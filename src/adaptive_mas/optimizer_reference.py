"""Independent SciPy reference comparison for the bounded portfolio optimizer.

This module is a verification aid only. SciPy remains an optional dependency and
is not used by the production decision path.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from adaptive_mas.portfolio.optimizer import PortfolioInfeasibleError, optimize_portfolio


class ReferenceCaseInfeasible(ValueError):
    """The declared box or turnover constraints have no feasible point."""


def representative_case() -> dict[str, object]:
    """Return a deterministic synthetic case with active portfolio constraints."""
    return {
        "case_id": "synthetic-constrained-allocation-v1",
        "fixture_label": "SYNTHETIC VERIFICATION FIXTURE ONLY",
        "nav_cny": 1_000_000.0,
        "expected_returns": {
            "600000": 0.16,
            "000001": 0.12,
            "000002": 0.075,
            "600001": 0.035,
        },
        "forecast_errors": {
            "600000": 0.025,
            "000001": 0.02,
            "000002": 0.018,
            "600001": 0.015,
        },
        "annualized_volatilities": {
            "600000": 0.28,
            "000001": 0.23,
            "000002": 0.19,
            "600001": 0.14,
        },
        "factor_betas": {
            "600000": 1.25,
            "000001": 1.0,
            "000002": 0.72,
            "600001": 0.35,
        },
        "market_annualized_volatility": 0.17,
        "current_weights": {
            "600000": 0.20,
            "000001": 0.15,
            "000002": 0.10,
            "600001": 0.05,
        },
        "horizon_sessions": 5,
        "maximum_symbol_weight": 0.32,
        "maximum_target_weights": {
            "600000": 0.28,
            "000001": 0.26,
            "000002": 0.22,
            "600001": 0.12,
        },
        "max_change_weights": {
            "600000": 0.09,
            "000001": 0.10,
            "000002": 0.08,
            "600001": 0.06,
        },
        "maximum_one_way_turnover": 0.12,
        "risk_budget_annualized": 0.115,
        "risk_aversion": 8.0,
        "forecast_error_multiplier": 0.5,
        "buy_cost_rate": 0.0011,
        "sell_cost_rate": 0.0016,
    }


def _normalized_inputs(case: Mapping[str, object]) -> dict[str, object]:
    names = sorted(set(case["expected_returns"]) | set(case["current_weights"]))  # type: ignore[arg-type]
    if not names:
        raise ValueError("reference case must contain at least one symbol")
    mappings = (
        "expected_returns", "forecast_errors", "annualized_volatilities",
        "factor_betas", "current_weights", "maximum_target_weights",
        "max_change_weights",
    )
    normalized: dict[str, object] = {"symbols": names}
    for key in mappings:
        raw = case.get(key, {})
        if not isinstance(raw, Mapping):
            raise ValueError(f"{key} must be a mapping")
        values = {symbol: float(raw.get(symbol, 0.0)) for symbol in names}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"{key} must contain finite values")
        normalized[key] = values
    for key in (
        "horizon_sessions", "maximum_symbol_weight", "maximum_one_way_turnover",
        "risk_budget_annualized", "risk_aversion", "forecast_error_multiplier",
        "buy_cost_rate", "sell_cost_rate", "market_annualized_volatility",
    ):
        value = float(case[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        normalized[key] = value
    return normalized


def _model(case: Mapping[str, object]) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row = _normalized_inputs(case)
    names = row["symbols"]
    assert isinstance(names, list)
    expected = row["expected_returns"]
    errors = row["forecast_errors"]
    volatilities = row["annualized_volatilities"]
    betas = row["factor_betas"]
    current = row["current_weights"]
    maximum_targets = row["maximum_target_weights"]
    capacities = row["max_change_weights"]
    assert all(isinstance(item, dict) for item in (expected, errors, volatilities, betas, current, maximum_targets, capacities))
    n = len(names)
    alpha = np.array([
        float(expected[name]) - float(row["forecast_error_multiplier"]) * float(errors[name])
        for name in names
    ])
    current_vector = np.array([float(current[name]) for name in names])
    beta_vector = np.array([float(betas[name]) for name in names])
    vol_vector = np.array([float(volatilities[name]) for name in names])
    market_variance = float(row["market_annualized_volatility"]) ** 2
    idiosyncratic = np.maximum(vol_vector**2 - beta_vector**2 * market_variance, 0.0)
    covariance = np.diag(idiosyncratic) + market_variance * np.outer(beta_vector, beta_vector)
    maximum_symbol = float(row["maximum_symbol_weight"])
    lower = np.array([
        max(current_vector[index] - float(capacities[name]), 0.0)
        for index, name in enumerate(names)
    ])
    upper = np.array([
        min(
            maximum_symbol,
            current_vector[index] + float(capacities[name]),
            float(maximum_targets[name]),
        )
        for index, name in enumerate(names)
    ])
    if np.any(lower > upper + 1e-12):
        raise ReferenceCaseInfeasible("target cap cannot be reached within the declared per-name capacity")
    row["lower_bounds"] = lower
    row["upper_bounds"] = upper
    row["alpha"] = alpha
    row["current_vector"] = current_vector
    row["covariance"] = covariance
    return row, alpha, current_vector, covariance, np.stack((lower, upper), axis=1)


def _transaction_cost(weights: np.ndarray, current: np.ndarray, buy_rate: float, sell_rate: float) -> float:
    buys = np.maximum(weights - current, 0.0)
    sells = np.maximum(current - weights, 0.0)
    return float(np.dot(buys, np.full_like(buys, buy_rate)) + np.dot(sells, np.full_like(sells, sell_rate)))


def _objective(row: Mapping[str, object], weights: np.ndarray, covariance: np.ndarray) -> float:
    alpha = np.asarray(row["alpha"], dtype=float)
    current = np.asarray(row["current_vector"], dtype=float)
    variance = float(weights @ covariance @ weights)
    risk_cost = 0.5 * float(row["risk_aversion"]) * (float(row["horizon_sessions"]) / 250.0) * variance
    costs = _transaction_cost(weights, current, float(row["buy_cost_rate"]), float(row["sell_cost_rate"]))
    return float(alpha @ weights - costs - risk_cost)


def _feasibility(row: Mapping[str, object], weights: np.ndarray, covariance: np.ndarray) -> dict[str, object]:
    current = np.asarray(row["current_vector"], dtype=float)
    lower = np.asarray(row["lower_bounds"], dtype=float)
    upper = np.asarray(row["upper_bounds"], dtype=float)
    turnover = float(np.abs(weights - current).sum() / 2.0)
    risk = float(math.sqrt(max(float(weights @ covariance @ weights), 0.0)))
    cost = _transaction_cost(weights, current, float(row["buy_cost_rate"]), float(row["sell_cost_rate"]))
    cash_used = float(weights.sum() + cost)
    violations = {
        "cash": max(0.0, cash_used - 1.0),
        "one_way_turnover": max(0.0, turnover - float(row["maximum_one_way_turnover"])),
        "annualized_risk": max(0.0, risk - float(row["risk_budget_annualized"])),
        "lower_bounds": float(np.maximum(lower - weights, 0.0).max(initial=0.0)),
        "upper_bounds": float(np.maximum(weights - upper, 0.0).max(initial=0.0)),
    }

    max_violation = max(violations.values(), default=0.0)
    return {
        "feasible": max_violation <= 2e-7,
        "maximum_violation": max_violation,
        "constraint_violations": violations,
        "cash_weight_after_costs": 1.0 - cash_used,
        "one_way_turnover": turnover,
        "annualized_risk": risk,
        "transaction_cost_weight": cost,
    }


def boundary_cases() -> dict[str, dict[str, object]]:
    """Return synthetic hard-cap repair and infeasible-boundary cases."""
    shared = {
        "fixture_label": "SYNTHETIC VERIFICATION FIXTURE ONLY",
        "nav_cny": 1_000_000.0,
        "expected_returns": {"600000": 0.16},
        "forecast_errors": {"600000": 0.0},
        "annualized_volatilities": {"600000": 0.10},
        "factor_betas": {"600000": 0.0},
        "market_annualized_volatility": 0.17,
        "current_weights": {"600000": 0.50},
        "horizon_sessions": 5,
        "maximum_symbol_weight": 0.80,
        "maximum_target_weights": {"600000": 0.10},
        "maximum_one_way_turnover": 1.0,
        "risk_budget_annualized": 1.0,
        "risk_aversion": 0.0,
        "forecast_error_multiplier": 0.0,
        "buy_cost_rate": 0.001,
        "sell_cost_rate": 0.001,
    }
    return {
        "current_holding_above_target_cap": {
            **shared,
            "case_id": "synthetic-current-holding-above-target-cap-v1",
            "max_change_weights": {"600000": 0.50},
        },
        "target_cap_incompatible_with_capacity": {
            **shared,
            "case_id": "synthetic-target-cap-capacity-conflict-v1",
            "max_change_weights": {"600000": 0.10},
        },
        "target_cap_incompatible_with_turnover": {
            **shared,
            "case_id": "synthetic-target-cap-turnover-conflict-v1",
            "max_change_weights": {"600000": 0.50},
            "maximum_one_way_turnover": 0.19,
        },
    }


def solve_reference(case: Mapping[str, object]) -> dict[str, object]:
    """Solve the convex quadratic portfolio objective with SciPy SLSQP."""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - exercised without the verification extra
        raise RuntimeError("install adaptive-mas[verification] to run the reference solver") from exc

    try:
        row, alpha, current, covariance, bounds_matrix = _model(case)
    except ReferenceCaseInfeasible as exc:
        return {
            "status": "infeasible_case",
            "solver": "independent target/capacity bound check",
            "success": False,
            "message": str(exc),
            "feasibility": {"feasible": False, "reason": str(exc)},
            "optimality_note": "The declared per-name target and capacity intervals do not intersect.",
        }
    names = row["symbols"]
    assert isinstance(names, list)
    n = len(names)
    buy_rate = float(row["buy_cost_rate"])
    sell_rate = float(row["sell_cost_rate"])
    horizon_scale = float(row["horizon_sessions"]) / 250.0
    risk_multiplier = float(row["risk_aversion"]) * horizon_scale
    risk_budget = float(row["risk_budget_annualized"])
    turnover_budget = float(row["maximum_one_way_turnover"])

    lower = np.asarray(row["lower_bounds"], dtype=float)
    upper = np.asarray(row["upper_bounds"], dtype=float)
    nearest_box_weights = np.minimum(np.maximum(current, lower), upper)
    minimum_box_turnover = float(np.abs(nearest_box_weights - current).sum() / 2.0)
    if minimum_box_turnover > turnover_budget + 1e-10:
        return {
            "status": "infeasible_case",
            "solver": "independent minimum-turnover bound check",
            "success": False,
            "message": "minimum one-way turnover to reach the target/capacity box exceeds the declared budget",
            "minimum_required_one_way_turnover": minimum_box_turnover,
            "maximum_one_way_turnover": turnover_budget,
            "feasibility": {"feasible": False, "reason": "turnover budget is below the box-distance lower bound"},
            "optimality_note": "A stricter risk or cash constraint cannot make this box-distance lower bound smaller.",
        }

    # x = [target weights, bought weights, sold weights]. The buy/sell
    # auxiliaries linearize asymmetric turnover costs and cash use.
    initial = np.concatenate((current, np.zeros(n), np.zeros(n)))
    variable_bounds = [
        (float(bounds_matrix[i, 0]), float(bounds_matrix[i, 1])) for i in range(n)
    ] + [(0.0, None)] * (2 * n)
    identity = np.eye(n)
    zero = np.zeros((n, n))
    buy_link_jac = np.concatenate((-identity, identity, zero), axis=1)
    sell_link_jac = np.concatenate((identity, zero, identity), axis=1)

    def objective(x: np.ndarray) -> float:
        weights, buys, sells = x[:n], x[n:2*n], x[2*n:]
        variance = float(weights @ covariance @ weights)
        return float(-alpha @ weights + buy_rate * buys.sum() + sell_rate * sells.sum() + 0.5 * risk_multiplier * variance)

    def objective_jac(x: np.ndarray) -> np.ndarray:
        weights = x[:n]
        gradient_weights = -alpha + risk_multiplier * (covariance @ weights)
        return np.concatenate((gradient_weights, np.full(n, buy_rate), np.full(n, sell_rate)))

    def cash_constraint(x: np.ndarray) -> float:
        return float(1.0 - x[:n].sum() - buy_rate * x[n:2*n].sum() - sell_rate * x[2*n:].sum())

    def cash_jac(_x: np.ndarray) -> np.ndarray:
        return np.concatenate((-np.ones(n), np.full(n, -buy_rate), np.full(n, -sell_rate)))

    def turnover_constraint(x: np.ndarray) -> float:
        return float(turnover_budget - 0.5 * (x[n:2*n].sum() + x[2*n:].sum()))

    def turnover_jac(_x: np.ndarray) -> np.ndarray:
        return np.concatenate((np.zeros(n), np.full(n, -0.5), np.full(n, -0.5)))

    def risk_constraint(x: np.ndarray) -> float:
        weights = x[:n]
        return float(risk_budget**2 - weights @ covariance @ weights)

    def risk_jac(x: np.ndarray) -> np.ndarray:
        return np.concatenate((-2.0 * covariance @ x[:n], np.zeros(2*n)))

    constraints = [
        {"type": "ineq", "fun": lambda x: x[n:2*n] - x[:n] + current, "jac": lambda _x: buy_link_jac},
        {"type": "ineq", "fun": lambda x: x[2*n:] + x[:n] - current, "jac": lambda _x: sell_link_jac},
        {"type": "ineq", "fun": cash_constraint, "jac": cash_jac},
        {"type": "ineq", "fun": turnover_constraint, "jac": turnover_jac},
        {"type": "ineq", "fun": risk_constraint, "jac": risk_jac},
    ]
    result = minimize(
        objective,
        initial,
        jac=objective_jac,
        method="SLSQP",
        bounds=variable_bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
    )
    weights = np.asarray(result.x[:n], dtype=float)
    feasibility = _feasibility(row, weights, covariance)
    if not result.success or not feasibility["feasible"]:
        raise RuntimeError(
            f"SciPy reference did not converge to a feasible result: {result.message}; "
            f"violation={feasibility['maximum_violation']}"
        )
    target_weights = {symbol: float(weights[index]) for index, symbol in enumerate(names)}
    return {
        "status": "optimal_numeric_reference",
        "solver": "scipy.optimize.minimize(method='SLSQP')",
        "solver_version": version("scipy"),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective_value": _objective(row, weights, covariance),
        "target_weights": target_weights,
        "feasibility": feasibility,
        "formulation": "concave quadratic utility maximization with convex risk, turnover, cash, target and capacity constraints; buy/sell auxiliaries linearize asymmetric costs",
        "optimality_note": "Numeric reference from a mature general-purpose solver; not a formal certificate of global optimality.",
    }


def compare_production(case: Mapping[str, object], iteration_budgets: tuple[int, ...] = (1, 5, 20, 100, 400, 1200)) -> dict[str, object]:
    row, _alpha, current, covariance, bounds_matrix = _model(case)
    names = row["symbols"]
    assert isinstance(names, list)
    reference = solve_reference(case)
    reference_objective = float(reference["objective_value"])
    reference_weights = np.array([float(reference["target_weights"][name]) for name in names])  # type: ignore[index]
    production_rows = []
    for iterations in iteration_budgets:
        production = optimize_portfolio(
            {name: float(row["expected_returns"][name]) for name in names},  # type: ignore[index]
            {name: float(row["forecast_errors"][name]) for name in names},  # type: ignore[index]
            {name: float(row["annualized_volatilities"][name]) for name in names},  # type: ignore[index]
            {name: float(current[index]) for index, name in enumerate(names)},
            horizon_sessions=int(row["horizon_sessions"]),
            maximum_symbol_weight=float(row["maximum_symbol_weight"]),
            maximum_one_way_turnover=float(row["maximum_one_way_turnover"]),
            risk_budget_annualized=float(row["risk_budget_annualized"]),
            risk_aversion=float(row["risk_aversion"]),
            forecast_error_multiplier=float(row["forecast_error_multiplier"]),
            buy_cost_rate=float(row["buy_cost_rate"]),
            sell_cost_rate=float(row["sell_cost_rate"]),
            factor_betas={name: float(row["factor_betas"][name]) for name in names},  # type: ignore[index]
            market_annualized_volatility=float(row["market_annualized_volatility"]),
            max_change_weights={name: float(row["max_change_weights"][name]) for name in names},  # type: ignore[index]
            maximum_target_weights={name: float(row["maximum_target_weights"][name]) for name in names},  # type: ignore[index]
            iterations=int(iterations),
        )
        weights = np.array([float(production.target_weights.get(name, 0.0)) for name in names])
        feasibility = _feasibility(row, weights, covariance)
        production_objective = _objective(row, weights, covariance)
        production_rows.append({
            "iteration_budget": int(iterations),
            "objective_value": production_objective,
            "objective_gap_to_reference": reference_objective - production_objective,
            "objective_gap_cny_at_case_nav": (reference_objective - production_objective) * float(case["nav_cny"]),
            "target_l1_distance_to_reference": float(np.abs(weights - reference_weights).sum()),
            "target_weights": {name: float(weights[index]) for index, name in enumerate(names)},
            "feasibility": feasibility,
        })
    final = production_rows[-1]
    return {
        "schema_version": "adaptive-mas.optimizer-reference-comparison.v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "fixture_label": str(case.get("fixture_label", "synthetic-only")),
        "case_id": str(case.get("case_id", "unnamed")),
        "case": dict(case),
        "reference": reference,
        "production_solver": "adaptive_mas.portfolio.optimizer.optimize_portfolio (proximal-gradient with constraint repair)",
        "convergence": production_rows,
        "boundary_cases": compare_boundary_cases(),
        "final_comparison": {
            "iteration_budget": final["iteration_budget"],
            "objective_gap_to_reference": final["objective_gap_to_reference"],
            "objective_gap_cny_at_case_nav": final["objective_gap_cny_at_case_nav"],
            "target_l1_distance_to_reference": final["target_l1_distance_to_reference"],
            "production_feasible": final["feasibility"]["feasible"],  # type: ignore[index]
            "reference_feasible": reference["feasibility"]["feasible"],  # type: ignore[index]
        },
        "interpretation": "This synthetic case measures numerical objective gap and feasibility only. A feasible production result is not represented as optimal; real market outcomes and strategy value are outside this comparison.",
    }


def compare_boundary_cases() -> dict[str, dict[str, object]]:
    """Compare production repair/fail-closed behavior with independent bounds/reference cases."""
    comparisons = {}
    for name, case in boundary_cases().items():
        row = _normalized_inputs(case)
        symbols = row["symbols"]
        assert isinstance(symbols, list)
        try:
            production = optimize_portfolio(
                {symbol: float(row["expected_returns"][symbol]) for symbol in symbols},  # type: ignore[index]
                {symbol: float(row["forecast_errors"][symbol]) for symbol in symbols},  # type: ignore[index]
                {symbol: float(row["annualized_volatilities"][symbol]) for symbol in symbols},  # type: ignore[index]
                {symbol: float(row["current_weights"][symbol]) for symbol in symbols},  # type: ignore[index]
                horizon_sessions=int(row["horizon_sessions"]),
                maximum_symbol_weight=float(row["maximum_symbol_weight"]),
                maximum_one_way_turnover=float(row["maximum_one_way_turnover"]),
                risk_budget_annualized=float(row["risk_budget_annualized"]),
                risk_aversion=float(row["risk_aversion"]),
                forecast_error_multiplier=float(row["forecast_error_multiplier"]),
                buy_cost_rate=float(row["buy_cost_rate"]),
                sell_cost_rate=float(row["sell_cost_rate"]),
                factor_betas={symbol: float(row["factor_betas"][symbol]) for symbol in symbols},  # type: ignore[index]
                market_annualized_volatility=float(row["market_annualized_volatility"]),
                max_change_weights={symbol: float(row["max_change_weights"][symbol]) for symbol in symbols},  # type: ignore[index]
                maximum_target_weights={symbol: float(row["maximum_target_weights"][symbol]) for symbol in symbols},  # type: ignore[index]
                iterations=1200,
            )
            production_row, _alpha, _current, covariance, _bounds = _model(case)
            weights = np.array([float(production.target_weights.get(symbol, 0.0)) for symbol in symbols])
            production_result = {
                "status": "feasible",
                "target_weights": production.target_weights,
                "feasibility": _feasibility(production_row, weights, covariance),
            }
        except PortfolioInfeasibleError as exc:
            production_result = {"status": "fail_closed", "reason": str(exc)}
        reference = solve_reference(case)
        comparisons[name] = {
            "case_id": case["case_id"],
            "fixture_label": case["fixture_label"],
            "production": production_result,
            "reference": reference,
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_production(representative_case())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["final_comparison"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
