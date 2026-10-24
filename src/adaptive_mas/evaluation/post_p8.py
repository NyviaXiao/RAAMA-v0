"""Predeclared post-P8 prediction diagnostics and matched-date comparisons."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict

import numpy as np

from adaptive_mas.agents.quantitative import AGENT_IDS
from adaptive_mas.evaluation.agent_research import rank_ic


def paired_block_mean_difference(
    left: list[float],
    right: list[float],
    *,
    block_sessions: int = 20,
    repetitions: int = 2000,
    seed: int = 20260926,
) -> dict[str, float | int]:
    """Moving-block interval for an arbitrary date-level paired metric difference."""
    if len(left) != len(right) or not left:
        raise ValueError("配对日期差异区间要求两组等长且非空")
    if block_sessions < 1 or repetitions < 1:
        raise ValueError("Bootstrap 区块长度和重复次数必须为正")
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise ValueError("配对日期差异包含非有限数值")
    count = len(lhs)
    blocks = math.ceil(count / block_sessions)
    offsets = np.arange(block_sessions)
    random = np.random.default_rng(seed)
    starts = random.integers(0, count, size=(repetitions, blocks))
    indices = ((starts[:, :, None] + offsets) % count).reshape(repetitions, -1)[:, :count]
    mean_differences = (lhs[indices] - rhs[indices]).mean(axis=1)
    lower, upper = np.quantile(mean_differences, [0.025, 0.975])
    return {
        "mean_difference": float(np.mean(lhs - rhs)),
        "mean_difference_95pct_low": float(lower),
        "mean_difference_95pct_high": float(upper),
        "probability_left_exceeds_right": float(np.mean(mean_differences > 0)),
        "paired_dates": count,
        "block_sessions": block_sessions,
        "repetitions": repetitions,
        "seed": seed,
    }


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    variance_left = math.fsum((value - mean_left) ** 2 for value in left)
    variance_right = math.fsum((value - mean_right) ** 2 for value in right)
    if variance_left <= 0 or variance_right <= 0:
        return None
    covariance = math.fsum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(left, right, strict=True)
    )
    return covariance / math.sqrt(variance_left * variance_right)


def _date_forecast_metrics(
    predicted: dict[str, float],
    actual: dict[str, float],
    top_fraction: float,
) -> dict[str, float | None]:
    symbols = sorted(predicted.keys() & actual.keys())
    if not symbols:
        return {
            "rank_ic": None,
            "mae": None,
            "rmse": None,
            "top_decile_gross_return": None,
        }
    residuals = [actual[symbol] - predicted[symbol] for symbol in symbols]
    ordered = sorted(symbols, key=lambda symbol: (-predicted[symbol], symbol))
    valid_rank = len(symbols) > 1 and len({predicted[symbol] for symbol in symbols}) > 1
    selected = ordered[: max(1, math.ceil(len(ordered) * top_fraction))]
    return {
        "rank_ic": (
            rank_ic(
                [predicted[symbol] for symbol in symbols],
                [actual[symbol] for symbol in symbols],
            )
            if valid_rank else None
        ),
        "mae": statistics.fmean(abs(value) for value in residuals),
        "rmse": math.sqrt(statistics.fmean(value * value for value in residuals)),
        "top_decile_gross_return": (
            statistics.fmean(actual[symbol] for symbol in selected) if valid_rank else None
        ),
    }


def expert_prediction_diagnostics(
    result: dict[str, object],
    *,
    forecast_error_multiplier: float = 0.5,
) -> dict[str, object]:
    """Report date-level expert/fusion skill, residual dependence, and LOO ablations."""
    if forecast_error_multiplier < 0 or not math.isfinite(forecast_error_multiplier):
        raise ValueError("预先冻结的预测误差乘数必须有限且非负")
    predictions = {
        str(row["prediction_date"]): row for row in result["prediction_decisions"]
    }
    settlements_by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in result["prediction_settlements"]:
        settlements_by_date[str(row["prediction_date"])].append(row)

    candidate_checks = []
    method_metrics: dict[str, list[dict[str, object]]] = defaultdict(list)
    method_metrics_by_date: dict[str, dict[str, dict[str, float | None]]] = defaultdict(dict)
    signal_correlations: dict[str, list[float]] = defaultdict(list)
    signal_universe_overlap: dict[str, list[float]] = defaultdict(list)
    residual_correlations: dict[str, list[float]] = defaultdict(list)
    loo_metrics: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    portfolio_uncertainty_rows: dict[str, list[dict[str, float | None]]] = defaultdict(list)

    for prediction_date, prediction in predictions.items():
        candidates = sorted(str(symbol) for symbol in prediction["scored_symbols"])
        expected_hash = hashlib.sha256("\n".join(candidates).encode("ascii")).hexdigest()
        date_settlements = settlements_by_date.get(prediction_date, [])
        settled = [row for row in date_settlements if row["status"] == "settled"]
        return_field = f"forward_return_{prediction['horizon_sessions']}s"
        actual = {
            str(row["symbol"]): float(row[return_field])
            for row in settled
            if row.get(return_field) is not None
        }
        candidate_checks.append(
            {
                "prediction_date": prediction_date,
                "input_cutoff_matches_decision": prediction["input_cutoff"] == prediction_date,
                "candidate_hash_matches_symbols": prediction["candidate_symbols_sha256"] == expected_hash,
                "settlement_rows_match_candidate_hash": all(
                    row["candidate_symbols_sha256"] == expected_hash for row in date_settlements
                ),
                "settled_symbols_subset_of_frozen_candidates": set(actual) <= set(candidates),
                "candidate_count": len(candidates),
                "settled_count": len(actual),
                "pending_or_missing_count": len(candidates) - len(actual),
            }
        )
        if not actual:
            continue

        for pair, overlap in prediction.get("expert_pairwise_symbol_overlap", {}).items():
            jaccard = overlap.get("jaccard")
            if jaccard is not None:
                signal_universe_overlap[str(pair)].append(float(jaccard))
        scores_by_method = prediction["scores_by_method"]
        expected_by_method = prediction["expected_returns_by_method"]
        for method, predicted in expected_by_method.items():
            if not predicted:
                continue
            metrics = _date_forecast_metrics(
                {str(key): float(value) for key, value in predicted.items()},
                actual,
                0.10,
            )
            method_metrics[str(method)].append(metrics)
            method_metrics_by_date[prediction_date][str(method)] = metrics

        for method, weights_raw in prediction.get("target_weights_by_method", {}).items():
            weights = {str(symbol): float(value) for symbol, value in weights_raw.items()}
            source_method = {
                "joint_optimization": "loss_gated_fusion",
                "joint_market_relative": "market_plus_relative_fusion",
            }.get(method, method)
            expected = expected_by_method.get(source_method)
            rmse = prediction.get("forecast_errors_by_method", {}).get(source_method)
            if not weights or expected is None or rmse is None:
                continue
            common = sorted(set(weights) & set(actual) & set(expected))
            if not common:
                continue
            target_weight = math.fsum(weights[symbol] for symbol in common)
            predicted_return = math.fsum(
                weights[symbol] * float(expected[symbol]) for symbol in common
            )
            realized_return = math.fsum(
                weights[symbol] * actual[symbol] for symbol in common
            )
            vol_by_symbol = prediction["annualized_volatility_by_symbol"]
            beta_by_symbol = prediction["factor_beta_by_symbol"]
            market_vol = float(prediction["market_annualized_volatility"])
            factor_exposure = math.fsum(
                weights[symbol] * float(beta_by_symbol[symbol])
                for symbol in common
                if symbol in beta_by_symbol
            )
            idiosyncratic_variance = math.fsum(
                weights[symbol] ** 2
                * max(
                    float(vol_by_symbol[symbol]) ** 2
                    - float(beta_by_symbol[symbol]) ** 2 * market_vol**2,
                    0.0,
                )
                for symbol in common
                if symbol in vol_by_symbol and symbol in beta_by_symbol
            )
            annualized_portfolio_risk = math.sqrt(
                idiosyncratic_variance + (factor_exposure * market_vol) ** 2
            )
            horizon = int(prediction["horizon_sessions"])
            horizon_sigma = annualized_portfolio_risk * math.sqrt(horizon / 250)
            error_allowance = (
                target_weight * forecast_error_multiplier * float(rmse)
            )
            absolute_error = abs(realized_return - predicted_return)
            portfolio_uncertainty_rows[str(method)].append(
                {
                    "absolute_portfolio_forecast_error": absolute_error,
                    "forecast_error_allowance": error_allowance,
                    "horizon_sigma": horizon_sigma,
                    "error_allowance_over_horizon_sigma": (
                        error_allowance / horizon_sigma if horizon_sigma > 0 else None
                    ),
                    "within_error_allowance": float(absolute_error <= error_allowance),
                    "within_historical_horizon_sigma": float(absolute_error <= horizon_sigma),
                }
            )

        for left_index, left_agent in enumerate(AGENT_IDS):
            for right_agent in AGENT_IDS[left_index + 1 :]:
                shared = sorted(
                    set(scores_by_method[left_agent])
                    & set(scores_by_method[right_agent])
                    & set(actual)
                )
                signal_corr = (
                    rank_ic(
                        [float(scores_by_method[left_agent][symbol]) for symbol in shared],
                        [float(scores_by_method[right_agent][symbol]) for symbol in shared],
                    )
                    if len(shared) > 1
                    and len({scores_by_method[left_agent][symbol] for symbol in shared}) > 1
                    and len({scores_by_method[right_agent][symbol] for symbol in shared}) > 1
                    else None
                )
                if signal_corr is not None:
                    signal_correlations[f"{left_agent}:{right_agent}"].append(signal_corr)

                left_prediction = expected_by_method[left_agent]
                right_prediction = expected_by_method[right_agent]
                residual_left = [
                    actual[symbol] - float(left_prediction[symbol]) for symbol in shared
                ]
                residual_right = [
                    actual[symbol] - float(right_prediction[symbol]) for symbol in shared
                ]
                residual_corr = _correlation(residual_left, residual_right)
                if residual_corr is not None:
                    residual_correlations[f"{left_agent}:{right_agent}"].append(residual_corr)

        equal_prediction = expected_by_method.get("equal_fusion", {})
        for removed_agent in AGENT_IDS:
            remaining = [agent for agent in AGENT_IDS if agent != removed_agent]
            loo = {
                symbol: statistics.fmean(
                    float(expected_by_method[agent][symbol]) for agent in remaining
                )
                for symbol in set.intersection(
                    *(set(expected_by_method[agent]) for agent in remaining)
                )
            }
            loo_metrics[f"without_{removed_agent}"].append(
                _date_forecast_metrics(
                    {str(key): float(value) for key, value in loo.items()},
                    actual,
                    0.10,
                )
            )
        if not equal_prediction:
            raise ValueError("留一消融缺少等权融合的成熟预测")

    def summarize_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
        output = {"matured_dates": len(rows)}
        for metric in ("rank_ic", "mae", "rmse", "top_decile_gross_return"):
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            output[f"mean_date_{metric}"] = statistics.fmean(values) if values else None
            output[f"defined_date_{metric}_count"] = len(values)
        return output

    loo_ablation = {}
    for name, rows in loo_metrics.items():
        equal_rows = method_metrics["equal_fusion"]
        common = min(len(rows), len(equal_rows))
        comparison = {"without_method": summarize_metrics(rows)}
        for metric in ("rank_ic", "top_decile_gross_return"):
            paired = [
                (float(rows[index][metric]), float(equal_rows[index][metric]))
                for index in range(common)
                if rows[index][metric] is not None and equal_rows[index][metric] is not None
            ]
            comparison[f"equal_fusion_minus_ablation_{metric}"] = (
                paired_block_mean_difference(
                    [right for _, right in paired],
                    [left for left, _ in paired],
                    seed=20260926 + len(loo_ablation),
                )
                if paired else None
            )
        loo_ablation[name] = comparison

    paired_forecast_comparisons = {}
    baseline_method = "loss_gated_fusion"
    comparison_methods = sorted({
        method
        for rows in method_metrics_by_date.values()
        for method in rows
        if method != baseline_method
    })
    for method in comparison_methods:
        common_dates = sorted(
            day
            for day, rows in method_metrics_by_date.items()
            if method in rows and baseline_method in rows
        )
        by_metric = {}
        for metric in ("rank_ic", "mae", "rmse", "top_decile_gross_return"):
            paired = [
                (method_metrics_by_date[day][method][metric], method_metrics_by_date[day][baseline_method][metric])
                for day in common_dates
                if method_metrics_by_date[day][method][metric] is not None
                and method_metrics_by_date[day][baseline_method][metric] is not None
            ]
            by_metric[metric] = (
                paired_block_mean_difference(
                    [float(candidate) for candidate, _ in paired],
                    [float(baseline) for _, baseline in paired],
                    seed=20261100 + len(paired_forecast_comparisons),
                )
                if paired else None
            )
        paired_forecast_comparisons[method] = {
            "baseline_method": baseline_method,
            "common_dates": len(common_dates),
            "by_metric_candidate_minus_baseline": by_metric,
        }

    def mean_ci_by_key(values_by_key: dict[str, list[float]]) -> dict[str, object]:
        output = {}
        for index, (key, values) in enumerate(sorted(values_by_key.items())):
            if not values:
                output[key] = {"matured_dates": 0, "mean": None, "interval_95pct": None}
                continue
            output[key] = {
                "matured_dates": len(values),
                "mean": statistics.fmean(values),
                "interval_95pct": paired_block_mean_difference(
                    values,
                    [0.0] * len(values),
                    seed=20261000 + index,
                ),
            }
        return output

    uncertainty_summary = {}
    for method, rows in sorted(portfolio_uncertainty_rows.items()):
        ratios = [
            float(row["error_allowance_over_horizon_sigma"])
            for row in rows
            if row["error_allowance_over_horizon_sigma"] is not None
        ]
        uncertainty_summary[method] = {
            "matured_weighted_decision_dates": len(rows),
            "forecast_error_multiplier_frozen": forecast_error_multiplier,
            "mean_absolute_portfolio_forecast_error": statistics.fmean(
                float(row["absolute_portfolio_forecast_error"]) for row in rows
            ),
            "mean_forecast_error_allowance": statistics.fmean(
                float(row["forecast_error_allowance"]) for row in rows
            ),
            "mean_historical_portfolio_horizon_sigma": statistics.fmean(
                float(row["horizon_sigma"]) for row in rows
            ),
            "mean_error_allowance_over_horizon_sigma": statistics.fmean(ratios) if ratios else None,
            "fraction_realized_error_within_error_allowance": statistics.fmean(
                float(row["within_error_allowance"]) for row in rows
            ),
            "fraction_realized_error_within_historical_horizon_sigma": statistics.fmean(
                float(row["within_historical_horizon_sigma"]) for row in rows
            ),
            "correlation_absolute_forecast_error_vs_horizon_sigma": _correlation(
                [float(row["absolute_portfolio_forecast_error"]) for row in rows],
                [float(row["horizon_sigma"]) for row in rows],
            ),
        }

    return {
        "candidate_freeze_audit": {
            "prediction_dates": len(candidate_checks),
            "all_input_cutoffs_match_decision": all(
                row["input_cutoff_matches_decision"] for row in candidate_checks
            ),
            "all_candidate_hashes_recompute": all(
                row["candidate_hash_matches_symbols"] for row in candidate_checks
            ),
            "all_settlement_rows_keep_candidate_hash": all(
                row["settlement_rows_match_candidate_hash"] for row in candidate_checks
            ),
            "all_settled_symbols_were_scored": all(
                row["settled_symbols_subset_of_frozen_candidates"] for row in candidate_checks
            ),
            "mean_candidate_count": (
                statistics.fmean(row["candidate_count"] for row in candidate_checks)
                if candidate_checks else 0.0
            ),
            "candidate_checks": candidate_checks,
        },
        "date_level_forecast_metrics": {
            method: summarize_metrics(rows)
            for method, rows in sorted(method_metrics.items())
        },
        "paired_date_level_forecast_comparisons_vs_loss_gated_fusion": paired_forecast_comparisons,
        "expert_signal_rank_correlations": mean_ci_by_key(signal_correlations),
        "expert_pairwise_signal_universe_overlap": mean_ci_by_key(signal_universe_overlap),
        "expert_forecast_error_correlations": mean_ci_by_key(residual_correlations),
        "portfolio_uncertainty_vs_risk_diagnostic": {
            "interpretation": "Forecast RMSE allowance is reported alongside historical factor-plus-idiosyncratic horizon sigma. The RMSE multiplier remains frozen; this comparison diagnoses overlap and does not tune the trading gate.",
            "by_allocation_method": uncertainty_summary,
        },
        "leave_one_expert_out_ablation": loo_ablation,
    }
