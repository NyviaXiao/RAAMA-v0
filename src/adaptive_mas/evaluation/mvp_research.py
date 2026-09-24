"""Walk-forward training, agent fusion, market state, and portfolio targets."""

import math
import statistics
from collections import defaultdict
from datetime import date

from adaptive_mas.agents.quantitative import AGENT_IDS, quantitative_agents
from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data.research import ResearchData
from adaptive_mas.evaluation.agent_research import percentile_scores, rank_ic
from adaptive_mas.evaluation.uncertainty import paired_block_bootstrap
from adaptive_mas.models.ridge import ridge_predictions
from adaptive_mas.portfolio.allocation import (
    apply_weight_and_turnover_limits,
    inverse_volatility_weights,
    top_fraction_weights,
)
from adaptive_mas.regime.market_state import conditional_weights, market_state
from adaptive_mas.reliability.gating import reliability_weights
from adaptive_mas.targets.five_session import five_session_forward_returns


def _evaluate_matured_window(
    research: ResearchData,
    pending: dict[str, object],
    top_fraction: float,
    available_by: date | None = None,
) -> tuple[dict[str, float], dict[str, object], list[tuple[tuple[float, ...], float]]]:
    labels = five_session_forward_returns(research, [pending["decision_date"]])
    outcome_by_symbol = {
        str(row["symbol"]): float(row["forward_return_5s"])
        for row in labels
        if available_by is None or row["available_at"] <= available_by
    }
    shared_symbols = sorted(
        outcome_by_symbol.keys() & set(pending["features_by_symbol"])
    )
    matured_agents = {}
    method_results = {}
    for method, method_scores in pending["scores_by_method"].items():
        common = [symbol for symbol in shared_symbols if symbol in method_scores]
        if len(common) < 2:
            continue
        method_values = [method_scores[symbol] for symbol in common]
        outcome_values = [outcome_by_symbol[symbol] for symbol in common]
        ordered = sorted(common, key=lambda symbol: (-method_scores[symbol], symbol))
        selected_count = max(1, math.ceil(len(ordered) * top_fraction))
        method_results[method] = {
            "rank_ic": rank_ic(method_values, outcome_values),
            "top_decile_gross_return": statistics.fmean(
                outcome_by_symbol[symbol] for symbol in ordered[:selected_count]
            ),
        }
        if method in AGENT_IDS:
            matured_agents[method] = method_results[method]["rank_ic"]
    if len(matured_agents) != len(AGENT_IDS):
        return {}, {}, []
    training_rows = [
        (pending["features_by_symbol"][symbol], outcome_by_symbol[symbol])
        for symbol in shared_symbols
    ]
    observation = {
        "decision_date": pending["decision_date"].isoformat(),
        "label_available_at": labels[0]["available_at"].isoformat(),
        "regime": pending["regime"],
        "cross_section": len(shared_symbols),
        "methods": method_results,
        "adaptive_weights": pending["adaptive_weights"],
        "regime_weights": pending["regime_weights"],
        "regime_weight_source": pending["regime_weight_source"],
        "regime_matured_windows": pending["regime_matured_windows"],
    }
    return matured_agents, observation, training_rows


def run_mvp_research(
    research: ResearchData,
    config: dict[str, float | int],
    initial_state: dict[str, object] | None = None,
) -> dict[str, object]:
    step = int(config["decision_step_sessions"])
    trend_window = int(config["feature_trend_sessions"])
    reversal_window = int(config["feature_reversal_sessions"])
    liquidity_return_window = int(config["feature_liquidity_return_sessions"])
    recent_amount = int(config["feature_recent_amount_sessions"])
    reference_amount = int(config["feature_reference_amount_sessions"])
    top_fraction = float(config["evaluation_top_fraction"])
    portfolio_fraction = float(config["portfolio_top_fraction"])
    invested_weight = float(config["invested_weight"])
    reliability_window = int(config["reliability_lookback_windows"])
    reliability_minimum = int(config["reliability_minimum_windows"])
    equal_weight_share = float(config["reliability_equal_weight_share"])
    regime_window = int(config["regime_lookback_sessions"])
    regime_minimum = int(config["regime_minimum_windows"])
    ridge_alpha = float(config["ridge_alpha"])
    ridge_minimum_rows = int(config["ridge_minimum_training_rows"])
    risk_window = int(config["risk_lookback_sessions"])
    risk_minimum_observations = int(config["risk_minimum_observations"])
    maximum_symbol_weight = float(config["maximum_symbol_weight"])
    maximum_turnover = float(config["maximum_one_way_turnover"])

    first_signal_index = max(
        trend_window,
        reversal_window,
        recent_amount + reference_amount - 1,
    )
    decision_indices = list(
        range(first_signal_index, len(research.trading_dates) - 5, step)
    )
    if not decision_indices:
        raise ValueError("训练区间不足以形成五交易日决策窗口")

    bars = {
        (str(row["symbol"]), row["trade_date"]): row
        for row in research.adjusted_bars
    }
    initial_state = initial_state or {}
    matured_rank_ics = list(initial_state.get("matured_rank_ics", []))
    matured_by_state: dict[str, list[dict[str, float]]] = defaultdict(
        list, {state: list(rows) for state, rows in initial_state.get("matured_by_state", {}).items()}
    )
    training_features = [tuple(row) for row in initial_state.get("training_features", [])]
    training_returns = [float(value) for value in initial_state.get("training_returns", [])]
    observations = []
    decision_log = []
    decisions_by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    previous_targets: dict[str, dict[str, float]] = defaultdict(dict)
    pending = None
    static_weights = None
    comparison_started = False

    for decision_index in decision_indices:
        decision_date = research.trading_dates[decision_index]

        # Only labels whose exit close is visible by this after-close decision mature here.
        if pending is not None:
            matured_row, observation, training_rows = _evaluate_matured_window(
                research, pending, top_fraction, decision_date
            )
            if matured_row:
                matured_rank_ics.append(matured_row)
                matured_by_state[pending["regime"]].append(matured_row)
                training_features.extend(features for features, _ in training_rows)
                training_returns.extend(value for _, value in training_rows)
                observations.append(observation)

        if static_weights is None and len(matured_rank_ics) >= reliability_minimum:
            static_weights = reliability_weights(
                matured_rank_ics[:reliability_minimum],
                AGENT_IDS,
                reliability_window,
                reliability_minimum,
                equal_weight_share,
            )

        snapshot = research.snapshot_at(decision_date)
        outputs = quantitative_agents(
            snapshot,
            trend_window,
            reversal_window,
            liquidity_return_window,
            recent_amount,
            reference_amount,
        )
        raw_scores = {
            agent_id: {signal.symbol: signal.score for signal in outputs[agent_id]}
            for agent_id in AGENT_IDS
        }
        common_symbols = set.intersection(*(set(scores) for scores in raw_scores.values()))
        if len(common_symbols) < 2:
            raise ValueError(f"{decision_date} 可比较的共同 Agent 信号不足")
        feature_by_symbol = {
            symbol: tuple(raw_scores[agent_id][symbol] for agent_id in AGENT_IDS)
            for symbol in sorted(common_symbols)
        }
        percentile_by_agent = {
            agent_id: percentile_scores(
                {symbol: raw_scores[agent_id][symbol] for symbol in common_symbols}
            )
            for agent_id in AGENT_IDS
        }
        equal_fusion = {
            symbol: statistics.fmean(
                percentile_by_agent[agent_id][symbol] for agent_id in AGENT_IDS
            )
            for symbol in common_symbols
        }
        current_adaptive_weights = reliability_weights(
            matured_rank_ics,
            AGENT_IDS,
            reliability_window,
            reliability_minimum,
            equal_weight_share,
        )
        static_for_decision = static_weights or {
            agent_id: 1 / len(AGENT_IDS) for agent_id in AGENT_IDS
        }
        static_fusion = {
            symbol: math.fsum(
                static_for_decision[agent_id] * percentile_by_agent[agent_id][symbol]
                for agent_id in AGENT_IDS
            )
            for symbol in common_symbols
        }
        adaptive_fusion = {
            symbol: math.fsum(
                current_adaptive_weights[agent_id] * percentile_by_agent[agent_id][symbol]
                for agent_id in AGENT_IDS
            )
            for symbol in common_symbols
        }
        current_regime = market_state(research, decision_date, regime_window)
        current_regime_weights, regime_weight_source, regime_history_count = conditional_weights(
            matured_by_state,
            current_regime,
            matured_rank_ics,
            AGENT_IDS,
            reliability_window,
            regime_minimum,
            equal_weight_share,
        )
        regime_fusion = {
            symbol: math.fsum(
                current_regime_weights[agent_id] * percentile_by_agent[agent_id][symbol]
                for agent_id in AGENT_IDS
            )
            for symbol in common_symbols
        }

        ridge_scores = ridge_predictions(
            training_features,
            training_returns,
            feature_by_symbol,
            ridge_alpha,
        )
        if len(training_returns) >= ridge_minimum_rows and not ridge_scores:
            raise ValueError("Ridge 已达到训练样本门槛但没有产生预测")

        score_by_method = {
            **{
                agent_id: percentile_by_agent[agent_id]
                for agent_id in AGENT_IDS
            },
            "ridge": ridge_scores,
            "equal_fusion": equal_fusion,
            "static_fusion": static_fusion,
            "adaptive_fusion": adaptive_fusion,
            "regime_fusion": regime_fusion,
        }

        # Comparison starts only after both supervised and adaptive systems have
        # a real matured training history; earlier signals are warm-up only.
        if len(training_returns) >= ridge_minimum_rows and len(matured_rank_ics) >= reliability_minimum:
            comparison_started = True
            for method, scores in score_by_method.items():
                if not scores:
                    continue
                target = top_fraction_weights(scores, portfolio_fraction, invested_weight)
                target = apply_weight_and_turnover_limits(
                    target,
                    previous_targets[method],
                    maximum_symbol_weight,
                    maximum_turnover,
                )
                previous_targets[method] = target
                decisions_by_method[method].append(
                    {"decision_date": decision_date, "target_weights": target}
                )

            risk_history = {}
            history_dates = research.trading_dates[
                max(0, decision_index - risk_window + 1) : decision_index + 1
            ]
            for symbol in common_symbols:
                returns = []
                for day in history_dates:
                    row = bars.get((symbol, day))
                    if row is not None and row["return_1d"] is not None:
                        returns.append(float(row["return_1d"]) / 100)
                risk_history[symbol] = returns
            if decision_index >= risk_window:
                risk_target = inverse_volatility_weights(
                    adaptive_fusion,
                    risk_history,
                    portfolio_fraction,
                    invested_weight,
                    maximum_symbol_weight,
                    risk_minimum_observations,
                )
                risk_status = "inverse_volatility"
            else:
                risk_target = top_fraction_weights(
                    adaptive_fusion, portfolio_fraction, invested_weight
                )
                risk_status = "equal_weight_warmup"
            risk_target = apply_weight_and_turnover_limits(
                risk_target,
                previous_targets["adaptive_inverse_volatility"],
                maximum_symbol_weight,
                maximum_turnover,
            )
            previous_targets["adaptive_inverse_volatility"] = risk_target
            decisions_by_method["adaptive_inverse_volatility"].append(
                {"decision_date": decision_date, "target_weights": risk_target}
            )
        else:
            risk_status = "warmup"

        decision_log.append(
            {
                "decision_date": decision_date.isoformat(),
                "input_cutoff": decision_date.isoformat(),
                "members": len(snapshot.members),
                "common_signal_count": len(common_symbols),
                "matured_rank_ic_windows": len(matured_rank_ics),
                "ridge_training_rows": len(training_returns),
                "regime": current_regime,
                "regime_weight_source": regime_weight_source,
                "regime_matured_windows": regime_history_count,
                "risk_allocator_status": risk_status,
                "weights": {
                    "static": static_for_decision,
                    "adaptive": current_adaptive_weights,
                    "regime": current_regime_weights,
                },
                "target_weights": {
                    method: {
                        symbol: weight
                        for symbol, weight in decisions[-1]["target_weights"].items()
                    }
                    for method, decisions in decisions_by_method.items()
                    if decisions and decisions[-1]["decision_date"] == decision_date
                },
            }
        )
        pending = {
            "decision_date": decision_date,
            "features_by_symbol": feature_by_symbol,
            "scores_by_method": score_by_method,
            "regime": current_regime,
            "adaptive_weights": current_adaptive_weights,
            "regime_weights": current_regime_weights,
            "regime_weight_source": regime_weight_source,
            "regime_matured_windows": regime_history_count,
        }

    if pending is not None:
        matured_row, observation, training_rows = _evaluate_matured_window(
            research, pending, top_fraction
        )
        if matured_row:
            matured_rank_ics.append(matured_row)
            matured_by_state[pending["regime"]].append(matured_row)
            training_features.extend(features for features, _ in training_rows)
            training_returns.extend(value for _, value in training_rows)
            observations.append(observation)

    if not comparison_started:
        raise ValueError("训练历史不足以形成可比较的多 Agent 组合实验")

    return {
        "protocol": {
            "universe": "historical BaoStock membership snapshots as visible at each decision date",
            "decision": "after close; targets execute at next session open",
            "daily_marking": "adjusted open/close price ratios; suspended positions carry last value",
            "execution_boundary": "unadjusted BaoStock tradestatus; no board-limit queue model",
            "label": "next-session open to fifth-session close adjusted-price return",
            "decision_step_sessions": step,
            "top_fraction": portfolio_fraction,
            "invested_weight": invested_weight,
            "maximum_symbol_weight": maximum_symbol_weight,
            "maximum_one_way_turnover": maximum_turnover,
            "liquidity_limit": "order notional capped by trailing mean traded amount times configured participation",
            "reliability": "only matured five-session Rank IC before each after-close decision",
            "static_weights": "first configured number of matured development windows, then frozen",
            "regime": f"risk_on when CSI 300 {regime_window}-session price return is nonnegative; otherwise risk_off",
            "regime_fallback": "global reliability until the state has enough matured windows",
            "ridge": "three existing Agent scores, standardized using matured training rows only",
            "risk_allocator": "top-decile eligibility, inverse trailing volatility, weight and turnover caps",
        "initial_comparison_date": decision_log[
                next(
                    index
                    for index, row in enumerate(decision_log)
                    if row["target_weights"]
                )
            ]["decision_date"],
            "costs_bps": {
                key: config[key]
                for key in (
                    "commission_per_side_bps",
                    "transfer_fee_per_side_bps",
                    "stamp_duty_sell_bps",
                    "slippage_per_side_bps",
                )
            },
        },
        "decision_windows": len(decision_indices),
        "evaluated_label_windows": len(observations),
        "comparison_decision_count": min(
            len(rows) for rows in decisions_by_method.values()
        ),
        "observations": observations,
        "decisions_by_method": dict(decisions_by_method),
        "decision_log": decision_log,
        "training_rows_at_end": len(training_returns),
        "matured_rank_ic_windows_at_end": len(matured_rank_ics),
        "training_seed": {
            "training_features": training_features,
            "training_returns": training_returns,
            "matured_rank_ics": matured_rank_ics,
            "matured_by_state": dict(matured_by_state),
        },
    }


def run_portfolio_comparisons(
    research: ResearchData,
    result: dict[str, object],
    config: dict[str, float | int],
) -> dict[str, object]:
    costs = {
        key: float(config[key])
        for key in (
            "commission_per_side_bps",
            "transfer_fee_per_side_bps",
            "stamp_duty_sell_bps",
            "slippage_per_side_bps",
        )
    }
    initial_capital = float(config["initial_capital_cny"])
    liquidity_lookback = int(config["liquidity_lookback_sessions"])
    maximum_participation = float(config["maximum_daily_participation"])
    comparisons = {}
    artifacts = {}
    sensitivity = {"0x": {}, "2x": {}}
    for method, decisions in result["decisions_by_method"].items():
        simulated = simulate_portfolio(
            research,
            decisions,
            initial_capital,
            costs,
            liquidity_lookback,
            maximum_participation,
        )
        comparisons[method] = simulated.pop("summary")
        comparisons[method]["paired_block_bootstrap_vs_benchmark"] = paired_block_bootstrap(
            [float(row["daily_return"]) for row in simulated["daily_nav"]],
            [float(row["benchmark_return"]) for row in simulated["daily_nav"]],
        )
        artifacts[method] = simulated
        for scenario, multiplier in (("0x", 0.0), ("2x", 2.0)):
            scenario_costs = {
                name: value * multiplier for name, value in costs.items()
            }
            sensitivity[scenario][method] = simulate_portfolio(
                research,
                decisions,
                initial_capital,
                scenario_costs,
                liquidity_lookback,
                maximum_participation,
            )["summary"]
    return {"summary": comparisons, "cost_sensitivity": sensitivity, "artifacts": artifacts}
