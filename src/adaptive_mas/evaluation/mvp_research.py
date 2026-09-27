"""Walk-forward training, agent fusion, market state, and portfolio targets."""

import math
import hashlib
import statistics
from collections import defaultdict
from datetime import date
from typing import Callable

from adaptive_mas.agents.quantitative import AGENT_IDS, quantitative_agents
from adaptive_mas.backtest.simulator import (
    simulate_fixed_horizon_portfolio,
    simulate_portfolio,
)
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
from adaptive_mas.targets.five_session import forward_returns


def _evaluate_matured_window(
    research: ResearchData,
    pending: dict[str, object],
    top_fraction: float,
    available_by: date | None = None,
) -> tuple[
    dict[str, float],
    dict[str, object],
    list[tuple[tuple[float, ...], float]],
    list[dict[str, object]],
    list[tuple[tuple[float, ...], float]],
    list[tuple[tuple[float, ...], float]],
    dict[str, float],
]:
    horizon = int(pending.get("horizon_sessions", 5))
    labels = forward_returns(research, [pending["decision_date"]], horizon)
    label_by_symbol = {
        str(row["symbol"]): float(row["forward_return"])
        for row in labels
    }
    decision_index = research.trading_dates.index(pending["decision_date"])
    entry_date = research.trading_dates[decision_index + 1]
    exit_date = research.trading_dates[decision_index + horizon]
    settled_symbols = {
        str(row["symbol"])
        for row in labels
        if available_by is None or row["available_at"] <= available_by
    }
    settlements = [
        {
            "prediction_date": pending["decision_date"].isoformat(),
            "symbol": symbol,
            "entry_date": entry_date.isoformat(),
            "planned_exit_date": exit_date.isoformat(),
            "available_at": exit_date.isoformat(),
            "status": (
                "settled"
                if symbol in settled_symbols
                else "not_yet_available"
                if symbol in label_by_symbol
                else "missing_label_data"
            ),
            "candidate_symbols_sha256": pending["candidate_symbols_sha256"],
            f"forward_return_{horizon}s": label_by_symbol.get(symbol)
            if symbol in settled_symbols
            else None,
        }
        for symbol in sorted(pending["features_by_symbol"])
    ]
    outcome_by_symbol = {
        symbol: label_by_symbol[symbol] for symbol in settled_symbols
    }
    index_by_date = {row["trade_date"]: row for row in research.index_bars}
    entry_bar = index_by_date.get(entry_date)
    exit_bar = index_by_date.get(exit_date)
    if entry_bar is None or exit_bar is None or not entry_bar.get("open") or not exit_bar.get("close"):
        raise ValueError(f"BaoStock 沪深300指数缺少成熟标签价格：{entry_date} 至 {exit_date}")
    market_realized_return = float(exit_bar["close"]) / float(entry_bar["open"]) - 1
    relative_outcome_by_symbol = {
        symbol: value - market_realized_return
        for symbol, value in outcome_by_symbol.items()
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
        if len(set(method_values)) < 2:
            continue
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
    forecast_errors = {}
    expected_returns = pending.get("expected_returns_by_method", {})
    for method, predictions in expected_returns.items():
        common = [symbol for symbol in shared_symbols if symbol in predictions]
        if not common:
            continue
        residuals = [
            outcome_by_symbol[symbol] - float(predictions[symbol]) for symbol in common
        ]
        forecast_errors[method] = {
            "mae": statistics.fmean(abs(value) for value in residuals),
            "rmse": math.sqrt(statistics.fmean(value * value for value in residuals)),
            "rank_ic": (
                rank_ic(
                    [float(predictions[symbol]) for symbol in common],
                    [outcome_by_symbol[symbol] for symbol in common],
                )
                if len(common) > 1
                and len({float(predictions[symbol]) for symbol in common}) > 1
                else None
            ),
            "calibration_buckets": (
                _return_calibration_buckets(
                    common,
                    predictions,
                    outcome_by_symbol,
                    bucket_count=5,
                )
                if len({float(predictions[symbol]) for symbol in common}) > 1
                else []
            ),
            "mean_realized_market_excess_return": statistics.fmean(
                relative_outcome_by_symbol[symbol] for symbol in common
            ),
        }
        if method in method_results:
            method_results[method].update(forecast_errors[method])
    if len(matured_agents) != len(AGENT_IDS):
        return {}, {}, [], settlements, [], {}
    training_rows = [
        (pending["features_by_symbol"][symbol], outcome_by_symbol[symbol])
        for symbol in shared_symbols
    ]
    observation = {
        "decision_date": pending["decision_date"].isoformat(),
        "label_available_at": labels[0]["available_at"].isoformat(),
        "realized_market_return": market_realized_return,
        "regime": pending["regime"],
        "cross_section": len(shared_symbols),
        "methods": method_results,
        "forecast_errors": forecast_errors,
        "adaptive_weights": pending["adaptive_weights"],
        "regime_weights": pending["regime_weights"],
        "regime_weight_source": pending["regime_weight_source"],
        "regime_matured_windows": pending["regime_matured_windows"],
    }
    calibration_features = pending.get("percentile_features_by_symbol", {})
    calibration_rows = [
        (tuple(calibration_features[symbol]), outcome_by_symbol[symbol])
        for symbol in shared_symbols
        if symbol in calibration_features
    ]
    relative_calibration_rows = [
        (tuple(calibration_features[symbol]), relative_outcome_by_symbol[symbol])
        for symbol in shared_symbols
        if symbol in calibration_features
    ]
    loss_row = {
        method: float(values["rmse"])
        for method, values in forecast_errors.items()
        if values["rmse"] is not None
    }
    return (
        matured_agents,
        observation,
        training_rows,
        settlements,
        calibration_rows,
        relative_calibration_rows,
        loss_row,
    )


def _market_horizon_forecast(
    research: ResearchData,
    decision_index: int,
    horizon_sessions: int,
    lookback_windows: int,
) -> float:
    """Use only completed BaoStock index windows to forecast the horizon return."""
    if lookback_windows < 1:
        raise ValueError("市场收益回看窗口必须为正")
    index_by_date = {row["trade_date"]: row for row in research.index_bars}
    last_historical_decision = decision_index - horizon_sessions
    first_historical_decision = max(0, last_historical_decision - lookback_windows + 1)
    returns = []
    for historical_index in range(first_historical_decision, last_historical_decision + 1):
        entry = index_by_date.get(research.trading_dates[historical_index + 1])
        exit_bar = index_by_date.get(research.trading_dates[historical_index + horizon_sessions])
        if entry is None or exit_bar is None or not entry.get("open") or not exit_bar.get("close"):
            continue
        returns.append(float(exit_bar["close"]) / float(entry["open"]) - 1)
    if not returns:
        raise ValueError("当前决策日前没有已成熟的沪深300同期限收益历史")
    return statistics.fmean(returns)


def _return_calibration_buckets(
    symbols: list[str],
    predictions: dict[str, float],
    outcomes: dict[str, float],
    bucket_count: int,
) -> list[dict[str, float | int]]:
    """Summarize equal-count forecast groups within one date cross-section."""
    if bucket_count < 1:
        raise ValueError("预测收益校准分桶数必须为正")
    ordered = sorted(symbols, key=lambda symbol: (float(predictions[symbol]), symbol))
    by_bucket: dict[int, list[str]] = defaultdict(list)
    for rank, symbol in enumerate(ordered):
        bucket = min(bucket_count - 1, rank * bucket_count // max(len(ordered), 1))
        by_bucket[bucket].append(symbol)
    return [
        {
            "bucket": bucket + 1,
            "symbol_count": len(bucket_symbols),
            "mean_predicted_return": statistics.fmean(
                float(predictions[symbol]) for symbol in bucket_symbols
            ),
            "mean_realized_return": statistics.fmean(
                outcomes[symbol] for symbol in bucket_symbols
            ),
        }
        for bucket, bucket_symbols in sorted(by_bucket.items())
        if bucket_symbols
    ]


def _fit_return_calibrators(
    training_features: list[tuple[float, ...]],
    training_returns: list[float],
    sufficient_statistics: list[list[float]] | None = None,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, object]]:
    """Fit shrinkage linear score-to-return maps using only matured labels."""
    if len(training_features) != len(training_returns):
        raise ValueError("校准训练特征与成熟标签数量不一致")
    moments = sufficient_statistics
    if moments is None:
        moments = _extend_calibration_moments(
            [[0.0] * 6 for _ in AGENT_IDS], training_features, training_returns
        )
    if len(moments) != len(AGENT_IDS) or any(len(row) != 6 for row in moments):
        raise ValueError("校准充分统计量结构无效")
    calibrators = {}
    diagnostics = {}
    for column, agent_id in enumerate(AGENT_IDS):
        count, sum_x, sum_y, sum_x_squared, sum_xy, sum_y_squared = moments[column]
        if count <= 0:
            calibrators[agent_id] = (0.0, 0.0, 0.0)
            diagnostics[agent_id] = {"status": "waiting_for_mature_labels", "rows": 0}
            continue
        mean_x = sum_x / count
        mean_y = sum_y / count
        numerator = sum_xy - sum_x * sum_y / count
        denominator = max(sum_x_squared - sum_x * sum_x / count, 0.0)
        # A fixed prior on the score slope avoids fitting noise from overlapping
        # cross-sectional labels as though every stock were an independent date.
        slope = numerator / (denominator + count * 0.04)
        centered_y_squared = max(sum_y_squared - sum_y * sum_y / count, 0.0)
        residual_sum = max(
            centered_y_squared - 2 * slope * numerator + slope * slope * denominator,
            0.0,
        )
        rmse = math.sqrt(residual_sum / count)
        calibrators[agent_id] = (mean_x, mean_y, slope)
        diagnostics[agent_id] = {
            "status": "matured_history",
            "rows": int(count),
            "mean_return": mean_y,
            "score_slope": slope,
            "training_fit_rmse": rmse,
        }
    return calibrators, diagnostics


def _extend_calibration_moments(
    moments: list[list[float]],
    features: list[tuple[float, ...]],
    returns: list[float],
) -> list[list[float]]:
    """Add matured rows to per-agent O(1)-per-date calibration statistics."""
    if len(features) != len(returns):
        raise ValueError("校准训练特征与成熟标签数量不一致")
    if len(moments) != len(AGENT_IDS) or any(len(row) != 6 for row in moments):
        raise ValueError("校准充分统计量结构无效")
    for row, target in zip(features, returns, strict=True):
        if len(row) != len(AGENT_IDS):
            raise ValueError("校准样本的 Agent 特征维数无效")
        y = float(target)
        for column in range(len(AGENT_IDS)):
            x = float(row[column])
            current = moments[column]
            current[0] += 1
            current[1] += x
            current[2] += y
            current[3] += x * x
            current[4] += x * y
            current[5] += y * y
    return moments


def _loss_gate(
    matured_losses: list[dict[str, float]],
    reliability_window: int,
    reliability_minimum: int,
    equal_weight_share: float,
) -> tuple[dict[str, float], str]:
    rows = matured_losses[-reliability_window:]
    usable = [row for row in rows if all(agent_id in row for agent_id in AGENT_IDS)]
    if len(usable) < reliability_minimum:
        return {agent_id: 1 / len(AGENT_IDS) for agent_id in AGENT_IDS}, "equal_weight_warmup"
    mean_error = {
        agent_id: statistics.fmean(row[agent_id] for row in usable)
        for agent_id in AGENT_IDS
    }
    inverse = {agent_id: 1 / max(mean_error[agent_id], 1e-6) for agent_id in AGENT_IDS}
    total = math.fsum(inverse.values())
    learned = {agent_id: inverse[agent_id] / total for agent_id in AGENT_IDS}
    return (
        {
            agent_id: equal_weight_share / len(AGENT_IDS)
            + (1 - equal_weight_share) * learned[agent_id]
            for agent_id in AGENT_IDS
        },
        "matured_prediction_loss",
    )


def _matured_error_estimates(
    matured_losses: list[dict[str, float]],
    reliability_window: int,
    reliability_minimum: int,
) -> tuple[dict[str, float | None], str]:
    rows = [
        row for row in matured_losses[-reliability_window:]
        if all(agent_id in row for agent_id in AGENT_IDS)
    ]
    if len(rows) < reliability_minimum:
        return {}, "warming_up_no_matured_oos_error"
    methods = set.intersection(*(set(row) for row in rows))
    return (
        {
            method: statistics.fmean(float(row[method]) for row in rows)
            for method in sorted(methods)
        },
        "matured_out_of_sample_errors",
    )


def _risk_estimates(
    research: ResearchData,
    decision_index: int,
    symbols: set[str],
    lookback_sessions: int,
    minimum_observations: int,
) -> tuple[dict[str, float], dict[str, float], float]:
    cached_indexes = getattr(research, "_p8_risk_history_indexes", None)
    if cached_indexes is None:
        returns_by_symbol: dict[str, dict[date, float]] = defaultdict(dict)
        for row in research.adjusted_bars:
            if row["return_1d"] is not None:
                returns_by_symbol[str(row["symbol"])][row["trade_date"]] = (
                    float(row["return_1d"]) / 100
                )
        index_by_date = {row["trade_date"]: row for row in research.index_bars}
        cached_indexes = (dict(returns_by_symbol), index_by_date)
        research._p8_risk_history_indexes = cached_indexes
    returns_by_symbol, index_by_date = cached_indexes
    history_dates = research.trading_dates[
        max(0, decision_index - lookback_sessions + 1) : decision_index + 1
    ]
    market_return_by_date = {}
    for history_position, day in enumerate(history_dates):
        index_bar = index_by_date.get(day)
        if index_bar is None:
            continue
        if index_bar.get("return_1d") is not None:
            market_return_by_date[day] = float(index_bar["return_1d"]) / 100
        elif history_position > 0:
            previous_index_bar = index_by_date.get(history_dates[history_position - 1])
            if previous_index_bar and previous_index_bar.get("close"):
                market_return_by_date[day] = (
                    float(index_bar["close"]) / float(previous_index_bar["close"]) - 1
                )
    market_returns = [
        market_return_by_date[day]
        for day in history_dates
        if day in market_return_by_date
    ]
    if len(market_returns) < minimum_observations or statistics.pstdev(market_returns) == 0:
        return {}, {}, 0.0
    market_mean = statistics.fmean(market_returns)
    market_variance = statistics.fmean((value - market_mean) ** 2 for value in market_returns)
    market_volatility = math.sqrt(market_variance * 250)
    result = {}
    betas = {}
    for symbol in sorted(symbols):
        paired = [
            (stock_return, market_return_by_date[day])
            for day in history_dates
            if (stock_return := returns_by_symbol.get(symbol, {}).get(day)) is not None
            and day in market_return_by_date
        ]
        if len(paired) < minimum_observations:
            continue
        stock_returns, paired_market = zip(*paired, strict=True)
        stock_mean = statistics.fmean(stock_returns)
        paired_market_mean = statistics.fmean(paired_market)
        paired_market_variance = statistics.fmean(
            (value - paired_market_mean) ** 2 for value in paired_market
        )
        if paired_market_variance <= 0:
            continue
        beta = statistics.fmean(
            (stock - stock_mean) * (market - paired_market_mean)
            for stock, market in paired
        ) / paired_market_variance
        result[symbol] = statistics.pstdev(stock_returns) * math.sqrt(250)
        betas[symbol] = beta
    return result, betas, market_volatility


def run_mvp_research(
    research: ResearchData,
    config: dict[str, float | int],
    initial_state: dict[str, object] | None = None,
    as_of_date: date | None = None,
    ensure_as_of_decision: bool = False,
    comparison_start_date: date | None = None,
    progress_callback: Callable[[str], None] | None = None,
    decision_start_date: date | None = None,
) -> dict[str, object]:
    step = int(config["decision_step_sessions"])
    horizon = int(config.get("prediction_horizon_sessions", 5))
    if step < 1 or horizon < 1:
        raise ValueError("决策步长和预测期限必须至少为 1 个交易日")
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
    maximum_target_change = float(config.get("maximum_target_one_way_change", maximum_turnover))

    first_signal_index = max(
        trend_window,
        reversal_window,
        liquidity_return_window,
        recent_amount + reference_amount - 1,
    )
    if as_of_date is not None:
        if as_of_date not in research.trading_dates:
            raise ValueError(f"指定决策日不在可用研究交易日历中：{as_of_date}")
        end_index = research.trading_dates.index(as_of_date)
    else:
        # Preserve historical non-daily replays, whose final prediction window
        # previously had to mature inside the data. Daily runs retain their last
        # pending decision; an explicit as_of call always includes that date.
        end_index = (
            len(research.trading_dates) - 1
            if step == 1
            else len(research.trading_dates) - horizon - 1
        )
    decision_indices = list(range(first_signal_index, end_index + 1, step))
    if comparison_start_date is not None and comparison_start_date not in research.trading_dates:
        raise ValueError(f"共同评价起始日不在研究交易日历中：{comparison_start_date}")
    if decision_start_date is not None:
        if decision_start_date not in research.trading_dates:
            raise ValueError(f"预测起始日不在研究交易日历中：{decision_start_date}")
        decision_start_index = research.trading_dates.index(decision_start_date)
        decision_indices = [index for index in decision_indices if index >= decision_start_index]
    if (
        end_index >= first_signal_index
        and (ensure_as_of_decision or (as_of_date is None and step == 1))
        and (not decision_indices or decision_indices[-1] != end_index)
    ):
        decision_indices.append(end_index)
    if not decision_indices:
        raise ValueError("训练区间不足以形成预测决策窗口")

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
    calibration_features = [
        tuple(row) for row in initial_state.get("calibration_features", [])
    ]
    calibration_returns = [
        float(value) for value in initial_state.get("calibration_returns", [])
    ]
    relative_calibration_features = [
        tuple(row) for row in initial_state.get("relative_calibration_features", [])
    ]
    relative_calibration_returns = [
        float(value) for value in initial_state.get("relative_calibration_returns", [])
    ]
    calibration_sufficient_statistics = [
        [float(value) for value in row]
        for row in initial_state.get("calibration_sufficient_statistics", [])
    ]
    if not calibration_sufficient_statistics:
        calibration_sufficient_statistics = _extend_calibration_moments(
            [[0.0] * 6 for _ in AGENT_IDS],
            calibration_features,
            calibration_returns,
        )
    relative_calibration_sufficient_statistics = [
        [float(value) for value in row]
        for row in initial_state.get("relative_calibration_sufficient_statistics", [])
    ]
    if not relative_calibration_sufficient_statistics:
        relative_calibration_sufficient_statistics = _extend_calibration_moments(
            [[0.0] * 6 for _ in AGENT_IDS],
            relative_calibration_features,
            relative_calibration_returns,
        )
    matured_losses: list[dict[str, float]] = [
        {str(name): float(value) for name, value in row.items()}
        for row in initial_state.get("matured_losses", [])
    ]
    observations = []
    decision_log = []
    prediction_decisions = []
    prediction_settlements = []
    decisions_by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    previous_targets: dict[str, dict[str, float]] = defaultdict(dict)
    pending_windows = []
    static_weights = None
    comparison_started = False

    for decision_count, decision_index in enumerate(decision_indices, start=1):
        decision_date = research.trading_dates[decision_index]
        if progress_callback and (decision_count == 1 or decision_count % 50 == 0 or decision_count == len(decision_indices)):
            progress_callback(
                f"research decision {decision_count}/{len(decision_indices)} at {decision_date.isoformat()}"
            )

        # Only labels whose exit close is visible by this after-close decision mature here.
        still_pending = []
        for pending in pending_windows:
            if pending["exit_date"] is None or pending["exit_date"] > decision_date:
                still_pending.append(pending)
                continue
            matured_row, observation, training_rows, settlements, calibration_rows, relative_calibration_rows, loss_row = _evaluate_matured_window(
                research, pending, top_fraction, decision_date
            )
            prediction_settlements.extend(settlements)
            if matured_row:
                matured_rank_ics.append(matured_row)
                matured_by_state[pending["regime"]].append(matured_row)
                training_features.extend(features for features, _ in training_rows)
                training_returns.extend(value for _, value in training_rows)
                calibration_features.extend(features for features, _ in calibration_rows)
                calibration_returns.extend(value for _, value in calibration_rows)
                calibration_sufficient_statistics = _extend_calibration_moments(
                    calibration_sufficient_statistics,
                    [features for features, _ in calibration_rows],
                    [value for _, value in calibration_rows],
                )
                relative_calibration_features.extend(features for features, _ in relative_calibration_rows)
                relative_calibration_returns.extend(value for _, value in relative_calibration_rows)
                relative_calibration_sufficient_statistics = _extend_calibration_moments(
                    relative_calibration_sufficient_statistics,
                    [features for features, _ in relative_calibration_rows],
                    [value for _, value in relative_calibration_rows],
                )
                if loss_row:
                    matured_losses.append(loss_row)
                observations.append(observation)
        pending_windows = still_pending

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
            horizon_sessions=horizon,
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

        calibrators, calibration_diagnostics = _fit_return_calibrators(
            calibration_features,
            calibration_returns,
            calibration_sufficient_statistics,
        )
        calibrated_agents = {
            agent_id: {
                symbol: mean_return + slope * (percentile_by_agent[agent_id][symbol] - mean_score)
                for symbol in common_symbols
            }
            for agent_id, (mean_score, mean_return, slope) in calibrators.items()
        }
        relative_calibrators, relative_calibration_diagnostics = _fit_return_calibrators(
            relative_calibration_features,
            relative_calibration_returns,
            relative_calibration_sufficient_statistics,
        )
        calibrated_relative_agents = {
            agent_id: {
                symbol: mean_return + slope * (percentile_by_agent[agent_id][symbol] - mean_score)
                for symbol in common_symbols
            }
            for agent_id, (mean_score, mean_return, slope) in relative_calibrators.items()
        }
        market_expected_return = _market_horizon_forecast(
            research,
            decision_index,
            horizon,
            int(config.get("market_return_lookback_windows", 60)),
        )
        market_only_returns = {
            symbol: market_expected_return for symbol in common_symbols
        }
        market_relative_returns = {
            symbol: market_expected_return
            + statistics.fmean(
                calibrated_relative_agents[agent_id][symbol] for agent_id in AGENT_IDS
            )
            for symbol in common_symbols
        }
        loss_gate_weights, loss_gate_source = _loss_gate(
            matured_losses,
            reliability_window,
            reliability_minimum,
            equal_weight_share,
        )

        def fuse_calibrated(weights: dict[str, float]) -> dict[str, float]:
            return {
                symbol: math.fsum(
                    weights[agent_id] * calibrated_agents[agent_id][symbol]
                    for agent_id in AGENT_IDS
                )
                for symbol in common_symbols
            }

        calibrated_equal = fuse_calibrated(
            {agent_id: 1 / len(AGENT_IDS) for agent_id in AGENT_IDS}
        )
        calibrated_static = fuse_calibrated(static_for_decision)
        calibrated_adaptive = fuse_calibrated(current_adaptive_weights)
        calibrated_regime = fuse_calibrated(current_regime_weights)
        calibrated_loss_gated = fuse_calibrated(loss_gate_weights)
        matured_oos_errors, forecast_error_status = _matured_error_estimates(
            matured_losses, reliability_window, reliability_minimum
        )

        calibrated_returns_by_method = {
            **calibrated_agents,
            "market_only": market_only_returns,
            "market_plus_relative_fusion": market_relative_returns,
            "ridge": ridge_scores,
            "equal_fusion": calibrated_equal,
            "static_fusion": calibrated_static,
            "adaptive_fusion": calibrated_adaptive,
            "regime_fusion": calibrated_regime,
            "loss_gated_fusion": calibrated_loss_gated,
        }
        forecast_errors_by_method = {
            method: matured_oos_errors.get(method)
            for method in calibrated_returns_by_method
        }

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
            "loss_gated_fusion": percentile_scores(calibrated_loss_gated),
            "market_plus_relative_fusion": percentile_scores(market_relative_returns),
        }

        risk_symbols = common_symbols
        annualized_volatility_by_symbol, factor_beta_by_symbol, market_annualized_volatility = _risk_estimates(
            research,
            decision_index,
            risk_symbols,
            risk_window,
            risk_minimum_observations,
        )

        # Comparison starts only after both supervised and adaptive systems have
        # a real matured training history; earlier signals are warm-up only.
        if (
            len(training_returns) >= ridge_minimum_rows
            and len(matured_rank_ics) >= reliability_minimum
            and annualized_volatility_by_symbol
            and (comparison_start_date is None or decision_date >= comparison_start_date)
        ):
            comparison_started = True
            for method, scores in score_by_method.items():
                if not scores:
                    continue
                target = top_fraction_weights(scores, portfolio_fraction, invested_weight)
                target = apply_weight_and_turnover_limits(
                    target,
                    previous_targets[method],
                    maximum_symbol_weight,
                    maximum_target_change,
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
                maximum_target_change,
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
        decision_position = research.trading_dates.index(decision_date)
        entry_position = decision_position + 1
        exit_position = decision_position + horizon
        entry_date = (
            research.trading_dates[entry_position]
            if entry_position < len(research.trading_dates)
            else None
        )
        exit_date = (
            research.trading_dates[exit_position]
            if exit_position < len(research.trading_dates)
            else None
        )
        targets_for_date = {
            method: decisions[-1]["target_weights"]
            for method, decisions in decisions_by_method.items()
            if decisions and decisions[-1]["decision_date"] == decision_date
        }
        candidate_symbols = sorted(feature_by_symbol)
        candidate_symbols_sha256 = hashlib.sha256(
            "\n".join(candidate_symbols).encode("ascii")
        ).hexdigest()
        prediction_decisions.append(
            {
                "prediction_date": decision_date.isoformat(),
                "input_cutoff": decision_date.isoformat(),
                "entry_date": entry_date.isoformat() if entry_date else None,
                "planned_exit_date": exit_date.isoformat() if exit_date else None,
                "horizon_sessions": horizon,
                "status": "pending" if exit_date is None or exit_date > research.trading_dates[end_index] else "awaiting_settlement",
                "universe_size": len(snapshot.members),
                "scored_symbols": candidate_symbols,
                "candidate_symbols_sha256": candidate_symbols_sha256,
                "candidate_symbols_count": len(feature_by_symbol),
                "scores_by_method": {
                    method: scores
                    for method, scores in score_by_method.items()
                },
                "target_weights_by_method": targets_for_date,
                "expected_returns_by_method": calibrated_returns_by_method,
                "forecast_errors_by_method": forecast_errors_by_method,
                "forecast_error_status": forecast_error_status,
                "calibration": calibration_diagnostics,
                "relative_calibration": relative_calibration_diagnostics,
                "market_expected_return": market_expected_return,
                "loss_gate_weights": loss_gate_weights,
                "loss_gate_source": loss_gate_source,
                "annualized_volatility_by_symbol": annualized_volatility_by_symbol,
                "factor_beta_by_symbol": factor_beta_by_symbol,
                "market_annualized_volatility": market_annualized_volatility,
            }
        )
        pending_windows.append({
            "decision_date": decision_date,
            "exit_date": exit_date,
            "horizon_sessions": horizon,
            "features_by_symbol": feature_by_symbol,
            "percentile_features_by_symbol": {
                symbol: tuple(percentile_by_agent[agent_id][symbol] for agent_id in AGENT_IDS)
                for symbol in sorted(common_symbols)
            },
            "candidate_symbols_sha256": candidate_symbols_sha256,
            "scores_by_method": score_by_method,
            "expected_returns_by_method": calibrated_returns_by_method,
            "regime": current_regime,
            "adaptive_weights": current_adaptive_weights,
            "regime_weights": current_regime_weights,
            "regime_weight_source": regime_weight_source,
            "regime_matured_windows": regime_history_count,
        })

    label_available_by = as_of_date or research.trading_dates[-1]
    for pending in pending_windows:
        if pending["exit_date"] is not None and pending["exit_date"] <= label_available_by:
            matured_row, observation, training_rows, settlements, calibration_rows, relative_calibration_rows, loss_row = _evaluate_matured_window(
                research, pending, top_fraction, label_available_by
            )
            prediction_settlements.extend(settlements)
            if matured_row:
                matured_rank_ics.append(matured_row)
                matured_by_state[pending["regime"]].append(matured_row)
                training_features.extend(features for features, _ in training_rows)
                training_returns.extend(value for _, value in training_rows)
                calibration_features.extend(features for features, _ in calibration_rows)
                calibration_returns.extend(value for _, value in calibration_rows)
                relative_calibration_features.extend(features for features, _ in relative_calibration_rows)
                relative_calibration_returns.extend(value for _, value in relative_calibration_rows)
                calibration_sufficient_statistics = _extend_calibration_moments(
                    calibration_sufficient_statistics,
                    [features for features, _ in calibration_rows],
                    [value for _, value in calibration_rows],
                )
                relative_calibration_sufficient_statistics = _extend_calibration_moments(
                    relative_calibration_sufficient_statistics,
                    [features for features, _ in relative_calibration_rows],
                    [value for _, value in relative_calibration_rows],
                )
                if loss_row:
                    matured_losses.append(loss_row)
                observations.append(observation)
            continue
        decision_position = research.trading_dates.index(pending["decision_date"])
        entry_position = decision_position + 1
        entry_date = research.trading_dates[entry_position] if entry_position < len(research.trading_dates) else None
        for symbol in sorted(pending["features_by_symbol"]):
            prediction_settlements.append(
                {
                    "prediction_date": pending["decision_date"].isoformat(),
                    "symbol": symbol,
                    "entry_date": entry_date.isoformat() if entry_date else None,
                    "planned_exit_date": pending["exit_date"].isoformat() if pending["exit_date"] else None,
                    "available_at": pending["exit_date"].isoformat() if pending["exit_date"] else None,
                    "horizon_sessions": pending["horizon_sessions"],
                    "status": "pending",
                    "candidate_symbols_sha256": pending["candidate_symbols_sha256"],
                    f"forward_return_{pending['horizon_sessions']}s": None,
                }
            )

    if not comparison_started:
        raise ValueError("训练历史不足以形成可比较的多 Agent 组合实验")

    quality_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    quality_buckets: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for observation in observations:
        for method, metrics in observation["forecast_errors"].items():
            if metrics["rmse"] is None:
                continue
            quality_rows[method].append(
                {
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "rank_ic": float(metrics["rank_ic"])
                    if metrics["rank_ic"] is not None else None,
                }
            )
            for bucket in metrics.get("calibration_buckets", []):
                quality_buckets[method][int(bucket["bucket"])].append(
                    {
                        "predicted": float(bucket["mean_predicted_return"]),
                        "realized": float(bucket["mean_realized_return"]),
                    }
                )
    prediction_quality = {
        method: {
            "matured_decision_dates": len(rows),
            "mean_date_mae": statistics.fmean(row["mae"] for row in rows),
            "mean_date_rmse": statistics.fmean(row["rmse"] for row in rows),
            "mean_date_rank_ic": (
                statistics.fmean(
                    row["rank_ic"] for row in rows if row["rank_ic"] is not None
                )
                if any(row["rank_ic"] is not None for row in rows) else None
            ),
            "defined_date_rank_ic_count": sum(row["rank_ic"] is not None for row in rows),
            "return_calibration_by_forecast_bucket": {
                str(bucket): {
                    "matured_decision_dates": len(bucket_rows),
                    "mean_date_predicted_return": statistics.fmean(
                        row["predicted"] for row in bucket_rows
                    ),
                    "mean_date_realized_return": statistics.fmean(
                        row["realized"] for row in bucket_rows
                    ),
                    "mean_date_calibration_gap": statistics.fmean(
                        row["realized"] - row["predicted"] for row in bucket_rows
                    ),
                }
                for bucket, bucket_rows in sorted(quality_buckets[method].items())
            },
        }
        for method, rows in quality_rows.items()
    }

    return {
        "protocol": {
            "universe": "historical BaoStock membership snapshots as visible at each decision date",
            "decision": "after close; targets execute at next session open",
            "daily_marking": "adjusted open/close price ratios; suspended positions carry last value",
            "execution_boundary": "unadjusted BaoStock tradestatus; no board-limit queue model",
            "label": f"next-session open to {horizon}-session close adjusted-price return",
            "prediction_horizon_sessions": horizon,
            "decision_step_sessions": step,
            "top_fraction": portfolio_fraction,
            "invested_weight": invested_weight,
            "maximum_symbol_weight": maximum_symbol_weight,
            "maximum_one_way_turnover": maximum_turnover,
            "maximum_target_one_way_change": maximum_target_change,
            "liquidity_limit": "order notional capped by trailing mean traded amount times configured participation",
            "reliability": f"only matured {horizon}-session Rank IC before each after-close decision",
            "static_weights": "first configured number of matured development windows, then frozen",
            "regime": f"risk_on when CSI 300 {regime_window}-session price return is nonnegative; otherwise risk_off",
            "regime_fallback": "global reliability until the state has enough matured windows",
            "ridge": "three existing Agent scores, standardized using matured training rows only",
            "risk_allocator": "top-decile eligibility, inverse trailing volatility, weight and turnover caps",
            "return_calibration": "shrinkage linear score-to-return maps trained only on matured labels; prediction quality includes five equal-count forecast buckets averaged equally by decision date",
            "forecast_error": "rolling date-level RMSE from frozen predictions after their labels mature",
            "forecast_error_multiplier": float(config.get("forecast_error_multiplier", 0.5)),
            "market_return": "trailing mean of completed BaoStock CSI 300 next-open-to-horizon-close returns; only labels mature by the decision close are included",
            "market_relative_forecast": "market-only forecast plus equal-weight cross-sectional expert calibration against stock return minus the same-horizon CSI 300 return; calibrated only on matured dates",
            "market_return_lookback_windows": int(config.get("market_return_lookback_windows", 60)),
            "loss_gate": "inverse matured out-of-sample prediction RMSE with equal-weight shrinkage",
            "joint_optimizer": "expected return less matured forecast error, factor-shrinkage risk penalty, and actual holding-to-target transaction cost; cash, concentration, risk, capacity, and one-way turnover constrained",
            "comparison_start_date": (comparison_start_date.isoformat() if comparison_start_date else None),
            "account_start_state": "cash; target turnover state begins at the common comparison start date",
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
        "prediction_quality": prediction_quality,
        "prediction_decisions": prediction_decisions,
        "prediction_settlements": prediction_settlements,
        "decisions_by_method": dict(decisions_by_method),
        "decision_log": decision_log,
        "training_rows_at_end": len(training_returns),
        "matured_rank_ic_windows_at_end": len(matured_rank_ics),
        "training_seed": {
            "training_features": training_features,
            "training_returns": training_returns,
            "calibration_features": calibration_features,
            "calibration_returns": calibration_returns,
            "calibration_sufficient_statistics": calibration_sufficient_statistics,
            "relative_calibration_features": relative_calibration_features,
            "relative_calibration_returns": relative_calibration_returns,
            "relative_calibration_sufficient_statistics": relative_calibration_sufficient_statistics,
            "matured_losses": matured_losses,
            "matured_rank_ics": matured_rank_ics,
            "matured_by_state": dict(matured_by_state),
        },
    }


def run_portfolio_comparisons(
    research: ResearchData,
    result: dict[str, object],
    config: dict[str, float | int],
    execution_protocol: str = "continuous_v1",
    progress_callback: Callable[[str], None] | None = None,
    common_start_date: date | None = None,
    methods: set[str] | None = None,
    run_sensitivities: bool = True,
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
    risk_sensitivity = {"15pct_budget": {}, "25pct_budget": {}}
    common_decisions = {
        method: [
            row for row in decisions
            if common_start_date is None or row["decision_date"] >= common_start_date
        ]
        for method, decisions in result["decisions_by_method"].items()
        if methods is None or method in methods
    }
    if execution_protocol == "continuous_v1":
        simulator = simulate_portfolio
    elif execution_protocol == "fixed_horizon_v2":
        simulator = simulate_fixed_horizon_portfolio
    else:
        raise ValueError(f"未知组合执行协议：{execution_protocol}")
    for method, decisions in common_decisions.items():
        if not decisions:
            continue
        if progress_callback:
            progress_callback(f"portfolio replay started: {method}")
        simulation_arguments = (
            research,
            decisions,
            initial_capital,
            costs,
            liquidity_lookback,
            maximum_participation,
        )
        if execution_protocol == "fixed_horizon_v2":
            simulated = simulator(
                *simulation_arguments,
                horizon_sessions=int(config.get("prediction_horizon_sessions", 5)),
                evaluation_start_date=common_start_date,
            )
        else:
            simulated = simulator(
                *simulation_arguments,
                maximum_one_way_turnover=float(config["maximum_one_way_turnover"]),
                progress_callback=(
                    (lambda message, name=method: progress_callback(f"{name}: {message}"))
                    if progress_callback else None
                ),
                evaluation_start_date=common_start_date,
            )
        comparisons[method] = simulated.pop("summary")
        comparisons[method]["paired_block_bootstrap_vs_benchmark"] = paired_block_bootstrap(
            [float(row["daily_return"]) for row in simulated["daily_nav"]],
            [float(row["benchmark_return"]) for row in simulated["daily_nav"]],
        )
        artifacts[method] = simulated
        if run_sensitivities:
            for scenario, multiplier in (("0x", 0.0), ("2x", 2.0)):
                scenario_costs = {
                    name: value * multiplier for name, value in costs.items()
                }
                scenario_arguments = (
                    research,
                    decisions,
                    initial_capital,
                    scenario_costs,
                    liquidity_lookback,
                    maximum_participation,
                )
                if execution_protocol == "fixed_horizon_v2":
                    sensitivity[scenario][method] = simulator(
                        *scenario_arguments,
                        horizon_sessions=int(config.get("prediction_horizon_sessions", 5)),
                        evaluation_start_date=common_start_date,
                    )["summary"]
                else:
                    sensitivity[scenario][method] = simulator(
                        *scenario_arguments,
                        maximum_one_way_turnover=float(config["maximum_one_way_turnover"]),
                        progress_callback=(
                            (lambda message, name=f"{method} {scenario}": progress_callback(f"{name}: {message}"))
                            if progress_callback else None
                        ),
                        evaluation_start_date=common_start_date,
                    )["summary"]
        if progress_callback:
            progress_callback(f"portfolio replay complete: {method}")

    if execution_protocol == "continuous_v1":
        forecasts = [
            row
            for row in result["prediction_decisions"]
            if row["target_weights_by_method"]
            and (
                common_start_date is None
                or date.fromisoformat(row["prediction_date"]) >= common_start_date
            )
        ]
        simulated = simulate_portfolio(
            research,
            [],
            initial_capital,
            costs,
            liquidity_lookback,
            maximum_participation,
            float(config["maximum_one_way_turnover"]),
            forecast_decisions=forecasts,
            optimizer_config=config,
            progress_callback=(
                (lambda message: progress_callback(f"joint_optimization: {message}"))
                if progress_callback else None
            ),
            evaluation_start_date=common_start_date,
        )
        summary = simulated.pop("summary")
        summary["paired_block_bootstrap_vs_benchmark"] = paired_block_bootstrap(
            [float(row["daily_return"]) for row in simulated["daily_nav"]],
            [float(row["benchmark_return"]) for row in simulated["daily_nav"]],
        )
        comparisons["joint_optimization"] = summary
        artifacts["joint_optimization"] = simulated
        if run_sensitivities:
            for scenario, multiplier in (("0x", 0.0), ("2x", 2.0)):
                scenario_costs = {name: value * multiplier for name, value in costs.items()}
                scenario_config = dict(config)
                for name in (
                "commission_per_side_bps",
                "transfer_fee_per_side_bps",
                "stamp_duty_sell_bps",
                "slippage_per_side_bps",
                ):
                    scenario_config[name] = float(config[name]) * multiplier
                scenario_result = simulate_portfolio(
                research,
                [],
                initial_capital,
                scenario_costs,
                liquidity_lookback,
                maximum_participation,
                float(config["maximum_one_way_turnover"]),
                forecast_decisions=forecasts,
                optimizer_config=scenario_config,
                progress_callback=(
                    (lambda message, name=scenario: progress_callback(
                        f"joint_optimization {name}: {message}"
                    ))
                    if progress_callback else None
                ),
                evaluation_start_date=common_start_date,
                )
                sensitivity[scenario]["joint_optimization"] = scenario_result["summary"]

        relative_forecasts = []
        for forecast in forecasts:
            relative_error = forecast["forecast_errors_by_method"].get(
                "market_plus_relative_fusion"
            )
            if relative_error is None:
                continue
            relative_forecasts.append(
                {
                    **forecast,
                    "expected_returns_by_method": {
                        **forecast["expected_returns_by_method"],
                        "loss_gated_fusion": forecast["expected_returns_by_method"][
                            "market_plus_relative_fusion"
                        ],
                    },
                    "forecast_errors_by_method": {
                        **forecast["forecast_errors_by_method"],
                        "loss_gated_fusion": relative_error,
                    },
                }
            )
        if len(relative_forecasts) != len(forecasts):
            raise ValueError(
                "共同评价窗中市场加相对收益预测没有完整成熟误差；拒绝与绝对收益优化器作不配对比较"
            )
        relative_simulation = simulate_portfolio(
            research,
            [],
            initial_capital,
            costs,
            liquidity_lookback,
            maximum_participation,
            float(config["maximum_one_way_turnover"]),
            forecast_decisions=relative_forecasts,
            optimizer_config=config,
            progress_callback=(
                (lambda message: progress_callback(f"joint_market_relative: {message}"))
                if progress_callback else None
            ),
            evaluation_start_date=common_start_date,
        )
        relative_summary = relative_simulation.pop("summary")
        relative_summary["paired_block_bootstrap_vs_benchmark"] = paired_block_bootstrap(
            [float(row["daily_return"]) for row in relative_simulation["daily_nav"]],
            [float(row["benchmark_return"]) for row in relative_simulation["daily_nav"]],
        )
        comparisons["joint_market_relative"] = relative_summary
        artifacts["joint_market_relative"] = relative_simulation
        if run_sensitivities:
            for scenario, multiplier in (("0x", 0.0), ("2x", 2.0)):
                scenario_costs = {name: value * multiplier for name, value in costs.items()}
                scenario_config = dict(config)
                for name in (
                "commission_per_side_bps",
                "transfer_fee_per_side_bps",
                "stamp_duty_sell_bps",
                "slippage_per_side_bps",
                ):
                    scenario_config[name] = float(config[name]) * multiplier
                relative_cost_result = simulate_portfolio(
                research,
                [],
                initial_capital,
                scenario_costs,
                liquidity_lookback,
                maximum_participation,
                float(config["maximum_one_way_turnover"]),
                forecast_decisions=relative_forecasts,
                optimizer_config=scenario_config,
                progress_callback=(
                    (lambda message, label=scenario: progress_callback(
                        f"joint_market_relative {label}: {message}"
                    ))
                    if progress_callback else None
                ),
                evaluation_start_date=common_start_date,
                )
                sensitivity[scenario]["joint_market_relative"] = relative_cost_result["summary"]
        result["decisions_by_method"]["joint_market_relative"] = [
            {
                "decision_date": date.fromisoformat(row["decision_date"]),
                "target_weights": row["target_weights"],
            }
            for row in relative_simulation["decisions"]
            if row["execution_date"] is not None
        ]
        for prediction in result["prediction_decisions"]:
            optimized = {
                row["decision_date"]: row
                for row in relative_simulation["decisions"]
            }.get(prediction["prediction_date"])
            if optimized is not None:
                prediction["target_weights_by_method"]["joint_market_relative"] = optimized[
                    "target_weights"
                ]
                prediction["joint_market_relative"] = {
                    key: optimized[key]
                    for key in (
                        "gross_expected_return",
                        "forecast_error_allowance",
                        "annualized_risk",
                        "objective_value",
                        "holding_objective_value",
                        "incremental_objective_value",
                        "risk_penalty_weight",
                        "portfolio_horizon_sigma",
                        "forecast_error_allowance_over_horizon_sigma",
                        "risk_adjusted_net_incremental_value_cny",
                    )
                }

        for scenario, risk_budget in (
            (("15pct_budget", 0.15), ("25pct_budget", 0.25))
            if run_sensitivities else ()
        ):
            scenario_config = dict(config)
            scenario_config["risk_budget_annualized"] = risk_budget
            for method_name, method_forecasts in (
                ("joint_optimization", forecasts),
                ("joint_market_relative", relative_forecasts),
            ):
                scenario_result = simulate_portfolio(
                    research,
                    [],
                    initial_capital,
                    costs,
                    liquidity_lookback,
                    maximum_participation,
                    float(config["maximum_one_way_turnover"]),
                    forecast_decisions=method_forecasts,
                    optimizer_config=scenario_config,
                    progress_callback=(
                        (lambda message, label=f"{method_name} {scenario}": progress_callback(
                            f"{label}: {message}"
                        ))
                        if progress_callback else None
                    ),
                    evaluation_start_date=common_start_date,
                )
                scenario_summary = scenario_result["summary"]
                scenario_summary["paired_block_bootstrap_vs_benchmark"] = paired_block_bootstrap(
                    [float(row["daily_return"]) for row in scenario_result["daily_nav"]],
                    [float(row["benchmark_return"]) for row in scenario_result["daily_nav"]],
                )
                scenario_summary["forecast_error_multiplier_unchanged"] = float(
                    scenario_config.get("forecast_error_multiplier", 0.5)
                )
                risk_sensitivity[scenario][method_name] = scenario_summary

        optimized_decisions = [
            {
                "decision_date": date.fromisoformat(row["decision_date"]),
                "target_weights": row["target_weights"],
            }
            for row in simulated["decisions"]
            if row["execution_date"] is not None
        ]
        result["decisions_by_method"]["joint_optimization"] = optimized_decisions
        optimized_by_date = {
            row["decision_date"]: row for row in simulated["decisions"]
        }
        for row in result["prediction_decisions"]:
            optimized = optimized_by_date.get(row["prediction_date"])
            if optimized is not None:
                row["target_weights_by_method"]["joint_optimization"] = optimized["target_weights"]
                row["joint_optimization"] = {
                    key: optimized[key]
                    for key in (
                        "current_cash_cny",
                        "current_holdings_cny",
                        "current_nav_cny",
                        "gross_expected_return",
                        "forecast_error_allowance",
                        "expected_transaction_cost_weight",
                        "annualized_risk",
                        "one_way_turnover",
                        "objective_value",
                        "target_cash_cny_before_costs",
                        "estimated_cash_after_costs_cny",
                        "buy_notional_cny",
                        "sell_notional_cny",
                        "holding_objective_value",
                        "incremental_objective_value",
                        "risk_penalty_weight",
                        "portfolio_horizon_sigma",
                        "forecast_error_allowance_over_horizon_sigma",
                        "risk_adjusted_net_incremental_value_cny",
                    )
                }
                decision_date = date.fromisoformat(row["prediction_date"])
                label_rows = forward_returns(
                    research,
                    [decision_date],
                    int(row["horizon_sessions"]),
                )
                realized_by_symbol = {
                    str(label["symbol"]): float(label["forward_return"])
                    for label in label_rows
                }
                current_nav = float(optimized["current_nav_cny"])
                current_weights = {
                    str(symbol): float(value) / current_nav
                    for symbol, value in optimized["current_holdings_cny"].items()
                    if current_nav > 0
                }
                target_weights = {
                    str(symbol): float(weight)
                    for symbol, weight in optimized["target_weights"].items()
                }
                target_realized = math.fsum(
                    weight * realized_by_symbol[symbol]
                    for symbol, weight in target_weights.items()
                    if symbol in realized_by_symbol
                )
                holding_realized = math.fsum(
                    weight * realized_by_symbol[symbol]
                    for symbol, weight in current_weights.items()
                    if symbol in realized_by_symbol
                )
                actual_symbols = set(realized_by_symbol)
                target_coverage = math.fsum(
                    weight for symbol, weight in target_weights.items()
                    if symbol in actual_symbols
                )
                hold_coverage = math.fsum(
                    weight for symbol, weight in current_weights.items()
                    if symbol in actual_symbols
                )
                label_index = research.trading_dates.index(decision_date) + int(row["horizon_sessions"])
                relation = {
                    "label_available_at": (
                        research.trading_dates[label_index].isoformat()
                        if label_index < len(research.trading_dates) else None
                    ),
                    "status": "settled" if realized_by_symbol else "pending_or_missing_label",
                    "prediction_horizon_sessions": int(row["horizon_sessions"]),
                    "target_weight_coverage": target_coverage,
                    "current_holding_weight_coverage": hold_coverage,
                    "predicted_gross_return": float(optimized["gross_expected_return"]),
                    "matured_forecast_error_allowance": float(optimized["forecast_error_allowance"]),
                    "estimated_transaction_cost": float(optimized["expected_transaction_cost_weight"]),
                    "portfolio_risk_penalty": float(optimized["risk_penalty_weight"]),
                    "annualized_portfolio_risk": float(optimized["annualized_risk"]),
                    "portfolio_horizon_sigma": float(optimized["portfolio_horizon_sigma"]),
                    "forecast_error_allowance_over_horizon_sigma": optimized[
                        "forecast_error_allowance_over_horizon_sigma"
                    ],
                    "predicted_risk_adjusted_net_incremental_return_vs_hold": float(
                        optimized["incremental_objective_value"]
                    ),
                    "realized_target_return_at_planned_weights": target_realized,
                    "realized_current_hold_return": holding_realized,
                    "realized_incremental_return_after_estimated_trade_cost": (
                        target_realized
                        - holding_realized
                        - float(optimized["expected_transaction_cost_weight"])
                    ),
                }
                relation["predicted_minus_realized_incremental_return"] = (
                    relation["predicted_risk_adjusted_net_incremental_return_vs_hold"]
                    - relation["realized_incremental_return_after_estimated_trade_cost"]
                )
                row["decision_level_cost_adjusted_prediction_relation"] = relation
        decision_log_by_date = {
            row["decision_date"]: row for row in result["decision_log"]
        }
        for record in simulated["decisions"]:
            log_row = decision_log_by_date.get(record["decision_date"])
            if log_row is not None:
                log_row["target_weights"]["joint_optimization"] = record["target_weights"]
                log_row["joint_actual_account"] = {
                    "cash_cny": record["current_cash_cny"],
                    "holdings_cny": record["current_holdings_cny"],
                    "nav_cny": record["current_nav_cny"],
                }
        result["comparison_decision_count"] = min(
            len(rows) for rows in result["decisions_by_method"].values()
        )
        matured_relations = [
            row["decision_level_cost_adjusted_prediction_relation"]
            for row in result["prediction_decisions"]
            if (
                row.get("decision_level_cost_adjusted_prediction_relation", {}).get("status")
                == "settled"
            )
        ]
        result["decision_value_quality"] = {
            "unit": "decision date; overlapping three- or five-session labels are averaged by decision date",
            "matured_decision_dates": len(matured_relations),
            "mean_predicted_risk_adjusted_net_incremental_return_vs_hold": (
                statistics.fmean(
                    float(row["predicted_risk_adjusted_net_incremental_return_vs_hold"])
                    for row in matured_relations
                ) if matured_relations else None
            ),
            "mean_realized_incremental_return_after_estimated_trade_cost": (
                statistics.fmean(
                    float(row["realized_incremental_return_after_estimated_trade_cost"])
                    for row in matured_relations
                ) if matured_relations else None
            ),
            "mean_predicted_minus_realized_incremental_return": (
                statistics.fmean(
                    float(row["predicted_minus_realized_incremental_return"])
                    for row in matured_relations
                ) if matured_relations else None
            ),
            "positive_predicted_and_realized_date_fraction": (
                statistics.fmean(
                    float(
                        row["predicted_risk_adjusted_net_incremental_return_vs_hold"] > 0
                        and row["realized_incremental_return_after_estimated_trade_cost"] > 0
                    )
                    for row in matured_relations
                ) if matured_relations else None
            ),
        }
    return {
        "summary": comparisons,
        "cost_sensitivity": sensitivity,
        "risk_sensitivity": risk_sensitivity,
        "common_start_date": common_start_date.isoformat() if common_start_date else None,
        "artifacts": artifacts,
    }
