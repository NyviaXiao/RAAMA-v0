"""Train the point-in-time Agentic RL policy and run one frozen A/B/C replay."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import tomllib
from collections import Counter
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data import load_research_data
from adaptive_mas.data.public_data import load_public_records
from adaptive_mas.decision import PortfolioState
from adaptive_mas.evaluation.agentic_rl import (
    ACTIONS,
    PROFILES,
    LinearExpectedSarsa,
    RidgeResidualModel,
    state_vector,
)
from adaptive_mas.evaluation.mvp_research import _risk_estimates

if __package__ in {None, ""}:
    from prepare_agentic_rl_qwen import _response_valid
else:
    from scripts.prepare_agentic_rl_qwen import _response_valid


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments" / "agentic_rl_historical_20261021"
SOURCE_DIR = ROOT / "experiments" / "post_p8_research_20260926" / "three_session_daily"
CALIBRATION_DIR = ROOT / "experiments" / "p9_joint_economics_20260927_full_exit"
OUTBOUND = RUN_ROOT / "outbound_payloads.jsonl"
QWEN_DIR = RUN_ROOT / "qwen_responses"
NORMALIZED_QWEN_DIR = RUN_ROOT / "qwen_normalized_responses"
TRAIN_END = date(2025, 6, 30)
VALIDATION_START = date(2025, 7, 1)
VALIDATION_END = date(2025, 8, 29)
EVALUATION_START = date(2025, 9, 1)
RISK_LOOKBACK_SESSIONS = 20
MARKET_STATE_LOOKBACK_SESSIONS = 20
ACCOUNT_RISK_LOOKBACK_SESSIONS = 60
ACCOUNT_RISK_MINIMUM_OBSERVATIONS = 40
TRAIN_EPSILON = 0.20
SHANGHAI = ZoneInfo("Asia/Shanghai")
FUNDAMENTAL_VALUE_FEATURES = (
    "revenue_growth",
    "parent_net_profit_growth",
    "adjusted_net_profit_growth",
    "operating_cashflow_to_revenue",
    "asset_liability_ratio",
    "roe",
)
FUNDAMENTAL_FEATURES = (
    *FUNDAMENTAL_VALUE_FEATURES,
    "valid_fact_fraction",
    *(f"{name}_available" for name in FUNDAMENTAL_VALUE_FEATURES),
)
RESEARCH_FEATURES = (
    "direction_score",
    "evidence_sufficiency",
    "conflict_resolved",
    "p9_expected_return",
    "fundamental_increment",
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _read_forecasts(path: Path) -> dict[date, dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return {
            date.fromisoformat(str(row["prediction_date"])): row
            for line in source if line.strip()
            for row in (json.loads(line),)
        }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _write_gzip_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split(day: date) -> str:
    if day <= TRAIN_END:
        return "training"
    if day <= VALIDATION_END:
        return "validation"
    return "evaluation"


def _filter_research(research, end: date):
    from adaptive_mas.data.research import ResearchData

    return ResearchData(
        trading_dates=[day for day in research.trading_dates if day <= end],
        adjusted_bars=[row for row in research.adjusted_bars if row["trade_date"] <= end],
        execution_bars=[row for row in research.execution_bars if row["trade_date"] <= end],
        index_bars=[row for row in research.index_bars if row["trade_date"] <= end],
        membership_snapshots=[row for row in research.membership_snapshots if row.available_at <= end],
    )


def _symbol(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


def _load_facts(cache_dir: Path, forecast_days: list[date], last_day: date) -> dict[date, dict[str, dict[str, object]]]:
    as_of = datetime.combine(last_day, time(23, 59), SHANGHAI)
    cached = load_public_records(cache_dir, "financial_indicator", as_of)
    visible_rows = []
    for row in cached:
        available = datetime.fromisoformat(str(row["available_at"])).astimezone(SHANGHAI)
        published = datetime.fromisoformat(str(row["published_at"])).astimezone(SHANGHAI)
        ready = max(available, published)
        if ready > as_of:
            continue
        visible_rows.append((ready, available, str(row["REPORT_DATE"]), _symbol(row["symbol"]), row))
    visible_rows.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    dates_by_day: dict[date, dict[str, dict[str, object]]] = {}
    latest: dict[str, tuple[datetime, str, dict[str, object]]] = {}
    cursor = 0
    for decision_day in sorted(forecast_days):
        decision_cutoff = datetime.combine(decision_day, time(23, 59), SHANGHAI)
        while cursor < len(visible_rows) and visible_rows[cursor][0] <= decision_cutoff:
            _, available, report_date, symbol, row = visible_rows[cursor]
            current = latest.get(symbol)
            if current is None or (available, report_date) > (current[0], current[1]):
                latest[symbol] = (available, report_date, row)
            cursor += 1
        features = {}
        for symbol, (_, _, row) in latest.items():
            revenue = _float(row.get("TOTALOPERATEREVETZ"))
            profit = _float(row.get("PARENTNETPROFITTZ"))
            adjusted_profit = _float(row.get("KCFJCXSYJLRTZ"))
            operating_cashflow_to_revenue = _float(row.get("JYXJLYYSR"))
            asset_liability_ratio = _float(row.get("ZCFZL"))
            roe = _float(row.get("ROEJQ"))
            raw_values = [revenue, profit, adjusted_profit, operating_cashflow_to_revenue, asset_liability_ratio, roe]
            valid = [value is not None for value in raw_values]
            revenue_value = max(-2.0, min(2.0, (revenue or 0.0) / 100))
            profit_value = max(-2.0, min(2.0, (profit or 0.0) / 100))
            adjusted_profit_value = max(-2.0, min(2.0, (adjusted_profit or 0.0) / 100))
            operating_cashflow_value = max(-5.0, min(5.0, operating_cashflow_to_revenue or 0.0))
            asset_liability_value = max(0.0, min(2.0, (asset_liability_ratio or 0.0) / 100))
            roe_value = max(-1.0, min(1.0, (roe or 0.0) / 100))
            features[symbol] = {
                "features": [
                    revenue_value, profit_value, adjusted_profit_value,
                    operating_cashflow_value, asset_liability_value, roe_value,
                    sum(valid) / len(valid), *(float(value) for value in valid),
                ],
                "revenue_yoy_pct": revenue,
                "parent_net_profit_yoy_pct": profit,
                "adjusted_net_profit_yoy_pct": adjusted_profit,
                "operating_cashflow_to_revenue_ratio": operating_cashflow_to_revenue,
                "asset_liability_ratio_pct": asset_liability_ratio,
                "roe_pct": roe,
                "available_fields": [name for name, is_valid in zip(FUNDAMENTAL_VALUE_FEATURES, valid, strict=True) if is_valid],
                "unavailable_fields": [name for name, is_valid in zip(FUNDAMENTAL_VALUE_FEATURES, valid, strict=True) if not is_valid],
                "record_id": str(row["source_record_id"]),
                "report_period": str(row["REPORT_DATE"]),
                "published_at": str(row["published_at"]),
                "available_at": str(row["available_at"]),
                "fetched_at": str(row["fetched_at"]),
                "vintage_verified": bool(row["vintage_verified"]),
            }
        dates_by_day[decision_day] = features
    return dates_by_day


def _float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _calibrated_forecasts(source_forecasts, calibration_rows, research):
    calibration = {date.fromisoformat(str(row["decision_date"])): row["by_method"] for row in calibration_rows}
    dates = sorted(set(source_forecasts) & set(calibration))
    bars = {(_symbol(row["symbol"]), row["trade_date"]): row for row in research.adjusted_bars}
    index = {row["trade_date"]: row for row in research.index_bars}
    forecasts = {}
    outcomes = {}
    for day in dates:
        if day not in research.trading_dates:
            continue
        row = json.loads(json.dumps(source_forecasts[day]))
        if row.get("input_cutoff") != day.isoformat():
            raise ValueError(f"P9 forecast cutoff mismatch on {day}")
        for method in ("market_plus_relative_fusion", "trend"):
            details = calibration[day][method]
            if str(details["window_end_label_date"]) >= day.isoformat():
                raise ValueError(f"P9 calibration uses an unmatured label on {day}")
            correction = float(details["total_mean_bias_estimate"])
            row["expected_returns_by_method"][method] = {
                symbol: float(value) + correction
                for symbol, value in row["expected_returns_by_method"][method].items()
            }
        for method in ("market_plus_relative_fusion", "trend", "reversal", "liquidity_confirmation"):
            if method not in row["expected_returns_by_method"] or row["forecast_errors_by_method"].get(method) is None:
                raise ValueError(f"{day} lacks a matured calibrated method for the RL state")
        if "market_plus_relative_fusion" not in row.get("target_weights_by_method", {}):
            raise ValueError(f"{day} lacks the frozen P9 candidate basket")
        forecasts[day] = row
        day_index = research.trading_dates.index(day)
        horizon = int(row["horizon_sessions"])
        if day_index + horizon >= len(research.trading_dates):
            outcomes[day] = {}
            continue
        entry_day = research.trading_dates[day_index + 1]
        exit_day = research.trading_dates[day_index + horizon]
        market_entry = index.get(entry_day)
        market_exit = index.get(exit_day)
        label_by_symbol = {}
        for symbol, p9_mu in row["expected_returns_by_method"]["market_plus_relative_fusion"].items():
            entry = bars.get((_symbol(symbol), entry_day))
            exit_bar = bars.get((_symbol(symbol), exit_day))
            if (
                entry is None or exit_bar is None or market_entry is None or market_exit is None
                or any(value is None for value in (
                    entry.get("open"), exit_bar.get("close"),
                    market_entry.get("open"), market_exit.get("close"),
                ))
            ):
                continue
            stock_return = float(exit_bar["close"]) / float(entry["open"]) - 1
            market_return = float(market_exit["close"]) / float(market_entry["open"]) - 1
            label_by_symbol[_symbol(symbol)] = {
                "absolute_return": stock_return,
                "market_return": market_return,
                "forward_excess_return": stock_return - market_return,
                "label_available_date": exit_day,
                "entry_date": entry_day,
                "exit_date": exit_day,
            }
        outcomes[day] = label_by_symbol
    if not forecasts:
        raise ValueError("No matching P8 forecasts and sequential P9 calibration rows")
    return forecasts, outcomes


def _fundamental_predictions(forecasts, outcomes, facts_by_day, trading_dates):
    sample_rows = []
    seen_facts = set()
    label_rows = []
    samples_by_decision_day: dict[date, list[dict[str, object]]] = {}
    date_index = {day: index for index, day in enumerate(trading_dates)}
    eligible_days = sorted(set(forecasts) & set(facts_by_day))
    for day in eligible_days:
        if day > TRAIN_END:
            continue
        forecast = forecasts[day]
        mu = forecast["expected_returns_by_method"]["market_plus_relative_fusion"]
        for symbol, fact in facts_by_day[day].items():
            fact_id = f"{symbol}:{fact['record_id']}"
            if fact_id in seen_facts:
                continue
            outcome = outcomes[day].get(symbol)
            if outcome is None:
                continue
            seen_facts.add(fact_id)
            sample = {
                "record_id": f"fundamental:{day.isoformat()}:{fact_id}",
                "decision_date": day.isoformat(),
                "label_available_date": outcome["label_available_date"].isoformat(),
                "symbol": symbol,
                "feature_values": fact["features"],
                "available_fields": fact["available_fields"],
                "unavailable_fields": fact["unavailable_fields"],
                "source_values": {
                    "revenue_yoy_pct": fact["revenue_yoy_pct"],
                    "parent_net_profit_yoy_pct": fact["parent_net_profit_yoy_pct"],
                    "adjusted_net_profit_yoy_pct": fact["adjusted_net_profit_yoy_pct"],
                    "operating_cashflow_to_revenue_ratio": fact["operating_cashflow_to_revenue_ratio"],
                    "asset_liability_ratio_pct": fact["asset_liability_ratio_pct"],
                    "roe_pct": fact["roe_pct"],
                },
                "report_period": fact["report_period"],
                "published_at": fact["published_at"],
                "available_at": fact["available_at"],
                "fetched_at": fact["fetched_at"],
                "source_record_id": fact["record_id"],
                "vintage_verified": fact["vintage_verified"],
                "target_residual": float(outcome["absolute_return"]) - float(mu[symbol]),
            }
            sample_rows.append(sample)
            label_rows.append((outcome["label_available_date"], sample))
            samples_by_decision_day.setdefault(day, []).append(sample)
    label_rows.sort(key=lambda item: (item[0], item[1]["decision_date"], item[1]["symbol"]))
    by_maturity: dict[date, list[dict[str, object]]] = {}
    for label_day, row in label_rows:
        if label_day <= TRAIN_END:
            by_maturity.setdefault(label_day, []).append(row)
    model = RidgeResidualModel(FUNDAMENTAL_FEATURES, alpha=10.0, minimum_rows=300)
    matured = []
    predictions: dict[date, dict[str, float]] = {}
    for day in eligible_days:
        if day <= TRAIN_END:
            for sample in by_maturity.get(day, []):
                model.add(list(sample["feature_values"]), float(sample["target_residual"]), str(sample["record_id"]))
                matured.append(sample)
            model.fit()
        for sample in samples_by_decision_day.get(day, []):
            sample["sequential_residual_prediction"] = model.predict(list(sample["feature_values"]))
        increments = {}
        for symbol, fact in facts_by_day[day].items():
            increments[symbol] = model.predict(list(fact["features"]))
        predictions[day] = increments
    return predictions, model, sample_rows, matured


def _load_queries() -> tuple[dict[date, dict[str, object]], dict[str, object], dict[str, object]]:
    events = _read_jsonl(OUTBOUND)
    by_day = {}
    response_status = Counter()
    split_coverage: dict[str, Counter] = {}
    provider_cost = 0.0
    response_count = 0
    invalid_response_count = 0
    normalized_response_count = 0
    effective_response_count = 0
    usage_known_response_count = 0
    unknown_cost_count = 0
    reserved_unknown_cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    completion_seconds = []
    for event in events:
        day = date.fromisoformat(str(event["prediction_date"]))
        split = str(event["data_split"])
        coverage = split_coverage.setdefault(split, Counter())
        coverage["request_count"] += 1
        state_path = QWEN_DIR / f"{event['event_id']}.json"
        if not state_path.exists():
            response_status["missing"] += 1
            coverage["missing_count"] += 1
            coverage["unusable_response_count"] += 1
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["request_sha256"] != event["request_sha256"]:
            raise ValueError(f"Qwen cache request hash mismatch for {event['event_id']}")
        status = str(state["status"])
        response_status[status] += 1
        coverage[f"{status}_count"] += 1
        actual_cost = state.get("actual_estimated_cost_cny")
        if actual_cost is not None:
            provider_cost += float(actual_cost)
        else:
            unknown_cost_count += 1
            reserved_unknown_cost += float(state.get("reserved_max_cost_cny", 0.0))
        usage = state.get("usage")
        if isinstance(usage, dict) and "prompt_tokens" in usage and "completion_tokens" in usage:
            usage_known_response_count += 1
            prompt_tokens += int(usage["prompt_tokens"])
            completion_tokens += int(usage["completion_tokens"])
        if status == "invalid_response":
            invalid_response_count += 1
        analysis = None
        if status == "completed":
            analysis = state["analysis"]
            response_count += 1
            effective_response_count += 1
            coverage["effective_usable_response_count"] += 1
            started = datetime.fromisoformat(str(state["submitted_at_utc"]))
            finished = datetime.fromisoformat(str(state["finished_at_utc"]))
            completion_seconds.append((finished - started).total_seconds())
        elif status == "invalid_response":
            normalized_path = NORMALIZED_QWEN_DIR / f"{event['event_id']}.json"
            normalized_record = (
                json.loads(normalized_path.read_text(encoding="utf-8"))
                if normalized_path.is_file() else None
            )
            if (
                normalized_record is not None
                and normalized_record.get("normalization_status") == "normalized_for_replay"
                and normalized_record.get("normalization_version") == "exact_public_title_citation_to_source_type_v1"
                and normalized_record.get("request_sha256") == event["request_sha256"]
                and normalized_record.get("original_status") == "invalid_response"
                and normalized_record.get("raw_response_sha256") == state.get("raw_response_sha256")
                and normalized_record.get("payload_manifest_sha256") == state.get("payload_manifest_sha256")
            ):
                analysis = normalized_record.get("normalized_analysis")
                if not isinstance(analysis, dict):
                    raise ValueError(f"Normalized Qwen analysis is not an object for {event['event_id']}")
                analysis = _response_valid(json.dumps(analysis, ensure_ascii=False), event)
                normalized_response_count += 1
                effective_response_count += 1
                coverage["normalized_response_count"] += 1
                coverage["effective_usable_response_count"] += 1
                event["analysis_normalization"] = {
                    "version": normalized_record["normalization_version"],
                    "basis": normalized_record.get("normalization_basis", []),
                }
            else:
                coverage["unrecovered_invalid_response_count"] += 1
                coverage["unusable_response_count"] += 1
        else:
            coverage["unusable_response_count"] += 1

        if analysis is not None:
            event["analysis"] = analysis
            event["actual_cost_cny"] = float(actual_cost)
            event["usage"] = usage
            by_day[day] = event
    return by_day, {
        "request_count": len(events),
        "completed_response_count": response_count,
        "invalid_response_count": invalid_response_count,
        "normalized_response_count": normalized_response_count,
        "effective_usable_response_count": effective_response_count,
        "unrecovered_invalid_response_count": sum(
            counts["unrecovered_invalid_response_count"] for counts in split_coverage.values()
        ),
        "split_coverage": {split: dict(counts) for split, counts in split_coverage.items()},
        "usage_known_response_count": usage_known_response_count,
        "response_status_counts": dict(response_status),
        "provider_actual_estimated_cost_cny": provider_cost,
        "provider_actual_cost_unknown_count": unknown_cost_count,
        "provider_reserved_max_cost_for_unknown_cny": reserved_unknown_cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "median_completed_latency_seconds": statistics.median(completion_seconds) if completion_seconds else None,
    }, {str(event["event_id"]): event for event in events}


def _risk_reward(daily_nav, risk_budget: float) -> tuple[float, float | None]:
    returns = [float(row["daily_return"]) for row in daily_nav]
    current = returns[-1]
    risk = statistics.stdev(returns[-RISK_LOOKBACK_SESSIONS:]) * math.sqrt(250) if len(returns) >= RISK_LOOKBACK_SESSIONS else None
    penalty = 0.5 * max(risk - risk_budget, 0.0) ** 2 / 250 if risk is not None else 0.0
    return math.log1p(current) - penalty, risk


def _event_features(event, fundamental_increment: float) -> list[float]:
    analysis = event["analysis"]
    sufficiency = {"insufficient": 0.0, "limited": 0.5, "sufficient": 1.0}[str(analysis["evidence_sufficiency"])]
    return [
        float(analysis["direction_score"]),
        sufficiency,
        float(bool(analysis["conflict_resolved"])),
        float(event["p9_expected_return"]),
        float(fundamental_increment),
    ]


def _ridge_from_dict(value: dict[str, object]) -> RidgeResidualModel:
    return RidgeResidualModel.from_dict(value)


def _state(
    forecast,
    account,
    research,
    day: date,
    facts,
    fundamental_increments,
    query_event,
    last_query_day: date | None,
    risk_budget: float,
) -> tuple[dict[str, float], dict[str, object]]:
    index = research.trading_dates.index(day)
    if index < MARKET_STATE_LOOKBACK_SESSIONS:
        raise ValueError(f"{day} has no complete 20-session market-state window")
    index_by_date = {
        row["trade_date"]: float(row["close"])
        for row in research.index_bars
        if row.get("close") is not None
    }
    market_return_20s = (
        index_by_date[day]
        / index_by_date[research.trading_dates[index - MARKET_STATE_LOOKBACK_SESSIONS]]
        - 1
    )
    market_state = "market_up" if market_return_20s >= 0 else "market_down"
    methods = forecast["expected_returns_by_method"]
    risk_symbols = set(account.holdings_cny) | set(methods["market_plus_relative_fusion"])
    volatility, betas, market_volatility = _risk_estimates(
        research,
        index,
        risk_symbols,
        ACCOUNT_RISK_LOOKBACK_SESSIONS,
        ACCOUNT_RISK_MINIMUM_OBSERVATIONS,
    )
    risk_audit_unavailable = sorted(
        (set(account.holdings_cny) - set(volatility))
        | (set(account.holdings_cny) - set(betas))
    )
    if risk_audit_unavailable:
        risk = None
        risk_state = "unavailable_missing_holding_risk"
    else:
        factor_exposure = math.fsum(
            value / account.nav_cny * float(betas[symbol])
            for symbol, value in account.holdings_cny.items()
        )
        idiosyncratic_variance = math.fsum(
            (value / account.nav_cny) ** 2
            * max(
                float(volatility[symbol]) ** 2
                - float(betas[symbol]) ** 2 * market_volatility**2,
                0.0,
            )
            for symbol, value in account.holdings_cny.items()
        )
        risk = math.sqrt(
            idiosyncratic_variance + (factor_exposure * market_volatility) ** 2
        )
        risk_state = "near_budget" if risk >= risk_budget * 0.8 else "within_budget"
    symbols = sorted(set(methods["market_plus_relative_fusion"]))
    spreads = [
        statistics.pstdev([float(methods[name][symbol]) for name in ("trend", "reversal", "liquidity_confirmation")])
        for symbol in symbols
        if all(symbol in methods[name] for name in ("trend", "reversal", "liquidity_confirmation"))
    ]
    median_spread = statistics.median(spreads)
    covered = sum(
        symbol in facts and bool(facts[symbol].get("available_fields"))
        for symbol in symbols
    )
    anomalies = [
        max(abs(float(value)) for value in facts[symbol]["features"][:len(FUNDAMENTAL_VALUE_FEATURES)])
        for symbol in facts if symbol in symbols
    ]
    cash_weight = float(account.cash_cny / account.nav_cny)
    holdings_weight = math.fsum(account.holdings_cny.values()) / account.nav_cny
    evidence_conflict = 0.0
    trade_importance = 0.0
    if query_event is not None:
        p9_direction = (float(query_event["p9_expected_return"]) > 0) - (float(query_event["p9_expected_return"]) < 0)
        evidence_conflict = min(2.0, abs(int(query_event["title_direction_score"]) - p9_direction)) / 2
        target = str(query_event["symbol"])
        held_weight = float(account.holdings_cny.get(target, 0.0)) / account.nav_cny
        trade_importance = min(1.0, max(float(query_event["p9_target_weight"]), held_weight) * 20)
    days_since_query = 30 if last_query_day is None else max(0, (day - last_query_day).days)
    state = {
        "market_return_20s": max(-1.0, min(1.0, market_return_20s * 5)),
        "expert_disagreement": max(0.0, min(3.0, median_spread / 0.01)),
        "fundamental_anomaly": max(0.0, min(3.0, statistics.median(anomalies) if anomalies else 0.0)),
        "fundamental_coverage": covered / len(symbols),
        "query_available": float(query_event is not None),
        "evidence_conflict": evidence_conflict,
        "query_trade_importance": trade_importance,
        "account_cash_weight": cash_weight,
        "account_holdings_weight": max(0.0, min(1.0, holdings_weight)),
        "risk_to_budget": max(0.0, min(3.0, float(risk or 0.0) / risk_budget)),
        "days_since_query": min(3.0, days_since_query / 10),
        "fusion_stage": 0.0,
        "research_performed": 0.0,
        "research_direction_score": 0.0,
        "research_evidence_sufficiency": 0.0,
        "research_conflict_resolved": 0.0,
        "research_supporting_evidence_count": 0.0,
        "research_contrary_evidence_count": 0.0,
    }
    return state, {
        "market_state": market_state,
        "market_return_20s": market_return_20s,
        "median_calibrated_expert_spread": median_spread,
        "fundamental_coverage_symbols": covered,
        "candidate_symbol_count": len(symbols),
        "account_cash_weight": cash_weight,
        "account_holdings_weight": holdings_weight,
        "current_account_annualized_risk_estimate": risk,
        "risk_budget_annualized": risk_budget,
        "risk_state": risk_state,
        "query_available": query_event is not None,
        "evidence_conflict": evidence_conflict,
        "query_trade_importance": trade_importance,
    }


def _study_actions(state: dict[str, float]) -> tuple[str, ...]:
    return ("study:finish", "study:query") if state["query_available"] else ("study:finish",)


def _fusion_actions(state: dict[str, float]) -> tuple[str, ...]:
    allowed = ["fusion:p9"]
    if state["fundamental_coverage"] > 0:
        allowed.append("fusion:fundamental")
    if state["research_performed"] > 0:
        allowed.extend(("fusion:research", "fusion:joint"))
    return tuple(allowed)


def _with_research_result(state: dict[str, float], event: dict[str, object] | None) -> dict[str, float]:
    if event is None:
        return {**state, "fusion_stage": 1.0}
    analysis = event["analysis"]
    sufficiency = {"insufficient": 0.0, "limited": 0.5, "sufficient": 1.0}[
        str(analysis["evidence_sufficiency"])
    ]
    return {
        **state,
        "fusion_stage": 1.0,
        "research_performed": 1.0,
        "research_direction_score": float(analysis["direction_score"]),
        "research_evidence_sufficiency": sufficiency,
        "research_conflict_resolved": float(bool(analysis["conflict_resolved"])),
        "research_supporting_evidence_count": min(1.0, len(analysis["supporting_evidence"]) / 2),
        "research_contrary_evidence_count": min(1.0, len(analysis["contrary_evidence"]) / 2),
    }


def _apply_profile(forecast, profile, facts, fundamental_increments, query_event, research_model):
    base_mu = forecast["expected_returns_by_method"]["market_plus_relative_fusion"]
    result_mu = {symbol: float(value) for symbol, value in base_mu.items()}
    if profile in {"fundamental", "joint"}:
        for symbol, adjustment in fundamental_increments.items():
            if symbol in result_mu:
                result_mu[symbol] = result_mu[symbol] + float(adjustment)
    text_adjustment = 0.0
    if profile in {"research", "joint"}:
        if query_event is None:
            raise ValueError("Research fusion action has no current Qwen response")
        symbol = str(query_event["symbol"])
        included_fundamental_increment = (
            fundamental_increments.get(symbol, 0.0) if profile == "joint" else 0.0
        )
        features = _event_features(query_event, included_fundamental_increment)
        text_adjustment = research_model.predict(features)
        result_mu[symbol] = result_mu[symbol] + text_adjustment
    expected = {**forecast["expected_returns_by_method"], "agentic_fusion": result_mu}
    errors = {**forecast["forecast_errors_by_method"]}
    errors["agentic_fusion"] = float(errors["market_plus_relative_fusion"])
    target_methods = forecast["target_weights_by_method"]
    if "market_plus_relative_fusion" not in target_methods:
        raise ValueError("P9 candidate basket is missing from the frozen forecast")
    return {
        **forecast,
        "expected_returns_by_method": expected,
        "forecast_errors_by_method": errors,
        "target_weights_by_method": {
            **target_methods,
            "agentic_fusion": dict(target_methods["market_plus_relative_fusion"]),
        },
        "controller_return_method": "agentic_fusion",
    }, text_adjustment


def _save_run(output: Path, name: str, run: dict[str, object]) -> None:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "summary.json", run["summary"])
    _write_csv(directory / "daily_nav.csv", run["daily_nav"])
    _write_csv(directory / "orders.csv", run["orders"])
    _write_csv(directory / "positions.csv", run["positions"])
    _write_jsonl(directory / "decisions.jsonl", run["decisions"])


def _average_ranks(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = np.zeros(len(values), dtype=float)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2
        for index in order[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks


def _forecast_metrics(trace: list[dict[str, object]], outcomes: dict[date, dict[str, dict[str, object]]]) -> dict[str, object]:
    squared_errors = []
    observations = []
    rank_ics = []
    for row in trace:
        day = date.fromisoformat(str(row["decision_date"]))
        realized = outcomes.get(day, {})
        paired = [
            (float(mu), float(realized[symbol]["absolute_return"]), float(realized[symbol]["forward_excess_return"]))
            for symbol, mu in row["expected_returns_by_symbol"].items()
            if symbol in realized and realized[symbol].get("absolute_return") is not None
        ]
        if not paired:
            continue
        predictions = [item[0] for item in paired]
        absolute_labels = [item[1] for item in paired]
        excess_labels = [item[2] for item in paired]
        squared_errors.extend((prediction - label) ** 2 for prediction, label in zip(predictions, absolute_labels, strict=True))
        observations.extend(zip(predictions, absolute_labels, strict=True))
        if len(paired) >= 20:
            prediction_rank = _average_ranks(predictions)
            label_rank = _average_ranks(excess_labels)
            if np.std(prediction_rank) and np.std(label_rank):
                rank_ics.append(float(np.corrcoef(prediction_rank, label_rank)[0, 1]))
    return {
        "matured_stock_date_observations": len(observations),
        "matured_decision_dates": len(rank_ics),
        "rmse": math.sqrt(statistics.fmean(squared_errors)) if squared_errors else None,
        "rank_ic_mean": statistics.fmean(rank_ics) if rank_ics else None,
        "rank_ic_median": statistics.median(rank_ics) if rank_ics else None,
        "rmse_label": "absolute stock return from next-session open to third-session close",
        "rank_ic_definition": "daily Spearman correlation of predicted absolute stock returns and realized excess returns; average ranks for ties; common market return does not change within-date label ranks",
    }


def _target_changes(reference, candidate) -> dict[str, object]:
    reference_by_day = {
        str(row["decision_date"]): row.get("target_weights")
        for row in reference["decisions"] if row.get("target_weights") is not None
    }
    candidate_by_day = {
        str(row["decision_date"]): row.get("target_weights")
        for row in candidate["decisions"] if row.get("target_weights") is not None
    }
    common = sorted(set(reference_by_day) & set(candidate_by_day))
    changed = []
    for day in common:
        left = reference_by_day[day]
        right = candidate_by_day[day]
        symbols = set(left) | set(right)
        maximum_difference = max(
            (abs(float(left.get(symbol, 0.0)) - float(right.get(symbol, 0.0))) for symbol in symbols),
            default=0.0,
        )
        if maximum_difference > 1e-8:
            changed.append({"decision_date": day, "maximum_absolute_weight_difference": maximum_difference})
    return {
        "compared_decision_count": len(common),
        "changed_target_decision_count": len(changed),
        "changed_target_decision_fraction": len(changed) / len(common) if common else 0.0,
        "changed_dates": changed,
    }


def _controller(
    *,
    mode: str,
    research,
    forecasts,
    outcomes,
    facts_by_day,
    fundamental_increments,
    query_by_day,
    risk_budget: float,
    sarsa: LinearExpectedSarsa | None = None,
    research_model: RidgeResidualModel | None = None,
):
    model = research_model or RidgeResidualModel(RESEARCH_FEATURES, alpha=10.0, minimum_rows=5)
    collect_training = mode in {"training", "fixed_training"}
    fusion_transitions = []
    pending_queries = []
    text_training_samples = []
    action_trace = []
    forecast_trace = []
    decision_state_trace = []
    transition_trace = []
    last_query_day = None
    last_nav_count = 0
    epsilon = TRAIN_EPSILON if mode == "training" else 0.0

    def decide(*, forecast, account, decision_date, daily_nav, decisions, decision_pause=None):
        nonlocal last_query_day, last_nav_count
        day = decision_date
        facts = facts_by_day[day]
        increments = fundamental_increments[day]
        query_event = query_by_day.get(day)
        pre_state, state_audit = _state(
            forecast, account, research, day, facts, increments,
            query_event, last_query_day, risk_budget,
        )
        allowed_study = _study_actions(pre_state)
        finish_fusion_state = {**pre_state, "fusion_stage": 1.0}
        query_fusion_state = _with_research_result(pre_state, query_event)
        decision_state_trace.append({
            "decision_date": day.isoformat(),
            "study_state_features": pre_state,
            "allowed_study_actions": list(allowed_study),
            "finish_fusion_state_features": finish_fusion_state,
            "finish_allowed_fusion_actions": list(_fusion_actions(finish_fusion_state)),
            "query_fusion_state_features": query_fusion_state if query_event is not None else None,
            "query_allowed_fusion_actions": (
                list(_fusion_actions(query_fusion_state)) if query_event is not None else None
            ),
            "query_event_id": query_event["event_id"] if query_event is not None else None,
            "query_text_residual_features": (
                _event_features(
                    query_event,
                    float(increments.get(str(query_event["symbol"]), 0.0)),
                )
                if query_event is not None else None
            ),
        })

        if collect_training:
            newly_observed = daily_nav[last_nav_count:]
            last_nav_count = len(daily_nav)
            for nav_offset, nav_row in enumerate(
                newly_observed,
                start=len(daily_nav) - len(newly_observed),
            ):
                trade_day = date.fromisoformat(str(nav_row["trade_date"]))
                reward, _ = _risk_reward(daily_nav[:nav_offset + 1], risk_budget)
                for transition in fusion_transitions:
                    if (
                        len(transition["rewards"]) >= 3
                        or trade_day <= transition["decision_date"]
                        or trade_day <= transition["last_reward_date"]
                    ):
                        continue
                    transition["rewards"].append(reward)
                    transition["reward_dates"].append(trade_day.isoformat())
                    transition["last_reward_date"] = trade_day

            for pending in list(pending_queries):
                if pending["label_available_date"] > day:
                    continue
                model.add(
                    pending["features"],
                    pending["target_residual"],
                    pending["record_id"],
                )
                model.fit()
                text_training_samples.append({
                    "record_id": pending["record_id"],
                    "event_id": pending["event_id"],
                    "decision_date": pending["decision_date"].isoformat(),
                    "label_available_date": pending["label_available_date"].isoformat(),
                    "symbol": pending["symbol"],
                    "feature_values": pending["features"],
                    "target_residual": pending["target_residual"],
                    "matured_at": day.isoformat(),
                })
                pending_queries.remove(pending)

            if mode == "training":
                remaining = []
                for transition in fusion_transitions:
                    if len(transition["rewards"]) < 3:
                        remaining.append(transition)
                        continue
                    update = sarsa.update_n_step(
                        transition["action"],
                        transition["features"],
                        transition["rewards"][:3],
                        state_vector(pre_state),
                        allowed_study,
                        epsilon,
                    )
                    transition_trace.append({
                        "transition_stage": "fusion",
                        "policy_version": "linear_expected_sarsa_two_stage_v1",
                        "decision_date": transition["decision_date"].isoformat(),
                        "action": transition["action"],
                        "state_features": transition["features"].tolist(),
                        "allowed_actions": transition["allowed_actions"],
                        "reward_dates": transition["reward_dates"][:3],
                        "matured_rewards": transition["rewards"][:3],
                        "next_decision_date": day.isoformat(),
                        "next_stage": "study",
                        "next_state_features": state_vector(pre_state).tolist(),
                        "next_allowed_actions": list(allowed_study),
                        "matured_at": day.isoformat(),
                        "policy_update_index": sarsa.updates,
                        "update": update,
                    })
                fusion_transitions[:] = remaining

        if mode in {"training", "policy"}:
            study_action, study_explored, study_estimates = sarsa.choose(
                state_vector(pre_state),
                allowed_study,
                epsilon,
                seed=int(hashlib.sha256(f"{day.isoformat()}:study".encode()).hexdigest()[:8], 16),
            )
        elif mode in {"fixed", "fixed_training"}:
            study_action = "study:query" if query_event is not None else "study:finish"
            study_explored = False
            study_estimates = {}
        else:
            raise ValueError(f"Unknown controller mode: {mode}")

        if study_action not in allowed_study:
            raise ValueError(f"Controller selected unavailable study action {study_action} on {day}")
        queried = study_action == "study:query"
        revealed_event = query_event if queried else None
        fusion_state = _with_research_result(pre_state, revealed_event)
        allowed_fusion = _fusion_actions(fusion_state)

        if mode in {"training", "policy"}:
            fusion_action, fusion_explored, fusion_estimates = sarsa.choose(
                state_vector(fusion_state),
                allowed_fusion,
                epsilon,
                seed=int(hashlib.sha256(f"{day.isoformat()}:fusion".encode()).hexdigest()[:8], 16),
            )
        else:
            fusion_action = "fusion:joint" if queried else (
                "fusion:fundamental"
                if "fusion:fundamental" in allowed_fusion
                else "fusion:p9"
            )
            fusion_explored = False
            fusion_estimates = {}
        if fusion_action not in allowed_fusion:
            raise ValueError(f"Controller selected unavailable fusion action {fusion_action} on {day}")

        profile = PROFILES[fusion_action]
        controlled, research_adjustment = _apply_profile(
            forecast,
            profile,
            facts,
            increments,
            revealed_event,
            model,
        )
        fee = float(revealed_event["actual_cost_cny"]) if queried else 0.0
        if queried:
            last_query_day = day

        stage_reward = 0.0
        study_update = None
        if mode == "training":
            study_update = sarsa.update_one_step(
                study_action,
                state_vector(pre_state),
                stage_reward,
                state_vector(fusion_state),
                allowed_fusion,
                epsilon,
                intraday_discount=1.0,
            )
            transition_trace.append({
                "transition_stage": "study",
                "policy_version": "linear_expected_sarsa_two_stage_v1",
                "decision_date": day.isoformat(),
                "action": study_action,
                "state_features": state_vector(pre_state).tolist(),
                "allowed_actions": list(allowed_study),
                "same_day_reward": stage_reward,
                "reward_type": "zero_same_day_information_outcome",
                "intraday_discount": 1.0,
                "next_decision_date": day.isoformat(),
                "next_stage": "fusion",
                "next_state_features": state_vector(fusion_state).tolist(),
                "next_allowed_actions": list(allowed_fusion),
                "matured_at": day.isoformat(),
                "policy_update_index": sarsa.updates,
                "update": study_update,
            })
            fusion_transitions.append({
                "decision_date": day,
                "last_reward_date": day,
                "action": fusion_action,
                "features": state_vector(fusion_state),
                "allowed_actions": list(allowed_fusion),
                "rewards": [],
                "reward_dates": [],
            })

        if queried and collect_training and profile in {"research", "joint"}:
            target = outcomes.get(day, {}).get(str(revealed_event["symbol"]))
            if target is not None and day <= TRAIN_END:
                symbol = str(revealed_event["symbol"])
                p9 = float(
                    forecast["expected_returns_by_method"]["market_plus_relative_fusion"][symbol]
                )
                fund = float(increments.get(symbol, 0.0)) if profile == "joint" else 0.0
                features = _event_features(revealed_event, fund)
                pending_queries.append({
                    "record_id": f"qwen:{revealed_event['event_id']}:{profile}",
                    "event_id": revealed_event["event_id"],
                    "decision_date": day,
                    "label_available_date": target["label_available_date"],
                    "symbol": symbol,
                    "features": features,
                    "target_residual": float(target["absolute_return"]) - p9 - fund,
                })

        expected_returns = {
            _symbol(symbol): float(value)
            for symbol, value in controlled["expected_returns_by_method"]["agentic_fusion"].items()
        }
        forecast_trace.append({
            "decision_date": day.isoformat(),
            "study_action": study_action,
            "fusion_action": fusion_action,
            "expected_returns_by_symbol": expected_returns,
        })
        action_trace.append({
            "decision_date": day.isoformat(),
            "study_action": study_action,
            "fusion_action": fusion_action,
            "profile": profile,
            "query": queried,
            "query_event_id": revealed_event["event_id"] if queried else None,
            "query_symbol": revealed_event["symbol"] if queried else None,
            "research_cost_cny": fee,
            "study_immediate_reward": stage_reward,
            "research_direction_score": revealed_event["analysis"]["direction_score"] if queried else None,
            "research_evidence_sufficiency": revealed_event["analysis"]["evidence_sufficiency"] if queried else None,
            "research_increment": research_adjustment,
            "fundamental_increment_for_query_symbol": (
                float(increments.get(str(revealed_event["symbol"]), 0.0))
                if queried else None
            ),
        })
        audit = {
            **state_audit,
            "data_split": _split(day),
            "study_state_features": pre_state,
            "allowed_study_actions": allowed_study,
            "study_action": study_action,
            "research_response_revealed_after_study_action": queried,
            "study_immediate_reward": stage_reward,
            "fusion_state_features": fusion_state,
            "allowed_fusion_actions": allowed_fusion,
            "fusion_action": fusion_action,
            "action_applied": True,
            "fusion_profile": profile,
            "supplementary_query": queried,
            "supplementary_query_status": "completed_cached_response" if queried else "not_selected",
            "supplementary_query_event_id": revealed_event["event_id"] if queried else None,
            "supplementary_query_target_symbol": revealed_event["symbol"] if queried else None,
            "supplementary_query_cost_cny": fee,
            "research_cost_cny": fee,
            "research_response": revealed_event["analysis"] if queried else None,
            "research_public_source_provenance": (
                revealed_event["local_source_reference_map"] if queried else None
            ),
            "research_numeric_adjustment": research_adjustment,
            "fundamental_adjustment_coverage": len(increments),
            "research_calibration_rows": model.fitted_rows,
            "study_policy_estimates": study_estimates,
            "fusion_policy_estimates": fusion_estimates,
            "study_explored": study_explored,
            "fusion_explored": fusion_explored,
            "exploration_rate": epsilon,
            "fusion_feedback_maturity_sessions": 3,
            "fusion_feedback_includes_research_fee_once": True,
            "study_td_update": study_update,
        }
        return controlled, audit

    def summary():
        return {
            "study_action_counts": dict(Counter(row["study_action"] for row in action_trace)),
            "fusion_action_counts": dict(Counter(row["fusion_action"] for row in action_trace)),
            "query_count": sum(bool(row["query"]) for row in action_trace),
            "query_cost_cny": math.fsum(float(row["research_cost_cny"]) for row in action_trace),
            "query_event_ids": [row["query_event_id"] for row in action_trace if row["query_event_id"]],
            "action_trace": action_trace,
            "forecast_trace": forecast_trace,
            "decision_state_trace": decision_state_trace,
            "transition_trace": transition_trace,
            "text_training_samples": text_training_samples,
            "research_model": model.to_dict(),
            "query_calibration_pending_at_end": len(pending_queries),
            "fusion_transition_pending_at_end": len(fusion_transitions),
            "unmatured_transition_count": len(fusion_transitions),
        }

    return decide, summary

def _run_portfolio(
    research,
    forecasts,
    initial_capital,
    costs,
    config,
    output,
    name,
    controller=None,
    start_date=None,
    progress=None,
):
    optimizer = dict(config)
    optimizer.update({
        "optimizer_return_method": "agentic_fusion" if controller else "market_plus_relative_fusion",
        "optimizer_candidate_scope": "frozen_method_target",
    })
    result = simulate_portfolio(
        research,
        [],
        initial_capital,
        costs,
        int(config["liquidity_lookback_sessions"]),
        float(config["maximum_daily_participation"]),
        float(config["maximum_one_way_turnover"]),
        forecast_decisions=forecasts,
        optimizer_config=optimizer,
        progress_callback=(lambda text: progress(f"{name}: {text}")) if progress else None,
        evaluation_start_date=start_date,
        forecast_controller=controller,
    )
    _save_run(output, name, result)
    return result


def _render_report(path: Path, summary: dict[str, object]) -> None:
    eval_rows = summary["evaluation"]["accounts"]
    lines = [
        "# RAAMA 历史 Agentic RL 评测",
        "",
        "## 研究边界",
        "",
        f"- 训练：{summary['windows']['training_start']} 至 {summary['windows']['training_end']}；验证：{summary['windows']['validation_start']} 至 {summary['windows']['validation_end']}；冻结评价：{summary['windows']['evaluation_start']} 至 {summary['windows']['evaluation_end']}。",
        "- 只读取 train.csv、指定 BaoStock 缓存和已有 AKShare 缓存；不读 test.csv。历史回填资料标记为回顾研究，未伪称当时真实归档。",
        "- 预测标签：次交易日开盘至第 3 个交易日收盘的股票绝对收益；每日连续持仓复审。",
        "",
        "## 一次统一 A/B/C 账户回放",
        "",
        "| 方案 | 净收益 | 超额沪深300 | 最大回撤 | Sharpe（无风险利率0） | 换手 | 交易费 | 研究费 | 查询次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("A · P9", "A"), ("B · 固定研究/融合", "B"), ("C · RL调度/融合", "C")):
        row = eval_rows[key]
        lines.append(
            f"| {label} | {row['total_return']:.2%} | {row['excess_return_vs_csi300']:.2%} | "
            f"{row['maximum_drawdown']:.2%} | {row['sharpe_rf_zero']:.2f} | {row['one_way_turnover_over_average_nav']:.2%} | "
            f"¥{row['transaction_cost_cny']:,.2f} | ¥{row['research_cost_cny']:,.2f} | {row['query_count']} |"
        )
    lines.append("")
    if eval_rows["A"]["annualized_risk"] == 0:
        lines.append("- A 全现金基线的年化波动率为 0，Sharpe 数学上未定义；表中 0.00 是现有显示约定。")
    split_names = {"training": "训练", "validation": "验证", "evaluation": "评价"}
    for split, label in split_names.items():
        coverage = summary["qwen"]["split_coverage"].get(split, {})
        lines.append(
            f"- {label}期研究响应覆盖：请求 {coverage.get('request_count', 0)}，原始有效 {coverage.get('completed_count', 0)}，"
            f"精确标题恢复 {coverage.get('normalized_response_count', 0)}，最终可用 {coverage.get('effective_usable_response_count', 0)}，"
            f"未恢复无效 {coverage.get('unrecovered_invalid_response_count', 0)}。"
        )
    lines.extend([
        "",
        f"- B−A净收益差：{summary['evaluation']['B_minus_A_return_difference']:.2%}；C−B净收益差：{summary['evaluation']['C_minus_B_return_difference']:.2%}。",
        f"- 目标权重相对 P9 实际变化的决策占比：B {eval_rows['B']['target_changes_vs_P9']['changed_target_decision_fraction']:.1%}，C {eval_rows['C']['target_changes_vs_P9']['changed_target_decision_fraction']:.1%}。",
        "- 评测是已查看历史数据上的冻结回放，稀疏文本调用的配对差异不构成普遍因果证明。B/C 查询后成熟的文本校准样本分别为 41/20；策略路径影响查询暴露与可成熟样本，因此 C−B 同时包含暴露和校准差异，不能解释为纯调度因果贡献。",
        "",
        "## 校准与覆盖",
        "",
        f"- 结构化基本面训练样本：{summary['calibration']['fundamental_training_rows']} 个唯一披露项；包含收入/利润/扣非利润同比、经营现金流/营业收入、资产负债率与 ROE；缺失字段以可用性掩码表示。",
        f"- 训练披露字段可用数：{summary['calibration']['fundamental_feature_coverage']}；来源字段为 `TOTALOPERATEREVETZ`、`PARENTNETPROFITTZ`、`KCFJCXSYJLRTZ`、`JYXJLYYSR`、`ZCFZL`、`ROEJQ`。",
        f"- 基本面残差 RMSE：顺序成熟标签预测 {summary['calibration']['fundamental_sequential_residual_rmse']}（{summary['calibration']['fundamental_sequential_prediction_count']} 个训练期预测）；最终训练拟合 RMSE {summary['calibration']['fundamental_training_fit_rmse']}（仅拟合内，不是样本外误差）。",
        f"- C 查询后成熟的文本残差样本：{summary['calibration']['C_text_training_rows']}；B 查询后成熟的文本残差样本：{summary['calibration']['B_text_training_rows']}。",
        f"- Qwen 请求/原始有效/原始无效/精确标题恢复/最终可用/未恢复无效：{summary['qwen']['request_count']} / {summary['qwen']['completed_response_count']} / {summary['qwen']['invalid_response_count']} / {summary['qwen']['normalized_response_count']} / {summary['qwen']['effective_usable_response_count']} / {summary['qwen']['unrecovered_invalid_response_count']}；provider 已知估算费用（含无效响应）¥{summary['qwen']['provider_actual_estimated_cost_cny']:.4f}；另有 {summary['qwen']['provider_actual_cost_unknown_count']} 项实际费用未知，预留上限 ¥{summary['qwen']['provider_reserved_max_cost_for_unknown_cny']:.4f}。账户只对策略实际使用的查询扣费一次。",
        f"- 用量已知响应的输入/输出 token：{summary['qwen']['prompt_tokens']} / {summary['qwen']['completion_tokens']}；成功响应中位延迟：{summary['qwen']['median_completed_latency_seconds']} 秒。只有经验证的原始或精确来源归一响应进入训练/评价，未恢复的无效响应排除。",
        f"- 预测校准：A {summary['evaluation']['forecast_metrics']['A']['rmse']}，B {summary['evaluation']['forecast_metrics']['B']['rmse']}，C {summary['evaluation']['forecast_metrics']['C']['rmse']}；Rank IC 见 `evaluation_summary.json`。",
        "",
        "## 限制与解释",
        "",
        "- 新闻正文没有存档的事件仅发送标题；缺失新闻/宏观资料不当作中性证据。财报/公告记录虽按来源披露日作回顾可见，抓取时间仍保留且 vintage 未验证。",
        "- 查询响应的方向、充分性、冲突和证据类型只作为查询后的状态特征，不直接计入奖励。研究步骤即时奖励为 0、同日折扣为 1；查询价值只按后续实际账户净收益、风险和费用评价。",
        "- B 在所有有缓存响应的候选日查询后固定使用联合融合；C 先由 RL 选择查询或结束研究，查询结果仅在该动作后进入同日研究状态，再由 RL 选择 P9/基本面/研究/联合融合。",
        "- 文本样本稀疏；新增查询行动进入市场时间只通过一次已扣费的决策，研究步骤本身不推进交易日。",
        "",
        "详细账户、决策轨迹、预测配对、冻结系数和模型调用账单见同目录文件。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data-dir", type=Path, default=Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data"))
    parser.add_argument("--baostock-run-dir", type=Path, default=ROOT / "data" / "raw" / "baostock" / "fetch-20260924T115050Z")
    parser.add_argument("--public-cache-dir", type=Path, default=ROOT / "data" / "raw" / "akshare")
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT / "replay")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    status = {"status": "incomplete", "started_at_utc": datetime.now(timezone.utc).isoformat(), "qwen_calls": 0}

    def progress(message: str) -> None:
        status["last_progress"] = message
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "RUN_STATUS.json", status)
        with (output / "progress.log").open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"{status['updated_at_utc']} {message}\n")
        print(f"[Agentic RL historical] {message}", flush=True)

    progress("loading only train.csv, the frozen BaoStock cache, P8 forecasts, P9 sequential calibration, and existing financial caches")
    research = load_research_data(args.training_data_dir, args.baostock_run_dir)
    source_forecasts = _read_forecasts(SOURCE_DIR / "prediction_decisions.jsonl.gz")
    calibration_rows = _read_jsonl(CALIBRATION_DIR / "rolling_calibration_details.jsonl")
    forecasts, outcomes = _calibrated_forecasts(source_forecasts, calibration_rows, research)
    days = sorted(day for day in forecasts if research.trading_dates.index(day) + 1 < len(research.trading_dates))
    training_days = [day for day in days if day <= TRAIN_END]
    validation_days = [day for day in days if VALIDATION_START <= day <= VALIDATION_END]
    evaluation_days = [day for day in days if day >= EVALUATION_START]
    if not training_days or not validation_days or not evaluation_days:
        raise ValueError("Fixed chronological split lacks training, validation, or evaluation forecast dates")
    actual_end = research.trading_dates[-1]
    if actual_end != date(2026, 3, 6):
        raise ValueError(f"Primary cache end changed from the approved 2026-03-06: {actual_end}")
    first_eval = evaluation_days[0]
    if first_eval < EVALUATION_START:
        raise ValueError("Evaluation started before the frozen date boundary")
    progress(f"loaded {len(days)} sequential P9 decisions; training {training_days[0]}..{training_days[-1]} ({len(training_days)}), validation {validation_days[0]}..{validation_days[-1]} ({len(validation_days)}), evaluation {first_eval}..{evaluation_days[-1]} ({len(evaluation_days)})")

    facts_by_day = _load_facts(args.public_cache_dir, days, actual_end)
    fundamental_increments, fundamental_model, fundamental_samples, matured_fundamental = _fundamental_predictions(
        forecasts, outcomes, facts_by_day, research.trading_dates,
    )
    sequential_fundamental_errors = [
        float(row["sequential_residual_prediction"]) - float(row["target_residual"])
        for row in fundamental_samples
        if row.get("sequential_residual_prediction") is not None
        and date.fromisoformat(str(row["label_available_date"])) <= TRAIN_END
    ]
    fundamental_sequential_rmse = (
        math.sqrt(statistics.fmean(error * error for error in sequential_fundamental_errors))
        if sequential_fundamental_errors else None
    )
    fundamental_feature_coverage = {
        name: sum(name in row["available_fields"] for row in fundamental_samples)
        for name in FUNDAMENTAL_VALUE_FEATURES
    }
    query_by_day, qwen_summary, all_query_events = _load_queries()
    if set(query_by_day) - set(forecasts):
        raise ValueError("Qwen response exists outside the frozen P9 forecast dates")
    train_responses = sum(day <= TRAIN_END for day in query_by_day)
    if train_responses == 0:
        progress("no successful training Qwen responses are available; request preparation is complete but numerical text learning is unavailable")
    progress(f"chronological fundamental residual calibration finished: {len(matured_fundamental)} matured unique fact labels; sequential residual RMSE={fundamental_sequential_rmse}; Qwen training responses={train_responses}/{sum(event['data_split']=='training' for event in all_query_events.values())}")

    config = tomllib.loads((SOURCE_DIR / "config.toml").read_text(encoding="utf-8"))
    costs = {key: float(config[key]) for key in (
        "commission_per_side_bps", "transfer_fee_per_side_bps",
        "stamp_duty_sell_bps", "slippage_per_side_bps",
    )}
    risk_budget = float(config["risk_budget_annualized"])
    initial_capital = float(config["initial_capital_cny"])
    train_research = _filter_research(research, TRAIN_END)
    train_forecasts = [forecasts[day] for day in training_days]
    train_c_policy = LinearExpectedSarsa(alpha=0.015, gamma=0.96)
    train_c, train_c_summary = _controller(
        mode="training", research=train_research, forecasts=forecasts, outcomes=outcomes,
        facts_by_day=facts_by_day, fundamental_increments=fundamental_increments,
        query_by_day=query_by_day, risk_budget=risk_budget, sarsa=train_c_policy,
    )
    progress("C training replay begins (3-session matured Expected SARSA; query cost is debited by the account simulator)")
    c_training = _run_portfolio(
        train_research, train_forecasts, initial_capital, costs, config, output,
        "training_C", train_c, start_date=training_days[0], progress=progress,
    )
    c_training_state = train_c_summary()
    progress(f"C training replay finished: {c_training['summary']['trading_days']} sessions; policy updates={train_c_policy.updates}; queries={c_training_state['query_count']}")

    b_text_model = RidgeResidualModel(RESEARCH_FEATURES, alpha=10.0, minimum_rows=5)
    train_b, train_b_summary = _controller(
        mode="fixed_training", research=train_research, forecasts=forecasts, outcomes=outcomes,
        facts_by_day=facts_by_day, fundamental_increments=fundamental_increments,
        query_by_day=query_by_day, risk_budget=risk_budget, research_model=b_text_model,
    )
    progress("B training replay begins using its predeclared fixed query-plus-joint-fusion rule")
    b_training = _run_portfolio(
        train_research, train_forecasts, initial_capital, costs, config, output,
        "training_B", train_b, start_date=training_days[0], progress=progress,
    )
    b_training_state = train_b_summary()
    progress(f"B training replay finished: {b_training['summary']['trading_days']} sessions; queries={b_training_state['query_count']}")

    validation_research = _filter_research(research, validation_days[-1])
    b_frozen_model = _ridge_from_dict(b_training_state["research_model"])
    validation_b, validation_b_summary = _controller(
        mode="fixed", research=validation_research, forecasts=forecasts, outcomes=outcomes,
        facts_by_day=facts_by_day, fundamental_increments=fundamental_increments,
        query_by_day=query_by_day, risk_budget=risk_budget, research_model=b_frozen_model,
    )
    progress("frozen B rule validation replay begins")
    b_validation = _run_portfolio(
        validation_research, [forecasts[day] for day in validation_days], initial_capital,
        costs, config, output, "validation_B", validation_b,
        start_date=validation_days[0], progress=progress,
    )
    validation_b_state = validation_b_summary()

    eval_forecasts = [forecasts[day] for day in evaluation_days]
    frozen_c_policy = LinearExpectedSarsa.from_dict(train_c_policy.to_dict())
    b_eval, b_eval_summary = _controller(
        mode="fixed", research=research, forecasts=forecasts, outcomes=outcomes,
        facts_by_day=facts_by_day, fundamental_increments=fundamental_increments,
        query_by_day=query_by_day, risk_budget=risk_budget,
        research_model=_ridge_from_dict(b_training_state["research_model"]),
    )
    c_eval, c_eval_summary = _controller(
        mode="policy", research=research, forecasts=forecasts, outcomes=outcomes,
        facts_by_day=facts_by_day, fundamental_increments=fundamental_increments,
        query_by_day=query_by_day, risk_budget=risk_budget, sarsa=frozen_c_policy,
        research_model=_ridge_from_dict(c_training_state["research_model"]),
    )
    progress("one unified frozen A/B/C evaluation begins from identical initial capital and date range")
    a_eval = _run_portfolio(
        research, eval_forecasts, initial_capital, costs, config, output,
        "evaluation_A_P9", None, start_date=EVALUATION_START, progress=progress,
    )
    b_eval_run = _run_portfolio(
        research, eval_forecasts, initial_capital, costs, config, output,
        "evaluation_B_fixed", b_eval, start_date=EVALUATION_START, progress=progress,
    )
    c_eval_run = _run_portfolio(
        research, eval_forecasts, initial_capital, costs, config, output,
        "evaluation_C_expected_sarsa", c_eval, start_date=EVALUATION_START, progress=progress,
    )
    a_forecast_trace = [
        {"decision_date": row["prediction_date"], "study_action": "study:finish", "fusion_action": "fusion:p9", "expected_returns_by_symbol": {
            _symbol(symbol): float(value)
            for symbol, value in row["expected_returns_by_method"]["market_plus_relative_fusion"].items()
        }} for row in eval_forecasts
    ]
    a_metrics = _forecast_metrics(a_forecast_trace, outcomes)
    b_metrics = _forecast_metrics(b_eval_summary()["forecast_trace"], outcomes)
    c_metrics = _forecast_metrics(c_eval_summary()["forecast_trace"], outcomes)
    b_eval_state = b_eval_summary()
    c_eval_state = c_eval_summary()
    b_target_changes = _target_changes(a_eval, b_eval_run)
    c_target_changes = _target_changes(a_eval, c_eval_run)
    eval_accounts = {}
    for key, run, controller_state, metric in (
        ("A", a_eval, {}, a_metrics),
        ("B", b_eval_run, b_eval_state, b_metrics),
        ("C", c_eval_run, c_eval_state, c_metrics),
    ):
        raw = run["summary"]
        eval_accounts[key] = {
            "total_return": float(raw["total_return"]),
            "excess_return_vs_csi300": float(raw["total_return"] - raw["benchmark_total_return"]),
            "benchmark_total_return": float(raw["benchmark_total_return"]),
            "maximum_drawdown": float(raw["maximum_drawdown"]),
            "sharpe_rf_zero": float(raw["sharpe_rf_zero"]),
            "one_way_turnover_over_average_nav": float(raw["one_way_turnover_over_average_nav"]),
            "transaction_cost_cny": float(raw["total_transaction_cost_cny"]),
            "research_cost_cny": float(raw["research_cost_cny"]),
            "total_cost_cny": float(raw["total_cost_including_research_cny"]),
            "average_cash_weight": float(raw["mean_cash_weight"]),
            "annualized_risk": float(raw["annualized_volatility"]),
            "risk_budget_annualized": risk_budget,
            "query_count": int(controller_state.get("query_count", 0)),
            "query_event_ids": controller_state.get("query_event_ids", []),
            "forecast_metrics": metric,
            "target_changes_vs_P9": b_target_changes if key == "B" else c_target_changes if key == "C" else None,
        }
    summary = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_input": "train.csv plus existing BaoStock and AKShare caches; test.csv not read",
        "data_provenance": {
            "training_csv_path": str(args.training_data_dir / "train.csv"),
            "baostock_cache_path": str(args.baostock_run_dir),
            "public_cache_path": str(args.public_cache_dir),
            "bao_trading_date_count": len(research.trading_dates),
            "bao_date_start": research.trading_dates[0].isoformat(),
            "bao_date_end": research.trading_dates[-1].isoformat(),
            "historical_research_not_blind": True,
            "backfilled_public_vintage_verified": False,
        },
        "windows": {
            "training_start": training_days[0].isoformat(), "training_end": TRAIN_END.isoformat(),
            "validation_start": VALIDATION_START.isoformat(), "validation_end": VALIDATION_END.isoformat(),
            "evaluation_start": EVALUATION_START.isoformat(), "evaluation_end": research.trading_dates[-1].isoformat(),
            "training_decision_count": len(training_days), "validation_decision_count": len(validation_days),
            "evaluation_decision_count": len(evaluation_days),
            "horizon_sessions": 3,
            "absolute_return_label": "next-session open to third-session close stock return",
            "market_return_label": "CSI 300 return over the same open-to-close interval",
            "excess_return_label": "absolute stock return minus the same-interval CSI 300 return",
        },
        "strategy_rules": {
            "A": "P9 market_plus_relative_fusion",
            "B": "query every available cached event, then use joint P9+fundamental+research residual; without query use fundamental increment when facts exist",
            "C": "two-stage linear Expected SARSA: choose study:query or study:finish, reveal public-evidence response only after query, then choose fusion:p9/fundamental/research/joint; epsilon 0.20 in training and 0 in frozen evaluation",
            "training_query_effects_enter_after_action": True,
            "query_and_fusion_are_separate_actions": True,
            "research_step_feedback": "zero immediate reward; response-derived fields remain post-query state features only, and study Expected SARSA bootstraps into the same-day fusion state with unit intraday discount",
            "fusion_step_feedback": "three matured daily NAV log returns including the once-debited query fee, minus the daily risk-budget penalty; separate n=3 Expected SARSA update",
            "evaluation_policy_and_incremental_calibrators_frozen_at": TRAIN_END.isoformat(),
            "query_target_selection": "one frozen date-symbol candidate per date, ranked from only same-date public evidence, P9 forecast disagreement/position importance; future outcomes and Qwen conclusions excluded",
            "research_fee_accounting": "debit actual cached response cost from account cash at its next execution open; daily NAV/reward include the debit once; no second reward subtraction",
            "risk_budget_annualized": risk_budget,
            "reward": "two separate TD transitions: zero immediate study reward with unit same-day bootstrap, and matured n=3 daily NAV/risk reward for the fusion step; fee is included in NAV once",
        },
        "calibration": {
            "fundamental_model": fundamental_model.to_dict(),
            "fundamental_training_rows": len(fundamental_samples),
            "fundamental_matured_rows_at_freeze": len(matured_fundamental),
            "fundamental_fitted_rows": fundamental_model.fitted_rows,
            "fundamental_training_fit_rmse": fundamental_model.rmse,
            "fundamental_sequential_residual_rmse": fundamental_sequential_rmse,
            "fundamental_sequential_prediction_count": len(sequential_fundamental_errors),
            "fundamental_feature_coverage": fundamental_feature_coverage,
            "C_text_training_rows": c_training_state["research_model"]["training_rows"],
            "B_text_training_rows": b_training_state["research_model"]["training_rows"],
            "C_text_model": c_training_state["research_model"],
            "B_text_model": b_training_state["research_model"],
        },
        "qwen": qwen_summary,
        "training": {
            "C_account_summary": c_training["summary"],
            "C_policy": train_c_policy.to_dict(),
            "C_controller": {key: value for key, value in c_training_state.items() if key not in {
                "forecast_trace", "action_trace", "decision_state_trace", "transition_trace", "text_training_samples",
            }},
            "C_transition_update_count": len(c_training_state["transition_trace"]),
            "C_unmatured_transition_count": c_training_state["unmatured_transition_count"],
            "C_text_training_sample_count": len(c_training_state["text_training_samples"]),
            "B_account_summary": b_training["summary"],
            "B_controller": {key: value for key, value in b_training_state.items() if key not in {
                "forecast_trace", "action_trace", "decision_state_trace", "transition_trace", "text_training_samples",
            }},
            "B_text_training_sample_count": len(b_training_state["text_training_samples"]),
        },
        "validation": {
            "fixed_B_account_summary": b_validation["summary"],
            "fixed_B_controller": {key: value for key, value in validation_b_state.items() if key not in {
                "forecast_trace", "action_trace", "decision_state_trace", "transition_trace", "text_training_samples",
            }},
        },
        "evaluation": {
            "accounts": eval_accounts,
            "B_minus_A_return_difference": eval_accounts["B"]["total_return"] - eval_accounts["A"]["total_return"],
            "C_minus_B_return_difference": eval_accounts["C"]["total_return"] - eval_accounts["B"]["total_return"],
            "same_initial_capital": initial_capital,
            "same_decision_and_evaluation_dates": True,
            "policy_frozen_before_evaluation": True,
            "forecast_metrics": {"A": a_metrics, "B": b_metrics, "C": c_metrics},
        },
    }
    _write_json(output / "evaluation_summary.json", summary)
    _write_json(output / "frozen_agentic_policy.json", {
        "policy": frozen_c_policy.to_dict(),
        "fundamental_residual_model": fundamental_model.to_dict(),
        "C_text_residual_model": c_training_state["research_model"],
        "B_text_residual_model": b_training_state["research_model"],
        "fixed_B_rule": "query each available event, select joint fusion",
        "freeze_date": TRAIN_END.isoformat(),
    })
    _write_gzip_jsonl(output / "fundamental_residual_samples.jsonl.gz", fundamental_samples)
    _write_gzip_jsonl(output / "matured_stock_market_outcomes.jsonl.gz", [
        {
            "decision_date": day.isoformat(),
            "symbol": symbol,
            "absolute_return": outcome["absolute_return"],
            "market_return": outcome["market_return"],
            "forward_excess_return": outcome["forward_excess_return"],
            "entry_date": outcome["entry_date"].isoformat(),
            "exit_date": outcome["exit_date"].isoformat(),
            "label_available_date": outcome["label_available_date"].isoformat(),
        }
        for day, by_symbol in outcomes.items()
        for symbol, outcome in by_symbol.items()
    ])
    _write_jsonl(output / "C_training_action_trace.jsonl", c_training_state["action_trace"])
    _write_jsonl(output / "C_training_transition_trace.jsonl", c_training_state["transition_trace"])
    _write_jsonl(output / "C_training_text_residual_samples.jsonl", c_training_state["text_training_samples"])
    _write_jsonl(output / "B_training_action_trace.jsonl", b_training_state["action_trace"])
    _write_jsonl(output / "B_training_text_residual_samples.jsonl", b_training_state["text_training_samples"])
    _write_jsonl(output / "B_validation_action_trace.jsonl", validation_b_state["action_trace"])
    _write_jsonl(output / "B_evaluation_action_trace.jsonl", b_eval_state["action_trace"])
    _write_jsonl(output / "C_evaluation_action_trace.jsonl", c_eval_state["action_trace"])
    _write_jsonl(output / "C_evaluation_state_trace.jsonl", c_eval_state["decision_state_trace"])
    _write_gzip_jsonl(output / "evaluation_forecast_traces.jsonl.gz", [
        {"strategy": name, **row}
        for name, trace in (("A", a_forecast_trace), ("B", b_eval_state["forecast_trace"]), ("C", c_eval_state["forecast_trace"]))
        for row in trace
    ])
    _render_report(output / "AGENTIC_RL_HISTORICAL_REPORT.md", summary)
    status.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "qwen_calls": qwen_summary["completed_response_count"]})
    _write_json(output / "RUN_STATUS.json", status)
    progress("unified A/B/C evaluation complete; report, frozen policy, calibration models, and traced account outputs saved")


if __name__ == "__main__":
    main()
