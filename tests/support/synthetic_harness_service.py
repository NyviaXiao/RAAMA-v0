"""Clearly labeled, local-only data and reusable research artifact for service tests.

This fixture exercises the real service, worker, CLI, calibrated forecast,
optimizer, and hard-risk path. It never reads repository market CSVs and never
contacts a provider.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import shutil
import tomllib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from adaptive_mas.agents.harness import _accept_report, _run_local_risk_role
from adaptive_mas.data.research import load_oos_research_data
from adaptive_mas.evaluation.mvp_research import run_mvp_research
from adaptive_mas.harness_runtime import _market_snapshot_identity
from adaptive_mas.package_identity import source_package_identity
from adaptive_mas.release_governance import HarnessReleaseStore
from adaptive_mas.research_provenance import build_research_package_provenance, stable_hash
from adaptive_mas.risk_profile import resolve_risk_profile


ROOT = Path(__file__).resolve().parents[2]
SHANGHAI = ZoneInfo("Asia/Shanghai")
AS_OF = date(2026, 10, 23)
TRAINED_THROUGH = date(2026, 10, 16)
HORIZON = 3
RISK_BUDGET = 0.20
SYMBOLS = ("sh.600000", "sz.000001")


def _business_dates(start: date, end: date) -> list[date]:
    output = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            output.append(cursor)
        cursor += timedelta(days=1)
    return output


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_market_data(run_dir: Path) -> None:
    dates = _business_dates(date(2025, 8, 1), AS_OF)
    index_rows = []
    prices = {symbol: 10.0 + i * 3.0 for i, symbol in enumerate(SYMBOLS)}
    prior_prices = dict(prices)
    rows_by_symbol = {symbol: [] for symbol in SYMBOLS}
    for index, day in enumerate(dates):
        index_close = 3_500 + index * 0.7 + 12 * math.sin(index / 8)
        index_rows.append({
            "date": day.isoformat(), "code": "sh.000300", "open": index_close * 0.998,
            "high": index_close * 1.004, "low": index_close * 0.996, "close": index_close,
            "preclose": index_close - 0.7, "pctChg": (index_close / (index_close - 0.7) - 1) * 100,
        })
        for offset, symbol in enumerate(SYMBOLS):
            daily_return = 0.00025 + 0.0017 * math.sin((index + 3 * offset) / (6 + offset)) + 0.0003 * math.cos(index / 13)
            prices[symbol] = prior_prices[symbol] * (1.0 + daily_return)
            code = symbol
            volume = 1_500_000 + 12_000 * (index % 19)
            for adjustment in ("1", "3"):
                rows_by_symbol[symbol].append({
                    "date": day.isoformat(), "code": code, "open": f"{prices[symbol] * 0.999:.6f}",
                    "high": f"{prices[symbol] * 1.006:.6f}", "low": f"{prices[symbol] * 0.994:.6f}",
                    "close": f"{prices[symbol]:.6f}", "preclose": f"{prior_prices[symbol]:.6f}",
                    "volume": volume, "amount": f"{prices[symbol] * volume:.4f}",
                    "adjustflag": adjustment, "turn": "0.015", "tradestatus": "1",
                    "pctChg": f"{daily_return * 100:.8f}",
                })
            prior_prices[symbol] = prices[symbol]

    index_fields = ["date", "code", "open", "high", "low", "close", "preclose", "pctChg"]
    equity_fields = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "adjustflag", "turn", "tradestatus", "pctChg"]
    _write_csv(run_dir / "index" / "sh.000300.csv", index_fields, index_rows)
    membership_rows = [{"updateDate": dates[0].isoformat(), "code": symbol, "code_name": f"Synthetic {symbol}"} for symbol in SYMBOLS]
    _write_csv(run_dir / "membership" / f"asof-{dates[0].isoformat()}.csv", ["updateDate", "code", "code_name"], membership_rows)
    for symbol, rows in rows_by_symbol.items():
        _write_csv(run_dir / "daily-adjusted" / f"{symbol}.csv", equity_fields, [row for row in rows if row["adjustflag"] == "1"])
        _write_csv(run_dir / "daily-unadjusted" / f"{symbol}.csv", equity_fields, [row for row in rows if row["adjustflag"] == "3"])
    (run_dir / "manifest.jsonl").write_text(json.dumps({
        "fixture_label": "SYNTHETIC ONLY; NOT PROVIDER DATA",
        "symbols": list(SYMBOLS), "start_date": dates[0].isoformat(), "end_date": AS_OF.isoformat(),
    }, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_industry_config() -> dict[str, object]:
    return {
        "risk_industry_sources": [{
            "symbol": "600000",
            "industry": "Synthetic Industry Basket",
            "member_symbols": ["600000", "000001"],
            "source_title": "Synthetic fixture industry classification (local test data)",
            "source_url": "https://fixture.invalid/synthetic-industry-map",
            "published_at": f"{AS_OF.isoformat()}T15:00:00+08:00",
            "available_at": f"{AS_OF.isoformat()}T15:00:00+08:00",
            "fetched_at": f"{AS_OF.isoformat()}T15:05:00+08:00",
        }],
    }


def prepare_synthetic_service_fixture(fixture_root: str | Path) -> dict[str, Any]:
    """Create a complete task input bundle and its source-bound synthetic research result."""
    fixture_root = Path(fixture_root).resolve()
    fixture_root.mkdir(parents=True, exist_ok=True)
    data_dir = fixture_root / "synthetic_baostock_run"
    public_cache_dir = fixture_root / "empty_public_cache"
    version_dir = fixture_root / "synthetic_harness_versions"
    public_cache_dir.mkdir(exist_ok=True)
    version_dir.mkdir(exist_ok=True)
    _synthetic_market_data(data_dir)
    market = load_oos_research_data(data_dir)

    source_config = ROOT / "configs" / "daily_three_session.toml"
    config_path = fixture_root / "daily_three_session.toml"
    industry = _synthetic_industry_config()["risk_industry_sources"][0]
    industry_toml = (
        "\n[[risk_industry_sources]]\n"
        f"symbol = {json.dumps(industry['symbol'])}\n"
        f"industry = {json.dumps(industry['industry'])}\n"
        f"member_symbols = {json.dumps(industry['member_symbols'])}\n"
        f"source_title = {json.dumps(industry['source_title'])}\n"
        f"source_url = {json.dumps(industry['source_url'])}\n"
        f"published_at = {json.dumps(industry['published_at'])}\n"
        f"available_at = {json.dumps(industry['available_at'])}\n"
        f"fetched_at = {json.dumps(industry['fetched_at'])}\n"
    )
    synthetic_config = source_config.read_text(encoding="utf-8")
    synthetic_config = synthetic_config.replace("maximum_symbol_weight = 0.05", "maximum_symbol_weight = 0.60")
    synthetic_config = synthetic_config.replace("portfolio_top_fraction = 0.10", "portfolio_top_fraction = 1.0")
    synthetic_config = synthetic_config.replace("maximum_one_way_turnover = 0.50", "maximum_one_way_turnover = 1.0")
    synthetic_config = synthetic_config.replace("maximum_daily_participation = 0.01", "maximum_daily_participation = 0.50")
    config_path.write_bytes(synthetic_config.encode("utf-8") + industry_toml.encode("utf-8"))
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    risk_template = "balanced"
    risk_expression = "年化风险不超过15%，单票不超过5%，单向换手不超过30%，至少保留20%现金"
    risk_profile = resolve_risk_profile(
        risk_template=risk_template,
        risk_expression=risk_expression,
        base_defaults={
            "risk_budget_annualized": config.get("risk_budget_annualized", RISK_BUDGET),
            "maximum_symbol_weight": config.get("maximum_symbol_weight", 0.05),
            "maximum_one_way_turnover": config.get("maximum_one_way_turnover", 0.50),
            "minimum_cash_weight": config.get("minimum_cash_weight", 0.0),
        },
    )
    if not risk_profile["execution_allowed"]:
        raise AssertionError(risk_profile["unresolved"])
    config.update({"prediction_horizon_sessions": HORIZON})

    cutoff = TRAINED_THROUGH
    training_run = run_mvp_research(market, config, as_of_date=cutoff, ensure_as_of_decision=True)
    training_state = training_run["training_seed"]
    seed_path = fixture_root / "synthetic_training_seed.json.gz"
    with gzip.open(seed_path, "wt", encoding="utf-8") as handle:
        json.dump(training_state, handle, ensure_ascii=False, sort_keys=True)

    manifest_path = data_dir / "manifest.jsonl"
    versions = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "baostock_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "training_seed_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        "training_seed_trained_through": cutoff.isoformat(),
        "training_seed_horizon_sessions": str(HORIZON),
    }
    seed_provenance = {
        "trained_through": cutoff.isoformat(),
        "cutoff_source": "cli_argument",
        "horizon_sessions": HORIZON,
        "horizon_source": "cli_argument",
    }
    release_store = HarnessReleaseStore(version_dir)
    candidate, candidate_version_id, version_record = release_store.load_active(
        config=config,
        training_state=training_state,
        training_seed_provenance=seed_provenance,
        versions=versions,
    )
    candidate_sha256 = hashlib.sha256(json.dumps(
        candidate, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")).hexdigest()
    market_identity = _market_snapshot_identity(market, AS_OF)

    research_diagnostic = run_mvp_research(
        market,
        config,
        initial_state=training_state,
        as_of_date=AS_OF,
        ensure_as_of_decision=True,
    )
    forecast = research_diagnostic["prediction_decisions"][-1]
    model_errors = forecast.get("forecast_errors_by_method", {})
    model_metrics = [
        {"metric": f"matured_forecast_error_{method}", "value": float(value), "unit": "return_fraction"}
        for method, value in model_errors.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ] if isinstance(model_errors, dict) else []
    model_risk_core = {
        "status": "available" if model_metrics else "unavailable",
        "available_at": datetime.combine(AS_OF, datetime.min.time()).replace(hour=16, tzinfo=SHANGHAI).isoformat(),
        "period_start": cutoff.isoformat(),
        "period_end": AS_OF.isoformat(),
        "metrics": model_metrics,
        "source_refs": [
            f"training-seed-cutoff:{cutoff.isoformat()}",
            f"calibrated-forecast:{AS_OF.isoformat()}",
            f"market-snapshot:{market_identity['sha256']}",
        ],
    }
    model_risk_context = {**model_risk_core, "context_sha256": stable_hash(model_risk_core)}

    source_id = "synthetic-market-evidence-600000"
    source_time = f"{AS_OF.isoformat()}T15:55:00+08:00"
    market_source = {
        "evidence_id": source_id,
        "source_record_id": "SYNTHETIC_LOCAL_MARKET_FIXTURE_600000",
        "symbol": "600000",
        "source_type": "public_market_data",
        "dataset": "synthetic_local_adjusted_bars",
        "published_at": source_time,
        "available_at": source_time,
        "fetched_at": source_time,
        "period_start": TRAINED_THROUGH.isoformat(),
        "period_end": AS_OF.isoformat(),
        "metric": "synthetic_fixture_return",
        "value": 0.01,
        "unit": "return_fraction",
        "vintage_verified": True,
        "source_ref": "SYNTHETIC LOCAL FIXTURE ONLY",
    }
    market_report = _accept_report({
        "summary": "A synthetic local market fixture supplied one source-bound observation.",
        "findings": [{
            "symbol": "600000",
            "conclusion": "Synthetic fixture data only; this observation has no real-market meaning.",
            "evidence_ids": [source_id],
            "observations": [{
                "evidence_id": source_id,
                "field": "synthetic_fixture_return",
                "value": 0.01,
                "unit": "return_fraction",
                "report_period": None,
                "period_start": TRAINED_THROUGH.isoformat(),
                "period_end": AS_OF.isoformat(),
            }],
            "uncertainties": ["Synthetic local fixture only; not vendor or exchange data."],
        }],
        "unresolved_items": [],
    }, {source_id: market_source}, "market", visible_evidence_ids={source_id},
       decision_time=datetime.combine(AS_OF, datetime.min.time()).replace(hour=16, tzinfo=SHANGHAI))["accepted_report"]

    risk_result = _run_local_risk_role(
        "industry",
        "600000",
        "Check the synthetic industry's realized volatility and drawdown.",
        research=market,
        config=config,
        decision_time=datetime.combine(AS_OF, datetime.min.time()).replace(hour=16, tzinfo=SHANGHAI),
        model_risk_context=model_risk_context,
        news_handlers={},
        news_catalog={},
    )
    if risk_result.get("status") != "completed" or not isinstance(risk_result.get("report"), dict):
        raise RuntimeError(f"synthetic industry risk fixture could not be formed: {risk_result.get('reason')}")
    risk_report = risk_result["report"]
    risk_ids = [str(row) for finding in risk_report["findings"] for row in finding["evidence_ids"]]
    evidence_catalog = {source_id: market_source, **risk_result["evidence_catalog"]}
    risk_action = {
        "role_id": "risk_directed_query_1",
        "status": "completed_new_evidence",
        "symbol": "600000",
        "question": "Check the synthetic industry's realized volatility and drawdown.",
        "risk_dimension": "industry",
        "target_claim": "Synthetic fixture data only.",
        "target_evidence_id": source_id,
        "newly_cited_evidence_ids": risk_ids,
        "execution_mode": "on_host_deterministic",
        "raw_source_text_sent_to_model": False,
        "tool_trace": [{"tool": step["tool"], "status": "completed"} for step in risk_result["tool_trace"]],
        "turn_count": 0,
    }
    manager_report = {
        "summary": "Synthetic local research artifact; one bounded industry-risk review was reused offline.",
        "priority_symbols": [{"symbol": "600000", "evidence_ids": [source_id, *risk_ids[:2]]}],
        "unresolved_items": ["Fixture results are not real-market evidence."],
        "no_query_reason": "The synthetic one-action fixture ended after the industry risk review.",
        "stop_reason": "The synthetic one-action fixture ended after the industry risk review.",
        "manager_stop_code": "fixture_adapter_completed",
        "action_count": 1,
        "research_plan": {"goal": "Exercise sourced risk feedback in the offline service."},
        "risk_research": [{
            "dimension": "industry",
            "status": "completed_new_evidence",
            "symbol": "600000",
            "question": risk_action["question"],
            "target_claim": risk_action["target_claim"],
            "target_evidence_id": source_id,
            "report": risk_report,
            "source_evidence_ids": risk_ids,
        }],
    }
    public_snapshot = {
        "schema_version": "public-expert-snapshot.v1",
        "data_mode": "captured_public_cache_cutoff",
        "evaluation_horizon_sessions": HORIZON,
        "requested_decision_time": f"{AS_OF.isoformat()}T16:00:00+08:00",
        "fixture_label": "SYNTHETIC LOCAL FIXTURE ONLY; NO PROVIDER SOURCES",
        "p9_outputs": [], "p10_outputs": [], "attention_outputs": [],
    }
    task_usage = {
        "provider_call_count": 0,
        "attempt_count": 0,
        "response_cache_reuse_count": 0,
        "model_turn_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "logical_estimated_cost_cny": 0.0,
        "api_incremental_cost_cny": 0.0,
    }
    research_result: dict[str, object] = {
        "schema_version": "adaptive-mas.harness-research.v1",
        "status": "completed",
        "fixture_label": "SYNTHETIC LOCAL FIXTURE ONLY; NOT PROVIDER OR MARKET EVIDENCE",
        "public_evidence_mode": "captured_public_cache_cutoff",
        "as_of": AS_OF.isoformat(),
        "decision_time": f"{AS_OF.isoformat()}T16:00:00+08:00",
        "agents": {
            "market": {"report": market_report, "tool_trace": [], "context_compression": [], "turn_count": 0},
            "fundamental": {"report": {"agent_id": "fundamental", "summary": "No public fundamental source was used in this synthetic fixture.", "findings": [], "unresolved_items": ["No synthetic fundamental fixture was supplied."], "source_record_ids": []}, "tool_trace": [], "context_compression": [], "turn_count": 0},
            "research_manager": {"report": manager_report, "tool_trace": [], "context_compression": [], "turn_count": 0, "directed_queries": [risk_action]},
        },
        "manager_actions": [risk_action],
        "optional_experts": {"risk": {"status": "completed", "last_dimension": "industry", "hard_checks_separate": True}},
        "research_loop": {"goal": manager_report["research_plan"]["goal"], "action_count": 1, "action_limit": 2, "stop_code": "fixture_adapter_completed", "actions": [risk_action], "duplicate_actions_blocked": 0},
        "evidence_catalog": evidence_catalog,
        "memory": {
            "procedural_experience_version": candidate_version_id,
            "procedural_experience_sha256": hashlib.sha256(candidate["procedural_experience"].encode("utf-8")).hexdigest(),
            "procedural_memory": version_record.get("procedural_memory"),
        },
        "task_usage": task_usage,
        "cost": {"call_count": 0, "logical_inference_count": 0, "logical_inference_estimated_cost_cny": 0.0, "incremental_api_cost_cny": 0.0},
    }
    implementation = source_package_identity()
    research_result["package_provenance"] = build_research_package_provenance(
        research_result,
        as_of=AS_OF,
        public_evidence_mode="captured_public_cache_cutoff",
        horizon=HORIZON,
        config=config,
        candidate_version_id=candidate_version_id,
        candidate_sha256=candidate_sha256,
        implementation_identity=implementation,
        market_identity=market_identity,
        model="synthetic-fixture-adapter",
        endpoint="http://127.0.0.1/synthetic-fixture-only",
        source_snapshot=public_snapshot,
        model_risk_context_sha256=str(model_risk_context["context_sha256"]),
    )
    research_path = fixture_root / "synthetic_research_reuse.json"
    research_path.write_text(json.dumps(research_result, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    return {
        "fixture_root": str(fixture_root),
        "as_of": AS_OF.isoformat(),
        "horizon_sessions": HORIZON,
        "risk_budget_annualized": RISK_BUDGET,
        "risk_template": risk_template,
        "risk_expression": risk_expression,
        "risk_profile": risk_profile,
        "training_seed_trained_through": cutoff.isoformat(),
        "training_seed_horizon_sessions": HORIZON,
        "baostock_run_dir": str(data_dir),
        "training_seed": str(seed_path),
        "reuse_research_json": str(research_path),
        "config": str(config_path),
        "public_cache_dir": str(public_cache_dir),
        "harness_version_dir": str(version_dir),
        "portfolio": {"cash_cny": 1_000_000.0, "holdings_cny": {"600000": 0.0, "000001": 0.0}},
        "mode": "decision_reuse",
        "fixture_label": "SYNTHETIC LOCAL ONLY; NO TEST CSV, PROVIDER, OR BROKER",
        "implementation_package_sha256": implementation["implementation_package_sha256"],
        "training_seed_rows": len(training_state.get("state_fusion_training_rows", [])),
        "synthetic_research_source_cost_cny": 0.0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_synthetic_service_fixture(args.fixture_root), ensure_ascii=False, indent=2))
