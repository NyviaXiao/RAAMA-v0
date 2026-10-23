"""Compare baseline and harness-filtered decisions on the same account and short horizon."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import tomllib
from datetime import date
from pathlib import Path

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data import load_oos_research_data
from adaptive_mas.data.research import ResearchData
from adaptive_mas.decision import PortfolioState, create_daily_decision
from adaptive_mas.evaluation.mvp_research import run_mvp_research
from adaptive_mas.harness_evolution import HarnessVersionStore


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _short_research_window(research: ResearchData, as_of: date, horizon: int) -> ResearchData:
    decision_index = research.trading_dates.index(as_of)
    end_index = decision_index + horizon
    if end_index >= len(research.trading_dates):
        raise ValueError("cached market data does not cover the requested forward horizon")
    end_date = research.trading_dates[end_index]
    return ResearchData(
        trading_dates=[day for day in research.trading_dates if day <= end_date],
        adjusted_bars=[row for row in research.adjusted_bars if row["trade_date"] <= end_date],
        execution_bars=[row for row in research.execution_bars if row["trade_date"] <= end_date],
        index_bars=[row for row in research.index_bars if row["trade_date"] <= end_date],
        membership_snapshots=[row for row in research.membership_snapshots if row.available_at <= end_date],
    )


def _decision_summary(decision, *, harness: bool) -> dict[str, object]:
    harness_trace = decision.evidence.get("harness") or {}
    cost_accounting = harness_trace.get("research_cost_accounting", {})
    return {
        "status": decision.status,
        "target_cash_cny": decision.target_cash_cny,
        "target_holdings_cny": dict(decision.target_holdings_cny),
        "target_weights": dict(decision.target_weights),
        "orders": [order.__dict__ for order in decision.orders],
        "harness_version_id": decision.versions.get("harness_version_id") if harness else None,
        "candidate_priority_symbols": (
            harness_trace.get("candidate_filter", {}).get("priority_symbols", []) if harness else []
        ),
        "logical_research_cost_cny": (
            cost_accounting.get("logical_research_cost_cny", 0.0)
            if harness else 0.0
        ),
        "source_task_api_cost_cny": (
            cost_accounting.get("source_task_api_cost_cny", 0.0) if harness else 0.0
        ),
        "new_provider_requests": 0,
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument("--portfolio-state-json", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--horizon-sessions", type=int, required=True)
    parser.add_argument("--risk-budget-annualized", type=float, required=True)
    parser.add_argument("--reuse-research-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=project_dir / "configs" / "daily_three_session.toml")
    parser.add_argument(
        "--training-seed",
        type=Path,
        default=project_dir / "data" / "processed" / "p8_daily_three_session_continuous_v1" / "training_seed.json.gz",
    )
    parser.add_argument("--training-seed-trained-through", type=date.fromisoformat)
    parser.add_argument("--training-seed-horizon-sessions", type=int)
    parser.add_argument("--public-cache-dir", type=Path, default=project_dir / "data" / "raw" / "akshare")
    parser.add_argument(
        "--harness-version-dir",
        type=Path,
        default=project_dir / "data" / "processed" / "harness_versions",
    )
    parser.add_argument("--retrospective-public-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.horizon_sessions < 1:
        raise ValueError("horizon-sessions must be positive")
    default_seed = (
        project_dir / "data" / "processed" / "p8_daily_three_session_continuous_v1" / "training_seed.json.gz"
    )
    if args.training_seed_trained_through is None:
        if args.training_seed.resolve() != default_seed.resolve():
            raise ValueError("custom training seeds require --training-seed-trained-through")
        args.training_seed_trained_through = date(2026, 3, 6)
    if args.training_seed_horizon_sessions is None:
        if args.training_seed.resolve() != default_seed.resolve():
            raise ValueError("custom training seeds require --training-seed-horizon-sessions")
        args.training_seed_horizon_sessions = 3
    if args.training_seed_horizon_sessions != args.horizon_sessions:
        raise ValueError("seed horizon must match --horizon-sessions")
    if args.as_of <= args.training_seed_trained_through:
        raise ValueError("as-of must be later than the training seed cutoff")

    account_data = json.loads(args.portfolio_state_json.read_text(encoding="utf-8"))
    if set(account_data) != {"cash_cny", "holdings_cny"} or not isinstance(account_data["holdings_cny"], dict):
        raise ValueError("portfolio JSON must contain only cash_cny and holdings_cny")
    portfolio = PortfolioState(
        cash_cny=float(account_data["cash_cny"]),
        holdings_cny={str(symbol): float(value) for symbol, value in account_data["holdings_cny"].items()},
    )
    portfolio.validate()

    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    config["prediction_horizon_sessions"] = args.horizon_sessions
    config["risk_budget_annualized"] = args.risk_budget_annualized
    training_seed_bytes = args.training_seed.read_bytes()
    with gzip.open(args.training_seed, "rt", encoding="utf-8") as stream:
        training_state = json.load(stream)

    research = load_oos_research_data(args.baostock_run_dir)
    research.trading_dates.index(args.as_of)
    research_window = _short_research_window(research, args.as_of, args.horizon_sessions)
    diagnostic = run_mvp_research(
        research,
        config,
        initial_state=training_state,
        as_of_date=args.as_of,
        ensure_as_of_decision=True,
    )
    if not diagnostic.get("prediction_decisions"):
        raise ValueError("calibrated forecast preflight produced no decision rows")
    forecast = diagnostic["prediction_decisions"][-1]
    if forecast.get("prediction_date") != args.as_of.isoformat():
        raise ValueError("calibrated forecast date does not match --as-of")
    if forecast.get("forecast_errors_by_method", {}).get("loss_gated_fusion") is None:
        raise ValueError("paired simulation requires a mature loss_gated_fusion forecast")

    expected_mode = (
        "retrospective_source_publication_proxy"
        if args.retrospective_public_cache else "captured_public_cache_cutoff"
    )
    harness_research = json.loads(args.reuse_research_json.read_text(encoding="utf-8"))
    if harness_research.get("status") != "completed" or harness_research.get("as_of") != args.as_of.isoformat():
        raise ValueError("reuse research must be completed for this exact as-of date")
    if harness_research.get("public_evidence_mode") != expected_mode:
        raise ValueError("reuse research evidence mode must match --retrospective-public-cache")

    version_store = HarnessVersionStore(args.harness_version_dir)
    candidate, version_id, version = version_store.load_active()
    memory = harness_research.get("memory", {})
    experience_sha256 = hashlib.sha256(candidate["procedural_experience"].encode("utf-8")).hexdigest()
    if (
        memory.get("procedural_experience_version") != version_id
        or memory.get("procedural_experience_sha256") != experience_sha256
    ):
        raise ValueError("reuse research must match the currently active candidate and procedural memory")

    reused_research = copy.deepcopy(harness_research)
    source_usage = dict(reused_research["task_usage"])
    task_cost = float(source_usage["logical_estimated_cost_cny"])
    source_api_cost = float(source_usage["api_incremental_cost_cny"])
    task_usage = {
        **source_usage,
        "provider_call_count": 0,
        "response_cache_reuse_count": int(source_usage["model_turn_count"]),
        "api_incremental_cost_cny": 0.0,
        "source_task_api_incremental_cost_cny": source_api_cost,
        "research_result_reused_from_artifact": True,
    }
    reused_research["task_usage"] = task_usage
    reused_research["cost"] = {
        "call_count": 0,
        "logical_inference_count": int(task_usage["model_turn_count"]),
        "logical_inference_estimated_cost_cny": task_cost,
        "incremental_api_cost_cny": 0.0,
        "reused_precomputed_result": True,
    }
    candidate_sha256 = hashlib.sha256(_json(candidate).encode("utf-8")).hexdigest()
    harness_context = {
        "status": "completed",
        "research": reused_research,
        "version_id": version_id,
        "candidate_sha256": candidate_sha256,
        "public_evidence_mode": expected_mode,
        "procedural_memory": version.get("procedural_memory"),
        "run_dir": "precomputed_research_artifact",
        "research_cost": {
            "task_logical_cost_cny": task_cost,
            "actual_incremental_api_cost_cny": 0.0,
            "source_task_api_cost_cny": source_api_cost,
            "cost_source": "reused research artifact; no new provider request",
        },
    }
    versions = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "training_seed_sha256": hashlib.sha256(training_seed_bytes).hexdigest(),
        "training_seed_trained_through": args.training_seed_trained_through.isoformat(),
        "training_seed_horizon_sessions": str(args.training_seed_horizon_sessions),
        "public_evidence_mode": expected_mode,
    }
    manifest_path = args.baostock_run_dir / "manifest.jsonl"
    if manifest_path.is_file():
        versions["baostock_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    baseline = create_daily_decision(
        research, config, args.as_of, portfolio, training_state,
        versions=versions, research_diagnostic=diagnostic,
    )
    harness = create_daily_decision(
        research, config, args.as_of, portfolio, training_state,
        versions=versions, harness_context=harness_context, research_diagnostic=diagnostic,
    )
    if baseline.status == "warmup_no_action" or harness.status == "warmup_no_action":
        raise ValueError("paired simulation requires executable optimizer decisions")

    cost_keys = (
        "commission_per_side_bps", "transfer_fee_per_side_bps",
        "stamp_duty_sell_bps", "slippage_per_side_bps",
    )
    costs = {key: float(config[key]) for key in cost_keys}
    common_simulation = {
        "research": research_window,
        "initial_capital_cny": portfolio.nav_cny,
        "costs_bps": costs,
        "liquidity_lookback_sessions": int(config["liquidity_lookback_sessions"]),
        "maximum_daily_participation": float(config["maximum_daily_participation"]),
        "maximum_one_way_turnover": float(config["maximum_one_way_turnover"]),
        "evaluation_start_date": args.as_of,
        "initial_account": account_data,
    }
    baseline_simulation = simulate_portfolio(
        decisions=[{"decision_date": args.as_of, "target_weights": dict(baseline.target_weights)}],
        **common_simulation,
    )
    harness_simulation = simulate_portfolio(
        decisions=[{
            "decision_date": args.as_of,
            "target_weights": dict(harness.target_weights),
            "research_cost_cny": task_cost,
        }],
        **common_simulation,
    )
    baseline_summary = baseline_simulation["summary"]
    harness_summary = harness_simulation["summary"]
    output = {
        "schema_version": "adaptive-mas.harness-paired-simulation.v1",
        "status": "completed",
        "as_of": args.as_of.isoformat(),
        "horizon_sessions": args.horizon_sessions,
        "simulation_scope": "one next-session rebalance followed by holding through the requested short horizon",
        "same_initial_account": True,
        "initial_account_sha256": hashlib.sha256(_json(account_data).encode("utf-8")).hexdigest(),
        "initial_nav_cny": portfolio.nav_cny,
        "shared_calibrated_forecast_sha256": hashlib.sha256(_json(forecast).encode("utf-8")).hexdigest(),
        "public_evidence_mode": expected_mode,
        "strict_historical_pit_verified": bool(harness_research.get("data_limits", {}).get("strict_historical_pit_verified", False)),
        "provider_requests_sent": False,
        "provider_call_count": 0,
        "baseline": {
            "decision": _decision_summary(baseline, harness=False),
            "simulation": baseline_summary,
        },
        "harness": {
            "decision": _decision_summary(harness, harness=True),
            "simulation": harness_summary,
        },
        "paired_differences": {
            "total_return": float(harness_summary["total_return"]) - float(baseline_summary["total_return"]),
            "final_nav_cny": float(harness_summary["final_nav_cny"]) - float(baseline_summary["final_nav_cny"]),
            "total_transaction_cost_cny": float(harness_summary["total_transaction_cost_cny"]) - float(baseline_summary["total_transaction_cost_cny"]),
            "research_cost_cny": float(harness_summary["research_cost_cny"]) - float(baseline_summary["research_cost_cny"]),
        },
        "interpretation_limit": "short retrospective counterfactual; it is not a live-fill result or evidence of investment performance",
    }
    _write_json(args.output, output)
    print(json.dumps({
        "status": output["status"],
        "output": str(args.output),
        "provider_requests_sent": False,
        "total_return_difference": output["paired_differences"]["total_return"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
