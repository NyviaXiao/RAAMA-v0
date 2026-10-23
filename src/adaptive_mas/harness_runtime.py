"""Stable Python entry point for one public-evidence harness decision."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

from adaptive_mas.agents.harness import MAX_CALLS_PER_EXPERIMENT, CallBudget, QwenClient, run_research_harness
from adaptive_mas.agents.public_experts import build_public_expert_snapshot
from adaptive_mas.data.research import ResearchData
from adaptive_mas.decision import DailyDecision, PortfolioState, create_daily_decision
from adaptive_mas.evaluation.mvp_research import run_mvp_research
from adaptive_mas.harness_evolution import HarnessVersionStore


SHANGHAI = ZoneInfo("Asia/Shanghai")


def create_harness_daily_decision(
    research: ResearchData,
    config: dict[str, object],
    as_of: date,
    portfolio: PortfolioState,
    training_state: dict[str, object],
    *,
    public_cache_dir: str | Path,
    harness_version_dir: str | Path,
    run_dir: str | Path,
    training_seed_trained_through: date,
    training_seed_horizon_sessions: int,
    training_seed_cutoff_source: str = "caller_supplied_provenance",
    training_seed_horizon_source: str = "caller_supplied_provenance",
    retrospective_public_cache: bool = False,
    call_ledger_path: str | Path | None = None,
    response_cache_dir: str | Path | None = None,
    versions: Mapping[str, str | None] | None = None,
    max_provider_calls: int = 10,
    announcement_text_limit: int = 0,
    announcement_document_dir: str | Path | None = None,
    client: QwenClient | None = None,
    precomputed_research_result: dict[str, object] | None = None,
) -> DailyDecision:
    """Run public agents, apply their sourced candidate filter and return a normal DailyDecision.

    Input validation and the calibrated forecast are checked before creating a model client or sending
    requests. A warm-up decision returns without using the public-agent model budget.
    """
    portfolio.validate()
    horizon = int(config.get("prediction_horizon_sessions", 0))
    risk_budget = float(config.get("risk_budget_annualized", math.nan))
    if horizon < 1:
        raise ValueError("prediction_horizon_sessions must be positive")
    if training_seed_horizon_sessions < 1:
        raise ValueError("training_seed_horizon_sessions must be positive")
    if horizon != training_seed_horizon_sessions:
        raise ValueError(
            "prediction_horizon_sessions must match the declared training-seed horizon"
        )
    if not math.isfinite(risk_budget) or risk_budget <= 0:
        raise ValueError("risk_budget_annualized must be finite and positive")
    if not 1 <= max_provider_calls <= 10:
        raise ValueError("one daily research task is limited to 1..10 provider calls")
    if announcement_text_limit and announcement_document_dir is None:
        raise ValueError("reading cached announcement text requires an announcement_document_dir")
    if retrospective_public_cache and announcement_text_limit:
        raise ValueError("retrospective public-cache mode does not permit announcement body reads")
    if as_of <= training_seed_trained_through:
        raise ValueError(
            "as_of must be later than the declared training-seed cutoff; the seed file has no embedded date metadata"
        )
    if not training_seed_cutoff_source.strip():
        raise ValueError("training_seed_cutoff_source must describe the cutoff provenance")
    if not training_seed_horizon_source.strip():
        raise ValueError("training_seed_horizon_source must describe the horizon provenance")
    research.trading_dates.index(as_of)

    # Compute the calibrated forecast exactly once before any provider request.
    # The final decision reuses this diagnostic rather than fitting/replaying the
    # full forecast path a second time.
    research_diagnostic = run_mvp_research(
        research,
        config,
        initial_state=training_state,
        as_of_date=as_of,
        ensure_as_of_decision=True,
    )
    if not research_diagnostic.get("prediction_decisions"):
        raise ValueError("calibrated forecast preflight produced no decision rows")
    calibrated_forecast = research_diagnostic["prediction_decisions"][-1]
    if calibrated_forecast.get("prediction_date") != as_of.isoformat():
        raise ValueError(f"no calibrated forecast was produced for {as_of.isoformat()}")
    decision_versions = {
        **dict(versions or {}),
        "training_seed_trained_through": training_seed_trained_through.isoformat(),
        "training_seed_cutoff_source": training_seed_cutoff_source,
        "training_seed_horizon_sessions": str(training_seed_horizon_sessions),
        "training_seed_horizon_source": training_seed_horizon_source,
        "public_evidence_mode": (
            "retrospective_source_publication_proxy"
            if retrospective_public_cache else "captured_public_cache_cutoff"
        ),
    }
    if calibrated_forecast["forecast_errors_by_method"].get("loss_gated_fusion") is None:
        baseline = create_daily_decision(
            research,
            config,
            as_of,
            portfolio,
            training_state,
            decision_versions,
            research_diagnostic=research_diagnostic,
        )
        return replace(
            baseline,
            evidence={
                **baseline.evidence,
                "harness": {
                    "status": "skipped_forecast_warmup",
                    "provider_call_count": 0,
                    "reason": "no mature calibrated forecast is available for the optimizer",
                },
            }
        )

    store = HarnessVersionStore(harness_version_dir)
    candidate, version_id, version = store.load_active()
    task_dir = Path(run_dir)
    candidate_sha256 = hashlib.sha256(
        json.dumps(candidate, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if precomputed_research_result is not None:
        research_result = copy.deepcopy(precomputed_research_result)
        if research_result.get("status") != "completed" or research_result.get("as_of") != as_of.isoformat():
            raise ValueError("precomputed public research must be completed for this exact decision date")
        expected_data_mode = (
            "retrospective_source_publication_proxy"
            if retrospective_public_cache else "captured_public_cache_cutoff"
        )
        if research_result.get("public_evidence_mode") != expected_data_mode:
            raise ValueError("precomputed research data mode must match the explicit API mode")
        memory = research_result.get("memory", {})
        experience_sha256 = hashlib.sha256(candidate["procedural_experience"].encode("utf-8")).hexdigest()
        if (
            memory.get("procedural_experience_version") != version_id
            or memory.get("procedural_experience_sha256") != experience_sha256
        ):
            raise ValueError("precomputed research must use the currently loaded harness version")
        source_usage = dict(research_result["task_usage"])
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
        research_result["task_usage"] = task_usage
        research_result["cost"] = {
            "call_count": 0,
            "logical_inference_count": int(task_usage["model_turn_count"]),
            "logical_inference_estimated_cost_cny": task_cost,
            "incremental_api_cost_cny": 0.0,
            "reused_precomputed_result": True,
        }
        task_incremental_cost = 0.0
        research_cost_source = "previously paid research result in the shared evolution experiment"
    else:
        decision_time = datetime.combine(as_of, datetime.min.time()).replace(hour=16, tzinfo=SHANGHAI)
        public_snapshot = build_public_expert_snapshot(
            public_cache_dir,
            decision_time,
            horizon,
            allowed_symbols=set(research.membership_at(as_of)) | set(portfolio.holdings_cny),
            announcement_text_limit=announcement_text_limit,
            announcement_document_dir=announcement_document_dir,
            include_qwen=False,
            retrospective_backfill_proxy=retrospective_public_cache,
        )
        budget = CallBudget(
            max_calls=MAX_CALLS_PER_EXPERIMENT,
            ledger_path=Path(call_ledger_path) if call_ledger_path else task_dir / "calls.jsonl",
            response_cache_dir=Path(response_cache_dir) if response_cache_dir else task_dir / "response_cache",
        )
        if budget.unresolved_call:
            raise RuntimeError("the shared call ledger has an unknown outbound outcome; inspect it before continuing")
        remaining_calls = budget.remaining_provider_calls
        if remaining_calls <= 0:
            raise RuntimeError("the shared 160-call experiment ledger has no provider requests remaining")
        budget.limit_next_task(min(max_provider_calls, remaining_calls))
        call_start = len(budget.calls)
        logical_start = len(budget.logical_calls)
        research_result = run_research_harness(
            research,
            as_of,
            public_snapshot,
            candidate,
            client or QwenClient.from_environment(),
            budget,
            config,
            candidate_version_id=version_id,
            procedural_memory=version.get("procedural_memory"),
        )
        task_calls = budget.calls[call_start:]
        task_logical_calls = budget.logical_calls[logical_start:]
        provider_task_call_count = sum(row.get("status") in {"started", "completed"} for row in task_calls)
        task_cost = sum(float(row["logical_estimated_cost_cny"]) for row in task_logical_calls)
        task_incremental_cost = sum(float(row.get("actual_estimated_cost_cny", 0.0)) for row in task_calls)
        task_usage = {
            "provider_call_count": provider_task_call_count,
            "attempt_count": len(task_calls),
            "response_cache_reuse_count": sum(row["source"] == "response_cache" for row in task_logical_calls),
            "model_turn_count": len(task_logical_calls),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in task_logical_calls),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in task_logical_calls),
            "logical_estimated_cost_cny": task_cost,
            "api_incremental_cost_cny": task_incremental_cost,
        }
        research_result["task_usage"] = task_usage
        research_cost_source = "provider responses and exact-response cache reuse in this task"
    harness_context = {
        "status": "completed",
        "research": research_result,
        "version_id": version_id,
        "candidate_sha256": candidate_sha256,
        "public_evidence_mode": (
            "retrospective_source_publication_proxy"
            if retrospective_public_cache else "captured_public_cache_cutoff"
        ),
        "procedural_memory": version.get("procedural_memory"),
        "run_dir": str(task_dir),
        "research_cost": {
            "task_logical_cost_cny": task_cost,
            "actual_incremental_api_cost_cny": task_incremental_cost,
            "source_task_api_cost_cny": task_usage.get("source_task_api_incremental_cost_cny", task_incremental_cost),
            "cost_source": research_cost_source,
            "accounting": "one task-local logical inference charge; exact-response cache reuse does not change comparison cost",
        },
    }
    updated_versions = dict(versions or {})
    updated_versions.update({
        "training_seed_trained_through": training_seed_trained_through.isoformat(),
        "training_seed_cutoff_source": training_seed_cutoff_source,
        "training_seed_horizon_sessions": str(training_seed_horizon_sessions),
        "training_seed_horizon_source": training_seed_horizon_source,
        "public_evidence_mode": (
            "retrospective_source_publication_proxy"
            if retrospective_public_cache else "captured_public_cache_cutoff"
        ),
    })
    updated_versions.update({
        "harness_version_id": version_id,
        "harness_candidate_sha256": candidate_sha256,
    })
    return create_daily_decision(
        research,
        config,
        as_of,
        portfolio,
        training_state,
        versions=updated_versions,
        harness_context=harness_context,
        research_diagnostic=research_diagnostic,
    )
