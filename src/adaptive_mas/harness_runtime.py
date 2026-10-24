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
from adaptive_mas.decision import DailyDecision, DecisionOrder, PortfolioState, create_daily_decision
from adaptive_mas.evaluation.mvp_research import run_mvp_research
from adaptive_mas.harness_state import HarnessStateStore, UnknownOutboundOutcome
from adaptive_mas.package_identity import source_package_identity
from adaptive_mas.release_governance import HarnessReleaseStore
from adaptive_mas.research_provenance import (
    build_research_package_provenance as _build_research_package_provenance,
    endpoint_sha256 as _endpoint_sha256,
    market_snapshot_identity as _market_snapshot_identity,
    research_artifact_sha256 as _research_artifact_sha256,
    research_cost_summary as _research_cost_summary,
    stable_hash as _stable_hash,
    validate_research_package_provenance as _validate_research_package_provenance,
)
from adaptive_mas.risk_profile import apply_risk_profile, validate_canonical_risk_profile


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _restore_decision(payload: Mapping[str, object]) -> DailyDecision:
    value = dict(payload)
    value["orders"] = tuple(
        DecisionOrder(**dict(row)) for row in value.get("orders", [])
    )
    return DailyDecision(**value)


def _implementation_identity() -> dict[str, object]:
    return source_package_identity()


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
    state_store_path: str | Path | None = None,
    task_id: str | None = None,
    idempotency_key: str | None = None,
    risk_profile: Mapping[str, object] | None = None,
) -> DailyDecision:
    """Run public agents, apply their sourced candidate filter and return a normal DailyDecision.

    Input validation and the calibrated forecast are checked before creating a model client or sending
    requests. A warm-up decision returns without using the public-agent model budget.
    """
    portfolio.validate()
    checked_risk_profile = (
        validate_canonical_risk_profile(risk_profile) if risk_profile is not None else None
    )
    decision_config = (
        apply_risk_profile(config, checked_risk_profile)
        if checked_risk_profile is not None else config
    )
    horizon = int(config.get("prediction_horizon_sessions", 0))
    risk_budget = float(decision_config.get("risk_budget_annualized", math.nan))
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
    forecast_errors = calibrated_forecast.get("forecast_errors_by_method", {})
    model_risk_metrics = [
        {"metric": f"matured_forecast_error_{method}", "value": float(value), "unit": "return_fraction"}
        for method, value in forecast_errors.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ] if isinstance(forecast_errors, Mapping) else []
    model_risk_core: dict[str, object] = {
        "status": "available" if model_risk_metrics else "unavailable",
        "available_at": datetime.combine(as_of, datetime.min.time()).replace(hour=16, tzinfo=SHANGHAI).isoformat(),
        "period_start": training_seed_trained_through.isoformat(),
        "period_end": as_of.isoformat(),
        "metrics": model_risk_metrics,
        "source_refs": [
            f"training-seed-cutoff:{training_seed_trained_through.isoformat()}",
            f"calibrated-forecast:{as_of.isoformat()}",
            f"market-snapshot:{_market_snapshot_identity(research, as_of)['sha256']}",
        ],
    }
    model_risk_context: dict[str, object] = {
        **model_risk_core,
        "context_sha256": _stable_hash(model_risk_core),
    }
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
    if checked_risk_profile is not None:
        decision_versions.update({
            "risk_profile_sha256": str(checked_risk_profile["profile_sha256"]),
            "risk_profile_schema_version": str(checked_risk_profile["schema_version"]),
        })
    if calibrated_forecast["forecast_errors_by_method"].get("loss_gated_fusion") is None:
        baseline = create_daily_decision(
            research,
            decision_config,
            as_of,
            portfolio,
            training_state,
            decision_versions,
            research_diagnostic=research_diagnostic,
        )
        profile_constraints = (
            checked_risk_profile.get("constraints", {})
            if checked_risk_profile is not None else {}
        )
        profile_template = (
            checked_risk_profile.get("template") if checked_risk_profile is not None else None
        )
        profile_template_label = {
            "conservative": "保守",
            "balanced": "平衡",
            "aggressive": "进取",
        }.get(profile_template, "系统默认")
        profile_summary = (
            f"风险档案 {profile_template_label}，年化波动率上限 {float(profile_constraints['risk_budget_annualized']):.2%}，"
            f"单票上限 {float(profile_constraints['maximum_symbol_weight']):.2%}，"
            f"单向换手上限 {float(profile_constraints['maximum_one_way_turnover']):.2%}，"
            f"最低现金 {float(profile_constraints['minimum_cash_weight']):.2%}。"
            if isinstance(profile_constraints, Mapping) and checked_risk_profile is not None
            else ""
        )
        warmup_evidence = {
            **baseline.evidence,
            "risk_profile": checked_risk_profile,
            "harness": {
                "status": "skipped_forecast_warmup",
                "provider_call_count": 0,
                "reason": "no mature calibrated forecast is available for the optimizer",
                "hard_risk_outcome": {
                    "status": "no_optimized_proposal",
                    "orders_blocked": False,
                    "new_orders": 0,
                    "reason": "no calibrated proposal was generated; no orders are permitted",
                },
                "investment_manager_feedback": {
                    "status": "warmup_no_action",
                    "summary": f"尚无成熟校准收益方案；未生成订单，等待成熟反馈。{profile_summary}年化波动率预算不保证最大回撤或最大亏损。",
                    "orders": [],
                    "risk_review_status": "not_applicable_without_proposal",
                    "risk_profile": checked_risk_profile,
                    "risk_profile_sha256": (
                        checked_risk_profile.get("profile_sha256")
                        if checked_risk_profile is not None else None
                    ),
                    "drawdown_guarantee": False,
                },
            },
        }
        warmup_decision = replace(
            baseline,
            evidence=warmup_evidence,
        )
        if state_store_path is not None:
            store = HarnessStateStore(state_store_path)
            warmup_task_id = task_id or Path(run_dir).name
            warmup_identity = {
                "as_of": as_of.isoformat(),
                "portfolio": {
                    "cash_cny": portfolio.cash_cny,
                    "holdings_cny": dict(sorted(portfolio.holdings_cny.items())),
                },
                "config_sha256": _stable_hash(config),
                "risk_profile_sha256": (
                    checked_risk_profile.get("profile_sha256")
                    if checked_risk_profile is not None else None
                ),
                "training_state_sha256": _stable_hash(training_state),
                "versions": dict(versions or {}),
                "forecast_status": calibrated_forecast.get("status"),
                "warmup": True,
            }
            task = store.open_task(
                warmup_task_id,
                idempotency_key or warmup_task_id,
                warmup_identity,
                max_provider_calls=0,
            )
            if task["status"] == "completed":
                return _restore_decision(task["result"])
            warmup_task_id = str(task["task_id"])
            store.checkpoint_task(warmup_task_id, "warmup_no_action")
            store.put_artifact(warmup_task_id, "decision", warmup_decision.to_dict())
            store.complete_task(
                warmup_task_id,
                warmup_decision.to_dict(),
                feedback_id="investment-manager-final-v1",
                feedback=warmup_evidence["harness"]["investment_manager_feedback"],
            )
        return warmup_decision

    release_store = HarnessReleaseStore(harness_version_dir)
    runtime_seed_provenance = {
        "trained_through": training_seed_trained_through.isoformat(),
        "cutoff_source": training_seed_cutoff_source,
        "horizon_sessions": training_seed_horizon_sessions,
        "horizon_source": training_seed_horizon_source,
    }
    candidate, version_id, version = release_store.load_active(
        config=config,
        training_state=training_state,
        training_seed_provenance=runtime_seed_provenance,
        versions=versions or {},
    )
    task_dir = Path(run_dir)
    candidate_sha256 = hashlib.sha256(
        json.dumps(candidate, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    public_snapshot = None
    if precomputed_research_result is None:
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

    active_client = None
    provider_identity: dict[str, object] = {"status": "not_used_for_precomputed_research"}
    if precomputed_research_result is None:
        active_client = client or QwenClient.from_environment()
        provider_identity = {
            "model": active_client.model,
            "endpoint_sha256": _endpoint_sha256(active_client.endpoint),
        }
    else:
        active_client = client
        source_provenance = precomputed_research_result.get("package_provenance", {})
        provider_identity = {
            "status": "precomputed_source_manifest",
            "source_manifest_sha256": (
                source_provenance.get("source_manifest_sha256")
                if isinstance(source_provenance, Mapping) else None
            ),
            "configured_model": client.model if client is not None else None,
            "configured_endpoint_sha256": (
                _endpoint_sha256(client.endpoint) if client is not None else None
            ),
        }

    ledger_path = Path(call_ledger_path) if call_ledger_path else task_dir / "calls.jsonl"
    cache_path = Path(response_cache_dir) if response_cache_dir else task_dir / "response_cache"
    task_identifier = task_id or task_dir.name
    implementation_identity = _implementation_identity()
    market_snapshot_identity = _market_snapshot_identity(research, as_of)
    training_seed_provenance = {
        "trained_through": training_seed_trained_through.isoformat(),
        "cutoff_source": training_seed_cutoff_source,
        "horizon_sessions": training_seed_horizon_sessions,
        "horizon_source": training_seed_horizon_source,
    }
    public_snapshot_sha256 = (
        _stable_hash(public_snapshot) if public_snapshot is not None else None
    )
    precomputed_research_artifact_sha256 = (
        _research_artifact_sha256(precomputed_research_result)
        if precomputed_research_result is not None else None
    )
    precomputed_provenance = (
        precomputed_research_result.get("package_provenance", {})
        if precomputed_research_result is not None else {}
    )
    source_manifest_sha256 = (
        precomputed_provenance.get("source_manifest_sha256")
        if isinstance(precomputed_provenance, Mapping) else None
    )
    package_identity = {
        "schema_version": "adaptive-mas.runtime-package-identity.v1",
        "as_of": as_of.isoformat(),
        "candidate_version_id": version_id,
        "candidate_sha256": candidate_sha256,
        "config_sha256": _stable_hash(config),
        "task_combination_config_sha256": _stable_hash(config),
        "risk_profile_sha256": (
            checked_risk_profile.get("profile_sha256")
            if checked_risk_profile is not None else None
        ),
        "training_state_sha256": _stable_hash(training_state),
        "training_seed_provenance": training_seed_provenance,
        "versions": dict(versions or {}),
        "market_snapshot_sha256": market_snapshot_identity["sha256"],
        "model_risk_context_sha256": model_risk_context["context_sha256"],
        "public_snapshot_sha256": public_snapshot_sha256,
        "precomputed_research_artifact_sha256": precomputed_research_artifact_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "provider_identity": provider_identity,
        "implementation_identity": implementation_identity,
        "retrospective_public_cache": retrospective_public_cache,
        "prediction_horizon_sessions": horizon,
        "public_evidence_mode": (
            "retrospective_source_publication_proxy"
            if retrospective_public_cache else "captured_public_cache_cutoff"
        ),
    }
    runtime_package_identity_sha256 = _stable_hash(package_identity)
    request_identity = {
        "as_of": as_of.isoformat(),
        "portfolio": {
            "cash_cny": portfolio.cash_cny,
            "holdings_cny": dict(sorted(portfolio.holdings_cny.items())),
        },
        "config_sha256": _stable_hash(config),
        "risk_profile_sha256": (
            checked_risk_profile.get("profile_sha256")
            if checked_risk_profile is not None else None
        ),
        "training_state_sha256": _stable_hash(training_state),
        "training_seed_provenance": training_seed_provenance,
        "versions": dict(versions or {}),
        "candidate_version_id": version_id,
        "candidate_sha256": candidate_sha256,
        "public_snapshot_sha256": public_snapshot_sha256,
        "precomputed_research_artifact_sha256": precomputed_research_artifact_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "market_snapshot": market_snapshot_identity,
        "model_risk_context_sha256": model_risk_context["context_sha256"],
        "provider_identity": provider_identity,
        "implementation_identity": implementation_identity,
        "runtime_package_identity": package_identity,
        "runtime_package_identity_sha256": runtime_package_identity_sha256,
        "retrospective_public_cache": retrospective_public_cache,
        "prediction_horizon_sessions": horizon,
    }
    state_store = HarnessStateStore(state_store_path) if state_store_path is not None else None
    task_state = None
    if state_store is not None:
        task_state = state_store.open_task(
            task_identifier,
            idempotency_key or task_identifier,
            request_identity,
            max_provider_calls=max_provider_calls,
        )
        if task_state["status"] == "completed":
            return _restore_decision(task_state["result"])
        task_identifier = str(task_state["task_id"])
        state_store.checkpoint_task(
            task_identifier,
            "preflight_completed",
            {"candidate_version_id": version_id, "candidate_sha256": candidate_sha256},
        )

    saved_research_result = (
        state_store.get_artifact(task_identifier, "research_result")
        if state_store is not None else None
    )
    reusable_research_result = (
        precomputed_research_result
        if precomputed_research_result is not None
        else saved_research_result
    )
    if reusable_research_result is not None:
        research_result = copy.deepcopy(reusable_research_result)
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
        provenance = research_result.get("package_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("precomputed research must include complete package provenance")
        validated_manifest_sha256 = _validate_research_package_provenance(
            research_result,
            provenance,
            as_of=as_of,
            public_evidence_mode=expected_data_mode,
            horizon=horizon,
            candidate_version_id=version_id,
            candidate_sha256=candidate_sha256,
            implementation_identity=implementation_identity,
            market_identity=market_snapshot_identity,
            client=active_client,
            config=config,
            model_risk_context_sha256=str(model_risk_context["context_sha256"]),
        )
        if (
            precomputed_research_result is not None
            and validated_manifest_sha256 != package_identity["source_manifest_sha256"]
        ):
            raise ValueError("validated source manifest does not match the task package identity")
        source_usage = dict(research_result["task_usage"])
        source_cost = dict(research_result["cost"])
        task_cost = float(source_usage["logical_estimated_cost_cny"])
        if saved_research_result is not None and precomputed_research_result is None:
            # A restart is the same logical task. Preserve its original provider spend and usage.
            task_usage = {**source_usage, "research_result_recovered_from_same_task": True}
            task_incremental_cost = float(source_usage["api_incremental_cost_cny"])
            research_cost_source = "original provider responses recovered from this task"
            research_result["receiving_task_usage"] = task_usage
            research_result["receiving_task_cost"] = {
                **source_cost,
                "recovered_same_task_artifact": True,
            }
        else:
            # A supplied artifact was paid for elsewhere; this task adds no provider spend.
            source_api_cost = float(source_cost["incremental_api_cost_cny"])
            task_usage = {
                **source_usage,
                "provider_call_count": 0,
                "response_cache_reuse_count": int(source_usage["model_turn_count"]),
                "api_incremental_cost_cny": 0.0,
                "source_task_api_incremental_cost_cny": source_api_cost,
                "research_result_reused_from_artifact": True,
            }
            research_result["receiving_task_usage"] = task_usage
            research_result["receiving_task_cost"] = {
                "call_count": 0,
                "logical_inference_count": int(task_usage["model_turn_count"]),
                "logical_inference_estimated_cost_cny": task_cost,
                "incremental_api_cost_cny": 0.0,
                "source_task_incremental_api_cost_cny": source_api_cost,
                "reused_precomputed_result": True,
            }
            task_incremental_cost = 0.0
            research_cost_source = "previously paid research result in the shared evolution experiment"
    else:
        if public_snapshot is None:
            raise RuntimeError("public evidence snapshot was not prepared")
        budget = CallBudget(
            max_calls=MAX_CALLS_PER_EXPERIMENT,
            ledger_path=ledger_path,
            response_cache_dir=cache_path,
            state_store=state_store,
            task_id=task_identifier if state_store is not None else None,
            budget_scope_id=str(ledger_path.resolve()) if state_store is not None else None,
            task_provider_call_limit=max_provider_calls if state_store is not None else None,
            imported_scope_calls=0,
            package_identity_sha256=runtime_package_identity_sha256,
        )
        shared_calls_used = 0
        if state_store is not None:
            shared_calls_used = state_store.ensure_budget_scope(
                str(ledger_path.resolve()),
                MAX_CALLS_PER_EXPERIMENT,
                imported_calls=budget.provider_call_count,
            )
            budget.imported_scope_calls = budget.provider_call_count
            unresolved_state_requests = state_store.reconcile_request_cache(task_identifier, cache_path)
            if unresolved_state_requests:
                state_store.fail_task(
                    task_identifier,
                    {"reason": "provider request has unknown delivery outcome", "request_hashes": unresolved_state_requests},
                    unknown_outcome=True,
                )
                raise UnknownOutboundOutcome(
                    "task has a provider request with unknown delivery outcome; inspect the durable request record before continuing"
                )
        if budget.unresolved_call:
            if state_store is not None:
                state_store.fail_task(
                    task_identifier,
                    {"reason": "legacy ledger contains an unresolved provider request"},
                    unknown_outcome=True,
                )
            raise RuntimeError("the shared call ledger has an unknown outbound outcome; inspect it before continuing")
        remaining_calls = min(
            budget.remaining_provider_calls,
            MAX_CALLS_PER_EXPERIMENT - shared_calls_used,
        )
        if remaining_calls <= 0:
            raise RuntimeError("the shared 160-call experiment ledger has no provider requests remaining")
        budget.limit_next_task(min(max_provider_calls, remaining_calls))
        call_start = len(budget.calls)
        logical_start = len(budget.logical_calls)
        if state_store is not None:
            state_store.checkpoint_task(task_identifier, "research_started")
        try:
            research_result = run_research_harness(
                research,
                as_of,
                public_snapshot,
                candidate,
                active_client if active_client is not None else QwenClient.from_environment(),
                budget,
                config,
                candidate_version_id=version_id,
                procedural_memory=version.get("procedural_memory"),
                model_risk_context=model_risk_context,
            )
            if state_store is not None:
                task_calls = [row for row in budget.calls if row.get("task_id") == task_identifier]
                task_logical_calls = [
                    row for row in budget.logical_calls if row.get("task_id") == task_identifier
                ]
            else:
                task_calls = budget.calls[call_start:]
                task_logical_calls = budget.logical_calls[logical_start:]
            provider_task_call_count = sum(
                row.get("status") in {"started", "completed"} for row in task_calls
            )
            task_cost = sum(float(row["logical_estimated_cost_cny"]) for row in task_logical_calls)
            task_incremental_cost = sum(
                float(row.get("actual_estimated_cost_cny", 0.0)) for row in task_calls
            )
            task_usage = {
                "provider_call_count": provider_task_call_count,
                "attempt_count": len(task_calls),
                "response_cache_reuse_count": sum(
                    row["source"] == "response_cache" for row in task_logical_calls
                ),
                "model_turn_count": len(task_logical_calls),
                "prompt_tokens": sum(int(row["prompt_tokens"]) for row in task_logical_calls),
                "completion_tokens": sum(int(row["completion_tokens"]) for row in task_logical_calls),
                "logical_estimated_cost_cny": task_cost,
                "api_incremental_cost_cny": task_incremental_cost,
            }
            research_result["task_usage"] = task_usage
            research_result["cost"] = _research_cost_summary(task_usage)
            research_result["receiving_task_usage"] = dict(task_usage)
            research_result["receiving_task_cost"] = dict(research_result["cost"])
            research_result["package_provenance"] = _build_research_package_provenance(
                research_result,
                as_of=as_of,
                public_evidence_mode=(
                    "retrospective_source_publication_proxy"
                    if retrospective_public_cache else "captured_public_cache_cutoff"
                ),
                horizon=horizon,
                config=config,
                candidate_version_id=version_id,
                candidate_sha256=candidate_sha256,
                implementation_identity=implementation_identity,
                market_identity=market_snapshot_identity,
                model=active_client.model,
                endpoint=active_client.endpoint,
                source_snapshot=public_snapshot,
                model_risk_context_sha256=str(model_risk_context["context_sha256"]),
            )
        except Exception as exc:
            if state_store is not None:
                state_store.fail_task(
                    task_identifier,
                    {"error_type": type(exc).__name__, "message": str(exc)},
                    unknown_outcome=budget.unresolved_call,
                )
            raise
        research_cost_source = "provider responses and exact-response cache reuse in this task"
    if state_store is not None:
        if saved_research_result is None:
            state_store.put_artifact(task_identifier, "research_result", research_result)
        state_store.checkpoint_task(
            task_identifier,
            "research_completed",
            {"task_usage": research_result.get("task_usage", {})},
        )
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
        "runtime_package_identity_sha256": runtime_package_identity_sha256,
        "runtime_package_identity": package_identity,
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
        "runtime_package_identity_sha256": runtime_package_identity_sha256,
        "tool_schema_sha256": implementation_identity["tool_schema_sha256"],
        "evaluator_sha256": implementation_identity["evaluator_sha256"],
    })
    if checked_risk_profile is not None:
        updated_versions.update({
            "risk_profile_sha256": str(checked_risk_profile["profile_sha256"]),
            "risk_profile_schema_version": str(checked_risk_profile["schema_version"]),
        })
    try:
        decision = create_daily_decision(
            research,
            decision_config,
            as_of,
            portfolio,
            training_state,
            versions=updated_versions,
            harness_context=harness_context,
            research_diagnostic=research_diagnostic,
        )
    except Exception as exc:
        if state_store is not None:
            state_store.fail_task(
                task_identifier,
                {"error_type": type(exc).__name__, "message": str(exc)},
            )
        raise
    if state_store is not None:
        decision_payload = decision.to_dict()
        state_store.put_artifact(task_identifier, "decision", decision_payload)
        feedback_payload = decision.evidence.get("investment_manager_feedback")
        state_store.complete_task(
            task_identifier,
            decision_payload,
            feedback_id=("investment-manager-final-v1" if isinstance(feedback_payload, Mapping) else None),
            feedback=feedback_payload if isinstance(feedback_payload, Mapping) else None,
        )
    return decision
