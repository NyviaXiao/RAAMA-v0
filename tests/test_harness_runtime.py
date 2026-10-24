from __future__ import annotations

import hashlib
import json
from datetime import date

import pytest

from adaptive_mas.agents.harness_prompts import seed_candidate
from adaptive_mas.agents.harness import CallBudget
from adaptive_mas.data.research import MembershipSnapshot, ResearchData
from adaptive_mas.decision import DailyDecision, PortfolioState
from adaptive_mas import harness_runtime
from adaptive_mas import harness_evolution
from adaptive_mas.harness_evolution import HarnessCase, evaluate_candidate


def _runtime_inputs(as_of: date):
    return (
        ResearchData([date(2026, 3, 6), as_of], [], [], [], []),
        {"prediction_horizon_sessions": 3, "risk_budget_annualized": 0.2},
        PortfolioState(cash_cny=100.0, holdings_cny={}),
        {},
    )


class _MockProvider:
    model = "fixture-model"
    endpoint = "https://fixture.invalid"

    def __init__(self):
        self.turns = {}

    @staticmethod
    def _call(name: str, arguments: dict[str, object], call_id: str):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    def complete(self, _messages, _tools, *, role, budget=None, **_kwargs):
        turn = self.turns.get(role, 0) + 1
        self.turns[role] = turn
        if budget is not None:
            request_sha256 = hashlib.sha256(f"{role}:{turn}".encode("utf-8")).hexdigest()
            call_index = len(budget.calls) + 1
            budget.begin_call(100, 128, role, request_sha256)
            budget.finish_call(
                call_index,
                {"prompt_tokens": 100, "completion_tokens": 20},
                elapsed_ms=0.1,
            )
        if role == "market" and turn == 1:
            call = self._call("read_market_signals", {"symbols": ["600000"], "limit_per_signal": 1}, f"{role}-{turn}")
        elif role == "market":
            call = self._call("submit_report", {
                "summary": "No eligible market evidence was available.",
                "findings": [], "unresolved_items": [],
            }, f"{role}-{turn}")
        elif role == "fundamental":
            call = self._call("submit_report", {
                "summary": "No eligible fundamental evidence was available.",
                "findings": [], "unresolved_items": [],
            }, f"{role}-{turn}")
        elif role == "research_manager" and turn == 1:
            call = self._call("plan_research", {"goal": "Check whether any cited public evidence supports a priority."}, f"{role}-{turn}")
        else:
            call = self._call("finalize_research", {
                "summary": "No symbols had cited public evidence to prioritize.",
                "priority_symbols": [], "unresolved_items": [],
                "no_query_reason": "No cited evidence supports a targeted follow-up.",
            }, f"{role}-{turn}")
        return {"choices": [{"message": {"tool_calls": [call]}}]}


def _fixture_precomputed(
    as_of: date,
    candidate: dict[str, str],
    config: dict[str, object],
    training_state: dict[str, object],
    research: ResearchData,
) -> dict[str, object]:
    implementation = harness_runtime._implementation_identity()
    endpoint = "https://fixture.invalid"
    source_snapshot = {
        "schema_version": "public-expert-snapshot.v1",
        "data_mode": "captured_public_cache_cutoff",
        "requested_decision_time": f"{as_of.isoformat()}T16:00:00+08:00",
        "evaluation_horizon_sessions": 3,
    }
    training_seed_provenance = {
        "trained_through": "2026-03-06",
        "cutoff_source": "caller_supplied_provenance",
        "horizon_sessions": 3,
        "horizon_source": "caller_supplied_provenance",
    }
    market_snapshot_identity = harness_runtime._market_snapshot_identity(research, as_of)
    candidate_sha256 = harness_runtime._stable_hash(candidate)
    result = {
        "status": "completed",
        "as_of": as_of.isoformat(),
        "public_evidence_mode": "captured_public_cache_cutoff",
        "memory": {
            "procedural_experience_version": "seed",
            "procedural_experience_sha256": hashlib.sha256(
                candidate["procedural_experience"].encode("utf-8")
            ).hexdigest(),
        },
        "task_usage": {
            "provider_call_count": 2,
            "attempt_count": 2,
            "response_cache_reuse_count": 0,
            "logical_estimated_cost_cny": 0.01,
            "api_incremental_cost_cny": 0.01,
            "model_turn_count": 2,
            "prompt_tokens": 100,
            "completion_tokens": 20,
        },
    }
    result["cost"] = harness_runtime._research_cost_summary(result["task_usage"])
    result["package_provenance"] = harness_runtime._build_research_package_provenance(
        result,
        as_of=as_of,
        public_evidence_mode="captured_public_cache_cutoff",
        horizon=3,
        config=config,
        candidate_version_id="seed",
        candidate_sha256=candidate_sha256,
        implementation_identity=implementation,
        market_identity=market_snapshot_identity,
        model="fixture-model",
        endpoint=endpoint,
        source_snapshot=source_snapshot,
    )
    return result


def test_runtime_reuses_single_preflight_forecast_for_final_decision(tmp_path, monkeypatch) -> None:
    as_of = date(2026, 9, 23)
    research, config, portfolio, training_state = _runtime_inputs(as_of)
    candidate = seed_candidate()
    diagnostic = {
        "prediction_decisions": [{
            "prediction_date": as_of.isoformat(),
            "forecast_errors_by_method": {"loss_gated_fusion": 0.01},
        }],
    }
    precomputed = _fixture_precomputed(as_of, candidate, config, training_state, research)
    forecast_runs = []
    final_calls = []

    def fake_forecast(*args, **kwargs):
        forecast_runs.append((args, kwargs))
        return diagnostic

    def fake_decision(*args, **kwargs):
        final_calls.append((args, kwargs))
        return diagnostic

    monkeypatch.setattr(harness_runtime, "run_mvp_research", fake_forecast)
    monkeypatch.setattr(harness_runtime, "create_daily_decision", fake_decision)

    result = harness_runtime.create_harness_daily_decision(
        research,
        config,
        as_of,
        portfolio,
        training_state,
        public_cache_dir=tmp_path / "public",
        harness_version_dir=tmp_path / "versions",
        run_dir=tmp_path / "run",
        training_seed_trained_through=date(2026, 3, 6),
        training_seed_horizon_sessions=3,
        precomputed_research_result=precomputed,
    )

    assert result is diagnostic
    assert len(forecast_runs) == 1
    assert len(final_calls) == 1
    assert final_calls[0][1]["research_diagnostic"] is diagnostic
    assert final_calls[0][1]["versions"]["training_seed_trained_through"] == "2026-03-06"


def test_runtime_does_not_load_research_selected_candidate_as_production(tmp_path, monkeypatch):
    as_of = date(2026, 9, 23)
    research, config, portfolio, training_state = _runtime_inputs(as_of)
    candidate = seed_candidate()
    precomputed = _fixture_precomputed(as_of, candidate, config, training_state, research)
    selected = seed_candidate()
    selected["procedural_experience"] += " Research-only selected candidate."
    versions = harness_evolution.HarnessVersionStore(tmp_path / "versions")
    versions.adopt(
        selected,
        previous_version_id="seed",
        selection={"fixture_only": True},
        reason="A local research selection must not publish the production bundle.",
        run_id="research-only-run",
        version_id="research-only-candidate",
    )
    diagnostic = {"prediction_decisions": [{
        "prediction_date": as_of.isoformat(),
        "forecast_errors_by_method": {"loss_gated_fusion": 0.01},
    }]}
    final_calls = []
    monkeypatch.setattr(harness_runtime, "run_mvp_research", lambda *_args, **_kwargs: diagnostic)
    monkeypatch.setattr(
        harness_runtime,
        "create_daily_decision",
        lambda *args, **kwargs: final_calls.append((args, kwargs)) or diagnostic,
    )

    result = harness_runtime.create_harness_daily_decision(
        research,
        config,
        as_of,
        portfolio,
        training_state,
        public_cache_dir=tmp_path / "public",
        harness_version_dir=tmp_path / "versions",
        run_dir=tmp_path / "run",
        training_seed_trained_through=date(2026, 3, 6),
        training_seed_horizon_sessions=3,
        precomputed_research_result=precomputed,
    )

    assert result is diagnostic
    assert final_calls[0][1]["harness_context"]["runtime_package_identity"][
        "candidate_version_id"
    ] == "seed"
    assert versions.load_active()[1] == "research-only-candidate"


def test_gepa_trace_round_trip_is_reusable_and_fails_closed_on_snapshot_drift(tmp_path, monkeypatch):
    as_of = date(2026, 9, 23)
    seed_cutoff = date(2026, 3, 6)
    research = ResearchData(
        [seed_cutoff, as_of], [], [], [],
        [MembershipSnapshot(seed_cutoff, seed_cutoff, frozenset({"sh.600000"}), "fixture-membership")],
    )
    runtime_config = {"prediction_horizon_sessions": 3, "risk_budget_annualized": 0.2}
    portfolio = PortfolioState(cash_cny=100.0, holdings_cny={})
    training_state = {"internal-training-state": "receiving-task-only"}
    source_config = {**runtime_config, "feature_trend_sessions": 1}
    snapshot = {
        "schema_version": "public-expert-snapshot.v1",
        "data_mode": "retrospective_source_publication_proxy",
        "requested_decision_time": f"{as_of.isoformat()}T16:00:00+08:00",
        "evaluation_horizon_sessions": 3,
    }
    candidate = seed_candidate()
    required_observations = []
    case = HarnessCase(as_of, snapshot, "reuse_demo", required_observations)
    source_identity = harness_evolution.source_package_identity()
    upstream_identity = {
        "schema_version": "adaptive-mas.gepa-package-identity.v1",
        "source_identity": source_identity,
        "active_version_id": "seed",
        "active_candidate_sha256": harness_evolution._stable_hash(candidate),
        "model": _MockProvider.model,
        "endpoint_sha256": harness_runtime._endpoint_sha256(_MockProvider.endpoint),
        "config_sha256": harness_evolution._stable_hash(source_config),
        "case_rows": [{
            "as_of": as_of.isoformat(),
            "split": "reuse_demo",
            "snapshot_sha256": harness_evolution._stable_hash(snapshot),
            "required_observations_sha256": harness_evolution._stable_hash(required_observations),
            "public_evidence_mode": snapshot["data_mode"],
        }],
    }
    budget = CallBudget(max_calls=20)
    budget.package_identity_manifest = upstream_identity
    budget.package_identity_sha256 = harness_evolution._stable_hash(upstream_identity)
    trace_sink = []
    evaluate_candidate(
        candidate,
        case,
        research=research,
        client=_MockProvider(),
        budget=budget,
        config=source_config,
        candidate_version_id="seed",
        trace_sink=trace_sink,
    )
    assert len(trace_sink) == 1
    serialized_trace = json.loads(json.dumps(trace_sink[0], ensure_ascii=False))
    source_manifest = serialized_trace["package_provenance"]["source_manifest"]
    assert source_manifest["schema_version"] == "adaptive-mas.harness-source-manifest.v4"
    assert "training_state_sha256" not in source_manifest
    assert "training_seed_provenance" not in source_manifest
    assert source_manifest["source_package_identity"]["upstream_experiment"]["identity"] == upstream_identity

    diagnostic = {"prediction_decisions": [{
        "prediction_date": as_of.isoformat(),
        "forecast_errors_by_method": {"loss_gated_fusion": 0.01},
    }]}
    final_calls = []
    monkeypatch.setattr(harness_runtime, "run_mvp_research", lambda *_args, **_kwargs: diagnostic)
    monkeypatch.setattr(
        harness_runtime,
        "create_daily_decision",
        lambda *args, **kwargs: final_calls.append((args, kwargs)) or diagnostic,
    )
    receiving_config = {**runtime_config, "task_combination_note": "receiving task config"}
    runtime_provider = _MockProvider()
    result = harness_runtime.create_harness_daily_decision(
        research,
        receiving_config,
        as_of,
        portfolio,
        training_state,
        public_cache_dir=tmp_path / "public",
        harness_version_dir=tmp_path / "versions",
        run_dir=tmp_path / "run",
        training_seed_trained_through=date(2026, 3, 6),
        training_seed_horizon_sessions=3,
        retrospective_public_cache=True,
        client=runtime_provider,
        precomputed_research_result=serialized_trace,
    )
    assert result is diagnostic
    task_context = final_calls[-1][1]["harness_context"]
    assert task_context["research"]["task_usage"]["provider_call_count"] > 0
    assert task_context["research"]["task_usage"]["api_incremental_cost_cny"] > 0.0
    assert task_context["research"]["cost"]["incremental_api_cost_cny"] > 0.0
    assert task_context["research"]["receiving_task_usage"]["provider_call_count"] == 0
    assert task_context["research"]["receiving_task_usage"]["api_incremental_cost_cny"] == 0.0
    assert task_context["research"]["receiving_task_cost"]["incremental_api_cost_cny"] == 0.0
    assert task_context["research_cost"]["actual_incremental_api_cost_cny"] == 0.0
    assert task_context["runtime_package_identity"]["training_state_sha256"] == harness_runtime._stable_hash(training_state)
    assert runtime_provider.turns == {}

    second_result = harness_runtime.create_harness_daily_decision(
        research,
        receiving_config,
        as_of,
        portfolio,
        training_state,
        public_cache_dir=tmp_path / "public",
        harness_version_dir=tmp_path / "versions-second",
        run_dir=tmp_path / "run-second",
        training_seed_trained_through=date(2026, 3, 6),
        training_seed_horizon_sessions=3,
        retrospective_public_cache=True,
        client=runtime_provider,
        precomputed_research_result=serialized_trace,
    )
    assert second_result is diagnostic
    second_research = final_calls[-1][1]["harness_context"]["research"]
    assert second_research["task_usage"]["api_incremental_cost_cny"] == task_context["research"]["task_usage"]["api_incremental_cost_cny"]
    assert second_research["receiving_task_usage"]["api_incremental_cost_cny"] == 0.0
    assert final_calls[-1][1]["harness_context"]["research_cost"]["actual_incremental_api_cost_cny"] == 0.0
    assert runtime_provider.turns == {}

    drifted_trace = json.loads(json.dumps(serialized_trace, ensure_ascii=False))
    drifted_manifest = drifted_trace["package_provenance"]["source_manifest"]
    drifted_manifest["source_snapshot"]["data_mode"] = "captured_public_cache_cutoff"
    drifted_trace["package_provenance"]["source_manifest_sha256"] = harness_runtime._stable_hash(
        drifted_manifest
    )
    with pytest.raises(ValueError, match="source snapshot hash does not match manifest content"):
        harness_runtime.create_harness_daily_decision(
            research,
            receiving_config,
            as_of,
            portfolio,
            training_state,
            public_cache_dir=tmp_path / "public",
            harness_version_dir=tmp_path / "versions",
            run_dir=tmp_path / "drift",
            training_seed_trained_through=date(2026, 3, 6),
            training_seed_horizon_sessions=3,
            retrospective_public_cache=True,
            client=runtime_provider,
            precomputed_research_result=drifted_trace,
        )


def test_runtime_rejects_dates_on_or_before_seed_cutoff_before_forecasting(tmp_path, monkeypatch) -> None:
    cutoff = date(2026, 3, 6)
    research, config, portfolio, training_state = _runtime_inputs(cutoff)
    forecast_runs = []
    monkeypatch.setattr(
        harness_runtime,
        "run_mvp_research",
        lambda *args, **kwargs: forecast_runs.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="later than the declared training-seed cutoff"):
        harness_runtime.create_harness_daily_decision(
            research,
            config,
            cutoff,
            portfolio,
            training_state,
            public_cache_dir=tmp_path / "public",
            harness_version_dir=tmp_path / "versions",
            run_dir=tmp_path / "run",
            training_seed_trained_through=cutoff,
            training_seed_horizon_sessions=3,
        )

    assert forecast_runs == []


def test_runtime_rejects_seed_horizon_mismatch_before_forecasting(tmp_path, monkeypatch) -> None:
    as_of = date(2026, 9, 23)
    research, config, portfolio, training_state = _runtime_inputs(as_of)
    forecast_runs = []
    monkeypatch.setattr(
        harness_runtime,
        "run_mvp_research",
        lambda *args, **kwargs: forecast_runs.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="must match the declared training-seed horizon"):
        harness_runtime.create_harness_daily_decision(
            research,
            config,
            as_of,
            portfolio,
            training_state,
            public_cache_dir=tmp_path / "public",
            harness_version_dir=tmp_path / "versions",
            run_dir=tmp_path / "run",
            training_seed_trained_through=date(2026, 3, 6),
            training_seed_horizon_sessions=5,
        )

    assert forecast_runs == []


def test_runtime_rejects_precomputed_research_from_a_different_package(tmp_path, monkeypatch):
    as_of = date(2026, 9, 23)
    research, config, portfolio, training_state = _runtime_inputs(as_of)
    candidate = seed_candidate()
    precomputed = _fixture_precomputed(as_of, candidate, config, training_state, research)
    precomputed["package_provenance"]["tool_schema_sha256"] = "0" * 64
    monkeypatch.setattr(
        harness_runtime,
        "run_mvp_research",
        lambda *_args, **_kwargs: {"prediction_decisions": [{
            "prediction_date": as_of.isoformat(),
            "forecast_errors_by_method": {"loss_gated_fusion": 0.01},
        }]},
    )

    with pytest.raises(ValueError, match="package identity does not match"):
        harness_runtime.create_harness_daily_decision(
            research,
            config,
            as_of,
            portfolio,
            training_state,
            public_cache_dir=tmp_path / "public",
            harness_version_dir=tmp_path / "versions",
            run_dir=tmp_path / "run",
            training_seed_trained_through=date(2026, 3, 6),
            training_seed_horizon_sessions=3,
            precomputed_research_result=precomputed,
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("format", "provenance format is unsupported"),
        ("manifest", "source manifest hash does not match its content"),
        ("snapshot", "source snapshot hash does not match manifest content"),
        ("model", "provider identity does not match package provenance"),
        ("configured_model", "does not match the configured model"),
        ("negative_cost", "must be finite and nonnegative"),
        ("nan_cost", "must be finite and nonnegative"),
        ("source_cost_drift", "research artifact content does not match its source manifest"),
    ],
)
def test_runtime_rejects_unverifiable_precomputed_provenance(tmp_path, monkeypatch, drift, message):
    as_of = date(2026, 9, 23)
    research, config, portfolio, training_state = _runtime_inputs(as_of)
    precomputed = _fixture_precomputed(
        as_of, seed_candidate(), config, training_state, research
    )
    provenance = precomputed["package_provenance"]
    configured_client = None
    if drift == "format":
        provenance["schema_version"] = "adaptive-mas.research-package-provenance.v1"
    elif drift == "manifest":
        provenance["source_manifest"]["source_package_identity"]["research_config"]["risk_budget_annualized"] = 0.3
    elif drift == "snapshot":
        provenance["source_manifest"]["source_snapshot"]["requested_decision_time"] = (
            "2026-09-24T16:00:00+08:00"
        )
        provenance["source_manifest_sha256"] = harness_runtime._stable_hash(
            provenance["source_manifest"]
        )
    elif drift == "model":
        provenance["model"] = "different-fixture-model"
    elif drift == "configured_model":
        from adaptive_mas.agents.harness import QwenClient

        configured_client = QwenClient(
            "https://different-fixture.invalid", "fixture-key", "different-fixture-model"
        )
    elif drift == "negative_cost":
        precomputed["task_usage"]["logical_estimated_cost_cny"] = -123.0
    elif drift == "nan_cost":
        precomputed["task_usage"]["api_incremental_cost_cny"] = float("nan")
    elif drift == "source_cost_drift":
        precomputed["task_usage"]["logical_estimated_cost_cny"] = 0.02
        precomputed["cost"]["logical_inference_estimated_cost_cny"] = 0.02

    monkeypatch.setattr(
        harness_runtime,
        "run_mvp_research",
        lambda *_args, **_kwargs: {"prediction_decisions": [{
            "prediction_date": as_of.isoformat(),
            "forecast_errors_by_method": {"loss_gated_fusion": 0.01},
        }]},
    )
    with pytest.raises(ValueError, match=message):
        harness_runtime.create_harness_daily_decision(
            research,
            config,
            as_of,
            portfolio,
            training_state,
            public_cache_dir=tmp_path / "public",
            harness_version_dir=tmp_path / "versions",
            run_dir=tmp_path / "run",
            training_seed_trained_through=date(2026, 3, 6),
            training_seed_horizon_sessions=3,
            client=configured_client,
            precomputed_research_result=precomputed,
        )


def test_runtime_separates_source_experiment_config_from_task_combination_config(
    tmp_path, monkeypatch
):
    as_of = date(2026, 9, 23)
    research, config, portfolio, training_state = _runtime_inputs(as_of)
    precomputed = _fixture_precomputed(
        as_of, seed_candidate(), config, training_state, research
    )
    task_config = {**config, "receiving_task_note": "different task combination"}
    diagnostic = {"prediction_decisions": [{
        "prediction_date": as_of.isoformat(),
        "forecast_errors_by_method": {"loss_gated_fusion": 0.01},
    }]}
    monkeypatch.setattr(harness_runtime, "run_mvp_research", lambda *_args, **_kwargs: diagnostic)
    monkeypatch.setattr(
        harness_runtime,
        "create_daily_decision",
        lambda *_args, **_kwargs: diagnostic,
    )

    result = harness_runtime.create_harness_daily_decision(
        research,
        task_config,
        as_of,
        portfolio,
        training_state,
        public_cache_dir=tmp_path / "public",
        harness_version_dir=tmp_path / "versions",
        run_dir=tmp_path / "run",
        training_seed_trained_through=date(2026, 3, 6),
        training_seed_horizon_sessions=3,
        precomputed_research_result=precomputed,
    )

    assert result is diagnostic


def test_runtime_persists_completed_decision_and_reuses_same_task_idempotently(tmp_path, monkeypatch):
    as_of = date(2026, 9, 23)
    research, config, portfolio, training_state = _runtime_inputs(as_of)
    candidate = seed_candidate()
    diagnostic = {
        "prediction_decisions": [{
            "prediction_date": as_of.isoformat(),
            "forecast_errors_by_method": {"loss_gated_fusion": 0.01},
        }],
    }
    precomputed = _fixture_precomputed(as_of, candidate, config, training_state, research)
    decision_calls = []

    def fake_forecast(*_args, **_kwargs):
        return diagnostic

    def fake_decision(*_args, **_kwargs):
        decision_calls.append(True)
        return DailyDecision(
            schema_version="adaptive-mas.decision.v1",
            as_of_date=as_of.isoformat(), execution_date=None,
            status="hold", prediction_status="pending", prediction_horizon_sessions=3,
            predictions_by_symbol={}, current_nav_cny=100.0, current_cash_cny=100.0,
            current_holdings_cny={}, target_cash_cny=100.0, target_holdings_cny={},
            target_weights={}, estimated_cash_after_costs_cny=100.0,
            buy_notional_cny=0.0, sell_notional_cny=0.0, orders=(),
            expected_gross_return=0.0, forecast_error_allowance=0.0,
            expected_transaction_cost_cny=0.0,
            current_holding_objective_value_cny=0.0,
            optimized_objective_value_cny=0.0,
            risk_adjusted_net_incremental_value_cny=0.0, risk_penalty_cny=0.0,
            annualized_risk=0.0, annualized_risk_budget=0.2,
            one_way_turnover=0.0,
            evidence={"investment_manager_feedback": {"risk_review_status": "passed"}},
            versions={},
        )

    monkeypatch.setattr(harness_runtime, "run_mvp_research", fake_forecast)
    monkeypatch.setattr(harness_runtime, "create_daily_decision", fake_decision)
    kwargs = {
        "public_cache_dir": tmp_path / "public",
        "harness_version_dir": tmp_path / "versions",
        "run_dir": tmp_path / "run",
        "training_seed_trained_through": date(2026, 3, 6),
        "training_seed_horizon_sessions": 3,
        "precomputed_research_result": precomputed,
        "state_store_path": tmp_path / "harness-state.sqlite3",
        "task_id": "stable-task-id",
        "idempotency_key": "stable-task-id",
    }
    first = harness_runtime.create_harness_daily_decision(
        research, config, as_of, portfolio, training_state, **kwargs,
    )
    resumed = harness_runtime.create_harness_daily_decision(
        research, config, as_of, portfolio, training_state,
        **{**kwargs, "task_id": "retry-completed-generated-id"},
    )

    assert first == resumed
    assert len(decision_calls) == 1
    from adaptive_mas.harness_state import HarnessStateStore

    task = HarnessStateStore(kwargs["state_store_path"]).get_task("stable-task-id")
    assert task["status"] == "completed"
    assert task["stage"] == "completed"
    assert HarnessStateStore(kwargs["state_store_path"]).get_task("retry-completed-generated-id") is None


def test_warmup_runtime_uses_canonical_task_id_when_idempotency_key_is_reused(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from adaptive_mas.harness_state import HarnessStateStore

    as_of = date(2026, 9, 23)
    research, config, portfolio, training_state = _runtime_inputs(as_of)
    diagnostic = {
        "prediction_decisions": [{
            "prediction_date": as_of.isoformat(),
            "status": "warmup",
            "forecast_errors_by_method": {"loss_gated_fusion": None},
        }],
    }
    monkeypatch.setattr(harness_runtime, "run_mvp_research", lambda *_args, **_kwargs: diagnostic)
    monkeypatch.setattr(
        harness_runtime,
        "create_daily_decision",
        lambda *_args, **_kwargs: SimpleNamespace(evidence={}),
    )
    monkeypatch.setattr(
        harness_runtime,
        "replace",
        lambda _baseline, *, evidence: SimpleNamespace(
            evidence=evidence,
            to_dict=lambda: {"as_of_date": as_of.isoformat(), "status": "warmup", "evidence": evidence},
        ),
    )
    warmup_identity = {
        "as_of": as_of.isoformat(),
        "portfolio": {
            "cash_cny": portfolio.cash_cny,
            "holdings_cny": dict(sorted(portfolio.holdings_cny.items())),
        },
        "config_sha256": harness_runtime._stable_hash(config),
        "training_state_sha256": harness_runtime._stable_hash(training_state),
        "versions": {},
        "forecast_status": "warmup",
        "warmup": True,
    }
    store = HarnessStateStore(tmp_path / "warmup-state.sqlite3")
    store.open_task(
        "canonical-warmup-task",
        "warmup-idempotency-key",
        warmup_identity,
        max_provider_calls=0,
    )

    result = harness_runtime.create_harness_daily_decision(
        research,
        config,
        as_of,
        portfolio,
        training_state,
        public_cache_dir=tmp_path / "public",
        harness_version_dir=tmp_path / "versions",
        run_dir=tmp_path / "run",
        training_seed_trained_through=date(2026, 3, 6),
        training_seed_horizon_sessions=3,
        state_store_path=tmp_path / "warmup-state.sqlite3",
        task_id="retry-warmup-generated-id",
        idempotency_key="warmup-idempotency-key",
    )

    assert result.evidence["harness"]["status"] == "skipped_forecast_warmup"
    assert store.get_task("canonical-warmup-task")["status"] == "completed"
    assert store.get_task("retry-warmup-generated-id") is None


def test_runtime_recovery_preserves_same_task_provider_cost_and_records_input_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from adaptive_mas.agents.harness import QwenClient
    from adaptive_mas.data.research import MembershipSnapshot
    from adaptive_mas.harness_state import HarnessStateStore

    as_of = date(2026, 9, 23)
    research, config, portfolio, training_state = _runtime_inputs(as_of)
    research.membership_snapshots = (
        MembershipSnapshot(as_of, as_of, frozenset(), "fixture-membership"),
    )
    diagnostic = {
        "prediction_decisions": [{
            "prediction_date": as_of.isoformat(),
            "forecast_errors_by_method": {"loss_gated_fusion": 0.01},
        }],
    }
    monkeypatch.setattr(harness_runtime, "run_mvp_research", lambda *_args, **_kwargs: diagnostic)
    monkeypatch.setattr(
        harness_runtime,
        "build_public_expert_snapshot",
        lambda *_args, **_kwargs: {
            "schema_version": "public-expert-snapshot.v1",
            "data_mode": "captured_public_cache_cutoff",
            "requested_decision_time": f"{as_of.isoformat()}T16:00:00+08:00",
            "evaluation_horizon_sessions": 3,
        },
    )

    research_runs = []

    def fake_research_harness(_research, _as_of, _snapshot, candidate, _client, budget, _config, **_kwargs):
        research_runs.append(True)
        request_hash = "fixture-provider-request"
        usage = {"prompt_tokens": 100, "completion_tokens": 20}
        budget.calls.append({
            "task_id": budget.task_id,
            "request_sha256": request_hash,
            "status": "completed",
            "actual_estimated_cost_cny": 0.002,
            "source": "provider",
        })
        budget.logical_calls.append({
            "task_id": budget.task_id,
            "request_sha256": request_hash,
            "role": "market",
            "source": "provider",
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "logical_estimated_cost_cny": 0.002,
        })
        return {
            "status": "completed",
            "as_of": as_of.isoformat(),
            "public_evidence_mode": "captured_public_cache_cutoff",
            "memory": {
                "procedural_experience_version": "seed",
                "procedural_experience_sha256": hashlib.sha256(
                    candidate["procedural_experience"].encode("utf-8")
                ).hexdigest(),
            },
        }

    monkeypatch.setattr(harness_runtime, "run_research_harness", fake_research_harness)
    decision_calls = []

    def flaky_decision(*_args, **kwargs):
        decision_calls.append(kwargs)
        if len(decision_calls) == 1:
            raise RuntimeError("fixture interruption after research artifact persistence")
        return SimpleNamespace(
            to_dict=lambda: {"status": "hold"},
            evidence={"investment_manager_feedback": {"status": "recorded"}},
        )

    monkeypatch.setattr(harness_runtime, "create_daily_decision", flaky_decision)
    common = {
        "public_cache_dir": tmp_path / "public",
        "harness_version_dir": tmp_path / "versions",
        "run_dir": tmp_path / "run",
        "training_seed_trained_through": date(2026, 3, 6),
        "training_seed_horizon_sessions": 3,
        "client": QwenClient("https://fixture.invalid/api", "fixture-key", "fixture-model"),
        "state_store_path": tmp_path / "state.sqlite3",
        "task_id": "recoverable-task",
        "idempotency_key": "recoverable-task",
    }

    with pytest.raises(RuntimeError, match="fixture interruption"):
        harness_runtime.create_harness_daily_decision(
            research, config, as_of, portfolio, training_state, **common,
        )
    resumed = harness_runtime.create_harness_daily_decision(
        research, config, as_of, portfolio, training_state,
        **{**common, "task_id": "retry-new-generated-id"},
    )

    assert resumed.to_dict() == {"status": "hold"}
    assert len(research_runs) == 1
    recovered_context = decision_calls[1]["harness_context"]
    assert recovered_context["research"]["task_usage"]["provider_call_count"] == 1
    assert recovered_context["research"]["task_usage"]["api_incremental_cost_cny"] == 0.002
    assert recovered_context["research_cost"]["actual_incremental_api_cost_cny"] == 0.002
    assert recovered_context["research_cost"]["cost_source"] == "original provider responses recovered from this task"

    task = HarnessStateStore(common["state_store_path"]).get_task("recoverable-task")
    identity = task["request"]
    assert identity["provider_identity"]["model"] == "fixture-model"
    assert identity["provider_identity"]["endpoint_sha256"]
    assert identity["market_snapshot"]["sha256"]
    assert identity["implementation_identity"]["tool_schema_sha256"]
    assert identity["implementation_identity"]["implementation_package_sha256"]
    assert identity["training_seed_provenance"] == {
        "trained_through": "2026-03-06",
        "cutoff_source": "caller_supplied_provenance",
        "horizon_sessions": 3,
        "horizon_source": "caller_supplied_provenance",
    }
    assert HarnessStateStore(common["state_store_path"]).get_task("retry-new-generated-id") is None
