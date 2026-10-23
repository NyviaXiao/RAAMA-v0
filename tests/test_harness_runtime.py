from __future__ import annotations

import hashlib
from datetime import date

import pytest

from adaptive_mas.agents.harness_prompts import seed_candidate
from adaptive_mas.data.research import ResearchData
from adaptive_mas.decision import PortfolioState
from adaptive_mas import harness_runtime


def _runtime_inputs(as_of: date):
    return (
        ResearchData([date(2026, 3, 6), as_of], [], [], [], []),
        {"prediction_horizon_sessions": 3, "risk_budget_annualized": 0.2},
        PortfolioState(cash_cny=100.0, holdings_cny={}),
        {},
    )


def test_runtime_reuses_single_preflight_forecast_for_final_decision(tmp_path, monkeypatch) -> None:
    as_of = date(2026, 9, 23)
    research, config, portfolio, training_state = _runtime_inputs(as_of)
    candidate = seed_candidate()
    experience_hash = hashlib.sha256(candidate["procedural_experience"].encode("utf-8")).hexdigest()
    diagnostic = {
        "prediction_decisions": [{
            "prediction_date": as_of.isoformat(),
            "forecast_errors_by_method": {"loss_gated_fusion": 0.01},
        }],
    }
    precomputed = {
        "status": "completed",
        "as_of": as_of.isoformat(),
        "public_evidence_mode": "captured_public_cache_cutoff",
        "memory": {
            "procedural_experience_version": "seed",
            "procedural_experience_sha256": experience_hash,
        },
        "task_usage": {
            "logical_estimated_cost_cny": 0.01,
            "api_incremental_cost_cny": 0.01,
            "model_turn_count": 2,
        },
    }
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
