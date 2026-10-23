import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from adaptive_mas.agents import qwen_expert
from adaptive_mas.agents import public_experts
from adaptive_mas.evaluation.expert_memory import (
    build_evidence_challenges,
    candidate_performance_effects,
    update_matured_performance_memory,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _at(value):
    return datetime.fromisoformat(value).replace(tzinfo=SHANGHAI)


def _forecast(record_id="rule:1", direction=1):
    return {
        "record_id": record_id,
        "expert": "fundamental_rule_v1",
        "symbol": "sh.600000",
        "event_type": "quarterly_fundamental_growth",
        "prediction_date": "2026-09-27",
        "event_time": "2026-09-26T00:00:00+08:00",
        "available_at": "2026-09-26T12:00:00+08:00",
        "horizon_sessions": 3,
        "evidence_text_scope": "structured_financial_fields",
        "direction_score": direction,
        "rule": "both_growth_rates_nonnegative",
        "evidence": [{"source_record_id": "src-1", "dataset": "financial_indicator"}],
    }


def test_qwen_client_skips_without_credential(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("missing credential must not make a request")

    monkeypatch.delenv("POPAGENT_LLM_KEY", raising=False)
    monkeypatch.setattr(qwen_expert, "urlopen", forbidden)

    result = qwen_expert.call_qwen_expert(
        [_forecast()], _at("2026-09-27T18:00:00"), 3
    )

    assert result["status"] == "not_run_missing_POPAGENT_LLM_KEY"
    assert result["outputs"] == []
    assert result["endpoint"] is None


def test_qwen_requires_endpoint_only_when_configured_to_make_a_call(monkeypatch):
    monkeypatch.setenv("POPAGENT_LLM_KEY", "in-memory-test-key")
    monkeypatch.delenv("POPAGENT_LLM_URL", raising=False)

    with pytest.raises(ValueError, match="必须设置 POPAGENT_LLM_URL"):
        qwen_expert.call_qwen_expert(
            [_forecast()], _at("2026-09-27T18:00:00"), 3
        )


def test_qwen_client_validates_citations_and_accounts_tokens(monkeypatch):
    seen_request = {}

    def response_for(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        seen_request.update(model=payload["model"], endpoint=request.full_url)
        source = json.loads(payload["messages"][1]["content"])["records"][0]
        body = {
            "choices": [{"message": {"content": json.dumps({"analyses": [{
                "record_id": source["record_id"],
                "event_type": "quarterly_results",
                "direction_score": 1,
                "event_time_scale": "slow",
                "rationale": "growth facts agree",
                "cited_source_record_ids": ["src-1"],
            }]})}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
        }
        return _Response(body)

    monkeypatch.setattr(qwen_expert, "urlopen", response_for)
    monkeypatch.setenv("POPAGENT_LLM_URL", "https://example.test/v1/chat/completions")
    monkeypatch.setenv("POPAGENT_LLM_MODEL", "qwen-test-model")
    result = qwen_expert.call_qwen_expert(
        [_forecast()], _at("2026-09-27T18:00:00"), 5, api_key="in-memory-test-key"
    )

    assert result["status"] == "completed"
    assert result["outputs"][0]["horizon_sessions"] == 5
    assert result["outputs"][0]["horizon_kind"] == "comparison_label_window"
    assert result["outputs"][0]["evidence"][0]["source_record_id"] == "src-1"
    assert result["usage"]["total_tokens"] == 1200
    assert result["estimated_token_cost_cny"] == pytest.approx(0.00288)
    assert result["model"] == "qwen-test-model"
    assert result["endpoint"] == "https://example.test/v1/chat/completions"
    assert result["outputs"][0]["model"] == "qwen-test-model"
    assert seen_request == {
        "model": "qwen-test-model",
        "endpoint": "https://example.test/v1/chat/completions",
    }
    assert result["completed_at"] == result["outputs"][0]["available_at"]


def test_qwen_rejects_unprovided_source_reference(monkeypatch):
    def response_for(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        source = json.loads(payload["messages"][1]["content"])["records"][0]
        body = {
            "choices": [{"message": {"content": json.dumps({"analyses": [{
                "record_id": source["record_id"],
                "event_type": "quarterly_results",
                "direction_score": -1,
                "event_time_scale": "short",
                "rationale": "invented evidence",
                "cited_source_record_ids": ["not-in-input"],
            }]})}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
        }
        return _Response(body)

    monkeypatch.setattr(qwen_expert, "urlopen", response_for)
    monkeypatch.setenv("POPAGENT_LLM_URL", "https://example.test/v1/chat/completions")
    with pytest.raises(ValueError, match="未随该分析提供"):
        qwen_expert.call_qwen_expert(
            [_forecast()], _at("2026-09-27T18:00:00"), 3, api_key="in-memory-test-key"
        )


def test_challenge_requires_conflicting_direction_and_cites_both_sources():
    positive = _forecast("positive", 1)
    negative = {**_forecast("negative", -1), "expert": "announcement_title_rule_v1"}

    challenges = build_evidence_challenges([positive, negative])

    assert len(challenges) == 1
    assert challenges[0]["status"] == "evidence_conflict_unresolved"
    assert challenges[0]["original_forecasts_changed"] is False
    assert {
        ref["evidence"][0]["source_record_id"]
        for ref in challenges[0]["evidence_refs"]
    } == {"src-1"}
    assert build_evidence_challenges([positive, {**positive, "record_id": "same-sign"}]) == []


def test_same_announcement_full_text_resolves_only_evidence_scope_precedence():
    full_text = {
        **_forecast("full-text", 1),
        "expert": "cninfo_full_text_llm_v1",
        "evidence_text_scope": "full_announcement_pdf_text",
        "evidence": [{"source_record_id": "src-1", "dataset": "cninfo_pdf"}],
    }
    headline = {
        **_forecast("headline", -1),
        "expert": "announcement_title_rule_v1",
        "evidence_text_scope": "headline_only_no_announcement_body",
    }

    challenge = build_evidence_challenges([full_text, headline])[0]

    assert challenge["status"] == "evidence_scope_precedence_resolved"
    assert challenge["direction_correctness"] == "not_established"
    assert challenge["judge_verdict"]["winner_record_id"] == "full-text"
    assert challenge["original_forecasts_changed"] is False


def test_performance_memory_excludes_unmatured_and_wrong_horizon_rows():
    outcomes = [
        {
            "record_id": "rule:1",
            "label_available_at": "2026-09-27T16:00:00+08:00",
            "horizon_sessions": 3,
            "forward_return": 0.02,
            "outcome_ref": "bars:20260927",
        },
        {
            "record_id": "rule:future",
            "label_available_at": "2026-09-29T16:00:00+08:00",
            "horizon_sessions": 3,
            "forward_return": 0.04,
            "outcome_ref": "bars:20260929",
        },
    ]

    memory = update_matured_performance_memory(
        [_forecast(), _forecast("rule:future")],
        outcomes,
        _at("2026-09-28T18:00:00"),
    )

    assert len(memory) == 1
    assert memory[0]["directional_return"] == pytest.approx(0.02)
    assert memory[0]["direction_hit"] is True
    with pytest.raises(ValueError, match="期限与预测期限不一致"):
        update_matured_performance_memory(
            [_forecast()], [{**outcomes[0], "horizon_sessions": 5}], _at("2026-09-28T18:00:00")
        )


def test_public_snapshot_filters_every_equity_dataset_to_allowed_symbols(monkeypatch, tmp_path):
    calls = []

    def read_records(cache_dir, dataset, decision_time, symbols=None, company_names=None):
        calls.append((dataset, symbols, company_names))
        return []

    monkeypatch.setattr(public_experts, "load_public_records", read_records)
    monkeypatch.delenv("POPAGENT_LLM_KEY", raising=False)
    snapshot = public_experts.build_public_expert_snapshot(
        tmp_path,
        _at("2026-09-27T22:00:00"),
        3,
        allowed_symbols={"600000", "000001"},
        announcement_text_limit=0,
    )

    expected_codes = {"sh.600000", "sz.000001"}
    equity_calls = [row for row in calls if row[0] not in {"news", "macro_cpi", "macro_pmi"}]
    assert equity_calls
    assert all(row[1] == expected_codes for row in equity_calls)
    assert snapshot["allowed_symbols"] == ["000001", "600000"]
    assert snapshot["source_coverage"]["captured_records"]["financial_indicator"] == 0


def test_public_snapshot_does_not_treat_attention_rows_as_financial_statements(monkeypatch, tmp_path):
    attention_row = {
        "source_record_id": "attention:600000:20260925",
        "symbol": "sh.600000",
        "content_sha256": "a" * 64,
        "raw_response_ref": "cache/attention.json",
        "published_at": "2026-09-25T18:00:00+08:00",
        "available_at": "2026-09-25T18:00:00+08:00",
        "fetched_at": "2026-09-25T18:01:00+08:00",
        "vintage_verified": True,
        "vintage_status": "captured",
        "交易日": "2026-09-25",
        "用户关注指数": 12.5,
    }

    def read_records(_cache_dir, dataset, _decision_time, symbols=None, company_names=None):
        return [attention_row] if dataset == "attention_index" else []

    monkeypatch.setattr(public_experts, "load_public_records", read_records)
    monkeypatch.delenv("POPAGENT_LLM_KEY", raising=False)

    snapshot = public_experts.build_public_expert_snapshot(
        tmp_path,
        _at("2026-09-27T22:00:00"),
        3,
        announcement_text_limit=0,
        include_qwen=False,
    )

    assert len(snapshot["attention_outputs"]) == 1
    assert snapshot["attention_outputs"][0]["symbol"] == "600000"
    assert snapshot["attention_outputs"][0]["attention_index"] == 12.5


def test_retrospective_proxy_retains_late_capture_audit_and_strict_mode_excludes(monkeypatch, tmp_path):
    historical_indicator = {
        "source_record_id": "indicator:600000:2024-12-31",
        "symbol": "sh.600000",
        "content_sha256": "b" * 64,
        "raw_response_ref": "cache/financial_indicator.json",
        "published_at": "2025-03-31T16:00:00+08:00",
        "available_at": "2025-03-31T16:00:00+08:00",
        "fetched_at": "2026-09-27T09:10:00+08:00",
        "vintage_verified": True,
        "vintage_status": "historical_backfill",
        "REPORT_DATE": "2024-12-31",
        "TOTALOPERATEREVETZ": 4.2,
        "PARENTNETPROFITTZ": 5.3,
        "ROEJQ": 11.1,
        "SECURITY_NAME_ABBR": "示例公司",
    }

    def read_records(_cache_dir, dataset, _decision_time, symbols=None, company_names=None):
        return [historical_indicator] if dataset == "financial_indicator" else []

    monkeypatch.setattr(public_experts, "load_public_records", read_records)
    monkeypatch.delenv("POPAGENT_LLM_KEY", raising=False)
    decision_time = _at("2026-04-20T16:00:00")

    strict_snapshot = public_experts.build_public_expert_snapshot(
        tmp_path, decision_time, 3, announcement_text_limit=0, include_qwen=False,
    )
    proxy_snapshot = public_experts.build_public_expert_snapshot(
        tmp_path, decision_time, 3, announcement_text_limit=0, include_qwen=False,
        retrospective_backfill_proxy=True,
    )

    assert strict_snapshot["p9_outputs"] == []
    proxy_source = proxy_snapshot["p9_outputs"][0]["evidence"][0]
    assert proxy_snapshot["data_mode"] == "retrospective_source_publication_proxy"
    assert proxy_snapshot["strict_historical_pit_verified"] is False
    assert proxy_source["published_at"] == historical_indicator["published_at"]
    assert proxy_source["available_at"] == historical_indicator["available_at"]
    assert proxy_source["fetched_at"] == historical_indicator["fetched_at"]
    assert proxy_source["retrospective_backfill_proxy"] is True
    assert proxy_source["vintage_verified"] is False


def test_nontrading_observation_does_not_advance_event_consumption_watermark():
    from datetime import date
    from pathlib import Path
    from scripts.run_p9_p12_research import _last_trading_event_start

    weekend = {
        "decision_time": "2026-09-27T22:00:00+08:00",
        "event_start_time": "2026-09-24T22:00:00+08:00",
    }
    monday = datetime.fromisoformat("2026-09-28T18:00:00+08:00")
    trading_dates = (date(2026, 9, 23), date(2026, 9, 28))

    event_start = _last_trading_event_start(
        [(Path("weekend"), weekend)], trading_dates, monday
    )

    assert event_start == datetime.fromisoformat("2026-09-24T22:00:00+08:00")
    assert event_start < datetime.fromisoformat(weekend["decision_time"])


def test_first_weekday_snapshot_starts_at_prior_trading_close():
    from datetime import date
    from pathlib import Path
    from scripts.run_p9_p12_research import _last_trading_event_start

    monday = datetime.fromisoformat("2026-09-28T18:00:00+08:00")
    event_start = _last_trading_event_start(
        [], (date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28)), monday
    )

    assert event_start == datetime.fromisoformat("2026-09-25T15:00:00+08:00")


def test_weekend_qwen_signal_carries_to_monday_once():
    from datetime import date
    from pathlib import Path
    from scripts.run_p9_p12_research import (
        _carry_forward_llm_signals,
        _last_trading_event_start,
    )

    friday_close = datetime.fromisoformat("2026-09-25T15:00:00+08:00")
    weekend = {
        "decision_time": "2026-09-27T22:24:00+08:00",
        "event_start_time": friday_close.isoformat(),
        "llm_outputs": {"outputs": [{
            "record_id": "qwen:announcement-1",
            "forecast_of_record_id": "announcement-1",
            "available_at": "2026-09-27T22:23:00+08:00",
            "symbol": "sh.600000",
            "direction_score": 1,
            "expert": "qwen3.7_plus_fundamental_announcement",
            "horizon_sessions": 3,
        }]},
    }
    monday = datetime.fromisoformat("2026-09-28T18:00:00+08:00")
    first_carry = _carry_forward_llm_signals(
        [(Path("weekend"), weekend)], {"600000"}, friday_close, monday
    )
    assert len(first_carry) == 1
    assert first_carry[0]["record_id"] == "qwen:announcement-1:carry:2026-09-28"

    monday_snapshot = {
        "decision_time": monday.isoformat(),
        "event_start_time": friday_close.isoformat(),
        "llm_outputs": {"outputs": []},
        "carry_forward_expert_signals": first_carry,
    }
    tuesday = datetime.fromisoformat("2026-09-29T18:00:00+08:00")
    next_watermark = _last_trading_event_start(
        [(Path("weekend"), weekend), (Path("monday"), monday_snapshot)],
        (date(2026, 9, 25), date(2026, 9, 28), date(2026, 9, 29)),
        tuesday,
    )
    assert next_watermark == monday
    assert _carry_forward_llm_signals(
        [(Path("weekend"), weekend), (Path("monday"), monday_snapshot)],
        {"600000"}, next_watermark, tuesday,
    ) == []


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")

def test_candidate_performance_effects_reuses_twenty_nonoverlapping_date_gate():
    trading_dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(60))
    memory = [
        {
            "expert": "fundamental_rule_v1",
            "horizon_sessions": 3,
            "prediction_date": day.isoformat(),
            "directional_return": 0.01,
        }
        for day in trading_dates
    ]

    eligible = candidate_performance_effects(memory, trading_dates)["fundamental_rule_v1:3s"]
    pending = candidate_performance_effects(memory[:57], trading_dates)["fundamental_rule_v1:3s"]

    assert eligible["matured_nonoverlapping_decision_dates"] == 20
    assert eligible["status"] == "candidate_effect_eligible"
    assert eligible["normal_95pct_lower_bound"] == pytest.approx(0.01)
    assert pending["matured_nonoverlapping_decision_dates"] == 19
    assert pending["candidate_return_adjustment"] == 0.0
