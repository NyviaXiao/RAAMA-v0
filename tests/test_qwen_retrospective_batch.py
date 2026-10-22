import hashlib
import json
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.qwen_retrospective_batch as batch
from scripts.qwen_retrospective_batch import (
    CHAT_TOKEN_OVERHEAD_RESERVE,
    MAX_INPUT_TOKENS,
    _build_payload,
    _validate_response,
)


def _row():
    return {
        "data_split": "training",
        "prediction_date": "2025-01-10",
        "symbol": "600000",
        "publication_date": "2025-01-09",
        "announcement_title": "公司公告标题",
        "announcement_source_record_id_citation": "https://example.invalid/a/1",
        "announcement_url": "https://example.invalid/a/1",
        "announcement_sha256": "a" * 64,
        "financial_indicator_source_record_id": "600000:indicator:20240930",
        "financial_indicator_sha256": "b" * 64,
        "financial_indicator_report_period": "2024-09-30",
        "financial_indicator_published_at": "2024-10-30T00:00:00+08:00",
        "revenue_yoy_pct": "8.2",
        "parent_net_profit_yoy_pct": "-3.1",
    }


def test_payload_contains_one_allowlisted_event_and_stays_under_input_cap():
    payload = _build_payload(_row())
    user = json.loads(payload["messages"][1]["content"])
    event = user["event"]
    assert event["symbol"] == "600000"
    assert event["financial"]["revenue_yoy_pct"] == 8.2
    assert event["financial"]["parent_net_profit_yoy_pct"] == -3.1
    assert len(event["sources"]) == 2
    assert "decision_date" not in event and "publication_date" not in event
    assert "url" not in payload["messages"][1]["content"]
    assert "p9_expected_return" not in payload["messages"][1]["content"]
    assert "data_split" not in payload["messages"][1]["content"]
    assert payload["max_tokens"] == 256
    assert payload["model"] == "qwen3.7-max-2026-06-08"
    assert payload["enable_thinking"] is False
    message_bytes = len(json.dumps(payload["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    assert message_bytes + CHAT_TOKEN_OVERHEAD_RESERVE <= MAX_INPUT_TOKENS


def test_response_requires_input_citations_and_fixed_direction_schema():
    payload = _build_payload(_row())
    request = json.loads(payload["messages"][1]["content"])
    allowed = {source["citation_ref"] for source in request["event"]["sources"]}
    valid = {
        "event_id": request["event"]["id"],
        "direction_score": 1,
        "evidence_sufficiency": "limited",
        "rationale": "仅据输入证据判断。",
        "cited_source_record_ids": [next(iter(allowed))],
    }
    assert _validate_response(json.dumps(valid, ensure_ascii=False), request["event"]["id"], allowed)["direction_score"] == 1
    valid["cited_source_record_ids"] = ["unprovided-source"]
    with pytest.raises(ValueError, match="not present"):
        _validate_response(json.dumps(valid), request["event"]["id"], allowed)


def _preview_record(event_id):
    citation = f"citation-{event_id}"
    user = {"event": {"id": event_id, "sources": [{"citation_ref": citation}]}}
    payload = {
        "model": batch.MODEL,
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": json.dumps(user)},
        ],
        "max_tokens": batch.MAX_OUTPUT_TOKENS,
    }
    request_sha = hashlib.sha256(event_id.encode()).hexdigest()
    return {
        "data_split": "training",
        "event_id": event_id,
        "request_sha256": request_sha,
        "payload": payload,
    }


def test_send_skips_unknown_delivery_counts_its_reserve_and_uses_direct_opener(tmp_path, monkeypatch):
    records = [
        _preview_record("unknown"),
        _preview_record("invalid"),
        _preview_record("new-1"),
        _preview_record("new-2"),
    ]
    outbound = tmp_path / "preview.jsonl"
    summary_path = tmp_path / "summary.json"
    preview_bytes = b"frozen test preview"
    outbound.write_bytes(preview_bytes)
    summary_path.write_text(json.dumps({"payload_manifest_sha256": hashlib.sha256(preview_bytes).hexdigest()}))
    response_dir = tmp_path / "responses"
    response_dir.mkdir()
    per_call_cap = (
        batch.MAX_INPUT_TOKENS * batch.INPUT_CNY_PER_MILLION_TOKENS
        + batch.MAX_OUTPUT_TOKENS * batch.OUTPUT_CNY_PER_MILLION_TOKENS
    ) / 1_000_000
    unknown, invalid = records[:2]
    for record, status in ((unknown, "delivery_unknown"), (invalid, "completed_invalid_response")):
        (response_dir / f"{record['event_id']}.json").write_text(json.dumps({
            "event_id": record["event_id"],
            "data_split": "training",
            "status": status,
            "request_sha256": record["request_sha256"],
            "attempt_count": 1,
            "reserved_cost_cny": per_call_cap,
            "actual_estimated_cost_cny": 0.01266 if status == "completed_invalid_response" else None,
            "usage": {"prompt_tokens": 554, "completion_tokens": 167} if status == "completed_invalid_response" else None,
        }))
    key_path = tmp_path / "data" / "raw" / "qwen_key.txt"
    key_path.parent.mkdir(parents=True)
    key_path.write_text("test-only-key")
    monkeypatch.setattr(batch, "PAYLOADS_PATH", outbound)
    monkeypatch.setattr(batch, "PREVIEW_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(batch, "RESPONSE_DIR", response_dir)
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "EXPECTED_SPLIT_COUNTS", {"training": 4, "evaluation": 0})
    monkeypatch.setattr(batch, "_load_preview", lambda: records)
    monkeypatch.setenv("POPAGENT_LLM_URL", "https://example.invalid/chat/completions")
    monkeypatch.setenv("POPAGENT_LLM_MODEL", batch.MODEL)

    opener_instances = []

    class FakeReason:
        errno = 10048
        winerror = 10048

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.body

    class FakeOpener:
        def __init__(self):
            self.event_ids = []

        def open(self, request, timeout):
            event_id = request.get_header("Idempotency-key")
            self.event_ids.append(event_id)
            if event_id == "new-2":
                raise urllib.error.URLError(FakeReason())
            citation = f"citation-{event_id}"
            response = {
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                "choices": [{"message": {"content": json.dumps({
                    "event_id": event_id,
                    "direction_score": 1,
                    "evidence_sufficiency": "limited",
                    "rationale": "test",
                    "cited_source_record_ids": [citation],
                })}}],
            }
            return FakeResponse(json.dumps(response).encode())

    def build_direct_opener(*handlers):
        assert any(
            isinstance(handler, urllib.request.ProxyHandler) and handler.proxies == {}
            for handler in handlers
        )
        opener = FakeOpener()
        opener_instances.append(opener)
        return opener

    monkeypatch.setattr(batch.urllib.request, "build_opener", build_direct_opener)
    with pytest.raises(ValueError, match="Worst-case reserved spend"):
        batch.send("training", 2, per_call_cap * 3, hashlib.sha256(preview_bytes).hexdigest())
    assert opener_instances == []

    with pytest.raises(RuntimeError, match="delivery failed or is unknown"):
        batch.send("training", 2, per_call_cap * 4, hashlib.sha256(preview_bytes).hexdigest())

    assert opener_instances[0].event_ids == ["new-1", "new-2"]
    states = batch._load_states()
    assert states["unknown"]["attempt_count"] == 1
    assert states["unknown"]["status"] == "delivery_unknown"
    assert states["invalid"]["attempt_count"] == 1
    assert states["invalid"]["status"] == "completed_invalid_response"
    assert states["invalid"]["reserved_cost_cny"] == per_call_cap
    assert states["invalid"]["actual_estimated_cost_cny"] == 0.01266
    assert states["new-1"]["status"] == "completed"
    failed = states["new-2"]
    assert failed["status"] == "delivery_unknown"
    assert failed["delivery_error_type"] == "URLError"
    assert failed["delivery_error_reason_type"] == "FakeReason"
    assert failed["delivery_error_errno"] == 10048
    assert failed["delivery_error_winerror"] == 10048
    assert not {"url", "key", "body"} & failed.keys()


def test_training_analysis_uses_completed_subset_and_reports_missing_event(tmp_path, monkeypatch):
    records = []
    states = {}
    for index in range(81):
        event_id = f"event-{index}"
        records.append({
            "event_id": event_id,
            "data_split": "training",
            "payload": {"messages": [{}, {"content": json.dumps({"event": {"id": event_id}})}]},
            "local_event_metadata": {"prediction_date": "2025-01-10", "symbol": "600000"},
        })
        if index < 79:
            states[event_id] = {"status": "completed", "analysis": {"direction_score": 1}}
        elif index == 79:
            states[event_id] = {"status": "completed_invalid_response"}
        else:
            states[event_id] = {"status": "delivery_unknown"}

    training_days = [date(2025, 1, day) for day in (10, 13, 14, 15)]
    research = SimpleNamespace(
        trading_dates=training_days,
        adjusted_bars=[
            {"symbol": "600000", "trade_date": date(2025, 1, 13), "open": 10.0, "close": 10.0},
            {"symbol": "600000", "trade_date": date(2025, 1, 15), "open": 10.0, "close": 11.0},
        ],
        index_bars=[
            {"trade_date": date(2025, 1, 13), "open": 100.0, "close": 100.0},
            {"trade_date": date(2025, 1, 15), "open": 100.0, "close": 102.0},
        ],
    )
    captured = {}

    import adaptive_mas.data as research_data
    import adaptive_mas.evaluation.expert_memory as expert_memory

    monkeypatch.setattr(research_data, "load_research_data", lambda *_args: research)

    def capture_matured(forecasts, outcomes, _cutoff):
        captured["forecast_ids"] = [row["record_id"] for row in forecasts]
        captured["outcome_ids"] = [row["record_id"] for row in outcomes]
        return []

    monkeypatch.setattr(expert_memory, "update_matured_performance_memory", capture_matured)
    monkeypatch.setattr(expert_memory, "candidate_performance_effects", lambda *_args: {})
    monkeypatch.setattr(batch, "_load_preview", lambda: records)
    monkeypatch.setattr(batch, "_load_states", lambda: states)
    memory_path = tmp_path / "training_memory.jsonl"
    report_path = tmp_path / "training_effect.json"
    monkeypatch.setattr(batch, "TRAINING_MEMORY_PATH", memory_path)
    monkeypatch.setattr(batch, "TRAINING_EFFECT_PATH", report_path)

    report = batch.analyze_training(tmp_path, tmp_path)

    assert len(captured["forecast_ids"]) == 79
    assert len(captured["outcome_ids"]) == 79
    assert "event-79" not in captured["forecast_ids"]
    assert "event-80" not in captured["forecast_ids"]
    assert report["training_qwen_responses"] == 79
    assert report["training_qwen_response_expected_count"] == 81
    assert report["training_qwen_response_missing_count"] == 2
    assert report["training_qwen_response_status_counts"] == {
        "completed": 79,
        "completed_invalid_response": 1,
        "delivery_unknown": 1,
    }
    assert report["missing_training_response_event_ids"] == ["event-79", "event-80"]
