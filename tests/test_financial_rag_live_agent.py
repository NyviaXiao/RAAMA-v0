from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from adaptive_mas.agents import harness as harness_module
from adaptive_mas.agents.harness import CallBudget, QwenClient
from adaptive_mas.harness_state import HarnessStateStore
from scripts.run_financial_rag_live_agent import FALLBACK_MODEL, QuotaFallbackClient


class _Response:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


def _http_error(code: int, body: dict[str, object]) -> HTTPError:
    return HTTPError(
        "https://fixture.invalid/chat/completions",
        code,
        "provider rejection",
        None,
        io.BytesIO(json.dumps(body).encode("utf-8")),
    )


def test_live_agent_falls_back_only_after_explicit_quota_and_keeps_both_attempts(tmp_path, monkeypatch) -> None:
    store = HarnessStateStore(tmp_path / "provider-state.sqlite3")
    store.open_task("task", "task", {"case": "synthetic"}, max_provider_calls=2)
    sent_models: list[str] = []

    def fake_urlopen(request, **_kwargs):
        payload = json.loads(request.data)
        sent_models.append(str(payload["model"]))
        if payload["model"] == "qwen3.7-max-2026-06-08":
            raise _http_error(429, {"error": {"code": "insufficient_quota", "message": "quota exhausted"}})
        return _Response({
            "choices": [{"message": {"tool_calls": []}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        })

    monkeypatch.setattr(harness_module, "urlopen", fake_urlopen)
    primary = QwenClient("https://fixture.invalid", "fixture-secret", "qwen3.7-max-2026-06-08")
    fallback_client = QuotaFallbackClient(primary, tmp_path / "model-events.jsonl")
    budget = CallBudget(
        max_calls=5,
        ledger_path=tmp_path / "requests.jsonl",
        response_cache_dir=tmp_path / "responses",
        request_archive_dir=tmp_path / "request-archive",
        state_store=store,
        task_id="task",
        budget_scope_id="scope",
        task_provider_call_limit=2,
    )
    budget.limit_next_task(4)

    result = fallback_client.complete(
        [{"role": "user", "content": "synthetic"}],
        budget=budget,
        role="fundamental",
    )

    assert result["usage"] == {"prompt_tokens": 11, "completion_tokens": 3}
    assert sent_models == ["qwen3.7-max-2026-06-08", FALLBACK_MODEL]
    assert fallback_client.fallback_used is True
    assert fallback_client.model == FALLBACK_MODEL
    assert budget.provider_call_count == 2
    assert budget.summary()["quota_rejection_count"] == 1
    assert budget.unresolved_call is False
    assert "fixture-secret" not in (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "model-events.jsonl").read_text(encoding="utf-8"))[
        "provider_error_code"
    ] == "insufficient_quota"


def test_live_agent_does_not_fallback_on_provider_rate_limit(tmp_path, monkeypatch) -> None:
    sent_models: list[str] = []

    def fake_urlopen(request, **_kwargs):
        sent_models.append(str(json.loads(request.data)["model"]))
        raise _http_error(429, {"error": {"code": "rate_limit_exceeded", "message": "rate limit reached"}})

    monkeypatch.setattr(harness_module, "urlopen", fake_urlopen)
    primary = QwenClient("https://fixture.invalid", "fixture-secret", "primary-model")
    client = QuotaFallbackClient(primary, tmp_path / "model-events.jsonl")
    budget = CallBudget(
        max_calls=5,
        ledger_path=tmp_path / "requests.jsonl",
        response_cache_dir=tmp_path / "responses",
    )
    budget.limit_next_task(4)

    with pytest.raises(HTTPError):
        client.complete(
            [{"role": "user", "content": "synthetic"}],
            budget=budget,
            role="fundamental",
        )

    assert sent_models == ["primary-model"]
    assert client.fallback_used is False
    assert budget.unresolved_call is True
    assert budget.summary()["quota_rejection_count"] == 0
    assert not (tmp_path / "model-events.jsonl").exists()
