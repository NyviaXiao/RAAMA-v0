from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
from concurrent.futures import ThreadPoolExecutor

import pytest

from adaptive_mas.harness_state import (
    HarnessStateStore,
    StateBudgetExhausted,
    StateConflict,
    UnknownOutboundOutcome,
)
from adaptive_mas.agents.harness import CallBudget, QwenClient
import adaptive_mas.agents.harness as harness_module


def test_task_request_artifact_feedback_and_execution_survive_reopen_idempotently(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    store = HarnessStateStore(db_path)
    request = {"as_of": "2026-09-01", "account_sha256": "fixture-account-hash"}
    task = store.open_task("task-1", "request-1", request, max_provider_calls=2)
    assert task["status"] == "running"
    store.ensure_budget_scope("experiment", 3)

    response = store.reserve_provider_request(
        "task-1", "request-hash-1", scope_id="experiment",
        scope_maximum_calls=3, task_maximum_calls=2,
    )
    assert response["kind"] == "reserved"
    assert response["attempt_id"]
    provider_response = {"choices": [{"message": {"content": "fixture"}}], "usage": {
        "prompt_tokens": 7, "completion_tokens": 3,
    }}
    store.complete_provider_request("task-1", "request-hash-1", provider_response)
    store.put_artifact("task-1", "research_result", {"status": "completed", "as_of": "2026-09-01"})
    store.record_feedback("task-1", "feedback-1", {"risk_status": "passed"})
    store.record_execution_result("task-1", "execution-1", {"filled_notional_cny": 0.0})
    store.complete_task(
        "task-1", {"status": "decision_completed"},
        feedback_id="final-feedback", feedback={"risk_status": "passed"},
    )

    reopened = HarnessStateStore(db_path)
    resumed = reopened.open_task("task-1", "request-1", request, max_provider_calls=2)
    assert resumed["status"] == "completed"
    assert resumed["stage"] == "completed"
    assert reopened.get_feedback("task-1", "final-feedback") == {"risk_status": "passed"}
    assert reopened.get_execution_result("task-1", "execution-1") == {"filled_notional_cny": 0.0}
    assert reopened.list_task_events("task-1")[-1]["stage"] == "completed"
    assert reopened.get_cached_provider_response("task-1", "request-hash-1") == provider_response
    assert reopened.get_artifact("task-1", "research_result") == {
        "status": "completed", "as_of": "2026-09-01",
    }
    # Repeating the same durable writes is safe; changing their content is rejected.
    reopened.record_feedback("task-1", "feedback-1", {"risk_status": "passed"})
    reopened.record_execution_result("task-1", "execution-1", {"filled_notional_cny": 0.0})
    reopened.complete_task(
        "task-1", {"status": "decision_completed"},
        feedback_id="final-feedback", feedback={"risk_status": "passed"},
    )
    with pytest.raises(StateConflict):
        reopened.record_feedback("task-1", "feedback-1", {"risk_status": "overridden"})
    with pytest.raises(StateConflict):
        reopened.open_task(
            "task-2", "request-1", {"as_of": "2026-09-02"}, max_provider_calls=2,
        )


def test_durable_task_lifecycle_and_dead_letter_are_audited_and_fenced(tmp_path):
    store = HarnessStateStore(tmp_path / "lifecycle.sqlite3")
    store.open_task("lifecycle-task", "lifecycle-key", {"operation": "local"}, max_provider_calls=0)

    pause_payload = {"action": "pause", "operator_id": "operator-1", "reason": "Review checkpoint."}
    assert store.transition_task_status(
        "lifecycle-task", expected_statuses={"running"}, status="paused", stage="paused",
        event_key="control-pause-1", payload=pause_payload,
    )
    assert store.get_task("lifecycle-task")["status"] == "paused"
    assert store.transition_task_status(
        "lifecycle-task", expected_statuses={"running"}, status="paused", stage="paused",
        event_key="control-pause-1", payload=pause_payload,
    )  # Retried activity is idempotent even after the status changed.

    assert store.transition_task_status(
        "lifecycle-task", expected_statuses={"paused"}, status="running", stage="resumed",
        event_key="control-resume-1", payload={"action": "resume", "operator_id": "operator-1"},
    )
    assert store.get_task("lifecycle-task")["status"] == "running"
    assert store.transition_task_status(
        "lifecycle-task", expected_statuses={"running"}, status="failed", stage="failed",
        event_key="workflow-failed", payload={"error": "synthetic failure"},
    )
    assert store.dead_letter_task(
        "lifecycle-task", expected_status="failed", operator_id="operator-2",
        reason="No safe retry remains.", event_key="dead-letter-1",
    )
    assert store.get_task("lifecycle-task")["status"] == "dead_letter"
    assert store.dead_letter_task(
        "lifecycle-task", expected_status="failed", operator_id="operator-2",
        reason="No safe retry remains.", event_key="dead-letter-1",
    )
    assert len(store.list_task_events("lifecycle-task")) == 5
    assert store.list_task_events("lifecycle-task")[-1]["payload"] == {
        "expected_status": "failed", "operator_id": "operator-2", "reason": "No safe retry remains.",
    }

    assert not store.transition_task_status(
        "lifecycle-task", expected_statuses={"running"}, status="cancelled", stage="cancelled",
        event_key="late-cancel", payload={"action": "cancel"},
    )
    assert store.get_task("lifecycle-task")["status"] == "dead_letter"
    assert store.list_task_events("lifecycle-task")[-1]["stage"] == "cancelled_rejected"
    with pytest.raises(StateConflict):
        store.transition_task_status(
            "lifecycle-task", expected_statuses={"running"}, status="cancelled", stage="cancelled",
            event_key="control-pause-1", payload={"action": "cancel"},
        )
    with pytest.raises(ValueError, match="only failed or unknown"):
        store.dead_letter_task(
            "lifecycle-task", expected_status="running", operator_id="operator-2",
            reason="unsafe", event_key="dead-letter-invalid",
        )


def test_unknown_request_prevents_another_outbound_request_until_reconciled(tmp_path):
    store = HarnessStateStore(tmp_path / "unknown.sqlite3")
    store.open_task("task", "task", {"fixture": True}, max_provider_calls=3)
    store.reserve_provider_request(
        "task", "unknown-hash", scope_id="scope", scope_maximum_calls=3,
        task_maximum_calls=3,
    )

    with pytest.raises(UnknownOutboundOutcome):
        store.reserve_provider_request(
            "task", "different-hash", scope_id="scope", scope_maximum_calls=3,
            task_maximum_calls=3,
        )

    store.complete_provider_request(
        "task", "unknown-hash", {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    )
    retry_reservation = store.reserve_provider_request(
        "task", "different-hash", scope_id="scope", scope_maximum_calls=3,
        task_maximum_calls=3,
    )
    assert retry_reservation["kind"] == "reserved"


def test_concurrent_task_reservations_obey_one_transactional_shared_budget(tmp_path):
    store = HarnessStateStore(tmp_path / "concurrent.sqlite3")
    store.ensure_budget_scope("shared", 3)
    task_ids = [f"task-{index}" for index in range(8)]
    for task_id in task_ids:
        store.open_task(task_id, task_id, {"task": task_id}, max_provider_calls=1)

    def reserve(task_id: str):
        try:
            store.reserve_provider_request(
                task_id,
                f"hash-{task_id}",
                scope_id="shared",
                scope_maximum_calls=3,
                task_maximum_calls=1,
            )
            return "reserved"
        except StateBudgetExhausted:
            return "exhausted"

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(reserve, task_ids))

    assert outcomes.count("reserved") == 3
    assert outcomes.count("exhausted") == 5
    assert store.ensure_budget_scope("shared", 3) == 3


def test_qwen_client_recovers_exact_response_from_sqlite_without_resending(tmp_path, monkeypatch):
    store = HarnessStateStore(tmp_path / "runtime.sqlite3")
    store.open_task("task", "task", {"fixture": True}, max_provider_calls=2)
    response_cache = tmp_path / "cache"
    ledger = tmp_path / "calls.jsonl"
    provider_calls = []
    provider_body = {
        "choices": [{"message": {"content": "fixture response"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(provider_body).encode("utf-8")

    def fake_urlopen(*_args, **_kwargs):
        provider_calls.append(True)
        return FakeResponse()

    monkeypatch.setattr(harness_module, "urlopen", fake_urlopen)
    client = QwenClient("https://fixture.invalid", "not-a-real-secret", "fixture-model")
    first_budget = CallBudget(
        max_calls=2, ledger_path=ledger, response_cache_dir=response_cache,
        state_store=store, task_id="task", budget_scope_id="scope",
        task_provider_call_limit=2,
    )
    first_budget.limit_next_task(2)
    first = client.complete([{"role": "user", "content": "fixture"}], budget=first_budget, role="market")
    assert first == provider_body
    assert len(provider_calls) == 1

    # Simulate losing the response-cache file after SQLite durably committed it.
    next(response_cache.glob("*.json")).unlink()
    resumed_budget = CallBudget(
        max_calls=2, ledger_path=ledger, response_cache_dir=response_cache,
        state_store=store, task_id="task", budget_scope_id="scope",
        task_provider_call_limit=2,
    )
    resumed_budget.limit_next_task(2)
    resumed = client.complete([{"role": "user", "content": "fixture"}], budget=resumed_budget, role="market")

    assert resumed == provider_body
    assert len(provider_calls) == 1
    assert resumed_budget.provider_call_count == 1
    assert len(list(response_cache.glob("*.json"))) == 1


def test_offline_only_qwen_reuses_exact_cache_and_rejects_cache_miss_before_network(tmp_path, monkeypatch):
    response_cache = tmp_path / "offline-response-cache"
    ledger = tmp_path / "offline-ledger.jsonl"
    response = {
        "choices": [{"message": {"content": "cached fixture"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(response).encode("utf-8")

    def fake_urlopen(*_args, **_kwargs):
        calls.append(True)
        return FakeResponse()

    monkeypatch.setattr(harness_module, "urlopen", fake_urlopen)
    online_client = QwenClient("https://fixture.invalid", "fixture-key", "fixture-model")
    online_budget = CallBudget(max_calls=2, ledger_path=ledger, response_cache_dir=response_cache)
    online_budget.limit_next_task(2)
    messages = [{"role": "user", "content": "cached query"}]
    assert online_client.complete(messages, budget=online_budget, role="market") == response
    assert len(calls) == 1

    offline_client = QwenClient(
        "https://fixture.invalid", "fixture-key", "fixture-model", offline_only=True,
    )
    offline_budget = CallBudget(max_calls=2, ledger_path=ledger, response_cache_dir=response_cache)
    offline_budget.limit_next_task(2)
    assert offline_client.complete(messages, budget=offline_budget, role="market") == response
    assert len(calls) == 1

    uncached_budget = CallBudget(
        max_calls=2,
        ledger_path=tmp_path / "cache-miss-ledger.jsonl",
        response_cache_dir=tmp_path / "empty-cache",
    )
    uncached_budget.limit_next_task(2)
    with pytest.raises(RuntimeError, match="offline-only Qwen client"):
        offline_client.complete(
            [{"role": "user", "content": "not cached"}], budget=uncached_budget, role="market",
        )
    assert len(calls) == 1
    assert uncached_budget.provider_call_count == 0


def test_concurrent_qwen_clients_write_unique_reconstructable_ledger_entries(tmp_path, monkeypatch):
    store = HarnessStateStore(tmp_path / "concurrent-runtime.sqlite3")
    store.ensure_budget_scope("shared-ledger", 10)
    for task_id in ("task-a", "task-b"):
        store.open_task(task_id, task_id, {"task": task_id}, max_provider_calls=1)
    ledger = tmp_path / "shared-calls.jsonl"
    response_cache = tmp_path / "shared-cache"
    barrier = Barrier(2)
    provider_requests = []
    response = {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1}}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(response).encode("utf-8")

    def fake_urlopen(request, **_kwargs):
        provider_requests.append(request.data)
        barrier.wait(timeout=5)
        return FakeResponse()

    monkeypatch.setattr(harness_module, "urlopen", fake_urlopen)
    client = QwenClient("https://fixture.invalid", "fixture-key", "fixture-model")

    def invoke(task_id: str):
        budget = CallBudget(
            max_calls=10, ledger_path=ledger, response_cache_dir=response_cache,
            state_store=store, task_id=task_id, budget_scope_id="shared-ledger",
            task_provider_call_limit=1,
        )
        budget.limit_next_task(2)
        return client.complete(
            [{"role": "user", "content": task_id}], budget=budget, role="market",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(invoke, ("task-a", "task-b")))

    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    started = [event for event in events if event["event"] == "request_started"]
    completed = [event for event in events if event["event"] == "request_completed"]
    assert results == [response, response]
    assert len(provider_requests) == 2
    assert len(started) == 2
    assert len({event["ledger_id"] for event in started}) == 2
    assert {event["ledger_id"] for event in started} == {
        event["ledger_id"] for event in completed
    }
    reconstructed = CallBudget(
        max_calls=10, ledger_path=ledger, response_cache_dir=response_cache,
        state_store=store, task_id="task-a", budget_scope_id="shared-ledger",
        task_provider_call_limit=1,
    )
    assert len(reconstructed.calls) == 2
    assert len(reconstructed.logical_calls) == 2
    assert {row["task_id"] for row in reconstructed.calls} == {"task-a", "task-b"}


def test_qwen_exact_response_cache_is_bound_to_package_identity(tmp_path, monkeypatch):
    ledger = tmp_path / "package-bound-calls.jsonl"
    response_cache = tmp_path / "package-bound-cache"
    provider_calls = []
    response = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(response).encode("utf-8")

    def fake_urlopen(*_args, **_kwargs):
        provider_calls.append(True)
        return FakeResponse()

    monkeypatch.setattr(harness_module, "urlopen", fake_urlopen)
    client = QwenClient("https://fixture.invalid", "fixture-key", "fixture-model")
    messages = [{"role": "user", "content": "same provider request"}]
    for package_hash in ("a" * 64, "b" * 64):
        budget = CallBudget(
            max_calls=10,
            ledger_path=ledger,
            response_cache_dir=response_cache,
            package_identity_sha256=package_hash,
        )
        budget.limit_next_task(2)
        assert client.complete(messages, budget=budget, role="market") == response

    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    started = [event for event in events if event["event"] == "request_started"]
    assert len(provider_calls) == 2
    assert len(list(response_cache.glob("*.json"))) == 2
    assert {event["package_identity_sha256"] for event in started} == {"a" * 64, "b" * 64}
