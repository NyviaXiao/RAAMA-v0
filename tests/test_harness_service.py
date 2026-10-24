from __future__ import annotations

import http.client
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

from adaptive_mas.harness_service import (
    HarnessJobWorker,
    HarnessRequestHandler,
    HarnessServiceStore,
    LocalHarnessHTTPServer,
    ServiceConflict,
    validate_decision_job,
    validate_execution_ledger_request,
)
from adaptive_mas.observability import decision_summary, export_span_if_configured, safe_attributes, span_record
from adaptive_mas.release_governance import HarnessReleaseStore


def _project(tmp_path):
    root = tmp_path / "project"
    for name in (
        "data/raw/baostock/cache",
        "data/processed",
        "experiments",
        "configs",
        "data/processed/harness_versions",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "data/processed/seed.json").write_text("{}", encoding="utf-8")
    (root / "experiments/research.json").write_text("{}", encoding="utf-8")
    (root / "configs/daily.toml").write_text("horizon_sessions = 3\n", encoding="utf-8")
    return root


def _request(root):
    return {
        "mode": "decision_reuse",
        "as_of": "2026-04-01",
        "horizon_sessions": 3,
        "risk_budget_annualized": 0.2,
        "training_seed_trained_through": "2026-03-01",
        "training_seed_horizon_sessions": 3,
        "baostock_run_dir": str(root / "data/raw/baostock/cache"),
        "training_seed": str(root / "data/processed/seed.json"),
        "reuse_research_json": str(root / "experiments/research.json"),
        "config": str(root / "configs/daily.toml"),
        "public_cache_dir": str(root / "data/raw/baostock/cache"),
        "harness_version_dir": str(root / "data/processed/harness_versions"),
        "portfolio": {"cash_cny": 10000, "holdings_cny": {"600000": 0}},
    }


def _rules():
    return {
        "ruleset_id": "fixture-rule-v1",
        "effective_date": "2026-01-01",
        "source_title": "Test fixture only",
        "source_url": "https://example.invalid/fixture",
        "symbols": {
            "600000": {
                "board": "fixture",
                "order_unit": 100,
                "minimum_buy_quantity": 100,
                "odd_lot_sell": True,
                "t_plus_one": True,
                "tick_size": 0.01,
                "price_limit_pct": 0.1,
                "maximum_daily_participation": 1.0,
                "liquidity_lookback_sessions": 1,
            }
        },
        "fees": {
            "commission_per_side_bps": 5,
            "minimum_commission_cny": 5,
            "transfer_fee_per_side_bps": 0.1,
            "stamp_duty_sell_bps": 5,
            "slippage_per_side_bps": 5,
        },
    }


def _ledger_request():
    return {
        "execution_mode": "quantity",
        "trading_dates": ["2026-01-05", "2026-01-06"],
        "market_bars": [
            {"symbol": "600000", "trade_date": "2026-01-05", "tradestatus": "1", "open": 10, "high": 10, "low": 10, "close": 10, "preclose": 10, "amount": 1000000, "volume": 100000},
            {"symbol": "600000", "trade_date": "2026-01-06", "tradestatus": "1", "open": 10.1, "high": 10.1, "low": 10.1, "close": 10.1, "preclose": 10, "amount": 1000000, "volume": 100000},
        ],
        "orders": [{
            "trade_date": "2026-01-06", "symbol": "600000", "side": "buy", "quantity": 100,
            "source_prediction_id": "pred-fixture", "prediction_date": "2026-01-05",
            "prediction_horizon_sessions": 3,
        }],
        "rule_pack": _rules(),
        "initial_cash_cny": 10000,
        "prediction_linkage_mode": "linked",
        "prediction_records": [{
            "prediction_id": "pred-fixture", "symbol": "600000", "prediction_date": "2026-01-05",
            "available_at": "2026-01-05T15:00:00+08:00", "horizon_sessions": 3,
            "expected_return": 0.01, "label_status": "pending", "realized_return": None,
        }],
        "research_cost_cny_by_date": {"2026-01-06": 2.5},
    }


def _server(tmp_path):
    database = tmp_path / "service.sqlite3"
    service_store = HarnessServiceStore(database)
    release_store = HarnessReleaseStore(tmp_path / "versions", project_root=tmp_path)
    server = LocalHarnessHTTPServer(("127.0.0.1", 0), HarnessRequestHandler)
    server.context = {
        "store": service_store,
        "release_store": release_store,
        "project_root": tmp_path,
        "release_manager_token": "local-test-token",
    }
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _http(server, method, path, payload=None, token=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    result = json.loads(response.read().decode("utf-8"))
    status = response.status
    connection.close()
    return status, result


def test_decision_job_validation_is_local_and_provider_disabled(tmp_path):
    root = _project(tmp_path)
    normalized = validate_decision_job(_request(root), root)
    assert normalized["mode"] == "decision_reuse"
    assert normalized["portfolio"]["cash_cny"] == 10000
    bad_mode = _request(root) | {"mode": "provider_run"}
    try:
        validate_decision_job(bad_mode, root)
    except ValueError as exc:
        assert "decision_reuse" in str(exc)
    else:
        raise AssertionError("provider mode must be rejected")
    outside = _request(root) | {"reuse_research_json": str(tmp_path / "outside.json")}
    try:
        validate_decision_job(outside, root)
    except ValueError as exc:
        assert "inside" in str(exc)
    else:
        raise AssertionError("project-root escape must be rejected")


def test_sqlite_idempotency_concurrent_claims_and_worker_recovery(tmp_path):
    root = _project(tmp_path)
    store = HarnessServiceStore(tmp_path / "state.sqlite3")
    request = validate_decision_job(_request(root), root)

    def submit(task_id):
        return store.submit(task_id, "stable-key", request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        submitted = list(pool.map(submit, ("task-a", "task-b")))
    assert submitted[0]["task_id"] == submitted[1]["task_id"]
    assert len(store.list_jobs()) == 1
    try:
        store.submit("task-c", "stable-key", request | {"risk_budget_annualized": 0.3})
    except ServiceConflict:
        pass
    else:
        raise AssertionError("changed request under an existing idempotency key must conflict")

    worker = HarnessJobWorker(
        store, project_root=root,
        runner=lambda _job: {"status": "warmup_no_action", "orders": [], "predictions_by_symbol": {}},
    )
    assert worker.run_once() is True
    task_id = submitted[0]["task_id"]
    saved = store.get_job(task_id)
    assert saved["status"] == "completed"
    assert saved["result"]["status"] == "warmup_no_action"
    assert [event["stage"] for event in store.events(task_id)][-1] == "completed"

    store.submit("task-c", "stable-c", request)
    store.submit("task-d", "stable-d", request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(store.claim_next, ("worker-a", "worker-b")))
    claimed_ids = [job["task_id"] for job in claims if job is not None]
    assert len(claimed_ids) == len(set(claimed_ids)) == 2


def test_execution_ledger_api_persists_immutable_simulation(tmp_path):
    server, thread = _server(tmp_path)
    try:
        assert _http(server, "GET", "/api/health")[1]["ok"] is True
        status, saved = _http(server, "POST", "/api/execution-ledgers", {"ledger_id": "ledger-1", "request": _ledger_request()})
        assert status == 201
        assert saved["result"]["broker_orders_enabled"] is False
        assert saved["result"]["summary"]["all_cash_reconciliations_exact"] is True
        assert saved["result"]["summary"]["all_quantity_reconciliations_exact"] is True
        day = saved["result"]["daily_nav"][1]
        assert day["fees_cny"] > 0
        assert day["slippage_cost_cny"] > 0
        assert day["research_cost_cny"] == 2.5
        status, repeated = _http(server, "POST", "/api/execution-ledgers", {"ledger_id": "ledger-1", "request": _ledger_request()})
        assert status == 200 and repeated["reused"] is True
        changed = _ledger_request() | {"initial_cash_cny": 9000}
        status, error = _http(server, "POST", "/api/execution-ledgers", {"ledger_id": "ledger-1", "request": changed})
        assert status == 409 and "immutable" in error["error"]
        listing = _http(server, "GET", "/api/execution-ledgers")[1]["ledgers"]
        assert len(listing) == 1 and listing[0]["ledger_id"] == "ledger-1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_release_manager_is_derived_from_bearer_token(tmp_path):
    server, thread = _server(tmp_path)
    body = {"package_id": "not-registered", "evidence_ids": [], "reason": "test"}
    try:
        status, error = _http(server, "POST", "/api/releases/publish", body)
        assert status == 403
        assert "authorization" in error["error"]
        status, error = _http(server, "POST", "/api/releases/publish", body, token="local-test-token")
        assert status in {400, 422}
        assert "authorization" not in error["error"].lower()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_evaluation_api_reports_registry_and_rejects_caller_pass_flags(tmp_path):
    server, thread = _server(tmp_path)
    try:
        assert _http(server, "GET", "/api/evaluations")[1] == {"protocols": [], "packages": [], "evidence": []}
        status, error = _http(server, "POST", "/api/evaluations", {"action": "evaluate", "passed": True})
        assert status == 400
        assert "protocol_id" in error["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_trace_redaction_and_summary_exclude_private_account_details():
    safe = safe_attributes({
        "api_key": "secret-value",
        "portfolio": {"600000": 250000},
        "source_path": "C:/private/cache/file.json",
        "source": "local-cache",
        "nested": {"account_number": "123456", "risk_budget": 0.2},
    })
    assert "api_key" not in safe and "portfolio" not in safe and "source_path" not in safe
    assert safe["nested"] == {"risk_budget": 0.2}
    summary = decision_summary({"portfolio": {"cash_cny": 1}, "orders": [], "versions": {}})
    assert "portfolio" not in summary
    assert summary["order_count"] == 0


def test_otel_export_is_opt_in(monkeypatch):
    monkeypatch.delenv("RAAMA_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert export_span_if_configured(span_record(trace_id="trace-1", name="test")) == "disabled"


def test_ledger_request_rejects_unsupported_fields():
    raw = _ledger_request() | {"provider_api_key": "must-not-be-accepted"}
    try:
        validate_execution_ledger_request(raw)
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unsupported execution fields must be rejected")
