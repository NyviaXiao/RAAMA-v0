from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import pytest

from adaptive_mas.harness_service import (
    HarnessJobWorker,
    HarnessRequestHandler,
    HarnessServiceStore,
    LocalHarnessHTTPServer,
    ServiceConflict,
    WorkerLeaseLost,
    _replay_runner,
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
        "workflow_operator_id": "local-supervisor",
        "workflow_operator_token": "workflow-test-token",
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


def test_risk_profile_validation_preserves_legacy_numeric_and_uses_template_without_hidden_numeric(tmp_path):
    root = _project(tmp_path)
    legacy = validate_decision_job(_request(root), root)
    assert legacy["risk_budget_annualized"] == 0.2
    assert legacy["risk_profile"]["sources"]["risk_budget_annualized"][0]["source"] == "legacy_numeric_field"

    template_request = _request(root) | {"risk_template": "conservative"}
    template_request.pop("risk_budget_annualized")
    profiled = validate_decision_job(template_request, root)
    assert profiled["risk_budget_annualized"] == 0.12
    assert profiled["risk_profile"]["template"] == "conservative"
    assert profiled["risk_profile"]["sources"]["risk_budget_annualized"][0]["source"] == "template_default"

    store = HarnessServiceStore(tmp_path / "risk-profile-identity.sqlite3")
    store.submit("template-task", "same-profile-key", profiled)
    changed_profile = validate_decision_job(
        template_request | {"risk_template": "balanced"}, root
    )
    with pytest.raises(ServiceConflict):
        store.submit("changed-template-task", "same-profile-key", changed_profile)


@pytest.mark.parametrize(
    "bad_field",
    [
        {"risk_expression": False},
        {"risk_template": True},
        {"risk_budget_annualized": True},
        {"risk_budget_annualized": None},
        {"risk_expression": "最多亏损10%"},
        {"risk_expression": "不要保守"},
    ],
)
def test_unsafe_or_wrongly_typed_risk_input_cannot_enter_worker_queue(tmp_path, bad_field):
    root = _project(tmp_path)
    with pytest.raises(ValueError):
        validate_decision_job(_request(root) | bad_field, root)


def test_preview_api_exposes_unresolved_without_creating_executable_task(tmp_path):
    server, thread = _server(tmp_path)
    root = _project(tmp_path)
    try:
        status, profile = _http(server, "POST", "/api/risk-profile/parse", {
            "config": str(root / "configs/daily.toml"),
            "risk_expression": "最大回撤10%",
        })
        assert status == 200
        assert profile["execution_allowed"] is False
        assert profile["unresolved"]
        status, error = _http(server, "POST", "/api/tasks", {
            "task_id": "blocked-risk",
            "idempotency_key": "blocked-risk",
            "request": _request(root) | {"risk_expression": "最大回撤10%"},
        })
        assert status == 400
        assert "needs clarification" in error["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_worker_handoff_preserves_the_same_canonical_profile(tmp_path, monkeypatch):
    root = _project(tmp_path)
    request = _request(root) | {"risk_template": "balanced", "risk_expression": "单票不超过4%"}
    request.pop("risk_budget_annualized")
    normalized = validate_decision_job(request, root)
    observed = {}

    class CompletedProcess:
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    def fake_popen(command, **kwargs):
        risk_path = command[command.index("--risk-profile-json") + 1]
        output_path = command[command.index("--output") + 1]
        observed["profile"] = json.loads((root / risk_path).read_text(encoding="utf-8")) if not Path(risk_path).is_absolute() else json.loads(Path(risk_path).read_text(encoding="utf-8"))
        observed["command"] = command
        Path(output_path).write_text("{}", encoding="utf-8")
        return CompletedProcess()

    monkeypatch.setattr("adaptive_mas.harness_service.subprocess.Popen", fake_popen)
    result = _replay_runner(
        {"task_id": "profile-handoff", "request": normalized},
        project_root=root,
        database_path=tmp_path / "isolated" / "service.sqlite3",
    )

    assert result == {}
    assert observed["profile"] == normalized["risk_profile"]
    assert "--risk-budget-annualized" not in observed["command"]
    assert observed["profile"]["constraints"]["maximum_symbol_weight"] == 0.04


def test_task_submission_replays_from_the_materialized_snapshot_and_rejects_tampering(tmp_path, monkeypatch):
    server, thread = _server(tmp_path)
    root = _project(tmp_path)
    request = _request(root)
    payload = {"task_id": "snapshot-task", "idempotency_key": "snapshot-key", "request": request}
    try:
        status, submitted = _http(server, "POST", "/api/tasks", payload)
        assert status == 202, submitted
        initial_store = server.context["store"]
        store = HarnessServiceStore(initial_store.database_path)
        job = store.get_job("snapshot-task")
        assert job is not None
        binding = job["request"]["runtime_binding"]
        materialization = binding["materialization"]
        snapshot_root = Path(materialization["snapshot_root"])
        assert (Path(materialization["source_root"]) / "scripts" / "run_harness_decision.py").is_file()
        assert materialization["runtime_package"]["package_id"] == "seed"

        retry_status, retry = _http(server, "POST", "/api/tasks", payload)
        assert retry_status == 202
        assert retry["task_id"] == submitted["task_id"]
        assert len(list(snapshot_root.parent.iterdir())) == 1

        snapshot_source_root = Path(materialization["source_root"])
        snapshot_environment = dict(os.environ)
        snapshot_environment["PYTHONPATH"] = os.pathsep.join(
            [str(snapshot_source_root / "src"), snapshot_environment.get("PYTHONPATH", "")]
        )
        binding_path = tmp_path / "queued-runtime-binding.json"
        binding_path.write_text(json.dumps(binding, ensure_ascii=False), encoding="utf-8")
        binding_check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, sys; from pathlib import Path; "
                "from adaptive_mas.runtime_binding import validate_workflow_runtime_binding; "
                "binding=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); "
                "paths=json.loads(sys.argv[3]); "
                "validate_workflow_runtime_binding(binding, project_root=sys.argv[2], request_paths=paths)",
                str(binding_path),
                str(tmp_path),
                json.dumps({key: value["path"] for key, value in materialization["inputs"].items()}),
            ],
            cwd=snapshot_source_root,
            env=snapshot_environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert binding_check.returncode == 0, binding_check.stderr

        original_config = Path(request["config"])
        original_config.write_text("horizon_sessions = 99\n", encoding="utf-8")
        Path(request["training_seed"]).unlink()
        help_result = subprocess.run(
            [
                sys.executable,
                str(snapshot_source_root / "scripts" / "run_harness_decision.py"),
                "--help",
            ],
            cwd=snapshot_source_root,
            env=snapshot_environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert help_result.returncode == 0, help_result.stderr
        observed = {"launches": 0}

        class CompletedProcess:
            returncode = 0

            def communicate(self, timeout=None):
                return "", ""

        def fake_popen(command, **kwargs):
            observed["launches"] += 1
            observed["command"] = command
            observed["cwd"] = Path(kwargs["cwd"])
            observed["pythonpath"] = kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0]
            copied_inputs = materialization["inputs"]
            for key in ("baostock_run_dir", "config", "public_cache_dir", "reuse_research_json", "training_seed"):
                argument = "--" + key.replace("_", "-")
                assert Path(command[command.index(argument) + 1]) == Path(copied_inputs[key]["path"])
            assert Path(copied_inputs["config"]["path"]).read_text(encoding="utf-8") == "horizon_sessions = 3\n"
            assert Path(command[command.index("--harness-version-dir") + 1]) == Path(materialization["runtime_package_store_root"])
            assert Path(command[command.index("--workflow-project-root") + 1]) == tmp_path.resolve()
            Path(command[command.index("--output") + 1]).write_text("{}", encoding="utf-8")
            return CompletedProcess()

        monkeypatch.setattr("adaptive_mas.harness_service.subprocess.Popen", fake_popen)
        assert _replay_runner(job, project_root=tmp_path, database_path=store.database_path) == {}
        assert observed["cwd"] == Path(materialization["source_root"])
        assert Path(observed["pythonpath"]) == Path(materialization["source_root"]) / "src"
        assert observed["launches"] == 1

        Path(materialization["inputs"]["config"]["path"]).write_text("tampered", encoding="utf-8")
        with pytest.raises(ValueError, match="materialized input failed integrity verification: config"):
            _replay_runner(job, project_root=tmp_path, database_path=store.database_path)
        assert observed["launches"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_task_submission_materializes_snapshot_below_windows_path_boundary():
    service_artifacts = Path("service_artifacts")
    example_snapshot_name = f"ws-{'0' * 12}"
    legacy_staging_name = f".{example_snapshot_name}.pending-{'0' * 32}"
    copied_schema_relative = Path("source") / "schemas" / "decision_api.v1.schema.json"
    legacy_path_tail = (
        len(str(service_artifacts))
        + 1
        + len(legacy_staging_name)
        + 1
        + len(str(copied_schema_relative))
    )
    target_project_root_length = 261 - legacy_path_tail

    with tempfile.TemporaryDirectory(prefix="raama-service-") as temporary_root:
        temporary_path = Path(temporary_root)
        root_segment_length = target_project_root_length - len(str(temporary_path)) - 1
        assert root_segment_length > 0
        project_root = temporary_path / ("r" * root_segment_length)
        project_root.mkdir()

        legacy_copy_path = project_root / service_artifacts / legacy_staging_name / copied_schema_relative
        compact_copy_path = project_root / service_artifacts / f".ws-{'0' * 24}" / copied_schema_relative
        assert len(str(legacy_copy_path)) >= 261
        assert len(str(compact_copy_path)) < 260

        server, thread = _server(project_root)
        request = _request(_project(project_root))
        try:
            status, submitted = _http(
                server,
                "POST",
                "/api/tasks",
                {"task_id": "path-boundary-task", "idempotency_key": "path-boundary", "request": request},
            )
            assert status == 202, submitted
            job = server.context["store"].get_job("path-boundary-task")
            assert job is not None
            materialization = job["request"]["runtime_binding"]["materialization"]
            copied_schema = Path(materialization["source_root"]) / "schemas" / "decision_api.v1.schema.json"
            assert copied_schema.is_file()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


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


def test_expired_worker_lease_is_quarantined_until_audited_reconciliation(tmp_path):
    root = _project(tmp_path)
    store = HarnessServiceStore(tmp_path / "unknown-worker.sqlite3")
    request = validate_decision_job(_request(root), root)
    store.submit("uncertain-job", "uncertain-idempotency", request)
    claimed = store.claim_next("worker-before-crash", lease_seconds=60)
    assert claimed is not None and claimed["attempt_count"] == 1

    connection = store._connect()
    try:
        connection.execute("UPDATE service_jobs SET lease_until=0 WHERE task_id=?", ("uncertain-job",))
    finally:
        connection.close()

    assert store.claim_next("worker-after-crash") is None
    unknown = store.get_job("uncertain-job")
    assert unknown["status"] == "unknown"
    assert unknown["error"]["previous_worker_id"] == "worker-before-crash"
    assert store.events("uncertain-job")[-1]["stage"] == "unknown"

    with pytest.raises(ServiceConflict, match="attempt changed"):
        store.reconcile_unknown_job(
            "uncertain-job", expected_attempt=2, operator_id="local-supervisor", reason="checked"
        )
    reconciled = store.reconcile_unknown_job(
        "uncertain-job",
        expected_attempt=1,
        operator_id="local-supervisor",
        reason="Confirmed the local replay had not started.",
    )
    assert reconciled["status"] == "queued"
    event = store.events("uncertain-job")[-1]
    assert event["stage"] == "reconciled_retry"
    assert event["payload"]["attributes"]["operator_id"] == "local-supervisor"
    assert event["payload"]["attributes"]["action"] == "retry_after_confirmed_no_external_effect"
    retried = store.claim_next("worker-after-crash")
    assert retried is not None and retried["attempt_count"] == 2


def test_expired_lease_generation_fences_every_stale_worker_write(tmp_path):
    root = _project(tmp_path)
    store = HarnessServiceStore(tmp_path / "lease-generation.sqlite3")
    request = validate_decision_job(_request(root), root)
    store.submit("fenced-job", "fenced-idempotency", request)
    first = store.claim_next("restarted-worker", lease_seconds=60)
    assert first is not None and first["attempt_count"] == 1
    first_token = first["lease_token"]

    connection = store._connect()
    try:
        connection.execute("UPDATE service_jobs SET lease_until=0 WHERE task_id=?", ("fenced-job",))
    finally:
        connection.close()
    assert store.claim_next("lease-checker") is None
    store.reconcile_unknown_job(
        "fenced-job", expected_attempt=1, operator_id="local-supervisor", reason="No replay effect was applied."
    )

    second = store.claim_next("restarted-worker", lease_seconds=60)
    assert second is not None and second["attempt_count"] == 2
    second_token = second["lease_token"]
    assert first_token != second_token
    before = store.get_job("fenced-job")
    before_events = store.events("fenced-job")
    before_workers = store.worker_status()

    with pytest.raises(WorkerLeaseLost):
        store.update_stage("fenced-job", "stale-stage", worker_id="restarted-worker", lease_token=first_token)
    with pytest.raises(WorkerLeaseLost):
        store.renew_lease("fenced-job", "restarted-worker", lease_token=first_token)
    with pytest.raises(WorkerLeaseLost):
        store.complete("fenced-job", "restarted-worker", first_token, {"attempt": 1}, elapsed_ms=1)
    with pytest.raises(WorkerLeaseLost):
        store.fail("fenced-job", "restarted-worker", first_token, {"error": "stale attempt"})
    assert store.acknowledge_control("fenced-job", "restarted-worker", first_token) is None

    after = store.get_job("fenced-job")
    assert after["status"] == "running" and after["attempt_count"] == 2
    assert after.get("result") is None and after["lease_token"] == second_token
    assert store.events("fenced-job") == before_events
    assert store.worker_status() == before_workers
    assert before.get("result") is None

    store.complete("fenced-job", "restarted-worker", second_token, {"attempt": 2}, elapsed_ms=1)
    completed = store.get_job("fenced-job")
    assert completed["status"] == "completed"
    assert completed["result"] == {"attempt": 2}


def test_pause_resume_and_cancel_are_durable_worker_control_transitions(tmp_path):
    root = _project(tmp_path)
    store = HarnessServiceStore(tmp_path / "workflow-controls.sqlite3")
    request = validate_decision_job(_request(root), root)
    store.submit("controlled-job", "controlled-idempotency", request)

    def pause_runner(job):
        store.request_control(
            str(job["task_id"]), action="pause", operator_id="local-supervisor", reason="Pause after this activity."
        )
        return {"attempt": 1}

    worker = HarnessJobWorker(store, project_root=root, worker_id="control-worker", runner=pause_runner)
    assert worker.run_once() is True
    paused = store.get_job("controlled-job")
    assert paused["status"] == "paused" and paused.get("result") is None
    assert store.events("controlled-job")[-1]["stage"] == "paused"

    resumed = store.request_control(
        "controlled-job", action="resume", operator_id="local-supervisor", reason="Resume for cancellation check."
    )
    assert resumed["status"] == "queued"

    def cancel_runner(job):
        store.request_control(
            str(job["task_id"]), action="cancel", operator_id="local-supervisor", reason="Stop this attempt."
        )
        return {"attempt": 2}

    worker = HarnessJobWorker(store, project_root=root, worker_id="control-worker", runner=cancel_runner)
    assert worker.run_once() is True
    cancelled = store.get_job("controlled-job")
    assert cancelled["status"] == "cancelled" and cancelled.get("result") is None
    events = store.events("controlled-job")
    controls = [event for event in events if event["stage"].startswith("control_")]
    assert [event["payload"]["attributes"]["operator_id"] for event in controls] == [
        "local-supervisor", "local-supervisor", "local-supervisor",
    ]
    assert any(event["stage"] == "paused" for event in events)
    assert any(event["stage"] == "cancelled" for event in events)


def test_task_control_api_requires_server_operator_and_rejects_forged_identity(tmp_path):
    server, thread = _server(tmp_path)
    try:
        store = server.context["store"]
        store.submit("api-control", "api-control-idempotency", {"mode": "decision_reuse"})
        body = {"action": "pause", "reason": "Operator requested a hold."}
        status, error = _http(server, "POST", "/api/tasks/api-control/control", body)
        assert status == 403 and "authorization" in error["error"]

        status, error = _http(
            server,
            "POST",
            "/api/tasks/api-control/control",
            body | {"operator_id": "forged"},
            token="workflow-test-token",
        )
        assert status == 400 and "requires action and reason" in error["error"]

        status, paused = _http(
            server, "POST", "/api/tasks/api-control/control", body, token="workflow-test-token"
        )
        assert status == 200 and paused["status"] == "paused"
        status, resumed = _http(
            server,
            "POST",
            "/api/tasks/api-control/control",
            {"action": "resume", "reason": "Operator resumed the task."},
            token="workflow-test-token",
        )
        assert status == 200 and resumed["status"] == "queued"
        event = store.events("api-control")[-1]
        assert event["payload"]["attributes"]["operator_id"] == "local-supervisor"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_unknown_job_can_be_closed_as_audited_dead_letter_without_retry(tmp_path):
    server, thread = _server(tmp_path)
    try:
        store = server.context["store"]
        store.submit("api-dead-letter", "api-dead-letter-idempotency", {"mode": "decision_reuse"})
        claimed = store.claim_next("worker-may-still-be-running", lease_seconds=60)
        assert claimed is not None
        connection = store._connect()
        try:
            connection.execute("UPDATE service_jobs SET lease_until=0 WHERE task_id=?", ("api-dead-letter",))
        finally:
            connection.close()
        assert store.claim_next("lease-checker") is None

        body = {"expected_attempt": 1, "reason": "The outcome is uncertain; do not repeat this attempt."}
        status, error = _http(server, "POST", "/api/tasks/api-dead-letter/dead-letter", body)
        assert status == 403 and "authorization" in error["error"]
        status, error = _http(
            server,
            "POST",
            "/api/tasks/api-dead-letter/dead-letter",
            body | {"operator_id": "forged"},
            token="workflow-test-token",
        )
        assert status == 400 and "requires expected_attempt and reason" in error["error"]
        status, error = _http(
            server,
            "POST",
            "/api/tasks/api-dead-letter/dead-letter",
            body | {"expected_attempt": 2},
            token="workflow-test-token",
        )
        assert status == 409 and "attempt changed" in error["error"]

        status, result = _http(
            server,
            "POST",
            "/api/tasks/api-dead-letter/dead-letter",
            body,
            token="workflow-test-token",
        )
        assert status == 200 and result["status"] == "dead_letter"
        job = store.get_job("api-dead-letter")
        assert job["status"] == "dead_letter" and job["attempt_count"] == 1
        with pytest.raises(WorkerLeaseLost):
            store.complete(
                "api-dead-letter", "worker-may-still-be-running", claimed["lease_token"],
                {"attempt": 1}, elapsed_ms=1,
            )
        assert store.get_job("api-dead-letter")["status"] == "dead_letter"
        event = store.events("api-dead-letter")[-1]
        assert event["stage"] == "dead_letter"
        assert event["payload"]["attributes"]["operator_id"] == "local-supervisor"
        assert event["payload"]["attributes"]["action"] == "close_without_retry"
        assert event["payload"]["attributes"]["reason"] == body["reason"]
        with pytest.raises(ServiceConflict, match="only an unknown job"):
            store.reconcile_unknown_job(
                "api-dead-letter", expected_attempt=1, operator_id="local-supervisor", reason="retry"
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_unknown_job_reconciliation_requires_server_operator_and_rejects_caller_identity(tmp_path):
    server, thread = _server(tmp_path)
    try:
        store = server.context["store"]
        store.submit("api-unknown", "api-unknown-idempotency", {"mode": "decision_reuse"})
        assert store.claim_next("worker-crashed", lease_seconds=60) is not None
        connection = store._connect()
        try:
            connection.execute("UPDATE service_jobs SET lease_until=0 WHERE task_id=?", ("api-unknown",))
        finally:
            connection.close()
        assert store.claim_next("lease-checker") is None

        body = {
            "action": "retry_after_confirmed_no_external_effect",
            "expected_attempt": 1,
            "reason": "operator checked local task artifacts",
        }
        status, error = _http(server, "POST", "/api/tasks/api-unknown/reconcile", body)
        assert status == 403 and "authorization" in error["error"]
        status, error = _http(
            server, "POST", "/api/tasks/api-unknown/reconcile",
            body | {"operator_id": "forged"}, token="workflow-test-token",
        )
        assert status == 400 and "reconciliation requires" in error["error"]
        status, error = _http(
            server, "POST", "/api/tasks/api-unknown/reconcile",
            body | {"reason": None}, token="workflow-test-token",
        )
        assert status == 400 and "reason must be" in error["error"]
        status, result = _http(
            server, "POST", "/api/tasks/api-unknown/reconcile", body, token="workflow-test-token"
        )
        assert status == 200 and result["status"] == "queued"
        event = store.events("api-unknown")[-1]
        assert event["payload"]["attributes"]["operator_id"] == "local-supervisor"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


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
