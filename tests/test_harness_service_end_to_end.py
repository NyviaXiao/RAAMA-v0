from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from tests.support.synthetic_harness_service import ROOT, prepare_synthetic_service_fixture


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _request(url: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _start_server(port: int, database: Path, version_dir: Path) -> subprocess.Popen:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.Popen(
        [
            sys.executable, "-m", "adaptive_mas.harness_service", "serve",
            "--database", str(database), "--project-root", str(ROOT),
            "--version-dir", str(version_dir), "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _start_worker(database: Path) -> subprocess.Popen:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.Popen(
        [
            sys.executable, "-m", "adaptive_mas.harness_service", "worker",
            "--database", str(database), "--project-root", str(ROOT),
            "--poll-seconds", "0.1", "--worker-id", "synthetic-e2e-worker",
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_for_health(base_url: str, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 20
    last_error = None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise AssertionError(f"local API exited during startup with code {server.returncode}")
        try:
            if _request(f"{base_url}/api/health").get("ok"):
                return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.15)
    raise AssertionError(f"local API did not become ready: {last_error}")


def test_loopback_service_worker_cli_and_sqlite_result_survive_restart() -> None:
    temporary_root = ROOT / ".tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="synthetic-harness-e2e-", dir=temporary_root) as temp_name:
        fixture_root = Path(temp_name).resolve()
        fixture = prepare_synthetic_service_fixture(fixture_root)
        database = fixture_root / "isolated_service.sqlite3"
        version_dir = Path(fixture["harness_version_dir"])
        port = _free_loopback_port()
        base_url = f"http://127.0.0.1:{port}"
        server = _start_server(port, database, version_dir)
        worker = None
        try:
            _wait_for_health(base_url, server)
            worker = _start_worker(database)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                health = _request(f"{base_url}/api/health")
                if health.get("worker_online"):
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("independent offline worker did not register with the local service")

            request = {
                key: value for key, value in fixture.items()
                if key in {
                    "mode", "as_of", "horizon_sessions", "risk_template", "risk_expression",
                    "training_seed_trained_through", "training_seed_horizon_sessions",
                    "baostock_run_dir", "training_seed", "reuse_research_json", "config",
                    "public_cache_dir", "harness_version_dir", "portfolio",
                }
            }
            for path_key in (
                "baostock_run_dir", "training_seed", "reuse_research_json", "config",
                "public_cache_dir", "harness_version_dir",
            ):
                request[path_key] = os.path.relpath(str(request[path_key]), ROOT)
            submitted = _request(f"{base_url}/api/tasks", method="POST", payload={
                "task_id": "synthetic-service-e2e-task",
                "idempotency_key": "synthetic-service-e2e-idempotency",
                "request": request,
            })
            task_id = submitted["task_id"]
            assert submitted["status"] in {"queued", "running"}

            deadline = time.monotonic() + 180
            completed = None
            while time.monotonic() < deadline:
                completed = _request(f"{base_url}/api/tasks/{task_id}?detail=1")
                if completed["status"] in {"completed", "failed", "unknown"}:
                    break
                time.sleep(0.3)
            assert completed is not None and completed["status"] == "completed", (
                completed.get("error") if completed else "task did not return"
            )
            result = completed["result"]
            harness = result["evidence"]["harness"]
            research = harness["research"]
            feedback = result["evidence"]["investment_manager_feedback"]
            assert research["fixture_label"].startswith("SYNTHETIC LOCAL FIXTURE ONLY")
            assert research["agents"]["research_manager"]["report"]["risk_research"][0]["dimension"] == "industry"
            assert feedback["risk_research"][0]["source_record_ids"]
            assert harness["hard_risk_outcome"]["status"]
            assert completed["request"]["risk_profile"]["profile_sha256"] == result["evidence"]["risk_profile"]["profile_sha256"]
            assert result["evidence"]["risk_profile"]["constraints"] == {
                "risk_budget_annualized": 0.15,
                "maximum_symbol_weight": 0.05,
                "maximum_one_way_turnover": 0.30,
                "minimum_cash_weight": 0.20,
            }
            assert result["versions"]["risk_profile_sha256"] == result["evidence"]["risk_profile"]["profile_sha256"]
            runtime_identity = harness["runtime_package_identity"]
            assert runtime_identity["risk_profile_sha256"] == result["evidence"]["risk_profile"]["profile_sha256"]
            assert feedback["risk_profile"] == result["evidence"]["risk_profile"]
            assert feedback["drawdown_guarantee"] is False
            assert "不保证最大回撤或最大亏损" in feedback["summary"]
            assert harness["hard_risk_outcome"]["checks"]["minimum_cash_ratio_after_costs"] is True
            assert result["estimated_cash_after_costs_cny"] >= result["current_nav_cny"] * 0.20 - 1e-7
            assert result["predictions_by_symbol"]
            assert result["evidence"]["prediction_record_status"] not in {None, "warmup"}
            assert completed["result_summary"]["logical_research_cost_cny"] == 0.0
            assert completed["result_summary"]["new_api_cost_cny"] == 0.0

            persisted_task_id = task_id
            _stop(server)
            server = _start_server(port, database, version_dir)
            _wait_for_health(base_url, server)
            restored = _request(f"{base_url}/api/tasks/{persisted_task_id}?detail=1")
            assert restored["status"] == "completed"
            assert restored["result"] == result
            assert restored["request"]["risk_profile"] == completed["request"]["risk_profile"]

            duplicate = _request(f"{base_url}/api/tasks", method="POST", payload={
                "task_id": "synthetic-service-e2e-retry",
                "idempotency_key": "synthetic-service-e2e-idempotency",
                "request": request,
            })
            assert duplicate["task_id"] == persisted_task_id
            assert duplicate["status"] == "completed"
        finally:
            _stop(worker)
            _stop(server)
