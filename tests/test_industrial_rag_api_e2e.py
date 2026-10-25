from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore
from adaptive_mas.industrial_rag.repository import IndustrialRagStore
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService
from adaptive_mas.industrial_rag.web import LocalRagHTTPServer, RagRequestHandler


def _cli(repo_root: Path, env: dict[str, str], *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "adaptive_mas.industrial_rag.cli", *arguments],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


def _request_json(url: str, value: dict[str, object] | None = None) -> dict[str, object]:
    if value is None:
        request = Request(url)
    else:
        request = Request(
            url,
            data=json.dumps(value, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_cli_ingest_release_status_and_live_local_api_flow(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    source = tmp_path / "synthetic_public_financial_table.csv"
    metadata_path = tmp_path / "synthetic_metadata.json"
    database = tmp_path / "rag.sqlite3"
    objects = tmp_path / "objects"
    scratch = tmp_path / "parser-scratch"
    source.write_text(
        "项目,2024年,单位\n营业收入,123456,人民币元\n净利润,789,人民币元\n",
        encoding="utf-8",
        newline="",
    )
    first_available = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    metadata = {
        "document_id": "synthetic-doc-1",
        "version_id": "synthetic-version-1",
        "source_id": "synthetic-source-1",
        "title": "SYNTHETIC API FIXTURE ONLY — 2024 financial table",
        "document_type": "synthetic_fixture",
        "security_code": "600036",
        "entity_name": "synthetic issuer fixture",
        "report_period": "2024",
        "accounting_basis": "synthetic basis fixture",
        "source_url": "https://example.invalid/synthetic/financial-table.csv",
        "rights_note": "Synthetic test fixture only; not an issuer filing.",
        "published_at": first_available,
        "first_available_at": first_available,
        "time_basis": "strict_forward",
        "read_principals": ["local-public"],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    ingested = _cli(
        repo_root, env,
        "ingest", "--source", str(source), "--metadata", str(metadata_path),
        "--database", str(database), "--objects", str(objects), "--scratch", str(scratch),
    )
    assert ingested["status"] == "ingested"
    assert ingested["parser"]["status"] == "parsed"

    released = _cli(
        repo_root, env,
        "release", "--release-id", "synthetic-release-1", "--version-id", "synthetic-version-1",
        "--database", str(database), "--objects", str(objects), "--reason", "isolated API contract test",
    )
    assert released["status"] == "published"
    assert released["release"]["release_id"] == "synthetic-release-1"

    status = _cli(repo_root, env, "status", "--database", str(database), "--objects", str(objects))
    assert status["release"]["active_release_id"] == "synthetic-release-1"

    store = IndustrialRagStore(database, LocalContentAddressedStore(objects))
    server = LocalRagHTTPServer(("127.0.0.1", 0), RagRequestHandler)
    server.context = {
        "store": store,
        "service": IndustrialRagService(store),
        "principal_id": "local-public",
        "time_mode": "strict_forward",
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        health = _request_json(base_url + "/api/health")
        assert health["ok"] is True
        assert health["release"]["active_release_id"] == "synthetic-release-1"

        query_scope = {
            "security_code": "600036",
            "query": "2024年营业收入",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "task_type": "fact",
            "channels": ["bm25"],
            "top_k": 5,
        }
        result = _request_json(base_url + "/api/search", query_scope)
        assert result["query_binding"]["release_id"] == "synthetic-release-1"
        assert result["retrieved"]
        assert any("营业收入" in item["text"] for item in result["retrieved"])

        research = _request_json(base_url + "/api/research", query_scope)
        assert research["status"] == "candidate_only"
        assert research["coverage"]
        assert research["coverage"][0]["status"] == "candidate_only"
        assert research["supporting_evidence"] == []
        assert all(item["citation"].get("numeric_eligible") is False for item in research["candidate_evidence"])

        table_item = next(
            item for item in result["retrieved"]
            if item["citation"].get("table_id") is not None
        )
        table_row = _request_json(base_url + "/api/table-row", {
            **{key: query_scope[key] for key in ("security_code", "query", "as_of")},
            "table_id": table_item["citation"]["table_id"],
            "row_start": table_item["citation"]["row_start"],
            "row_end": table_item["citation"]["row_end"],
        })
        assert table_row["rows"]

        original_url = base_url + "/api/original/synthetic-doc-1/synthetic-version-1?" + urlencode({
            key: query_scope[key] for key in ("security_code", "query", "as_of")
        })
        with urlopen(original_url, timeout=5) as response:
            assert response.read() == source.read_bytes()

        try:
            _request_json(base_url + "/api/search", {**query_scope, "principal_id": "untrusted"})
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("caller-controlled principal_id must be rejected")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
