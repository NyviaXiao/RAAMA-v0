"""Run a synthetic, isolated PostgreSQL/S3 and direct service integration probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.industrial_rag.cli import _outbox_verify_once  # noqa: E402
from adaptive_mas.industrial_rag.models import (  # noqa: E402
    Block,
    CorpusReleaseSpec,
    Document,
    DocumentBundle,
    DocumentVersion,
    Span,
)
from adaptive_mas.industrial_rag.object_store import S3CompatibleObjectStore  # noqa: E402
from adaptive_mas.industrial_rag.postgres import PostgresIndustrialRagStore  # noqa: E402
from adaptive_mas.research_provenance import stable_hash  # noqa: E402


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"private env file line {line_number} must use KEY=VALUE syntax")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or key in values or not value:
            raise ValueError(f"private env file line {line_number} has an invalid or duplicate key")
        values[key] = value
    for key in ("POSTGRES_PASSWORD", "RUSTFS_ACCESS_KEY", "RUSTFS_SECRET_KEY"):
        if not values.get(key):
            raise ValueError(f"private env file must define {key}")
    return values


def _compose(args: argparse.Namespace, *command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker", "compose", "--project-name", args.project_name,
            "--env-file", str(args.env_file), "-f", str(args.compose_file), *command,
        ],
        check=check,
        capture_output=True,
        text=True,
        timeout=90,
    )


def _published_port(args: argparse.Namespace, service: str, container_port: int) -> int:
    result = _compose(args, "port", service, str(container_port))
    match = re.fullmatch(r"127\.0\.0\.1:(\d+)\s*", result.stdout)
    if match is None:
        raise RuntimeError(f"Compose did not report a loopback-only host mapping for {service}")
    return int(match.group(1))


def _http_json(url: str, method: str = "GET", body: object | None = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=15) as response:
            raw = response.read()
            decoded = json.loads(raw) if raw.startswith((b"{", b"[")) else raw.decode("utf-8", "replace")
            return response.status, decoded
    except HTTPError as error:
        detail = error.read(512).decode("utf-8", "replace")
        raise RuntimeError(f"service returned HTTP {error.code}: {detail}") from error


def _delete_if_present(url: str) -> None:
    try:
        _http_json(url, "DELETE")
    except RuntimeError as error:
        if "HTTP 404" not in str(error):
            raise
    except OSError:
        # The isolated service may have stopped after a failed probe; Compose teardown owns cleanup.
        return


def _wait_for_postgres(store: PostgresIndustrialRagStore, timeout_seconds: float = 45) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status = store.backend_status()
            if status["postgres_connection_verified"]:
                return status
        except RuntimeError as error:
            last_error = error
        time.sleep(1)
    raise RuntimeError("PostgreSQL did not recover within the bounded reconnect window") from last_error


def _wait_for_http(url: str, timeout_seconds: float = 60) -> tuple[int, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return _http_json(url)
        except (OSError, RuntimeError) as error:
            last_error = error
            time.sleep(1)
    raise RuntimeError("service did not become reachable within the bounded readiness window") from last_error


def _postgres_dsn(port: int, password: str) -> str:
    import psycopg

    return psycopg.conninfo.make_conninfo(
        host="127.0.0.1", port=port, dbname="raama_smoke", user="raama_smoke",
        password=password, connect_timeout=8,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-file", type=Path, default=ROOT / "deploy" / "compose" / "raama-backend-smoke.yml")
    parser.add_argument("--env-file", type=Path, required=True, help="Private KEY=VALUE file; never include it in Git or probe output")
    parser.add_argument("--project-name", required=True, help="Unique isolated Compose project name, for example raama-smoke-<random>")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests" / "fixtures" / "backend_smoke_document.v1.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,48}", args.project_name):
        raise ValueError("project-name must be a unique lowercase Compose-safe identifier")
    args.compose_file = args.compose_file.resolve(strict=True)
    args.env_file = args.env_file.resolve(strict=True)
    fixture_path = args.fixture.resolve(strict=True)
    secrets = _read_env_file(args.env_file)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != "raama.financial-rag.backend-smoke-document.v1":
        raise ValueError("unsupported synthetic backend fixture schema")

    import boto3
    from botocore.config import Config

    ports = {
        "postgres": _published_port(args, "postgres", 5432),
        "rustfs": _published_port(args, "rustfs", 9000),
        "opensearch": _published_port(args, "opensearch", 9200),
        "qdrant": _published_port(args, "qdrant", 6333),
        "temporal_ui": _published_port(args, "temporal", 8233),
    }
    suffix = uuid.uuid4().hex[:12]
    release_id = f"smoke-release-{suffix}"
    version_id = f"{fixture['version_suffix']}-{suffix}"
    document_id = f"{fixture['document_suffix']}-{suffix}"
    source_id = f"{fixture['source_suffix']}-{suffix}"
    payload = str(fixture["payload_utf8"]).encode("utf-8")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    bucket = "raama-backend-smoke"
    prefix = f"smoke/{suffix}/objects"
    index_name = f"raama-smoke-{suffix}"
    collection_name = f"raama_smoke_{suffix}"
    output: dict[str, object] = {"schema_version": "raama.financial-rag.backend-smoke-result.v1", "run_id": suffix}

    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{ports['rustfs']}",
        aws_access_key_id=secrets["RUSTFS_ACCESS_KEY"],
        aws_secret_access_key=secrets["RUSTFS_SECRET_KEY"],
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 1}),
    )
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
    objects = S3CompatibleObjectStore(s3, bucket, prefix=prefix)
    object_hash = objects.put(payload)
    if object_hash != payload_sha256 or objects.get(object_hash) != payload:
        raise AssertionError("RustFS content-addressed round trip did not preserve the synthetic bytes")
    output["rustfs_s3_content_addressed_roundtrip"] = True

    store = PostgresIndustrialRagStore(_postgres_dsn(ports["postgres"], secrets["POSTGRES_PASSWORD"]), objects)
    output["postgres_backend"] = store.backend_status()
    now = datetime.now(timezone.utc)
    published = now - timedelta(minutes=1)
    text = payload.decode("utf-8")
    span = Span(span_id=f"span-{suffix}", version_id=version_id, page_number=1, char_start=0, char_end=len(text), text=text)
    block = Block(
        block_id=f"block-{suffix}", version_id=version_id, text=text,
        block_type=str(fixture["block_type"]), page_number=1, char_start=0,
        char_end=len(text), span_id=span.span_id, parser_confidence=1.0,
    )
    document = Document(
        document_id=document_id, title=str(fixture["title"]), document_type="text",
        security_code=str(fixture["security_code"]), entity_name=str(fixture["entity_name"]),
        report_period=str(fixture["report_period"]), accounting_basis=str(fixture["accounting_basis"]),
    )
    version = DocumentVersion(
        version_id=version_id, source_id=source_id, content_sha256=payload_sha256,
        source_url=str(fixture["source_url"]), rights_note=str(fixture["rights_note"]),
        published_at=published, first_available_at=published, captured_at=now,
        ingested_at=now + timedelta(milliseconds=1), time_basis="strict_forward",
        parser_name="synthetic", parser_version="1", parser_confidence=1.0, parser_status="parsed",
    )
    bundle = DocumentBundle(
        document=document, version=version, payload=payload, spans=(span,), blocks=(block,),
        read_principals=tuple(str(item) for item in fixture["read_principals"]),
    )
    ingested = store.ingest_bundle(bundle)
    if ingested.get("status") != "ingested" or objects.get(payload_sha256) != payload:
        raise AssertionError("PostgreSQL/S3 application ingest did not reopen the source object")
    outbox = _outbox_verify_once(store)
    if outbox is None or outbox.get("committed") is not True:
        raise AssertionError("PostgreSQL outbox consume did not commit its receipt")
    with store._connect() as connection:
        receipt = connection.execute(
            "SELECT result_json FROM rag_consumer_receipts WHERE consumer_id=? AND event_key=?",
            ("sqlite-reference-indexer-v1", str(outbox["event_key"])),
        ).fetchone()
    if receipt is None:
        raise AssertionError("PostgreSQL outbox receipt is missing after commit")

    source_manifest = store.source_manifest([version_id])
    text_index = store.preview_reference_index_manifest(release_id, [version_id])
    vector_index_sha256 = stable_hash({
        "release_id": release_id, "backend": "qdrant", "status": "unavailable_not_configured",
        "embedding_model": "none", "embedding_dimension": None, "points": [],
    })
    release_spec = CorpusReleaseSpec(
        release_id=release_id, source_manifest_sha256=stable_hash(source_manifest),
        parser_version="synthetic@1", chunker_version="raama-financial-chunker-v1",
        embedding_model="none", embedding_dimension=None, required_backends=("sqlite_reference",),
        text_index_manifest_sha256=str(text_index["index_manifest_sha256"]),
        vector_index_manifest_sha256=vector_index_sha256,
    )
    store.create_release(release_spec, [version_id])
    reference_receipt = store.build_reference_index_receipt(release_id)
    ready = store.mark_backend_ready(reference_receipt, store.verify_reference_index)
    publication = store.publish_release(release_id, actor_id="integration-smoke", reason="Verify isolated PostgreSQL-backed application contract.")
    if not ready["ready"] or publication["status"] != "published" or store.release_summaries()["active_release_id"] != release_id:
        raise AssertionError("PostgreSQL reference receipt and release publication did not complete")
    output["postgres_application_ingest_outbox_release_publish"] = {
        "ingested": True, "object_reopened": True, "outbox_receipt_committed": True,
        "reference_receipt_ready": True, "release_published": True,
    }

    try:
        _compose(args, "stop", "postgres")
        try:
            store.backend_status()
        except RuntimeError as error:
            if str(error) != "PostgreSQL connection failed; connection details are omitted from diagnostics":
                raise AssertionError("PostgreSQL outage did not produce the adapter's generic fail-closed error") from error
            output["postgres_outage_fails_closed"] = True
        else:
            raise AssertionError("PostgreSQL outage was not surfaced by the repository adapter")
    finally:
        _compose(args, "start", "postgres")
    prior_postgres_port = ports["postgres"]
    for service, container_port in (("postgres", 5432), ("rustfs", 9000), ("opensearch", 9200), ("qdrant", 6333), ("temporal", 8233)):
        ports["postgres" if service == "postgres" else ("temporal_ui" if service == "temporal" else service)] = _published_port(args, service, container_port)
    store.dsn = _postgres_dsn(ports["postgres"], secrets["POSTGRES_PASSWORD"])
    recovered = _wait_for_postgres(store)
    with store._connect() as connection:
        recovered_receipt = connection.execute(
            "SELECT result_json FROM rag_consumer_receipts WHERE consumer_id=? AND event_key=?",
            ("sqlite-reference-indexer-v1", str(outbox["event_key"])),
        ).fetchone()
    if (
        store.release_summaries()["active_release_id"] != release_id
        or objects.get(payload_sha256) != payload
        or recovered_receipt is None
        or store.source_manifest([version_id])[0]["content_sha256"] != payload_sha256
    ):
        raise AssertionError("PostgreSQL recovery did not retain the committed source, receipt, and release state")
    output["postgres_reconnect_preserves_committed_release"] = {
        "server_version": recovered["server_version"], "active_release_preserved": True,
        "object_bytes_preserved": True, "outbox_receipt_preserved": True,
        "dynamic_port_rechecked": True, "host_port_rebound": ports["postgres"] != prior_postgres_port,
    }

    os_url = f"http://127.0.0.1:{ports['opensearch']}"
    qdrant_url = f"http://127.0.0.1:{ports['qdrant']}"
    try:
        status, info = _wait_for_http(f"{os_url}/")
        _http_json(f"{os_url}/{index_name}", "PUT", {"mappings": {"properties": {"text": {"type": "text"}}}})
        _http_json(f"{os_url}/{index_name}/_doc/smoke?refresh=true", "PUT", {"text": fixture["index_text"]})
        search_status, found = _http_json(f"{os_url}/{index_name}/_search", "POST", {"query": {"match": {"text": "synthetic query"}}})
        if search_status != 200 or found["hits"]["total"]["value"] != 1:
            raise AssertionError("OpenSearch direct REST synthetic search failed")
        output["opensearch_direct_rest"] = {"http_status": status, "version": info.get("version", {}).get("number"), "index_search": True}
    finally:
        _delete_if_present(f"{os_url}/{index_name}")

    try:
        _http_json(f"{qdrant_url}/collections/{collection_name}", "PUT", {"vectors": {"size": len(fixture["vector"]), "distance": "Cosine"}})
        _http_json(
            f"{qdrant_url}/collections/{collection_name}/points?wait=true", "PUT",
            {"points": [{"id": 1, "vector": fixture["vector"], "payload": {"kind": "synthetic"}}]},
        )
        query_status, query_result = _http_json(
            f"{qdrant_url}/collections/{collection_name}/points/query", "POST",
            {"query": fixture["vector"], "limit": 1, "with_payload": True},
        )
        if query_status != 200 or query_result["result"]["points"][0]["id"] != 1:
            raise AssertionError("Qdrant direct REST synthetic vector query failed")
        output["qdrant_direct_rest"] = {"http_status": query_status, "query_returned_fixture": True}
    finally:
        _delete_if_present(f"{qdrant_url}/collections/{collection_name}")

    status, _ = _wait_for_http(f"http://127.0.0.1:{ports['temporal_ui']}/")
    temporal_health = _compose(args, "exec", "-T", "temporal", "temporal", "operator", "cluster", "health", "--address", "temporal:7233")
    if "SERVING" not in temporal_health.stdout.upper():
        raise AssertionError("Temporal development service did not report SERVING")
    output["temporal_dev_ui_and_rpc"] = {"ui_http_status": status, "rpc_health": "SERVING"}
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
