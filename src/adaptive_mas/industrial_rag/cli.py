"""Isolated import, release and loopback workbench commands for industrial financial RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from adaptive_mas.industrial_rag.models import CorpusReleaseSpec, Document, DocumentBundle, DocumentVersion
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore
from adaptive_mas.industrial_rag.parser import parse_isolated, records_from_parse
from adaptive_mas.industrial_rag.repository import IndustrialRagStore
from adaptive_mas.industrial_rag.web import serve
from adaptive_mas.research_provenance import stable_hash


SUPERVISION_ROOT = Path(".tmp/industrial_rag_supervision_20261025")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata file must contain one JSON object")
    return value


def _store(database_path: str | Path, object_root: str | Path) -> IndustrialRagStore:
    return IndustrialRagStore(database_path, LocalContentAddressedStore(object_root))


def ingest_file(
    *,
    source_path: str | Path,
    metadata_path: str | Path,
    database_path: str | Path,
    object_root: str | Path,
    scratch_root: str | Path,
) -> dict[str, object]:
    source = Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError("source document does not exist")
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    metadata = _load_json(metadata_path)
    allowed = {
        "document_id", "version_id", "source_id", "title", "document_type", "security_code", "entity_name",
        "report_period", "accounting_basis", "aliases", "source_url", "rights_note",
        "expected_sha256", "published_at", "first_available_at", "time_basis", "valid_from",
        "valid_to", "revised_at", "read_principals",
    }
    if set(metadata) - allowed:
        raise ValueError("source metadata contains unknown or unsupported fields")
    if metadata.get("expected_sha256") and metadata["expected_sha256"] != digest:
        raise ValueError("local source SHA-256 does not match the supplied manifest")
    parsed = parse_isolated(payload, source.name, scratch_root=scratch_root)
    captured_at = _now()
    version_id = str(metadata.get("version_id") or f"v-{digest[:24]}")
    document = Document(
        document_id=str(metadata["document_id"]), title=str(metadata["title"]),
        document_type=str(metadata["document_type"]),
        security_code=str(metadata["security_code"]) if metadata.get("security_code") else None,
        entity_name=str(metadata["entity_name"]) if metadata.get("entity_name") else None,
        report_period=str(metadata["report_period"]) if metadata.get("report_period") else None,
        accounting_basis=str(metadata["accounting_basis"]) if metadata.get("accounting_basis") else None,
        aliases=tuple(str(item) for item in metadata.get("aliases", [])),
    )
    parser_confidence = float(parsed.get("parser_confidence", 0.0))
    parser_status = str(parsed.get("status", "quarantined"))
    version = DocumentVersion(
        version_id=version_id,
        source_id=str(metadata["source_id"]),
        content_sha256=digest,
        source_url=str(metadata["source_url"]),
        rights_note=str(metadata["rights_note"]),
        published_at=metadata.get("published_at"),
        first_available_at=metadata.get("first_available_at"),
        captured_at=captured_at,
        ingested_at=_now(),
        valid_from=metadata.get("valid_from"),
        valid_to=metadata.get("valid_to"),
        revised_at=metadata.get("revised_at"),
        time_basis=str(metadata.get("time_basis", "unknown")),  # type: ignore[arg-type]
        parser_name=str(parsed.get("parser_name", "unavailable")),
        parser_version=str(parsed.get("parser_version", "unavailable")),
        parser_confidence=parser_confidence,
        parser_status="parsed" if parser_status == "parsed" and parser_confidence >= 0.6 else "quarantined",
        parser_error_code=(str(parsed.get("error_code") or "low_parser_quality") if parser_status != "parsed" or parser_confidence < 0.6 else None),
        parser_error=(str(parsed.get("error") or "structural text coverage below release threshold") if parser_status != "parsed" or parser_confidence < 0.6 else None),
    )
    records = records_from_parse(parsed, version_id) if version.parser_status == "parsed" else {}
    bundle = DocumentBundle(
        document=document, version=version, payload=payload,
        sections=records.get("sections", ()), spans=records.get("spans", ()),
        blocks=records.get("blocks", ()), tables=records.get("tables", ()),
        cells=records.get("cells", ()),
        read_principals=tuple(str(value) for value in metadata.get("read_principals", ["local-public"])),
    )
    store = _store(database_path, object_root)
    result = store.ingest_bundle(bundle)
    result["parser"] = {
        "status": version.parser_status, "name": version.parser_name,
        "version": version.parser_version, "confidence": parser_confidence,
        "quality": parsed.get("quality"), "error_code": version.parser_error_code,
    }
    return result


def _outbox_verify_once(store: IndustrialRagStore) -> dict[str, object] | None:
    def verify(event_key: str, event_type: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        digest = str(payload.get("content_sha256", ""))
        stored = store.object_store.get(digest)
        if hashlib.sha256(stored).hexdigest() != digest:
            raise ValueError("outbox source bytes failed checksum verification")
        return {"verified_content_sha256": digest, "event_type": event_type, "handler_key": event_key}

    return store.consume_outbox_once("sqlite-reference-indexer-v1", verify)


def build_release(
    *,
    release_id: str,
    version_ids: list[str],
    database_path: str | Path,
    object_root: str | Path,
    actor_id: str,
    reason: str,
) -> dict[str, object]:
    store = _store(database_path, object_root)
    for version_id in version_ids:
        source = next((item for item in store.source_manifest([version_id])), None)
        if source is None:
            raise FileNotFoundError(f"document version {version_id!r} is not registered")
    handled_events = 0
    while True:
        event = _outbox_verify_once(store)
        if event is None:
            break
        handled_events += 1
    source_manifest_sha256 = stable_hash(store.source_manifest(version_ids))
    text_index_sha256 = str(store.preview_reference_index_manifest(release_id, version_ids)["index_manifest_sha256"])
    vector_index_sha256 = stable_hash({
        "release_id": release_id,
        "backend": "qdrant",
        "status": "unavailable_not_configured",
        "embedding_model": "none",
        "embedding_dimension": None,
        "points": [],
    })
    parser_versions = sorted({
        f"{item['parser_name']}@{item['parser_version']}"
        for item in store.source_manifest(version_ids)
    })
    spec = CorpusReleaseSpec(
        release_id=release_id,
        source_manifest_sha256=source_manifest_sha256,
        parser_version=f"pinned:{stable_hash(parser_versions)[:20]}",
        chunker_version="raama-financial-chunker-v1",
        embedding_model="none",
        embedding_dimension=None,
        required_backends=("sqlite_reference",),
        text_index_manifest_sha256=text_index_sha256,
        vector_index_manifest_sha256=vector_index_sha256,
    )
    created = store.create_release(spec, version_ids)
    receipt = store.build_reference_index_receipt(release_id)
    backend = store.mark_backend_ready(receipt, store.verify_reference_index)
    published = store.publish_release(release_id, actor_id=actor_id, reason=reason)
    return {"status": "published", "release": published, "backend_receipt": backend, "outbox_events_verified": handled_events, "created": created}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    root = Path(__file__).resolve().parents[3]
    default_supervision = root / SUPERVISION_ROOT

    serve_parser = subparsers.add_parser("serve", help="serve the isolated Chinese query workbench")
    serve_parser.add_argument("--database", type=Path, default=default_supervision / "industrial_rag.sqlite3")
    serve_parser.add_argument("--objects", type=Path, default=default_supervision / "objects")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8877)
    serve_parser.add_argument("--principal-id", default="local-public")
    serve_parser.add_argument("--time-mode", choices=["strict_forward", "retrospective_diagnostic"], default="strict_forward")

    ingest_parser = subparsers.add_parser("ingest", help="parse and register one source with explicit provenance metadata")
    ingest_parser.add_argument("--source", type=Path, required=True)
    ingest_parser.add_argument("--metadata", type=Path, required=True)
    ingest_parser.add_argument("--database", type=Path, default=default_supervision / "industrial_rag.sqlite3")
    ingest_parser.add_argument("--objects", type=Path, default=default_supervision / "objects")
    ingest_parser.add_argument("--scratch", type=Path, default=default_supervision / "parser_scratch")

    release_parser = subparsers.add_parser("release", help="build, verify and publish an immutable local corpus release")
    release_parser.add_argument("--release-id", required=True)
    release_parser.add_argument("--version-id", action="append", required=True)
    release_parser.add_argument("--database", type=Path, default=default_supervision / "industrial_rag.sqlite3")
    release_parser.add_argument("--objects", type=Path, default=default_supervision / "objects")
    release_parser.add_argument("--actor-id", default="local-operator")
    release_parser.add_argument("--reason", required=True)

    status_parser = subparsers.add_parser("status", help="read local corpus, backend and quarantine state")
    status_parser.add_argument("--database", type=Path, default=default_supervision / "industrial_rag.sqlite3")
    status_parser.add_argument("--objects", type=Path, default=default_supervision / "objects")

    rollback_parser = subparsers.add_parser("rollback", help="switch the isolated corpus pointer to a published release or seed")
    rollback_parser.add_argument("--release-id", required=True)
    rollback_parser.add_argument("--database", type=Path, default=default_supervision / "industrial_rag.sqlite3")
    rollback_parser.add_argument("--objects", type=Path, default=default_supervision / "objects")
    rollback_parser.add_argument("--actor-id", default="local-operator")
    rollback_parser.add_argument("--reason", required=True)

    args = parser.parse_args()
    if args.command == "serve":
        serve(database_path=args.database, object_root=args.objects, host=args.host, port=args.port,
              principal_id=args.principal_id, time_mode=args.time_mode)
        return
    if args.command == "ingest":
        result = ingest_file(source_path=args.source, metadata_path=args.metadata, database_path=args.database,
                             object_root=args.objects, scratch_root=args.scratch)
    elif args.command == "release":
        result = build_release(release_id=args.release_id, version_ids=args.version_id,
                               database_path=args.database, object_root=args.objects,
                               actor_id=args.actor_id, reason=args.reason)
    elif args.command == "rollback":
        store = _store(args.database, args.objects)
        result = store.rollback_release(args.release_id, actor_id=args.actor_id, reason=args.reason)
    else:
        store = _store(args.database, args.objects)
        result = {"backends": store.backend_status(), "release": store.release_summaries(), "quarantine": store.quarantine_summary()}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
