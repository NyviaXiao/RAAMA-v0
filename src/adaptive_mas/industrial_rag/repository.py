"""SQLite local adapter for the canonical financial RAG data contract."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from adaptive_mas.industrial_rag.models import (
    Block, CorpusReleaseSpec, DocumentBundle, IndexBuildReceipt, ContractError, parse_datetime,
    validate_identifier, validate_security_code, validate_sha256,
)
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore, ObjectStore, ObjectStoreError
from adaptive_mas.research_provenance import stable_hash


class ImmutableConflict(RuntimeError):
    """An immutable source/version identity was reused with changed bytes or metadata."""


class ReleaseNotReady(RuntimeError):
    """A corpus release has one or more required backends that are not ready."""


_SQLITE_SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS rag_documents (
  document_id TEXT PRIMARY KEY, security_code TEXT, entity_name TEXT, title TEXT NOT NULL,
  document_type TEXT NOT NULL, report_period TEXT, accounting_basis TEXT, aliases_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_document_versions (
  version_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES rag_documents(document_id),
  source_id TEXT NOT NULL UNIQUE, content_sha256 TEXT NOT NULL, source_url TEXT NOT NULL,
  rights_note TEXT NOT NULL, published_at TEXT, first_available_at TEXT, captured_at TEXT NOT NULL,
  ingested_at TEXT NOT NULL, valid_from TEXT, valid_to TEXT, revised_at TEXT,
  time_basis TEXT NOT NULL, parser_name TEXT NOT NULL, parser_version TEXT NOT NULL,
  parser_confidence REAL, parser_status TEXT NOT NULL, parser_error_code TEXT, parser_error TEXT, metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS rag_versions_lookup_idx ON rag_document_versions(document_id, first_available_at);
CREATE TABLE IF NOT EXISTS rag_sections (
  section_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  title TEXT NOT NULL, level INTEGER NOT NULL, parent_section_id TEXT, start_page INTEGER, end_page INTEGER, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_spans (
  span_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  page_number INTEGER, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, text TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_blocks (
  block_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  section_id TEXT, page_number INTEGER, block_type TEXT NOT NULL, text TEXT NOT NULL,
  parent_block_id TEXT, table_id TEXT, row_start INTEGER, row_end INTEGER, parser_confidence REAL,
  quarantine_reason TEXT, payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS rag_blocks_version_page_idx ON rag_blocks(version_id, page_number);
CREATE TABLE IF NOT EXISTS rag_tables (
  table_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  title TEXT, header_rows INTEGER NOT NULL, continued_from_table_id TEXT, continuation_confidence REAL,
  unit_text TEXT, accounting_basis TEXT, report_period TEXT, page_start INTEGER, page_end INTEGER, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_cells (
  cell_id TEXT PRIMARY KEY, table_id TEXT NOT NULL REFERENCES rag_tables(table_id), version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  row_index INTEGER NOT NULL, column_index INTEGER NOT NULL, raw_text TEXT NOT NULL, normalized_value TEXT,
  value_kind TEXT NOT NULL, page_number INTEGER, unit TEXT, multiplier TEXT, currency TEXT, accounting_basis TEXT,
  report_period TEXT, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_facts (
  fact_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  security_code TEXT NOT NULL, entity_name TEXT NOT NULL, metric TEXT NOT NULL, raw_value TEXT NOT NULL,
  normalized_value TEXT NOT NULL, unit TEXT NOT NULL, multiplier TEXT NOT NULL, currency TEXT NOT NULL,
  report_period TEXT NOT NULL, consolidation TEXT NOT NULL, confidence REAL NOT NULL, review_status TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS rag_facts_lookup_idx ON rag_facts(security_code, metric, report_period);
CREATE TABLE IF NOT EXISTS rag_quarantine (
  quarantine_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  error_code TEXT NOT NULL, reason TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'open',
  reviewed_by TEXT, reviewed_at TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_access_grants (
  principal_id TEXT NOT NULL, document_id TEXT NOT NULL REFERENCES rag_documents(document_id),
  can_read INTEGER NOT NULL, changed_at TEXT NOT NULL, PRIMARY KEY(principal_id, document_id)
);
CREATE TABLE IF NOT EXISTS rag_access_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL);
INSERT OR IGNORE INTO rag_access_state(singleton, revision) VALUES(1,0);
CREATE TABLE IF NOT EXISTS rag_outbox (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, event_key TEXT NOT NULL UNIQUE, event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
  consumer_id TEXT, lease_until REAL, created_at TEXT NOT NULL, acknowledged_at TEXT
);
CREATE TABLE IF NOT EXISTS rag_consumer_receipts (
  consumer_id TEXT NOT NULL, event_key TEXT NOT NULL, result_json TEXT NOT NULL, completed_at TEXT NOT NULL,
  PRIMARY KEY(consumer_id, event_key)
);
CREATE TABLE IF NOT EXISTS rag_corpus_releases (
  release_id TEXT PRIMARY KEY, spec_json TEXT NOT NULL, spec_sha256 TEXT NOT NULL,
  status TEXT NOT NULL, created_at TEXT NOT NULL, published_at TEXT
);
CREATE TABLE IF NOT EXISTS rag_release_versions (
  release_id TEXT NOT NULL REFERENCES rag_corpus_releases(release_id),
  version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  PRIMARY KEY(release_id, version_id)
);
CREATE TABLE IF NOT EXISTS rag_release_backends (
  release_id TEXT NOT NULL REFERENCES rag_corpus_releases(release_id), backend TEXT NOT NULL,
  physical_index_handle TEXT NOT NULL, backend_version TEXT NOT NULL,
  source_manifest_sha256 TEXT NOT NULL, index_manifest_sha256 TEXT NOT NULL,
  document_count INTEGER NOT NULL, record_count INTEGER NOT NULL,
  embedding_model TEXT NOT NULL, embedding_dimension INTEGER, ready_at TEXT NOT NULL,
  PRIMARY KEY(release_id, backend)
);
CREATE TABLE IF NOT EXISTS rag_active_release (singleton INTEGER PRIMARY KEY CHECK(singleton=1), release_id TEXT REFERENCES rag_corpus_releases(release_id), updated_at TEXT NOT NULL);
INSERT OR IGNORE INTO rag_active_release(singleton, release_id, updated_at) VALUES(1,NULL,'1970-01-01T00:00:00+00:00');
CREATE TABLE IF NOT EXISTS rag_release_audit (
  audit_id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL, release_id TEXT NOT NULL,
  previous_release_id TEXT, reason TEXT NOT NULL, actor_id TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_query_cache (
  cache_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload(value: object) -> dict[str, object]:
    return asdict(value)  # type: ignore[arg-type,return-value]


def _normalise_record(value: object) -> dict[str, object]:
    result = _payload(value)
    for key, item in list(result.items()):
        if isinstance(item, datetime):
            result[key] = item.astimezone(timezone.utc).isoformat()
    return result


class IndustrialRagStore:
    """Authoritative local SQLite adapter; metadata and object bytes are independently verified."""

    def __init__(self, database_path: str | Path, object_store: ObjectStore | None = None) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.object_store = object_store or LocalContentAddressedStore(self.database_path.parent / "objects")
        with closing(self._connect()) as connection:
            connection.executescript(_SQLITE_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _record_hash(value: object) -> str:
        # Hash the same UTC-normalized mapping that is persisted as JSON so a
        # release receipt survives a JSON round trip without changing meaning.
        return stable_hash(_normalise_record(value))

    def ingest_bundle(self, bundle: DocumentBundle) -> dict[str, object]:
        document, version = bundle.document, bundle.version
        if not bundle.payload:
            raise ValueError("document bytes are required")
        payload_sha = hashlib.sha256(bundle.payload).hexdigest()
        if payload_sha != version.content_sha256:
            raise ImmutableConflict("document payload SHA-256 does not match version metadata")
        if any(item.version_id != version.version_id for group in (bundle.sections, bundle.spans, bundle.blocks, bundle.tables, bundle.facts) for item in group):
            raise ContractError("every section, span, block, table, and fact must bind to this document version")
        if any(cell.version_id != version.version_id for cell in bundle.cells):
            raise ContractError("every table cell must bind to this document version")
        if any(cell.table_id not in {table.table_id for table in bundle.tables} for cell in bundle.cells):
            raise ContractError("table cell references an absent table")
        if any(fact.security_code != document.security_code for fact in bundle.facts):
            raise ContractError("fact security code must match its parent document")
        if any(fact.entity_name != document.entity_name for fact in bundle.facts):
            raise ContractError("fact entity must match its parent document")
        if any(fact.consolidation != document.accounting_basis for fact in bundle.facts):
            raise ContractError("fact accounting basis must match its parent document")
        if version.parser_status != "parsed" and any((bundle.blocks, bundle.sections, bundle.spans, bundle.tables, bundle.cells, bundle.facts)):
            raise ContractError("quarantined or unavailable versions cannot persist searchable content")
        span_ids = {item.span_id for item in bundle.spans}
        if any(block.span_id is not None and block.span_id not in span_ids for block in bundle.blocks):
            raise ContractError("block source span reference is missing")
        spans_by_id = {item.span_id: item for item in bundle.spans}
        for fact in bundle.facts:
            for span_id in fact.span_ids:
                span = spans_by_id.get(span_id)
                if (
                    span is None
                    or span.version_id != version.version_id
                    or span.page_number is None
                    or not span.text.strip()
                    or span.char_end <= span.char_start
                ):
                    raise ContractError("financial fact must cite an existing, nonempty, locatable span from the same version")
        object_hash = self.object_store.put(bundle.payload)
        if object_hash != version.content_sha256:
            raise ObjectStoreError("object store returned a different content hash")
        now = _now()
        metadata = _normalise_record(document)
        version_data = _normalise_record(version)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM rag_document_versions WHERE source_id=? OR version_id=?",
                    (version.source_id, version.version_id),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["source_id"] != version.source_id
                        or existing["version_id"] != version.version_id
                        or existing["document_id"] != document.document_id
                        or existing["content_sha256"] != version.content_sha256
                        or existing["metadata_json"] != _json(version_data)
                    ):
                        raise ImmutableConflict("source/version identity was reused with changed bytes or metadata")
                    prior_document = connection.execute(
                        "SELECT metadata_json FROM rag_documents WHERE document_id=?",
                        (document.document_id,),
                    ).fetchone()
                    if prior_document is None or prior_document["metadata_json"] != _json(metadata):
                        raise ImmutableConflict("document identity was reused with changed metadata")
                    self._verify_existing_children(connection, bundle)
                    connection.commit()
                    return {"status": "reused", "document_id": document.document_id, "version_id": version.version_id, "content_sha256": version.content_sha256}
                prior_document = connection.execute("SELECT * FROM rag_documents WHERE document_id=?", (document.document_id,)).fetchone()
                if prior_document is None:
                    connection.execute(
                        "INSERT INTO rag_documents VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (document.document_id, document.security_code, document.entity_name, document.title,
                         document.document_type, document.report_period, document.accounting_basis,
                         _json(document.aliases), _json(metadata), now),
                    )
                elif prior_document["security_code"] != document.security_code:
                    raise ImmutableConflict("document identity cannot change security code")
                connection.execute(
                    "INSERT INTO rag_document_versions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (version.version_id, document.document_id, version.source_id, version.content_sha256,
                     version.source_url, version.rights_note, version_data["published_at"],
                     version_data["first_available_at"], version_data["captured_at"], version_data["ingested_at"],
                     version_data["valid_from"], version_data["valid_to"], version_data["revised_at"],
                     version.time_basis, version.parser_name, version.parser_version, version.parser_confidence,
                     version.parser_status, version.parser_error_code, version.parser_error,
                     _json(version_data)),
                )
                self._insert_children(connection, bundle)
                if version.parser_status == "quarantined":
                    if bundle.facts:
                        raise ContractError("quarantined document cannot contain usable financial facts")
                    connection.execute(
                        "INSERT INTO rag_quarantine VALUES(?,?,?,?,'open',NULL,NULL,?)",
                        (f"q-{version.version_id}", version.version_id, version.parser_error_code, version.parser_error or "parser did not provide a reason", now),
                    )
                changed = False
                for principal in sorted(set(bundle.read_principals)):
                    if not principal or len(principal) > 128:
                        raise ContractError("read principal is empty or too long")
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO rag_access_grants VALUES(?,?,1,?)",
                        (principal, document.document_id, now),
                    )
                    changed = changed or cursor.rowcount > 0
                if changed:
                    connection.execute("UPDATE rag_access_state SET revision=revision+1 WHERE singleton=1")
                event_key = f"rag.document.ingested:{version.version_id}:{version.content_sha256}"
                event_payload = {"document_id": document.document_id, "version_id": version.version_id,
                                 "source_id": version.source_id, "content_sha256": version.content_sha256}
                connection.execute(
                    "INSERT INTO rag_outbox(event_key,event_type,payload_json,status,attempts,created_at) VALUES(?,?,?,'pending',0,?)",
                    (event_key, "document.quarantined" if version.parser_status == "quarantined" else "document.ingested", _json(event_payload), now),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {"status": "ingested", "document_id": document.document_id, "version_id": version.version_id, "content_sha256": version.content_sha256, "outbox_event_key": event_key}

    @staticmethod
    def _verify_existing_children(connection: sqlite3.Connection, bundle: DocumentBundle) -> None:
        child_tables = (
            ("rag_sections", "section_id", bundle.sections),
            ("rag_spans", "span_id", bundle.spans),
            ("rag_blocks", "block_id", bundle.blocks),
            ("rag_tables", "table_id", bundle.tables),
            ("rag_cells", "cell_id", bundle.cells),
            ("rag_facts", "fact_id", bundle.facts),
        )
        for table, id_column, incoming in child_tables:
            stored = {
                str(row[id_column]): str(row["payload_json"])
                for row in connection.execute(
                    f"SELECT {id_column},payload_json FROM {table} WHERE version_id=?",
                    (bundle.version.version_id,),
                ).fetchall()
            }
            expected = {
                str(getattr(item, id_column.removesuffix("_id") + "_id")): _json(_normalise_record(item))
                for item in incoming
            }
            if stored != expected:
                raise ImmutableConflict("source/version identity was reused with changed parsed child records")

    @staticmethod
    def _insert_children(connection: sqlite3.Connection, bundle: DocumentBundle) -> None:
        for section in bundle.sections:
            payload = _normalise_record(section)
            connection.execute("INSERT INTO rag_sections VALUES(?,?,?,?,?,?,?,?)", (
                section.section_id, section.version_id, section.title, section.level, section.parent_section_id,
                section.start_page, section.end_page, _json(payload),
            ))
        for span in bundle.spans:
            payload = _normalise_record(span)
            connection.execute("INSERT INTO rag_spans VALUES(?,?,?,?,?,?,?)", (
                span.span_id, span.version_id, span.page_number, span.char_start, span.char_end, span.text, _json(payload),
            ))
        for block in bundle.blocks:
            payload = _normalise_record(block)
            connection.execute("INSERT INTO rag_blocks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                block.block_id, block.version_id, block.section_id, block.page_number, block.block_type, block.text,
                block.parent_block_id, block.table_id, block.row_start, block.row_end,
                block.parser_confidence, block.quarantine_reason, _json(payload),
            ))
        for table in bundle.tables:
            payload = _normalise_record(table)
            connection.execute("INSERT INTO rag_tables VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                table.table_id, table.version_id, table.title, table.header_rows, table.continued_from_table_id,
                table.continuation_confidence, table.unit_text, table.accounting_basis, table.report_period,
                table.page_start, table.page_end, _json(payload),
            ))
        for cell in bundle.cells:
            payload = _normalise_record(cell)
            connection.execute("INSERT INTO rag_cells VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                cell.cell_id, cell.table_id, cell.version_id, cell.row_index, cell.column_index,
                cell.raw_text, cell.normalized_value, cell.value_kind, cell.page_number, cell.unit,
                cell.multiplier, cell.currency, cell.accounting_basis, cell.report_period, _json(payload),
            ))
        for fact in bundle.facts:
            payload = _normalise_record(fact)
            connection.execute("INSERT INTO rag_facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                fact.fact_id, fact.version_id, fact.security_code, fact.entity_name, fact.metric, fact.raw_value,
                fact.normalized_value, fact.unit, fact.multiplier, fact.currency, fact.report_period,
                fact.consolidation, fact.confidence, fact.review_status, _json(payload),
            ))

    def permission_revision(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT revision FROM rag_access_state WHERE singleton=1").fetchone()
        return int(row["revision"])

    def set_access(self, principal_id: str, document_id: str, *, can_read: bool) -> int:
        if not principal_id or len(principal_id) > 128:
            raise ValueError("principal_id must contain 1..128 characters")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute("SELECT 1 FROM rag_documents WHERE document_id=?", (document_id,)).fetchone()
            if exists is None:
                connection.rollback()
                raise FileNotFoundError("document not found")
            current = connection.execute("SELECT can_read FROM rag_access_grants WHERE principal_id=? AND document_id=?", (principal_id, document_id)).fetchone()
            allowed = int(can_read)
            if current is None or int(current["can_read"]) != allowed:
                connection.execute(
                    "INSERT INTO rag_access_grants VALUES(?,?,?,?) ON CONFLICT(principal_id,document_id) DO UPDATE SET can_read=excluded.can_read,changed_at=excluded.changed_at",
                    (principal_id, document_id, allowed, _now()),
                )
                connection.execute("UPDATE rag_access_state SET revision=revision+1 WHERE singleton=1")
            revision = int(connection.execute("SELECT revision FROM rag_access_state WHERE singleton=1").fetchone()["revision"])
            connection.commit()
        return revision

    def permitted_versions(self, principal_id: str, security_code: str, as_of: datetime, *, time_mode: str, report_periods: Sequence[str] = (), accounting_basis: str | None = None, release_id: str | None = None) -> list[sqlite3.Row]:
        validate_security_code(security_code)
        as_of_utc = parse_datetime(as_of, "as_of", required=True).isoformat()
        if not release_id:
            raise ReleaseNotReady("query requires a published, immutable corpus release")
        sql = (
            "SELECT v.*,d.security_code,d.entity_name,d.title,d.document_type,d.report_period,d.accounting_basis,d.aliases_json "
            "FROM rag_document_versions v JOIN rag_documents d USING(document_id) "
            "JOIN rag_access_grants g ON g.document_id=d.document_id AND g.principal_id=? AND g.can_read=1 "
            "JOIN rag_release_versions rv ON rv.version_id=v.version_id AND rv.release_id=? "
            "WHERE d.security_code=? AND v.first_available_at IS NOT NULL AND v.first_available_at<=? "
            "AND (v.valid_from IS NULL OR v.valid_from<=?) AND (v.valid_to IS NULL OR v.valid_to>?)"
            " AND (v.revised_at IS NULL OR v.revised_at<=?)"
        )
        params: list[object] = [principal_id, release_id, security_code, as_of_utc, as_of_utc, as_of_utc, as_of_utc]
        if time_mode == "strict_forward":
            sql += " AND v.time_basis='strict_forward' AND v.captured_at<=? AND v.ingested_at<=?"
            params.extend([as_of_utc, as_of_utc])
        elif time_mode == "retrospective_diagnostic":
            sql += " AND ((v.time_basis='strict_forward' AND v.captured_at<=? AND v.ingested_at<=?) OR v.time_basis='retrospective_proxy')"
            params.extend([as_of_utc, as_of_utc])
        else:
            raise ValueError("unsupported time_mode")
        if report_periods:
            sql += " AND (d.report_period IN (" + ",".join("?" for _ in report_periods) + ") OR EXISTS (SELECT 1 FROM rag_facts f WHERE f.version_id=v.version_id AND f.report_period IN (" + ",".join("?" for _ in report_periods) + ")))"
            params.extend(report_periods)
            params.extend(report_periods)
        if accounting_basis:
            sql += " AND (d.accounting_basis=? OR EXISTS (SELECT 1 FROM rag_facts f WHERE f.version_id=v.version_id AND f.consolidation=?))"
            params.extend([accounting_basis, accounting_basis])
        sql += " ORDER BY v.first_available_at DESC, v.version_id"
        with closing(self._connect()) as connection:
            return list(connection.execute(sql, params).fetchall())

    def read_blocks(self, version_ids: Sequence[str]) -> list[sqlite3.Row]:
        if not version_ids:
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT b.*,v.source_id,v.source_url,v.published_at,v.first_available_at,v.captured_at,v.ingested_at,v.content_sha256,v.time_basis,d.document_id,d.title,d.security_code "
                "FROM rag_blocks b JOIN rag_document_versions v USING(version_id) JOIN rag_documents d USING(document_id) "
                "WHERE b.version_id IN (" + ",".join("?" for _ in version_ids) + ") AND b.quarantine_reason IS NULL",
                list(version_ids),
            ).fetchall()
        return list(rows)

    def read_facts(self, version_ids: Sequence[str]) -> list[sqlite3.Row]:
        if not version_ids:
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT f.*,v.source_id,v.source_url,v.published_at,v.first_available_at,v.captured_at,v.ingested_at,v.content_sha256,v.time_basis,d.document_id,d.title "
                "FROM rag_facts f JOIN rag_document_versions v USING(version_id) JOIN rag_documents d USING(document_id) "
                "WHERE f.version_id IN (" + ",".join("?" for _ in version_ids) + ") AND f.review_status!='rejected'",
                list(version_ids),
            ).fetchall()
        return list(rows)

    def _get_span(self, span_id: str) -> dict[str, object] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT s.*,v.source_id,v.source_url,v.published_at,v.first_available_at,v.captured_at,v.ingested_at,v.content_sha256,v.time_basis,d.document_id,d.title,d.security_code "
                "FROM rag_spans s JOIN rag_document_versions v USING(version_id) JOIN rag_documents d USING(document_id) WHERE span_id=?",
                (span_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def read_spans(self, span_ids: Sequence[str]) -> list[dict[str, object]]:
        if not span_ids:
            return []
        unique_ids = list(dict.fromkeys(span_ids))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT s.*,v.source_id,v.source_url,v.published_at,v.first_available_at,v.captured_at,v.ingested_at,v.content_sha256,v.time_basis,d.document_id,d.title,d.security_code "
                "FROM rag_spans s JOIN rag_document_versions v USING(version_id) JOIN rag_documents d USING(document_id) "
                "WHERE s.span_id IN (" + ",".join("?" for _ in unique_ids) + ")",
                unique_ids,
            ).fetchall()
        return [dict(row) for row in rows]

    def _get_document_bytes(self, document_id: str, version_id: str, principal_id: str) -> bytes:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT v.content_sha256 FROM rag_document_versions v JOIN rag_access_grants g USING(document_id) "
                "WHERE v.document_id=? AND v.version_id=? AND g.principal_id=? AND g.can_read=1",
                (document_id, version_id, principal_id),
            ).fetchone()
        if row is None:
            raise PermissionError("document is not currently readable by this principal")
        return self.object_store.get(str(row["content_sha256"]))

    def active_release_id(self) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT release_id FROM rag_active_release WHERE singleton=1").fetchone()
        return str(row["release_id"]) if row and row["release_id"] else None

    def source_manifest(self, version_ids: Sequence[str]) -> list[dict[str, object]]:
        if not version_ids or len(version_ids) != len(set(version_ids)):
            raise ValueError("source manifest requires a nonempty unique version list")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT version_id,source_id,content_sha256,time_basis,parser_name,parser_version,parser_status FROM rag_document_versions WHERE version_id IN (" + ",".join("?" for _ in version_ids) + ") ORDER BY version_id",
                list(version_ids),
            ).fetchall()
        manifest = [dict(row) for row in rows]
        if {str(row["version_id"]) for row in manifest} != set(version_ids):
            raise FileNotFoundError("source manifest refers to an unregistered document version")
        return manifest

    def create_release(self, spec: CorpusReleaseSpec, version_ids: Sequence[str]) -> dict[str, object]:
        if not version_ids or len(version_ids) != len(set(version_ids)):
            raise ValueError("release must bind a nonempty unique version list")
        with closing(self._connect()) as connection:
            existing = connection.execute("SELECT * FROM rag_corpus_releases WHERE release_id=?", (spec.release_id,)).fetchone()
            if existing is not None:
                if existing["spec_sha256"] != self._record_hash(spec):
                    raise ImmutableConflict("release ID was reused with changed specification")
                bound = [row[0] for row in connection.execute("SELECT version_id FROM rag_release_versions WHERE release_id=? ORDER BY version_id", (spec.release_id,))]
                if bound != sorted(version_ids):
                    raise ImmutableConflict("release ID was reused with changed version manifest")
                return {"release_id": spec.release_id, "status": existing["status"], "reused": True}
            found = {row[0] for row in connection.execute("SELECT version_id FROM rag_document_versions WHERE version_id IN (" + ",".join("?" for _ in version_ids) + ")", list(version_ids))}
            if found != set(version_ids):
                raise FileNotFoundError("release refers to an unregistered document version")
            actual_sources = self.source_manifest(version_ids)
            if any(row["parser_status"] != "parsed" for row in actual_sources):
                raise ReleaseNotReady("quarantined or unavailable documents cannot enter a corpus release")
            if any(row["parser_confidence"] is None or float(row["parser_confidence"]) < 0.6 for row in connection.execute(
                "SELECT parser_confidence FROM rag_document_versions WHERE version_id IN (" + ",".join("?" for _ in version_ids) + ")",
                list(version_ids),
            ).fetchall()):
                raise ReleaseNotReady("documents without a sufficient recorded parser quality signal cannot enter a release")
            actual_manifest_hash = stable_hash(actual_sources)
            if actual_manifest_hash != spec.source_manifest_sha256:
                raise ImmutableConflict("source manifest hash does not match the selected immutable document versions")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO rag_corpus_releases VALUES(?,?,?,'building',?,NULL)",
                    (spec.release_id, _json(_normalise_record(spec)), self._record_hash(spec), _now()),
                )
                connection.executemany(
                    "INSERT INTO rag_release_versions VALUES(?,?)",
                    [(spec.release_id, version_id) for version_id in sorted(version_ids)],
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {"release_id": spec.release_id, "status": "building", "reused": False, "source_manifest_sha256": actual_manifest_hash}

    def release_spec(self, release_id: str) -> tuple[dict[str, object], str]:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT spec_json,spec_sha256 FROM rag_corpus_releases WHERE release_id=?", (release_id,)).fetchone()
        if row is None:
            raise FileNotFoundError("corpus release not found")
        spec = json.loads(row["spec_json"])
        if stable_hash(spec) != row["spec_sha256"]:
            raise ImmutableConflict("corpus release specification integrity check failed")
        return spec, str(row["spec_sha256"])

    def mark_backend_ready(
        self,
        receipt: IndexBuildReceipt,
        verifier: Callable[[str], Mapping[str, object]],
    ) -> dict[str, object]:
        release_id, backend = receipt.release_id, receipt.backend
        spec, _ = self.release_spec(release_id)
        required = set(spec["required_backends"])
        if backend not in required:
            raise ValueError("backend is not part of the immutable release requirement")
        expected_manifests = {
            "text": spec["text_index_manifest_sha256"],
            "text_index": spec["text_index_manifest_sha256"],
            "opensearch": spec["text_index_manifest_sha256"],
            "sqlite_reference": spec["text_index_manifest_sha256"],
            "vector": spec["vector_index_manifest_sha256"],
            "vector_index": spec["vector_index_manifest_sha256"],
            "qdrant": spec["vector_index_manifest_sha256"],
        }
        expected_manifest = expected_manifests.get(backend)
        if expected_manifest is None:
            raise ValueError("backend must map to a declared text/vector index manifest")
        if receipt.index_manifest_sha256 != expected_manifest:
            raise ImmutableConflict("backend receipt does not match the release-bound physical index manifest")
        if receipt.source_manifest_sha256 != spec["source_manifest_sha256"]:
            raise ImmutableConflict("backend receipt is built from a different source manifest")
        try:
            actual = dict(verifier(receipt.physical_index_handle))
        except Exception as error:
            raise ImmutableConflict("backend receipt does not match a verifiable release-bound physical index") from error
        expected = {
            "backend": receipt.backend,
            "physical_index_handle": receipt.physical_index_handle,
            "backend_version": receipt.backend_version,
            "source_manifest_sha256": receipt.source_manifest_sha256,
            "index_manifest_sha256": receipt.index_manifest_sha256,
            "document_count": receipt.document_count,
            "record_count": receipt.record_count,
            "embedding_model": receipt.embedding_model,
            "embedding_dimension": receipt.embedding_dimension,
        }
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ImmutableConflict("backend verification did not match its immutable build receipt")
        expected_model = spec["embedding_model"]
        expected_dimension = spec["embedding_dimension"]
        if backend in {"vector", "vector_index", "qdrant"} and (
            receipt.embedding_model != expected_model or receipt.embedding_dimension != expected_dimension
        ):
            raise ImmutableConflict("vector index model or dimension differs from the corpus release")
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT status FROM rag_corpus_releases WHERE release_id=?", (release_id,)).fetchone()
            if row is None:
                raise FileNotFoundError("corpus release not found")
            if row["status"] == "published":
                raise ValueError("published release backend manifests are immutable")
            prior = connection.execute("SELECT index_manifest_sha256,physical_index_handle FROM rag_release_backends WHERE release_id=? AND backend=?", (release_id, backend)).fetchone()
            if prior and (prior["index_manifest_sha256"] != receipt.index_manifest_sha256 or prior["physical_index_handle"] != receipt.physical_index_handle):
                raise ImmutableConflict("backend readiness cannot be changed after recording")
            connection.execute(
                "INSERT OR IGNORE INTO rag_release_backends VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (release_id, backend, receipt.physical_index_handle, receipt.backend_version,
                 receipt.source_manifest_sha256, receipt.index_manifest_sha256, receipt.document_count,
                 receipt.record_count, receipt.embedding_model, receipt.embedding_dimension, _now()),
            )
            ready = [r[0] for r in connection.execute("SELECT backend FROM rag_release_backends WHERE release_id=?", (release_id,))]
        return {"release_id": release_id, "backend": backend, "physical_index_handle": receipt.physical_index_handle, "index_manifest_sha256": receipt.index_manifest_sha256, "ready": True, "missing_backends": sorted(required - set(ready))}

    def preview_reference_index_manifest(self, release_id: str, version_ids: Sequence[str]) -> dict[str, object]:
        """Hash actual permitted block rows for the explicit local reference backend."""
        if not version_ids or len(version_ids) != len(set(version_ids)):
            raise ValueError("reference index requires a nonempty unique version list")
        source_manifest = self.source_manifest(version_ids)
        if any(row["parser_status"] != "parsed" for row in source_manifest):
            raise ReleaseNotReady("quarantined or unavailable documents cannot be indexed")
        version_set = set(version_ids)
        placeholders = ",".join("?" for _ in version_ids)
        with closing(self._connect()) as connection:
            document_rows = connection.execute(
                f"SELECT DISTINCT document_id FROM rag_document_versions WHERE version_id IN ({placeholders})", list(version_ids)
            ).fetchall()
            block_rows = connection.execute(
                f"SELECT block_id,version_id,block_type,page_number,text FROM rag_blocks WHERE version_id IN ({placeholders}) AND quarantine_reason IS NULL ORDER BY version_id,page_number,block_id",
                list(version_ids),
            ).fetchall()
        if {str(row["version_id"]) for row in source_manifest} != version_set:
            raise FileNotFoundError("reference index version list is incomplete")
        source_hash = stable_hash(source_manifest)
        block_manifest = [
            {"block_id": row["block_id"], "version_id": row["version_id"], "block_type": row["block_type"],
             "page_number": row["page_number"], "text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()}
            for row in block_rows
        ]
        index_data = {
            "release_id": release_id, "backend": "sqlite_reference",
            "backend_version": sqlite3.sqlite_version, "source_manifest_sha256": source_hash,
            "document_count": len(document_rows), "record_count": len(block_rows),
            "block_manifest_sha256": stable_hash(block_manifest),
            "embedding_model": "none", "embedding_dimension": None,
        }
        return {**index_data, "index_manifest_sha256": stable_hash(index_data)}

    def build_reference_index_receipt(self, release_id: str) -> IndexBuildReceipt:
        versions = self.release_versions(release_id)
        data = self.preview_reference_index_manifest(release_id, versions)
        spec, _ = self.release_spec(release_id)
        if data["index_manifest_sha256"] != spec["text_index_manifest_sha256"]:
            raise ImmutableConflict("reference text index differs from the corpus release's expected manifest")
        return IndexBuildReceipt(
            release_id=release_id, backend="sqlite_reference",
            physical_index_handle=f"sqlite-reference:{release_id}", backend_version=str(data["backend_version"]),
            source_manifest_sha256=str(data["source_manifest_sha256"]),
            index_manifest_sha256=str(data["index_manifest_sha256"]),
            document_count=int(data["document_count"]), record_count=int(data["record_count"]),
            embedding_model="none", embedding_dimension=None, built_at=_now(),
        )

    def verify_reference_index(self, physical_index_handle: str) -> Mapping[str, object]:
        prefix = "sqlite-reference:"
        if not physical_index_handle.startswith(prefix):
            raise ValueError("unknown local reference index handle")
        release_id = physical_index_handle.removeprefix(prefix)
        versions = self.release_versions(release_id)
        data = self.preview_reference_index_manifest(release_id, versions)
        return {**data, "physical_index_handle": physical_index_handle}

    def publish_release(self, release_id: str, *, actor_id: str, reason: str) -> dict[str, object]:
        if not reason.strip() or not actor_id.strip():
            raise ValueError("release publication requires an actor and reason")
        spec, spec_sha = self.release_spec(release_id)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                release = connection.execute("SELECT status,spec_sha256 FROM rag_corpus_releases WHERE release_id=?", (release_id,)).fetchone()
                if release is None or release["spec_sha256"] != spec_sha:
                    raise FileNotFoundError("corpus release not found or changed")
                ready = {row[0] for row in connection.execute("SELECT backend FROM rag_release_backends WHERE release_id=?", (release_id,))}
                missing = set(spec["required_backends"]) - ready
                if missing:
                    raise ReleaseNotReady(f"required index backends are not ready: {', '.join(sorted(missing))}")
                previous = connection.execute("SELECT release_id FROM rag_active_release WHERE singleton=1").fetchone()[0]
                connection.execute("UPDATE rag_corpus_releases SET status='published',published_at=? WHERE release_id=?", (_now(), release_id))
                connection.execute("UPDATE rag_active_release SET release_id=?,updated_at=? WHERE singleton=1", (release_id, _now()))
                connection.execute("INSERT INTO rag_release_audit(action,release_id,previous_release_id,reason,actor_id,created_at) VALUES('publish',?,?,?,?,?)", (release_id, previous, reason, actor_id, _now()))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {"release_id": release_id, "status": "published", "previous_release_id": previous, "spec_sha256": spec_sha}

    def rollback_release(self, release_id: str, *, actor_id: str, reason: str) -> dict[str, object]:
        if not reason.strip() or not actor_id.strip():
            raise ValueError("rollback requires an actor and reason")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                target = None if release_id == "seed" else connection.execute("SELECT status FROM rag_corpus_releases WHERE release_id=?", (release_id,)).fetchone()
                if release_id != "seed" and (target is None or target["status"] != "published"):
                    raise ValueError("rollback target must be an already published immutable corpus release or seed")
                previous = connection.execute("SELECT release_id FROM rag_active_release WHERE singleton=1").fetchone()[0]
                target_id = None if release_id == "seed" else release_id
                connection.execute("UPDATE rag_active_release SET release_id=?,updated_at=? WHERE singleton=1", (target_id, _now()))
                connection.execute("INSERT INTO rag_release_audit(action,release_id,previous_release_id,reason,actor_id,created_at) VALUES('rollback',?,?,?,?,?)", (release_id, previous, reason, actor_id, _now()))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {"active_release_id": release_id, "previous_release_id": previous, "status": "rolled_back"}

    def release_versions(self, release_id: str) -> list[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT version_id FROM rag_release_versions WHERE release_id=? ORDER BY version_id", (release_id,)).fetchall()
        return [str(row[0]) for row in rows]

    def release_summaries(self) -> dict[str, object]:
        with closing(self._connect()) as connection:
            releases = connection.execute("SELECT release_id,status,created_at,published_at,spec_json,spec_sha256 FROM rag_corpus_releases ORDER BY created_at DESC").fetchall()
            active = connection.execute("SELECT release_id FROM rag_active_release WHERE singleton=1").fetchone()[0]
            quarantine = connection.execute("SELECT COUNT(*) FROM rag_quarantine WHERE state='open'").fetchone()[0]
        summary = []
        for row in releases:
            spec = json.loads(row["spec_json"])
            with closing(self._connect()) as connection:
                ready = [str(item[0]) for item in connection.execute("SELECT backend FROM rag_release_backends WHERE release_id=? ORDER BY backend", (row["release_id"],))]
                versions = [str(item[0]) for item in connection.execute("SELECT version_id FROM rag_release_versions WHERE release_id=? ORDER BY version_id", (row["release_id"],))]
            summary.append({
                "release_id": row["release_id"], "status": row["status"],
                "created_at": row["created_at"], "published_at": row["published_at"],
                "active": row["release_id"] == active, "spec_sha256": row["spec_sha256"],
                "required_backends": spec["required_backends"], "ready_backends": ready,
                "missing_backends": sorted(set(spec["required_backends"]) - set(ready)),
                "version_count": len(versions), "versions": versions,
            })
        return {"active_release_id": active, "releases": summary, "open_quarantine_count": int(quarantine)}

    def quarantine_summary(self, *, limit: int = 100) -> dict[str, object]:
        if not 1 <= limit <= 500:
            raise ValueError("quarantine limit must be between 1 and 500")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT q.quarantine_id,q.version_id,q.error_code,q.reason,q.state,q.created_at,d.document_id,d.title,d.security_code,v.source_id,v.content_sha256,v.source_url "
                "FROM rag_quarantine q JOIN rag_document_versions v USING(version_id) JOIN rag_documents d USING(document_id) "
                "WHERE q.state='open' ORDER BY q.created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {"items": [dict(row) for row in rows], "count": len(rows), "limit": limit}

    def claim_outbox(self, consumer_id: str, *, lease_seconds: float = 60) -> dict[str, object] | None:
        if not consumer_id or lease_seconds <= 0:
            raise ValueError("consumer_id and positive lease_seconds are required")
        now_epoch = time.time()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM rag_outbox WHERE status='pending' OR (status='processing' AND lease_until<?) ORDER BY event_id LIMIT 1",
                (now_epoch,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute("UPDATE rag_outbox SET status='processing',consumer_id=?,lease_until=?,attempts=attempts+1 WHERE event_id=?", (consumer_id, now_epoch + lease_seconds, row["event_id"]))
            connection.commit()
        generation = int(row["attempts"]) + 1
        return {
            "event_id": int(row["event_id"]), "event_key": row["event_key"],
            "event_type": row["event_type"], "payload": json.loads(row["payload_json"]),
            "attempt": generation, "lease_generation": generation, "consumer_id": consumer_id,
        }

    def finish_outbox(
        self,
        event_key: str,
        consumer_id: str,
        result: Mapping[str, object],
        *,
        lease_generation: int,
    ) -> bool:
        if lease_generation < 1:
            raise ValueError("lease_generation must be positive")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM rag_outbox WHERE event_key=?", (event_key,)).fetchone()
            if row is None:
                connection.rollback()
                raise FileNotFoundError("outbox event not found")
            receipt = connection.execute("SELECT 1 FROM rag_consumer_receipts WHERE consumer_id=? AND event_key=?", (consumer_id, event_key)).fetchone()
            if receipt is not None and row["status"] == "done":
                connection.commit()
                return False
            if (
                row["status"] != "processing"
                or row["consumer_id"] != consumer_id
                or int(row["attempts"]) != lease_generation
                or row["lease_until"] is None
                or float(row["lease_until"]) < time.time()
            ):
                connection.rollback()
                raise RuntimeError("consumer no longer owns this outbox lease")
            connection.execute("INSERT OR IGNORE INTO rag_consumer_receipts VALUES(?,?,?,?)", (consumer_id, event_key, _json(dict(result)), _now()))
            connection.execute("UPDATE rag_outbox SET status='done',acknowledged_at=?,lease_until=NULL WHERE event_key=?", (_now(), event_key))
            connection.commit()
        return True

    def consume_outbox_once(self, consumer_id: str, handler: Callable[[str, str, Mapping[str, object]], Mapping[str, object]], *, lease_seconds: float = 60) -> dict[str, object] | None:
        event = self.claim_outbox(consumer_id, lease_seconds=lease_seconds)
        if event is None:
            return None
        output = handler(str(event["event_key"]), str(event["event_type"]), event["payload"])
        committed = self.finish_outbox(
            str(event["event_key"]), consumer_id, output,
            lease_generation=int(event["lease_generation"]),
        )
        return {"event_key": event["event_key"], "attempt": event["attempt"], "committed": committed}

    def cache_get(self, cache_key: str) -> dict[str, object] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT payload_json FROM rag_query_cache WHERE cache_key=?", (cache_key,)).fetchone()
        return json.loads(row[0]) if row is not None else None

    def cache_put(self, cache_key: str, payload: Mapping[str, object]) -> None:
        with closing(self._connect()) as connection:
            connection.execute("INSERT OR REPLACE INTO rag_query_cache VALUES(?,?,?)", (cache_key, _json(dict(payload)), _now()))

    def backend_status(self) -> dict[str, object]:
        try:
            import psycopg  # noqa: F401
            postgres_driver = "available"
        except ImportError:
            postgres_driver = "unavailable"
        try:
            import boto3  # noqa: F401
            s3_sdk = "available"
        except ImportError:
            s3_sdk = "unavailable"
        return {
            "metadata_backend": "sqlite",
            "metadata_backend_role": "local reference adapter; PostgreSQL is production authoritative target",
            "postgres_driver": postgres_driver,
            "s3_sdk": s3_sdk,
            "object_store": type(self.object_store).__name__,
            "postgres_connection_verified": False,
            "postgres_contract_suite_verified": False,
            "s3_connection_verified": False,
            "s3_contract_suite_verified": False,
        }
