"""PostgreSQL adapter for the same financial RAG store contract as the SQLite adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from adaptive_mas.industrial_rag.repository import IndustrialRagStore


_SQLITE_JSON_EXTRACT = re.compile(
    r"(?is)json_extract\(\s*([A-Za-z_][A-Za-z0-9_.]*)\s*,\s*"
    r"'(\$\.[A-Za-z_][A-Za-z0-9_.]*)'\s*\)"
)


def _translate_json_extract(match: re.Match[str]) -> str:
    column, json_path = match.groups()
    path_parts = json_path[2:].split(".")
    return f"({column}::jsonb #>> '{{{','.join(path_parts)}}}')"


class _HybridRow(dict[str, Any]):
    def __init__(self, names: tuple[str, ...], values: tuple[Any, ...]) -> None:
        super().__init__(zip(names, values, strict=True))
        self._names = names
        self._values = values

    def __getitem__(self, key: object) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)


def _row_factory(cursor: Any) -> Any:
    # Psycopg calls the factory before executing a statement. DDL and writes
    # without RETURNING have no tuple description, and SELECT descriptions are
    # available only after execution, so resolve the names for each returned row.
    def make_row(values: Any) -> Any:
        description = cursor.description
        if description is None:
            return tuple(values)
        names = tuple(column.name for column in description)
        return _HybridRow(names, tuple(values))

    return make_row


def translate_sqlite_contract(sql: str) -> str:
    """Translate the small, audited SQL portability subset used by the shared store."""
    stripped = sql.strip()
    if re.match(r"(?is)^BEGIN\s+IMMEDIATE\s*;?$", stripped):
        stripped = "BEGIN ISOLATION LEVEL SERIALIZABLE"
    elif re.match(r"(?is)^SELECT\s+\*\s+FROM\s+rag_outbox\s+WHERE\s+status='pending'", stripped):
        stripped = stripped.rstrip(";") + " FOR UPDATE SKIP LOCKED"
    elif re.match(r"(?is)^INSERT\s+OR\s+IGNORE\s+INTO\s+", stripped):
        stripped = re.sub(r"(?is)^INSERT\s+OR\s+IGNORE\s+INTO\s+", "INSERT INTO ", stripped)
        stripped = stripped.rstrip(";") + " ON CONFLICT DO NOTHING"
    elif re.match(r"(?is)^INSERT\s+OR\s+REPLACE\s+INTO\s+rag_query_cache\s+VALUES\s*\(\s*\?\s*,\s*\?\s*,\s*\?\s*\)", stripped):
        stripped = (
            "INSERT INTO rag_query_cache VALUES(?, ?, ?) ON CONFLICT(cache_key) DO UPDATE "
            "SET payload_json=EXCLUDED.payload_json, created_at=EXCLUDED.created_at"
        )
    else:
        stripped = re.sub(r"(?is)^INSERT\s+OR\s+REPLACE\s+INTO\s+", "INSERT INTO ", stripped)
    stripped = _SQLITE_JSON_EXTRACT.sub(_translate_json_extract, stripped)
    return stripped.replace("?", "%s")


class _PostgresConnection:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def execute(self, sql: str, parameters: Any = ()) -> Any:
        return self._connection.execute(translate_sqlite_contract(sql), parameters)

    def executemany(self, sql: str, parameters: Any) -> Any:
        return self._connection.cursor().executemany(translate_sqlite_contract(sql), parameters)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "_PostgresConnection":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None:
            self.rollback()
        self.close()


class PostgresIndustrialRagStore(IndustrialRagStore):
    """PostgreSQL metadata authority; use SQLite only for local single-user deployments."""

    def __init__(self, dsn: str, object_store: Any) -> None:
        if not dsn.strip():
            raise ValueError("PostgreSQL DSN is required")
        self.dsn = dsn
        self.object_store = object_store
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("PostgreSQL adapter unavailable: install the pinned psycopg 3 dependency") from error
        self._psycopg = psycopg
        self.database_path = Path(":postgresql:")
        self._migrate()

    def _connect(self) -> _PostgresConnection:
        try:
            raw = self._psycopg.connect(self.dsn, autocommit=True, row_factory=_row_factory)
        except Exception:
            raise RuntimeError("PostgreSQL connection failed; connection details are omitted from diagnostics") from None
        return _PostgresConnection(raw)

    def _migrate(self) -> None:
        path = Path(__file__).parent / "migrations" / "0001_industrial_rag.sql"
        statements = [item.strip() for item in path.read_text(encoding="utf-8").split(";") if item.strip()]
        with self._connect() as connection:  # type: ignore[attr-defined]
            for statement in statements:
                connection.execute(statement)

    def backend_status(self) -> dict[str, object]:
        with self._connect() as connection:  # type: ignore[attr-defined]
            row = connection.execute("SELECT current_setting('server_version') AS server_version").fetchone()
        return {
            "metadata_backend": "postgresql",
            "metadata_backend_role": "authoritative metadata adapter",
            "server_version": str(row["server_version"]) if row is not None else "unknown",
            "postgres_connection_verified": row is not None,
            "postgres_contract_suite_verified": False,
            "object_store": type(self.object_store).__name__,
        }
