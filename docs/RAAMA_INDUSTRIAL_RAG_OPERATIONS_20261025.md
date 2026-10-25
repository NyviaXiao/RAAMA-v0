# Industrial financial RAG local operations — 2026-10-25

This runbook starts the isolated local reference implementation. It does not configure the existing Harness ports or databases, connect to a provider, or create a production corpus.

## Install and inspect

From the repository root in PowerShell:

```powershell
uv sync --extra dev
uv run raama-rag status
```

The default state lives under `.tmp/industrial_rag_supervision_20261025/`: SQLite metadata, content-addressed source bytes, and parser scratch files. These paths are local diagnostics and are ignored by Git.

## Register a source

Create a metadata JSON file for each source. It must name a stable `document_id`, unique `version_id` and `source_id`, title, type, six-digit security code when applicable, source URL, rights note, and timing basis. `strict_forward` requires a verified first-available time. Use `retrospective_proxy` only when the actual historical first-available/capture time is unknown; provide the best known publication proxy and retain that uncertainty in the rights/time note. The CLI records this run’s `captured_at` and `ingested_at` itself.

Example metadata shape:

```json
{
  "document_id": "issuer-2024-annual-report",
  "version_id": "issuer-2024-annual-report-v1",
  "source_id": "issuer-disclosure-2025-03-26-001",
  "title": "Example issuer 2024 annual report",
  "document_type": "annual_report",
  "security_code": "600036",
  "entity_name": "Example issuer",
  "report_period": "2024",
  "accounting_basis": "consolidated",
  "source_url": "https://example.invalid/public/report.pdf",
  "rights_note": "Public disclosure for local analysis; no redistribution license asserted.",
  "expected_sha256": "<64 lowercase hex characters>",
  "published_at": "2025-03-26T00:00:00+08:00",
  "first_available_at": "2025-03-26T00:00:00+08:00",
  "time_basis": "retrospective_proxy",
  "read_principals": ["local-public"]
}
```

Import a source file and its metadata:

```powershell
uv run raama-rag ingest --source .\report.pdf --metadata .\report.metadata.json
```

The current local parser can extract text from PDF text layers, TXT/Markdown, HTML, CSV, JSON, XBRL/XML, DOCX, PPTX, and XLSX. CSV, XLSX, and Markdown tables keep row/cell boundaries and repeat table headers in retrieval row groups; reopening a row checks current permission and time scope. Image-only files and scanned PDFs are quarantined because OCR is not configured; XLSM, external Office relationships, and legacy XLS are rejected. Parser page coverage is a structural signal, not an accuracy score. Parsed content does not automatically become reviewed financial facts.

## Build and publish a local corpus snapshot

Use the `version_id` from the ingest response:

```powershell
uv run raama-rag release --release-id cmb-dev-2026-10-25-r1 --version-id issuer-2024-annual-report-v1 --reason "Initial local diagnostic corpus"
uv run raama-rag status
```

Publication builds and verifies the SQLite reference index receipt, then atomically advances the local active release. Its required backend set is currently only `sqlite_reference`; the CLI records Qdrant as unavailable and uses no embeddings. It verifies outbox bytes before publishing. A release cannot include parser-quarantined or low-coverage sources.

Rollback changes the active pointer to a previously published immutable snapshot or the initial seed:

```powershell
uv run raama-rag rollback --release-id seed --reason "Return the local corpus to its initial state"
```

## Start the Chinese query workbench

```powershell
uv run raama-rag serve --host 127.0.0.1 --port 8877 --principal-id local-public --time-mode strict_forward
```

Open `http://127.0.0.1:8877/`. The service binds to loopback and the API rejects requests that try to set a principal, release pointer, or time mode. The default is strict forward time. `--time-mode retrospective_diagnostic` is an operator-selected development mode; results carry a warning and cannot be presented as verified historical availability. The workbench has no source-upload route; use the local CLI for ingestion.

Available query channels are the structured-fact table, local BM25 reference channel, section/page reference, and explicit adapter seams. Dense, vision, source-relation, and cross-encoder reranking channels report unavailable until separately configured. The seed manifest and schema are under `benchmarks/financial_rag/` and `schemas/`.

## Current service boundaries

- The local reference implementation uses SQLite and filesystem content storage. PostgreSQL and S3-compatible adapters are not yet verified against live services.
- PostgreSQL row conversion, SQL translation, and migrations have code-level coverage only; live connection, migration, concurrency, isolation, CRUD, and crash-recovery checks remain pending.
- OpenSearch, Qdrant, OCR, embedding/reranking models, Temporal, and external observability are not configured.
- The sample corpus is a viewed, single-issuer public diagnostic set, not an independent acceptance set.
- Do not place broker credentials, account documents, private research, or restricted source bytes in the public diagnostic store.
