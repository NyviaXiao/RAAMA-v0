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

The default parser extracts text from PDF text layers, TXT/Markdown, HTML, CSV, JSON, XBRL/XML, DOCX, PPTX, and XLSX. CSV, XLSX, and Markdown tables keep row/cell boundaries and repeat table headers in retrieval row groups; reopening a row checks current permission and time scope. An opt-in Docling path runs CPU layout analysis, Chinese RapidOCR, and TableFormer against pinned local assets; it records page boxes, structured table cells, and page coverage without inventing exact character spans. Docling counts column-heading rows separately from row-header stubs, retains explicit unit and period labels with cell coordinates, and quarantines a table when heading rows are missing or ambiguous. The synthetic scan now produces two searchable rows that can be reopened with the source row/column, value, unit, and period intact. XLSM, macro-marked Office files, external Office relationships, and legacy XLS are rejected. Parser page coverage is a structural signal, not an accuracy score. Parsed content does not automatically become reviewed financial facts.

Fetch the pinned parser assets explicitly before ingesting scanned PDFs. The model files and generated SHA-256 inventory remain in the ignored `.tmp` directory:

```powershell
uv run raama-rag fetch-docling-models
uv run raama-rag ingest --source .\scanned-report.pdf --metadata .\scanned-report.metadata.json --pdf-backend docling
```

The Docling asset contract pins Docling 2.129.0, RapidOCR 3.9.2, Heron layout commit `8f39ad3c0b4c58e9c2d2c84a38465abf757272d8`, and TableFormer commit `fc0f2d45e2218ea24bce5045f58a389aed16dc23` (tag `v2.3.0`). Inference requires the local asset manifest and checks file paths, sizes, and SHA-256 values; external services and plugins are disabled and Hugging Face is placed in offline mode. A synthetic scanned financial-table PDF has passed the local CPU smoke path. This does not establish OCR accuracy on real filings or cross-page table quality.

Tables whose next page omits its column headings remain quarantined until a reviewer confirms the adjacent-page candidate. To expose review controls in the workbench, configure a server-side principal that already has read access to the source, including it in that source's `read_principals` metadata before ingest:

```powershell
uv run raama-rag serve --host 127.0.0.1 --port 8877 --principal-id local-public --continuation-review-principal-id local-reviewer
```

The review identity comes from server configuration; request bodies cannot choose it. Confirm, reject, and revoke decisions are kept in an audit table. Confirming or revoking changes the continuation revision and invalidates cached searches and pagination cursors.

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

To opt into the pinned local Qwen CPU embedding and reranker, fetch both immutable revisions and pass the asset directory to the workbench:

```powershell
uv run raama-rag fetch-qwen-models
uv run raama-rag serve --host 127.0.0.1 --port 8877 --principal-id local-public --time-mode strict_forward --qwen-models .tmp\industrial_rag_supervision_20261025\qwen-models
```

Qwen3 Embedding and Reranker 0.6B are Apache-2.0 models pinned to embedding revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` and reranker revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`. They are loaded from local files only; each first load checks the recorded file paths, sizes, and SHA-256 values. The embedding uses last-token pooling and L2 normalization; the cross-encoder records a yes/no logit difference as a ranking score, never as a calibrated probability. Dense retrieval currently scans permission-filtered SQLite blocks and keeps a bounded in-memory vector cache; it is a real model-inference reference, not a persistent Qdrant index. Vision and source-relation channels remain unavailable. The seed manifest and schema are under `benchmarks/financial_rag/` and `schemas/`.

## Offline evaluation

Evaluate an explicit frozen QA manifest against a stored prediction file:

```powershell
uv run raama-rag evaluate --manifest .\benchmark.json --predictions .\predictions.json --output .\evaluation.json
```

Evidence labels must identify an exact source hash, page, and original cell/table row, fact, or text span. Numeric evidence compares `Decimal` values; units, multipliers, periods, and accounting basis must match exactly. Missing predictions count as failures, unknown case IDs are rejected, and repeated source locations contribute only once to ranking gain. The report includes Recall@K, complete evidence coverage, MRR, nDCG, answerability/refusal accuracy, source match, duplicates, and latency percentiles, grouped by data partition and view status. It records benchmark and prediction hashes. Evaluation reads labels offline and must not be called from live agents or query paths. The current viewed seed is diagnostic; no independent acceptance evaluation or 95% claim has been run.

## Current service boundaries

- The local reference implementation uses SQLite and filesystem content storage. PostgreSQL and S3-compatible adapters are not yet verified against live services.
- PostgreSQL row conversion, SQL translation, and migrations have code-level coverage only; live connection, migration, concurrency, isolation, CRUD, and crash-recovery checks remain pending.
- OpenSearch, persistent Qdrant, Temporal, and external observability are not configured. Local Docling and Qwen inference are optional and require their separate fetch commands; neither proves live backend integration.
- The sample corpus is a viewed, single-issuer public diagnostic set, not an independent acceptance set.
- Do not place broker credentials, account documents, private research, or restricted source bytes in the public diagnostic store.

## Shared CPU host resource template

`deploy/systemd/raama.slice` caps the slice at 2000% aggregate CPU, sets `MemoryHigh=180000000000`, `MemoryMax=200000000000`, disables additional swap, and sets a separate `TasksMax=4096` process/thread budget. TasksMax is not a CPU count. The host must add an `AllowedCPUs` drop-in containing only the CPUs delegated to RAAMA; no host CPU list is assumed here. `deploy/systemd/raama-cpu.env` sets model and numerical-library threads to one per process and workbench request threads to four. These are templates only: place every RAAMA process in the same delegated slice and confirm effective `cpu.max`, `cpuset.cpus.effective`, memory, and task limits on the target host before starting heavy work. Per-process thread settings do not enforce the aggregate budget by themselves.
