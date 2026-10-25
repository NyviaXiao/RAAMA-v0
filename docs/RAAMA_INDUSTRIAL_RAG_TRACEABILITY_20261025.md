# Industrial financial RAG traceability — 2026-10-25

Baseline: `420cc4f4dc26f875d568095da0fa6ed79e35732a` on `harness/gepa-evolution-20261022`.
This is an implementation checkpoint, not stage acceptance. “Code checked” means only the local checks listed here; it does not imply a real PostgreSQL, S3, OpenSearch, Qdrant, OCR, model, workflow, or independent review.

## A — task, benchmark, and migration boundary

| ID | Artifact / entry point | Evidence at this checkpoint | Status |
|---|---|---|---|
| A01 | This file; `RAAMA_INDUSTRIAL_RAG_IMPLEMENTATION_PLAN_20261025.md` | Maps A01–G05 and the proposal’s exact 20-module experiment matrix; preserves existing contract checks below | Implemented; parent review pending |
| A02 | `schemas/financial_qa_case.schema.json`; `benchmarks/financial_rag/seed_manifest.json` | Typed necessary evidence, permitted arithmetic, refusal reason, source hash/rights/times, annotation and viewed-state fields | Implemented; schema and annotation still require independent review |
| A03 | `benchmarks/financial_rag/seed_manifest.json` | Six viewed diagnostic cases from one public CMB 2024 report family plus four separately labeled synthetic adversarial cases; both source hashes and capture timestamps recorded | Small diagnostic seed only; cross-page continuation, verified restatement conflict, independent issuer holdout, and expert-sized set remain absent |
| A04 | Seed manifest `baseline` record | No pre-change immutable page/fact index or complete evidence labels existed at the authorized baseline; metrics are explicitly `unavailable` | Unavailable, with reason recorded; no fabricated score |

The seed set is development material. Every public case was viewed; none is independent acceptance data. The historical capture time is unknown, so the source is marked `retrospective_proxy`. These examples cannot establish point-in-time performance.

## B — authoritative source, version, and publication contract

| ID | Code / artifact | Local evidence | Status |
|---|---|---|---|
| B01 | `src/adaptive_mas/industrial_rag/models.py`; SQLite schema and PostgreSQL migration | Typed document/version/section/block/table/cell/span/fact/release/receipt records; Decimal and source-location constraints | Code checked by focused contract tests |
| B02 | `models.py`; `repository.py::permitted_versions` | UTC-normalized published/first-available/captured/ingested/valid/revised times; strict forward excludes late/unknown capture; retrospective proxy is explicitly labeled; changed immutable identity is rejected | Code checked by focused contract tests |
| B03 | `repository.py`; `postgres.py`; `object_store.py`; `migrations/0001_industrial_rag.sql` | Local SQLite and content-addressed file store are exercised. PostgreSQL SQL translation and row-factory behavior have isolated unit coverage. Psycopg migration/CRUD/concurrency, S3-compatible storage, and service restart have not run against real backends | Partial; actual PostgreSQL and S3 contract suites pending |
| B04 | `repository.py::consume_outbox_once`, `build_reference_index_receipt`, `publish_release`, `rollback_release` | SQLite lease/reclaim/idempotency, source-hash checks, receipt-vs-index checks, required-backend atomic gate, old-snapshot and rollback paths are covered. The CLI publishes only the local reference index. | Local reference backend code checked; external backend build receipts and crash/failure integration remain pending |
| B05 | `repository.py::permitted_versions`; `retrieval.py`; `web.py` | Current grants and temporal/release scope filter before retrieval; grant revision participates in cache identity; span and original download recheck current permission; HTTP query does not accept caller-controlled principal, time mode, or release | Code checked by focused contract tests; independent security review pending |

The local BM25 implementation is a deterministic Chinese bigram reference channel. It is not OpenSearch, a dense embedding, or a neural reranker. No production release is implied by the local corpus pointer.

## Proposal optimization matrix — 20 module crosswalk

The module numbering and names follow section 10 of `RAAMA_INDUSTRIAL_RAG_HARNESS_REDESIGN_PROPOSAL_20261024.md`. The stage is deliberately still pending where a lightweight adapter or fixture is not the named real capability.

| # | Module | Current entry point | Status / boundary |
|---:|---|---|---|
| 1 | Layout/OCR routing | `parser.py` | Partial: PDF text-layer coverage signal; scans/images quarantine; no OCR model |
| 2 | Cross-page table reconstruction | `parser.py::validate_cross_page_table_continuation` | Conservative rule and synthetic mismatch check; no real financial multi-page table evaluation |
| 3 | Semantic and row-group chunking | `parser.py::_blocks_for_pages`; `parser.py::_table_row_group_blocks` | Page/paragraph plus CSV/XLSX/Markdown row groups with repeated headers; no calibrated semantic chunker or PDF table detector |
| 4 | Hierarchy/page/cell linkage | `models.py`; `parser.py::records_from_parse`; `retrieval.py::read_table_row` | Page/section/block/span and table-row/cell linkage; row expansion rechecks current permission/release/time. Production-grade coordinates are not available on all formats |
| 5 | Entity and alias disambiguation | `models.py::Document`; `QueryPlan` | Metadata fields exist; entity resolution/query alias extraction is not implemented |
| 6 | Time/period/basis extraction | `models.py`; `repository.py` | Explicit metadata and filtering implemented; automated extraction from source text is not implemented |
| 7 | Constrained query rewrite | `retrieval.py::QueryPlan` | Not implemented |
| 8 | Multi-hop question decomposition | `retrieval.py::route` | Not implemented |
| 9 | BM25 | `retrieval.py::_bm25` | Deterministic local reference only; no OpenSearch index |
| 10 | Embeddings and multi-recall | `retrieval.py` channel adapters | Adapter seam only; dense model and Qdrant unavailable |
| 11 | RRF and source deduplication | `retrieval.py::reciprocal_rank_fusion` | Implemented and local-checkable; scores are labeled rank scores, not probabilities |
| 12 | Independent reranker | `retrieval.py::IndustrialRagService` | Adapter seam only; no cross-encoder configured |
| 13 | Parent/footnote expansion | `retrieval.py::read_span`; `web.py` | Cited span expansion exists; automatic parent/footnote graph expansion is not implemented |
| 14 | Financial facts and table normalization | `models.py::FinancialFact`; `parser.py::normalize_financial_cell` | Type/Decimal normalization implemented; PDF-to-reviewed-financial-fact extraction is not implemented |
| 15 | Faithful compression/context budget | — | Not implemented |
| 16 | Answer plan and evidence coverage | `models.py::QueryPlan`; search trace | Query plan and channel traces exist; no answer planner or complete evidence-set coverage judge |
| 17 | Follow-up/challenge/refusal/stop | `retrieval.py` no-evidence status | Basic no-evidence response only; bounded research loop/challenge/stopping policy not implemented |
| 18 | Claim citation alignment | `retrieval.py` citations | Source-bound evidence citations exist; per-generated-claim entailment/alignment judge not implemented |
| 19 | Deterministic arithmetic validation | `parser.py::normalize_financial_cell` | Exact Decimal parsing is present; answer arithmetic/claim verification is not implemented |
| 20 | Conflict/basis/unsupported-claim checks | `models.py`; `repository.py` | Basis/time filtering and review gate exist; restatement/conflict resolution and semantic support checks are not implemented |

## Existing contracts carried into this change

These are preservation checks, not newly re-certified capabilities:

| Existing contract | Preserved reference |
|---|---|
| Numeric and natural-language risk profiles; optimizer target caps, feasibility, fee-adjusted cash floor | `docs/RAAMA_RISK_PROFILE_20261024.md`; `docs/RAAMA_OPTIMIZER_REFERENCE_COMPARISON_20261024.json` |
| Hard risk remains deterministic; unsupported candidates cannot bypass the optimizer | Existing decision and optimizer contract regressions referenced by `docs/RAAMA_REQUIREMENTS_TRACEABILITY_20261024.md` |
| Quantity execution ledger, fees/slippage, and prediction labels remain separate from NAV | Existing S6-04/S6-05 traceability and ledger checks in `docs/RAAMA_REQUIREMENTS_TRACEABILITY_20261024.md` |
| Harness production/research release pointer, audit, and complete rollback governance | Existing S5-01–S5-04 traceability in `docs/RAAMA_REQUIREMENTS_TRACEABILITY_20261024.md` |
| Contextual RL and Expected SARSA remain distinct replay paths; bounded GEPA candidate path retained | Existing PLAN-01 and S3-01–S3-03 traceability in `docs/RAAMA_REQUIREMENTS_TRACEABILITY_20261024.md` |
| No `test.csv` access and no edits/staging of `experiments/**` | Explicit task boundary; this change’s focused check uses only new isolated tests and local code |

## Stage log

- Focused A/B local contract suite: `14 passed` in 2.07 seconds using `.venv` on Windows.
- PostgreSQL row-maker unit checks exercise a no-result cursor and a later populated SELECT description. This is not a live PostgreSQL contract run.
- Full legacy regression suite, real PostgreSQL, S3, external retrieval models, independent A/B review, and backup publication are not included in this checkpoint.
- Follow-up regression pass: fact ingestion now requires every citation span to exist, belong to the same version, contain text, and identify a page; entity and accounting basis must match the parent document. Fact citations use the cited spans and retain each span's page and source. Reusing a source/version now checks parent document metadata and all parsed child records. Outbox acknowledgements require the current lease generation, including when the same worker reclaims an expired lease.
- Optional Harness bridge: `run_research_harness` can receive an `IndustrialRagService`, pins the active corpus release at task start, and exposes bounded search plus permission-rechecked span reopening to the fundamental agent. This is an injectable research path; the production daily-decision runtime still does not configure the service.
- Follow-up checks: `tests/test_industrial_rag.py`, `tests/test_harness_evolution.py`, and `tests/test_harness_unified_workflow.py` passed together (`63 passed`). Live PostgreSQL and S3-compatible contract runs remain pending.
- E bridge correction: omitted `limit` now defaults to its supported maximum, and repeated evidence identities are reused idempotently with an explicit `no_new_evidence` result. The shared source reader now reopens table row groups as well as exact text spans, rechecking access, release, and time scope.
- C parser increment: CSV/XLSX/Markdown table rows retain cell boundaries and repeat headers in row-group blocks; parent/previous/next/table/row links are persisted, while synthesized row text is not mislabeled as an exact source span. CSV/workbook size limits and Office external relationship checks are enforced. Negative and positive cases are in `tests/test_industrial_rag.py`.
- Latest focused verification: RAG, Harness evolution, and unified-workflow suites passed together (`66 passed`). Native OCR/layout models, PDF table extraction, live PostgreSQL/S3/OpenSearch/Qdrant, and production daily-decision runtime wiring remain pending.
