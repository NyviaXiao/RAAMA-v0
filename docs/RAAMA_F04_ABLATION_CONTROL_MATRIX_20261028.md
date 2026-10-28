# F04 paired RAG controls and evidence matrix — 2026-10-28

## Scope

The fixed catalog has 20 IDs, but it does not contain 20 executable module removals. The paired runner distinguishes bounded runtime interventions, parameter sensitivities, and controls that cannot safely be changed by a candidate. Every catalog row names its runtime control and the paired evidence required. The output explicitly sets `module_removal_claim_permitted` to false: the runner records bounded policy switches, not full causal removal of an entire subsystem.

The runner pins the same candidate, four fixed cases (two selection and two frozen evaluation), corpus release, adapter identities, principal, permission revision, and continuation revision across each pair. It enumerates every recorded search-tool invocation before validation, including invocations with a missing or wrong ablation ID. Every invocation must carry the expected policy ID (including an explicit `null` on baseline), nonempty coverage, a planner, and nonempty well-formed execution steps. Each step is checked against the runtime control; for ID 11 every fusion method must match the arm's configured strategy, and for ID 12 every reranking step must show the disabled state. Missing or malformed evidence remains in the trace summary and makes the pair incomplete. Provider-enabled runs reserve up to 80 calls before starting. An `offline_only` client reserves zero; an exact cache miss fails closed. No provider-backed run is authorized by this matrix.

## Per-ID control contract

| ID | Catalog item | Current classification | Control in the comparison arm | Pair evidence / limit |
|---:|---|---|---|---|
| 1 | Layout/OCR routing | Requires isolated source/index rebuild; not runnable in the paired query runner | Parser backend and layout-model bundle | Same captured source bytes, metadata, parser/model fingerprints, and a receipt for each rebuilt index are required. |
| 2 | Cross-page table reconstruction | Bounded retrieval-context intervention | `include_continuation_context`: `true` → `false` | Executed planner values are recorded for both arms. This suppresses reviewed continuation context at retrieval; it leaves reconstruction and review status fixed. |
| 3 | Semantic row-group chunking | Bounded retrieval candidate-set intervention | `row_group_chunking`: `true` → `false` | Executed planner values and search steps are recorded. This excludes existing row-group candidates at retrieval; indexed row-group records stay fixed. |
| 4 | Hierarchy/page/cell linkage | Immutable evidence-location contract; not runnable | No candidate switch | Exact source, page, table, row, column, and cell linkage must remain valid in every arm. |
| 5 | Entity alias disambiguation | Immutable identity contract; not runnable | No candidate switch | Source-backed entity and alias resolution must remain fixed. |
| 6 | Time/period/basis extraction | Immutable temporal contract; not runnable | No candidate switch | Strict-forward visibility and exact period/accounting-basis checks remain fixed. |
| 7 | Constrained query rewrite | Bounded runtime intervention | `query_rewrite_enabled`: `true` → `false` | Executed planner value and query-step hashes/counts are recorded. Mandatory correction search remains active. |
| 8 | Multi-hop decomposition | Bounded runtime intervention | `multi_hop_enabled`: `true` → `false` | Executed planner value and query-step hashes/counts are recorded. One supporting search remains required. |
| 9 | BM25 support channel | Bounded runtime intervention | BM25 membership in support channels: `true` → `false` | Executed channel set is recorded; mandatory BM25 correction search behavior remains fixed. |
| 10 | Dense retrieval | Conditional runtime intervention | Dense membership in support channels: `true` → `false` | Requires dense in the baseline and a configured, versioned dense adapter. The seed candidate omits dense, so its catalog row is not runnable until that condition is met. |
| 11 | Reciprocal-rank fusion | Bounded runtime intervention | `fusion_strategy`: `rrf` → `first_available_channel` | Both planner strategy and every actual fusion-step method are checked; a single mismatching step rejects the pair. |
| 12 | Independent reranker | Conditional runtime intervention | Configured reranking: `true` → `false` | Requires a versioned reranker in the baseline. The comparison planner and every executed reranking trace must record disabled. |
| 13 | Reviewed parent context | Bounded runtime intervention | `table_expansion_limit`: positive baseline → `0` | Executed planner value is checked; exact child-row evidence remains available. |
| 14 | Financial fact normalization | Bounded structured-fact channel intervention | Structured-fact channel membership: `true` → `false` | Executed channel set is checked. This omits the retrieval channel; parser normalization and report validation remain active. |
| 15 | Context allocation | Parameter sensitivity; not a module removal | `context_char_limit`: baseline → `16000` | Baseline and comparison limits are recorded from the same candidate and fixed cases. |
| 16 | Evidence coverage budget | Parameter sensitivity; not a module removal | `evidence_limit`: baseline → `64` | Baseline and comparison limits are recorded from the same candidate and fixed cases. |
| 17 | Challenge and stop | Immutable safety policy; not runnable | No candidate switch | Negative and counter-evidence cases must keep the same challenge and stop behavior. |
| 18 | Claim/citation alignment | Immutable evaluator contract; not runnable | No candidate switch | Claim-to-citation support evaluation stays identical across arms. |
| 19 | Deterministic arithmetic validation | Immutable calculation safety policy; not runnable | No candidate switch | Exact Decimal recomputation and operand-evidence checks stay identical across arms. |
| 20 | Conflict/unsupported-claim checks | Immutable safety policy; not runnable | No candidate switch | Conflicting-source and unsupported-number negative cases stay enforced across arms. |

For the seed candidate, nine runtime interventions are planned (IDs 2, 3, 7–9, and 11–14), two sensitivities are separate (IDs 15–16), and nine catalog rows are not runnable without a different safe baseline or an offline rebuild (IDs 1, 4–6, 10, and 17–20). ID 10 is conditional rather than permanently unavailable.

## Local execution evidence

`tests/test_harness_evolution.py` targeted control slice: **15 passed, 55 deselected in 5.50s**. It executes the ten bounded runtime controls (including conditional dense and reranker switches after configuring synthetic versioned adapters) and both parameter sensitivities against the local synthetic RAG service. The evaluator is stubbed, and the offline-only client has a zero planned provider-call ceiling; all recorded provider-call counts are zero. The test verifies paired control observations and sensitivity labels, not report quality or causal effects.

No blind holdout was used. No scientific acceptance, model-backed effect, public-filing result, or provider generation is established by this local control check.
