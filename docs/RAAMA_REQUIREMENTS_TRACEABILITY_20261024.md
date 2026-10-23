# RAAMA implementation requirements traceability

Baseline: `772b4fd20a4cc27342434eb16aec6515c2cad419` on `harness/gepa-evolution-20261022`.

This matrix distinguishes implementation, automated checks, actual main-chain execution, independent review, and evidence that must await real future data. Test fixtures must be labelled and never described as real market outcomes. `test.csv` is prohibited outside a final isolated scoring step; this implementation does not read it.

| ID | Requirement | Code entry / artifact | Verification and evidence | Status |
|---|---|---|---|---|
| S0-01 | Record baseline, current entry points, capabilities, versions, test contracts, and preserve historical pathways | `docs/RAAMA_IMPLEMENTATION_INVENTORY_20261024.md`; existing CLI/runtime/evaluation entry points | `uv run python -m pytest -p no:cacheprovider -q`: exit 1, 164 passed, 2 known overlay contract failures; history/data boundary tests passed | implemented; tested at baseline; pending parent review |
| S1-01 | Context compaction preserves dynamic financial facts plus symbol, field, value, unit, period, timestamps, vintage, provenance, conflicts, and reopen reference | pending | targeted regression and reset/reopen run | pending |
| S1-02 | Runtime fact validation rejects wrong security, field, value, unit, reporting period, unavailable source, or unexposed citation; unknown remains unresolved | pending | positive/negative validator cases and runtime path | pending |
| S1-03 | Structured fact correctness and semantic claim support are separate; candidate numeric eligibility is explicit | pending | runtime pipeline output and negative cases | pending |
| S1-04 | Recall enforces permissions and as-of visibility, identity/period relevance, dedupe/source independence, support/counterevidence, budget, and discard reasons | pending | recall path with labelled fixtures | pending |
| S1-05 | Parse, tool, and transport failures have bounded correction/retry/reconciliation; budget reservation and idempotency survive interruption | pending | failure and concurrent-worker checks | pending |
| S1-06 | Existing public expert overlay contract is reconciled without weakening intended assertions | pending | overlay regression suite | pending |
| S2-01 | Optional market, fundamental, news, and attention experts use real eligible sources and report unavailable honestly | pending | backend execution with available/absent sources | pending |
| S2-02 | Research manager's plan/action/challenge/stop choice drives tool use; new evidence can change its next action; duplicate/no-gain actions stop | pending | persisted end-to-end run and action trace | pending |
| S2-03 | Evidence, counterevidence, claims, provenance, and structured handoffs are linked to immutable run/task traces | pending | trace/artifact inspection | pending |
| S2-04 | Hard risk checks always run; optional risk research is separate; final investment-manager feedback consumes actual risk decisions | pending | blocked and permitted portfolio runs | pending |
| S2-05 | Task/request state, feedback, and execution results are durable, transactional, concurrent-safe, and idempotent | pending | SQLite concurrency/recovery checks | pending |
| S3-01 | Offline GEPA evolves only bounded strategy/routing parameters; hard policy, tools, validators, evaluator, optimizer, and production pointer are immutable | pending | candidate-schema and mutation rejection checks | pending |
| S3-02 | Research strategy is compared to a fixed baseline with frozen partitions, action/cost traces, and bounded budgets | pending | paired offline smoke/report | pending |
| S3-03 | Production runs bind a fixed complete package identity; cache/reuse verifies package identity | pending | identity mismatch/reuse checks | pending |
| S4-01 | Mature expert calibration/error, sample size, coverage, source overlap, uncertainty, and dissonance are explicit | pending | calibration artifact and coverage checks | pending |
| S4-02 | Error-shrunk baseline remains; state-conditional low-dimensional fusion is trained only on mature allowed labels and frozen for inference | pending | chronological split and frozen inference check | pending |
| S4-03 | Text is reproducible features first; no uncalibrated text is converted directly into numeric return | pending | eligibility and leakage checks | pending |
| S4-04 | Fusion, query policy, account-utility variants, and ablations are separated; costs and units are consistent | pending | predeclared comparison artifact | pending |
| S5-01 | Development, selection, independent acceptance, and forward/shadow partitions are isolated and versioned; `test.csv` stays untouched | pending | partition boundary audit | pending |
| S5-02 | Research adoption cannot update production; backend enforces acceptance/shadow/publish roles and hard-policy invariants | pending | unauthorized and authorized transition checks | pending |
| S5-03 | Rollback restores a complete compatible immutable bundle and records audit history | pending | publish/rollback execution | pending |
| S5-04 | Engineering, research quality, economic value, forward evidence, and scientific claims have separate states | pending | emitted report/status audit | pending |
| S6-01 | Unified task/run/trace/span observability records source, validation, cost, latency, budget, decisions, recovery, and controlled detail | pending | local trace inspection and redaction check | pending |
| S6-02 | Local API and independent worker persist jobs/artifacts in SQLite with real locking and idempotency | pending | start/submit/poll/restart run | pending |
| S6-03 | Workbench is a real Chinese UI connected to task lifecycle, evidence/challenges, portfolio/risk/cost, versions, evaluation, and release controls | pending | browser interaction evidence; unavailable state if environment blocks | pending |
| S6-04 | Financial execution supports both nominal research and quantity modes, sellable shares, lot rules, suspensions/limits, corporate actions, fees/slippage/capacity, risk, and exact reconciliation | pending | fixture-based reconciliation and market-rule source/version audit | pending |
| S6-05 | Prediction labels and investment/NAV accounting are separate; no real broker order or public deployment is triggered | pending | ledger invariants and shadow execution record | pending |
| PLAN-01 | Preserve existing contextual RL and Expected SARSA histories as distinct replayable paths; no concurrent strategy controllers | pending | API/path inventory and history regression | pending |
| PLAN-02 | Evaluations group by time/event, control leakage/embargo, register attempts and viewed ranges, and distinguish pending external/forward evidence | pending | protocol and output audit | pending |
| PLAN-03 | No synthetic fixture is represented as a real data result or as proof of scientific/investment improvement | pending | report wording and provenance audit | pending |

## Stage acceptance log

Stage 0 inventory is recorded. No implementation stage has been accepted by the parent yet. See `.tmp/harness_supervision_20261024/implementation_notes.md` for exact commands, exit codes, elapsed times, and commits as work proceeds.
