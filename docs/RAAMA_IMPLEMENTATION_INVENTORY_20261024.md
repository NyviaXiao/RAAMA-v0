# RAAMA implementation inventory at baseline

Baseline: `772b4fd20a4cc27342434eb16aec6515c2cad419` on `harness/gepa-evolution-20261022`.

## Existing main paths

| Concern | Existing entry point | Baseline behavior |
|---|---|---|
| Public research orchestration | `scripts/run_harness_decision.py` → `adaptive_mas.harness_runtime.create_harness_daily_decision` → `agents.harness.run_research_harness` | Fixed market and fundamental specialists, then a research manager with one bounded directed query. A complete research result is passed into `decision.create_daily_decision`. |
| Candidate effect research/evolution | `scripts/run_harness_evolution.py` → `adaptive_mas.harness_evolution` | GEPA proposal/selection/frozen-case workflow; the version store's `adopt()` writes the shared `active.json` pointer directly. |
| Numeric decision | `decision.create_daily_decision` → `optimize_forecast_target` → portfolio optimizer | Calibrated forecasts are numeric inputs; manager priority symbols filter candidates. The GEPA harness path has deterministic optimizer risk constraints; separate role-based risk reviews are emitted only by the legacy contextual-collaboration path. |
| Public evidence acquisition/selection | `agents.public_experts.build_public_expert_snapshot`; `data.public_data`; `agents.public_prompt_context.build_public_context` | Cached public sources, captured or explicitly retrospective mode, citation/context selection, and separate public signal expert components exist. |
| Historical adaptive controllers | `evaluation.contextual_rl` and `evaluation.agentic_rl`, with distinct replay/development scripts | Separate preserved pathways; tests cover their development/evaluation boundary. They are not called the GEPA runtime manager. |
| Local product surface | No API server, worker, task database, or interactive workbench found in `src`, `scripts`, or `pyproject.toml` | Existing scripts and generated showcase/report files do not provide the required persisted API-worker-workbench flow. |

## Safe baseline check

`test.csv` was not opened. One full collection-and-test run completed with `uv run python -m pytest -p no:cacheprovider -q`: **164 passed, 2 failed** in 1.85 s. Both failures are the known public-expert-overlay contract drift:

- `test_matured_effect_changes_only_same_day_candidate_and_replays_same_account` expects a matured public effect to mutate the base `market_plus_relative_fusion` forecast. The implementation instead retains base returns and exposes calibrated-eligible effects as distinct `fundamental_p11`/`news_p11` components, consistent with the final plan's separation of forecast components.
- `test_weekend_carry_signal_is_available_to_next_trading_overlay_once` expects a Qwen direction signal to become numeric return. The implementation intentionally labels Qwen as evidence-only and does not add it to eligible numeric experts.

The implementation will update those tests to assert the current separation/evidence-only contract, not weaken the intended safety checks.

The exact baseline command and result, along with the two failed test names and assertions, are recorded in `.tmp/harness_supervision_20261024/implementation_notes.md`. The other two baseline attempts failed before tests ran: default uv cache access was denied, then console pytest collection omitted the repository root from its import path.

## Baseline data boundary

Repository rules were read from `AGENTS.md`. Existing market/research loaders target `train.csv`; the evaluation boundary tests create a temporary invalid `test.csv` fixture to prove that research ignores it. No baseline run opened user scoring data. Historical experiment folders under `experiments/` remain untouched and untracked.
