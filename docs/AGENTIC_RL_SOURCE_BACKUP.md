# Agentic RL clean-baseline overlay

Base repository revision: `1d2ae1910d5ac3920d7da1abf45e96cf8a973835`.

This directory contains task-owned source, tools, tests, and documentation plus a complete snapshot of the current working-tree `src/adaptive_mas/backtest/simulator.py`. The simulator snapshot is deliberately the only bundled file from the pre-existing simulator edits. It includes the pre-existing `untradeable_risk_pause` behavior as well as the Agentic RL forecast callback, one-time research-fee debit/funding, and shared per-symbol daily liquidity capacity. The completed replay records 78 paused decisions in B training (`replay_complete/training_B/summary.json`) and zero in the other evaluation splits; this pause behavior was present before this Agentic RL work and was used by the B training run. Retaining the complete simulator snapshot is required to reproduce that run. The overlay does not include modified `decision.py` or `public_data.py`.

## Files

- `scripts/run_agentic_rl_historical.py`
- `scripts/prepare_agentic_rl_qwen.py`
- `scripts/infer_agentic_rl_frozen.py`
- `scripts/normalize_agentic_rl_qwen_responses.py`
- `src/adaptive_mas/agents/public_prompt_context.py`
- `src/adaptive_mas/evaluation/agentic_rl.py`
- `src/adaptive_mas/backtest/simulator.py`
- `tests/test_agentic_rl.py`
- `tests/test_public_prompt_context.py`
- `docs/AGENTIC_RL_PROMPT_CONTEXT_V3.md`

`run_agentic_rl_historical.py` has no dependency on the unrelated `contextual_rl.py`. It reuses HEAD's `adaptive_mas.evaluation.mvp_research._risk_estimates`; the unchanged HEAD modules `adaptive_mas.decision`, `adaptive_mas.data` / `adaptive_mas.data.public_data`, `adaptive_mas.data.research`, and `adaptive_mas.portfolio.optimizer` provide the remaining project interfaces. A read-only import audit found no other task-owned imports of missing repository modules. The prior clean-HEAD dependency staging resolved these imports; its 11 focused tests ran before the complete simulator snapshot was synchronized here. This final simulator sync was not re-tested, per the follow-up instruction.

## Existing inputs needed for a historical replay

The overlay does not copy data or experiment artifacts. The historical runner still needs the already configured `train.csv`, selected BaoStock cache, AKShare public cache, P8 forecast files, P9 calibration files, and frozen outbound/response/normalized-response inputs under their existing workspace paths. Python project dependencies, including NumPy, are supplied by the repository environment. It does not need a Qwen key to replay cached responses.

The new offline preview needs only the existing frozen outbound manifest. It writes a separate v3 preview and makes zero network requests:

```powershell
.venv\Scripts\python.exe scripts\prepare_agentic_rl_qwen.py preview --event-id <existing-event-id> --output .tmp\agentic_rl_20261021_review\prompt_context_v3_preview.json
```

The public-context builder keeps caller-supplied `支持` and `反证` source tags local. When counter evidence is explicitly tagged, it reserves the top counter source and, if available, the top support source before relevance fill; unmarked sources stay unclassified. Its context cap measures UTF-8 bytes of the complete serialized messages plus 128 framing-reserve bytes, then adds the 600-token output reserve. The combined amount must fit the 8,600-token default bound. It drops the lowest-priority unreserved sources first while keeping a citable source, then drops summaries from the end of the supplied priority order; if reserved material still cannot fit, it fails instead of truncating. Section/table citations retain supplied section, paragraph, and page locators, and table headers, units, qualifiers, and rows stay together.

The v2 request manifest, its approved hash, response caches, accounts, raw data, logs, and frozen results are deliberately outside this overlay.
