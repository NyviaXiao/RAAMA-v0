# P9 Return-First Progress

Updated: 2026-09-27 (Asia/Shanghai)

## Research boundary

- Preserve all existing P8/post-P8 artifacts and working-tree changes. Development used BaoStock plus `train.csv`; `test.csv` has not been read.
- Frozen common start is 2024-07-10 after close, with 399 NAV/execution days from 2024-07-11 through 2026-03-06.
- There are 400 after-close decision dates through 2026-03-06. The final signal has no next session and is not an executed return day.
- All historical results are development evidence, not forward validation.

## Completed research

- P8 report: `experiments/post_p8_research_20260926/RESEARCH_REPORT.md`.
- Decision attribution: `experiments/p9_return_first_20260926_v2/RESEARCH_REPORT.md`. The frozen joint optimizer bought on 0/400 dates; the forecast-error allowance was the main blocker. Error-relaxed probes are diagnostic only.
- Sparse fusion full replay: `experiments/p9_sparse_fusion_20260926_full`. Net +10.15%, 399 days, 12,767 fills, ¥229,071 fees, 50.0% mean cash, −17.56% max drawdown; P8 trend +23.23%; paired daily mean difference −0.03985% [−0.15379%, +0.07365%]. Sparse marked risk exceeded 20% on 64 days and reached 25.01%. It does not establish multi-expert incremental value; no further sparse-gate search was done.
- Rolling mean-calibration same-simulator replay: `experiments/p9_joint_economics_20260927_full_exit`, complete under Python PID 53916 / Windows launcher PID 27760. Five accounts each have 400 decision rows, 399 NAV days, no negative cash, zero outside-basket additions, fees reconcile, and NAV cash-plus-holdings error below ¥1e-9. All 1,200 method-date calibrations satisfy strict maturity.
- Five P9 net returns: old RMSE/soft8 0%; point/soft8 +4.72%; point/hard20 +20.74%; lower-bound/hard20 +2.51%; trend point/hard20 +10.86%. Main fees ¥50,435, mean cash 47.87%, 28,001 fills, max marked risk 22.91%, 65 closes above 20%. Versus P8 simple trend the main paired daily mean difference is −0.02058% [−0.17556%, +0.13066%]; it does not establish outperformance.
- Pre-specified risk-budget sensitivity: `experiments/p9_joint_economics_20260927_risk_sensitivity`, complete under Python PID 53488. 15%: +21.26%, 52.39% cash, ¥47,592 fees, 77 days above budget; 20% main: +20.74%, 47.87%, ¥50,435, 65 days; 25%: +21.92%, 43.84%, ¥55,035, 41 days. Keep 20% as the frozen main setting; do not pick by realized return.
- P9 outcome-noise reference: 1.64% market-relative portfolio mean absolute error and 2.04% average horizon portfolio sigma. These are not conditional-mean confidence intervals.
- The shared workspace reports 72 passing tests after the point-in-time no-forecast holding rule was added.

## No-forecast held-position rule and audit

A held position with no forecast from the selected method has zero forecast alpha, stays in the account, and can only be held or reduced. Its target weight is capped at its current weight. When a real forecast exists outside that day's frozen basket, retain the real forecast but prohibit new additions. Do not fill missing forecasts with later realized returns.

Rebuildable audit: `experiments/p9_return_first_20260927/zero_alpha_exit_audit/zero_alpha_exit_audit.json`, with detailed per-variant event files. Main point/hard20 had 778 decision-date × stock events across 19 stocks, zero cap violations; soft8 had 2 events; lower-bound/hard20 had 1,750 across 16 stocks; trend point/hard20 had 793 across 28 stocks; legacy all-cash had zero. These are position events, not trading-day counts. Builder: `scripts/audit_p9_zero_alpha_exits.py`.

The first complete replay failed and is preserved at `experiments/p9_joint_economics_20260927_full`; it stopped on 2025-07-22 in variant 2 at 250/399 days when 601298 left the forecast universe. This was a diagnosed data-universe boundary, not an environment-start failure.

## Actual daily choice policy: complete development replay

The first P9 four-way screen was diagnostic only: it scored cash, current holdings, trend and fusion from the fixed market-relative account state but did not execute the chosen action. Its highest-scored choices were trend 177 dates, fusion 126, cash 81 and hold 16, out of 400.

The real state-dependent daily selector in `scripts/run_p9_daily_selector.py` and `src/adaptive_mas/backtest/simulator.py` recomputes trend and market-relative targets from its actual current account under point-in-time calibration, then selects the higher feasible net objective; cash and continuing holdings arise as feasible target outcomes. A 60-decision real-data smoke completed with all four action types and passed ledger, fee and cash checks. Its −3.864% short-window return is implementation-only.

Full replay `experiments/p9_daily_selector_20260927_full` completed under Python PID 57912. It has 400 frozen after-close decisions and 399 executable NAV days; the final 2026-03-06 after-close signal has no following session. Net return +11.05%, mean cash 20.75%, fees ¥39,670, maximum drawdown −13.08%, 43,291 filled orders. The 399 executable targets classify as trend 125, market-relative 131, cash 22, hold 121. Ledger identity, fees, nonnegative cash, frozen candidate scope and target hard constraints passed independent review. Marked closing risk breached 20% on 87 days, maximum 22.91%; this does not undo decision-time feasibility but remains a real exposure limitation. No-forecast exit-only events: 713.

Paired 20-session-block comparison on the same 399 dates: daily mean selector minus P8 simple trend −0.03855% [−0.17535%, +0.07959%]; minus fixed P9 main −0.01797% [−0.09602%, +0.05816%]; minus cash +0.03308% [−0.05421%, +0.12232%]. All intervals cross zero. Cumulative return is below both fixed P9 +20.74% and simple trend +23.23%, so the daily highest-objective choice has not shown realized incremental value. Reproducible pair audit: `scripts/audit_p9_daily_selector_pairs.py`; artifact: `experiments/p9_daily_selector_20260927_full/paired_baseline_comparison.json`.

## BaoStock feasibility and forward state

- Installed BaoStock client 0.9.40 exposes quarterly profit/balance calls plus performance-express and forecast-report calls. Client comments describe date filters as publication/update or publication/statistical dates. This establishes interface-level structured data feasibility only, not response-field verification, full announcement text, intraday availability time or retrospective vintages.
- Current manifests contain only `query_history_k_data_plus` and `query_hs300_stocks`. The latest local index row is 2026-09-23. One live login attempt failed (`10002007`, socket `WinError 10057`) before any market/report query; no second attempt was made.
- Feasibility details: `experiments/p9_return_first_20260927/BAOSTOCK_DATA_FEASIBILITY.md`.
- `scripts/run_p9_forward_validation.py` now runs P9's actual selector, fixed P9 market-relative comparator, P8 simple trend, and CSI 300 price proxy from an independent viewed-period shadow account. Its 2026-09-23 cutover replay completed at `experiments/p9_forward_cutover_20260927_v9`: first tradeable decision 2026-05-14, 93 viewed NAV warm-up days, ledger/fees/cash/target checks pass, 0 fresh dates and 0 matured fresh decisions. The shadow account warm-up returns are excluded from forward results. Exact code/config/candidate/seed/BaoStock manifest hashes, 1,176 raw BaoStock response files and the cutover model/account states pass verification. Matured-date gating now requires both frozen method baskets to have complete available labels. All frozen inputs and account states are in `P9_FROZEN_MANIFEST_20260927.json`.
- The genuinely-new-date branch was exercised only as a development smoke with an artificially earlier 2026-09-17 cutover. It replayed four already-viewed 2026-09-18–23 dates and one matured decision in `experiments/p9_forward_historical_branch_smoke_20260927/`, explicitly marked `historical_branch_smoke_only`. This is not future evidence. No real fresh trading dates after 2026-09-23 are available locally. A three-session forward decision needs 123 fresh trading dates before 120 decisions can mature. The P8-only old status file remains `awaiting_new_dates`; the P9 cutover report independently reports the same boundary.

## Final report

Canonical integrated report: `P9_FINAL_REPORT_20260927.md`; the completed selector and paired comparison are incorporated. Do not consume a reset credit.
