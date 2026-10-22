# P9 前向影子账户操作说明

账户切换仍为 2026-09-23 收盘的暖启动状态。策略参数于 2026-09-27 22:57（北京时间）冻结；9 月 28 日首个收盘前仅修正冻结后统计边界和账户决策过滤，并重建哈希清单。新清单的 `created_at_utc` 记录修正后代码版本的生成时间。9 月 24 日行情只作冻结前诊断，9 月 25–27 日休市。9 月 24 日预测不得进入账户行动，因为它的下一交易日订单会在 9 月 28 日开盘执行；9 月 24 日行情可作为 9 月 28 日收盘决策的可见特征。首个冻结后决策日为 9 月 28 日收盘，最早于 9 月 29 日执行。入口、冻结哈希与截止日回放分别见 `scripts/run_p9_forward_validation.py`、`P9_P12_FROZEN_MANIFEST_20260928.json` 和 `experiments/p9_forward_cutover_20260927_v9/`。

在 BaoStock 能正常连接、目标交易日已完整收盘后，采集一次截至目标日的行情，随后为该决策日生成点时专家快照并运行前向账户。采集目录必须从 2026-03-16 开始，覆盖整个暖启动期和新日期；采集命令会打印新批次目录，后续命令都使用同一个目录：

```powershell
.venv\Scripts\python.exe scripts\acquire_phase2_market_data.py --oos-start-date 2026-03-16 --oos-end-date YYYY-MM-DD
.venv\Scripts\python.exe scripts\acquire_public_data.py --mode daily --as-of-date YYYY-MM-DD --output-dir data\raw\akshare --symbol 000001 --symbol 600000
# Replace the example codes with the unique held and chosen candidate codes; do not pass the full CSI 300.
# Set decision-time to the actual Shanghai time after the market and public-data refreshes have completed; do not backdate it to 15:00.
# In the current PowerShell session, configure POPAGENT_LLM_URL, POPAGENT_LLM_MODEL (optional), and POPAGENT_LLM_KEY from your local secret store.
.venv\Scripts\python.exe scripts\run_p9_p12_research.py --decision-time 'YYYY-MM-DDTHH:MM:SS+08:00' --horizon-sessions 3 --account-positions-csv experiments\p9_forward_前一日\p9_fixed_market_relative\positions.csv --baostock-run-dir data\raw\baostock\fetch-实际采集批次 --output-dir experiments\p9_p12_forward\snapshots\YYYY-MM-DD
Remove-Item Env:\POPAGENT_LLM_URL, Env:\POPAGENT_LLM_MODEL, Env:\POPAGENT_LLM_KEY -ErrorAction SilentlyContinue
.venv\Scripts\python.exe scripts\run_p9_forward_validation.py --as-of-date YYYY-MM-DD --baostock-run-dir data\raw\baostock\fetch-实际采集批次 --expert-snapshot-root experiments\p9_p12_forward\snapshots --frozen-manifest P9_P12_FROZEN_MANIFEST_20260928.json --output-dir experiments\p9_forward_YYYY-MM-DD
```

## 2026-09-28 首次收盘：复用同一批次

只采集一次截至 2026-09-28 的完整行情。先把采集命令输出的 `fetch-...` 路径代入下面命令；9 月 24 日诊断和 9 月 28 日正式账户使用同一批次。诊断 runner 会在加载后把数据截到 9 月 24 日，报告同时记录原缓存末日和被排除的更晚日期；9 月 28 日行情不会进入诊断账户。

```powershell
.venv\Scripts\python.exe scripts\acquire_phase2_market_data.py --oos-start-date 2026-03-16 --oos-end-date 2026-09-28 --output-root data\raw\baostock\p9_forward
.venv\Scripts\python.exe scripts\run_p9_forward_validation.py --as-of-date 2026-09-24 --baostock-run-dir data\raw\baostock\p9_forward\fetch-实际采集批次 --expert-snapshot-root experiments\p9_pre_freeze_diagnostic_20260928\no_expert_snapshots --frozen-manifest P9_P12_FROZEN_MANIFEST_20260928.json --output-dir experiments\p9_pre_freeze_diagnostic_20260928
.venv\Scripts\python.exe scripts\acquire_public_data.py --mode daily --as-of-date 2026-09-28 --output-dir data\raw\akshare --symbol 000001 --symbol 600000
# Replace the example codes with the unique held and chosen candidate codes; do not pass the full CSI 300.
# Set POPAGENT_LLM_URL, POPAGENT_LLM_MODEL (optional), and POPAGENT_LLM_KEY from the local secret store before the snapshot command.
# Replace 18:00 with the actual Shanghai time after the BaoStock and public-data refreshes completed; if later, use the later time.
.venv\Scripts\python.exe scripts\run_p9_p12_research.py --decision-time '2026-09-28T18:00:00+08:00' --horizon-sessions 3 --account-positions-csv experiments\p9_pre_freeze_diagnostic_20260928\p9_fixed_market_relative\positions.csv --baostock-run-dir data\raw\baostock\p9_forward\fetch-实际采集批次 --output-dir experiments\p9_p12_forward\snapshots\2026-09-28
Remove-Item Env:\POPAGENT_LLM_URL, Env:\POPAGENT_LLM_MODEL, Env:\POPAGENT_LLM_KEY -ErrorAction SilentlyContinue
.venv\Scripts\python.exe scripts\run_p9_forward_validation.py --as-of-date 2026-09-28 --baostock-run-dir data\raw\baostock\p9_forward\fetch-实际采集批次 --expert-snapshot-root experiments\p9_p12_forward\snapshots --frozen-manifest P9_P12_FROZEN_MANIFEST_20260928.json --output-dir experiments\p9_forward_20260928
```

诊断账户从 9 月 23 日暖启动状态接续：如 9 月 23 日信号委托在 9 月 24 日开盘成交，会保留在 9 月 24 日收盘持仓中；9 月 24 日预测已从账户动作中排除。9 月 24 日行情用于生成 9 月 28 日快照时的持仓范围和收盘决策可见特征。9 月 28 日收盘新信号最早在 9 月 29 日开盘执行。

`acquire_public_data.py --mode daily` 的逐股请求只覆盖显式列出的代码；仅列出当前非零持仓和准备纳入当日评估的候选，不要把 300 个成分股全部重复传入。该命令对这些代码采集财务报表、财务指标和当日公告；宏观数据与最近 20 条 CLS 快讯每个采集日只抓取一次。请求缓存键含采集日期和代码：同日成功响应会跳过，失败请求可重试，所以应在各数据源更新完毕后对所需代码运行一次，再记录实际的上海时间作为快照截点。快照程序要求截点不早于 15:00，并在 `expert_snapshot.json` 与 `RESEARCH_REPORT.md` 记录形成时点及“下一交易日开盘最早执行”的规则；不把 15:00 之后抓到的资料伪装成 15:00 已可见。

后续交易日按上面的常规步骤每日只采集一次；传给 runner 的 as-of 必须是 BaoStock 采集目录中存在的交易日。runner 会按 as-of 截断市场表和成分快照，因此目录里如含有更晚日期，这些日期不会进入该次重放，并会记入报告的 `source_cache_dates_after_as_of_excluded`。

专家快照的股票范围按决策日最新可见的 CSI 300 成分与上次账户的非零持仓取并集；可用 `--candidate-symbols-file` 追加当日候选（每行六位代码）。每日需把前一日账户 `positions.csv` 传给 `--account-positions-csv`，不要沿用 9/23 的默认切换账本。每次最多请求 3 份近 30 日公告全文 PDF；Qwen 输入上限可用 `--llm-max-records` 控制。`pypdf` 需先按 `pyproject.toml`/`uv.lock` 安装。公告 PDF 或文本解析不可用时，快照会保留明确失败状态；历史 13.6 万条公告全文不会批量下载。Qwen 状态、token 用量、估算成本、延迟及文档哈希都写入专家快照。

若未配置 `POPAGENT_LLM_KEY`，Qwen 明确跳过；有凭证且存在新分析记录时，必须同时配置 `POPAGENT_LLM_URL`。`POPAGENT_LLM_MODEL` 可选，默认 `qwen3.7-plus`。不要把 endpoint 或凭证值写入仓库。

`forward_validation_report.json` 的 `new_date_account_results` 与配对比较只计 2026-09-28 起的冻结后 NAV 日期；收益从最近的冻结前锚点净值计算，并保留 2026-09-23 暖启动净值供核对。9 月 24 日决策从账户输入中移除，避免在 9 月 28 日开盘执行冻结前预测。9 月 23 日暖启动决策的委托若于 9 月 24 日开盘成交，列为冻结前暖启动执行，不算新交易。9 月 28 日 NAV 首日变化标注为冻结前持仓的隔夜/当日持有收益；9 月 28 日收盘信号最早 9 月 29 日才能执行。`pre_freeze_diagnostic_dates` 单独列出冻结前日期；20 个共同冻结后日期不足时不计算 20 日区块 bootstrap 区间。`account_full_span_summaries_including_viewed_warmup` 含暖启动及诊断区间，不能拿来宣称冻结后收益。报告同时列出逐日选择、固定 P9、公共专家 overlay、P8 简单趋势、现金和 CSI 300 价格代理的共同日期比较。公共专家 overlay 只改冻结 P9 候选中已成熟且满足 20 个不重叠样本/95% 下界为正的效果；否则当日预测与固定 P9 相同。指数价格代理是合成基线，未被验证为可交易 ETF。

自 2026-09-28 起，三日预测至少需要 123 个冻结后交易日，才可能累计 120 个新成熟决策日。计数要求当天趋势与市场相对收益两个冻结候选篮子中的股票都有完整、截至观察日已可用的三日标签；单只股票先结算不算一个成熟组合决策。此前状态只能是等待，不能宣布前向通过或失败。真正新日期来临后，研究规则和冻结文件不得依照收益重新选择；若源数据修订导致冻结核对失败，应先记录差异并单独决定是否启动新的研究，而不是覆写本次前向证据。
