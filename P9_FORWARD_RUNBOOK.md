# P9 前向影子账户操作说明

P9 当前冻结于 2026-09-23 收盘。这个日期及以前的行情、模型和影子账户仅用于暖启动，**不计入前向成绩**。入口、冻结哈希与截止日回放分别见 `scripts/run_p9_forward_validation.py`、`P9_P12_FROZEN_MANIFEST_20260927.json` 和 `experiments/p9_forward_cutover_20260927_v9/`。

在 BaoStock 能正常连接、目标交易日已完整收盘后，先为该决策日生成点时专家快照，再运行前向账户：

```powershell
.venv\Scripts\python.exe scripts\acquire_phase2_market_data.py --oos-start-date 2026-03-16 --oos-end-date YYYY-MM-DD
# In the current PowerShell session, configure POPAGENT_LLM_URL, POPAGENT_LLM_MODEL (optional), and POPAGENT_LLM_KEY from your local secret store.
.venv\Scripts\python.exe scripts\run_p9_p12_research.py --decision-time 'YYYY-MM-DDTHH:MM:SS+08:00' --horizon-sessions 3 --account-positions-csv experiments\p9_forward_前一日\p9_fixed_market_relative\positions.csv --baostock-run-dir data\raw\baostock\fetch-实际采集批次 --output-dir experiments\p9_p12_forward\snapshots\YYYY-MM-DD
Remove-Item Env:\POPAGENT_LLM_URL, Env:\POPAGENT_LLM_MODEL, Env:\POPAGENT_LLM_KEY -ErrorAction SilentlyContinue
.venv\Scripts\python.exe scripts\run_p9_forward_validation.py --as-of-date YYYY-MM-DD --baostock-run-dir data\raw\baostock\fetch-实际采集批次 --expert-snapshot-root experiments\p9_p12_forward\snapshots --frozen-manifest P9_P12_FROZEN_MANIFEST_20260927.json --output-dir experiments\p9_forward_YYYY-MM-DD
```

采集目录必须从 2026-03-16 开始，覆盖整个暖启动期和新日期；第二条命令的 `--as-of-date` 必须等于该目录最后一个交易日。运行会核对已查看期间行情、冻结的代码/配置/训练状态/候选预测哈希，并重新确认 2026-09-23 的模型和账户状态没有变化。每天使用新的输出目录，保留旧批次和失败日志。

专家快照的股票范围按决策日最新可见的 CSI 300 成分与上次账户的非零持仓取并集；可用 `--candidate-symbols-file` 追加当日候选（每行六位代码）。每日需把前一日账户 `positions.csv` 传给 `--account-positions-csv`，不要沿用 9/23 的默认切换账本。每次最多请求 3 份近 30 日公告全文 PDF；Qwen 输入上限可用 `--llm-max-records` 控制。`pypdf` 需先按 `pyproject.toml`/`uv.lock` 安装。公告 PDF 或文本解析不可用时，快照会保留明确失败状态；历史 13.6 万条公告全文不会批量下载。Qwen 状态、token 用量、估算成本、延迟及文档哈希都写入专家快照。

若未配置 `POPAGENT_LLM_KEY`，Qwen 明确跳过；有凭证且存在新分析记录时，必须同时配置 `POPAGENT_LLM_URL`。`POPAGENT_LLM_MODEL` 可选，默认 `qwen3.7-plus`。不要把 endpoint 或凭证值写入仓库。

`forward_validation_report.json` 的 `new_date_account_results` 只列 2026-09-23 **之后**的账户净收益及费用；`account_full_span_summaries_including_viewed_warmup` 混合了暖启动期，不能拿来宣称前向收益。报告同时列出逐日选择、固定 P9、公共专家 overlay、P8 简单趋势、现金和 CSI 300 价格代理的共同日期比较。公共专家 overlay 只改冻结 P9 候选中已成熟且满足 20 个不重叠样本/95% 下界为正的效果；否则当日预测与固定 P9 相同。指数价格代理是合成基线，未被验证为可交易 ETF。

三日预测需要至少 123 个新交易日，才可能累计 120 个新成熟决策日。计数要求当天趋势与市场相对收益两个冻结候选篮子中的股票都有完整、截至观察日已可用的三日标签；单只股票先结算不算一个成熟组合决策。此前状态只能是等待，不能宣布前向通过或失败。真正新日期来临后，研究规则和冻结文件不得依照收益重新选择；若源数据修订导致冻结核对失败，应先记录差异并单独决定是否启动新的研究，而不是覆写本次前向证据。
