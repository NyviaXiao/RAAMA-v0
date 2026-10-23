# Harness Python 与 JSON 接口

## 作用

Harness 负责公共资料研究和候选范围筛选。它不会生成预期收益、持仓金额或交易指令；这些继续由成熟校准预测与原有组合优化器产生。现有单日输出维持 `adaptive-mas.decision.v1`，账户输入维持 `cash_cny` / `holdings_cny` JSON 契约，因此 app 可复用既有决策消费者。

## Python API

主入口为 `adaptive_mas.harness_runtime.create_harness_daily_decision`。调用方提供加载好的点时 `ResearchData`、决策配置、日期、真实账户名义金额、成熟训练状态和本地缓存路径。调用方还必须提供训练 seed 的已知截止日和预测期限：现有 seed JSON 不自带这两项元数据，不能从其数组内容推断。

```python
from datetime import date
from pathlib import Path

from adaptive_mas.data import load_oos_research_data
from adaptive_mas.decision import PortfolioState
from adaptive_mas.harness_runtime import create_harness_daily_decision

research = load_oos_research_data(Path("data/raw/baostock/fetch-20260924T150656Z"))
decision = create_harness_daily_decision(
    research,
    config,
    date(2026, 9, 23),
    PortfolioState(cash_cny=250_000, holdings_cny={"600000": 250_000}),
    training_state,
    public_cache_dir="data/raw/akshare",
    harness_version_dir="data/processed/harness_versions",
    run_dir="experiments/gepa_harness/daily/2026-09-23/run-id",
    training_seed_trained_through=date(2026, 3, 6),
    training_seed_horizon_sessions=3,
    training_seed_cutoff_source="repository_default_seed_protocol",
    training_seed_horizon_source="repository_default_seed_protocol",
    call_ledger_path="experiments/gepa_harness/calls.jsonl",
    response_cache_dir="experiments/gepa_harness/response_cache",
)
payload = decision.to_dict()
```

API 在构造 Qwen 客户端或发请求前校验账户、期限、风险预算、交易日、训练 seed 截止日、seed 期限和校准预测成熟度。seed 的 `trained_through` 必须早于 `as_of`，其期限必须等于 `config["prediction_horizon_sessions"]`。暖启动时返回 `warmup_no_action`，不发研究请求。正式路径在研究前算一次校准预测，并将同一个诊断交给最终决策，避免重复回放。

`precomputed_research_result` 可传入已完成研究任务的 JSON 对象，适用于 GEPA `reuse_demo`。它必须与决策日、当前激活的 harness 版本及程序性经验哈希一致；该路径不会产生新的 provider 请求，任务新增 API 成本记为 0，来源任务实际成本单列。

## 公共缓存时点模式

默认模式 `captured_public_cache_cutoff` 要求来源的 `available_at` 和真实 `fetched_at` 都不晚于决策时点，适合正常前向使用。对旧的回顾性样本，可显式设置 Python 参数 `retrospective_public_cache=True` 或 CLI 开关 `--retrospective-public-cache`。该模式只对本地已缓存的财务指标、财报和公告标题使用来源披露/可用日期代理；真实 `fetched_at` 原样保留，晚于 `as_of` 的资料标记 `vintage_verified=false`。公告正文、新闻、宏观和关注度数据不采用该代理。它不证明供应商历史版本或当时系统已捕获，因此 `strict_historical_pit_verified` 始终为 `false`。不下载新数据。

回顾性模式只应用于明确标注为历史代理的评估或演示；日常前向决策保持默认严格捕获时点模式。

## 同账户短窗口比较

`scripts/run_harness_paired_simulation.py` 接收同一份已完成 harness 研究 JSON、账户 JSON、seed 和决策日期，使用一次校准预测分别生成基准与 harness 候选筛选决策，再交给现有 `simulate_portfolio` 在同一初始现金/持仓上模拟一个调仓与指定交易日数的持有窗口。入口只复用研究 JSON，不创建模型客户端或发送 provider 请求；输入研究版本必须与当前活动 harness 版本相符。模拟结果是历史反事实，不能当成真实成交或投资绩效证明。

## 安全导出活动版本

`scripts/export_harness_version.py --output "active-harness-prompts.json"` 只导出活动版本的六段提示文本、版本/内容哈希和固定 schema 标识，不导出运行轨迹、公开来源目录、账户、预测、选择评价或本地路径。导出前会核对活动程序性经验的内容哈希。

## CLI 与 JSON 文件

命令行入口为 `scripts/run_harness_decision.py`。`--portfolio-state-json` 接受既有账户 JSON：

```json
{
  "cash_cny": 250000.0,
  "holdings_cny": {
    "600000": 250000.0
  }
}
```

自定义 training seed 必须从 provenance 一并给出 `--training-seed-trained-through` 和 `--training-seed-horizon-sessions`。项目默认 seed 的代码库来源约定为截止日 `2026-03-06`、期限 3 个交易日；这是维护的来源记录，不是 seed 文件内部的自描述字段。

```powershell
.venv\Scripts\python.exe scripts\run_harness_decision.py `
  --baostock-run-dir "data\raw\baostock\fetch-20260924T150656Z" `
  --portfolio-state-json "portfolio-state.json" `
  --as-of 2026-09-23 `
  --horizon-sessions 3 `
  --risk-budget-annualized 0.20 `
  --training-seed "data\processed\p8_daily_three_session_final\training_seed.json.gz" `
  --training-seed-trained-through 2026-03-06 `
  --training-seed-horizon-sessions 3 `
  --output "daily-decision.json"
```

输出是 `DailyDecision.to_dict()` 的 JSON，遵循 [`schemas/decision_api.v1.schema.json`](schemas/decision_api.v1.schema.json)。关键 harness 字段：

- `versions.training_seed_trained_through`、`versions.training_seed_horizon_sessions` 记录用于检查的来源声明；对应的 `*_source` 字段记录来源标识。
- `versions.harness_version_id` 和 `versions.harness_candidate_sha256` 标识被加载的提示/程序性经验版本。
- `evidence.harness.research` 保存三角色结构化发现、证据目录和工具轨迹；所有证据来自本地公共缓存。
- `evidence.harness.candidate_filter` 记录经理引用支持的候选与校准预测股票池的交集。它不修改校准收益。
- `evidence.harness.research_cost_accounting` 使用该任务的 `task_usage` 记录逻辑研究成本、新增 API 成本、缓存复用和来源任务成本。
- `evidence.harness.decision_cost_accounting` 把任务逻辑研究成本与预计交易成本各计一次。实际账户 NAV 回放如需入账，应扣除该任务逻辑研究成本一次；不可把累计账单再次摊到一个决策上。

## 预算、缓存与失败状态

日常决策最多 10 次模型请求。演化与日常入口使用同一个实验根目录时，共享 160 次 provider-call 硬上限和响应缓存。任务级逻辑成本按模型轮次记录，精确响应缓存仍保留可比较的逻辑成本，但新增 API 成本为零。已启动而无法确认结果的请求留作 unresolved 状态；在人工核对前，后续调用会停止，避免重复扣费。

请求限制为每次最多 24,000 UTF-8 输入字节和 1,200 个输出 token。出站提示只含决策日、点时公共市场/财务/公告信息、专家交接和固定公共评价诊断。账户现金/持仓、私有预测、优化器目标、交易指令、`test.csv` 行和本地路径不发给模型。



## 已完成运行摘要

2026-10-23 的 GEPA 运行生成一个唯一提案并拒绝；活动版本仍为 seed，没有投资改善证据。已完成的日常 API 演示复用研究 JSON，产生 hold 决策、0 个订单、0 次新增 provider 请求。短窗口模拟两臂均未交易，观测差异只来自逻辑研究成本计入方式。

可发布的汇总、seed 提示、候选差异和审计见 [安全运行报告](reports/harness_20261023/RUN_REPORT.md)。原始运行日志、请求账本、精确响应缓存和账户/模拟细节留在本机，未作为发布链接或安全包文件。
