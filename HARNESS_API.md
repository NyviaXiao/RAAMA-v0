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

## 本地工作台、worker 与执行账本

本地工作台提供回环地址上的中文 API 和静态界面。API 只允许 `127.0.0.1`/`localhost`，任务仅接受 `decision_reuse`，会在子进程环境中移除 provider API key；必须另行启动 worker 才会处理队列。两者默认共用 `data/processed/harness_service/harness_state.sqlite3`，状态和账本结果不会写入 GEPA 的 `experiments/` 历史目录：

```powershell
.venv\Scripts\python.exe -m adaptive_mas.harness_service serve
.venv\Scripts\python.exe -m adaptive_mas.harness_service worker
```

打开 `http://127.0.0.1:8765`。HTTP API 包含 `/api/health`、`/api/tasks`、`/api/evaluations`、`/api/execution-ledgers` 和 `/api/releases`。任务用 SQLite `BEGIN IMMEDIATE` 领取，租约过期后可恢复；同一 idempotency key 重复提交相同输入会复用原任务，改变输入则冲突。发布与回滚接口要求 `RAAMA_RELEASE_MANAGER_TOKEN` Bearer token，角色由服务端验证，不接受请求体中的角色声明。工作台保留脱敏摘要和本地受控明细；不设置 token 时发布按钮不可用。

本地 SQLite 默认保存 task/trace/span、阶段、延迟、预算/费用摘要、结果与恢复事件。遥测默认不会外发；需要 OTLP 时显式安装 `observability` 可选依赖，并设置 `RAAMA_OTEL_EXPORTER_OTLP_ENDPOINT`。导出采用字段脱敏，不包含账户持仓、提示正文、凭证和本地路径。未配置 endpoint 时，服务完全依赖本地检查记录。

执行账本 API 接收 `ledger_id` 和 `request`。`request` 的字段对应 `simulate_execution_ledger`：`execution_mode`（`nominal_research` 或 `quantity`）、必填 `prediction_linkage_mode`（`linked` 或明确的 `independent`）、严格递增的 `trading_dates`、`market_bars`、`orders`、`rule_pack`、`initial_cash_cny`，以及可选 `initial_lots`、`initial_holdings_cny`、`corporate_actions`、`source_artifact_hashes`、`prediction_records` 和 `risk_limits`。`linked` 模式要求订单引用唯一存在的同证券预测 ID、匹配预测日与期限；预测须在预测日收盘后可用，并且订单只能在下一交易日开盘执行。没有预测链路的独立账户回放必须显式指定 `independent`，不得填写预测 ID。预测数值仅写入标签账，不改变 NAV。同一 `ledger_id` 只能对应相同输入；重复相同请求返回已保存结果。账本返回逐日现金/NAV/股数核对、成交/受限订单、费用、滑点、风险检查、独立预测标签账和实际规则包哈希。两种模式都不会连接券商。

规则包必须明确规则 ID、生效日期、HTTPS 来源标题/链接、逐证券交易规则与费用配置；系统不提供未经核实的通用市场参数。规则审核应采用适用交易所和证券类别的当前官方规则，例如[上交所 2026 年交易规则修订](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml)及[深交所现行交易规则公告](https://www.szse.cn/lawrules/rule/trade/current/t20260424_620190.html)，并单独版本化实际费率和适用证券规则。未录入的数据不会被自动猜测。公司行动必须附带来源引用和与本地文件相符的 SHA-256。

独立评估 API 的 `POST /api/evaluations` 只接受三种操作：`predeclare`（冻结交易日历、分区、purge/embargo 和经济风险门槛）、`register_package`（登记含训练 seed/state、校准、模型、数据、工具 schema 的完整包；可通过 `package_path` 读取项目目录内 JSON 文件）和 `evaluate`（指定已登记协议/包、唯一 run ID、分区和项目目录内证据文件）。评估器从文件哈希和指标自行计算结果，不接受调用方提供的“通过”标志。`GET /api/evaluations` 显示冻结协议、包和已计算证据；没有成熟的前向影子记录时状态保持 `pending_forward`。

生产协议使用 `adaptive-mas.release-evaluation-protocol.v2`，预先固定 `evaluation_scope_sha256`、共同 `execution_context_sha256`、训练基准候选 ID/哈希、交易日历、预测期限和分区边界。独立验收的候选与基准必须使用相同起始资金/持仓/股数、交易模式、风险限制、规则费率、结算行情和公司行动；实际订单与预测必须逐项对应各自的哈希决策工件。相同冻结数据范围跨协议 ID 共享独立验收占用与已查看事件/预测区间；标签成熟日必须至少晚于预测日 `horizon_sessions` 个冻结交易日，再满足下一分区的 purge/embargo。

生产评估输入必须使用 `adaptive-mas.release-evaluation-input.v2`，来源清单为每个工件登记唯一 `artifact_id`、语义 `role`、项目内路径和 SHA-256。开发/选择行引用候选与基准的完整 research-result 工件；评估器校验来源包、候选/实现身份、原始研究源快照、市场快照及捕获时点，并重新运行研究评分。独立验收行还要引用执行请求和完整账本结果；请求中的市场 bars、公司行动和费用规则必须逐项匹配哈希来源工件，评估器重算执行账本并要求 NAV、现金、股数和费用结果完全一致，再从日 NAV 重新计算净收益、风险、回撤和成本。行内自报的质量、收益、风险、回撤或费用数字不参与通过判断。旧版汇总输入只可在明确的测试模式中用于 fixture 生命周期测试，无法登记为生产证据。

前向影子行必须引用 `decision` 工件 ID/hash 与持久任务 ID。服务端从自己的任务 SQLite 库读取完成状态、原决策 JSON、请求日期/期限以及创建和完成时间，并核对决策候选版本/哈希、日期与期限；单独导入的 `observed_at` 声明不构成前向证据。缺失或不匹配的持久任务保持 `pending_forward`，测试 fixture 不可发布到生产。



## 已完成运行摘要

2026-10-23 的 GEPA 运行生成一个唯一提案并拒绝；活动版本仍为 seed，没有投资改善证据。已完成的日常 API 演示复用研究 JSON，产生 hold 决策、0 个订单、0 次新增 provider 请求。短窗口模拟两臂均未交易，观测差异只来自逻辑研究成本计入方式。

可发布的汇总、seed 提示、候选差异和审计见 [安全运行报告](reports/harness_20261023/RUN_REPORT.md)。原始运行日志、请求账本、精确响应缓存和账户/模拟细节留在本机，未作为发布链接或安全包文件。
