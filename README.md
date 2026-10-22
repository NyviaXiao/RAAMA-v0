# Risk-Aware Adaptive Multi-Agent Portfolio Decision System

这是沪深300日线研究与组合决策项目。P8 提供每个交易日收盘后的点时预测与组合优化、下一交易日执行模拟、实际现金/持仓账本、成熟预测结算和可审计结果。它是可由应用调用的 Python 决策 API 与 JSON 命令行入口；项目没有交互式应用界面或网络服务。

完整研究口径、结果和限制见 [P8 最终报告](P8_FINAL_REPORT_20260926.md)、[MVP 研究记录](MVP_RESEARCH.md)、[M1 固定期限研究](M1_FIXED_HORIZON_RESEARCH.md)、[P9 前向研究报告](P9_FINAL_REPORT_20260927.md) 和 [P9–P12 实施验证补充](P9_P12_FINAL_REPORT_20260927.md)。模块职责见 [架构说明](ARCHITECTURE.md)，阶段验收依据见 [P8 完成清单](P8_COMPLETION_PLAN.md)。`base.md` 仅作为可讨论的项目参考。

## 数据边界

BaoStock 是唯一外部市场数据源。研究期本地行情仅读 `train.csv`；时间外研究只读指定的 BaoStock 缓存。任何研究、特征、训练、组合、调参和日报入口都不读取 `test.csv`。六位股票代码保留为字符串，历史成分按决策当时可见的快照确定。研究入口默认使用已有缓存；AKShare 公开资料另由 [PUBLIC_DATA.md](PUBLIC_DATA.md) 中的显式采集命令写入独立原始缓存。

开发缓存 `data/raw/baostock/fetch-20260924T115050Z` 覆盖 2024-01-02 至 2026-03-06。时间外诊断缓存 `data/raw/baostock/fetch-20260924T150656Z` 覆盖 2026-03-16 至 2026-09-23；该区间已被查看，只能作为已查看诊断，不能称为新系统的未见测试。

## 配置与重放

三个 P8 配置把“预测期限”和“复审间隔”分开：

| 配置 | 预测期限 | 复审间隔 | 用途 |
|---|---:|---:|---|
| `configs/p8_daily_three_session.toml` | 3 个交易日 | 每 1 个交易日 | P8 主配置 |
| `configs/p8_five_session_review_three_session.toml` | 3 个交易日 | 每 5 个交易日 | 固定预测期限，对比每日复审与每五个交易日复审 |
| `configs/p8_daily_five_session.toml` | 5 个交易日 | 每 1 个交易日 | 固定每日复审，单独检查预测期限变化 |

每五个交易日复审按交易日序号计数，不按日历周计算。相同预测期限的复审对照使用相同标签期限；每日三日与每日五日对照在相同决策日期生成观点，但预测标签的结束日期相差两个交易日。

以下命令使用本地 BaoStock 开发缓存和本地训练目录。`--processed-name` 为本次运行指定单独的工件目录，避免覆盖其他结果：

```powershell
.venv\Scripts\python.exe scripts\run_mvp_research.py --training-data-dir "C:\Users\xiao\Desktop\THU-BDC2026\data" --baostock-run-dir "data\raw\baostock\fetch-20260924T115050Z" --config configs\p8_daily_three_session.toml --output experiments\p8_daily_three_session_final_research.json --processed-name p8_daily_three_session_final
.venv\Scripts\python.exe scripts\run_mvp_research.py --training-data-dir "C:\Users\xiao\Desktop\THU-BDC2026\data" --baostock-run-dir "data\raw\baostock\fetch-20260924T115050Z" --config configs\p8_five_session_review_three_session.toml
.venv\Scripts\python.exe scripts\run_mvp_research.py --training-data-dir "C:\Users\xiao\Desktop\THU-BDC2026\data" --baostock-run-dir "data\raw\baostock\fetch-20260924T115050Z" --config configs\p8_daily_five_session.toml
```

每次运行会生成研究汇总、预测决策与结算账、每日订单/持仓/现金/净值文件、账户决策账和成熟训练种子。概要位于 `experiments/`，逐日账本位于对应的 Git 忽略目录 `data/processed/`。结果记录配置、BaoStock 清单和训练种子的 SHA-256。

## 应用调用与 JSON 契约

应用可直接调用 [decision.py](src/adaptive_mas/decision.py) 中的 `create_daily_decision(research, config, as_of_date, portfolio, training_state, versions)`。传入截至 `as_of_date` 可见的 `ResearchData`、冻结配置、实际现金和持仓市值，以及只含已成熟标签的训练种子；返回 `DailyDecision`，可用 `to_dict()` 序列化为 JSON。预测期限内尚未出现的价格不会阻止决策生成，输出的 `prediction_status` 为 `pending`，建议订单标记为下一交易日待执行。

输入账户 JSON 必须且只能包含 `cash_cny` 与 `holdings_cny`：

```json
{
  "cash_cny": 250000.0,
  "holdings_cny": {
    "000001": 500000.0,
    "600000": 250000.0
  }
}
```

`schemas/decision_api.v1.schema.json` 描述该账户输入与 `adaptive-mas.decision.v1` 输出。命令行入口会加载已缓存的 BaoStock 研究数据和对应的开发期成熟训练种子，并写出单日决策 JSON：

```powershell
.venv\Scripts\python.exe scripts\run_daily_decision.py --baostock-run-dir "data\raw\baostock\fetch-20260924T150656Z" --portfolio-state-json "portfolio-state.json" --training-seed "data\processed\p8_daily_three_session_final\training_seed.json.gz" --as-of 2026-09-23 --output "daily-decision.json"
```

未提供 `--as-of` 时，入口使用指定缓存中最新可用交易日。它不发单；输出订单只表示待执行建议。`target_cash_cny` 是扣费前的目标现金，`estimated_cash_after_costs_cny` 已扣预计交易成本。模拟账本中的 `cash_cny` 和净值按实际成交及费用更新。

`risk_adjusted_net_incremental_value_cny` 表示与当前持仓继续不动相比的组合目标函数增量，已纳入成熟样本外误差惩罚、买卖方向成本和组合风险惩罚。风险惩罚金额单独提供；买入、卖出名义金额分别列出。暖启动期间尚无成熟样本外误差时，API 明确返回 `warmup_no_action`，保留实际组合，不生成优化订单。

每张建议订单还带有历史成交额估算的单日容量、决策日 BaoStock 交易状态，以及“次日执行状态未确认”的标记。下一交易日是否停牌、是否有足够成交额和实际成交金额只能在执行时由市场数据或模拟账本确认；计划订单不会标记为已成交。`evidence.expert_matured_performance` 提供三个专家各自的损失门控权重、近窗口平均成熟 Rank-IC 和滚动成熟样本外 RMSE。

## RAAMA contextual RL 训练闭环

`run_raama_decision.py` 默认 `--exploration-rate 0`，表示保守地沿用固定 P9；空策略在这个设置下会一直保持 P9。开发期训练必须显式设置非零探索率，并给出评价起始日。起始日及之后会冻结探索和策略更新；回放以这个日期为硬边界校验决策标签。训练示例使用 `1.0` 让每个历史训练日都采样一个当日可用动作，使 20 条/动作门槛在约 279 个训练日内有实际达到的可能。该命令须按日期顺序逐日运行，并只用历史模拟器生成成交和 NAV；不可用于用户真实资金或下真实订单。日期须处于所用缓存范围内：

```powershell
.venv\Scripts\python.exe scripts\run_raama_decision.py --baostock-run-dir "data\raw\baostock\fetch-20260924T150656Z" --portfolio-state-json "shadow-portfolio-state.json" --as-of 2026-09-22 --evaluation-start-date 2026-09-29 --horizon-sessions 3 --risk-budget-annualized 0.20 --exploration-rate 1.0 --seed 7 --rl-state-file "state-day1.json" --output "daily-decision.json"
```

训练反馈须来自历史模拟器的成交和含费用 NAV；实时保守模式（默认零探索）也可使用实际账户 NAV。两种情况下，都可把单日决策 JSON（也接受 JSONL）直接交给反馈入口；输出的策略状态可供下一次决策加载：

```powershell
.venv\Scripts\python.exe scripts\replay_contextual_rl_feedback.py --decision-log "daily-decision.json" --daily-nav-csv "daily-nav.csv" --evaluation-start-date 2026-09-29 --as-of 2026-09-23 --initial-policy-json "state-day1.json" --feedback-output "mature-feedback.jsonl" --report-output "feedback-report.json" --policy-output "state-after-feedback.json"
```

下一次日决策将 `--rl-state-file` 指向新建的 `state-after-feedback.json`；回放输出路径须保持新路径，因为它只创建新策略文件。只有已执行账户的成熟 NAV 结果会更新策略；回放会保持幂等，评价期结果不会训练。风险窗口需有 20 个账户日收益。探索种子按决策日期派生，同日可复现、跨日变化。补充挑战目前不进入资金决策，因为尚无预先校准的数字映射；Qwen 角色调用与补充挑战成本会列入决策成本，并在反馈奖励中合计扣除一次。成交费用已由 NAV 收益反映，不会在奖励中再扣第二次。此入口和短跑尚未证明收益改善。

`scripts/run_contextual_rl_development.py` 提供一次按日期顺序训练、冻结、再评价的数值策略研究，并在运行目录写入 `frozen_contextual_rl_state.json`。该文件使用 `contextual-rl-state.v1`，可由上述日报入口直接读取；在冻结评价边界及之后加载时保持零探索与冻结策略更新：

```powershell
.venv\Scripts\python.exe scripts\run_contextual_rl_development.py --output-dir ".tmp\contextual-rl-development-run"
.venv\Scripts\python.exe scripts\run_raama_decision.py --baostock-run-dir "data\raw\baostock\fetch-20260924T150656Z" --portfolio-state-json "portfolio-state.json" --as-of 2026-09-23 --evaluation-start-date 2025-09-01 --horizon-sessions 3 --risk-budget-annualized 0.20 --exploration-rate 0 --rl-state-file ".tmp\contextual-rl-development-run\frozen_contextual_rl_state.json" --output "daily-decision.json"
```

策略只学习数值融合动作的成熟 NAV 反馈。`no_query` 是现有全部动作的查询设置；RL 尚未学习是否调用补充分析，Qwen 意见也不会改变数值策略或组合权重。分析查询动作与其历史效果证据均未实现。历史评价即使完整跑完，也不能视为未见样本或实盘收益验证。

2026-10-12 的点时重放记录在 `.tmp/codex-contextual-rl-forward-20261012-paused-final/`。279 个训练决策日中，600837 因无当日成交数据且风险窗口从 39/40 个有效收益样本继续缩短，导致 122 日暂停；其余 157 日选择了动作，138 条成熟反馈进入策略。各非 P9 动作在部分市场状态下未达到 20 条，本地控制器改用跨状态的合并计数与 Q 值；四个可用的非 P9 动作合并后分别有 34、29、24、21 条成熟反馈，均达到门槛。P9、trend、reversal、liquidity、balanced 的合并 Q 值依次为 +0.001145、-0.001211、-0.002893、-0.006023、-0.002670，因此冻结的 120 日评价全部选择 P9，非 P9 动作日为 0。RL 与 P9 的累计收益均为 -1.2986%，最大回撤均为 -11.9821%。这次结果验证了数值反馈更新与 P9 保守选择的运行链路；没有证明 RL 收益改善，也没有实现自主补充分析或 Qwen 观点影响权重。

本地公告/新闻与 Qwen 归档审查（2026-10-12）未发现可训练 query/no-query 的历史样本：研究目录只有一个已完成的专家快照（2026-09-27），其中 Qwen 仅有一条基于结构化财务字段的方向记录，0 条成熟结果、0 条证据质询、309 条待结算记录；该快照在此前 P9 数值回放日期之后。公告缓存虽有约 11.3 万条记录，但由 2026-09-27 批量抓取，公告发布时间精度仅到日期，快照标记财务/供应商 vintage 未验证；当次没有获取公告全文，新闻只匹配到 1 条 CLS 记录。因此它们不能代表早期决策日可见内容，也没有 query 与 no-query 的同状态配对结果。诚实训练此动作至少还需要：覆盖训练/评价日的不可变逐日新闻与公告标题/正文、发布时间和首次可见时间；同一输入与策略状态下留存的 query/no-query 决策、Qwen 模型/提示版本、数值输出和调用成本；以及两种决策各自成熟的同账户 NAV 或前向收益。新增历史文本还需有合法的数据来源/使用授权；未来执行付费 Qwen 调用需单独获准。本地现有材料不足以校准 Qwen 对预测、候选排序或组合权重的作用。

## Agentic RL 历史研究（2026-10-21）

这项回顾研究新增线性 Expected SARSA 的查询/融合调度、基本面与文本残差校准、查询费现金扣款和统一 A/B/C 账户回放。只读 `train.csv` 与已有 BaoStock、AKShare 缓存；回填财务/公告的历史 vintage 未核实，因此结果不是盲测或真实历史归档验证。

先离线生成固定日期分区的请求清单并审阅 `outbound_summary.json` 与 payload SHA-256。仅在审阅和明确继续后发送；请求数、manifest SHA 和费用上限必须与本次审阅一致：

```powershell
.venv\Scripts\python.exe scripts\prepare_agentic_rl_qwen.py prepare
.venv\Scripts\python.exe scripts\prepare_agentic_rl_qwen.py send --expected-sha256 88b4c7395ab265552d9e3b913a01a39b795720a6c286dbb02942048b776296d3 --max-calls 73 --max-budget-cny 5.0808 --exclude-event-id 31c749f3cedcc4123db730aabcb9d0b70eab8c93133fa27d270d898715fec091
.venv\Scripts\python.exe scripts\run_agentic_rl_historical.py --output-dir experiments\agentic_rl_historical_20261021\replay
```

`send` 只发送公开财务事实、公告标题及其公开时间信息、证券代码、决策日和期限；内部 P9 预测、目标权重、策略标识、引用和源记录 ID 仅保留在本地清单与回放。它只使用已配置的北京 Qwen endpoint，密钥从 `data/raw/qwen_key.txt` 读入进程环境，不重试投递未知的调用。Schema 无效的响应会先原样保存在 `qwen_raw_responses/`，记录具体校验原因、用量和费用后排除训练/评价，并继续其他独立请求；重新运行时已完成和无效事件只跳过，不重发。未知投递仍需使用下方单事件排除参数；已知实际费用、未知事件预留和新请求上限会合并校验总预算。请求清单为回顾分析，不能当作当时可见的归档内容。研究步骤的即时奖励为 0，同日折扣为 1；响应中的自报充分性、冲突和证据数量只作为查询后的状态特征。查询价值只依据后续账户净收益、风险和费用评价，查询费从现金/NAV 扣一次。重放生成冻结策略/残差模型、训练/验证/评价账户、逐日决策与两阶段转移轨迹，以及 `AGENTIC_RL_HISTORICAL_REPORT.md`；运行会拒绝覆盖已有输出目录。结果解释应区分基本面校准、稀疏文本证据、RL 调度和账户差异，不能宣称收益提升具有普遍因果性。

历史回放结束后，可从冻结策略和文本残差模型恢复并对一个留存历史状态做确定性推断：

```powershell
.venv\Scripts\python.exe scripts\infer_agentic_rl_frozen.py --artifact experiments\agentic_rl_historical_20261021\replay\frozen_agentic_policy.json --state-trace experiments\agentic_rl_historical_20261021\replay\C_evaluation_state_trace.jsonl --decision-date 2025-09-15
```

`run_raama_decision.py` 当前仍调用原 `contextual_rl` 控制器并读取 `contextual-rl-state.v1`；它尚未加载这份双阶段 Agentic RL 工件。上面的推断脚本用于单日历史策略恢复检查，不代表日常决策入口已切换。

授权的 Qwen 标题回顾实验（2026-10-12）结果见[简要报告](experiments/qwen_retrospective_dry_run_20261012/QWEN_RETROSPECTIVE_REPORT_20261012.md)。81 个训练事件中，79 条有效响应进入成熟标签，1 条投递状态不明、1 条响应无效均未计分。P11 在 79 条上得到 28 个非重叠日期、平均方向性超额收益 -1.0793%、95% 下界 -2.8217%；正向门槛未通过，收益调整为 0，研究停止。37 条评价请求未发送，也未进行 RL/query 回放。该研究属于事后合成：公告与财务 vintage 未核实，模型输入只有公告标题和结构化财务指标、没有公告全文；服务商账单尚未核对。因此这项结果不能证明策略改善，也不能作为严格点时或样本外验证。

## 验证

在项目目录的 PowerShell 中运行：

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

测试覆盖预测成熟时点、待结算日报、换手按实际组合变动计算、风险越界最小减仓、单票集中度、现金与费用约束、停牌/部分成交后的账户续接，以及账本净值核对。完整开发与已查看诊断重放命令、数据窗和研究结论见 [P8 最终报告](P8_FINAL_REPORT_20260926.md)。

## 研究边界

模拟使用名义金额账本，不表示真实股数、券商订单确认或逐笔成交；未模拟涨跌停排队与完整公司行动现金流。风险估计采用历史个股波动与单一市场因子，不使用完整协方差或 CVaR。研究收益不构成实盘验证。若预测误差超过估计收益，维持现金可能是当前约束下的正确输出；这说明系统不认可这些交易的净增益，不代表策略取得收益。

## Historical Agentic RL delivery (2026-10-22)

Two-stage Expected SARSA learns query and fusion decisions, with fundamental/text residual calibration and once-debited research costs. See [delivery results and boundaries](AGENTIC_RL_HISTORICAL_ACCEPTANCE.md) and [offline prompt/context v3](docs/AGENTIC_RL_PROMPT_CONTEXT_V3.md). The frozen historical experiment did not demonstrate a return improvement; the new artifact is not wired into the legacy daily entry point. Existing local data and cached responses are required for replay and are not included in this source backup.
