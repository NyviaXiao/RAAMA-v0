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

## 验证

在项目目录的 PowerShell 中运行：

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

测试覆盖预测成熟时点、待结算日报、换手按实际组合变动计算、风险越界最小减仓、单票集中度、现金与费用约束、停牌/部分成交后的账户续接，以及账本净值核对。完整开发与已查看诊断重放命令、数据窗和研究结论见 [P8 最终报告](P8_FINAL_REPORT_20260926.md)。

## 研究边界

模拟使用名义金额账本，不表示真实股数、券商订单确认或逐笔成交；未模拟涨跌停排队与完整公司行动现金流。风险估计采用历史个股波动与单一市场因子，不使用完整协方差或 CVaR。研究收益不构成实盘验证。若预测误差超过估计收益，维持现金可能是当前约束下的正确输出；这说明系统不认可这些交易的净增益，不代表策略取得收益。
