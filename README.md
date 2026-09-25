# Risk-Aware Adaptive Multi-Agent Portfolio Decision System

这是一个从零实现的沪深300量化多智能体研究项目。当前已完成 P3–P8 首个可重放研究 MVP，并按优化路线图完成 M1 首版：除原连续持仓协议外，新增严格匹配五日目标期限的计划退出协议、预测决策账与成熟结算账，以及停牌跨期后按实际持仓调整后续买单。完整实验协议、开发期与时间外数字、实现限制见 [MVP_RESEARCH.md](MVP_RESEARCH.md) 和 [M1_FIXED_HORIZON_RESEARCH.md](M1_FIXED_HORIZON_RESEARCH.md)；模块职责见 [ARCHITECTURE.md](ARCHITECTURE.md)，阶段计划见 [EXECUTION_PLAN.md](EXECUTION_PLAN.md)。

当前实验显示开发区间的自适应融合未超过等权基线；时间外只有 21 个评价窗口，不足以支持稳定性结论。模拟采用名义金额账本，不记录真实股数或券商成交回报，也未实现涨跌停排队、完整协方差优化或 CVaR。结果用于研究，不构成实盘策略验证。尚未实现文本 LLM、新闻/基本面 Agent、辩论、记忆或强化学习。

## 数据边界

研究运行器通过 BaoStock P2 点时数据层读取训练期间行情与成分快照；本地 `train.csv` 用于训练期代码与日期边界。OOS 运行器只读取明确指定的 BaoStock 时间外缓存。研究源码不加载或引用 `test.csv`，该文件不进入训练、特征、决策、组合和调参。原始外部 CSV 保持在原目录，不复制、不修改；BaoStock 原始/派生缓存由 Git 忽略。

## 环境与验证

在项目目录的 PowerShell 中运行：

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q -p no:cacheprovider
```

已验证项目虚拟环境下 30 项测试全部通过。

## 重放研究实验

依赖当前机器已有的 BaoStock 缓存；本次没有新增下载：

```powershell
.venv\Scripts\python.exe scripts\run_mvp_research.py --training-data-dir "C:\Users\xiao\Desktop\THU-BDC2026\data" --baostock-run-dir "data\raw\baostock\fetch-20260924T115050Z"
.venv\Scripts\python.exe scripts\run_mvp_oos_research.py --baostock-run-dir "data\raw\baostock\fetch-20260924T150656Z"
.venv\Scripts\python.exe scripts\run_mvp_research.py --training-data-dir "C:\Users\xiao\Desktop\THU-BDC2026\data" --baostock-run-dir "data\raw\baostock\fetch-20260924T115050Z" --execution-protocol fixed_horizon_v2
.venv\Scripts\python.exe scripts\run_mvp_oos_research.py --baostock-run-dir "data\raw\baostock\fetch-20260924T150656Z" --execution-protocol fixed_horizon_v2
```

公开实验汇总保存在 `experiments/`。大型逐日决策、订单、持仓及净值轨迹保存在 Git 忽略的 `data/processed/mvp/`。
