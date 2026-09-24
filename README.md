# Risk-Aware Adaptive Multi-Agent Portfolio Decision System

本仓库从零开始，不复用旧项目代码。当前包含 BaoStock 时点数据层、三个确定性量化专家、五日成熟结果驱动的基础可靠性门控，以及开发区间和锁定时间外区间的滚动诊断。当前成本仅为声明假设的情景计算，完整订单执行、风险优化、市场状态与文本 LLM Agent 仍属于后续阶段。系统决策链与模块责任见 [ARCHITECTURE.md](ARCHITECTURE.md)，阶段顺序与验收门槛见 [EXECUTION_PLAN.md](EXECUTION_PLAN.md)。

## 数据入口

当前唯一研究数据入口读取指定数据目录中的 `train.csv`：

```python
from adaptive_mas.data import load_training_bars

bars = load_training_bars(r"C:\Users\xiao\Desktop\THU-BDC2026\data")
```

加载器不读取 `test.csv` 或 `stock_data.csv`。源 CSV 保持在原目录，不复制到本项目。字段与时间点规则见 [DATA_CONTRACT.md](DATA_CONTRACT.md)，实测限制和审计结果见 [DATA_AUDIT.md](DATA_AUDIT.md)。

P2 研究数据层把信号用后复权日线、执行用未复权日线、沪深300指数和可用成分快照分开提供：

```python
from datetime import date
from adaptive_mas.data import load_research_data

research = load_research_data(
    r"C:\Users\xiao\Desktop\THU-BDC2026\data",
    r"data\raw\baostock\fetch-20260924T115050Z",
)
snapshot = research.snapshot_at(date(2024, 1, 2))
```

## BaoStock 原始数据

采集脚本只从 `train.csv` 确定日期范围，读取周度沪深300历史成分快照、348 个历史成员的未复权日线、本地训练集未覆盖 48 个成员的后复权日线，以及 `sh.000300` 指数日线。原始响应、抓取时间与 SHA-256 保存在 `data/raw/baostock/`，该目录已排除在 Git 跟踪之外。续传可通过 `--resume-run-dir` 指向已有采集目录。

本次数据包位于 `data/raw/baostock/fetch-20260924T115050Z/`，截至 2026-03-06；基础采集已完成，清单含 509 个响应。重跑时可使用：

重新采集方式：

```text
python scripts/acquire_phase2_market_data.py --training-data-dir "C:\Users\xiao\Desktop\THU-BDC2026\data"
```

## 验证

在获得 BaoStock 原始数据后运行第一组固定参数诊断：

```text
python scripts/run_momentum_diagnostic.py --training-data-dir "C:\Users\xiao\Desktop\THU-BDC2026\data" --baostock-run-dir "data\raw\baostock\fetch-20260924T115050Z"
```

当前结果见 [baseline_momentum.json](experiments/baseline_momentum.json)。这是每五个交易日抽样一次的历史横截面 Rank IC 诊断，不是扣成本后的交易回测或独立最终测试。

量化 Agent 研究重放：

```text
python scripts/run_agent_research.py --training-data-dir "C:\Users\xiao\Desktop\THU-BDC2026\data" --baostock-run-dir "data\raw\baostock\fetch-20260924T115050Z"
```

实现了 20 日趋势、5 日反转和成交额确认三个独立打分专家。融合以等权为基线；自适应门控仅使用此前已成熟窗口的 Rank IC，前 20 个窗口等权暖启动。真实数据指标、权重轨迹、配置哈希和限制见 [AGENT_RESEARCH.md](AGENT_RESEARCH.md) 与 [agent_research.json](experiments/agent_research.json)。这些 Top-Decile 数字是未扣成本的信号诊断，不表示已验证可交易策略。

冻结后的时间外区间使用 BaoStock 2026-03-16 至 2026-09-23 数据，独立于本地未来评分文件；采集和运行方式如下：

```text
python scripts/acquire_phase2_market_data.py --oos-start-date 2026-03-16 --oos-end-date 2026-09-23 --resume-run-dir "data\raw\baostock\fetch-20260924T150656Z"
python scripts/run_oos_agent_research.py --baostock-run-dir "data\raw\baostock\fetch-20260924T150656Z"
```

21 个非重叠窗口及成本假设下的净收益见 [OOS_RESEARCH.md](OOS_RESEARCH.md) 和 [oos_agent_research.json](experiments/oos_agent_research.json)。这仍是 Top-Decile 收益诊断，没有模拟真实成交、停牌延期退出或按账户规模收取的最低佣金。

在本项目目录的 PowerShell 中执行（需要 Python 3.11 或更新版本）：

```text
python -m pip install -e ".[dev]"
python -m pytest -q
```

测试采用 pytest 作为统一入口；覆盖本地训练数据边界、BaoStock 响应错误和格式错误、样本外独立加载、门控成熟历史初始化与成本公式。当前 22 项测试通过。真实 API 采集只在用户显式运行采集脚本或授权采集时执行。
