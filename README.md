# Risk-Aware Adaptive Multi-Agent Portfolio Decision System

本仓库从零开始，不复用旧项目代码。当前包含数据审计、训练数据边界、BaoStock 单源接口、时点数据层、20 日动量与五日目标的诊断基线；尚未实现多 Agent、可靠性门控、风险优化或交易回测。系统决策链与模块责任见 [ARCHITECTURE.md](ARCHITECTURE.md)，阶段顺序与验收门槛见 [EXECUTION_PLAN.md](EXECUTION_PLAN.md)。

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

在本项目目录的 PowerShell 中执行（需要 Python 3.11 或更新版本）：

```text
python -m pip install -e ".[dev]"
python -m pytest -q
```

测试采用 pytest 作为统一入口；覆盖本地训练数据边界、BaoStock 响应错误和格式错误。真实 API 采集只在用户显式运行采集脚本或授权采集时执行。
