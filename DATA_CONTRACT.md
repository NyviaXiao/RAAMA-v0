# 数据契约：本地训练行情与 BaoStock 历史数据

本契约记录已经观测到的字段和当前研究入口的行为。字段来源、范围、缺失值、文件间比较与原始文件哈希见 [DATA_AUDIT.md](DATA_AUDIT.md)。后复权口径依据用户提供的 BaoStock 来源和单点接口比对采用；数据单位与历史盘中发布时间未由源文件证明。

## 研究数据边界

- 唯一训练行情源是外部数据目录中的 `train.csv`。生产加载器 `load_training_bars(data_dir)` 只读取这个固定文件名，不合并其他文件，也不改写源文件。
- `stock_data.csv` 包含评分区间日期，不能作为训练入口；`test.csv` 仅可用于数据审计和未来隔离的最终本地评分器，不能进入研究数据层、特征、目标、训练、归一化、可靠性、状态识别、调参或组合决策。
- `hs300_stock_list.csv` 只有 2026-03-16 一个成分快照。用它回测 2024 年以来的“历史沪深300”会产生幸存者偏差。研究加载器使用单独缓存的 BaoStock 历史日期快照。
- `load_research_data(data_dir, baostock_run_dir)` 只读 `train.csv`、指定 BaoStock 采集目录中的历史成分、行情和 `sh.000300` 指数。它不会读 `test.csv` 或 `stock_data.csv`。
- 信号用后复权行情由本地 `train.csv` 提供 300 个代码；其余 48 个历史成分从 BaoStock `adjustflag=1` 补齐。执行用未复权行情则为 348 个历史成员单独保存在 `daily-unadjusted/`。

## 规范行情记录

| 字段 | `train.csv` 来源列 | Python 类型 | 当前语义 |
| --- | --- | --- | --- |
| `symbol` | `股票代码` | `str` | 1 至 6 位 ASCII 数字左补零，保留六位字符串，例如 `000001` |
| `trade_date` | `日期` | `datetime.date` | ISO 日期；仅为交易日标签，没有日内时刻 |
| `open` | `开盘` | `float` | 原始数值尺度 |
| `high` | `最高` | `float` | 原始数值尺度 |
| `low` | `最低` | `float` | 原始数值尺度 |
| `close` | `收盘` | `float` | 原始数值尺度 |
| `volume` | `成交量` | `float \| None` | 原始数值尺度；单位未证实 |
| `amount` | `成交额` | `float \| None` | 原始数值尺度；币种/单位未证实 |
| `amplitude` | `振幅` | `float \| None` | 原始数值尺度；不擅自除以 100 |
| `price_change` | `涨跌额` | `float \| None` | 原始数值尺度 |
| `turnover` | `换手率` | `float \| None` | 原始数值尺度；不擅自除以 100 |
| `return_1d` | `涨跌幅` | `float \| None` | 原始数值尺度；不擅自除以 100 |

`load_training_bars` 返回上述字段的字典列表，不生成收益标签或特征。`load_research_data` 保留本地训练源中的这些值，并为来自 BaoStock 的历史成员映射同一字段：`return_1d` 来自 `pctChg`，`turnover` 来自 `turn`，`price_change` 由 `close - preclose` 得到，`amplitude` 由 `(high - low) / preclose × 100` 得到。执行行情与后复权信号行情单独保存，不相互覆盖。

外部 CSV 进入系统时检查：必需列齐全、列名不重复、行宽正确、代码和日期可解析、数值有限、OHLC 满足 `low <= open/close <= high`，以及规范化后的 `(symbol, trade_date)` 唯一。错误必须直接抛出，不尝试备用文件。

## 时间点规则

- `event_time`：对日线行情而言是 `trade_date`；对成员快照而言是 BaoStock 返回的 `updateDate`。两者都是日期标签，不是盘中时间戳。
- `available_at`：研究快照中采用 `datetime.date` 的日期粒度。日线记录只在当日收盘后进入该日快照，信号最早在下一交易日执行；因此 `snapshot_at(D)` 允许 `trade_date <= D`，但策略不得在 D 日收盘价上假设同日成交。
- 成分快照的 `available_at` 按交易日历设为严格晚于源 `updateDate` 的第一个交易日。源没有公告时间戳；这是一项保守使用规则，不等于精确历史公布时刻。
- 历史快照只返回 `available_at <= decision_date` 的行情和成分。`decision_date` 必须是已取得的沪深300交易日。
- `fetched_at` 是本系统获取数据的时间，不等于历史公告、发布或生效时间。周度成员快照能按上述日期规则重放，但不能据此声称已恢复其正式公告时点。

本地 `train.csv` 的一个重叠样本与 BaoStock `adjustflag=1` 的 OHLC、成交量、成交额、换手率和涨跌幅一致；项目据此结合用户提供的来源说明使用 BaoStock 后复权口径。BaoStock 的 `adjustflag=1` 为后复权、`adjustflag=3` 为不复权；原始复权因子没有纳入本轮必要数据。价格和成交量单位仍沿用源数值，不做额外缩放。五交易日目标、交易成本和具体成交规则留在后续研究协议中确定。

## 后续数据类型提案：只定义字段，不实现

所有未来记录保留 `source`、`source_record_id`、`fetched_at`、`raw_response_ref`、`available_at`；时间戳应带时区，并按来源文档分别保存事件、公布、生效和修订时间。

| 类型 | 额外字段建议 | 时间点重点 |
| --- | --- | --- |
| `FundamentalRecord` | `symbol, metric, fiscal_period, value, unit, published_at, revision_id` | 财务期间结束日不等于报告公开日；修订保留版本 |
| `AnnouncementRecord` | `announcement_id, symbol, announcement_type, published_at` | 以实际公告公开时间决定可见性 |
| `MacroRecord` | `series_id, observation_period, value, unit, released_at, revision_id` | 观察期不等于公布日；保留初值与修订值 |
| `IndexMembershipRecord` | `index_id, symbol, effective_from, effective_to, announced_at` | 调整公告日和成分生效日分别保存 |

这些是未来来源接入的最小语义提案，不表示本地数据已含这些字段，也不构成 Phase 1 的实现 API。
