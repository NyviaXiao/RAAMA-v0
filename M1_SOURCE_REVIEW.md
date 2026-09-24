# BaoStock 接入与 M1 进度

更新时间：2026-09-24（Asia/Shanghai）

## 结论

根据用户提供的来源信息和一次定点 API 调用，本项目采用 **BaoStock 作为唯一行情数据源**。不接入 Tushare，也不为交叉验证引入第二个数据商。BaoStock Python SDK 在 PyPI 标注为 BSD License，平台公开提供股票和指数日线、沪深300成分查询等接口；本项目已能成功登录并读取所需记录。[BaoStock Python SDK](https://pypi.org/project/baostock/)，[BaoStock 官方站点](https://www.baostock.com/)

## 定点调用结果

同一 BaoStock 会话内完成以下读取：

- 查询日期 `2024-01-02` 的沪深300快照返回 300 行，`updateDate` 为 `2024-01-01`，字段为 `updateDate, code, code_name`。
- 浦发银行 `sh.600000` 的未复权日线接口返回字段 `date, code, open, high, low, close, preclose, volume, amount, adjustflag, turn, tradestatus, pctChg`。2024-01-02 的原始收盘价为 6.60，成交量为 22,066,700，成交额为 146,066,303.72。
- `sh.000300` 指数日线接口返回 `date, code, open, high, low, close, preclose, pctChg`。
- 本地 `train.csv` 同日浦发银行收盘价为 78.86858760。BaoStock `adjustflag=1` 后复权接口返回同一 OHLC 数值；成交量、成交额、换手率和涨跌幅也相同。由此将本地训练行情按 BaoStock 后复权行情使用。未复权 `adjustflag=3` 行情将单独保存，用于交易价格与执行模拟，不覆盖或拼接到后复权列。
- 按本地训练日历每周请求历史成分快照，成功保存 112 个快照、33,600 条成员记录，覆盖 348 个不同股票代码。快照保留源 `updateDate`，不含可验证的历史公告发布时间。

这次抽样足以确定当前数据接法与本地训练价口径；不继续做全文件或多源交叉审计。

## 已实现的最小数据边界

- [baostock_source.py](src/adaptive_mas/data/baostock_source.py)：单一来源会话，提供历史成分、显式复权标记的个股日线和指数日线；SDK/接口错误直接抛出。
- [acquire_phase2_market_data.py](scripts/acquire_phase2_market_data.py)：只从 `train.csv` 确定研究日期范围；按周取得历史成分快照，为历史成分并集拉取 `adjustflag=3` 日线，并为本地训练集未覆盖的成员获取 `adjustflag=1` 日线；另取得 `sh.000300` 指数序列。支持显式续传。每份响应保存来源、请求参数、UTC 抓取时间、行数和 SHA-256。
- 原始响应落在 `data/raw/baostock/`，由 `.gitignore` 排除，不会提交到代码仓库。采集范围截至训练集最后日期，不读取评分数据文件。
- 本次清单共 509 个响应、240,296 行：112 个成员快照（33,600 行）、348 只未复权日线（181,381 行）、48 只补充后复权日线（24,791 行）、沪深300指数 524 行。采集区间由 `train.csv` 决定，截至 2026-03-06。
- BaoStock 数据源边界、Phase 1 与 P2 时点层测试共 14 项通过。

## 点时和研究流程约定

- 行情 `event_time` 使用源字段 `date`；成分快照保留 `updateDate`。不创建未被接口提供的盘中时间。
- 日线信号在交易日结束后形成，最早在下一交易日执行。实际代码按日期过滤，避免同一日完整 OHLCV 被用于同日成交。
- 历史成分快照按周采样；缓存文件名记录请求日，成员行保留 BaoStock `updateDate`。读取层不把抓取时间当成历史可用时间。
- 日线源没有历史盘中发布时间。P2 的 `decision_date` 表示收盘后决策：当日完整日线仅在当日收盘后可见，最早下一交易日执行；成员快照则在源 `updateDate` 后的第一个交易日可见。这是日期粒度研究约定，不是供应商提供的精确发布时间戳。
- M1 与必要的原始行情采集已完成。交易费用、无法成交规则、基准收益口径和训练/验证窗口留到第一组基线实验前冻结。

## 后续数据边界

对当前架构必要的数据限于：历史成分快照及其成员行情、匹配日期的 `sh.000300` 日线。现有本地训练行情作为 BaoStock 后复权输入保留。基本面、新闻、宏观及分钟线目前都不需要。

AKShare 是另一个开源财经数据接口库，覆盖的品类很多；但当前 BaoStock 已提供本项目所需的核心接口，因此暂不添加第二套 SDK，避免增加依赖与数据口径分叉。[AKShare 官方项目](https://github.com/akfamily/akshare)
