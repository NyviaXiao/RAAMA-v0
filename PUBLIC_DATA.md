# AKShare 公共资料入口

本说明覆盖财报、财务指标、巨潮资讯公告、轻量新闻和少量宏观资料的采集与点时读取。行情、历史指数成分和价格仍由 BaoStock 提供；本入口不改变行情缓存、P8/P9 模型、前向账户、模拟器或交易决定。

## 当前验证和覆盖

2026-09-27 的小样本接口检查记录如下。AKShare 数据读取由主任务在获准联网的环境执行；本地沙箱不能直接连接上游站点。

| 资料 | 接口与观察到的样本 | 点时字段及限制 |
|---|---|---|
| 财务报表 | `stock_financial_report_sina(stock="sh600000", symbol="利润表")`：108 行、94 列；包含 `报告日`、`公告日期`、`更新日期`。2026-06-30 报告的公告日期为 2026-08-28。 | 报告期不是可用日。新浪日期只有日粒度，读取层将紧接披露日的日历日设为最早可见日；策略只在交易日收盘决策，故披露日不会进入当日决策。源未提供完整历史修订版本，记录带 `vintage_verified=false`。 |
| 财务指标 | `stock_financial_analysis_indicator_em(symbol="600000.SH", indicator="按报告期")`：108 行、141 列；包含 `REPORT_DATE`、`NOTICE_DATE`。 | 按 `NOTICE_DATE` 而不是 `REPORT_DATE` 筛选；公告日只有日粒度，也采用次日可见。历史值是否后来重述无法由当前接口证明。 |
| 巨潮公告 | `stock_zh_a_disclosure_report_cninfo(symbol="600000", market="沪深京", ...)`：2024-01 区间返回 1 条，2026-08 返回 9 条。 | `公告时间` 实际为 `YYYY-MM-DD`，没有时分秒；标题与公告链接保留，最早次日可见。没有把日期伪装成精确发布时间。 |
| 个股新闻 | `stock_news_em` 最近个股新闻接口在联网样本中因 `search-api-web.eastmoney.com` TLS 连接中断而失败，不作为已验证入口，也不做静默回退。 | 当前固定轻量入口选用已通的 `stock_info_global_cls()`：只返回最近 20 条市场快讯，按候选/持仓公司名在标题和内容中匹配。没有逐股覆盖保证，也不回填启动日前新闻。AKShare 未给每条快讯原文链接；缓存中的链接明确指向财联社栏目页（`链接范围=栏目页`），不可当作原文链接。 |
| 宏观 | `macro_china_cpi()` 样本返回 224 行，`月份` 是观察期而非发布日期；`macro_china_pmi()` 同样没有正式发布日期。国家统计局官方文章的 `meta name="PubDate"` 已实测解析成功：CPI 2026-08 于 2026-09-09 09:30 发布，PMI 2026-08 于 2026-08-31 09:30 发布（北京时间）。 | 只有能与国家统计局官方发布文章按系列和观察月份匹配的值才可读取，并且还要求 AKShare 数值响应的抓取时间不晚于决策时点。历史抓取从当前首页开始分页，遇到起始月份（默认 2024-01）即停止，最多 100 页；日更只取首页。全量历史页尚未回填入本地缓存。 |

真实日更小样本命令：

```powershell
.venv\Scripts\python.exe scripts\acquire_public_data.py --mode daily --symbol 600000 --as-of-date 2026-09-27 --output-dir data\raw\akshare-smoke-20260927
```

该次清单实际保存了三张财报 108/108/102 行、财务指标 108 行、当日巨潮公告 0 行、CPI 224 行、PMI 224 行、国家统计局正式发布日期 2 行、财联社快讯 20 行。按当日时点读取可见财报 108/108/102 行、指标 108 行、CPI 1 行、PMI 1 行、新闻 20 行。首轮 PMI 请求遇到 `ChunkedEncodingError`，失败清单已保存；同命令重跑只重试失败 PMI 并继续完成 NBS/新闻，没有重复抓取成功响应。

财报接口返回的是采集时可见的供应商历史表。`公告日期`/`NOTICE_DATE` 能防止把未来报告期提前使用，但不能证明今天返回的数值与公告当日的原始数值完全一致。因此本入口记录内容版本和抓取时间，并在读取结果标记版本未验证；不声称具备完整的财报修订时点历史。

历史成员池只从指定 BaoStock 缓存的 `membership/asof-*.csv` 快照并集构造，不请求最新沪深300名单。两段目录的离线聚合实测得到 365 个不同成员、140 个快照，日期范围为 2024-01-02 至 2026-09-21。开发缓存 `fetch-20260924T115050Z` 覆盖至 2026-03-02；已查看的时间外诊断缓存 `fetch-20260924T150656Z` 从 2026-03-16 起。清单记录 7 段超过 8 个日历日没有新快照的区间；这些区间可能覆盖春节、国庆等休市时间，不代表已核实的交易日缺失。采集清单还记录每个 BaoStock 目录、其 `manifest.jsonl` 哈希、快照数和成员数；不读取 `test.csv`。

## 采集

原始资料写入 Git 忽略的 `data/raw/akshare/`。AKShare 已解析出的 DataFrame 原样按列序列化为 UTF-8 CSV；这不是供应商 HTTP 原始报文。每个响应按内容 SHA-256 保存到 `objects/`，`manifest.jsonl` 记录来源、AKShare 接口、实际请求参数、实体代码、抓取时间、响应行数、内容哈希和缺口。抓取失败会追加失败记录并重新抛出；重新运行只跳过已有成功响应，失败请求可重试。

历史资料以 BaoStock 缓存快照并集为代码池。可传多个目录合并断开的快照区间；来源目录及其 manifest 哈希、成员数、快照范围和超过 8 日的日历间隔写入 `universe-sources.jsonl`。只做一个代码的接口样本时加 `--symbol`：

```powershell
.venv\Scripts\python.exe scripts\acquire_public_data.py --mode history --baostock-run-dir data\raw\baostock\fetch-20260924T115050Z --baostock-run-dir data\raw\baostock\fetch-20260924T150656Z
.venv\Scripts\python.exe scripts\acquire_public_data.py --mode history --baostock-run-dir data\raw\baostock\fetch-20260924T115050Z --symbol 600000 --output-dir data\raw\akshare-sample
```

历史回填逐股票抓取三张财务报表、按报告期财务指标和该代码在快照范围内的巨潮公告，并按 `--macro-start-month`（默认 `2024-01`）为 CPI/PMI 发布文章读取有限页的国家统计局栏目。它不采集新闻历史。开发缓存与第二个已查看的后续诊断缓存均只贡献其原 BaoStock 成分快照；这不使已查看区间成为未见测试。示例不读取 `test.csv`，也不声称快照日历区间等于交易日缺失。

随后完成的单股历史实测使用了上述两段 BaoStock 快照并集和 `600000`：三张财报分别归档 108/108/102 行，财务指标 108 行，2024-01 至 2026-09 巨潮公告 269 行；CPI、PMI 各归档 224 行，并取得 2 条国家统计局正式发布日期记录。以 2025-06-30 16:00（北京时间）读取时，巨潮公告为 155 条（最新公告日 2025-06-28），财务指标为 103 条（最新公告日 2025-04-30）；未来公告已排除，指标仍标记 `vintage_verified=false`。这验证的是单一股票，不代表 365 个历史成员已完成回填。

每日运行时，把当日候选股和持仓代码逐个传入。每日入口更新这些代码的财报/指标与当日公告，采集 CPI、PMI、当前国家统计局发布页，并归档一次财联社最近 20 条快讯：

```powershell
.venv\Scripts\python.exe scripts\acquire_public_data.py --mode daily --symbol 600000 --symbol 000001
```

新闻从第一次运行开始增量归档；不把 AKShare“最近 20 条”扩写成多年新闻库。财联社接口没有逐条链接或股票代码，筛选由消费者用公司名匹配标题/内容完成。

## 最小读取接口

```python
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from adaptive_mas.data import load_public_records

decision_time = datetime(2026, 9, 23, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
announcements = load_public_records(
    Path("data/raw/akshare"), "announcement", decision_time, symbols={"sh.600000"}
)
news = load_public_records(
    Path("data/raw/akshare"), "news", decision_time, company_names={"浦发银行"}
)
```

数据集名称为 `financial_statement_balance_sheet`、`financial_statement_income_statement`、`financial_statement_cash_flow_statement`、`financial_indicator`、`announcement`、`news`、`macro_cpi`、`macro_pmi`。`decision_time` 必须带时区。读取记录包含源字段以及 `published_at`、`available_at`、`source_record_id`、`fetched_at`、`raw_response_ref`、`content_sha256` 和 `vintage_verified`。新闻不能传 `symbols`，因为所选快讯源不提供个股代码。

同一缓存反复读取会先校验原始响应哈希，再重建相同记录。财务报告和公告按真实披露日期限制可见性；新闻按来源发布时间和本系统抓取时间共同限制；宏观值按国家统计局 PubDate 与 AKShare 值抓取时间共同限制。缺少这些时点信息的记录不会进入读取结果。

## 缺口与不可声称范围

- 已完成日更小样本和 `600000` 单股历史样本；365 个历史成员的全量财报、指标和公告回填仍未跑完，因此当前仓库不包含也不声称包含全量 AKShare 历史原始数据。
- 最近宏观发布时间、两页分页和 PubDate 解析均已用真实响应验证；单股历史样本归档了 2 条国家统计局正式发布日期记录。完整的 `2024-01` 起历史发布日期覆盖仍未完成；宏观读接口对没有 PubDate 证据或尚未抓取的月份返回空。
- 当前新闻是财联社栏目快讯，不是逐股新闻；单次 20 条、可能没有目标公司相关内容，且每行链接是栏目页，不是文章原文链接。
- 财报历史值的供应商修订版本无法从当前 API 恢复；时间字段能阻止未来会计期穿越，不能证明每个历史数值都是当年首发版本。
- 上游 API 变更、封锁或返回空会被记录为缺口并抛错；没有引入第二新闻源或自动备用源。
