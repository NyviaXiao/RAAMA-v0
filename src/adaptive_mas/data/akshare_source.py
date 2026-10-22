"""AKShare boundary for point-in-time public company, news, and macro data."""

import re
from datetime import date, datetime
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import requests
from bs4 import BeautifulSoup


FINANCIAL_STATEMENTS = ("资产负债表", "利润表", "现金流量表")
SHANGHAI = ZoneInfo("Asia/Shanghai")
NBS_RELEASE_INDEX = "https://www.stats.gov.cn/sj/zxfb/"
CLS_NEWS_INDEX = "https://www.cls.cn/telegraph"


def _vendor_code(code: str) -> tuple[str, str, str]:
    exchange, separator, symbol = code.partition(".")
    if (
        not separator
        or exchange not in {"sh", "sz"}
        or len(symbol) != 6
        or not symbol.isascii()
        or not symbol.isdigit()
    ):
        raise ValueError(f"BaoStock 证券代码格式无效：{code!r}")
    return symbol, f"{exchange}{symbol}", f"{symbol}.{exchange.upper()}"


def _validate_date_range(start_date: date, end_date: date) -> None:
    if start_date > end_date:
        raise ValueError("开始日期不能晚于结束日期")


def _nbs_published_at(soup: BeautifulSoup) -> str:
    published_meta = soup.find("meta", attrs={"name": re.compile("^PubDate$", re.I)})
    if published_meta is None:
        return ""
    value = published_meta.get("content", "")
    for date_format in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M"):
        try:
            published = datetime.strptime(value, date_format)
            return published.replace(tzinfo=SHANGHAI).isoformat()
        except ValueError:
            continue
    raise ValueError(f"国家统计局 PubDate 格式无效：{value!r}")


class AkShareSource:
    """Thin, single-source adapter; AKShare and upstream errors propagate."""

    def financial_statement(self, code: str, statement: str) -> pd.DataFrame:
        _, sina_code, _ = _vendor_code(code)
        if statement not in FINANCIAL_STATEMENTS:
            raise ValueError(f"不支持的财务报表类型：{statement!r}")
        return ak.stock_financial_report_sina(stock=sina_code, symbol=statement)

    def financial_indicator(self, code: str) -> pd.DataFrame:
        symbol, _, eastmoney_code = _vendor_code(code)
        if symbol.startswith(("4", "8")):
            raise ValueError(f"沪深300股票代码无效：{code!r}")
        return ak.stock_financial_analysis_indicator_em(
            symbol=eastmoney_code, indicator="按报告期"
        )

    def disclosures(
        self, code: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        symbol, _, _ = _vendor_code(code)
        _validate_date_range(start_date, end_date)
        return ak.stock_zh_a_disclosure_report_cninfo(
            symbol=symbol,
            market="沪深京",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )

    def recent_market_news(self) -> pd.DataFrame:
        frame = ak.stock_info_global_cls()
        frame = frame.rename(columns={"标题": "新闻标题", "内容": "新闻内容"}).copy()
        frame["新闻来源"] = "财联社电报"
        frame["新闻链接"] = CLS_NEWS_INDEX
        frame["链接范围"] = "栏目页"
        return frame

    def user_focus_index(self, code: str) -> pd.DataFrame:
        symbol, _, _ = _vendor_code(code)
        return ak.stock_comment_detail_scrd_focus_em(symbol=symbol)

    def macro_cpi(self) -> pd.DataFrame:
        return ak.macro_china_cpi()

    def macro_pmi(self) -> pd.DataFrame:
        return ak.macro_china_pmi()

    def nbs_macro_releases(
        self, start_month: str = "2024-01", max_pages: int | None = None
    ) -> pd.DataFrame:
        month_match = re.fullmatch(r"(\d{4})-(\d{2})", start_month)
        if month_match is None or not 1 <= int(month_match.group(2)) <= 12:
            raise ValueError(f"宏观发布起始月份格式无效：{start_month!r}")
        if max_pages is not None and max_pages < 1:
            raise ValueError("NBS 列表页数必须至少为 1")
        links: dict[str, tuple[str, str, str, str]] = {}
        reached_start = False
        pages_to_read = max_pages if max_pages is not None else 100
        for page_number in range(pages_to_read):
            page_url = (
                NBS_RELEASE_INDEX
                if page_number == 0
                else urljoin(NBS_RELEASE_INDEX, f"index_{page_number}.html")
            )
            index_response = requests.get(page_url, timeout=20)
            index_response.raise_for_status()
            soup = BeautifulSoup(index_response.content, "html.parser")
            page_periods = []
            for anchor in soup.find_all("a", href=True):
                title = anchor.get_text(" ", strip=True)
                cpi_match = re.search(r"(\d{4})年(\d{1,2})月份居民消费价格", title)
                pmi_match = re.search(r"(\d{4})年(\d{1,2})月中国采购经理指数", title)
                match = cpi_match or pmi_match
                if match is None:
                    continue
                period = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
                page_periods.append(period)
                if period < start_month:
                    continue
                url = urljoin(page_url, anchor["href"])
                parsed_url = urlparse(url)
                if parsed_url.hostname != "www.stats.gov.cn":
                    raise ValueError(f"国家统计局链接域名无效：{url!r}")
                series_id = "cpi" if cpi_match else "pmi"
                links[url] = (series_id, period, title, page_url)
            if max_pages is not None and page_number + 1 == max_pages:
                reached_start = True
                break
            if page_periods and min(page_periods) <= start_month:
                reached_start = True
                break
        if not reached_start:
            raise RuntimeError(
                f"NBS 最近 {pages_to_read} 个列表页没有到达起始月份 {start_month}"
            )

        rows = []
        for url, (series_id, period, title, listing_page) in links.items():
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            detail_soup = BeautifulSoup(response.content, "html.parser")
            released_at = _nbs_published_at(detail_soup)
            rows.append(
                {
                    "series_id": series_id,
                    "observation_month": period,
                    "released_at": released_at,
                    "announcement_title": title,
                    "announcement_url": url,
                    "listing_page": listing_page,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "series_id",
                "observation_month",
                "released_at",
                "announcement_title",
                "announcement_url",
                "listing_page",
            ],
        )
