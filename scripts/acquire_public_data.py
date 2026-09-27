"""Backfill and incrementally archive point-in-time public data."""

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from adaptive_mas.data.akshare_source import AkShareSource, FINANCIAL_STATEMENTS
from adaptive_mas.data.public_data import archive_failure, archive_frame, request_is_cached


ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "akshare"
STATEMENT_NAMES = {
    "资产负债表": "balance_sheet",
    "利润表": "income_statement",
    "现金流量表": "cash_flow_statement",
}


def _stock_code(symbol: str) -> str:
    if len(symbol) != 6 or not symbol.isascii() or not symbol.isdigit():
        raise ValueError(f"股票代码必须是六位 ASCII 数字：{symbol!r}")
    if symbol.startswith("6"):
        return f"sh.{symbol}"
    if symbol.startswith(("0", "3")):
        return f"sz.{symbol}"
    raise ValueError(f"不支持的沪深300股票代码：{symbol!r}")


def _membership_union(run_dirs: list[Path]) -> tuple[list[str], date, date, list[date], list[dict[str, object]]]:
    resolved_run_dirs = [path.resolve() for path in run_dirs]
    if len(resolved_run_dirs) != len(set(resolved_run_dirs)):
        raise ValueError("BaoStock 历史缓存目录不能重复传入")
    codes = set()
    snapshot_dates = []
    source_records = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"BaoStock 采集目录不存在：{run_dir}")
        manifest_path = run_dir / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"BaoStock 采集目录缺少清单：{manifest_path}")
        paths = sorted((run_dir / "membership").glob("asof-*.csv"))
        if not paths:
            raise FileNotFoundError(f"没有 BaoStock 历史成分快照：{run_dir / 'membership'}")
        run_dates = []
        run_codes = set()
        for path in paths:
            query_date = date.fromisoformat(path.stem.removeprefix("asof-"))
            snapshot_dates.append(query_date)
            run_dates.append(query_date)
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames is None or not {"updateDate", "code"}.issubset(reader.fieldnames):
                    raise ValueError(f"BaoStock 成分快照缺少必需列：{path}")
                rows = list(reader)
            if not rows:
                raise ValueError(f"BaoStock 成分快照没有成分记录：{path}")
            for row in rows:
                code = row["code"]
                exchange, separator, symbol = code.partition(".")
                if (
                    not separator
                    or exchange not in {"sh", "sz"}
                    or len(symbol) != 6
                    or not symbol.isascii()
                    or not symbol.isdigit()
                ):
                    raise ValueError(f"BaoStock 成分代码无效：{code!r}（{path}）")
                if date.fromisoformat(row["updateDate"]) > query_date:
                    raise ValueError(f"成分快照 updateDate 晚于请求日期：{path}")
                codes.add(code)
                run_codes.add(code)
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        source_records.append(
            {
                "run_dir": str(run_dir.resolve()),
                "baostock_manifest_sha256": manifest_sha256,
                "snapshot_count": len(paths),
                "snapshot_start": min(run_dates).isoformat(),
                "snapshot_end": max(run_dates).isoformat(),
                "distinct_member_count": len(run_codes),
            }
        )
    dates = sorted(set(snapshot_dates))
    calendar_intervals = [
        {
            "after": left.isoformat(),
            "before": right.isoformat(),
            "gap_days": (right - left).days - 1,
        }
        for left, right in zip(dates, dates[1:])
        if (right - left).days > 8
    ]
    return sorted(codes), min(dates), max(dates), dates, [
        *source_records,
        {"snapshot_calendar_intervals": calendar_intervals},
    ]


def _record_universe_sources(
    cache_dir: Path,
    source_records: list[dict[str, object]],
    snapshot_dates: list[date],
    member_codes: list[str],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    calendar_intervals = source_records[-1]["snapshot_calendar_intervals"]
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": "BaoStock cached membership snapshots",
        "runs": source_records[:-1],
        "snapshot_count": len(snapshot_dates),
        "snapshot_start": min(snapshot_dates).isoformat(),
        "snapshot_end": max(snapshot_dates).isoformat(),
        "distinct_member_count": len(member_codes),
        "snapshot_calendar_intervals": calendar_intervals,
    }
    record["content_id"] = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    with (cache_dir / "universe-sources.jsonl").open(
        "a", encoding="utf-8", newline=""
    ) as target:
        target.write(json.dumps(record, ensure_ascii=False) + "\n")


def _capture(
    cache_dir: Path,
    dataset: str,
    provider: str,
    endpoint: str,
    parameters: dict[str, str],
    fetch,
    required_columns: set[str],
    entity_code: str | None,
    collection_date: str | None,
) -> bool:
    if request_is_cached(
        cache_dir, dataset, endpoint, parameters, entity_code, collection_date
    ):
        return False
    try:
        frame = fetch()
    except requests.exceptions.RequestException as error:
        archive_failure(
            cache_dir,
            dataset,
            provider,
            endpoint,
            parameters,
            datetime.now(timezone.utc),
            error,
            entity_code,
            collection_date,
        )
        raise

    missing = required_columns - set(frame.columns)
    gaps = []
    if frame.empty:
        gaps.append("empty_response")
    if missing:
        gaps.append(f"missing_required_columns:{','.join(sorted(missing))}")
    provenance_fields = {
        "financial_indicator": ("NOTICE_DATE",),
        "announcement": ("公告时间", "公告链接"),
        "news": ("发布日期", "发布时间", "新闻来源", "新闻链接"),
    }
    if dataset.startswith("financial_statement_"):
        provenance_fields[dataset] = ("公告日期",)
        gaps.append("financial_value_vintage_unverified")
    if dataset == "financial_indicator":
        gaps.append("financial_value_vintage_unverified")
    for column in provenance_fields.get(dataset, ()):
        if column in frame.columns:
            missing_count = int(
                (frame[column].isna() | frame[column].astype(str).str.strip().eq("")).sum()
            )
            if missing_count:
                gaps.append(f"missing_{column}_rows:{missing_count}")
    if dataset in {
        "financial_indicator",
        "financial_statement_balance_sheet",
        "financial_statement_income_statement",
        "financial_statement_cash_flow_statement",
        "announcement",
    }:
        gaps.append("source_publication_precision_date_only")
    if dataset == "news":
        gaps.append("article_link_unavailable_source_url_is_section_page")
    if dataset in {"macro_cpi", "macro_pmi"}:
        gaps.append("AKShare_series_has_no_official_release_timestamp")
    if dataset == "macro_release" and "released_at" in frame.columns:
        count = int((frame["released_at"] == "").sum())
        if count:
            gaps.append(f"official_release_time_missing:{count}")

    entry = archive_frame(
        cache_dir,
        dataset,
        provider,
        endpoint,
        parameters,
        frame,
        datetime.now(timezone.utc),
        gaps,
        entity_code,
        collection_date,
    )
    print(
        f"{dataset}: {entry['row_count']} 行，sha256={entry['sha256']}；"
        f"{'；'.join(gaps) if gaps else '必需字段齐全'}",
        flush=True,
    )
    if missing:
        raise ValueError(f"{endpoint} 响应缺少必需列：{sorted(missing)}")
    return True


def _capture_macro(
    cache_dir: Path,
    source: AkShareSource,
    collection_date: str | None,
    nbs_start_month: str,
    nbs_max_pages: int | None,
) -> None:
    for dataset, endpoint, fetch in (
        ("macro_cpi", "macro_china_cpi", source.macro_cpi),
        ("macro_pmi", "macro_china_pmi", source.macro_pmi),
    ):
        _capture(
            cache_dir,
            dataset,
            "AKShare/Eastmoney",
            endpoint,
            {},
            fetch,
            {"月份"},
            None,
            collection_date,
        )
    _capture(
        cache_dir,
        "macro_release",
        "National Bureau of Statistics of China",
        "NBS data release pages",
        {
            "start_month": nbs_start_month,
            "max_pages": str(nbs_max_pages) if nbs_max_pages is not None else "until_start_month",
        },
        lambda: source.nbs_macro_releases(nbs_start_month, nbs_max_pages),
        {"series_id", "observation_month", "released_at", "announcement_url"},
        None,
        collection_date,
    )


def _capture_symbol(
    cache_dir: Path,
    source: AkShareSource,
    code: str,
    start_date: date,
    end_date: date,
    collection_date: str | None,
) -> None:
    exchange, symbol = code.split(".")
    sina_code = f"{exchange}{symbol}"
    eastmoney_code = f"{symbol}.{exchange.upper()}"
    for statement in FINANCIAL_STATEMENTS:
        _capture(
            cache_dir,
            f"financial_statement_{STATEMENT_NAMES[statement]}",
            "AKShare/Sina",
            "stock_financial_report_sina",
            {"stock": sina_code, "symbol": statement},
            lambda statement=statement: source.financial_statement(code, statement),
            {"报告日", "公告日期"},
            code,
            collection_date,
        )
    _capture(
        cache_dir,
        "financial_indicator",
        "AKShare/Eastmoney",
        "stock_financial_analysis_indicator_em",
        {"symbol": eastmoney_code, "indicator": "按报告期"},
        lambda: source.financial_indicator(code),
        {"REPORT_DATE", "NOTICE_DATE"},
        code,
        collection_date,
    )
    _capture(
        cache_dir,
        "announcement",
        "AKShare/CNInfo",
        "stock_zh_a_disclosure_report_cninfo",
        {
            "symbol": symbol,
            "market": "沪深京",
            "start_date": start_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
        },
        lambda: source.disclosures(code, start_date, end_date),
        {"代码", "公告标题", "公告时间", "公告链接"},
        code,
        collection_date,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("history", "daily"), required=True)
    parser.add_argument(
        "--baostock-run-dir", type=Path, action="append",
        help="BaoStock 缓存目录；历史回填可重复传入以聚合多个快照区间",
    )
    parser.add_argument("--symbol", action="append", help="六位代码；可重复传入")
    parser.add_argument("--as-of-date", type=date.fromisoformat)
    parser.add_argument("--macro-start-month", default="2024-01")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    cache_dir = args.output_dir
    source = AkShareSource()
    if args.mode == "history":
        if not args.baostock_run_dir:
            raise ValueError("历史回填必须提供至少一个 --baostock-run-dir")
        members, start_date, end_date, snapshot_dates, source_records = _membership_union(args.baostock_run_dir)
        _record_universe_sources(cache_dir, source_records, snapshot_dates, members)
        requested = {_stock_code(symbol) for symbol in args.symbol or []}
        if requested and not requested.issubset(members):
            raise ValueError("指定代码不在所选 BaoStock 历史快照并集中")
        codes = sorted(requested or set(members))
        for index, code in enumerate(codes, start=1):
            _capture_symbol(cache_dir, source, code, start_date, end_date, None)
            if index % 10 == 0 or index == len(codes):
                print(f"历史成员资料进度：{index}/{len(codes)}", flush=True)
        _capture_macro(cache_dir, source, None, args.macro_start_month, None)
        print(f"历史回填股票池来自 BaoStock 快照并集：{len(codes)} 个代码。", flush=True)
        if source_records[-1]["snapshot_calendar_intervals"]:
            print(
                "超过 8 个日历日未出现新快照的区间（不等同于交易日缺失）："
                f"{source_records[-1]['snapshot_calendar_intervals']}",
                flush=True,
            )
        return

    symbols = sorted({_stock_code(symbol) for symbol in args.symbol or []})
    if not symbols:
        raise ValueError("日更采集至少需要一个 --symbol 候选股或持仓代码")
    capture_date = args.as_of_date or datetime.now(SHANGHAI).date()
    capture_date_text = capture_date.isoformat()
    for index, code in enumerate(symbols, start=1):
        _capture_symbol(
            cache_dir, source, code, capture_date, capture_date, capture_date_text
        )
        print(f"日更个股资料进度：{index}/{len(symbols)}", flush=True)
    recent_month = (
        date(capture_date.year, capture_date.month, 1) - timedelta(days=45)
    ).strftime("%Y-%m")
    _capture_macro(cache_dir, source, capture_date_text, recent_month, 1)
    _capture(
        cache_dir,
        "news",
        "AKShare/CLS",
        "stock_info_global_cls",
        {},
        source.recent_market_news,
        {"新闻标题", "新闻内容", "发布日期", "发布时间", "新闻来源", "新闻链接"},
        None,
        capture_date_text,
    )
    print(
        "新闻只从本次启动起归档当前财联社快讯；仅最近 20 条，按公司名筛选命中，不是完整个股新闻库。",
        flush=True,
    )


if __name__ == "__main__":
    main()
