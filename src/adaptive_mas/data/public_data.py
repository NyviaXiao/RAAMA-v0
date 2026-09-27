"""Immutable AKShare response cache and minimal point-in-time reads."""

import csv
import hashlib
import json
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


SHANGHAI = ZoneInfo("Asia/Shanghai")
MANIFEST_NAME = "manifest.jsonl"
PUBLIC_DATASETS = {
    "financial_statement_balance_sheet",
    "financial_statement_income_statement",
    "financial_statement_cash_flow_statement",
    "financial_indicator",
    "announcement",
    "news",
    "macro_cpi",
    "macro_pmi",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_key(
    dataset: str,
    endpoint: str,
    parameters: dict[str, str],
    entity_code: str | None,
    collection_date: str | None,
) -> str:
    material = _canonical_json(
        {
            "dataset": dataset,
            "endpoint": endpoint,
            "parameters": parameters,
            "entity_code": entity_code,
            "collection_date": collection_date,
        }
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def read_manifest(cache_dir: str | Path) -> list[dict[str, object]]:
    path = Path(cache_dir) / MANIFEST_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def completed_request_keys(cache_dir: str | Path) -> set[str]:
    return {
        str(entry["request_key"])
        for entry in read_manifest(cache_dir)
        if entry.get("response_file")
    }


def archive_frame(
    cache_dir: str | Path,
    dataset: str,
    source: str,
    endpoint: str,
    parameters: dict[str, str],
    frame: pd.DataFrame,
    fetched_at: datetime,
    gaps: list[str],
    entity_code: str | None = None,
    collection_date: str | None = None,
) -> dict[str, object]:
    if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
        raise ValueError("fetched_at 必须带时区")
    root = Path(cache_dir)
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    response_path = root / "objects" / f"{digest}.csv"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    if response_path.exists():
        if hashlib.sha256(response_path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"内容寻址缓存校验失败：{response_path}")
    else:
        with response_path.open("xb") as output:
            output.write(payload)

    entry: dict[str, object] = {
        "schema_version": 1,
        "dataset": dataset,
        "source": source,
        "endpoint": endpoint,
        "parameters": parameters,
        "entity_code": entity_code,
        "collection_date": collection_date,
        "request_key": request_key(
            dataset, endpoint, parameters, entity_code, collection_date
        ),
        "fetched_at": fetched_at.astimezone(timezone.utc).isoformat(),
        "response_file": response_path.relative_to(root).as_posix(),
        "row_count": len(frame),
        "columns": [str(column) for column in frame.columns],
        "sha256": digest,
        "gaps": gaps,
    }
    with (root / MANIFEST_NAME).open("a", encoding="utf-8", newline="") as manifest:
        manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def archive_failure(
    cache_dir: str | Path,
    dataset: str,
    source: str,
    endpoint: str,
    parameters: dict[str, str],
    fetched_at: datetime,
    error: Exception,
    entity_code: str | None = None,
    collection_date: str | None = None,
) -> None:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema_version": 1,
        "dataset": dataset,
        "source": source,
        "endpoint": endpoint,
        "parameters": parameters,
        "entity_code": entity_code,
        "collection_date": collection_date,
        "request_key": request_key(
            dataset, endpoint, parameters, entity_code, collection_date
        ),
        "fetched_at": fetched_at.astimezone(timezone.utc).isoformat(),
        "failure": f"{type(error).__name__}: {error}",
        "gaps": ["request_failed"],
    }
    with (root / MANIFEST_NAME).open("a", encoding="utf-8", newline="") as manifest:
        manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")


def request_is_cached(
    cache_dir: str | Path,
    dataset: str,
    endpoint: str,
    parameters: dict[str, str],
    entity_code: str | None,
    collection_date: str | None,
) -> bool:
    expected = request_key(
        dataset, endpoint, parameters, entity_code, collection_date
    )
    return expected in completed_request_keys(cache_dir)


def _read_cached_rows(
    cache_dir: Path, dataset: str
) -> list[tuple[dict[str, object], dict[str, str]]]:
    rows: list[tuple[dict[str, object], dict[str, str]]] = []
    for entry in read_manifest(cache_dir):
        if entry.get("dataset") != dataset or not entry.get("response_file"):
            continue
        response_path = cache_dir / str(entry["response_file"])
        payload = response_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"原始响应缓存哈希不符：{response_path}")
        with response_path.open("r", encoding="utf-8-sig", newline="") as source:
            rows.extend((entry, row) for row in csv.DictReader(source))
    return rows


def _as_local_datetime(value: str, date_only_next_day: bool = False) -> datetime:
    if len(value) == 8 and value.isdigit():
        parsed = datetime.strptime(value, "%Y%m%d")
        is_date_only = True
    else:
        parsed = datetime.fromisoformat(value)
        is_date_only = len(value) == 10
    if parsed.tzinfo is not None:
        return parsed.astimezone(SHANGHAI)
    if is_date_only and date_only_next_day:
        parsed = datetime.combine(parsed.date() + timedelta(days=1), time.min)
    return parsed.replace(tzinfo=SHANGHAI)


def _manifest_fetched_at(entry: dict[str, object]) -> datetime:
    return datetime.fromisoformat(str(entry["fetched_at"])).astimezone(SHANGHAI)


def _record_fetched_at(record: dict[str, object]) -> datetime:
    return datetime.fromisoformat(str(record["fetched_at"])).astimezone(SHANGHAI)


def _macro_release_map(cache_dir: Path, decision_time: datetime) -> dict[tuple[str, str], datetime]:
    release_map: dict[tuple[str, str], datetime] = {}
    for entry, row in _read_cached_rows(cache_dir, "macro_release"):
        fetched_at = _manifest_fetched_at(entry)
        if fetched_at > decision_time or not row["released_at"]:
            continue
        released_at = datetime.fromisoformat(row["released_at"]).astimezone(SHANGHAI)
        if released_at <= decision_time:
            key = (row["series_id"], row["observation_month"])
            if key not in release_map or released_at > release_map[key]:
                release_map[key] = released_at
    return release_map


def load_public_records(
    cache_dir: str | Path,
    dataset: str,
    decision_time: datetime,
    symbols: set[str] | None = None,
    company_names: set[str] | None = None,
) -> list[dict[str, object]]:
    """Read cached records whose source publication time was visible by decision_time.

    A date-only filing timestamp is conservatively visible from the next calendar day.
    Incrementally archived news and macro values require a cache fetch no later than
    decision_time. For historical financial and announcement backfills, source dates
    provide approximate visibility; when no pre-decision snapshot exists, the earliest
    captured version is returned and marked unverified.
    """
    if decision_time.tzinfo is None or decision_time.utcoffset() is None:
        raise ValueError("decision_time 必须带时区")
    if dataset not in PUBLIC_DATASETS:
        raise ValueError(f"未知的 AKShare 数据集：{dataset!r}")
    if dataset == "news" and symbols is not None:
        raise ValueError("财联社快讯没有逐股代码，新闻筛选请使用 company_names")
    if symbols is not None:
        for code in symbols:
            exchange, separator, symbol = code.partition(".")
            if (
                not separator
                or exchange not in {"sh", "sz"}
                or len(symbol) != 6
                or not symbol.isascii()
                or not symbol.isdigit()
            ):
                raise ValueError(f"BaoStock 证券代码格式无效：{code!r}")
    if company_names is not None and any(
        not isinstance(name, str) or not name.strip() for name in company_names
    ):
        raise ValueError("公司名称筛选值不能为空")
    decision_time = decision_time.astimezone(SHANGHAI)
    root = Path(cache_dir)
    records: list[dict[str, object]] = []
    release_map = _macro_release_map(root, decision_time) if dataset in {"macro_cpi", "macro_pmi"} else {}

    for entry, raw in _read_cached_rows(root, dataset):
        symbol = str(entry.get("entity_code") or "")
        if dataset != "news" and symbols is not None and symbol not in symbols:
            continue

        if dataset.startswith("financial_statement_"):
            fiscal_period = raw["报告日"]
            published_value = raw["公告日期"]
            if not published_value:
                continue
            published_at = _as_local_datetime(published_value)
            available_at = _as_local_datetime(published_value, date_only_next_day=True)
            source_record_id = f"{symbol}:{dataset.removeprefix('financial_statement_')}:{fiscal_period}"
        elif dataset == "financial_indicator":
            period_key = raw.get("REPORT_DATE", "")
            published_value = raw.get("NOTICE_DATE", "")
            if not period_key or not published_value:
                continue
            published_at = _as_local_datetime(published_value[:10])
            available_at = _as_local_datetime(published_value[:10], date_only_next_day=True)
            source_record_id = f"{symbol}:indicator:{period_key}"
        elif dataset == "announcement":
            published_value = raw["公告时间"]
            if not published_value:
                continue
            published_at = _as_local_datetime(published_value[:10])
            available_at = _as_local_datetime(published_value[:10], date_only_next_day=True)
            source_record_id = raw["公告链接"]
        elif dataset == "news":
            published_at = _as_local_datetime(
                f"{raw['发布日期']}T{raw['发布时间']}+08:00"
            )
            available_at = _manifest_fetched_at(entry)
            if company_names is not None:
                searchable = f"{raw['新闻标题']} {raw['新闻内容']}"
                if not any(name in searchable for name in company_names):
                    continue
            source_record_id = hashlib.sha256(
                _canonical_json(
                    {
                        "title": raw["新闻标题"],
                        "date": raw["发布日期"],
                        "time": raw["发布时间"],
                        "content": raw["新闻内容"],
                    }
                ).encode("utf-8")
            ).hexdigest()
        elif dataset in {"macro_cpi", "macro_pmi"}:
            series_id = dataset.removeprefix("macro_")
            observation_month = raw["月份"]
            month_match = re.search(r"(\d{4})年\s*(\d{1,2})月", observation_month)
            if month_match is None:
                raise ValueError(f"AKShare 宏观观察期格式无效：{observation_month!r}")
            month = f"{int(month_match.group(1)):04d}-{int(month_match.group(2)):02d}"
            published_at = release_map.get((series_id, month))
            if published_at is None:
                continue
            available_at = max(published_at, _manifest_fetched_at(entry))
            source_record_id = f"{series_id}:{month}"
        else:
            raise ValueError(f"未知的 AKShare 数据集：{dataset!r}")

        if available_at > decision_time or published_at > decision_time:
            continue
        if dataset in {"news", "macro_cpi", "macro_pmi"} and _manifest_fetched_at(entry) > decision_time:
            continue
        records.append(
            {
                **raw,
                "symbol": symbol or None,
                "source_record_id": source_record_id,
                "published_at": published_at.isoformat(),
                "available_at": available_at.isoformat(),
                "source": entry["source"],
                "endpoint": entry["endpoint"],
                "fetched_at": entry["fetched_at"],
                "raw_response_ref": entry["response_file"],
                "content_sha256": entry["sha256"],
                "vintage_verified": dataset not in {
                    "financial_indicator",
                    "financial_statement_balance_sheet",
                    "financial_statement_income_statement",
                    "financial_statement_cash_flow_statement",
                    "macro_cpi",
                    "macro_pmi",
                },
                "vintage_status": (
                    "current_vendor_history_revision_unverified"
                    if dataset.startswith("financial_statement_")
                    or dataset in {"financial_indicator", "macro_cpi", "macro_pmi"}
                    else "captured_response_version"
                ),
            }
        )

    unique_records: dict[tuple[str, ...], dict[str, object]] = {}
    for record in records:
        identity = (
            str(record.get("symbol") or ""),
            str(record["source_record_id"]),
        )
        existing = unique_records.get(identity)
        if existing is None:
            unique_records[identity] = record
            continue

        fetched_at = _record_fetched_at(record)
        existing_fetched_at = _record_fetched_at(existing)
        if dataset == "news":
            # Repeated CLS windows keep the first capture at which the event was visible.
            replace_existing = fetched_at < existing_fetched_at
        elif dataset in {"macro_cpi", "macro_pmi"}:
            # Macro snapshots must already be captured by decision_time; prefer newest.
            replace_existing = fetched_at > existing_fetched_at
        else:
            # Prefer the newest snapshot available at the decision. If all snapshots
            # were captured later, retain the earliest historical backfill version.
            fetched_before_decision = fetched_at <= decision_time
            existing_before_decision = existing_fetched_at <= decision_time
            if fetched_before_decision != existing_before_decision:
                replace_existing = fetched_before_decision
            elif fetched_before_decision:
                replace_existing = fetched_at > existing_fetched_at
            else:
                replace_existing = fetched_at < existing_fetched_at
        if replace_existing:
            unique_records[identity] = record

    historical_backfill_datasets = {
        "financial_statement_balance_sheet",
        "financial_statement_income_statement",
        "financial_statement_cash_flow_statement",
        "financial_indicator",
        "announcement",
    }
    for record in unique_records.values():
        if (
            dataset in historical_backfill_datasets
            and _record_fetched_at(record) > decision_time
        ):
            record["vintage_verified"] = False
            if record["vintage_status"] == "captured_response_version":
                record["vintage_status"] = "historical_backfill_after_decision_unverified"

    result = list(unique_records.values())
    result.sort(key=lambda row: (str(row["published_at"]), str(row["source_record_id"])))
    return result
