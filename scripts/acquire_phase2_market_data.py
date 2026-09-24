"""Cache the BaoStock records required for a point-in-time CSI 300 prototype."""

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from adaptive_mas.data.baostock_source import BaoStockSource
from adaptive_mas.data.market_data import load_training_bars


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_response(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    manifest_path: Path,
    endpoint: str,
    parameters: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    entry = {
        "source": "BaoStock",
        "endpoint": endpoint,
        "parameters": parameters,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "response_file": path.relative_to(manifest_path.parent).as_posix(),
        "row_count": len(rows),
        "sha256": sha256_file(path),
    }
    with manifest_path.open("a", encoding="utf-8", newline="") as manifest:
        manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "baostock",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="续传一个已有采集目录，沿用其中已缓存的原始响应。",
    )
    args = parser.parse_args()

    training_bars = load_training_bars(args.training_data_dir)
    training_symbols = {str(row["symbol"]) for row in training_bars}
    sessions = sorted({row["trade_date"] for row in training_bars})
    start_date = sessions[0]
    end_date = sessions[-1]
    snapshots_by_week: dict[tuple[int, int], date] = {}
    for trade_date in sessions:
        iso_date = trade_date.isocalendar()
        snapshots_by_week.setdefault((iso_date.year, iso_date.week), trade_date)
    snapshot_dates = sorted(snapshots_by_week.values())

    if args.resume_run_dir:
        run_dir = args.resume_run_dir
        if not run_dir.is_dir():
            raise FileNotFoundError(f"续传目录不存在：{run_dir}")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.output_root / f"fetch-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.jsonl"
    if args.resume_run_dir:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"续传目录缺少清单：{manifest_path}")
    else:
        manifest_path.touch(exist_ok=False)

    member_codes: set[str] = set()
    completed_responses = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line:
            completed_responses.add(json.loads(line)["response_file"])

    pending_snapshot_dates = []
    for query_date in snapshot_dates:
        relative_path = f"membership/asof-{query_date.isoformat()}.csv"
        response_path = run_dir / relative_path
        if relative_path in completed_responses:
            with response_path.open(encoding="utf-8", newline="") as cached:
                member_codes.update(row["code"] for row in csv.DictReader(cached))
            continue
        if response_path.exists():
            raise FileExistsError(f"响应文件已存在但未写入清单：{response_path}")
        pending_snapshot_dates.append(query_date)

    request_batch_size = 30
    for batch_start in range(0, len(pending_snapshot_dates), request_batch_size):
        batch_dates = pending_snapshot_dates[batch_start : batch_start + request_batch_size]
        with BaoStockSource() as source:
            for query_date in batch_dates:
                fields, rows = source.hs300_membership(query_date)
                if not rows:
                    raise RuntimeError(f"BaoStock 未返回历史成分：{query_date}")
                response_path = run_dir / "membership" / f"asof-{query_date.isoformat()}.csv"
                save_response(
                    response_path,
                    fields,
                    rows,
                    manifest_path,
                    "query_hs300_stocks",
                    {"date": query_date.isoformat()},
                )
                member_codes.update(row["code"] for row in rows)

    print(f"已缓存 {len(snapshot_dates)} 个周度成分快照，涉及 {len(member_codes)} 个代码。")
    pending_codes = []
    for code in sorted(member_codes):
        relative_path = f"daily-unadjusted/{code}.csv"
        response_path = run_dir / relative_path
        if relative_path in completed_responses:
            continue
        if response_path.exists():
            raise FileExistsError(f"响应文件已存在但未写入清单：{response_path}")
        pending_codes.append(code)

    completed_daily_count = sum(
        path.startswith("daily-unadjusted/") for path in completed_responses
    )
    daily_count = completed_daily_count
    for batch_start in range(0, len(pending_codes), request_batch_size):
        batch_codes = pending_codes[batch_start : batch_start + request_batch_size]
        with BaoStockSource() as source:
            for code in batch_codes:
                fields, rows = source.daily_bars(
                    code, start_date, end_date, adjustflag="3"
                )
                response_path = run_dir / "daily-unadjusted" / f"{code}.csv"
                save_response(
                    response_path,
                    fields,
                    rows,
                    manifest_path,
                    "query_history_k_data_plus",
                    {
                        "code": code,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "frequency": "d",
                        "adjustflag": "3",
                    },
                )
                daily_count += 1
                completed_responses.add(f"daily-unadjusted/{code}.csv")
                if daily_count % 25 == 0 or daily_count == len(member_codes):
                    print(
                        "已缓存未复权成分股日线："
                        f"{daily_count}/{len(member_codes)}。"
                    )

    missing_adjusted_codes = sorted(
        code for code in member_codes if code.split(".", 1)[1] not in training_symbols
    )
    completed_adjusted = sum(
        path.startswith("daily-adjusted-missing/") for path in completed_responses
    )
    adjusted_count = completed_adjusted
    pending_adjusted_codes = []
    for code in missing_adjusted_codes:
        relative_path = f"daily-adjusted-missing/{code}.csv"
        response_path = run_dir / relative_path
        if relative_path in completed_responses:
            continue
        if response_path.exists():
            raise FileExistsError(f"响应文件已存在但未写入清单：{response_path}")
        pending_adjusted_codes.append(code)

    for batch_start in range(0, len(pending_adjusted_codes), request_batch_size):
        batch_codes = pending_adjusted_codes[batch_start : batch_start + request_batch_size]
        with BaoStockSource() as source:
            for code in batch_codes:
                fields, rows = source.daily_bars(
                    code, start_date, end_date, adjustflag="1"
                )
                response_path = run_dir / "daily-adjusted-missing" / f"{code}.csv"
                save_response(
                    response_path,
                    fields,
                    rows,
                    manifest_path,
                    "query_history_k_data_plus",
                    {
                        "code": code,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "frequency": "d",
                        "adjustflag": "1",
                    },
                )
                adjusted_count += 1
                completed_responses.add(f"daily-adjusted-missing/{code}.csv")
                if adjusted_count % 25 == 0 or adjusted_count == len(missing_adjusted_codes):
                    print(
                        "已缓存训练集未覆盖股票的后复权日线："
                        f"{adjusted_count}/{len(missing_adjusted_codes)}。"
                    )

    index_relative_path = "index/sh.000300.csv"
    index_path = run_dir / index_relative_path
    if index_relative_path not in completed_responses:
        if index_path.exists():
            raise FileExistsError(f"响应文件已存在但未写入清单：{index_path}")
        with BaoStockSource() as source:
            fields, rows = source.index_daily("sh.000300", start_date, end_date)
        save_response(
            index_path,
            fields,
            rows,
            manifest_path,
            "query_history_k_data_plus",
            {
                "code": "sh.000300",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "frequency": "d",
            },
        )

    print(f"BaoStock 原始数据已写入：{run_dir}")


if __name__ == "__main__":
    main()
