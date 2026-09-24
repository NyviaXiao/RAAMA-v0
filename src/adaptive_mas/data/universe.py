"""Point-in-time reads over cached BaoStock membership snapshots."""

import csv
from datetime import date
from pathlib import Path


def hs300_members_at(snapshot_dir: str | Path, decision_date: date) -> set[str]:
    latest_update_date: date | None = None
    latest_members: set[str] | None = None
    for path in Path(snapshot_dir).glob("asof-*.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or not {"updateDate", "code"}.issubset(reader.fieldnames):
                raise ValueError(f"沪深300快照缺少必需列：{path}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"沪深300快照没有成分记录：{path}")
        update_dates = {date.fromisoformat(row["updateDate"]) for row in rows}
        if len(update_dates) != 1:
            raise ValueError(f"同一沪深300快照含多个 updateDate：{path}")
        update_date = next(iter(update_dates))
        members = set()
        for row in rows:
            exchange, separator, symbol = row["code"].partition(".")
            if (
                not separator
                or exchange not in {"sh", "sz"}
                or len(symbol) != 6
                or not symbol.isascii()
                or not symbol.isdigit()
            ):
                raise ValueError(f"沪深300快照证券代码格式无效：{row['code']!r}（{path}）")
            members.add(symbol)
        if len(members) != len(rows):
            raise ValueError(f"沪深300快照存在重复股票代码：{path}")
        if update_date >= decision_date:
            continue
        if latest_update_date is None or update_date > latest_update_date:
            latest_update_date = update_date
            latest_members = members

    if latest_members is None:
        raise ValueError(f"没有在决策日前可用的沪深300成分快照：{decision_date}")
    return latest_members
