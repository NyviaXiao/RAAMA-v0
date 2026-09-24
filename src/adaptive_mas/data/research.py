"""Load and replay the project's date-granularity historical market view."""

import csv
import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from adaptive_mas.data.market_data import load_training_bars


EQUITY_FIELDS = {
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "turn",
    "tradestatus",
    "pctChg",
}
INDEX_FIELDS = {"date", "code", "open", "high", "low", "close", "preclose", "pctChg"}


@dataclass(frozen=True)
class MembershipSnapshot:
    event_time: date
    available_at: date
    members: frozenset[str]
    source_ref: str


@dataclass(frozen=True)
class MarketSnapshot:
    decision_date: date
    trading_dates: tuple[date, ...]
    members: frozenset[str]
    adjusted_bars: tuple[dict[str, object], ...]
    execution_bars: tuple[dict[str, object], ...]
    index_bars: tuple[dict[str, object], ...]


class ResearchData:
    def __init__(
        self,
        trading_dates: list[date],
        adjusted_bars: list[dict[str, object]],
        execution_bars: list[dict[str, object]],
        index_bars: list[dict[str, object]],
        membership_snapshots: list[MembershipSnapshot],
    ) -> None:
        self.trading_dates = tuple(trading_dates)
        self.adjusted_bars = tuple(adjusted_bars)
        self.execution_bars = tuple(execution_bars)
        self.index_bars = tuple(index_bars)
        self.membership_snapshots = tuple(membership_snapshots)

    def snapshot_at(self, decision_date: date) -> MarketSnapshot:
        members = self.membership_at(decision_date)

        return MarketSnapshot(
            decision_date=decision_date,
            trading_dates=tuple(day for day in self.trading_dates if day <= decision_date),
            members=members,
            adjusted_bars=tuple(
                row for row in self.adjusted_bars if row["available_at"] <= decision_date
            ),
            execution_bars=tuple(
                row for row in self.execution_bars if row["available_at"] <= decision_date
            ),
            index_bars=tuple(
                row for row in self.index_bars if row["available_at"] <= decision_date
            ),
        )

    def membership_at(self, decision_date: date) -> frozenset[str]:
        if decision_date not in self.trading_dates:
            raise ValueError(f"决策日不在沪深300交易日历中：{decision_date}")

        available_memberships = [
            snapshot
            for snapshot in self.membership_snapshots
            if snapshot.available_at <= decision_date
        ]
        if not available_memberships:
            raise ValueError(f"决策日 {decision_date} 没有已可用的沪深300成分快照")
        membership = max(
            available_memberships,
            key=lambda snapshot: (snapshot.available_at, snapshot.event_time),
        )
        return membership.members


def _read_rows(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"外部数据表头缺失或存在重复列：{path}")
        missing = required_fields - set(reader.fieldnames)
        if missing:
            raise ValueError(f"外部数据缺少必需列 {sorted(missing)}：{path}")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"外部数据第 {line_number} 行列数不匹配：{path}")
            rows.append(row)
        return rows


def _number(row: dict[str, str], column: str, path: Path) -> float | None:
    value = row[column]
    if value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"外部数据包含非有限数值 {column}：{path}")
    return number


def _symbol(code: str, path: Path) -> str:
    exchange, separator, symbol = code.partition(".")
    if (
        not separator
        or exchange not in {"sh", "sz"}
        or len(symbol) != 6
        or not symbol.isascii()
        or not symbol.isdigit()
    ):
        raise ValueError(f"BaoStock 证券代码格式无效：{code!r}（{path}）")
    return symbol


def _provider_bar(
    row: dict[str, str], path: Path, available_at: date, expected_adjustflag: str | None
) -> dict[str, object]:
    trade_date = date.fromisoformat(row["date"])
    symbol = _symbol(row["code"], path)
    if expected_adjustflag is not None and row["adjustflag"] != expected_adjustflag:
        raise ValueError(f"行情复权标记与目录用途不符：{path}")

    values = {
        name: _number(row, name, path)
        for name in ("open", "high", "low", "close", "preclose", "volume", "amount")
    }
    status = row["tradestatus"] if "tradestatus" in row else "1"
    if status not in {"0", "1"}:
        raise ValueError(f"BaoStock 成交状态无效：{status!r}（{path}）")
    open_price, high, low, close = (
        values["open"],
        values["high"],
        values["low"],
        values["close"],
    )
    if status == "1" and any(value is None for value in (open_price, high, low, close)):
        raise ValueError(f"正常交易记录缺少 OHLC：{path} {symbol} {trade_date}")
    if all(value is not None for value in (open_price, high, low, close)) and (
        low > min(open_price, close)
        or high < max(open_price, close)
        or low > high
    ):
        raise ValueError(f"BaoStock 行情 OHLC 关系无效：{path} {symbol} {trade_date}")

    preclose = values["preclose"]
    amplitude = None
    if preclose is not None and preclose > 0 and high is not None and low is not None:
        amplitude = (high - low) / preclose * 100
    price_change = close - preclose if close is not None and preclose is not None else None
    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "available_at": available_at,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": values["volume"],
        "amount": values["amount"],
        "amplitude": amplitude,
        "price_change": price_change,
        "turnover": _number(row, "turn", path) if "turn" in row else None,
        "return_1d": _number(row, "pctChg", path),
        "tradestatus": status,
        "source": "BaoStock",
        "source_ref": path.as_posix(),
    }


def _provider_bars(
    directory: Path,
    files: set[str],
    required_fields: set[str],
    adjustflag: str | None,
    trading_dates: tuple[date, ...],
) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[str, date], dict[str, object]] = {}
    valid_dates = set(trading_dates)
    for filename in sorted(files):
        path = directory / filename
        for row in _read_rows(path, required_fields):
            trade_date = date.fromisoformat(row["date"])
            if trade_date not in valid_dates:
                raise ValueError(f"行情日期不在已取得的沪深300交易日历中：{path} {trade_date}")
            bar = _provider_bar(row, path, trade_date, adjustflag)
            key = (str(bar["symbol"]), trade_date)
            if key in rows_by_key:
                raise ValueError(f"外部行情存在重复 (symbol, trade_date)：{key}")
            rows_by_key[key] = bar
    return list(rows_by_key.values())


def load_research_data(data_dir: str | Path, baostock_run_dir: str | Path) -> ResearchData:
    """Load local adjusted training bars and the explicitly cached BaoStock records."""
    data_path = Path(data_dir)
    run_dir = Path(baostock_run_dir)
    index_path = run_dir / "index" / "sh.000300.csv"
    index_rows = _read_rows(index_path, INDEX_FIELDS)
    index_bars = []
    seen_index_dates = set()
    for row in index_rows:
        trade_date = date.fromisoformat(row["date"])
        if trade_date in seen_index_dates:
            raise ValueError(f"沪深300指数存在重复交易日：{trade_date}")
        seen_index_dates.add(trade_date)
        index_bars.append(
            {
                "symbol": row["code"],
                "trade_date": trade_date,
                "available_at": trade_date,
                "open": _number(row, "open", index_path),
                "high": _number(row, "high", index_path),
                "low": _number(row, "low", index_path),
                "close": _number(row, "close", index_path),
                "return_1d": _number(row, "pctChg", index_path),
                "source": "BaoStock",
                "source_ref": index_path.as_posix(),
            }
        )
    trading_dates = tuple(sorted(seen_index_dates))
    if not trading_dates:
        raise ValueError(f"沪深300指数没有交易日：{index_path}")

    training_bars = load_training_bars(data_path)
    training_symbols = {str(row["symbol"]) for row in training_bars}
    adjusted_bars = []
    for row in training_bars:
        trade_date = row["trade_date"]
        if trade_date not in seen_index_dates:
            raise ValueError(f"训练行情日期不在指数交易日历中：{trade_date}")
        adjusted_bars.append(
            {
                **row,
                "available_at": trade_date,
                "source": "train.csv",
                "source_ref": "train.csv",
            }
        )

    membership_snapshots = []
    membership_codes: set[str] = set()
    for path in sorted((run_dir / "membership").glob("asof-*.csv")):
        query_date = date.fromisoformat(path.stem.removeprefix("asof-"))
        rows = _read_rows(path, {"updateDate", "code", "code_name"})
        if not rows:
            raise ValueError(f"沪深300成分快照为空：{path}")
        update_dates = {date.fromisoformat(row["updateDate"]) for row in rows}
        if len(update_dates) != 1:
            raise ValueError(f"同一成分快照含多个 updateDate：{path}")
        update_date = next(iter(update_dates))
        if update_date > query_date:
            raise ValueError(f"成分快照 updateDate 晚于请求日期：{path}")
        snapshot_members = frozenset(_symbol(row["code"], path) for row in rows)
        if len(snapshot_members) != len(rows):
            raise ValueError(f"成分快照存在重复股票代码：{path}")
        membership_codes.update(row["code"] for row in rows)
        available_index = bisect_right(trading_dates, update_date)
        if available_index == len(trading_dates):
            raise ValueError(f"成分快照之后没有可用的研究交易日：{path}")
        membership_snapshots.append(
            MembershipSnapshot(
                event_time=update_date,
                available_at=trading_dates[available_index],
                members=snapshot_members,
                source_ref=path.as_posix(),
            )
        )

    if not membership_codes:
        raise ValueError(f"没有可用沪深300历史成分快照：{run_dir}")
    expected_adjusted_symbols = {
        code for code in membership_codes if _symbol(code, run_dir) not in training_symbols
    }
    adjusted_dir = run_dir / "daily-adjusted-missing"
    adjusted_files = {path.name for path in adjusted_dir.glob("*.csv")}
    expected_adjusted_files = {f"{code}.csv" for code in expected_adjusted_symbols}
    if adjusted_files != expected_adjusted_files:
        raise ValueError(
            "训练集未覆盖的历史成分复权行情文件不完整或有多余文件："
            f"缺少 {sorted(expected_adjusted_files - adjusted_files)}，"
            f"多余 {sorted(adjusted_files - expected_adjusted_files)}"
        )
    adjusted_bars.extend(
        _provider_bars(
            adjusted_dir,
            adjusted_files,
            EQUITY_FIELDS,
            "1",
            trading_dates,
        )
    )
    adjusted_symbols = {str(row["symbol"]) for row in adjusted_bars}
    if adjusted_symbols != {_symbol(code, run_dir) for code in membership_codes}:
        raise ValueError("后复权行情与历史成分股票池覆盖不一致")

    execution_dir = run_dir / "daily-unadjusted"
    execution_files = {path.name for path in execution_dir.glob("*.csv")}
    expected_execution_files = {f"{code}.csv" for code in membership_codes}
    if execution_files != expected_execution_files:
        raise ValueError(
            "历史成分未复权行情文件不完整或有多余文件："
            f"缺少 {sorted(expected_execution_files - execution_files)}，"
            f"多余 {sorted(execution_files - expected_execution_files)}"
        )
    execution_bars = _provider_bars(
        execution_dir,
        execution_files,
        EQUITY_FIELDS,
        "3",
        trading_dates,
    )

    return ResearchData(
        list(trading_dates),
        adjusted_bars,
        execution_bars,
        index_bars,
        membership_snapshots,
    )
