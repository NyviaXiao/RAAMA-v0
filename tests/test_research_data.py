import csv
from datetime import date

import pytest

from adaptive_mas.data import load_research_data
from adaptive_mas.features.momentum import momentum
from adaptive_mas.targets.five_session import five_session_forward_returns


TRAIN_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]


def _write_rows(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _provider_row(code, day, adjustflag, close="10"):
    return {
        "date": day,
        "code": code,
        "open": "10",
        "high": "11.5" if close == "11" else "10.5",
        "low": "9.5",
        "close": close,
        "preclose": "9.5",
        "volume": "100",
        "amount": "1000",
        "adjustflag": adjustflag,
        "turn": "0.1",
        "tradestatus": "1",
        "pctChg": "5.2632",
    }


def test_research_snapshot_excludes_future_bars_and_uses_available_membership(tmp_path):
    days = (
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
    )
    data_dir = tmp_path / "input"
    train_rows = [
        {
            "股票代码": "000001",
            "日期": day,
            "开盘": "10",
            "收盘": "10",
            "最高": "10.5",
            "最低": "9.5",
            "成交量": "100",
            "成交额": "1000",
            "振幅": "10.5",
            "涨跌额": "0.5",
            "换手率": "0.1",
            "涨跌幅": "5.2632",
        }
        for day in days
    ]
    _write_rows(data_dir / "train.csv", TRAIN_COLUMNS, train_rows)
    (data_dir / "test.csv").write_text("not a research input", encoding="utf-8")

    raw_dir = tmp_path / "raw"
    _write_rows(
        raw_dir / "index" / "sh.000300.csv",
        ["date", "code", "open", "high", "low", "close", "preclose", "pctChg"],
        [
            {
                "date": day,
                "code": "sh.000300",
                "open": "10",
                "high": "11",
                "low": "9",
                "close": "10",
                "preclose": "9.5",
                "pctChg": "5.2",
            }
            for day in days
        ],
    )
    _write_rows(
        raw_dir / "membership" / "asof-2024-01-02.csv",
        ["updateDate", "code", "code_name"],
        [
            {"updateDate": "2024-01-01", "code": "sz.000001", "code_name": "平安银行"},
            {"updateDate": "2024-01-01", "code": "sz.000002", "code_name": "万科A"},
        ],
    )
    _write_rows(
        raw_dir / "membership" / "asof-2024-01-03.csv",
        ["updateDate", "code", "code_name"],
        [{"updateDate": "2024-01-03", "code": "sz.000002", "code_name": "万科A"}],
    )
    for code in ("sz.000001", "sz.000002"):
        _write_rows(
            raw_dir / "daily-unadjusted" / f"{code}.csv",
            list(_provider_row(code, "2024-01-02", "3")),
            [_provider_row(code, day, "3") for day in days],
        )
    _write_rows(
        raw_dir / "daily-adjusted-missing" / "sz.000002.csv",
        list(_provider_row("sz.000002", "2024-01-02", "1")),
        [
            _provider_row(
                "sz.000002",
                day,
                "1",
                close="11" if day == "2024-01-09" else "10",
            )
            for day in days
        ],
    )

    research_data = load_research_data(data_dir, raw_dir)
    snapshot = research_data.snapshot_at(date(2024, 1, 2))

    assert snapshot.members == {"000001", "000002"}
    assert {row["trade_date"] for row in snapshot.adjusted_bars} == {
        date(2024, 1, 2)
    }
    assert len(snapshot.adjusted_bars) == 2
    assert len(snapshot.execution_bars) == 2
    assert len(snapshot.index_bars) == 1
    assert research_data.snapshot_at(date(2024, 1, 3)).members == {
        "000001",
        "000002",
    }
    assert research_data.snapshot_at(date(2024, 1, 4)).members == {"000002"}
    feature_snapshot = research_data.snapshot_at(date(2024, 1, 3))
    assert max(feature_snapshot.trading_dates) == date(2024, 1, 3)
    signals = momentum(feature_snapshot, lookback_sessions=1)
    assert {row["symbol"]: row["momentum"] for row in signals} == {
        "000001": 0,
        "000002": 0,
    }
    assert all(row["input_cutoff"] == date(2024, 1, 3) for row in signals)

    label = five_session_forward_returns(research_data, [date(2024, 1, 2)])
    label_by_symbol = {row["symbol"]: row for row in label}
    assert label_by_symbol["000002"]["entry_date"] == date(2024, 1, 3)
    assert label_by_symbol["000002"]["exit_date"] == date(2024, 1, 9)
    assert label_by_symbol["000002"]["forward_return_5s"] == pytest.approx(0.1)
    assert label_by_symbol["000002"]["available_at"] == date(2024, 1, 9)
