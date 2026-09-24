import csv
from datetime import date

import pytest

from adaptive_mas.data import load_oos_research_data
from adaptive_mas.evaluation.agent_research import net_return_after_costs


def _write_rows(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _provider_row(day, adjustflag):
    return {
        "date": day,
        "code": "sh.600000",
        "open": "10",
        "high": "11",
        "low": "9",
        "close": "10.5",
        "preclose": "10",
        "volume": "100",
        "amount": "1000",
        "adjustflag": adjustflag,
        "turn": "0.1",
        "tradestatus": "1",
        "pctChg": "5",
    }


def test_oos_loader_uses_only_the_separate_baostock_cache(tmp_path):
    run_dir = tmp_path / "oos-cache"
    days = [f"2026-03-{day:02d}" for day in range(16, 21)]
    _write_rows(
        run_dir / "index" / "sh.000300.csv",
        ["date", "code", "open", "high", "low", "close", "preclose", "pctChg"],
        [
            {
                "date": day,
                "code": "sh.000300",
                "open": "100",
                "high": "102",
                "low": "99",
                "close": "101",
                "preclose": "100",
                "pctChg": "1",
            }
            for day in days
        ],
    )
    _write_rows(
        run_dir / "membership" / "asof-2026-03-16.csv",
        ["updateDate", "code", "code_name"],
        [{"updateDate": "2026-03-13", "code": "sh.600000", "code_name": "浦发银行"}],
    )
    for folder, adjustflag in (("daily-adjusted", "1"), ("daily-unadjusted", "3")):
        _write_rows(
            run_dir / folder / "sh.600000.csv",
            list(_provider_row(days[0], adjustflag)),
            [_provider_row(day, adjustflag) for day in days],
        )

    research = load_oos_research_data(run_dir)

    assert research.trading_dates == tuple(date.fromisoformat(day) for day in days)
    assert research.membership_at(date.fromisoformat(days[0])) == {"600000"}
    assert len(research.adjusted_bars) == len(days)
    assert len(research.execution_bars) == len(days)


def test_oos_loader_rejects_missing_symbol_bars(tmp_path):
    run_dir = tmp_path / "oos-cache"
    _write_rows(
        run_dir / "index" / "sh.000300.csv",
        ["date", "code", "open", "high", "low", "close", "preclose", "pctChg"],
        [{
            "date": "2026-03-16",
            "code": "sh.000300",
            "open": "100",
            "high": "102",
            "low": "99",
            "close": "101",
            "preclose": "100",
            "pctChg": "1",
        }],
    )
    _write_rows(
        run_dir / "membership" / "asof-2026-03-16.csv",
        ["updateDate", "code", "code_name"],
        [{"updateDate": "2026-03-13", "code": "sh.600000", "code_name": "浦发银行"}],
    )
    (run_dir / "daily-adjusted").mkdir()
    (run_dir / "daily-unadjusted").mkdir()

    with pytest.raises(ValueError, match="样本外后复权行情文件不完整"):
        load_oos_research_data(run_dir)


def test_net_return_costs_apply_both_sides_and_sell_stamp_duty():
    costs = {
        "commission_per_side": 3.0,
        "transfer_fee_per_side": 0.1,
        "stamp_duty_sell": 5.0,
        "slippage_per_side": 5.0,
    }
    expected = (1.01 * 0.9995 * (1 - 0.00081)) / (1.0005 * (1 + 0.00031)) - 1

    assert net_return_after_costs(0.01, costs) == pytest.approx(expected)
    assert net_return_after_costs(0.01, {name: 0.0 for name in costs}) == pytest.approx(0.01)
