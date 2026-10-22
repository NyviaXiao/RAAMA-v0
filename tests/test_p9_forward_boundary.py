"""The forward gate needs complete mature baskets and intact BaoStock responses."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_p9_forward_validation import (
    _new_decision_label_coverage,
    _partition_forward_dates,
    _post_freeze_account_decisions,
    _research_for_as_of,
    _verify_baostock_responses,
)
from adaptive_mas.data.research import MembershipSnapshot, ResearchData


def test_pre_freeze_2026_09_24_is_not_a_post_freeze_mature_decision() -> None:
    forecast = {
        "prediction_date": "2026-09-24",
        "planned_exit_date": "2026-09-29",
        "horizon_sessions": 3,
        "target_weights_by_method": {
            "trend": {"000001": 0.05},
            "market_plus_relative_fusion": {"000001": 0.05, "000002": 0.05},
        },
    }
    settlements = [
        {
            "prediction_date": "2026-09-24",
            "symbol": symbol,
            "planned_exit_date": "2026-09-29",
            "available_at": "2026-09-29",
            "status": "settled",
            "forward_return_3s": 0.01,
        }
        for symbol in ("000001", "000002")
    ]
    partial, partial_dates = _new_decision_label_coverage(
        [forecast], settlements[:1], date(2026, 9, 23), date(2026, 9, 28), date(2026, 9, 29)
    )
    assert partial == []
    assert partial_dates == set()


def test_post_freeze_coverage_still_requires_complete_mature_baskets() -> None:
    forecast = {
        "prediction_date": "2026-09-28",
        "planned_exit_date": "2026-10-08",
        "horizon_sessions": 3,
        "target_weights_by_method": {
            "trend": {"000001": 0.05},
            "market_plus_relative_fusion": {"000001": 0.05, "000002": 0.05},
        },
    }
    settlements = [
        {
            "prediction_date": "2026-09-28",
            "symbol": symbol,
            "planned_exit_date": "2026-10-08",
            "available_at": "2026-10-08",
            "status": "settled",
            "forward_return_3s": 0.01,
        }
        for symbol in ("000001", "000002")
    ]

    partial, partial_dates = _new_decision_label_coverage(
        [forecast], settlements[:1], date(2026, 9, 23), date(2026, 9, 28), date(2026, 10, 8)
    )
    assert partial_dates == set()
    assert partial[0]["settled_required_symbols"] == 1
    early, early_dates = _new_decision_label_coverage(
        [forecast], settlements, date(2026, 9, 23), date(2026, 9, 28), date(2026, 10, 7)
    )
    assert early_dates == set()
    assert not early[0]["complete_matured_decision"]
    complete, complete_dates = _new_decision_label_coverage(
        [forecast], settlements, date(2026, 9, 23), date(2026, 9, 28), date(2026, 10, 8)
    )
    assert complete_dates == {"2026-09-28"}
    assert complete[0]["complete_matured_decision"]


def test_2026_09_24_order_is_excluded_and_2026_09_28_is_first_eligible_decision() -> None:
    dates = (date(2026, 9, 23), date(2026, 9, 24), date(2026, 9, 28))

    diagnostic_dates, post_freeze_dates = _partition_forward_dates(dates)
    account_decisions = _post_freeze_account_decisions([
        {"prediction_date": "2026-09-23", "planned_execution_date": "2026-09-24"},
        {"prediction_date": "2026-09-24", "planned_execution_date": "2026-09-28"},
        {"decision_date": date(2026, 9, 28), "planned_execution_date": "2026-09-29"},
    ])

    assert diagnostic_dates == [date(2026, 9, 24)]
    assert post_freeze_dates == [date(2026, 9, 28)]
    actual_decisions = []
    for row in account_decisions:
        value = row.get("prediction_date", row.get("decision_date"))
        actual_decisions.append(value.isoformat() if isinstance(value, date) else value)
    assert actual_decisions == ["2026-09-23", "2026-09-28"]
    assert account_decisions[-1]["planned_execution_date"] == "2026-09-29"


def test_as_of_reuses_later_verified_cache_but_excludes_future_rows() -> None:
    research = ResearchData(
        [date(2026, 9, 23), date(2026, 9, 24), date(2026, 9, 28)],
        [
            {"trade_date": day, "available_at": day}
            for day in (date(2026, 9, 23), date(2026, 9, 24), date(2026, 9, 28))
        ],
        [{"trade_date": date(2026, 9, 28), "available_at": date(2026, 9, 28)}],
        [{"trade_date": day, "available_at": day}
         for day in (date(2026, 9, 23), date(2026, 9, 24), date(2026, 9, 28))],
        [
            MembershipSnapshot(date(2026, 9, 21), date(2026, 9, 23), frozenset({"000001"}), "old"),
            MembershipSnapshot(date(2026, 9, 28), date(2026, 9, 28), frozenset({"000002"}), "future"),
        ],
    )

    sliced, cache_through, excluded_dates = _research_for_as_of(research, date(2026, 9, 24))

    assert cache_through == date(2026, 9, 28)
    assert excluded_dates == [date(2026, 9, 28)]
    assert sliced.trading_dates == (date(2026, 9, 23), date(2026, 9, 24))
    assert all(row["available_at"] <= date(2026, 9, 24) for row in sliced.adjusted_bars)
    assert all(row["available_at"] <= date(2026, 9, 24) for row in sliced.index_bars)
    assert [row.available_at for row in sliced.membership_snapshots] == [date(2026, 9, 23)]


def test_as_of_must_be_a_session_in_the_source_cache() -> None:
    research = ResearchData([date(2026, 9, 23)], [], [], [], [])
    with pytest.raises(ValueError, match="不在 BaoStock 缓存交易日中"):
        _research_for_as_of(research, date(2026, 9, 24))


def test_manifest_response_hash_is_checked(tmp_path: Path) -> None:
    response = tmp_path / "bars.csv"
    response.write_text("date,close\n2026-09-23,10\n", encoding="utf-8")
    expected = hashlib.sha256(response.read_bytes()).hexdigest()
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"response_file": "bars.csv", "sha256": expected}) + "\n",
        encoding="utf-8",
    )
    assert _verify_baostock_responses(tmp_path) == 1
    response.write_text("date,close\n2026-09-23,11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="原始响应哈希不符"):
        _verify_baostock_responses(tmp_path)
