"""The forward gate needs complete mature baskets and intact BaoStock responses."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_p9_forward_validation import _new_decision_label_coverage, _verify_baostock_responses


def test_partial_symbol_settlement_is_not_a_mature_forward_decision() -> None:
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
        [forecast], settlements[:1], date(2026, 9, 23), date(2026, 9, 29)
    )
    assert partial_dates == set()
    assert partial[0]["settled_required_symbols"] == 1
    early, early_dates = _new_decision_label_coverage(
        [forecast], settlements, date(2026, 9, 23), date(2026, 9, 28)
    )
    assert early_dates == set()
    assert not early[0]["complete_matured_decision"]
    complete, complete_dates = _new_decision_label_coverage(
        [forecast], settlements, date(2026, 9, 23), date(2026, 9, 29)
    )
    assert complete_dates == {"2026-09-24"}
    assert complete[0]["complete_matured_decision"]


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
