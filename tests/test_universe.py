from datetime import date

import pytest

from adaptive_mas.data import hs300_members_at


def test_hs300_members_at_never_uses_a_future_snapshot(tmp_path):
    (tmp_path / "asof-2024-01-02.csv").write_text(
        "updateDate,code,code_name\n2024-01-01,sz.000001,平安银行\n",
        encoding="utf-8",
    )
    (tmp_path / "asof-2024-01-08.csv").write_text(
        "updateDate,code,code_name\n2024-01-08,sh.600000,浦发银行\n",
        encoding="utf-8",
    )

    assert hs300_members_at(tmp_path, date(2024, 1, 5)) == {"000001"}
    assert hs300_members_at(tmp_path, date(2024, 1, 8)) == {"000001"}
    assert hs300_members_at(tmp_path, date(2024, 1, 9)) == {"600000"}


def test_hs300_members_at_preserves_leading_zero_and_rejects_missing_history(tmp_path):
    (tmp_path / "asof-2024-01-02.csv").write_text(
        "updateDate,code,code_name\n2024-01-01,sz.000001,平安银行\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="没有在决策日前可用"):
        hs300_members_at(tmp_path, date(2023, 12, 29))

    assert hs300_members_at(tmp_path, date(2024, 1, 2)) == {"000001"}
