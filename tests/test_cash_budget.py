from datetime import date

import pytest

from adaptive_mas.backtest.simulator import _cash_after_fill


def test_cash_after_fill_clamps_only_rounding_noise() -> None:
    assert _cash_after_fill(-5e-9, trade_date=date(2024, 1, 2), symbol="000001") == (0.0, 5e-9)
    assert _cash_after_fill(2.5, trade_date=date(2024, 1, 2), symbol="000001") == (2.5, 0.0)


def test_cash_after_fill_rejects_real_deficit_with_context() -> None:
    with pytest.raises(ValueError, match="超出可用现金.*2024-01-02 000001 cash_cny=-"):
        _cash_after_fill(-1e-6, trade_date=date(2024, 1, 2), symbol="000001")
