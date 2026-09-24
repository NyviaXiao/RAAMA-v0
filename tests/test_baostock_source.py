from datetime import date

import pytest

from adaptive_mas.data import baostock_source
from adaptive_mas.data.baostock_source import BaoStockSource


class FakeResponse:
    def __init__(self, fields, rows, error_code="0", error_msg=""):
        self.fields = fields
        self.rows = rows
        self.error_code = error_code
        self.error_msg = error_msg
        self.position = 0

    def next(self):
        if self.position == len(self.rows):
            return False
        self.position += 1
        return True

    def get_row_data(self):
        return self.rows[self.position - 1]


def test_hs300_membership_uses_explicit_date_and_keeps_vendor_codes(monkeypatch):
    monkeypatch.setattr(baostock_source.bs, "login", lambda: FakeResponse([], []))
    monkeypatch.setattr(baostock_source.bs, "logout", lambda: FakeResponse([], []))
    monkeypatch.setattr(
        baostock_source.bs,
        "query_hs300_stocks",
        lambda date: FakeResponse(
            ["updateDate", "code", "code_name"],
            [["2024-01-01", "sh.600000", "浦发银行"]],
        ),
    )

    with BaoStockSource() as source:
        fields, rows = source.hs300_membership(date(2024, 1, 2))

    assert fields == ["updateDate", "code", "code_name"]
    assert rows[0]["code"] == "sh.600000"


def test_query_error_fails_loudly(monkeypatch):
    monkeypatch.setattr(baostock_source.bs, "login", lambda: FakeResponse([], []))
    monkeypatch.setattr(baostock_source.bs, "logout", lambda: FakeResponse([], []))
    monkeypatch.setattr(
        baostock_source.bs,
        "query_history_k_data_plus",
        lambda *args, **kwargs: FakeResponse([], [], "100010", "query denied"),
    )

    with BaoStockSource() as source:
        with pytest.raises(RuntimeError, match="query denied"):
            source.daily_bars(
                "sh.600000", date(2024, 1, 2), date(2024, 1, 5), adjustflag="3"
            )


def test_daily_bars_pass_explicit_adjustment_flag(monkeypatch):
    monkeypatch.setattr(baostock_source.bs, "login", lambda: FakeResponse([], []))
    monkeypatch.setattr(baostock_source.bs, "logout", lambda: FakeResponse([], []))
    requested = {}

    def query_history(code, fields, **parameters):
        requested.update(parameters)
        return FakeResponse([], [])

    monkeypatch.setattr(
        baostock_source.bs, "query_history_k_data_plus", query_history
    )

    with BaoStockSource() as source:
        source.daily_bars(
            "sh.600000", date(2024, 1, 2), date(2024, 1, 5), adjustflag="1"
        )

    assert requested["adjustflag"] == "1"


def test_malformed_external_row_fails_loudly(monkeypatch):
    monkeypatch.setattr(baostock_source.bs, "login", lambda: FakeResponse([], []))
    monkeypatch.setattr(baostock_source.bs, "logout", lambda: FakeResponse([], []))
    monkeypatch.setattr(
        baostock_source.bs,
        "query_hs300_stocks",
        lambda date: FakeResponse(["updateDate", "code"], [["2024-01-01"]]),
    )

    with BaoStockSource() as source:
        with pytest.raises(ValueError):
            source.hs300_membership(date(2024, 1, 2))


def test_invalid_request_code_fails_before_external_call(monkeypatch):
    monkeypatch.setattr(baostock_source.bs, "login", lambda: FakeResponse([], []))
    monkeypatch.setattr(baostock_source.bs, "logout", lambda: FakeResponse([], []))
    monkeypatch.setattr(
        baostock_source.bs,
        "query_history_k_data_plus",
        lambda *args, **kwargs: pytest.fail("API must not be called"),
    )

    with BaoStockSource() as source:
        with pytest.raises(ValueError, match="证券代码格式无效"):
            source.daily_bars(
                "600000", date(2024, 1, 2), date(2024, 1, 5), adjustflag="3"
            )
