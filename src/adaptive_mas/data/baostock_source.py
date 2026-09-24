"""Direct BaoStock boundary for historical market records."""

from datetime import date
from typing import Any

import baostock as bs


class BaoStockSource:
    def __enter__(self) -> "BaoStockSource":
        result = bs.login()
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败：{result.error_code} {result.error_msg}")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        result = bs.logout()
        if result.error_code != "0":
            logout_error = RuntimeError(
                f"BaoStock 登出失败：{result.error_code} {result.error_msg}"
            )
            if exc_value is not None:
                raise ExceptionGroup(
                    "BaoStock 查询与登出均失败", [exc_value, logout_error]
                )
            raise logout_error

    @staticmethod
    def _records(result: Any) -> tuple[list[str], list[dict[str, str]]]:
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock 查询失败：{result.error_code} {result.error_msg}")

        fields = list(result.fields)
        rows = []
        while result.next():
            rows.append(dict(zip(fields, result.get_row_data(), strict=True)))
        return fields, rows

    def hs300_membership(self, as_of: date) -> tuple[list[str], list[dict[str, str]]]:
        return self._records(bs.query_hs300_stocks(date=as_of.isoformat()))

    def daily_bars(
        self, code: str, start_date: date, end_date: date, adjustflag: str
    ) -> tuple[list[str], list[dict[str, str]]]:
        self._validate_request(code, start_date, end_date)
        if adjustflag not in {"1", "3"}:
            raise ValueError(f"不支持的 BaoStock 复权标记：{adjustflag!r}")
        return self._records(
            bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg",
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                frequency="d",
                adjustflag=adjustflag,
            )
        )

    def index_daily(
        self, code: str, start_date: date, end_date: date
    ) -> tuple[list[str], list[dict[str, str]]]:
        self._validate_request(code, start_date, end_date)
        return self._records(
            bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,preclose,pctChg",
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                frequency="d",
            )
        )

    @staticmethod
    def _validate_request(code: str, start_date: date, end_date: date) -> None:
        exchange, separator, security_code = code.partition(".")
        if (
            not separator
            or exchange not in {"sh", "sz"}
            or len(security_code) != 6
            or not security_code.isascii()
            or not security_code.isdigit()
        ):
            raise ValueError(f"BaoStock 证券代码格式无效：{code!r}")
        if start_date > end_date:
            raise ValueError("开始日期不能晚于结束日期")
