"""Canonical mapping and validation for the selected training CSV."""

import csv
import math
from datetime import date
from pathlib import Path


SOURCE_TO_CANONICAL = {
    "股票代码": "symbol",
    "日期": "trade_date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌额": "price_change",
    "换手率": "turnover",
    "涨跌幅": "return_1d",
}
PRICE_COLUMNS = ("open", "high", "low", "close")


def normalize_stock_code(code: str) -> str:
    """Return a numeric mainland stock code as a six-character string."""
    if not code or not code.isascii() or not code.isdigit() or len(code) > 6:
        raise ValueError(f"股票代码必须是 1 到 6 位 ASCII 数字：{code!r}")
    return code.zfill(6)


def load_training_bars(data_dir: str | Path) -> list[dict[str, object]]:
    """Load and validate only `train.csv` from an external data directory."""
    path = Path(data_dir) / "train.csv"
    bars: list[dict[str, object]] = []
    seen_keys: set[tuple[str, date]] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"训练数据没有表头：{path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"训练数据含重复列名：{path}")

        missing_columns = [name for name in SOURCE_TO_CANONICAL if name not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"训练数据缺少必需列 {missing_columns}：{path}")

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"训练数据第 {line_number} 行列数与表头不符：{path}")

            symbol_value = row["股票代码"]
            date_value = row["日期"]
            if not symbol_value or not date_value:
                raise ValueError(f"训练数据第 {line_number} 行代码或日期为空：{path}")

            bar: dict[str, object] = {
                "symbol": normalize_stock_code(symbol_value),
                "trade_date": date.fromisoformat(date_value),
            }
            for source_name, canonical_name in SOURCE_TO_CANONICAL.items():
                if canonical_name in ("symbol", "trade_date"):
                    continue
                value = row[source_name]
                if value is None or value == "":
                    if canonical_name in PRICE_COLUMNS:
                        raise ValueError(
                            f"训练数据第 {line_number} 行必需价格列 {source_name} 为空：{path}"
                        )
                    bar[canonical_name] = None
                    continue

                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(
                        f"训练数据第 {line_number} 行数值列 {source_name} 非有限：{path}"
                    )
                bar[canonical_name] = number

            open_price = bar["open"]
            high_price = bar["high"]
            low_price = bar["low"]
            close_price = bar["close"]
            if (
                low_price > min(open_price, close_price)
                or high_price < max(open_price, close_price)
                or low_price > high_price
            ):
                raise ValueError(f"训练数据第 {line_number} 行 OHLC 关系无效：{path}")

            key = (bar["symbol"], bar["trade_date"])
            if key in seen_keys:
                raise ValueError(f"训练数据存在重复 (symbol, trade_date)：{key}")
            seen_keys.add(key)
            bars.append(bar)

    if not bars:
        raise ValueError(f"训练数据没有记录：{path}")
    return bars
