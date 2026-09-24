import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from adaptive_mas.data import load_training_bars, normalize_stock_code


SOURCE_COLUMNS = [
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
VALID_ROW = {
    "股票代码": "600000",
    "日期": "2024-01-02",
    "开盘": "10",
    "收盘": "11",
    "最高": "12",
    "最低": "9",
    "成交量": "100",
    "成交额": "1100.5",
    "振幅": "30",
    "涨跌额": "1",
    "换手率": "0.5",
    "涨跌幅": "10",
}


class MarketDataBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_training_csv(self, rows, columns=SOURCE_COLUMNS):
        path = self.data_dir / "train.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                [{column: row.get(column, "") for column in columns} for row in rows]
            )
        return path

    def test_stock_codes_remain_six_character_strings(self):
        self.assertEqual(normalize_stock_code("600000"), "600000")
        self.assertEqual(normalize_stock_code("000001"), "000001")
        self.assertEqual(normalize_stock_code("1"), "000001")

    def test_missing_required_source_column_fails_loudly(self):
        columns = [column for column in SOURCE_COLUMNS if column != "最高"]
        self.write_training_csv([VALID_ROW], columns=columns)
        with self.assertRaisesRegex(ValueError, "缺少必需列"):
            load_training_bars(self.data_dir)

    def test_duplicate_canonical_key_fails_loudly(self):
        first = dict(VALID_ROW, **{"股票代码": "1"})
        second = dict(VALID_ROW, **{"股票代码": "000001"})
        self.write_training_csv([first, second])
        with self.assertRaisesRegex(ValueError, "重复"):
            load_training_bars(self.data_dir)

    def test_invalid_ohlc_relationship_fails_loudly(self):
        invalid = dict(VALID_ROW, **{"最高": "10.5"})
        self.write_training_csv([invalid])
        with self.assertRaisesRegex(ValueError, "OHLC"):
            load_training_bars(self.data_dir)

    def test_loading_does_not_modify_source_csv(self):
        source = self.write_training_csv([VALID_ROW])
        before = hashlib.sha256(source.read_bytes()).digest()
        load_training_bars(self.data_dir)
        after = hashlib.sha256(source.read_bytes()).digest()
        self.assertEqual(before, after)

    def test_research_loader_reads_training_file_only(self):
        self.write_training_csv([VALID_ROW])
        (self.data_dir / "test.csv").write_text("invalid scoring file", encoding="utf-8")
        self.assertEqual(load_training_bars(self.data_dir)[0]["symbol"], "600000")

        production_dir = Path(__file__).parents[1] / "src" / "adaptive_mas"
        production_sources = list(production_dir.rglob("*.py"))
        self.assertTrue(production_sources)
        self.assertTrue(
            all("test.csv" not in source.read_text(encoding="utf-8") for source in production_sources)
        )


if __name__ == "__main__":
    unittest.main()
