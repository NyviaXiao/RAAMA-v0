import csv
import json

from scripts.acquire_public_data import _membership_union, _record_universe_sources


def _write_snapshot(run_dir, query_date, update_date, code):
    membership = run_dir / "membership"
    membership.mkdir(parents=True)
    with (membership / f"asof-{query_date}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=["updateDate", "code"])
        writer.writeheader()
        writer.writerow({"updateDate": update_date, "code": code})
    (run_dir / "manifest.jsonl").write_text("{}\n", encoding="utf-8")


def test_history_pool_unions_two_baostock_runs_and_records_coverage_gap(tmp_path):
    first_run = tmp_path / "first"
    second_run = tmp_path / "second"
    first_run.mkdir()
    second_run.mkdir()
    _write_snapshot(first_run, "2024-01-02", "2024-01-01", "sz.000001")
    _write_snapshot(second_run, "2024-01-16", "2024-01-15", "sh.600000")

    codes, start, end, dates, provenance = _membership_union(
        [first_run, second_run]
    )
    output_dir = tmp_path / "akshare"
    _record_universe_sources(output_dir, provenance, dates, codes)
    recorded = json.loads(
        (output_dir / "universe-sources.jsonl").read_text(encoding="utf-8")
    )

    assert codes == ["sh.600000", "sz.000001"]
    assert start.isoformat() == "2024-01-02"
    assert end.isoformat() == "2024-01-16"
    assert len(recorded["runs"]) == 2
    assert all(run["baostock_manifest_sha256"] for run in recorded["runs"])
    assert recorded["snapshot_calendar_intervals"] == [
        {"after": "2024-01-02", "before": "2024-01-16", "gap_days": 13}
    ]
