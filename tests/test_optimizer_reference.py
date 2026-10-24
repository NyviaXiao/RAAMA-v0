from __future__ import annotations

import pytest

pytest.importorskip("scipy.optimize")

from adaptive_mas.optimizer_reference import compare_production, representative_case


def test_production_optimizer_comparison_reports_gap_convergence_and_feasibility():
    report = compare_production(representative_case())
    assert report["fixture_label"] == "SYNTHETIC VERIFICATION FIXTURE ONLY"
    assert report["reference"]["success"] is True
    assert report["reference"]["feasibility"]["feasible"] is True
    assert report["final_comparison"]["production_feasible"] is True
    assert len(report["convergence"]) >= 5
    assert all(row["feasibility"]["feasible"] for row in report["convergence"])
    assert report["final_comparison"]["objective_gap_to_reference"] >= -1e-8
    assert report["interpretation"].find("not represented as optimal") >= 0
    boundary = report["boundary_cases"]
    over_cap = boundary["current_holding_above_target_cap"]
    assert over_cap["production"]["status"] == "feasible"
    assert over_cap["production"]["target_weights"]["600000"] <= 0.1 + 1e-7
    assert over_cap["reference"]["success"] is True
    assert over_cap["reference"]["target_weights"]["600000"] <= 0.1 + 1e-7
    for case_name in (
        "target_cap_incompatible_with_capacity",
        "target_cap_incompatible_with_turnover",
    ):
        assert boundary[case_name]["production"]["status"] == "fail_closed"
        assert boundary[case_name]["reference"]["status"] == "infeasible_case"
