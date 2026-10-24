from __future__ import annotations

import pytest

from adaptive_mas.risk_profile import (
    TEMPLATE_VERSION,
    apply_risk_profile,
    resolve_risk_profile,
    validate_canonical_risk_profile,
)


@pytest.mark.parametrize(
    ("template", "risk", "symbol", "turnover", "cash"),
    [
        ("conservative", 0.12, 0.05, 0.10, 0.25),
        ("balanced", 0.20, 0.05, 0.50, 0.05),
        ("aggressive", 0.30, 0.10, 0.75, 0.00),
    ],
)
def test_versioned_templates_resolve_to_auditable_constraints(template, risk, symbol, turnover, cash):
    profile = resolve_risk_profile(risk_template=template)

    assert profile["execution_allowed"] is True
    assert profile["template_version"] == TEMPLATE_VERSION
    assert profile["constraints"] == {
        "risk_budget_annualized": risk,
        "maximum_symbol_weight": symbol,
        "maximum_one_way_turnover": turnover,
        "minimum_cash_weight": cash,
    }
    assert all(source["source"] == "template_default" for rows in profile["sources"].values() for source in rows)
    assert validate_canonical_risk_profile(profile) == profile


def test_explicit_values_override_template_defaults_and_percent_units_convert():
    profile = resolve_risk_profile(
        risk_template="conservative",
        risk_expression="年化波动率不超过15%，单票不超过5%，单向换手不超过30%，至少保留20%现金",
    )

    assert profile["execution_allowed"] is True
    assert profile["constraints"] == {
        "risk_budget_annualized": 0.15,
        "maximum_symbol_weight": 0.05,
        "maximum_one_way_turnover": 0.30,
        "minimum_cash_weight": 0.20,
    }
    assert profile["sources"]["risk_budget_annualized"][0]["source"] == "natural_language_expression"
    assert profile["sources"]["maximum_one_way_turnover"][0]["evidence"] == "单向换手不超过30%"


def test_explicit_numeric_field_overrides_template_without_treating_template_as_conflict():
    profile = resolve_risk_profile(risk_template="conservative", risk_budget_annualized=0.20)
    assert profile["execution_allowed"] is True
    assert profile["constraints"]["risk_budget_annualized"] == 0.20
    assert profile["sources"]["risk_budget_annualized"][0]["source"] == "legacy_numeric_field"


@pytest.mark.parametrize(
    "expression",
    [
        "年化风险不超过0.15",
        "现金不低于20%",
        "至少保留20%现金",
        "我偏保守",
    ],
)
def test_common_supported_chinese_expressions_resolve(expression):
    profile = resolve_risk_profile(risk_expression=expression)
    assert profile["execution_allowed"] is True


@pytest.mark.parametrize(
    "expression",
    [
        "不要保守",
        "年化风险至少15%",
        "不需要保留20%现金",
        "单票不超过5%且不低于8%",
        "最多亏损10%",
        "最大回撤10%",
    ],
)
def test_unsupported_negated_or_contradictory_requests_fail_closed(expression):
    profile = resolve_risk_profile(risk_expression=expression)
    assert profile["execution_allowed"] is False
    assert profile["status"] == "needs_clarification"
    assert profile["unresolved"]


def test_same_metric_conflict_between_explicit_sources_is_not_resolved_by_precedence():
    profile = resolve_risk_profile(
        risk_template="balanced",
        risk_expression="年化风险不超过15%",
        risk_budget_annualized=0.20,
        risk_constraints={"risk_budget_annualized": 0.10},
    )
    assert profile["execution_allowed"] is False
    assert any("冲突" in issue for issue in profile["unresolved"])


def test_boolean_numeric_input_and_tampered_profile_are_rejected():
    profile = resolve_risk_profile(risk_budget_annualized=True)
    assert profile["execution_allowed"] is False
    resolved = resolve_risk_profile(risk_template="balanced")
    with pytest.raises(ValueError, match="integrity"):
        validate_canonical_risk_profile(resolved | {"profile_sha256": "0" * 64})


def test_profile_application_preserves_profile_and_sets_optimizer_constraints():
    profile = resolve_risk_profile(risk_template="balanced")
    config = apply_risk_profile({"maximum_symbol_weight": 0.99, "risk_aversion": 8}, profile)
    assert config["risk_budget_annualized"] == 0.20
    assert config["maximum_symbol_weight"] == 0.05
    assert config["minimum_cash_weight"] == 0.05
    assert config["risk_aversion"] == 8
    assert config["risk_profile"] == profile
