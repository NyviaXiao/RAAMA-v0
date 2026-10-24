"""Versioned, offline parsing for hard portfolio risk constraints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any


PROFILE_SCHEMA_VERSION = "raama-risk-profile.v1"
TEMPLATE_VERSION = "raama-risk-templates.v1"

PARAMETER_UNITS = {
    "risk_budget_annualized": "annualized_volatility_fraction",
    "maximum_symbol_weight": "portfolio_weight_fraction",
    "maximum_one_way_turnover": "one_way_turnover_fraction",
    "minimum_cash_weight": "portfolio_weight_fraction",
}

TEMPLATES: dict[str, dict[str, float]] = {
    "conservative": {
        "risk_budget_annualized": 0.12,
        "maximum_symbol_weight": 0.05,
        "maximum_one_way_turnover": 0.10,
        "minimum_cash_weight": 0.25,
    },
    "balanced": {
        "risk_budget_annualized": 0.20,
        "maximum_symbol_weight": 0.05,
        "maximum_one_way_turnover": 0.50,
        "minimum_cash_weight": 0.05,
    },
    "aggressive": {
        "risk_budget_annualized": 0.30,
        "maximum_symbol_weight": 0.10,
        "maximum_one_way_turnover": 0.75,
        "minimum_cash_weight": 0.00,
    },
}

_TEMPLATE_WORDS: dict[str, tuple[str, ...]] = {
    "conservative": ("保守", "稳健", "conservative"),
    "balanced": ("平衡", "均衡", "balanced"),
    "aggressive": ("进取", "积极", "aggressive"),
}
_NUMBER = r"(?P<value>\d+(?:\.\d+)?)(?P<unit>%|％)?"
_UPPER_RELATION = r"(?:不超过|不高于|最多|最高|上限|≤|<=|小于等于|控制在|限制为|为)"
_LOWER_RELATION = r"(?:至少|不少于|不低于|最低|下限|大于|超过|>=|≥|>)"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "risk_budget_annualized",
        re.compile(
            rf"(?:年化(?:波动率|风险)(?:预算|上限)|风险预算(?:上限)?)\s*(?:{_UPPER_RELATION}\s*)?{_NUMBER}",
            re.IGNORECASE,
        ),
    ),
    (
        "risk_budget_annualized",
        re.compile(
            rf"(?:年化(?:波动率|风险))\s*{_UPPER_RELATION}\s*{_NUMBER}",
            re.IGNORECASE,
        ),
    ),
    (
        "maximum_symbol_weight",
        re.compile(
            rf"(?:单票|单只(?:证券|股票)|单个(?:标的|证券)|单名)(?:持仓)?(?:权重)?\s*{_UPPER_RELATION}\s*{_NUMBER}",
            re.IGNORECASE,
        ),
    ),
    (
        "maximum_one_way_turnover",
        re.compile(
            rf"(?:单向)?(?:换手|换手率|换手预算)(?:预算|上限)?\s*{_UPPER_RELATION}\s*{_NUMBER}",
            re.IGNORECASE,
        ),
    ),
    (
        "minimum_cash_weight",
        re.compile(
            rf"(?:(?:至少|不少于|不低于|最低)\s*)?(?:(?:保留|维持|保持)\s*)?(?:(?:至少|不少于|不低于|最低)\s*)?{_NUMBER}\s*(?:的)?现金",
            re.IGNORECASE,
        ),
    ),
    (
        "minimum_cash_weight",
        re.compile(
            rf"现金(?:比例)?\s*(?:最低|下限|至少|不低于|保留|为)?\s*{_NUMBER}",
            re.IGNORECASE,
        ),
    ),
    (
        "minimum_cash_weight",
        re.compile(
            rf"(?:最低|最少)(?:现金|现金比例)\s*(?:为|至少|不低于)?\s*{_NUMBER}",
            re.IGNORECASE,
        ),
    ),
)

_UNSUPPORTED_LOSS = re.compile(r"最大回撤|回撤|最多亏损|最大亏损|亏损|损失|本金损失|max(?:imum)?\s*(?:drawdown|loss)", re.IGNORECASE)
_NEGATION = re.compile(r"不要|不需要|无需|不用|不保留|不采取|不选|不能|不得|避免|否定")
_RISK_TERMS = re.compile(r"年化|风险|波动|单票|单只|单个|单名|换手|现金|亏损|回撤|损失|本金|仓位|持仓", re.IGNORECASE)
_DIRECTIONAL = re.compile(_LOWER_RELATION)
_LEFTOVER_ALLOWED = re.compile(
    r"(?:我|希望|想|偏|倾向于|偏向|选择|采用|使用|请|按|设为|设置为|给我|的|并且|以及|同时|和|与|且|，|。|、|；|;|,|\s)+"
)
_CONSTRAINT_KEYS = set(PARAMETER_UNITS)


def _stable_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _number_value(raw: str, unit: str | None) -> float:
    value = float(raw)
    if unit:
        value /= 100.0
    elif value >= 1:
        raise ValueError("百分比数值需要带 %；不带单位的整数有歧义")
    return value


def _validate_parameter(key: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key} 必须是有限数值")
    if key == "risk_budget_annualized":
        if not 0 < number <= 2:
            raise ValueError("risk_budget_annualized 必须处于 (0, 2]")
    elif key == "maximum_symbol_weight":
        if not 0 < number <= 1:
            raise ValueError("maximum_symbol_weight 必须处于 (0, 1]")
    elif key == "maximum_one_way_turnover":
        if not 0 <= number <= 1:
            raise ValueError("maximum_one_way_turnover 必须处于 [0, 1]")
    elif key == "minimum_cash_weight":
        if not 0 <= number < 1:
            raise ValueError("minimum_cash_weight 必须处于 [0, 1)")
    else:
        raise ValueError(f"unsupported risk constraint: {key}")
    return number


def _parse_expression(expression: str) -> tuple[dict[str, list[dict[str, object]]], list[str], list[tuple[int, int, str]]]:
    found: dict[str, list[dict[str, object]]] = {key: [] for key in _CONSTRAINT_KEYS}
    unresolved: list[str] = []
    consumed: list[tuple[int, int, str]] = []

    if not expression.strip():
        return found, unresolved, consumed
    if _UNSUPPORTED_LOSS.search(expression):
        unresolved.append("最大回撤/亏损要求不属于年化波动率预算，当前不支持将其换算成其他约束。")
    if _NEGATION.search(expression) and _RISK_TERMS.search(expression):
        unresolved.append("检测到风险条件或模板前存在否定表达，无法安全判断用户意图。")

    # Upper bounds applied to cash and lower bounds applied to other risk limits
    # are not accepted as if they were the supported direction.
    for match in re.finditer(rf"(?:年化(?:波动率|风险)|风险预算|单票(?:权重)?|单只(?:证券|股票)(?:权重)?|单个(?:标的|证券)(?:权重)?|单名(?:权重)?|(?:单向)?换手(?:率|预算)?)\s*{_LOWER_RELATION}\s*{_NUMBER}", expression, re.IGNORECASE):
        unresolved.append(f"不支持或方向相反的风险关系：{match.group(0)}")
        consumed.append((*match.span(), "unsupported"))
    for match in re.finditer(rf"现金(?:比例)?\s*{_UPPER_RELATION}\s*{_NUMBER}", expression, re.IGNORECASE):
        unresolved.append(f"现金要求需要明确最低比例；无法将该表达当作最低现金约束：{match.group(0)}")
        consumed.append((*match.span(), "unsupported"))

    for key, pattern in _PATTERNS:
        for match in pattern.finditer(expression):
            try:
                raw, unit = match.group("value"), match.group("unit")
                value = _validate_parameter(key, _number_value(raw, unit))
            except (ValueError, TypeError) as exc:
                unresolved.append(f"{match.group(0)}：{exc}")
                consumed.append((*match.span(), "unsupported"))
                continue
            found[key].append({"value": value, "evidence": match.group(0), "source": "natural_language_expression"})
            consumed.append((*match.span(), "explicit"))

    # Template terms in the expression are allowed alongside numeric clauses.
    template_hits: list[tuple[str, int, int, str]] = []
    for template_id, words in _TEMPLATE_WORDS.items():
        for word in words:
            for match in re.finditer(re.escape(word), expression, re.IGNORECASE):
                template_hits.append((template_id, *match.span(), match.group(0)))
    return found, unresolved, consumed + [(start, end, "template:" + template_id) for template_id, start, end, _ in template_hits]


def _canonical_profile_hash(profile: Mapping[str, object]) -> str:
    return _stable_hash({key: value for key, value in profile.items() if key != "profile_sha256"})


def resolve_risk_profile(
    *,
    risk_template: str | None = None,
    risk_expression: str | None = None,
    risk_constraints: Mapping[str, object] | None = None,
    risk_budget_annualized: object | None = None,
    base_defaults: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve templates, supported Chinese clauses, and explicit numeric fields.

    Explicit numeric fields outrank a template. Conflicting explicit sources are
    retained in ``unresolved`` and make the profile non-executable.
    """
    unresolved: list[str] = []
    if risk_expression is not None and not isinstance(risk_expression, str):
        unresolved.append("risk_expression 必须是文本。")
    expression = risk_expression if isinstance(risk_expression, str) else ""
    natural, parse_issues, consumed = _parse_expression(expression)
    unresolved.extend(parse_issues)

    chosen_templates = {item[2].split(":", 1)[1] for item in consumed if item[2].startswith("template:")}
    if risk_template is not None and not isinstance(risk_template, str):
        unresolved.append("risk_template 必须是文本标识。")
        risk_template = None
    selected_template = (risk_template or "").strip().lower() or None
    if selected_template is not None and selected_template not in TEMPLATES:
        unresolved.append(f"未知风险模板：{selected_template}")
        selected_template = None
    if len(chosen_templates) > 1:
        unresolved.append("表达中同时出现多个风险模板，需明确选择一个。")
    if selected_template and chosen_templates and selected_template not in chosen_templates:
        unresolved.append("所选风险模板与文本中提到的模板不一致。")
    if not selected_template and len(chosen_templates) == 1:
        selected_template = next(iter(chosen_templates))

    # Remove parsed clauses and template words, then reject leftover numerics,
    # relation words, or risk clauses rather than partially executing them.
    residue = list(expression)
    for start, end, _ in consumed:
        for index in range(start, end):
            residue[index] = " "
    remaining = "".join(residue)
    remaining = re.sub(r"保守|稳健|conservative|平衡|均衡|balanced|进取|积极|aggressive", " ", remaining, flags=re.IGNORECASE)
    remaining = _LEFTOVER_ALLOWED.sub(" ", remaining)
    remaining = re.sub(r"[：:，。；;、,\-—()（）\s]+", " ", remaining).strip()
    if remaining and (_RISK_TERMS.search(remaining) or _DIRECTIONAL.search(remaining) or re.search(r"\d|[%％]", remaining)):
        unresolved.append(f"有未能完整解析的风险表达：{remaining}")
    elif remaining:
        # Free-form prose is unsupported unless all remaining text is ordinary
        # request framing. Do not silently ignore it as a valid constraint.
        if not re.fullmatch(r"(?:请|我|希望|想|倾向于|偏向|选择|采用|使用|给我|按|的|模式|模板|方案|风险|约束|组合|投资|设为|设置为|就好|即可|可以|吧|呢| and|or|the|profile)+", remaining.replace(" ", ""), re.IGNORECASE):
            unresolved.append(f"有未能完整解析的表达：{remaining}")

    explicit: dict[str, list[dict[str, object]]] = {key: list(values) for key, values in natural.items()}
    if risk_constraints is not None:
        if not isinstance(risk_constraints, Mapping):
            unresolved.append("risk_constraints 必须是结构化对象。")
        else:
            unknown = set(risk_constraints) - _CONSTRAINT_KEYS
            if unknown:
                unresolved.append(f"不支持的结构化风险字段：{sorted(unknown)}")
            for key, raw_value in risk_constraints.items():
                if key not in _CONSTRAINT_KEYS:
                    continue
                try:
                    value = _validate_parameter(key, raw_value)
                except ValueError as exc:
                    unresolved.append(str(exc))
                    continue
                explicit[key].append({"value": value, "evidence": f"risk_constraints.{key}", "source": "structured_constraint"})
    if risk_budget_annualized is not None:
        try:
            explicit["risk_budget_annualized"].append({
                "value": _validate_parameter("risk_budget_annualized", risk_budget_annualized),
                "evidence": "risk_budget_annualized",
                "source": "legacy_numeric_field",
            })
        except ValueError as exc:
            unresolved.append(str(exc))

    defaults = {
        "risk_budget_annualized": 0.20,
        "maximum_symbol_weight": 0.05,
        "maximum_one_way_turnover": 0.50,
        "minimum_cash_weight": 0.0,
    }
    if base_defaults:
        for key in _CONSTRAINT_KEYS:
            if key in base_defaults:
                try:
                    defaults[key] = _validate_parameter(key, base_defaults[key])
                except ValueError as exc:
                    unresolved.append(f"系统配置默认值无效：{exc}")

    template_values = TEMPLATES.get(selected_template, {}) if selected_template else {}
    constraints: dict[str, float] = {}
    sources: dict[str, list[dict[str, object]]] = {}
    for key in sorted(_CONSTRAINT_KEYS):
        candidates = explicit[key]
        distinct = {round(float(item["value"]), 12) for item in candidates}
        if len(distinct) > 1:
            unresolved.append(f"同一约束存在冲突的明确数值：{key}={sorted(distinct)}")
        if candidates:
            constraints[key] = float(candidates[0]["value"])
            sources[key] = candidates
        elif key in template_values:
            constraints[key] = float(template_values[key])
            sources[key] = [{"value": constraints[key], "evidence": selected_template, "source": "template_default"}]
        else:
            constraints[key] = float(defaults[key])
            sources[key] = [{"value": constraints[key], "evidence": "system_configuration", "source": "system_default"}]

    profile: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "template": selected_template,
        "template_version": TEMPLATE_VERSION if selected_template else None,
        "original_expression": expression,
        "constraints": constraints,
        "units": dict(PARAMETER_UNITS),
        "sources": sources,
        "unresolved": list(dict.fromkeys(unresolved)),
        "status": "resolved" if not unresolved else "needs_clarification",
        "execution_allowed": not unresolved,
        "drawdown_guarantee": False,
        "scope_note": "用户选定的约束模板，不构成适当性评估；年化波动率预算不保证最大回撤或最大亏损。",
    }
    profile["profile_sha256"] = _canonical_profile_hash(profile)
    return profile


def validate_canonical_risk_profile(profile: Mapping[str, object]) -> dict[str, object]:
    """Check a profile handed from the API queue to the independent worker."""
    if not isinstance(profile, Mapping):
        raise ValueError("risk profile must be a JSON object")
    value = dict(profile)
    if value.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("risk profile schema version is unsupported")
    if value.get("execution_allowed") is not True or value.get("unresolved"):
        raise ValueError("risk profile is unresolved and cannot be executed")
    supplied_hash = value.get("profile_sha256")
    if not isinstance(supplied_hash, str) or supplied_hash != _canonical_profile_hash(value):
        raise ValueError("risk profile integrity check failed")
    constraints = value.get("constraints")
    if not isinstance(constraints, Mapping) or set(constraints) != _CONSTRAINT_KEYS:
        raise ValueError("risk profile constraints are incomplete")
    for key, raw_value in constraints.items():
        _validate_parameter(key, raw_value)
    if value.get("units") != PARAMETER_UNITS:
        raise ValueError("risk profile units are invalid")
    return value


def apply_risk_profile(config: Mapping[str, Any], profile: Mapping[str, object]) -> dict[str, Any]:
    """Return config with profile limits and auditable profile payload applied."""
    checked = validate_canonical_risk_profile(profile)
    constraints = checked["constraints"]
    assert isinstance(constraints, Mapping)
    return {
        **dict(config),
        **{key: float(value) for key, value in constraints.items()},
        "risk_profile": checked,
    }
