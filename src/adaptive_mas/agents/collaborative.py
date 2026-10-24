"""Evidence-grounded roles, one-round challenge, and final-plan explanations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from zoneinfo import ZoneInfo

from adaptive_mas.agents.qwen_expert import call_qwen_research_role
from adaptive_mas.agents.quantitative import AGENT_IDS, quantitative_agents
from adaptive_mas.evaluation.expert_memory import build_evidence_challenges


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _unique_sources(findings):
    return sorted({
        str(source["source_record_id"])
        for finding in findings
        for source in finding["evidence"]
    })


def _summary(agent_id, title, findings, decision_time, conclusion_if_empty):
    by_symbol = {}
    for finding in findings:
        symbol = finding.get("symbol")
        if symbol:
            by_symbol.setdefault(str(symbol), []).append(finding)
    return {
        "agent_id": agent_id,
        "role": title,
        "status": "completed" if findings else "unavailable_no_source_records",
        "input_cutoff": decision_time.isoformat(),
        "available_at": max(
            [decision_time.isoformat()]
            + [str(row.get("available_at", decision_time.isoformat())) for row in findings]
        ),
        "conclusion": conclusion_if_empty if not findings else (
            f"{len(findings)} 条有来源的研究记录；各证券结论见 findings。"
        ),
        "findings": findings,
        "direction_scores_by_symbol": {
            symbol: [int(row["direction_score"]) for row in rows]
            for symbol, rows in sorted(by_symbol.items())
        },
        "source_record_ids": _unique_sources(findings),
        "tool_cost_cny": 0.0,
        "estimated_token_cost_cny": 0.0,
        "elapsed_ms": 0.0,
    }


def _public_finding(record, agent_id):
    score = int(record.get("direction_score", 0))
    return {
        "record_id": str(record["record_id"]),
        "agent_id": agent_id,
        "symbol": (
            str(record["symbol"]).rsplit(".", 1)[-1]
            if record.get("symbol") else None
        ),
        "event_type": str(record.get("event_type", "public_record")),
        "direction_score": score,
        "conclusion": str(record.get("rule", record.get("interpretation_scope", "source record"))),
        "evidence_text_scope": str(record.get("evidence_text_scope", record.get("interpretation_scope", "structured_public_record"))),
        "available_at": str(record.get("available_at", "")),
        "evidence": [
            {
                "source_record_id": str(item["source_record_id"]),
                "dataset": str(item.get("dataset", "")),
                "published_at": str(item.get("published_at", "")),
                "available_at": str(item.get("available_at", "")),
                **{
                    key: item[key]
                    for key in (
                        "title", "report_period", "revenue_yoy_pct",
                        "parent_net_profit_yoy_pct", "roe_pct",
                        "income_statement_revenue_yoy_pct",
                        "income_statement_parent_profit_yoy_pct",
                        "attention_index", "trade_date", "content", "url",
                        "pdf_url", "pdf_sha256", "text_sha256",
                        "full_text", "vintage_verified",
                    )
                    if key in item
                },
            }
            for item in record.get("evidence", [])
        ],
    }


def build_agent_reports(
    research,
    decision_date: date,
    forecast: dict[str, object],
    public_snapshot: dict[str, object] | None,
    *,
    config: dict[str, object] | None = None,
    enable_qwen: bool = False,
) -> dict[str, object]:
    """Run independent market, fundamental, news and sourced-attention roles."""
    if enable_qwen and decision_date < datetime.now(SHANGHAI).date():
        raise ValueError("live Qwen role reviews cannot be generated for a historical decision date")
    decision_time = datetime.combine(decision_date, datetime.min.time()).replace(
        hour=16, tzinfo=SHANGHAI
    )
    effective_decision_time = decision_time
    snapshot = research.snapshot_at(decision_date)
    parameters = config or {}
    outputs = quantitative_agents(
        snapshot,
        int(parameters.get("feature_trend_sessions", 20)),
        int(parameters.get("feature_reversal_sessions", 5)),
        int(parameters.get("feature_liquidity_return_sessions", 5)),
        int(parameters.get("feature_recent_amount_sessions", 5)),
        int(parameters.get("feature_reference_amount_sessions", 20)),
        horizon_sessions=int(forecast["horizon_sessions"]),
    )
    market_findings = []
    for agent_id in AGENT_IDS:
        for signal in outputs[agent_id]:
            record_id = hashlib.sha256(
                f"{agent_id}:{signal.symbol}:{decision_date}:{signal.version}".encode("utf-8")
            ).hexdigest()
            market_findings.append({
                "record_id": record_id,
                "agent_id": agent_id,
                "symbol": signal.symbol,
                "tool": agent_id,
                "direction_score": 1 if signal.score > 0 else -1 if signal.score < 0 else 0,
                "tool_score": signal.score,
                "conclusion": signal.evidence.feature,
                "input_cutoff": signal.input_cutoff.isoformat(),
                "available_at": decision_time.isoformat(),
                "evidence": [{
                    "source_record_id": f"quant:{record_id}",
                    "dataset": "BaoStock_adjusted_market_bars",
                    "feature": signal.evidence.feature,
                    "start_date": signal.evidence.start_date.isoformat(),
                    "end_date": signal.evidence.end_date.isoformat(),
                    "source_refs": list(signal.evidence.source_refs),
                    "available_at": decision_time.isoformat(),
                }],
            })

    public = public_snapshot or {}
    fundamental_findings = [
        _public_finding(row, "fundamental")
        for row in public.get("p9_outputs", [])
        if row.get("expert") == "fundamental_rule_v1"
    ]
    announcement_findings = [
        _public_finding(row, "fundamental")
        for row in public.get("p9_outputs", [])
        if row.get("expert") in {"announcement_title_rule_v1", "cninfo_full_text_llm_v1"}
    ]
    fundamental_findings.extend(announcement_findings)
    news_findings = [
        _public_finding(row, "news")
        for row in public.get("p10_outputs", [])
    ]
    news_findings.extend(
        _public_finding(row, "news_macro_context")
        for row in public.get("macro_outputs", [])
    )
    attention_findings = [
        _public_finding(row, "market_attention")
        for row in public.get("attention_outputs", [])
    ]

    reports = {
        "market": _summary(
            "market",
            "市场 Agent：调用趋势、反转、量价确认三个确定性量化工具",
            market_findings,
            decision_time,
            "没有达到特征窗口的点时行情工具输出。",
        ),
        "fundamental": _summary(
            "fundamental",
            "基本面 Agent：财务事实与公告证据",
            fundamental_findings,
            decision_time,
            "时点缓存中没有可用财务事实或公告记录。",
        ),
        "news": _summary(
            "news",
            "新闻 Agent：已归档轻量快讯与官方宏观上下文",
            news_findings,
            decision_time,
            "时点缓存中没有可用的快讯或已发布宏观记录。",
        ),
        "market_attention": _summary(
            "market_attention",
            "关注度 Agent：来源标记的市场关注指数",
            attention_findings,
            decision_time,
            "没有可用的 AKShare 用户关注指数；没有讨论帖文本，因此不报告文本情绪。",
        ),
    }
    reports["market_attention"]["interpretation_scope"] = (
        "source_reported_market_attention_only_not_discussion_text_sentiment"
    )

    if enable_qwen:
        for agent_id in ("market", "fundamental", "news", "market_attention"):
            report = reports[agent_id]
            if not report["findings"]:
                continue
            qwen_records = report["findings"][:12]
            result = call_qwen_research_role(
                agent_id,
                (
                    f"Review the {report['role']} findings for the configured "
                    "comparison horizon. Distinguish support, contrary evidence and "
                    "uncertainty; do not produce a return forecast."
                ),
                qwen_records,
                decision_time,
                int(forecast["horizon_sessions"]),
            )
            report["qwen_review"] = result
            report["estimated_token_cost_cny"] = result["estimated_token_cost_cny"] or 0.0
            report["elapsed_ms"] = result["elapsed_ms"]
            if result.get("completed_at"):
                completed_at = datetime.fromisoformat(str(result["completed_at"])).astimezone(SHANGHAI)
                if completed_at.date() != decision_date:
                    raise ValueError("Qwen role review completed after the decision date")
                effective_decision_time = max(effective_decision_time, completed_at)
                report["available_at"] = max(
                    datetime.fromisoformat(str(report["available_at"])).astimezone(SHANGHAI),
                    completed_at,
                ).isoformat()

    reports["research_inputs"] = {
        "decision_date": decision_date.isoformat(),
        "input_cutoff": decision_date.isoformat(),
        "effective_decision_time": effective_decision_time.isoformat(),
        "horizon_sessions": int(forecast["horizon_sessions"]),
        "calibrated_forecast_methods": sorted(
            forecast["expected_returns_by_method"]
        ),
        "forecast_errors": {
            name: value
            for name, value in forecast["forecast_errors_by_method"].items()
        },
        "public_snapshot_available_at": public.get("decision_time"),
        "public_snapshot_source_cache": public.get("source_cache"),
    }
    return reports


def one_round_challenge(reports: dict[str, object]) -> dict[str, object]:
    """Use P11's cited, one-round evidence challenge without changing forecasts."""
    inputs = reports["research_inputs"]
    prediction_date = str(inputs["decision_date"])
    horizon = int(inputs["horizon_sessions"])
    p11_records = []
    findings_by_id = {}
    for role in ("market", "fundamental", "news", "market_attention"):
        for finding in reports[role]["findings"]:
            if not finding.get("evidence"):
                continue
            record_id = str(finding["record_id"])
            findings_by_id[record_id] = finding
            p11_records.append({
                "expert": str(finding["agent_id"]),
                "record_id": record_id,
                "symbol": finding.get("symbol"),
                "prediction_date": prediction_date,
                "horizon_sessions": horizon,
                "direction_score": int(finding["direction_score"]),
                "rule": str(finding["conclusion"]),
                "rationale": str(finding["conclusion"]),
                "evidence_text_scope": str(finding.get("evidence_text_scope", "structured_market_data")),
                "evidence": finding["evidence"],
            })

    qwen_role_reviews = []
    for role in ("market", "fundamental", "news", "market_attention"):
        review = reports[role].get("qwen_review")
        if not review or review.get("status") != "completed":
            continue
        cited_ids = sorted({
            str(citation["source_record_id"])
            for side in ("supports", "opposes")
            for citation in review.get(side, [])
        })
        qwen_role_reviews.append({
            "role": role,
            "status": review["status"],
            "model": review.get("model"),
            "conclusion": review.get("conclusion"),
            "role_level_direction_score": review.get("direction_score"),
            "direction_scope": "role_level_only_not_a_per_security_return",
            "supports": review.get("supports", []),
            "opposes": review.get("opposes", []),
            "uncertainties": review.get("uncertainties", []),
            "cited_source_record_ids": cited_ids,
            "estimated_token_cost_cny": review.get("estimated_token_cost_cny"),
        })

    p11_challenges = build_evidence_challenges(p11_records)
    conflicts = []
    for challenge in p11_challenges:
        round_record = challenge["discussion_rounds"][0]
        conflicts.append({
            "symbol": challenge["symbol"],
            "horizon_sessions": challenge["horizon_sessions"],
            "challenge_id": challenge["challenge_id"],
            "bull_case": [round_record["positive_argument"]],
            "bear_case": [round_record["negative_argument"]],
            "round_count": 1,
            "resolution": round_record["judge"]["reason"],
            "p11_status": challenge["status"],
            "p11_judge_verdict": challenge["judge_verdict"],
            "evidence_refs": challenge["evidence_refs"],
            "qwen_role_reviews": [
                review for review in qwen_role_reviews
                if set(review["cited_source_record_ids"]) & {
                    str(source["source_record_id"])
                    for ref in challenge["evidence_refs"]
                    for source in ref["evidence"]
                }
            ],
            "direction_correctness": challenge["direction_correctness"],
            "original_forecasts_changed": challenge["original_forecasts_changed"],
            "decision_effect": challenge["decision_effect"],
        })
    return {
        "round_count": 1,
        "status": "evidence_conflicts_found" if conflicts else "no_cross_role_conflict",
        "conflicts": conflicts,
        "p11_evidence_challenges": p11_challenges,
        "qwen_role_reviews": qwen_role_reviews,
        "supplementary_query_status": "not_requested",
        "supplementary_query_cost_cny": 0.0,
    }


def run_supplementary_challenge(
    debate: dict[str, object],
    reports: dict[str, object],
    decision_date: date,
    horizon_sessions: int,
) -> dict[str, object]:
    """Make at most one Qwen challenge on the highest-priority cited disagreement."""
    if not debate["conflicts"]:
        return debate
    conflict = debate["conflicts"][0]
    findings = {
        row["record_id"]: row
        for role in ("market", "fundamental", "news")
        for row in reports[role]["findings"]
    }
    source_ids = sorted({
        source_id
        for side in ("bull_case", "bear_case")
        for item in conflict[side]
        for source_id in item["source_record_ids"]
    })
    evidence = [
        {
            "source_record_id": source_id,
            "dataset": next(
                source["dataset"]
                for row in findings.values()
                for source in row["evidence"]
                if source["source_record_id"] == source_id
            ),
            "available_at": next(
                source["available_at"]
                for row in findings.values()
                for source in row["evidence"]
                if source["source_record_id"] == source_id
            ),
        }
        for source_id in source_ids
    ]
    challenge = {
        "record_id": f"one-round-challenge:{conflict['symbol']}:{decision_date.isoformat()}",
        "symbol": conflict["symbol"],
        "direction_score": 0,
        "conclusion": json.dumps(conflict, ensure_ascii=False),
        "evidence": evidence,
    }
    review = call_qwen_research_role(
        "supplementary_bull_bear_challenge",
        "For this one disputed stock, test each side against the cited evidence. "
        "Name what the evidence supports, what it contradicts, and what remains unknown. "
        "Do not add facts and do not create an expected return.",
        [challenge],
        datetime.combine(decision_date, datetime.min.time()).replace(hour=16, tzinfo=SHANGHAI),
        horizon_sessions,
    )
    return {
        **debate,
        "supplementary_query_status": review["status"],
        "supplementary_query": review,
        "supplementary_query_cost_cny": review["estimated_token_cost_cny"] or 0.0,
        "supplementary_query_target_symbol": conflict["symbol"],
    }


def research_manager(
    forecast: dict[str, object],
    method: str,
    disagreement_reports: dict[str, object],
    forecast_error_multiplier: float,
    maximum_candidates: int = 30,
) -> dict[str, object]:
    """Rank calibrated returns and attach cited Qwen context without changing scores."""
    returns = forecast["expected_returns_by_method"][method]
    error = forecast["forecast_errors_by_method"][method]
    if error is None:
        return {
            "status": "warmup_no_matured_calibration_error",
            "candidate_priorities": [],
        }
    reports_by_symbol = {}
    symbols_by_source = {}
    for role in ("market", "fundamental", "news", "market_attention"):
        for finding in disagreement_reports[role]["findings"]:
            symbol = finding.get("symbol")
            if symbol:
                reports_by_symbol.setdefault(str(symbol), []).append(finding)
                for source in finding["evidence"]:
                    symbols_by_source.setdefault(str(source["source_record_id"]), set()).add(str(symbol))
    qwen_by_symbol = {}
    for role in ("market", "fundamental", "news", "market_attention"):
        review = disagreement_reports[role].get("qwen_review")
        if not review or review.get("status") != "completed":
            continue
        for stance in ("supports", "opposes"):
            for citation in review.get(stance, []):
                source_id = str(citation["source_record_id"])
                for symbol in symbols_by_source.get(source_id, set()):
                    if symbol not in returns:
                        continue
                    qwen_by_symbol.setdefault(symbol, []).append({
                        "role": role,
                        "stance": "supporting_citation" if stance == "supports" else "opposing_citation",
                        "reason": str(citation["reason"]),
                        "source_record_id": source_id,
                        "review_conclusion": str(review["conclusion"]),
                        "review_uncertainties": list(review.get("uncertainties", [])),
                        "review_model": review.get("model"),
                        "role_level_direction_score": review.get("direction_score"),
                        "direction_scope": "role_level_only_not_a_per_security_return",
                    })
    ranked = sorted(
        returns,
        key=lambda symbol: (
            -(float(returns[symbol]) - forecast_error_multiplier * float(error)),
            symbol,
        ),
    )
    candidates = []
    for rank, symbol in enumerate(ranked[:maximum_candidates], start=1):
        findings = reports_by_symbol.get(symbol, [])
        qwen_evidence = qwen_by_symbol.get(symbol, [])
        supporting_qwen = sum(row["stance"] == "supporting_citation" for row in qwen_evidence)
        opposing_qwen = sum(row["stance"] == "opposing_citation" for row in qwen_evidence)
        candidates.append({
            "priority": rank,
            "symbol": symbol,
            "calibrated_expected_return": float(returns[symbol]),
            "matured_forecast_error": float(error),
            "ranking_score": (
                float(returns[symbol])
                - forecast_error_multiplier * float(error)
            ),
            "ranking_rule": "calibrated_expected_return - configured forecast_error_multiplier * matured forecast error",
            "supporting_evidence": [
                {
                    "agent_id": row["agent_id"],
                    "conclusion": row["conclusion"],
                    "source_record_ids": [item["source_record_id"] for item in row["evidence"]],
                }
                for row in findings
                if int(row["direction_score"]) > 0
            ],
            "opposing_evidence": [
                {
                    "agent_id": row["agent_id"],
                    "conclusion": row["conclusion"],
                    "source_record_ids": [item["source_record_id"] for item in row["evidence"]],
                }
                for row in findings
                if int(row["direction_score"]) < 0
            ],
            "qwen_review_evidence": qwen_evidence,
            "qwen_evidence_view": (
                "Qwen cited both supporting and opposing evidence; retain a mixed narrative view."
                if supporting_qwen and opposing_qwen else
                "Qwen cited supporting evidence for this symbol."
                if supporting_qwen else
                "Qwen cited opposing evidence for this symbol."
                if opposing_qwen else
                "No symbol-linked Qwen citations were available."
            ),
            "qwen_evidence_changes_numeric_priority": False,
        })
    return {
        "status": "completed_from_mature_calibrated_returns",
        "method": method,
        "forecast_error_multiplier": forecast_error_multiplier,
        "candidate_priorities": candidates,
    }


def risk_role_reviews(decision: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    """Report three deterministic views of the same optimizer output without editing it."""
    target = decision["target_weights"]
    highest_weight = max(target.values(), default=0.0)
    nav = float(decision["current_nav_cny"])
    risk = decision["annualized_risk"]
    risk_budget = float(decision["annualized_risk_budget"])
    minimum_cash_weight = float(config.get("minimum_cash_weight", 0.0))
    cash_after_costs_weight = float(decision["estimated_cash_after_costs_cny"]) / nav
    common_checks = {
        "annualized_risk_within_user_budget": (
            risk is not None and risk <= risk_budget + 1e-10
        ),
        "single_name_weight_within_limit": (
            highest_weight <= float(config["maximum_symbol_weight"]) + 1e-10
        ),
        "cash_after_costs_nonnegative": decision["estimated_cash_after_costs_cny"] >= -1e-8,
        "minimum_cash_ratio_after_costs": (
            cash_after_costs_weight + 1e-10 >= minimum_cash_weight
        ),
        "turnover_within_limit": (
            decision["one_way_turnover"]
            <= float(config["maximum_one_way_turnover"]) + 1e-10
        ),
    }
    reviews = {}
    for role, lens, view_metrics in (
        (
            "aggressive",
            "weighs incremental value against forecast uncertainty",
            {
                "expected_gross_return": decision["expected_gross_return"],
                "forecast_error_allowance": decision["forecast_error_allowance"],
                "risk_adjusted_net_incremental_value_cny": decision["risk_adjusted_net_incremental_value_cny"],
            },
        ),
        (
            "neutral",
            "balances risk use and estimated trading cost",
            {
                "annualized_risk": risk,
                "risk_budget_annualized": risk_budget,
                "estimated_transaction_cost_cny": decision["expected_transaction_cost_cny"],
                "transaction_cost_nav_fraction": decision["expected_transaction_cost_cny"] / nav,
            },
        ),
        (
            "conservative",
            "focuses on remaining cash, concentration and turnover",
            {
                "cash_after_costs_cny": decision["estimated_cash_after_costs_cny"],
                "cash_after_costs_nav_fraction": decision["estimated_cash_after_costs_cny"] / nav,
                "minimum_cash_weight": minimum_cash_weight,
                "highest_target_weight": highest_weight,
                "one_way_turnover": decision["one_way_turnover"],
            },
        ),
    ):
        reviews[role] = {
            "lens": lens,
            "view_metrics": view_metrics,
            "checks": dict(common_checks),
            "status": "within_constraints" if all(common_checks.values()) else "constraint_needs_attention",
            "proposal_changed": False,
        }
    reviewed_values = {
        "target_weights": target,
        "risk": decision["annualized_risk"],
        "cash_after_costs_cny": decision["estimated_cash_after_costs_cny"],
        "cost_cny": decision["expected_transaction_cost_cny"],
    }
    proposal_sha256 = hashlib.sha256(
        json.dumps(reviewed_values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for review in reviews.values():
        review["proposal_sha256"] = proposal_sha256
    return reviews


def investment_manager_explanation(decision: dict[str, object], research_manager_view=None) -> dict[str, object]:
    """Tie buy/hold/avoid language directly to the numerical target and order list."""
    status_by_symbol = {}
    for symbol in sorted(set(decision["current_holdings_cny"]) | set(decision["target_holdings_cny"])):
        current = float(decision["current_holdings_cny"].get(symbol, 0.0))
        target = float(decision["target_holdings_cny"].get(symbol, 0.0))
        if target > current + 0.01:
            action = "buy"
        elif target <= 0.01 and current <= 0.01:
            action = "avoid"
        elif target < current - 0.01:
            action = "reduce"
        else:
            action = "hold"
        status_by_symbol[symbol] = {
            "action": action,
            "current_value_cny": current,
            "target_value_cny": target,
            "target_weight": float(decision["target_weights"].get(symbol, 0.0)),
        }
    candidates = {
        str(row["symbol"]): row
        for row in (research_manager_view or {}).get("candidate_priorities", [])
    }
    risk_value = decision["annualized_risk"]
    risk_summary = (
        f"{float(risk_value):.2%} 年化风险，预算 {float(decision['annualized_risk_budget']):.2%}"
        if isinstance(risk_value, (int, float))
        else "风险估计不可用"
    )
    hard_risk_outcome = (decision.get("evidence") or {}).get("hard_risk_outcome", {})
    risk_profile = (decision.get("evidence") or {}).get("risk_profile")
    actual_hold_state = hard_risk_outcome.get("actual_hold_state")
    risk_unresolved = isinstance(actual_hold_state, dict) and (
        actual_hold_state.get("actual_constraints_satisfied") is False
        or actual_hold_state.get("risk_within_budget") is False
    )
    summary = (
        f"{decision['as_of_date']} 决策：{risk_summary}；"
        f"预计交易成本 ¥{decision['expected_transaction_cost_cny']:,.2f}，"
        f"扣费后现金 ¥{decision['estimated_cash_after_costs_cny']:,.2f}。"
    )
    if isinstance(risk_profile, Mapping):
        constraints = risk_profile.get("constraints", {})
        template = {
            "conservative": "保守",
            "balanced": "平衡",
            "aggressive": "进取",
        }.get(risk_profile.get("template"), "系统默认")
        if isinstance(constraints, Mapping):
            summary += (
                f"风险档案 {template}（版本 {risk_profile.get('template_version') or risk_profile.get('schema_version')}）："
                f"单票上限 {float(constraints.get('maximum_symbol_weight', 0)):.2%}，"
                f"单向换手上限 {float(constraints.get('maximum_one_way_turnover', 0)):.2%}，"
                f"最低现金 {float(constraints.get('minimum_cash_weight', 0)):.2%}。"
            )
    summary += "年化波动率预算不保证最大回撤或最大亏损。"
    if decision["status"] == "risk_blocked":
        summary += "硬性风控未通过，所有新增订单已阻断。"
    if risk_unresolved:
        summary += "现有持仓风险仍超预算或无法确认，风险未解决。"
    return {
        "status": decision["status"],
        "prediction_horizon_sessions": decision["prediction_horizon_sessions"],
        "summary": summary,
        "buy_hold_reduce_avoid_by_symbol": status_by_symbol,
        "research_context_by_symbol": {
            symbol: {
                "calibrated_priority": candidates[symbol]["priority"],
                "qwen_review_evidence": candidates[symbol]["qwen_review_evidence"],
                "qwen_evidence_view": candidates[symbol]["qwen_evidence_view"],
            }
            for symbol in status_by_symbol
            if symbol in candidates
        },
        "orders": decision["orders"],
        "risk_review_status": hard_risk_outcome.get("status"),
        "risk_failed_checks": hard_risk_outcome.get("failed_checks", []),
        "risk_orders_blocked": bool(hard_risk_outcome.get("orders_blocked", False)),
        "risk_actual_hold_state": actual_hold_state,
        "risk_unresolved": risk_unresolved,
        "risk_profile": risk_profile,
        "risk_profile_sha256": (
            risk_profile.get("profile_sha256") if isinstance(risk_profile, Mapping) else None
        ),
        "risk_profile_execution_allowed": (
            risk_profile.get("execution_allowed") if isinstance(risk_profile, Mapping) else True
        ),
        "drawdown_guarantee": False,
        "source_of_numeric_plan": "existing calibrated-return optimizer; narrative agent scores do not create expected returns or order amounts",
    }


def fixed_p9_comparison(decision, p9_optimization):
    """Serialize a same-account, same-constraints fixed-P9 optimizer comparison."""
    if p9_optimization is None:
        return {
            "status": "unavailable_without_mature_p9_forecast_error",
            "method": "market_plus_relative_fusion",
        }
    return {
        "status": "computed_same_account_same_constraints",
        "method": "market_plus_relative_fusion",
        "current_nav_cny": decision.current_nav_cny,
        "current_cash_cny": decision.current_cash_cny,
        "current_holdings_cny": dict(decision.current_holdings_cny),
        "target_weights": p9_optimization.target_weights,
        "annualized_risk": p9_optimization.annualized_risk,
        "annualized_risk_budget": p9_optimization.risk_budget_annualized,
        "one_way_turnover": p9_optimization.maximum_one_way_turnover,
        "expected_transaction_cost_cny": (
            p9_optimization.expected_transaction_cost * decision.current_nav_cny
        ),
        "objective_value_cny": (
            p9_optimization.objective_value * decision.current_nav_cny
        ),
    }
