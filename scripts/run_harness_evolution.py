"""Prepare a GEPA request review bundle or run the approved retrospective evolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from adaptive_mas.agents.harness import (
    DEFAULT_MODEL,
    INPUT_PRICE_CNY_PER_MILLION,
    MAX_CALLS_PER_EXPERIMENT,
    MAX_INPUT_BYTES_PER_CALL,
    MAX_OUTPUT_TOKENS_PER_CALL,
    OUTPUT_PRICE_CNY_PER_MILLION,
    CallBudget,
    QwenClient,
)
from adaptive_mas.data import load_oos_research_data
from adaptive_mas.harness_evolution import (
    GEPA_METRIC_CALL_LIMIT,
    GEPA_PROPOSAL_LIMIT,
    GEPA_REFLECTION_REQUESTS_PER_PROPOSAL,
    HarnessVersionStore,
    build_retrospective_cases,
    run_gepa_evolution,
    split_retrospective_cases,
    summarize_retrospective_cases,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
OBJECTIVE = (
    "Improve public research factual accuracy, citation precision, report-period correctness, "
    "point-in-time compliance, research-tool efficiency, and measured model cost."
)
BACKGROUND = (
    "Evolve only the supplied market, fundamental, manager, context, handoff and procedural-experience "
    "text components. Tool schemas, evaluator weights, data split, expected-return calibration and portfolio "
    "optimizer are fixed. All cases are retrospective cached data; never put accounts, private forecasts, "
    "trades, internal source paths or portfolio results in reflection feedback."
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _preview_feedback(as_of: str, required_facts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "as_of": as_of,
        "quality_metrics": {
            "score": 0.0,
            "fact_accuracy": 0.0,
            "required_fact_coverage": 0.0,
            "citation_precision": 0.0,
            "report_period_accuracy": 0.0,
            "point_in_time_accuracy": 0.0,
            "tool_efficiency": 0.0,
            "cost_efficiency": 0.0,
            "verified_observation_count": 0,
            "observation_count": 0,
            "unique_citation_count": 0,
            "duplicate_citation_count": 0,
            "valid_citation_count": 0,
            "citation_count": 0,
            "research_tool_call_count": 0,
            "estimated_model_cost_cny": 0.0,
        },
        "required_fact_checks": [
            {**fact, "status": "preview_only", "expected_value": "<public cached value>"}
            for fact in required_facts[:2]
        ],
        "incorrect_observations": [],
        "invalid_citation_examples": [],
        "public_report_summaries": {
            "market": {"summary": "<bounded public report summary>", "findings": [], "unresolved_count": 0},
            "fundamental": {"summary": "<bounded public report summary>", "findings": [], "unresolved_count": 0},
            "research_manager": {"summary": "<bounded public report summary>", "findings": [], "unresolved_count": 0},
        },
        "public_tool_summaries": {},
        "task_model_usage": {
            "provider_call_count": 0,
            "response_cache_reuse_count": 0,
            "model_turn_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "logical_estimated_cost_cny": 0.0,
            "api_incremental_cost_cny": 0.0,
        },
        "scope": "retrospective public cached research; no account, private forecast, trade, return or raw source ID",
    }


def _build_review(candidate: dict[str, str], cases: dict[str, list], model: str) -> dict[str, object]:
    from gepa.optimize_anything import _build_reflection_prompt_template
    from gepa.strategies.instruction_proposal import InstructionProposalSignature

    prompt_template = _build_reflection_prompt_template(OBJECTIVE, BACKGROUND)
    sample_case = cases["exploration"][0]
    feedback = _preview_feedback(sample_case.as_of.isoformat(), sample_case.required_observations)
    reflection_payloads = []
    for component, instruction in sorted(candidate.items()):
        prompt = InstructionProposalSignature.prompt_renderer({
            "current_instruction_doc": instruction,
            "dataset_with_feedback": [feedback],
            "prompt_template": prompt_template,
        })
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": MAX_OUTPUT_TOKENS_PER_CALL,
            "enable_thinking": False,
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(serialized) > MAX_INPUT_BYTES_PER_CALL:
            raise ValueError(f"reflection payload for {component} exceeds the {MAX_INPUT_BYTES_PER_CALL}-byte limit")
        reflection_payloads.append({
            "component": component,
            "serialized_bytes": len(serialized),
            "request_payload": payload,
        })

    unique_exploration_pairs = len(cases["exploration"]) * (1 + GEPA_PROPOSAL_LIMIT)
    external_research_tasks = (
        2 * len(cases["selection"])
        + 2 * len(cases["frozen_evaluation"])
        + len(cases["reuse_demo"])
    )
    research_task_count = unique_exploration_pairs + external_research_tasks
    research_turns = research_task_count * 10
    reflection_turns = GEPA_REFLECTION_REQUESTS_PER_PROPOSAL * GEPA_PROPOSAL_LIMIT
    total_calls = research_turns + reflection_turns
    per_call_cny = (
        MAX_INPUT_BYTES_PER_CALL * INPUT_PRICE_CNY_PER_MILLION
        + MAX_OUTPUT_TOKENS_PER_CALL * OUTPUT_PRICE_CNY_PER_MILLION
    ) / 1_000_000
    review = {
        "schema_version": "adaptive-mas.gepa-outgoing-review.v1",
        "review_content_sha256": "computed_after_serialization",
        "mode": "preparation_only_no_provider_requests",
        "public_evidence_mode": "retrospective_source_publication_proxy",
        "strict_historical_pit_verified": False,
        "model": model,
        "outgoing_request_fields": {
            "research_agents": ["model", "messages", "tools", "tool_choice", "temperature", "max_tokens", "enable_thinking"],
            "gepa_reflection": ["model", "messages", "temperature", "max_tokens", "enable_thinking"],
            "authentication": "Authorization header only; credential is not part of prompts or request-review files",
        },
        "public_context_sent": [
            "decision date and time",
            "public market tool results from the local BaoStock cache",
            "financial facts and announcement titles whose source publication/availability dates precede as_of; later fetched_at is preserved and marked unverified",
            "this is a retrospective source-date proxy, not strict historical point-in-time capture",
            "the research manager's public specialist handoffs and evidence identifiers",
            "fixed public evaluation diagnostics and measured model token usage",
        ],
        "excluded_from_model_requests": [
            "portfolio cash and holdings",
            "private forecasts and internal return estimates",
            "optimizer state, target weights and trade instructions",
            "test.csv and all test-set rows",
            "local paths and raw source_record_ids",
        ],
        "retrospective_split": {
            name: [case.as_of.isoformat() for case in cases[name]]
            for name in ("exploration", "selection", "frozen_evaluation", "reuse_demo")
        },
        "retrospective_case_summary": summarize_retrospective_cases([
            case
            for name in ("exploration", "selection", "frozen_evaluation", "reuse_demo")
            for case in cases[name]
        ]),
        "evolution_limits": {
            "unique_exploration_candidate_as_of_pairs": unique_exploration_pairs,
            "gepa_metric_call_limit": GEPA_METRIC_CALL_LIMIT,
            "gepa_metric_call_limit_scope": "evaluator-callback cap; evaluation cache limits exploration to unique (candidate, as_of) pairs",
            "external_paired_research_tasks": external_research_tasks,
            "research_task_count_upper_bound": research_task_count,
            "exploration_task_count_upper_bound": unique_exploration_pairs,
            "research_turns": research_turns,
            "reflection_turns": reflection_turns,
            "provider_call_upper_bound": total_calls,
            "hard_provider_call_ceiling": MAX_CALLS_PER_EXPERIMENT,
            "model_turn_limit_per_case": 10,
            "candidate_mutations": 1,
            "reflection_component_selector": "all",
            "reflection_component_pool": sorted(candidate),
            "reflected_text_components": sorted(candidate),
            "max_input_bytes_per_request": MAX_INPUT_BYTES_PER_CALL,
            "max_output_tokens_per_request": MAX_OUTPUT_TOKENS_PER_CALL,
        },
        "price_estimate": {
            "planning_input_cny_per_million_tokens": INPUT_PRICE_CNY_PER_MILLION,
            "planning_output_cny_per_million_tokens": OUTPUT_PRICE_CNY_PER_MILLION,
            "public_beijing_list_price_usd_per_million": {"input": 1.65, "output": 4.951},
            "source": "https://www.alibabacloud.com/help/zh/model-studio/qwen3-7-max",
            "per_call_upper_bound_cny": per_call_cny,
            "run_upper_bound_cny": total_calls * per_call_cny,
            "hard_160_call_upper_bound_cny": MAX_CALLS_PER_EXPERIMENT * per_call_cny,
            "basis": "Conservative planning calculation treats every permitted UTF-8 input byte as a token and applies no cache discount.",
        },
        "gepa_reflection_payload_previews": {
            "feedback_is_synthetic_shape_preview": True,
            "payloads": reflection_payloads,
        },
        "data_policy": {
            "test_csv_read": False,
            "new_data_downloads": False,
            "announcement_network_fetch": False,
            "retrospective_only": True,
        },
    }
    serialized_review = _json(review).encode("utf-8")
    review["review_content_sha256"] = hashlib.sha256(serialized_review).hexdigest()
    return review


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument("--public-cache-dir", type=Path, default=project_dir / "data" / "raw" / "akshare")
    parser.add_argument("--config", type=Path, default=project_dir / "configs" / "daily_three_session.toml")
    parser.add_argument("--horizon-sessions", type=int, required=True)
    parser.add_argument(
        "--version-dir", type=Path,
        default=project_dir / "data" / "processed" / "harness_versions",
    )
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=project_dir / "experiments" / "gepa_harness",
    )
    parser.add_argument("--execute-gepa", action="store_true", help="Run the reviewed retrospective GEPA experiment.")
    parser.add_argument("--approved-review-sha256", help="Optional review hash recorded for this run.")
    parser.add_argument("--resume-run-id", help="Resume an existing GEPA checkpoint without creating a new run.")
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    research = load_oos_research_data(args.baostock_run_dir)
    cases = split_retrospective_cases(build_retrospective_cases(
        research,
        args.public_cache_dir,
        horizon_sessions=args.horizon_sessions,
        config=config,
    ))
    version_store = HarnessVersionStore(args.version_dir)
    candidate, active_version_id, _ = version_store.load_active()
    model = os.environ.get("POPAGENT_LLM_MODEL", DEFAULT_MODEL)
    review = _build_review(candidate, cases, model)
    _write_json(args.artifact_dir / "outgoing_review.json", review)

    case_summary = review["retrospective_case_summary"]
    if not case_summary["eligible_for_model_execution"]:
        print(json.dumps({
            "status": "review_prepared_blocked_no_point_in_time_evidence",
            "review_file": str(args.artifact_dir / "outgoing_review.json"),
            "review_sha256": review["review_content_sha256"],
            "reason": case_summary["eligibility_reason"],
            "provider_calls": 0,
            "provider_requests_sent": False,
        }, ensure_ascii=False, indent=2))
        return

    if not args.execute_gepa:
        print(json.dumps({
            "status": "review_prepared_no_provider_requests",
            "review_file": str(args.artifact_dir / "outgoing_review.json"),
            "review_sha256": review["review_content_sha256"],
            "active_version_id": active_version_id,
            "provider_call_upper_bound": review["evolution_limits"]["provider_call_upper_bound"],
            "cost_upper_bound_cny": review["price_estimate"]["run_upper_bound_cny"],
        }, ensure_ascii=False, indent=2))
        return

    if args.approved_review_sha256 and args.approved_review_sha256 != review["review_content_sha256"]:
        raise ValueError("--approved-review-sha256 does not match the current outgoing review file")
    if args.resume_run_id:
        if Path(args.resume_run_id).name != args.resume_run_id:
            raise ValueError("--resume-run-id must be a plain run-directory name")
        run_id = args.resume_run_id
        checkpoint = args.version_dir / "runs" / run_id / "gepa_state.bin"
        if not checkpoint.is_file():
            raise ValueError(f"GEPA checkpoint does not exist for run {run_id}")
    else:
        run_id = datetime.now(SHANGHAI).strftime("%Y%m%d%H%M%S")
    run_artifact_dir = args.artifact_dir / "runs" / run_id
    budget = CallBudget(
        max_calls=MAX_CALLS_PER_EXPERIMENT,
        ledger_path=args.artifact_dir / "calls.jsonl",
        response_cache_dir=args.artifact_dir / "response_cache",
    )
    result = run_gepa_evolution(
        cases,
        research=research,
        client=QwenClient.from_environment(),
        budget=budget,
        version_store=version_store,
        run_id=run_id,
        config=config,
    )
    result["outgoing_review_sha256"] = review["review_content_sha256"]
    result["outgoing_review_file"] = str(args.artifact_dir / "outgoing_review.json")
    _write_json(run_artifact_dir / "evolution_result.json", result)
    print(json.dumps({
        "status": result["status"],
        "run_id": run_id,
        "review_sha256": review["review_content_sha256"],
        "provider_calls": result["cost"]["call_count"],
        "provider_call_upper_bound": result["planned_provider_call_upper_bound"],
        "incremental_api_cost_cny": result["cost"]["incremental_api_cost_cny"],
        "active_version_after": result["active_version_after"],
        "result_dir": str(run_artifact_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
