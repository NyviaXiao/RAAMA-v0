"""Run one daily decision with the persistent GEPA harness and calibrated optimizer."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tomllib
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from adaptive_mas.data import load_oos_research_data
from adaptive_mas.decision import PortfolioState
from adaptive_mas.harness_runtime import create_harness_daily_decision
from adaptive_mas.risk_profile import resolve_risk_profile, validate_canonical_risk_profile
from adaptive_mas.runtime_binding import validate_workflow_runtime_binding


SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_TRAINING_SEED_TRAINED_THROUGH = date(2026, 3, 6)
DEFAULT_TRAINING_SEED_HORIZON_SESSIONS = 3


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument("--portfolio-state-json", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--horizon-sessions", type=int, required=True)
    parser.add_argument("--risk-budget-annualized", type=float)
    parser.add_argument("--risk-template", choices=("conservative", "balanced", "aggressive"))
    parser.add_argument("--risk-expression")
    parser.add_argument("--risk-constraints-json", help="JSON object containing explicit fractional risk limits")
    parser.add_argument("--risk-profile-json", type=Path, help="Validated canonical profile passed by the local task worker")
    parser.add_argument("--config", type=Path, default=project_dir / "configs" / "daily_three_session.toml")
    default_training_seed = (
        project_dir / "data" / "processed" / "p8_daily_three_session_continuous_v1" / "training_seed.json.gz"
    )
    parser.add_argument("--training-seed", type=Path, default=default_training_seed)
    parser.add_argument(
        "--training-seed-trained-through",
        type=date.fromisoformat,
        help=(
            "Required for custom seed files because they contain no cutoff metadata. "
            "The repository's default seed is tied to the documented 2026-03-06 training cutoff."
        ),
    )
    parser.add_argument(
        "--training-seed-horizon-sessions",
        type=int,
        help=(
            "Required for custom seed files because they contain no horizon metadata. "
            "The repository's default seed is tied to the documented three-session horizon."
        ),
    )
    parser.add_argument("--public-cache-dir", type=Path, default=project_dir / "data" / "raw" / "akshare")
    parser.add_argument(
        "--harness-version-dir",
        type=Path,
        default=project_dir / "data" / "processed" / "harness_versions",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Shared experiment root; defaults to experiments/gepa_harness and shares the 160-call ledger/cache with GEPA evolution.",
    )
    parser.add_argument(
        "--task-id",
        help="Stable task and idempotency key for resuming the same exact request after interruption.",
    )
    parser.add_argument("--announcement-text-limit", type=int, default=0)
    parser.add_argument("--announcement-document-dir", type=Path)
    parser.add_argument(
        "--retrospective-public-cache",
        action="store_true",
        help="Use source publication dates as a retrospective proxy for backfilled finance/announcements; never treat it as strict historical PIT.",
    )
    parser.add_argument(
        "--reuse-research-json",
        type=Path,
        help="Use a completed research_result_file from the shared GEPA evolution run without making another provider request.",
    )
    parser.add_argument("--workflow-binding-json", type=Path)
    parser.add_argument("--pinned-runtime-package-id")
    parser.add_argument("--pinned-runtime-package-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.horizon_sessions < 1:
        raise ValueError("prediction horizon must be a positive integer")
    if args.announcement_text_limit and args.announcement_document_dir is None:
        raise ValueError("reading cached announcement text requires --announcement-document-dir")
    workflow_binding_sha256 = None
    if args.workflow_binding_json is not None:
        workflow_binding = json.loads(args.workflow_binding_json.read_text(encoding="utf-8"))
        if not isinstance(workflow_binding, dict):
            raise ValueError("workflow binding must be a JSON object")
        if args.pinned_runtime_package_id is None:
            raise ValueError("workflow binding requires a pinned runtime package ID")
        pinned_package = workflow_binding.get("runtime_package")
        if not isinstance(pinned_package, dict):
            raise ValueError("workflow binding has no complete runtime package reference")
        if (
            pinned_package.get("package_id") != args.pinned_runtime_package_id
            or pinned_package.get("package_sha256") != args.pinned_runtime_package_sha256
        ):
            raise ValueError("workflow runtime package arguments differ from the queued snapshot")
        workflow_binding_sha256 = validate_workflow_runtime_binding(
            workflow_binding,
            project_root=project_dir,
            request_paths={
                "baostock_run_dir": args.baostock_run_dir,
                "config": args.config,
                "public_cache_dir": args.public_cache_dir,
                "reuse_research_json": args.reuse_research_json,
                "training_seed": args.training_seed,
            },
        )

    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    config["prediction_horizon_sessions"] = args.horizon_sessions
    if args.risk_profile_json is not None:
        if args.risk_template or args.risk_expression or args.risk_constraints_json or args.risk_budget_annualized is not None:
            raise ValueError("--risk-profile-json cannot be combined with other risk profile inputs")
        risk_profile = validate_canonical_risk_profile(
            json.loads(args.risk_profile_json.read_text(encoding="utf-8"))
        )
    else:
        risk_constraints = json.loads(args.risk_constraints_json) if args.risk_constraints_json else None
        if risk_constraints is not None and not isinstance(risk_constraints, dict):
            raise ValueError("--risk-constraints-json must contain a JSON object")
        risk_profile = resolve_risk_profile(
            risk_template=args.risk_template,
            risk_expression=args.risk_expression,
            risk_constraints=risk_constraints,
            risk_budget_annualized=args.risk_budget_annualized,
            base_defaults={
                "risk_budget_annualized": config.get("risk_budget_annualized", 0.20),
                "maximum_symbol_weight": config.get("maximum_symbol_weight", 0.05),
                "maximum_one_way_turnover": config.get("maximum_one_way_turnover", 0.50),
                "minimum_cash_weight": config.get("minimum_cash_weight", 0.0),
            },
        )
        if not risk_profile["execution_allowed"]:
            raise ValueError("risk profile needs clarification: " + " ".join(risk_profile["unresolved"]))
    account = json.loads(args.portfolio_state_json.read_text(encoding="utf-8"))
    if set(account) != {"cash_cny", "holdings_cny"} or not isinstance(account["holdings_cny"], dict):
        raise ValueError("portfolio JSON must contain only cash_cny and holdings_cny")
    portfolio = PortfolioState(
        cash_cny=float(account["cash_cny"]),
        holdings_cny={str(symbol): float(value) for symbol, value in account["holdings_cny"].items()},
    )
    portfolio.validate()

    with gzip.open(args.training_seed, "rt", encoding="utf-8") as seed_file:
        training_state = json.load(seed_file)
    training_seed_trained_through = args.training_seed_trained_through
    training_seed_cutoff_source = "cli_argument"
    training_seed_horizon_sessions = args.training_seed_horizon_sessions
    training_seed_horizon_source = "cli_argument"
    if training_seed_trained_through is None:
        if args.training_seed.resolve() != default_training_seed.resolve():
            raise ValueError(
                "custom training seeds have no embedded cutoff; provide --training-seed-trained-through from their provenance"
            )
        training_seed_trained_through = DEFAULT_TRAINING_SEED_TRAINED_THROUGH
        training_seed_cutoff_source = "repository_default_seed_protocol"
    if training_seed_horizon_sessions is None:
        if args.training_seed.resolve() != default_training_seed.resolve():
            raise ValueError(
                "custom training seeds have no embedded horizon; provide --training-seed-horizon-sessions from their provenance"
            )
        training_seed_horizon_sessions = DEFAULT_TRAINING_SEED_HORIZON_SESSIONS
        training_seed_horizon_source = "repository_default_seed_protocol"
    if training_seed_horizon_sessions != args.horizon_sessions:
        raise ValueError(
            "--horizon-sessions must match --training-seed-horizon-sessions before running a decision"
        )
    research = load_oos_research_data(args.baostock_run_dir)
    run_root = args.run_dir or project_dir / "experiments" / "gepa_harness"
    run_id = args.task_id or datetime.now(SHANGHAI).strftime("%Y%m%dT%H%M%S")
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in run_id):
        raise ValueError("--task-id may contain only letters, digits, dot, underscore and hyphen")
    run_dir = run_root / "daily" / args.as_of.isoformat() / run_id
    manifest_path = args.baostock_run_dir / "manifest.jsonl"
    decision = create_harness_daily_decision(
        research,
        config,
        args.as_of,
        portfolio,
        training_state,
        training_seed_trained_through=training_seed_trained_through,
        training_seed_horizon_sessions=training_seed_horizon_sessions,
        training_seed_cutoff_source=training_seed_cutoff_source,
        training_seed_horizon_source=training_seed_horizon_source,
        retrospective_public_cache=args.retrospective_public_cache,
        public_cache_dir=args.public_cache_dir,
        harness_version_dir=args.harness_version_dir,
        run_dir=run_dir,
        call_ledger_path=run_root / "calls.jsonl",
        response_cache_dir=run_root / "response_cache",
        versions={
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "baostock_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "training_seed_sha256": hashlib.sha256(args.training_seed.read_bytes()).hexdigest(),
            "training_seed_trained_through": training_seed_trained_through.isoformat(),
            "training_seed_horizon_sessions": str(training_seed_horizon_sessions),
        },
        announcement_text_limit=args.announcement_text_limit,
        announcement_document_dir=args.announcement_document_dir,
        precomputed_research_result=(
            json.loads(args.reuse_research_json.read_text(encoding="utf-8"))
            if args.reuse_research_json else None
        ),
        state_store_path=run_root / "harness_state.sqlite3",
        task_id=run_id,
        idempotency_key=run_id,
        risk_profile=risk_profile,
        workflow_binding_sha256=workflow_binding_sha256,
        pinned_runtime_package_id=args.pinned_runtime_package_id,
        pinned_runtime_package_sha256=args.pinned_runtime_package_sha256,
    )
    payload = json.dumps(decision.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".pending")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
