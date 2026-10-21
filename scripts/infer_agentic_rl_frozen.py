"""Restore a frozen Agentic RL artifact and infer one recorded historical decision."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from adaptive_mas.evaluation.agentic_rl import (
    LinearExpectedSarsa,
    RidgeResidualModel,
    state_vector,
)


def infer_one(
    artifact: dict[str, object],
    state_trace: list[dict[str, object]],
    decision_date: date,
    seed: int = 0,
) -> dict[str, object]:
    freeze_date = date.fromisoformat(str(artifact["freeze_date"]))
    if decision_date <= freeze_date:
        raise ValueError("Frozen inference date must be after the policy freeze date")
    matches = [row for row in state_trace if row.get("decision_date") == decision_date.isoformat()]
    if len(matches) != 1:
        raise ValueError("State trace must contain exactly one row for the requested date")
    row = matches[0]
    policy = LinearExpectedSarsa.from_dict(artifact["policy"])
    residual_model = RidgeResidualModel.from_dict(artifact["C_text_residual_model"])

    study_state = row["study_state_features"]
    allowed_study = tuple(str(action) for action in row["allowed_study_actions"])
    study_action, _, study_values = policy.choose(
        state_vector(study_state), allowed_study, epsilon=0.0, seed=seed,
    )
    queried = study_action == "study:query"
    if queried:
        fusion_state = row.get("query_fusion_state_features")
        raw_fusion_actions = row.get("query_allowed_fusion_actions")
        if fusion_state is None or raw_fusion_actions is None:
            raise ValueError("Query was selected but the historical trace has no query-result state")
    else:
        fusion_state = row["finish_fusion_state_features"]
        raw_fusion_actions = row["finish_allowed_fusion_actions"]
    allowed_fusion = tuple(str(action) for action in raw_fusion_actions)
    fusion_action, _, fusion_values = policy.choose(
        state_vector(fusion_state), allowed_fusion, epsilon=0.0, seed=seed,
    )

    residual_candidate = None
    residual_applied = fusion_action in {"fusion:research", "fusion:joint"}
    if queried and residual_applied:
        features = row.get("query_text_residual_features")
        if features is None:
            raise ValueError("Query was selected but the historical trace has no residual features")
        profile_features = [float(value) for value in features]
        if fusion_action == "fusion:research":
            profile_features[-1] = 0.0
        residual_candidate = residual_model.predict(profile_features)
    return {
        "decision_date": decision_date.isoformat(),
        "freeze_date": freeze_date.isoformat(),
        "policy_updates": policy.updates,
        "residual_model_fitted_rows": residual_model.fitted_rows,
        "study_action": study_action,
        "study_action_values": study_values,
        "fusion_action": fusion_action,
        "fusion_action_values": fusion_values,
        "query_event_id": row.get("query_event_id") if queried else None,
        "research_residual_candidate": residual_candidate,
        "research_residual_applied": residual_applied,
        "historical_inference_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, required=True,
        help="frozen_agentic_policy.json produced by the historical replay",
    )
    parser.add_argument(
        "--state-trace", type=Path, required=True,
        help="C_evaluation_state_trace.jsonl produced by the historical replay",
    )
    parser.add_argument("--decision-date", type=date.fromisoformat, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    state_trace = [
        json.loads(line)
        for line in args.state_trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = infer_one(artifact, state_trace, args.decision_date, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
