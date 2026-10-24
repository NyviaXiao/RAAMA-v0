"""Content-addressed provenance for reusable public research artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import date
from typing import Mapping


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()


def endpoint_sha256(endpoint: str) -> str:
    return hashlib.sha256(endpoint.rstrip("/").encode("utf-8")).hexdigest()


def research_artifact_sha256(research_result: Mapping[str, object]) -> str:
    immutable_payload = {
        key: value
        for key, value in research_result.items()
        if key not in {
            "package_provenance",
            "receiving_task_usage",
            "receiving_task_cost",
        }
    }
    return stable_hash(immutable_payload)


def _validate_source_usage_cost(research_result: Mapping[str, object]) -> None:
    usage = research_result.get("task_usage")
    cost = research_result.get("cost")
    if not isinstance(usage, Mapping) or not isinstance(cost, Mapping):
        raise ValueError("research source must include immutable task usage and cost summaries")
    integer_fields = (
        "provider_call_count",
        "attempt_count",
        "response_cache_reuse_count",
        "model_turn_count",
        "prompt_tokens",
        "completion_tokens",
    )
    if "unknown_outcome_call_count" in usage:
        integer_fields += ("unknown_outcome_call_count",)
    counts = {}
    for field in integer_fields:
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"research source usage {field} must be a nonnegative integer")
        counts[field] = value
    if (
        counts["provider_call_count"] > counts["attempt_count"]
        or counts["response_cache_reuse_count"] > counts["model_turn_count"]
        or counts["model_turn_count"]
        > counts["provider_call_count"] + counts["response_cache_reuse_count"]
        or counts.get("unknown_outcome_call_count", 0) > counts["provider_call_count"]
    ):
        raise ValueError("research source usage call counts are inconsistent")
    for field in ("logical_estimated_cost_cny", "api_incremental_cost_cny"):
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"research source usage {field} must be finite and nonnegative")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"research source usage {field} must be finite and nonnegative")
    if (
        float(usage["api_incremental_cost_cny"])
        > float(usage["logical_estimated_cost_cny"]) + 1e-12
    ):
        raise ValueError("research source API cost cannot exceed its logical estimated cost")
    if counts["model_turn_count"] == 0 and (
        counts["prompt_tokens"] != 0
        or counts["completion_tokens"] != 0
        or float(usage["logical_estimated_cost_cny"]) != 0.0
        or float(usage["api_incremental_cost_cny"]) != 0.0
    ):
        raise ValueError("research source usage tokens and costs require a model turn")
    cost_fields = {
        "call_count": counts["provider_call_count"],
        "logical_inference_count": counts["model_turn_count"],
        "logical_inference_estimated_cost_cny": float(usage["logical_estimated_cost_cny"]),
        "incremental_api_cost_cny": float(usage["api_incremental_cost_cny"]),
    }
    for field, expected in cost_fields.items():
        actual = cost.get(field)
        if isinstance(expected, float):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or float(actual) < 0
                or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError("research source task usage does not match its cost summary")
        elif isinstance(actual, bool) or actual != expected:
            raise ValueError("research source task usage does not match its cost summary")


def research_cost_summary(task_usage: Mapping[str, object]) -> dict[str, object]:
    return {
        "call_count": int(task_usage["provider_call_count"]),
        "logical_inference_count": int(task_usage["model_turn_count"]),
        "logical_inference_estimated_cost_cny": float(task_usage["logical_estimated_cost_cny"]),
        "incremental_api_cost_cny": float(task_usage["api_incremental_cost_cny"]),
    }


def market_snapshot_identity(research: object, as_of: date) -> dict[str, object]:
    def path_stable_row(row: Mapping[str, object]) -> dict[str, object]:
        normalized = dict(row)
        source_ref = normalized.get("source_ref")
        if isinstance(source_ref, str):
            normalized["source_ref"] = source_ref.replace("\\", "/").rsplit("/", 1)[-1]
        return normalized

    trading_dates = tuple(day for day in research.trading_dates if day <= as_of)
    adjusted_bars = tuple(
        path_stable_row(row) for row in research.adjusted_bars if row["available_at"] <= as_of
    )
    execution_bars = tuple(
        path_stable_row(row) for row in research.execution_bars if row["available_at"] <= as_of
    )
    index_bars = tuple(
        path_stable_row(row) for row in research.index_bars if row["available_at"] <= as_of
    )
    memberships = research.membership_snapshots
    members = research.membership_at(as_of) if memberships else frozenset()
    applicable_memberships = [
        {
            "event_time": item.event_time.isoformat(),
            "available_at": item.available_at.isoformat(),
            "members": sorted(item.members),
            "source_ref": path_stable_row({"source_ref": item.source_ref})["source_ref"],
        }
        for item in memberships
        if item.available_at <= as_of
    ]
    payload = {
        "decision_date": as_of.isoformat(),
        "trading_dates": [item.isoformat() for item in trading_dates],
        "members": sorted(members),
        "adjusted_bars": adjusted_bars,
        "execution_bars": execution_bars,
        "index_bars": index_bars,
        "membership_snapshots": applicable_memberships,
    }
    return {"sha256": stable_hash(payload), "row_counts": {
        "trading_dates": len(trading_dates),
        "members": len(members),
        "adjusted_bars": len(adjusted_bars),
        "execution_bars": len(execution_bars),
        "index_bars": len(index_bars),
    }}


def _source_package_identity(
    *,
    as_of: date,
    public_evidence_mode: str,
    horizon: int,
    config: Mapping[str, object],
    candidate_version_id: str,
    candidate_sha256: str,
    implementation_identity: Mapping[str, object],
    market_identity: Mapping[str, object],
    model: str,
    endpoint: str,
    source_snapshot_sha256: str,
    upstream_experiment: Mapping[str, object] | None,
    model_risk_context_sha256: str | None = None,
) -> dict[str, object]:
    endpoint_hash = endpoint_sha256(endpoint)
    return {
        "schema_version": "adaptive-mas.research-source-package-identity.v1",
        "as_of": as_of.isoformat(),
        "public_evidence_mode": public_evidence_mode,
        "prediction_horizon_sessions": horizon,
        "research_config": dict(config),
        "research_config_sha256": stable_hash(config),
        "candidate_version_id": candidate_version_id,
        "candidate_sha256": candidate_sha256,
        "implementation_identity": dict(implementation_identity),
        "market_snapshot_identity": dict(market_identity),
        "provider_identity": {
            "model": model,
            "endpoint_sha256": endpoint_hash,
        },
        "public_snapshot_sha256": source_snapshot_sha256,
        "model_risk_context_sha256": model_risk_context_sha256,
        "upstream_experiment": copy.deepcopy(dict(upstream_experiment)) if upstream_experiment else None,
    }


def build_research_package_provenance(
    research_result: Mapping[str, object],
    *,
    as_of: date,
    public_evidence_mode: str,
    horizon: int,
    config: Mapping[str, object],
    candidate_version_id: str,
    candidate_sha256: str,
    implementation_identity: Mapping[str, object],
    market_identity: Mapping[str, object],
    model: str,
    endpoint: str,
    source_snapshot: Mapping[str, object],
    upstream_experiment: Mapping[str, object] | None = None,
    model_risk_context_sha256: str | None = None,
) -> dict[str, object]:
    _validate_source_usage_cost(research_result)
    snapshot = copy.deepcopy(dict(source_snapshot))
    snapshot_sha256 = stable_hash(snapshot)
    source_package_identity = _source_package_identity(
        as_of=as_of,
        public_evidence_mode=public_evidence_mode,
        horizon=horizon,
        config=config,
        candidate_version_id=candidate_version_id,
        candidate_sha256=candidate_sha256,
        implementation_identity=implementation_identity,
        market_identity=market_identity,
        model=model,
        endpoint=endpoint,
        source_snapshot_sha256=snapshot_sha256,
        upstream_experiment=upstream_experiment,
        model_risk_context_sha256=model_risk_context_sha256,
    )
    source_package_identity_sha256 = stable_hash(source_package_identity)
    endpoint_hash = endpoint_sha256(endpoint)
    source_manifest = {
        "schema_version": "adaptive-mas.harness-source-manifest.v4",
        "source_package_identity": source_package_identity,
        "source_package_identity_sha256": source_package_identity_sha256,
        "provider": {
            "model": model,
            "endpoint_sha256": endpoint_hash,
        },
        "source_snapshot": snapshot,
        "source_snapshot_sha256": snapshot_sha256,
        "research_artifact_sha256": research_artifact_sha256(research_result),
    }
    return {
        "schema_version": "adaptive-mas.research-package-provenance.v4",
        "source_manifest": source_manifest,
        "source_manifest_sha256": stable_hash(source_manifest),
        "source_package_identity_sha256": source_package_identity_sha256,
        "implementation_package_sha256": implementation_identity["implementation_package_sha256"],
        "tool_schema_sha256": implementation_identity["tool_schema_sha256"],
        "evaluator_sha256": implementation_identity["evaluator_sha256"],
        "candidate_version_id": candidate_version_id,
        "candidate_sha256": candidate_sha256,
        "model": model,
        "endpoint_sha256": endpoint_hash,
        "public_snapshot_sha256": snapshot_sha256,
    }


def _validate_upstream_experiment(
    value: object,
    *,
    as_of: date,
    public_evidence_mode: str,
    source_snapshot_sha256: str,
    required_observations: object,
    config_sha256: str,
    implementation_identity: Mapping[str, object],
    model: str,
    endpoint_hash: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError("source upstream experiment identity is invalid")
    identity = value.get("identity")
    identity_sha256 = value.get("identity_sha256")
    if (
        not isinstance(identity, Mapping)
        or identity_sha256 != stable_hash(identity)
        or identity.get("schema_version") != "adaptive-mas.gepa-package-identity.v1"
    ):
        raise ValueError("source upstream experiment hash does not match its identity")
    if (
        identity.get("source_identity") != dict(implementation_identity)
        or identity.get("model") != model
        or identity.get("endpoint_sha256") != endpoint_hash
        or identity.get("config_sha256") != config_sha256
    ):
        raise ValueError("source upstream experiment does not match research source identity")
    case_rows = identity.get("case_rows")
    matching = [
        row for row in case_rows
        if isinstance(row, Mapping)
        and row.get("as_of") == as_of.isoformat()
        and row.get("split") == "reuse_demo"
    ] if isinstance(case_rows, list) else []
    if len(matching) != 1:
        raise ValueError("source upstream experiment does not contain one matching reuse case")
    reuse_case = matching[0]
    if (
        reuse_case.get("snapshot_sha256") != source_snapshot_sha256
        or reuse_case.get("public_evidence_mode") != public_evidence_mode
        or reuse_case.get("required_observations_sha256") != stable_hash(required_observations)
    ):
        raise ValueError("source upstream reuse case does not match this research artifact")


def validate_research_package_provenance(
    research_result: Mapping[str, object],
    provenance: Mapping[str, object],
    *,
    as_of: date,
    public_evidence_mode: str,
    horizon: int,
    candidate_version_id: str,
    candidate_sha256: str,
    implementation_identity: Mapping[str, object],
    market_identity: Mapping[str, object],
    client: object | None,
    config: Mapping[str, object] | None = None,
    model_risk_context_sha256: str | None = None,
) -> str:
    if provenance.get("schema_version") != "adaptive-mas.research-package-provenance.v4":
        raise ValueError("precomputed research provenance format is unsupported")
    source_manifest = provenance.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        raise ValueError("precomputed research must include a source manifest")
    source_manifest_sha256 = provenance.get("source_manifest_sha256")
    if source_manifest_sha256 != stable_hash(source_manifest):
        raise ValueError("source manifest hash does not match its content")
    if source_manifest.get("schema_version") != "adaptive-mas.harness-source-manifest.v4":
        raise ValueError("source manifest format is unsupported")
    source_identity = source_manifest.get("source_package_identity")
    identity_sha256 = source_manifest.get("source_package_identity_sha256")
    if (
        not isinstance(source_identity, Mapping)
        or identity_sha256 != stable_hash(source_identity)
        or provenance.get("source_package_identity_sha256") != identity_sha256
    ):
        raise ValueError("source package identity hash does not match manifest content")
    if (
        source_identity.get("as_of") != as_of.isoformat()
        or source_identity.get("public_evidence_mode") != public_evidence_mode
        or source_identity.get("prediction_horizon_sessions") != horizon
    ):
        raise ValueError("source manifest date, evidence mode, or forecast horizon does not match this task")
    source_config = source_identity.get("research_config")
    config_sha256 = source_identity.get("research_config_sha256")
    if not isinstance(source_config, Mapping) or config_sha256 != stable_hash(source_config):
        raise ValueError("source research configuration hash does not match its content")
    if (
        source_identity.get("candidate_version_id") != candidate_version_id
        or source_identity.get("candidate_sha256") != candidate_sha256
        or provenance.get("candidate_version_id") != candidate_version_id
        or provenance.get("candidate_sha256") != candidate_sha256
    ):
        raise ValueError("source manifest candidate does not match the currently loaded harness version")
    if source_identity.get("implementation_identity") != dict(implementation_identity):
        raise ValueError("source manifest implementation identity does not match the current package")
    if source_identity.get("market_snapshot_identity") != dict(market_identity):
        raise ValueError("source manifest market snapshot does not match the current as-of data")
    manager = research_result.get("agents", {}).get("research_manager", {}) if isinstance(research_result.get("agents"), Mapping) else {}
    manager_report = manager.get("report", {}) if isinstance(manager, Mapping) else {}
    directed_queries = manager.get("directed_queries", []) if isinstance(manager, Mapping) else []
    has_model_risk_action = (
        isinstance(manager_report, Mapping)
        and any(isinstance(item, Mapping) and item.get("dimension") == "model" for item in manager_report.get("risk_research", []))
    ) or (
        isinstance(directed_queries, list)
        and any(isinstance(item, Mapping) and item.get("risk_dimension") == "model" for item in directed_queries)
    )
    has_industry_risk_action = (
        isinstance(manager_report, Mapping)
        and any(isinstance(item, Mapping) and item.get("dimension") == "industry" for item in manager_report.get("risk_research", []))
    ) or (
        isinstance(directed_queries, list)
        and any(isinstance(item, Mapping) and item.get("risk_dimension") == "industry" for item in directed_queries)
    )
    if has_industry_risk_action and config is not None and any(
        config.get(key) != source_config.get(key)
        for key in ("risk_industry_sources", "risk_lookback_sessions")
    ):
        raise ValueError("precomputed industry-risk research does not match the current sourced industry configuration")
    if has_model_risk_action and (
        not model_risk_context_sha256
        or source_identity.get("model_risk_context_sha256") != model_risk_context_sha256
    ):
        raise ValueError("precomputed model-risk research does not match the current calibrated model context")
    source_snapshot = source_manifest.get("source_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise ValueError("source manifest does not contain the raw public source snapshot")
    source_snapshot_sha256 = stable_hash(source_snapshot)
    if (
        source_manifest.get("source_snapshot_sha256") != source_snapshot_sha256
        or provenance.get("public_snapshot_sha256") != source_snapshot_sha256
        or source_identity.get("public_snapshot_sha256") != source_snapshot_sha256
    ):
        raise ValueError("source snapshot hash does not match manifest content")
    if (
        source_snapshot.get("schema_version") != "public-expert-snapshot.v1"
        or source_snapshot.get("data_mode") != public_evidence_mode
        or source_snapshot.get("evaluation_horizon_sessions") != horizon
        or str(source_snapshot.get("requested_decision_time", ""))[:10] != as_of.isoformat()
    ):
        raise ValueError("source snapshot schema, date, evidence mode, or horizon is invalid")
    provider = source_manifest.get("provider")
    if not isinstance(provider, Mapping):
        raise ValueError("source manifest provider identity is missing")
    model = provider.get("model")
    endpoint_hash = provider.get("endpoint_sha256")
    if not isinstance(endpoint_hash, str) or len(endpoint_hash) != 64:
        endpoint_hash = ""
    try:
        bytes.fromhex(endpoint_hash)
    except ValueError:
        endpoint_hash = ""
    expected_provider = {"model": model, "endpoint_sha256": endpoint_hash}
    if (
        not isinstance(model, str)
        or not model.strip()
        or not endpoint_hash
        or source_identity.get("provider_identity") != expected_provider
        or provenance.get("model") != model
        or provenance.get("endpoint_sha256") != endpoint_hash
    ):
        raise ValueError("source manifest provider identity does not match package provenance")
    if client is not None and (
        client.model != model or endpoint_sha256(client.endpoint) != endpoint_hash
    ):
        raise ValueError("source manifest provider identity does not match the configured model")
    required_observations = research_result.get("required_observations", [])
    _validate_upstream_experiment(
        source_identity.get("upstream_experiment"),
        as_of=as_of,
        public_evidence_mode=public_evidence_mode,
        source_snapshot_sha256=source_snapshot_sha256,
        required_observations=required_observations,
        config_sha256=str(config_sha256),
        implementation_identity=implementation_identity,
        model=str(model),
        endpoint_hash=endpoint_hash,
    )
    _validate_source_usage_cost(research_result)
    if source_manifest.get("research_artifact_sha256") != research_artifact_sha256(research_result):
        raise ValueError("research artifact content does not match its source manifest")
    if (
        provenance.get("implementation_package_sha256")
        != implementation_identity["implementation_package_sha256"]
        or provenance.get("tool_schema_sha256") != implementation_identity["tool_schema_sha256"]
        or provenance.get("evaluator_sha256") != implementation_identity["evaluator_sha256"]
    ):
        raise ValueError("precomputed research package identity does not match the current implementation")
    return str(source_manifest_sha256)
