import hashlib
import json
from datetime import date, timedelta

import pytest

from adaptive_mas.agents.harness_prompts import seed_candidate
from adaptive_mas.harness_evolution import HarnessVersionStore
from adaptive_mas.package_identity import source_package_identity
from adaptive_mas.release_governance import HarnessReleaseStore


def _package():
    return {
        "candidate_version_id": "candidate-17",
        "candidate": seed_candidate(),
        "config": {"prediction_horizon_sessions": 3, "risk_budget_annualized": 0.2},
        "runtime_context": {
            "training_state": {"mature_rows": [1, 2, 3]},
            "training_seed_provenance": {
                "trained_through": "2024-12-31",
                "cutoff_source": "fixture provenance",
                "horizon_sessions": 3,
                "horizon_source": "fixture provenance",
            },
            "versions": {"training_seed_sha256": "a" * 64, "config_sha256": "b" * 64},
        },
        "artifacts": {
            "training_seed": {"version": "seed-v1", "payload": {"weights": [0.2, 0.3, 0.5]}},
            "training_state": {"version": "state-v1", "payload": {"mature_rows": [1, 2, 3]}},
            "calibration": {"version": "cal-v1", "payload": {"agent_errors": {"a": 0.1}}},
            "model_versions": {"version": "models-v1", "payload": {"forecaster": "ridge-v2"}},
            "data_versions": {"version": "data-v1", "payload": {"market_manifest": "c" * 64}},
            "tool_schema": {"version": "tools-v1", "payload": {"tool_schema_sha256": "d" * 64}},
        },
    }


def _protocol():
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(40)]
    return {
        "schema_version": "adaptive-mas.release-evaluation-protocol.v1",
        "trading_dates": [day.isoformat() for day in days],
        "horizon_sessions": 3,
        "embargo_sessions": 3,
        "minimum_event_groups": 3,
        "partitions": {
            "development": {"start": days[0].isoformat(), "end": days[4].isoformat()},
            "selection": {"start": days[9].isoformat(), "end": days[13].isoformat()},
            "independent_acceptance": {"start": days[18].isoformat(), "end": days[22].isoformat()},
            "forward_shadow": {"start": days[27].isoformat(), "end": days[31].isoformat()},
        },
        "economic_gate": {
            "maximum_noninferiority_loss": 0.0,
            "maximum_drawdown_worsening": 0.0,
            "annualized_risk_budget": 0.2,
            "minimum_cost_savings_cny": 0.0,
            "one_sided_confidence": 0.95,
        },
    }


def _write_input(root, *, package_id, partition, name, bad_economics=False, fixture=True):
    source_path = root / f"{name}.source"
    source_path.write_text(f"source artifact for {name}", encoding="utf-8")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(40)]
    starts = {"development": 0, "selection": 9, "independent_acceptance": 18, "forward_shadow": 27}
    rows = []
    for offset in range(3):
        position = starts[partition] + offset + 1
        row = {
            "event_group_id": f"{partition}-event-{offset}",
            "prediction_date": days[position].isoformat(),
        }
        if partition != "forward_shadow":
            row["label_matured_date"] = days[position + 1].isoformat()
            row.update({
                "candidate_research_score": 0.8,
                "baseline_research_score": 0.7,
                "candidate_fact_accuracy": 0.95,
                "baseline_fact_accuracy": 0.9,
            })
        if partition == "independent_acceptance":
            row.update({
                "candidate_net_return": -0.1 if bad_economics and offset == 0 else 0.001,
                "baseline_net_return": 0.001,
                "candidate_total_cost_cny": 90.0,
                "baseline_total_cost_cny": 100.0,
                "candidate_annualized_risk": 0.15,
                "baseline_maximum_drawdown": -0.10,
                "candidate_maximum_drawdown": -0.10,
            })
        if partition == "forward_shadow":
            row.update({
                "real_orders": 0,
                "public_deployment": False,
                "decision_sha256": hashlib.sha256(f"decision-{offset}".encode()).hexdigest(),
            })
        rows.append(row)
    payload = {
        "schema_version": "adaptive-mas.release-evaluation-input.v1",
        "package_id": package_id,
        "source_manifest": {
            "fixture_only": fixture,
            "data_mode": "forward_observed" if partition == "forward_shadow" else "captured_public_cache_cutoff",
            "source_artifacts": [{"path": source_path.name, "sha256": source_hash}],
        },
        "records": rows,
    }
    input_path = root / f"{name}.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return input_path


def _registry(tmp_path, *, test_mode=True):
    store = HarnessReleaseStore(tmp_path / "release_store", project_root=tmp_path, test_mode=test_mode)
    protocol = _protocol()
    store.predeclare_protocol("protocol-fixture-1", protocol)
    package = store.register_package(_package())
    evidence = {}
    for partition in ("development", "selection", "independent_acceptance", "forward_shadow"):
        path = _write_input(
            tmp_path,
            package_id=package["package_id"],
            partition=partition,
            name=partition,
            fixture=True,
        )
        evidence[partition] = store.evaluate_partition(
            protocol_id="protocol-fixture-1",
            partition=partition,
            run_id=f"{partition}-run-1",
            package_id=package["package_id"],
            input_artifact_path=path,
        )
    return store, package, evidence


def test_research_pointer_isolated_and_production_stays_seed_without_verified_release(tmp_path):
    store, package, evidence = _registry(tmp_path)
    active, version_id, metadata = store.load_active(config=_package()["config"])
    assert active == seed_candidate()
    assert version_id == "seed"
    assert metadata["status"] == "no_accepted_production_bundle"

    research_candidate = seed_candidate()
    research_candidate["procedural_experience"] += " Research-only candidate."
    research_versions = HarnessVersionStore(tmp_path / "release_store")
    research_versions.adopt(
        research_candidate,
        previous_version_id="seed",
        selection={"fixture_only": True},
        reason="A local research selection is not a release.",
        run_id="research-only-run",
        version_id="research-only-candidate",
    )
    active_after, active_id_after, _ = store.load_active(config=_package()["config"])
    assert active_after == seed_candidate()
    assert active_id_after == "seed"
    assert research_versions.load_active()[1] == "research-only-candidate"

    production_store = HarnessReleaseStore(tmp_path / "release_store", project_root=tmp_path)
    with pytest.raises(ValueError, match="fixture evidence can never qualify for a production release"):
        production_store.publish_release(
            package_id=package["package_id"],
            evidence_ids={name: report["evidence_id"] for name, report in evidence.items()},
            actor_role="release_manager",
            reason="fixture-only evidence cannot publish",
        )
    assert store.status_summary()["status"] == "seed"


def test_registry_computes_acceptance_metrics_and_fixture_bundle_roundtrips_completely(tmp_path):
    store, package, evidence = _registry(tmp_path)
    acceptance = evidence["independent_acceptance"]
    assert acceptance["evidence_status"] == "fixture_only"
    assert acceptance["economic_gate"]["passed"] is True
    assert acceptance["economic_gate"]["mean_cost_savings_cny"] == 10.0
    assert acceptance["economic_gate"]["one_sided_95_percent_lower_bound"] == 0.0

    published = store.publish_release(
        package_id=package["package_id"],
        evidence_ids={name: report["evidence_id"] for name, report in evidence.items()},
        actor_role="release_manager",
        reason="Test-only verification of complete bundle serialization and restoration.",
        release_id="test-release-1",
    )
    assert published["test_only"] is True
    assert published["evidence_status"]["economic_value"] == "test_only_gate_calculated"
    restored, version_id, metadata = store.load_active(
        config=_package()["config"],
        training_state=_package()["runtime_context"]["training_state"],
        training_seed_provenance=_package()["runtime_context"]["training_seed_provenance"],
        versions=_package()["runtime_context"]["versions"],
    )
    restored_package = metadata["restored_package"]
    assert restored == seed_candidate()
    assert version_id == "candidate-17"
    assert restored_package["artifacts"] == published["complete_runtime_package"]["artifacts"]
    assert restored_package["config"] == _package()["config"]
    assert restored_package["runtime_context"] == _package()["runtime_context"]

    production_view = HarnessReleaseStore(tmp_path / "release_store", project_root=tmp_path)
    with pytest.raises(ValueError, match="test-only evidence bundles"):
        production_view.load_active(config=_package()["config"])


def test_role_boundaries_and_release_accepts_registry_ids_only(tmp_path):
    store, package, evidence = _registry(tmp_path)
    evidence_ids = {name: report["evidence_id"] for name, report in evidence.items()}
    with pytest.raises(PermissionError, match="release_manager"):
        store.publish_release(
            package_id=package["package_id"], evidence_ids=evidence_ids,
            actor_role="researcher", reason="unauthorized",
        )
    with pytest.raises(ValueError, match="four registry evidence IDs"):
        store.publish_release(
            package_id=package["package_id"], evidence_ids={"development": {"status": "passed"}},
            actor_role="release_manager", reason="self-reported evidence is not accepted",
        )


def test_protocol_purge_embargo_and_acceptance_view_are_enforced(tmp_path):
    store = HarnessReleaseStore(tmp_path, project_root=tmp_path, test_mode=True)
    protocol = _protocol()
    protocol["partitions"]["selection"]["start"] = "2025-01-06"
    with pytest.raises(ValueError, match="embargo"):
        store.predeclare_protocol("bad-protocol", protocol)

    protocol = _protocol()
    store.predeclare_protocol("protocol-fixture-1", protocol)
    package = store.register_package(_package())
    bad_purge_input = _write_input(
        tmp_path, package_id=package["package_id"], partition="development", name="bad-purge"
    )
    raw = json.loads(bad_purge_input.read_text(encoding="utf-8"))
    raw["records"][0]["label_matured_date"] = "2025-01-09"
    bad_purge_input.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="purge/embargo"):
        store.evaluate_partition(
            protocol_id="protocol-fixture-1", partition="development", run_id="bad-purge-run",
            package_id=package["package_id"], input_artifact_path=bad_purge_input,
        )

    acceptance_input = _write_input(
        tmp_path, package_id=package["package_id"], partition="independent_acceptance", name="acceptance-view"
    )
    store.evaluate_partition(
        protocol_id="protocol-fixture-1", partition="independent_acceptance", run_id="acceptance-view-run",
        package_id=package["package_id"], input_artifact_path=acceptance_input,
    )
    second_input = _write_input(
        tmp_path, package_id=package["package_id"], partition="independent_acceptance", name="acceptance-view-again"
    )
    with pytest.raises(ValueError, match="already been viewed/claimed"):
        store.evaluate_partition(
            protocol_id="protocol-fixture-1", partition="independent_acceptance", run_id="acceptance-second-run",
            package_id=package["package_id"], input_artifact_path=second_input,
        )
    with pytest.raises(ValueError, match="already been attempted"):
        store.evaluate_partition(
            protocol_id="protocol-fixture-1", partition="independent_acceptance", run_id="acceptance-view-run",
            package_id=package["package_id"], input_artifact_path=acceptance_input,
        )


def test_failed_computed_economic_gate_cannot_be_published(tmp_path):
    store = HarnessReleaseStore(tmp_path, project_root=tmp_path, test_mode=True)
    store.predeclare_protocol("protocol-fixture-1", _protocol())
    package = store.register_package(_package())
    reports = {}
    for partition in ("development", "selection", "independent_acceptance", "forward_shadow"):
        path = _write_input(
            tmp_path, package_id=package["package_id"], partition=partition, name=f"fail-{partition}",
            bad_economics=(partition == "independent_acceptance"),
        )
        reports[partition] = store.evaluate_partition(
            protocol_id="protocol-fixture-1", partition=partition, run_id=f"fail-{partition}-run",
            package_id=package["package_id"], input_artifact_path=path,
        )
    assert reports["independent_acceptance"]["economic_gate"]["passed"] is False
    with pytest.raises(ValueError, match="computed research/economic/shadow gates did not pass"):
        store.publish_release(
            package_id=package["package_id"],
            evidence_ids={name: report["evidence_id"] for name, report in reports.items()},
            actor_role="release_manager", reason="failed acceptance gate",
        )


def test_evidence_recompute_detects_changed_source_and_forward_fixture_stays_pending(tmp_path):
    store, package, evidence = _registry(tmp_path)
    report = evidence["development"]
    (tmp_path / "development.source").write_text("altered source artifact", encoding="utf-8")
    with pytest.raises(ValueError, match="source artifact hash mismatch"):
        store._verified_evidence(report["evidence_id"])
    assert evidence["forward_shadow"]["evidence_status"] == "fixture_only"
    assert evidence["forward_shadow"]["production_eligible"] is False


def test_rollback_restores_compatible_complete_bundle_and_audit(tmp_path):
    store, package, evidence = _registry(tmp_path)
    published = store.publish_release(
        package_id=package["package_id"],
        evidence_ids={name: report["evidence_id"] for name, report in evidence.items()},
        actor_role="release_manager",
        reason="test-only full restoration verification",
        release_id="test-release-rollback",
    )
    with pytest.raises(PermissionError, match="release_manager"):
        store.rollback("seed", config={}, actor_role="researcher", reason="unauthorized")
    with pytest.raises(ValueError, match="incompatible with the requested configuration"):
        store.rollback(
            "test-release-rollback", config={"prediction_horizon_sessions": 5},
            actor_role="release_manager", reason="wrong config",
        )
    result = store.rollback(
        "test-release-rollback",
        config=_package()["config"],
        actor_role="release_manager",
        reason="restore complete compatible package",
        training_state=_package()["runtime_context"]["training_state"],
        training_seed_provenance=_package()["runtime_context"]["training_seed_provenance"],
        versions=_package()["runtime_context"]["versions"],
    )
    active, version_id, metadata = store.load_active(
        config=_package()["config"],
        training_state=_package()["runtime_context"]["training_state"],
        training_seed_provenance=_package()["runtime_context"]["training_seed_provenance"],
        versions=_package()["runtime_context"]["versions"],
    )
    assert result["restored_bundle"]["complete_package_sha256"] == published["package_sha256"]
    assert version_id == "candidate-17"
    assert active == seed_candidate()
    assert metadata["restored_package"] == published["complete_runtime_package"]
    assert [row["action"] for row in store.audit_history()] == ["publish", "rollback"]


def test_package_and_bundle_tampering_fails_closed(tmp_path):
    store, package, evidence = _registry(tmp_path)
    with pytest.raises(ValueError, match="four registry evidence IDs"):
        store.publish_release(
            package_id=package["package_id"], evidence_ids={},
            actor_role="release_manager", reason="missing evidence",
        )
    with pytest.raises(ValueError, match="complete package must include"):
        store.register_package({"candidate_version_id": "bad", "candidate": seed_candidate()})
