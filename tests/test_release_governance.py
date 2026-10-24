import hashlib
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from adaptive_mas.agents.harness_prompts import seed_candidate
from adaptive_mas.harness_evolution import HarnessVersionStore
from adaptive_mas.package_identity import source_package_identity
from adaptive_mas.research_provenance import build_research_package_provenance, stable_hash
from adaptive_mas.release_governance import HarnessReleaseStore, _qualify_evidence


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
        "schema_version": "adaptive-mas.release-evaluation-protocol.v2",
        "evaluation_scope_sha256": "e" * 64,
        "baseline_candidate": {
            "candidate_version_id": "seed",
            "candidate_sha256": stable_hash(seed_candidate()),
        },
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
        position = starts[partition] + offset
        row = {
            "event_group_id": f"{partition}-event-{offset}",
            "prediction_date": days[position].isoformat(),
        }
        if partition != "forward_shadow":
            row["label_matured_date"] = days[position + 3].isoformat()
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


def _write_artifact(root, artifact_id, role, value):
    relative = f"fixtures/{artifact_id}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return {
        "artifact_id": artifact_id,
        "role": role,
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _research_artifact(*, as_of, candidate_version_id, candidate_sha256, complete=True, research_cost=0.01):
    final_tools = {
        "market": "submit_report",
        "fundamental": "submit_report",
        "research_manager": "finalize_research",
    }
    agents = {}
    for role, final_tool in final_tools.items():
        findings = []
        if role == "fundamental":
            findings = [{
                "symbol": "600000",
                "evidence_ids": ["ev-1"],
                "observations": [{
                    "evidence_id": "ev-1",
                    "field": "revenue",
                    "report_period": "2024Q3",
                    "value": 10.0,
                }],
                "conclusion": "Captured source supports the reported value.",
            }]
        tool_trace = [{"tool": "read_cached_evidence", "result": {"evidence_id": "ev-1"}}]
        if complete:
            tool_trace.append({"tool": final_tool, "result": {}})
        agents[role] = {
            "report": {"summary": "Captured source findings were checked.", "findings": findings},
            "tool_trace": tool_trace,
        }
    result = {
        "schema_version": "adaptive-mas.harness-research.v1",
        "status": "completed",
        "public_evidence_mode": "captured_public_cache_cutoff",
        "as_of": as_of.isoformat(),
        "agents": agents,
        "evidence_catalog": {
            "ev-1": {
                "symbol": "600000",
                "metric": "revenue",
                "value": 10.0,
                "report_period": "2024Q3",
                "title": "Captured fixture source",
                "source_type": "fixture_public_source",
                "published_at": f"{(as_of - timedelta(days=1)).isoformat()}T09:00:00+08:00",
                "available_at": f"{(as_of - timedelta(days=1)).isoformat()}T09:00:00+08:00",
                "fetched_at": f"{(as_of - timedelta(days=1)).isoformat()}T10:00:00+08:00",
            },
        },
        "task_usage": {
            "provider_call_count": 1,
            "attempt_count": 1,
            "response_cache_reuse_count": 0,
            "model_turn_count": 1,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "logical_estimated_cost_cny": research_cost,
            "api_incremental_cost_cny": research_cost,
        },
        "cost": {
            "call_count": 1,
            "logical_inference_count": 1,
            "logical_inference_estimated_cost_cny": research_cost,
            "incremental_api_cost_cny": research_cost,
        },
        "required_observations": [],
    }
    market_identity = _research_market_snapshot(as_of)["market_snapshot_identity"]
    result["package_provenance"] = build_research_package_provenance(
        result,
        as_of=as_of,
        public_evidence_mode="captured_public_cache_cutoff",
        horizon=3,
        config={"prediction_horizon_sessions": 3},
        candidate_version_id=candidate_version_id,
        candidate_sha256=candidate_sha256,
        implementation_identity=source_package_identity(),
        market_identity=market_identity,
        model="fixture-model",
        endpoint="https://fixture.invalid",
        source_snapshot={
            "schema_version": "public-expert-snapshot.v1",
            "data_mode": "captured_public_cache_cutoff",
            "requested_decision_time": f"{as_of.isoformat()}T16:00:00+08:00",
            "evaluation_horizon_sessions": 3,
        },
    )
    return result


def _research_market_snapshot(as_of):
    payload = {
        "decision_date": as_of.isoformat(),
        "trading_dates": [as_of.isoformat()],
        "members": [],
        "adjusted_bars": [],
        "execution_bars": [],
        "index_bars": [],
        "membership_snapshots": [],
    }
    return {
        "schema_version": "adaptive-mas.research-market-snapshot.v1",
        "snapshot_payload": payload,
        "market_snapshot_identity": {
            "sha256": stable_hash(payload),
            "row_counts": {
                "trading_dates": 1,
                "members": 0,
                "adjusted_bars": 0,
                "execution_bars": 0,
                "index_bars": 0,
            },
        },
    }


def _artifact_backed_development_input(root, package, *, candidate_complete=True):
    scope = {"schema_version": "fixture-evaluation-scope.v1", "scope": "explicit-test-fixture"}
    scope_hash = stable_hash(scope)
    protocol = _protocol()
    protocol["evaluation_scope_sha256"] = scope_hash
    store = HarnessReleaseStore(root / "release_store", project_root=root, test_mode=True)
    store.predeclare_protocol("artifact-protocol", protocol)
    registered = store.register_package(package)
    descriptors = [_write_artifact(root, "scope", "frozen_scope_manifest", scope)]
    records = []
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(40)]
    for offset in range(3):
        as_of = days[offset]
        descriptors.append(_write_artifact(
            root, f"research-market-{offset}", "research_market_snapshot", _research_market_snapshot(as_of)
        ))
        candidate_id = f"candidate-research-{offset}"
        baseline_id = f"baseline-research-{offset}"
        descriptors.append(_write_artifact(
            root, candidate_id, "research_result",
            _research_artifact(
                as_of=as_of,
                candidate_version_id=package["candidate_version_id"],
                candidate_sha256=stable_hash(package["candidate"]),
                complete=candidate_complete,
            ),
        ))
        descriptors.append(_write_artifact(
            root, baseline_id, "research_result",
            _research_artifact(
                as_of=as_of,
                candidate_version_id="seed",
                candidate_sha256=stable_hash(seed_candidate()),
                complete=True,
            ),
        ))
        records.append({
            "event_group_id": f"artifact-event-{offset}",
            "prediction_date": as_of.isoformat(),
            "label_matured_date": days[offset + 3].isoformat(),
            "candidate_research_artifact_id": candidate_id,
            "baseline_research_artifact_id": baseline_id,
            "candidate_research_score": 999.0,
            "baseline_research_score": -999.0,
            "candidate_fact_accuracy": 1.0,
            "baseline_fact_accuracy": 0.0,
        })
    input_path = root / "fixtures" / "artifact-development.json"
    input_path.write_text(json.dumps({
        "schema_version": "adaptive-mas.release-evaluation-input.v2",
        "package_id": registered["package_id"],
        "source_manifest": {
            "fixture_only": True,
            "data_mode": "captured_public_cache_cutoff",
            "source_artifacts": descriptors,
        },
        "records": records,
    }, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    report = store.evaluate_partition(
        protocol_id="artifact-protocol",
        partition="development",
        run_id="artifact-development-run",
        package_id=registered["package_id"],
        input_artifact_path=input_path,
    )
    return store, registered, report


def _fixture_rule_pack(trading_dates):
    days = [date.fromisoformat(value) for value in trading_dates]
    return {
        "ruleset_id": "fixture-rules-v1",
        "effective_date": days[0].isoformat(),
        "source_title": "Explicit test-only market rules",
        "source_url": "https://example.invalid/fixture-rules",
        "symbols": {
            "600000": {
                "board": "fixture-board",
                "order_unit": 100,
                "minimum_buy_quantity": 100,
                "odd_lot_sell": True,
                "t_plus_one": True,
                "tick_size": 0.01,
                "price_limit_pct": 0.1,
                "maximum_daily_participation": 1.0,
                "liquidity_lookback_sessions": 1,
            },
        },
        "fees": {
            "commission_per_side_bps": 5,
            "minimum_commission_cny": 5,
            "transfer_fee_per_side_bps": 0.1,
            "stamp_duty_sell_bps": 5,
            "slippage_per_side_bps": 5,
        },
    }


def _fixture_execution_request(trading_dates, market_hash, rule_hash):
    days = [date.fromisoformat(value) for value in trading_dates]
    market_bars = [{
        "symbol": "600000",
        "trade_date": day.isoformat(),
        "tradestatus": "1",
        "open": 10.0,
        "high": 10.0,
        "low": 10.0,
        "close": 10.0,
        "preclose": 10.0,
        "amount": 1_000_000.0,
        "volume": 100_000,
    } for day in days]
    return {
        "execution_mode": "nominal_research",
        "trading_dates": trading_dates,
        "market_bars": market_bars,
        "orders": [],
        "rule_pack": _fixture_rule_pack(trading_dates),
        "initial_cash_cny": 10_000,
        "source_artifact_hashes": {"market-bars": market_hash, "execution-rules": rule_hash},
    }


def _artifact_backed_acceptance_input(root, package, *, tamper_result=False):
    scope = {"schema_version": "fixture-evaluation-scope.v1", "scope": "explicit-test-fixture"}
    protocol = _protocol()
    protocol["evaluation_scope_sha256"] = stable_hash(scope)
    store = HarnessReleaseStore(root / "release_store", project_root=root, test_mode=True)
    store.predeclare_protocol("artifact-acceptance-protocol", protocol)
    registered = store.register_package(package)
    descriptors = [_write_artifact(root, "scope", "frozen_scope_manifest", scope)]
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(40)]
    records = []
    from adaptive_mas.execution_ledger import simulate_execution_ledger

    for offset in range(3):
        as_of = days[18 + offset]
        descriptors.append(_write_artifact(
            root, f"research-market-{offset}", "research_market_snapshot", _research_market_snapshot(as_of)
        ))
        identities = {}
        for arm, version_id, candidate_sha, logical_cost in (
            ("candidate", package["candidate_version_id"], stable_hash(package["candidate"]), 0.01),
            ("baseline", "seed", stable_hash(seed_candidate()), 0.02),
        ):
            research_id = f"{arm}-research-{offset}"
            research = _research_artifact(
                as_of=as_of,
                candidate_version_id=version_id,
                candidate_sha256=candidate_sha,
                research_cost=logical_cost,
            )
            descriptors.append(_write_artifact(root, research_id, "research_result", research))
            market_identity = research["package_provenance"]["source_manifest"]["source_package_identity"]["market_snapshot_identity"]
            execution_dates = [day.isoformat() for day in days[18 + offset:22 + offset]]
            rule_artifact = _write_artifact(
                root,
                f"execution-rules-{arm}-{offset}",
                "execution_rules",
                {
                    "schema_version": "adaptive-mas.execution-rule-pack.v1",
                    "rule_pack": _fixture_rule_pack(execution_dates),
                },
            )
            descriptors.append(rule_artifact)
            request = _fixture_execution_request(execution_dates, "0" * 64, rule_artifact["sha256"])
            settlement_market = {
                "schema_version": "adaptive-mas.settlement-market-data.v1",
                "market_bars": request["market_bars"],
                "corporate_actions": [],
            }
            market_descriptor = _write_artifact(
                root, f"settlement-market-{arm}-{offset}", "market_data", settlement_market
            )
            descriptors.append(market_descriptor)
            request["source_artifact_hashes"]["market-bars"] = market_descriptor["sha256"]
            request_id = f"{arm}-execution-request-{offset}"
            wrapper = {
                "schema_version": "adaptive-mas.execution-ledger-request.v1",
                "market_snapshot_identity": market_identity,
                "request": request,
            }
            descriptors.append(_write_artifact(root, request_id, "execution_request", wrapper))
            ledger_request = {
                **request,
                "research_cost_cny_by_date": {as_of.isoformat(): logical_cost},
            }
            execution_result = simulate_execution_ledger(**ledger_request)
            if tamper_result and arm == "candidate" and offset == 0:
                execution_result["summary"]["net_total_return"] = 0.75
            result_id = f"{arm}-execution-result-{offset}"
            descriptors.append(_write_artifact(root, result_id, "execution_result", execution_result))
            identities[f"{arm}_research_artifact_id"] = research_id
            identities[f"{arm}_execution_request_artifact_id"] = request_id
            identities[f"{arm}_execution_result_artifact_id"] = result_id
        records.append({
            "event_group_id": f"economic-event-{offset}",
            "prediction_date": as_of.isoformat(),
            "label_matured_date": days[21 + offset].isoformat(),
            **identities,
            "candidate_net_return": 999.0,
            "baseline_net_return": -999.0,
            "candidate_total_cost_cny": 0.0,
            "baseline_total_cost_cny": 999.0,
            "candidate_annualized_risk": 999.0,
        })
    input_path = root / "fixtures" / "artifact-acceptance.json"
    input_path.write_text(json.dumps({
        "schema_version": "adaptive-mas.release-evaluation-input.v2",
        "package_id": registered["package_id"],
        "source_manifest": {
            "fixture_only": True,
            "data_mode": "captured_public_cache_cutoff",
            "source_artifacts": descriptors,
        },
        "records": records,
    }, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    report = store.evaluate_partition(
        protocol_id="artifact-acceptance-protocol",
        partition="independent_acceptance",
        run_id="artifact-acceptance-run",
        package_id=registered["package_id"],
        input_artifact_path=input_path,
    )
    return report


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
    with pytest.raises(ValueError, match="caller-supplied evaluation summaries cannot qualify for production evidence"):
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
    store.predeclare_protocol("protocol-renamed", _protocol())
    renamed_input = _write_input(
        tmp_path, package_id=package["package_id"], partition="independent_acceptance", name="acceptance-renamed"
    )
    with pytest.raises(ValueError, match="already been viewed/claimed for this frozen data scope"):
        store.evaluate_partition(
            protocol_id="protocol-renamed", partition="independent_acceptance", run_id="acceptance-renamed-run",
            package_id=package["package_id"], input_artifact_path=renamed_input,
        )


def test_horizon_maturity_is_measured_in_frozen_trading_sessions(tmp_path):
    store = HarnessReleaseStore(tmp_path, project_root=tmp_path, test_mode=True)
    store.predeclare_protocol("horizon-protocol", _protocol())
    package = store.register_package(_package())
    path = _write_input(tmp_path, package_id=package["package_id"], partition="development", name="short-label")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["label_matured_date"] = "2025-01-03"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="at least horizon_sessions"):
        store.evaluate_partition(
            protocol_id="horizon-protocol", partition="development", run_id="short-label-run",
            package_id=package["package_id"], input_artifact_path=path,
        )


def test_production_rejects_caller_supplied_summary_and_normal_gate_paths_are_distinct(tmp_path):
    assert _qualify_evidence(
        "development", fixture_only=False, quality_pass=True
    ) == ("passed", True)
    assert _qualify_evidence(
        "development", fixture_only=False, quality_pass=False
    ) == ("rejected", False)
    assert _qualify_evidence(
        "selection", fixture_only=False, quality_pass=True
    ) == ("passed", True)
    assert _qualify_evidence(
        "selection", fixture_only=False, quality_pass=False
    ) == ("rejected", False)
    assert _qualify_evidence(
        "independent_acceptance", fixture_only=False, quality_pass=True, economic_pass=False
    ) == ("rejected", False)
    assert _qualify_evidence(
        "independent_acceptance", fixture_only=False, quality_pass=True, economic_pass=True
    ) == ("passed", True)

    store = HarnessReleaseStore(tmp_path / "production-registry", project_root=tmp_path)
    store.predeclare_protocol("production-protocol", _protocol())
    package = store.register_package(_package())
    manual_summary = _write_input(
        tmp_path, package_id=package["package_id"], partition="development", name="manual-summary", fixture=False
    )
    with pytest.raises(ValueError, match="caller-supplied evaluation summaries cannot qualify"):
        store.evaluate_partition(
            protocol_id="production-protocol", partition="development", run_id="manual-summary-run",
            package_id=package["package_id"], input_artifact_path=manual_summary,
        )
    assert store.status_summary()["status"] == "seed"


def test_artifact_backed_fixture_recomputes_research_scores_and_ignores_input_metrics(tmp_path):
    store, package, report = _artifact_backed_development_input(tmp_path, _package())
    assert report["test_only"] is True
    assert report["fixture_only"] is True
    assert report["research_quality_passed"] is True
    assert report["evidence_status"] == "fixture_only"
    assert report["quality_metrics"]["candidate_research_score_mean"] < 999.0
    assert report["quality_metrics"]["baseline_research_score_mean"] > -999.0
    production_view = HarnessReleaseStore(tmp_path / "release_store", project_root=tmp_path)
    with pytest.raises(ValueError, match="test and fixture paths cannot be registered"):
        production_view._verified_evidence(report["evidence_id"])


def test_artifact_backed_quality_gate_rejects_recomputed_candidate_regression(tmp_path):
    _store, _package_record, report = _artifact_backed_development_input(
        tmp_path, _package(), candidate_complete=False
    )
    assert report["quality_metrics"]["candidate_research_score_mean"] < report["quality_metrics"]["baseline_research_score_mean"]
    assert report["research_quality_passed"] is False
    assert report["evidence_status"] == "fixture_only"


def test_acceptance_economics_are_recomputed_from_reconciled_nav_and_cost_artifacts(tmp_path):
    report = _artifact_backed_acceptance_input(tmp_path, _package())
    assert report["fixture_only"] is True
    assert report["quality_metrics"]["passed"] is True
    assert report["economic_gate"]["passed"] is True
    assert report["economic_gate"]["mean_cost_savings_cny"] == pytest.approx(0.01)
    assert report["economic_gate"]["mean_candidate_minus_baseline_net_return"] > 0
    assert report["evidence_status"] == "fixture_only"


def test_acceptance_rejects_hashed_but_nonreproducible_nav_claim(tmp_path):
    with pytest.raises(ValueError, match="does not match recomputed execution"):
        _artifact_backed_acceptance_input(tmp_path, _package(), tamper_result=True)


def test_shadow_hash_must_match_a_same_day_persisted_completed_task(tmp_path):
    task_db = tmp_path / "service.sqlite3"
    store = HarnessReleaseStore(tmp_path / "registry", project_root=tmp_path, task_store_path=task_db)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    created = datetime.combine(today, datetime.min.time(), ZoneInfo("Asia/Shanghai")).replace(hour=9).astimezone(timezone.utc)
    updated = created + timedelta(minutes=1)
    package_identity = {"candidate_version_id": "candidate-x", "candidate_sha256": "a" * 64}
    decision = {
        "schema_version": "adaptive-mas.decision.v1",
        "as_of_date": today.isoformat(),
        "prediction_horizon_sessions": 3,
        "versions": {
            "harness_version_id": package_identity["candidate_version_id"],
            "harness_candidate_sha256": package_identity["candidate_sha256"],
        },
        "orders": [],
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    request = {"mode": "decision_reuse", "as_of": today.isoformat(), "horizon_sessions": 3}
    with sqlite3.connect(task_db) as connection:
        connection.execute(
            "CREATE TABLE service_jobs(task_id TEXT PRIMARY KEY,status TEXT,result_json TEXT,created_at TEXT,updated_at TEXT,request_json TEXT)"
        )
        connection.execute(
            "INSERT INTO service_jobs VALUES(?,?,?,?,?,?)",
            ("forward-task-1", "completed", json.dumps(decision, sort_keys=True), created.isoformat(), updated.isoformat(), json.dumps(request)),
        )
    artifact = {"artifact_id": "decision-1", "role": "decision", "path": "decision.json", "sha256": digest}
    evidence_row = {
        "task_id": "forward-task-1", "decision_artifact_id": "decision-1", "decision_sha256": digest,
    }
    declared_at = (created - timedelta(hours=1)).isoformat()
    assert store._verify_persisted_shadow_decision(
        row=evidence_row, artifacts_by_id={"decision-1": artifact}, prediction_date=today, declared_at=declared_at,
        package=package_identity, horizon_sessions=3,
    ) is True
    with sqlite3.connect(task_db) as connection:
        tampered = {**decision, "as_of_date": (today - timedelta(days=1)).isoformat()}
        connection.execute("UPDATE service_jobs SET result_json=? WHERE task_id=?", (json.dumps(tampered), "forward-task-1"))
    assert store._verify_persisted_shadow_decision(
        row=evidence_row, artifacts_by_id={"decision-1": artifact}, prediction_date=today, declared_at=declared_at,
        package=package_identity, horizon_sessions=3,
    ) is False


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
