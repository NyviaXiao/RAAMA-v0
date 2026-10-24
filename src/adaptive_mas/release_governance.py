"""Immutable local evidence registry and fail-closed production release gates."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from adaptive_mas.agents.harness_prompts import seed_candidate, validate_candidate
from adaptive_mas.package_identity import source_package_identity
from adaptive_mas.research_provenance import stable_hash


_RELEASE_ROLE = "release_manager"
_PARTITIONS = ("development", "selection", "independent_acceptance", "forward_shadow")
_PARTITION_ALIASES = {"acceptance": "independent_acceptance", "shadow": "forward_shadow"}
_PACKAGE_ARTIFACTS = (
    "training_seed", "training_state", "calibration", "model_versions", "data_versions", "tool_schema"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _qualify_evidence(
    partition: str,
    *,
    fixture_only: bool,
    quality_pass: bool,
    economic_pass: bool = False,
    shadow_pass: bool = False,
) -> tuple[str, bool]:
    """Classify partition evidence without conflating research and release gates."""
    if fixture_only:
        return "fixture_only", False
    if partition in {"development", "selection"}:
        passed = quality_pass
    elif partition == "independent_acceptance":
        passed = quality_pass and economic_pass
    elif partition == "forward_shadow":
        passed = shadow_pass
    else:
        raise ValueError("unknown release-evidence partition")
    return ("passed" if passed else "rejected"), passed


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _iso_day(value: object, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite numeric data") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return number


class HarnessReleaseStore:
    """The registry and production pointer share one transactional SQLite database.

    ``test_mode`` is intentionally an explicit constructor-only switch. It lets
    tests exercise bundle restoration with synthetic evidence, while such bundles
    are marked test-only and cannot be loaded by a normal production store.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        project_root: str | Path | None = None,
        task_store_path: str | Path | None = None,
        test_mode: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.task_store_path = Path(
            task_store_path or self.project_root / "data" / "processed" / "harness_service" / "harness_state.sqlite3"
        ).resolve()
        self.test_mode = bool(test_mode)
        self.database_path = self.root / "release_governance.sqlite3"
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_protocols (
                    protocol_id TEXT PRIMARY KEY,
                    protocol_json TEXT NOT NULL,
                    protocol_sha256 TEXT NOT NULL,
                    declared_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS package_artifacts (
                    package_id TEXT PRIMARY KEY,
                    package_json TEXT NOT NULL,
                    package_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluation_attempts (
                    run_id TEXT PRIMARY KEY,
                    protocol_id TEXT NOT NULL REFERENCES evaluation_protocols(protocol_id),
                    partition_name TEXT NOT NULL,
                    package_id TEXT NOT NULL REFERENCES package_artifacts(package_id),
                    input_path TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    report_json TEXT
                );
                CREATE TABLE IF NOT EXISTS partition_claims (
                    protocol_id TEXT NOT NULL REFERENCES evaluation_protocols(protocol_id),
                    partition_name TEXT NOT NULL,
                    run_id TEXT NOT NULL REFERENCES evaluation_attempts(run_id),
                    PRIMARY KEY(protocol_id, partition_name)
                );
                CREATE TABLE IF NOT EXISTS viewed_ranges (
                    protocol_id TEXT NOT NULL,
                    run_id TEXT NOT NULL REFERENCES evaluation_attempts(run_id),
                    partition_name TEXT NOT NULL,
                    event_group_id TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    label_matured_date TEXT,
                    PRIMARY KEY(run_id, event_group_id)
                );
                CREATE TABLE IF NOT EXISTS global_partition_claims (
                    evaluation_scope_sha256 TEXT NOT NULL,
                    partition_name TEXT NOT NULL,
                    protocol_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    PRIMARY KEY(evaluation_scope_sha256, partition_name)
                );
                CREATE TABLE IF NOT EXISTS global_viewed_ranges (
                    evaluation_scope_sha256 TEXT NOT NULL,
                    protocol_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    partition_name TEXT NOT NULL,
                    event_group_id TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    label_matured_date TEXT
                );
                CREATE INDEX IF NOT EXISTS global_viewed_ranges_event_idx
                    ON global_viewed_ranges(evaluation_scope_sha256, event_group_id);
                CREATE INDEX IF NOT EXISTS global_viewed_ranges_dates_idx
                    ON global_viewed_ranges(evaluation_scope_sha256, prediction_date, label_matured_date);
                CREATE TABLE IF NOT EXISTS evidence_artifacts (
                    evidence_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE REFERENCES evaluation_attempts(run_id),
                    protocol_id TEXT NOT NULL,
                    partition_name TEXT NOT NULL,
                    package_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    report_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS release_bundles (
                    release_id TEXT PRIMARY KEY,
                    bundle_json TEXT NOT NULL,
                    bundle_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS production_pointer (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    release_id TEXT REFERENCES release_bundles(release_id),
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS release_audit (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    release_id TEXT,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _check_release_actor(actor_role: str) -> None:
        if actor_role != _RELEASE_ROLE:
            raise PermissionError("only the release_manager role may publish or roll back a bundle")

    def _resolve_project_file(self, value: object) -> tuple[Path, str]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("artifact path is required")
        raw = Path(value)
        path = (self.project_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            relative = path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("evidence artifacts must stay inside the configured project root") from exc
        if not path.is_file() or path.suffix.lower() == ".csv" or path.name.lower() == "test.csv":
            raise ValueError("evidence path must be an existing non-CSV file")
        if not self.test_mode and any(
            part.lower() in {"test", "tests", "fixture", "fixtures", "synthetic", "testdata"}
            or part.lower().startswith("pytest-")
            for part in relative.parts
        ):
            raise ValueError("test and fixture paths cannot be registered as production evidence")
        return path, relative.as_posix()

    @staticmethod
    def _validate_protocol(protocol: Mapping[str, object]) -> dict[str, object]:
        if protocol.get("schema_version") != "adaptive-mas.release-evaluation-protocol.v2":
            raise ValueError("unsupported release evaluation protocol schema")
        scope_sha256 = protocol.get("evaluation_scope_sha256")
        if not isinstance(scope_sha256, str) or not _HEX64.fullmatch(scope_sha256):
            raise ValueError("protocol must bind one frozen evaluation_scope_sha256")
        baseline = protocol.get("baseline_candidate")
        if not isinstance(baseline, Mapping):
            raise ValueError("protocol must freeze a baseline candidate identity")
        baseline_version = str(baseline.get("candidate_version_id", ""))
        baseline_sha256 = baseline.get("candidate_sha256")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", baseline_version) or not isinstance(baseline_sha256, str) or not _HEX64.fullmatch(baseline_sha256):
            raise ValueError("protocol baseline candidate identity is invalid")
        raw_calendar = protocol.get("trading_dates")
        if not isinstance(raw_calendar, list) or not raw_calendar:
            raise ValueError("protocol must declare the frozen trading calendar")
        calendar = [_iso_day(item, "trading calendar date") for item in raw_calendar]
        if calendar != sorted(set(calendar)):
            raise ValueError("protocol trading calendar must be strictly increasing and unique")
        horizon = int(protocol.get("horizon_sessions", 0))
        embargo = int(protocol.get("embargo_sessions", -1))
        if horizon < 1 or embargo < horizon:
            raise ValueError("embargo_sessions must be at least the prediction horizon")
        minimum_groups = int(protocol.get("minimum_event_groups", 0))
        if minimum_groups < 3:
            raise ValueError("minimum_event_groups must be at least three grouped events")
        partition_input = protocol.get("partitions")
        if not isinstance(partition_input, Mapping):
            raise ValueError("protocol must declare all four partitions")
        partitions: dict[str, dict[str, str]] = {}
        calendar_index = {day: index for index, day in enumerate(calendar)}
        for name in _PARTITIONS:
            bounds = partition_input.get(name)
            if not isinstance(bounds, Mapping):
                raise ValueError(f"protocol partition {name} is required")
            start = _iso_day(bounds.get("start"), f"{name}.start")
            end = _iso_day(bounds.get("end"), f"{name}.end")
            if start not in calendar_index or end not in calendar_index or start > end:
                raise ValueError(f"{name} bounds must be ordered dates on the frozen trading calendar")
            partitions[name] = {"start": start.isoformat(), "end": end.isoformat()}
        for left, right in zip(_PARTITIONS, _PARTITIONS[1:]):
            left_end = calendar_index[_iso_day(partitions[left]["end"], left)]
            right_start = calendar_index[_iso_day(partitions[right]["start"], right)]
            if right_start <= left_end:
                raise ValueError("evaluation partitions must be chronological and nonoverlapping")
            if right_start - left_end - 1 < embargo:
                raise ValueError("partition gap is shorter than the predeclared embargo")
        gate = protocol.get("economic_gate")
        if not isinstance(gate, Mapping):
            raise ValueError("protocol must predeclare an economic and risk gate")
        normalized_gate = {
            "maximum_noninferiority_loss": _finite(gate.get("maximum_noninferiority_loss"), "noninferiority margin", minimum=0),
            "maximum_drawdown_worsening": _finite(gate.get("maximum_drawdown_worsening"), "drawdown margin", minimum=0),
            "annualized_risk_budget": _finite(gate.get("annualized_risk_budget"), "annualized risk budget", minimum=1e-12),
            "minimum_cost_savings_cny": _finite(gate.get("minimum_cost_savings_cny", 0), "minimum cost savings", minimum=0),
            "one_sided_confidence": float(gate.get("one_sided_confidence", 0.95)),
        }
        if normalized_gate["one_sided_confidence"] != 0.95:
            raise ValueError("current evaluator supports a predeclared one-sided 95% confidence bound")
        return {
            "schema_version": protocol["schema_version"],
            "evaluation_scope_sha256": scope_sha256,
            "baseline_candidate": {
                "candidate_version_id": baseline_version,
                "candidate_sha256": baseline_sha256,
            },
            "trading_dates": [day.isoformat() for day in calendar],
            "horizon_sessions": horizon,
            "embargo_sessions": embargo,
            "minimum_event_groups": minimum_groups,
            "partitions": partitions,
            "economic_gate": normalized_gate,
        }

    def predeclare_protocol(self, protocol_id: str, protocol: Mapping[str, object]) -> dict[str, object]:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", protocol_id):
            raise ValueError("protocol_id must be a safe identifier")
        normalized = self._validate_protocol(protocol)
        encoded = _json(normalized)
        digest = stable_hash(normalized)
        now = self._now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT protocol_sha256 FROM evaluation_protocols WHERE protocol_id=?", (protocol_id,)
            ).fetchone()
            if prior is not None:
                if prior["protocol_sha256"] != digest:
                    connection.rollback()
                    raise ValueError("predeclared protocol id is immutable")
                connection.commit()
                return {"protocol_id": protocol_id, "protocol_sha256": digest, "status": "already_declared"}
            connection.execute(
                "INSERT INTO evaluation_protocols VALUES (?,?,?,?)",
                (protocol_id, encoded, digest, now),
            )
            connection.commit()
        return {"protocol_id": protocol_id, "protocol_sha256": digest, "status": "predeclared"}

    @staticmethod
    def _normalize_package(package: Mapping[str, object]) -> dict[str, object]:
        candidate = validate_candidate(dict(package.get("candidate", {})))
        candidate_version_id = str(package.get("candidate_version_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", candidate_version_id):
            raise ValueError("candidate_version_id must be a safe identifier")
        config = package.get("config")
        runtime_context = package.get("runtime_context")
        artifacts = package.get("artifacts")
        if not isinstance(config, Mapping) or not config:
            raise ValueError("complete package must include the full runtime configuration")
        if not isinstance(runtime_context, Mapping):
            raise ValueError("complete package must include its runtime context")
        required_context = {"training_state", "training_seed_provenance", "versions"}
        if not required_context.issubset(runtime_context) or any(
            not isinstance(runtime_context[name], Mapping) for name in required_context
        ):
            raise ValueError("runtime context must include training state, seed provenance, and versions")
        if not isinstance(artifacts, Mapping) or not set(_PACKAGE_ARTIFACTS).issubset(artifacts):
            raise ValueError("complete package must include training, calibration, model, data, and tool artifacts")
        normalized_artifacts = {}
        for name in _PACKAGE_ARTIFACTS:
            value = artifacts[name]
            if not isinstance(value, Mapping) or "payload" not in value:
                raise ValueError(f"package artifact {name} must include a payload")
            payload = value["payload"]
            if payload is None or payload == {} or payload == []:
                raise ValueError(f"package artifact {name} cannot be empty")
            normalized_artifacts[name] = {
                "payload": payload,
                "payload_sha256": stable_hash(payload),
                "version": str(value.get("version", "unversioned")),
            }
        implementation = source_package_identity()
        package_body = {
            "schema_version": "adaptive-mas.complete-runtime-package.v1",
            "candidate_version_id": candidate_version_id,
            "candidate": candidate,
            "candidate_sha256": _sha256_text(_json(candidate)),
            "config": dict(config),
            "config_sha256": stable_hash(dict(config)),
            "runtime_context": {key: dict(value) for key, value in runtime_context.items()},
            "runtime_context_sha256": stable_hash(dict(runtime_context)),
            "artifacts": normalized_artifacts,
            "implementation_identity": implementation,
            "implementation_package_sha256": implementation["implementation_package_sha256"],
            "hard_policy_sha256": implementation["implementation_files_sha256"]["src/adaptive_mas/agents/harness.py"],
        }
        return package_body

    def register_package(self, package: Mapping[str, object]) -> dict[str, object]:
        normalized = self._normalize_package(package)
        encoded = _json(normalized)
        digest = stable_hash(normalized)
        package_id = f"pkg-{digest[:24]}"
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT package_sha256 FROM package_artifacts WHERE package_id=?", (package_id,)
            ).fetchone()
            if prior is not None and prior["package_sha256"] != digest:
                connection.rollback()
                raise ValueError("package artifact id collision")
            connection.execute(
                "INSERT OR IGNORE INTO package_artifacts VALUES (?,?,?,?)",
                (package_id, encoded, digest, self._now()),
            )
            connection.commit()
        return {"package_id": package_id, "package_sha256": digest, "candidate_sha256": normalized["candidate_sha256"]}

    def register_package_file(self, artifact_path: str | Path) -> dict[str, object]:
        path, _relative = self._resolve_project_file(str(artifact_path))
        try:
            package = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("package artifact must be a readable JSON object") from exc
        if not isinstance(package, Mapping):
            raise ValueError("package artifact must be a JSON object")
        return self.register_package(package)

    def _load_package(self, package_id: str) -> dict[str, object]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT package_json,package_sha256 FROM package_artifacts WHERE package_id=?", (package_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"unknown immutable package artifact {package_id}")
        package = json.loads(row["package_json"])
        if stable_hash(package) != row["package_sha256"]:
            raise ValueError("complete package artifact failed its content hash")
        if package.get("implementation_identity") != source_package_identity():
            raise ValueError("complete package artifact is incompatible with the current implementation")
        for name in _PACKAGE_ARTIFACTS:
            artifact = package["artifacts"][name]
            if stable_hash(artifact["payload"]) != artifact["payload_sha256"]:
                raise ValueError(f"complete package artifact {name} failed its content hash")
        candidate = validate_candidate(package["candidate"])
        if _sha256_text(_json(candidate)) != package["candidate_sha256"]:
            raise ValueError("complete package candidate hash mismatch")
        if stable_hash(package["config"]) != package["config_sha256"]:
            raise ValueError("complete package configuration hash mismatch")
        if stable_hash(package["runtime_context"]) != package["runtime_context_sha256"]:
            raise ValueError("complete package runtime context hash mismatch")
        return package

    def _load_protocol(self, protocol_id: str) -> tuple[dict[str, object], str, str]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT protocol_json,protocol_sha256,declared_at FROM evaluation_protocols WHERE protocol_id=?",
                (protocol_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"protocol {protocol_id} has not been predeclared")
        protocol = json.loads(row["protocol_json"])
        if stable_hash(protocol) != row["protocol_sha256"]:
            raise ValueError("predeclared protocol failed its content hash")
        return protocol, str(row["protocol_sha256"]), str(row["declared_at"])

    def _verified_source_artifacts(
        self, manifest: Mapping[str, object], *, require_descriptors: bool = False
    ) -> list[dict[str, str]]:
        raw = manifest.get("source_artifacts")
        if not isinstance(raw, list) or not raw:
            raise ValueError("source manifest must reference at least one source artifact")
        result = []
        artifact_ids: set[str] = set()
        allowed_roles = {
            "frozen_scope_manifest", "research_market_snapshot", "market_data", "research_result",
            "execution_rules", "execution_request", "execution_result", "decision",
        }
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("source artifact records must be objects")
            path, relative = self._resolve_project_file(item.get("path"))
            actual = _sha256_bytes(path.read_bytes())
            if item.get("sha256") != actual:
                raise ValueError(f"source artifact hash mismatch: {Path(relative).name}")
            artifact_id = str(item.get("artifact_id", Path(relative).name))
            role = str(item.get("role", "legacy_summary"))
            if not artifact_id or artifact_id in artifact_ids:
                raise ValueError("source artifact IDs must be unique and nonempty")
            artifact_ids.add(artifact_id)
            if require_descriptors and ("artifact_id" not in item or "role" not in item):
                raise ValueError("production artifacts must declare an artifact_id and semantic role")
            if require_descriptors and role not in allowed_roles:
                raise ValueError(f"unsupported production evidence artifact role: {role}")
            result.append({
                "artifact_id": artifact_id,
                "role": role,
                "path": relative,
                "sha256": actual,
            })
        return sorted(result, key=lambda row: row["path"])

    def _artifact_json(self, artifact: Mapping[str, str]) -> dict[str, object]:
        path = self.project_root / artifact["path"]
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"evidence artifact is not readable JSON: {path.name}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"evidence artifact must contain a JSON object: {path.name}")
        return dict(value)

    def _derive_production_rows(
        self,
        *,
        protocol: Mapping[str, object],
        package: Mapping[str, object],
        partition: str,
        rows: Sequence[object],
        source_artifacts: Sequence[Mapping[str, str]],
    ) -> list[dict[str, object]]:
        """Rebuild quality and finance metrics from hashed, role-bound source artifacts."""
        from adaptive_mas.execution_ledger import simulate_execution_ledger
        from adaptive_mas.harness_evolution import score_research_result
        from adaptive_mas.research_provenance import validate_research_package_provenance

        by_id = {item["artifact_id"]: item for item in source_artifacts}
        role_counts: dict[str, int] = {}
        for item in source_artifacts:
            role_counts[item["role"]] = role_counts.get(item["role"], 0) + 1
        if role_counts.get("frozen_scope_manifest", 0) != 1:
            raise ValueError("production input must reference one frozen scope manifest")
        scope_artifact = next(item for item in source_artifacts if item["role"] == "frozen_scope_manifest")
        if stable_hash(self._artifact_json(scope_artifact)) != protocol["evaluation_scope_sha256"]:
            raise ValueError("frozen scope manifest does not match the predeclared evaluation scope")
        research_market_artifacts = [item for item in source_artifacts if item["role"] == "research_market_snapshot"]
        if not research_market_artifacts:
            raise ValueError("production input must bind research_market_snapshot artifacts")
        market_identities = []
        for artifact in research_market_artifacts:
            snapshot = self._artifact_json(artifact)
            payload = snapshot.get("snapshot_payload")
            identity = snapshot.get("market_snapshot_identity")
            if snapshot.get("schema_version") != "adaptive-mas.research-market-snapshot.v1" or not isinstance(payload, Mapping) or not isinstance(identity, Mapping):
                raise ValueError("research market snapshot artifact has an unsupported shape")
            expected_counts = {
                "trading_dates": len(payload.get("trading_dates", [])),
                "members": len(payload.get("members", [])),
                "adjusted_bars": len(payload.get("adjusted_bars", [])),
                "execution_bars": len(payload.get("execution_bars", [])),
                "index_bars": len(payload.get("index_bars", [])),
            }
            if identity.get("sha256") != stable_hash(dict(payload)) or identity.get("row_counts") != expected_counts:
                raise ValueError("research market snapshot identity does not match its source rows")
            market_identities.append(dict(identity))
        market_hashes = {item["sha256"] for item in source_artifacts if item["role"] == "market_data"}
        implementation = source_package_identity()
        baseline = protocol["baseline_candidate"]
        package_identity = {
            "candidate_version_id": package["candidate_version_id"],
            "candidate_sha256": package["candidate_sha256"],
        }
        expected_identities = {
            "candidate": package_identity,
            "baseline": dict(baseline),
        }
        derived: list[dict[str, object]] = []
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError("production evaluation records must be objects")
            row = dict(raw_row)
            prediction_date = _iso_day(row.get("prediction_date"), "prediction_date")
            output: dict[str, object] = dict(row)
            paired_execution_calendars: dict[str, list[str]] = {}
            for arm in ("candidate", "baseline"):
                research_id = row.get(f"{arm}_research_artifact_id")
                if not isinstance(research_id, str) or research_id not in by_id:
                    raise ValueError(f"{arm} evaluation row must reference a hashed research artifact")
                research_artifact = by_id[str(research_id)]
                if research_artifact["role"] != "research_result":
                    raise ValueError(f"{arm} evaluation artifact role is not research_result")
                research_result = self._artifact_json(research_artifact)
                if (
                    research_result.get("schema_version") != "adaptive-mas.harness-research.v1"
                    or research_result.get("status") != "completed"
                    or research_result.get("as_of") != prediction_date.isoformat()
                    or research_result.get("public_evidence_mode") != "captured_public_cache_cutoff"
                ):
                    raise ValueError(f"{arm} research artifact is incomplete or has a mismatched date/mode")
                provenance = research_result.get("package_provenance")
                if not isinstance(provenance, Mapping):
                    raise ValueError(f"{arm} research artifact is missing package provenance")
                source_manifest = provenance.get("source_manifest")
                source_identity = source_manifest.get("source_package_identity") if isinstance(source_manifest, Mapping) else None
                if not isinstance(source_identity, Mapping):
                    raise ValueError(f"{arm} research artifact is missing source package identity")
                identity = expected_identities[arm]
                if (
                    source_identity.get("candidate_version_id") != identity["candidate_version_id"]
                    or source_identity.get("candidate_sha256") != identity["candidate_sha256"]
                ):
                    raise ValueError(f"{arm} research artifact does not match the frozen candidate identity")
                market_identity = source_identity.get("market_snapshot_identity")
                if not isinstance(market_identity, Mapping) or dict(market_identity) not in market_identities:
                    raise ValueError(f"{arm} research artifact is missing its market snapshot identity")
                matching_snapshots = [
                    self._artifact_json(artifact)
                    for artifact in research_market_artifacts
                    if self._artifact_json(artifact).get("market_snapshot_identity") == dict(market_identity)
                ]
                if not matching_snapshots or matching_snapshots[0]["snapshot_payload"].get("decision_date") != prediction_date.isoformat():
                    raise ValueError(f"{arm} research market snapshot date does not match its task")
                validate_research_package_provenance(
                    research_result,
                    provenance,
                    as_of=prediction_date,
                    public_evidence_mode="captured_public_cache_cutoff",
                    horizon=int(protocol["horizon_sessions"]),
                    candidate_version_id=str(identity["candidate_version_id"]),
                    candidate_sha256=str(identity["candidate_sha256"]),
                    implementation_identity=implementation,
                    market_identity=market_identity,
                    client=None,
                )
                metrics = score_research_result(research_result)
                output[f"{arm}_research_score"] = _finite(metrics.get("score"), f"{arm} recomputed research score", minimum=0)
                output[f"{arm}_fact_accuracy"] = _finite(metrics.get("fact_accuracy"), f"{arm} recomputed fact accuracy", minimum=0)
                output[f"{arm}_strict_pit_verified"] = metrics.get("strict_historical_pit_verified") is True

                if partition != "independent_acceptance":
                    continue
                execution_id = row.get(f"{arm}_execution_request_artifact_id")
                result_id = row.get(f"{arm}_execution_result_artifact_id")
                if not all(isinstance(value, str) and value in by_id for value in (execution_id, result_id)):
                    raise ValueError(f"{arm} acceptance row must reference hashed execution request and result artifacts")
                request_artifact = by_id[str(execution_id)]
                result_artifact = by_id[str(result_id)]
                if request_artifact["role"] != "execution_request" or result_artifact["role"] != "execution_result":
                    raise ValueError(f"{arm} acceptance artifact roles do not match their declared references")

                execution_wrapper = self._artifact_json(request_artifact)
                if execution_wrapper.get("schema_version") != "adaptive-mas.execution-ledger-request.v1":
                    raise ValueError(f"{arm} execution request artifact has an unsupported schema")
                market_request_identity = execution_wrapper.get("market_snapshot_identity")
                if market_request_identity != dict(market_identity):
                    raise ValueError(f"{arm} execution request is not linked to its research market snapshot")
                request = execution_wrapper.get("request")
                if not isinstance(request, Mapping):
                    raise ValueError(f"{arm} execution request artifact has no request object")
                request = dict(request)
                source_hashes = request.get("source_artifact_hashes")
                if not isinstance(source_hashes, Mapping) or not source_hashes:
                    raise ValueError(f"{arm} execution request is not linked to market data artifact hashes")
                referenced_hashes = {str(value) for value in source_hashes.values()}
                rule_hashes = {item["sha256"] for item in source_artifacts if item["role"] == "execution_rules"}
                if not referenced_hashes.issubset(market_hashes | rule_hashes):
                    raise ValueError(f"{arm} execution request references an unregistered source artifact")
                if not (referenced_hashes & market_hashes) or not (referenced_hashes & rule_hashes):
                    raise ValueError(f"{arm} execution request must bind both settlement data and its rule pack")
                referenced_market_hashes = referenced_hashes & market_hashes
                referenced_market_data_by_hash = {
                    artifact["sha256"]: self._artifact_json(artifact)
                    for artifact in source_artifacts
                    if artifact["role"] == "market_data" and artifact["sha256"] in referenced_market_hashes
                }
                referenced_market_data = list(referenced_market_data_by_hash.values())
                if set(referenced_market_data_by_hash) != referenced_market_hashes or any(
                    item.get("schema_version") != "adaptive-mas.settlement-market-data.v1"
                    or not isinstance(item.get("market_bars"), list)
                    for item in referenced_market_data
                ):
                    raise ValueError(
                        f"{arm} market_data artifacts do not contain versioned settlement bars "
                        f"(matched={len(referenced_market_data)}, expected={len(referenced_market_hashes)}, "
                        f"schemas={[item.get('schema_version') for item in referenced_market_data]})"
                    )
                rule_artifacts = [
                    self._artifact_json(artifact)
                    for artifact in source_artifacts
                    if artifact["role"] == "execution_rules" and artifact["sha256"] in (referenced_hashes & rule_hashes)
                ]
                if not rule_artifacts or not any(
                    asset.get("schema_version") == "adaptive-mas.execution-rule-pack.v1"
                    and asset.get("rule_pack") == request.get("rule_pack")
                    for asset in rule_artifacts
                ):
                    raise ValueError(f"{arm} execution rule pack does not match its hashed source artifact")
                available_bars = {
                    stable_hash(dict(bar))
                    for asset in referenced_market_data
                    for bar in asset["market_bars"]
                    if isinstance(bar, Mapping)
                }
                request_bars = request.get("market_bars")
                if not isinstance(request_bars, list) or any(
                    not isinstance(bar, Mapping) or stable_hash(dict(bar)) not in available_bars
                    for bar in request_bars
                ):
                    raise ValueError(f"{arm} execution request market bars do not match hashed settlement data")
                requested_actions = request.get("corporate_actions", [])
                available_actions = {
                    stable_hash(dict(action))
                    for asset in referenced_market_data
                    for action in asset.get("corporate_actions", [])
                    if isinstance(action, Mapping)
                }
                if not isinstance(requested_actions, list) or any(
                    not isinstance(action, Mapping) or stable_hash(dict(action)) not in available_actions
                    for action in requested_actions
                ):
                    raise ValueError(f"{arm} execution request corporate actions do not match hashed settlement data")
                usage = research_result.get("task_usage")
                if not isinstance(usage, Mapping):
                    raise ValueError(f"{arm} research artifact has no source task usage")
                research_cost = _finite(usage.get("logical_estimated_cost_cny"), f"{arm} research logical cost", minimum=0)
                request["research_cost_cny_by_date"] = {prediction_date.isoformat(): research_cost}
                try:
                    recomputed_ledger = simulate_execution_ledger(**request)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{arm} execution request could not be recomputed: {exc}") from exc
                supplied_result = self._artifact_json(result_artifact)
                if stable_hash(supplied_result) != stable_hash(recomputed_ledger):
                    raise ValueError(f"{arm} settlement/NAV/fee artifact does not match recomputed execution")
                summary = recomputed_ledger.get("summary")
                nav_rows = recomputed_ledger.get("daily_nav")
                reconciliation = recomputed_ledger.get("reconciliation")
                if not isinstance(summary, Mapping) or not isinstance(nav_rows, list) or len(nav_rows) < 2:
                    raise ValueError(f"{arm} execution result lacks a usable NAV series")
                maturity_date = _iso_day(row.get("label_matured_date"), "label_matured_date")
                execution_dates = [
                    str(item.get("trade_date"))
                    for item in nav_rows if isinstance(item, Mapping)
                ]
                if (
                    not execution_dates
                    or execution_dates[0] != prediction_date.isoformat()
                    or execution_dates[-1] != maturity_date.isoformat()
                ):
                    raise ValueError(f"{arm} NAV series must cover exactly prediction through label maturity")
                paired_execution_calendars[arm] = execution_dates
                if (
                    summary.get("all_cash_reconciliations_exact") is not True
                    or summary.get("all_nav_reconciliations_exact") is not True
                    or not isinstance(reconciliation, list)
                ):
                    raise ValueError(f"{arm} execution result failed exact reconciliation")
                nav_values = [_finite(item.get("nav_cny"), f"{arm} NAV", minimum=1e-12) for item in nav_rows if isinstance(item, Mapping)]
                daily_returns = [
                    _finite(item.get("daily_return"), f"{arm} daily return")
                    for item in nav_rows[1:] if isinstance(item, Mapping)
                ]
                mean_return = math.fsum(daily_returns) / len(daily_returns) if daily_returns else 0.0
                variance = math.fsum((value - mean_return) ** 2 for value in daily_returns) / max(len(daily_returns) - 1, 1)
                annualized_risk = math.sqrt(variance) * math.sqrt(252)
                peak = nav_values[0]
                drawdown = 0.0
                for value in nav_values:
                    peak = max(peak, value)
                    drawdown = min(drawdown, value / peak - 1.0)
                total_cost = (
                    _finite(summary.get("total_fees_cny"), f"{arm} fees", minimum=0)
                    + _finite(summary.get("total_slippage_cost_cny"), f"{arm} slippage", minimum=0)
                    + _finite(summary.get("total_research_cost_cny"), f"{arm} research cost", minimum=0)
                )
                output[f"{arm}_net_return"] = _finite(summary.get("net_total_return"), f"{arm} net return")
                output[f"{arm}_total_cost_cny"] = total_cost
                output[f"{arm}_annualized_risk"] = annualized_risk
                output[f"{arm}_maximum_drawdown"] = drawdown
            if paired_execution_calendars.get("candidate") != paired_execution_calendars.get("baseline"):
                raise ValueError("candidate and baseline settlement calendars must match exactly")
            derived.append(output)
        return derived

    def _verify_persisted_shadow_decision(
        self,
        *,
        row: Mapping[str, object],
        artifacts_by_id: Mapping[str, Mapping[str, str]],
        prediction_date: date,
        declared_at: str,
        package: Mapping[str, object],
        horizon_sessions: int,
    ) -> bool:
        decision_id = row.get("decision_artifact_id")
        task_id = row.get("task_id")
        if not isinstance(decision_id, str) or decision_id not in artifacts_by_id or not isinstance(task_id, str):
            return False
        artifact = artifacts_by_id[decision_id]
        if artifact["role"] != "decision" or row.get("decision_sha256") != artifact["sha256"]:
            return False
        try:
            decision = self._artifact_json(artifact)
            if (
                decision.get("schema_version") != "adaptive-mas.decision.v1"
                or decision.get("as_of_date") != prediction_date.isoformat()
                or decision.get("prediction_horizon_sessions") != horizon_sessions
            ):
                return False
            versions = decision.get("versions")
            if not isinstance(versions, Mapping) or (
                versions.get("harness_version_id") != package.get("candidate_version_id")
                or versions.get("harness_candidate_sha256") != package.get("candidate_sha256")
            ):
                return False
            with closing(sqlite3.connect(self.task_store_path, timeout=5)) as connection:
                persisted = connection.execute(
                    "SELECT status,result_json,created_at,updated_at,request_json FROM service_jobs WHERE task_id=?",
                    (task_id,),
                ).fetchone()
            if persisted is None or persisted[0] != "completed":
                return False
            persisted_decision = json.loads(persisted[1])
            created_at = datetime.fromisoformat(str(persisted[2]))
            updated_at = datetime.fromisoformat(str(persisted[3]))
            declared = datetime.fromisoformat(declared_at)
            if any(value.tzinfo is None for value in (created_at, updated_at, declared)):
                return False
            created_at = created_at.astimezone(timezone.utc)
            updated_at = updated_at.astimezone(timezone.utc)
            declared = declared.astimezone(timezone.utc)
            if created_at <= declared or updated_at < created_at or updated_at > datetime.now(timezone.utc):
                return False
            if (
                created_at.astimezone(ZoneInfo("Asia/Shanghai")).date() != prediction_date
                or updated_at.astimezone(ZoneInfo("Asia/Shanghai")).date() != prediction_date
            ):
                return False
            if stable_hash(persisted_decision) != stable_hash(decision):
                return False
            request = json.loads(persisted[4])
            return (
                isinstance(request, Mapping)
                and request.get("mode") == "decision_reuse"
                and request.get("as_of") == prediction_date.isoformat()
                and request.get("horizon_sessions") == horizon_sessions
            )
        except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _compute_report(
        self,
        *,
        protocol_id: str,
        protocol: Mapping[str, object],
        protocol_sha256: str,
        package_id: str,
        package: Mapping[str, object],
        partition: str,
        run_id: str,
        input_path: str,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        path, relative_path = self._resolve_project_file(input_path)
        input_bytes = path.read_bytes()
        artifact_sha256 = _sha256_bytes(input_bytes)
        source = json.loads(input_bytes.decode("utf-8"))
        if not isinstance(source, Mapping) or source.get("schema_version") not in {
            "adaptive-mas.release-evaluation-input.v1",
            "adaptive-mas.release-evaluation-input.v2",
        }:
            raise ValueError("unsupported immutable evaluation input artifact")
        if source.get("schema_version") == "adaptive-mas.release-evaluation-input.v1" and not self.test_mode:
            raise ValueError("caller-supplied evaluation summaries cannot qualify for production evidence")
        if source.get("package_id") != package_id:
            raise ValueError("evaluation input package ID does not match the registered package")
        manifest = source.get("source_manifest")
        rows = source.get("records")
        if not isinstance(manifest, Mapping) or not isinstance(rows, list) or not rows:
            raise ValueError("evaluation input must include a source manifest and grouped records")
        production_artifact_mode = source.get("schema_version") == "adaptive-mas.release-evaluation-input.v2"
        source_artifacts = self._verified_source_artifacts(
            manifest, require_descriptors=production_artifact_mode
        )
        fixture_only = bool(manifest.get("fixture_only", False))
        data_mode = str(manifest.get("data_mode", "unknown"))
        if not self.test_mode:
            if fixture_only:
                raise ValueError("fixture evidence can never qualify for a production release")
            if not production_artifact_mode:
                raise ValueError("production evidence requires artifact-backed evaluation input v2")
            scope_artifacts = [item for item in source_artifacts if item["role"] == "frozen_scope_manifest"]
            if len(scope_artifacts) != 1 or stable_hash(self._artifact_json(scope_artifacts[0])) != protocol["evaluation_scope_sha256"]:
                raise ValueError("frozen scope manifest does not match the predeclared evaluation scope")
            allowed_modes = {
                "development": {"captured_public_cache_cutoff"},
                "selection": {"captured_public_cache_cutoff"},
                "independent_acceptance": {"captured_public_cache_cutoff"},
                "forward_shadow": {"forward_observed"},
            }
            if data_mode not in allowed_modes[partition]:
                raise ValueError(f"{partition} evidence has an ineligible data mode")
            if not any(row["path"].startswith(("data/", "experiments/")) for row in source_artifacts):
                raise ValueError("production evidence must be rooted in versioned local data or run artifacts")

        trading_dates = [_iso_day(item, "protocol trading date") for item in protocol["trading_dates"]]
        date_index = {day: index for index, day in enumerate(trading_dates)}
        bounds = protocol["partitions"][partition]
        start = _iso_day(bounds["start"], f"{partition}.start")
        end = _iso_day(bounds["end"], f"{partition}.end")
        next_partition = _PARTITIONS[_PARTITIONS.index(partition) + 1] if partition != "forward_shadow" else None
        next_start = (
            _iso_day(protocol["partitions"][next_partition]["start"], f"{next_partition}.start")
            if next_partition else None
        )
        calendar_index = {day: idx for day, idx in date_index.items()}
        views: list[dict[str, object]] = []
        groups: set[str] = set()
        normalized_rows = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("evaluation records must be objects")
            event_group = str(row.get("event_group_id", "")).strip()
            if not event_group or event_group in groups:
                raise ValueError("each evaluation record needs one unique event_group_id")
            groups.add(event_group)
            prediction_date = _iso_day(row.get("prediction_date"), "prediction_date")
            if prediction_date not in calendar_index or not start <= prediction_date <= end:
                raise ValueError("prediction date is outside its predeclared partition")
            maturity_raw = row.get("label_matured_date")
            maturity = _iso_day(maturity_raw, "label_matured_date") if maturity_raw is not None else None
            if partition == "forward_shadow":
                if maturity is not None:
                    raise ValueError("forward-shadow rows must remain unlabelled until they mature")
            else:
                horizon_sessions = int(protocol["horizon_sessions"])
                if (
                    maturity is None
                    or maturity not in calendar_index
                    or maturity < prediction_date
                    or calendar_index[maturity] - calendar_index[prediction_date] < horizon_sessions
                ):
                    raise ValueError("label maturity must occur at least horizon_sessions after prediction on the frozen calendar")
                if next_start is not None:
                    gap = calendar_index[next_start] - calendar_index[maturity] - 1
                    if gap < int(protocol["embargo_sessions"]):
                        raise ValueError("label maturity crosses the next partition's purge/embargo boundary")
            views.append({
                "event_group_id": event_group,
                "prediction_date": prediction_date.isoformat(),
                "label_matured_date": maturity.isoformat() if maturity else None,
            })
            normalized_rows.append(dict(row))

        if production_artifact_mode and partition != "forward_shadow":
            normalized_rows = self._derive_production_rows(
                protocol=protocol,
                package=package,
                partition=partition,
                rows=normalized_rows,
                source_artifacts=source_artifacts,
            )

        minimum = int(protocol["minimum_event_groups"])
        quality_pass = False
        quality_metrics: dict[str, object] = {}
        economic_gate: dict[str, object] | None = None
        evidence_status = "recorded"
        production_eligible = False
        if partition == "forward_shadow":
            required = len(normalized_rows) >= minimum
            artifacts_by_id = {item["artifact_id"]: item for item in source_artifacts}
            declared_at = self._load_protocol(protocol_id)[2]
            if self.test_mode:
                safe_shadow = all(
                    row.get("real_orders") == 0
                    and row.get("public_deployment") is False
                    and isinstance(row.get("decision_sha256"), str)
                    and _HEX64.fullmatch(str(row["decision_sha256"]))
                    for row in normalized_rows
                )
            else:
                safe_shadow = (
                    data_mode == "forward_observed"
                    and all(
                        self._verify_persisted_shadow_decision(
                            row=row,
                            artifacts_by_id=artifacts_by_id,
                            prediction_date=_iso_day(row["prediction_date"], "prediction_date"),
                            declared_at=declared_at,
                        package=package,
                        horizon_sessions=int(protocol["horizon_sessions"]),
                        )
                        for row in normalized_rows
                    )
                )
            acceptance_end = _iso_day(protocol["partitions"]["independent_acceptance"]["end"], "acceptance.end")
            forward_after_acceptance = start > acceptance_end
            observed_after_predeclare = self.test_mode or safe_shadow
            shadow_checks_passed = required and safe_shadow and forward_after_acceptance and observed_after_predeclare and data_mode == "forward_observed"
            if not fixture_only and not observed_after_predeclare:
                evidence_status = "pending_forward"
                production_eligible = False
            else:
                evidence_status, production_eligible = _qualify_evidence(
                    partition,
                    fixture_only=fixture_only,
                    quality_pass=False,
                    shadow_pass=shadow_checks_passed,
                )
                if evidence_status == "rejected" and data_mode != "forward_observed":
                    evidence_status = "pending_forward"
        else:
            quality_values = []
            diff_returns = []
            savings = []
            candidate_risks = []
            drawdown_worsening = []
            for row in normalized_rows:
                candidate_quality = _finite(row.get("candidate_research_score"), "candidate research score")
                baseline_quality = _finite(row.get("baseline_research_score"), "baseline research score")
                candidate_fact = _finite(row.get("candidate_fact_accuracy"), "candidate fact accuracy", minimum=0)
                baseline_fact = _finite(row.get("baseline_fact_accuracy"), "baseline fact accuracy", minimum=0)
                quality_values.append((candidate_quality, baseline_quality, candidate_fact, baseline_fact))
                if partition == "independent_acceptance":
                    diff_returns.append(
                        _finite(row.get("candidate_net_return"), "candidate net return")
                        - _finite(row.get("baseline_net_return"), "baseline net return")
                    )
                    savings.append(
                        _finite(row.get("baseline_total_cost_cny"), "baseline total cost", minimum=0)
                        - _finite(row.get("candidate_total_cost_cny"), "candidate total cost", minimum=0)
                    )
                    candidate_risks.append(_finite(row.get("candidate_annualized_risk"), "candidate annualized risk", minimum=0))
                    drawdown_worsening.append(
                        _finite(row.get("baseline_maximum_drawdown"), "baseline drawdown")
                        - _finite(row.get("candidate_maximum_drawdown"), "candidate drawdown")
                    )
            quality_pass = len(quality_values) >= minimum and all(
                candidate_quality >= baseline_quality and candidate_fact >= baseline_fact
                for candidate_quality, baseline_quality, candidate_fact, baseline_fact in quality_values
            )
            if production_artifact_mode:
                quality_pass = quality_pass and all(
                    row.get("candidate_strict_pit_verified") is True
                    and row.get("baseline_strict_pit_verified") is True
                    for row in normalized_rows
                )
            quality_metrics = {
                "event_group_count": len(quality_values),
                "candidate_research_score_mean": math.fsum(row[0] for row in quality_values) / len(quality_values),
                "baseline_research_score_mean": math.fsum(row[1] for row in quality_values) / len(quality_values),
                "candidate_fact_accuracy_mean": math.fsum(row[2] for row in quality_values) / len(quality_values),
                "baseline_fact_accuracy_mean": math.fsum(row[3] for row in quality_values) / len(quality_values),
                "passed": quality_pass,
            }
            if production_artifact_mode:
                quality_metrics["strict_point_in_time_verified_group_count"] = sum(
                    row.get("candidate_strict_pit_verified") is True
                    and row.get("baseline_strict_pit_verified") is True
                    for row in normalized_rows
                )
                quality_metrics["strict_point_in_time_required"] = True
            if partition == "independent_acceptance":
                if len(diff_returns) >= 2:
                    mean_diff = math.fsum(diff_returns) / len(diff_returns)
                    sample_variance = math.fsum((value - mean_diff) ** 2 for value in diff_returns) / (len(diff_returns) - 1)
                    standard_error = math.sqrt(sample_variance / len(diff_returns))
                    lower_95 = mean_diff - 1.645 * standard_error
                else:
                    mean_diff, standard_error, lower_95 = 0.0, math.inf, -math.inf
                gate = protocol["economic_gate"]
                average_savings = math.fsum(savings) / len(savings) if savings else -math.inf
                max_risk = max(candidate_risks, default=math.inf)
                worst_drawdown_worsening = max(drawdown_worsening, default=-math.inf)
                economic_pass = (
                    len(diff_returns) >= minimum
                    and average_savings > float(gate["minimum_cost_savings_cny"])
                    and lower_95 >= -float(gate["maximum_noninferiority_loss"])
                    and max_risk <= float(gate["annualized_risk_budget"])
                    and worst_drawdown_worsening <= float(gate["maximum_drawdown_worsening"])
                )
                economic_gate = {
                    "status": "passed" if economic_pass else "failed",
                    "event_group_count": len(diff_returns),
                    "mean_candidate_minus_baseline_net_return": mean_diff,
                    "standard_error": standard_error if math.isfinite(standard_error) else None,
                    "one_sided_95_percent_lower_bound": lower_95 if math.isfinite(lower_95) else None,
                    "mean_cost_savings_cny": average_savings,
                    "maximum_candidate_annualized_risk": max_risk if math.isfinite(max_risk) else None,
                    "annualized_risk_budget": float(gate["annualized_risk_budget"]),
                    "worst_drawdown_worsening": worst_drawdown_worsening,
                    "predeclared_maximum_noninferiority_loss": float(gate["maximum_noninferiority_loss"]),
                    "predeclared_maximum_drawdown_worsening": float(gate["maximum_drawdown_worsening"]),
                    "passed": economic_pass,
                }
                evidence_status, production_eligible = _qualify_evidence(
                    partition,
                    fixture_only=fixture_only,
                    quality_pass=quality_pass,
                    economic_pass=economic_pass,
                )
            else:
                evidence_status, production_eligible = _qualify_evidence(
                    partition,
                    fixture_only=fixture_only,
                    quality_pass=quality_pass,
                )

        package_runtime = package["runtime_context"]
        report = {
            "schema_version": "adaptive-mas.release-evidence.v2",
            "run_id": run_id,
            "protocol_id": protocol_id,
            "protocol_sha256": protocol_sha256,
            "evaluation_scope_sha256": protocol["evaluation_scope_sha256"],
            "partition": partition,
            "package_id": package_id,
            "package_sha256": stable_hash(package),
            "candidate_sha256": package["candidate_sha256"],
            "implementation_package_sha256": package["implementation_package_sha256"],
            "input_artifact_path": relative_path,
            "input_artifact_sha256": artifact_sha256,
            "source_artifacts": source_artifacts,
            "artifact_binding_sha256": stable_hash(source_artifacts),
            "metric_source_mode": "artifact_recomputed" if production_artifact_mode else "test_fixture_summary",
            "source_manifest_sha256": stable_hash(dict(manifest)),
            "data_mode": data_mode,
            "fixture_only": fixture_only,
            "production_eligible": production_eligible,
            "source_range_sha256": stable_hash(views),
            "event_groups": sorted(groups),
            "views": views,
            "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "quality_metrics": quality_metrics,
            "research_quality_passed": quality_pass,
            "economic_gate": economic_gate,
            "forward_shadow_checks_passed": shadow_checks_passed if partition == "forward_shadow" else None,
            "evidence_status": evidence_status,
            "test_only": self.test_mode,
            "runtime_context_sha256": stable_hash(package_runtime),
        }
        return report, views

    def evaluate_partition(
        self,
        *,
        protocol_id: str,
        partition: str,
        run_id: str,
        package_id: str,
        input_artifact_path: str | Path,
    ) -> dict[str, object]:
        partition = _PARTITION_ALIASES.get(partition, partition)
        if partition not in _PARTITIONS:
            raise ValueError("unknown evaluation partition")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", run_id):
            raise ValueError("run_id must be a safe, immutable identifier")
        protocol, protocol_sha, _declared_at = self._load_protocol(protocol_id)
        package = self._load_package(package_id)
        input_path, relative = self._resolve_project_file(str(input_artifact_path))
        input_bytes = input_path.read_bytes()
        input_sha = _sha256_bytes(input_bytes)
        try:
            raw_input = json.loads(input_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raw_input = {}
        raw_records = raw_input.get("records", []) if isinstance(raw_input, Mapping) else []
        if not isinstance(raw_records, list):
            raw_records = []
        preliminary_views: list[dict[str, object]] = []
        for row in raw_records:
            if not isinstance(row, Mapping):
                continue
            try:
                prediction = _iso_day(row.get("prediction_date"), "prediction_date")
                maturity = (
                    _iso_day(row.get("label_matured_date"), "label_matured_date")
                    if row.get("label_matured_date") is not None else None
                )
            except ValueError:
                continue
            event_group = str(row.get("event_group_id", "")).strip()
            if event_group:
                preliminary_views.append({
                    "event_group_id": event_group,
                    "prediction_date": prediction.isoformat(),
                    "label_matured_date": maturity.isoformat() if maturity else None,
                })
        scope_sha256 = str(protocol["evaluation_scope_sha256"])
        now = self._now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM evaluation_attempts WHERE run_id=?", (run_id,)).fetchone():
                connection.rollback()
                raise ValueError("run_id is immutable and has already been attempted")
            prior_claim = connection.execute(
                "SELECT 1 FROM global_partition_claims WHERE evaluation_scope_sha256=? AND partition_name=?",
                (scope_sha256, partition),
            ).fetchone()
            prior_scope_views = connection.execute(
                "SELECT event_group_id,partition_name,prediction_date,label_matured_date FROM global_viewed_ranges "
                "WHERE evaluation_scope_sha256=?",
                (scope_sha256,),
            ).fetchall()
            connection.execute(
                "INSERT INTO evaluation_attempts(run_id,protocol_id,partition_name,package_id,input_path,input_sha256,status,created_at,report_json) "
                "VALUES(?,?,?,?,?,?,?,?,NULL)",
                (run_id, protocol_id, partition, package_id, relative, input_sha, "started", now),
            )
            if prior_claim is None:
                connection.execute(
                    "INSERT OR IGNORE INTO global_partition_claims VALUES(?,?,?,?)",
                    (scope_sha256, partition, protocol_id, run_id),
                )
            if partition in {"independent_acceptance", "forward_shadow"}:
                connection.execute(
                    "INSERT OR IGNORE INTO partition_claims VALUES(?,?,?)", (protocol_id, partition, run_id)
                )
            for view in preliminary_views:
                connection.execute(
                    "INSERT OR IGNORE INTO viewed_ranges VALUES(?,?,?,?,?,?)",
                    (protocol_id, run_id, partition, view["event_group_id"], view["prediction_date"], view["label_matured_date"]),
                )
                connection.execute(
                    "INSERT INTO global_viewed_ranges VALUES(?,?,?,?,?,?,?)",
                    (scope_sha256, protocol_id, run_id, partition, view["event_group_id"], view["prediction_date"], view["label_matured_date"]),
                )
            connection.commit()

        def reject_attempt(message: str) -> None:
            with closing(self._connect()) as connection:
                connection.execute("UPDATE evaluation_attempts SET status='rejected' WHERE run_id=?", (run_id,))
                connection.commit()
            raise ValueError(message)

        if prior_claim is not None:
            reject_attempt(f"{partition} has already been viewed/claimed for this frozen data scope")
        for view in preliminary_views:
            this_start = _iso_day(view["prediction_date"], "prediction_date")
            this_end = _iso_day(view["label_matured_date"], "label_matured_date") if view["label_matured_date"] else this_start
            for old in prior_scope_views:
                old_start = _iso_day(old["prediction_date"], "previous prediction date")
                old_end = _iso_day(old["label_matured_date"], "previous label maturity") if old["label_matured_date"] else old_start
                if old["event_group_id"] == view["event_group_id"] or max(this_start, old_start) <= min(this_end, old_end):
                    reject_attempt("an event group or prediction/label interval has already been viewed in this frozen data scope")
        try:
            report, views = self._compute_report(
                protocol_id=protocol_id,
                protocol=protocol,
                protocol_sha256=protocol_sha,
                package_id=package_id,
                package=package,
                partition=partition,
                run_id=run_id,
                input_path=str(input_path),
            )
        except Exception:
            with closing(self._connect()) as connection:
                connection.execute("UPDATE evaluation_attempts SET status='rejected' WHERE run_id=?", (run_id,))
                connection.commit()
            raise
        report_json = _json(report)
        report_sha = stable_hash(report)
        evidence_id = f"ev-{report_sha[:24]}"
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE evaluation_attempts SET status=?,report_json=? WHERE run_id=?",
                (str(report["evidence_status"]), report_json, run_id),
            )
            connection.execute(
                "INSERT INTO evidence_artifacts VALUES(?,?,?,?,?,?,?,?)",
                (evidence_id, run_id, protocol_id, partition, package_id, report_json, report_sha, self._now()),
            )
            connection.commit()
        return {"evidence_id": evidence_id, "evidence_sha256": report_sha, **report}

    def _verified_evidence(self, evidence_id: str) -> dict[str, object]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM evidence_artifacts WHERE evidence_id=?", (evidence_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"unknown registry evidence ID {evidence_id}")
        report = json.loads(row["report_json"])
        if stable_hash(report) != row["report_sha256"]:
            raise ValueError("registry evidence report failed its integrity hash")
        protocol, protocol_sha, _ = self._load_protocol(str(row["protocol_id"]))
        package = self._load_package(str(row["package_id"]))
        if protocol_sha != report["protocol_sha256"]:
            raise ValueError("registry evidence references changed protocol content")
        recomputed, _ = self._compute_report(
            protocol_id=str(row["protocol_id"]),
            protocol=protocol,
            protocol_sha256=protocol_sha,
            package_id=str(row["package_id"]),
            package=package,
            partition=str(row["partition_name"]),
            run_id=str(row["run_id"]),
            input_path=str(self.project_root / str(report["input_artifact_path"])),
        )
        if recomputed != report:
            raise ValueError("registry evidence no longer matches the source artifact and evaluator output")
        return report

    def publish_release(
        self,
        *,
        package_id: str,
        evidence_ids: Mapping[str, str],
        actor_role: str,
        reason: str,
        release_id: str | None = None,
    ) -> dict[str, object]:
        self._check_release_actor(actor_role)
        if not reason.strip():
            raise ValueError("publishing a release requires a reason")
        package = self._load_package(package_id)
        required = {"development", "selection", "independent_acceptance", "forward_shadow"}
        if set(evidence_ids) != required:
            raise ValueError("publish accepts only the four registry evidence IDs")
        reports = {name: self._verified_evidence(str(evidence_ids[name])) for name in required}
        if any(report["package_id"] != package_id for report in reports.values()):
            raise ValueError("release evidence and complete package IDs do not match")
        if any(report["test_only"] or report["fixture_only"] for report in reports.values()):
            if not self.test_mode:
                raise ValueError("fixture-only evidence cannot qualify for production")
        for name, report in reports.items():
            if report["partition"] != name:
                raise ValueError(f"{name} evidence ID references a different partition")
        acceptance = reports["independent_acceptance"]
        shadow = reports["forward_shadow"]
        if self.test_mode and not (
            all(bool(reports[name]["research_quality_passed"]) for name in ("development", "selection", "independent_acceptance"))
            and acceptance["economic_gate"] is not None
            and bool(acceptance["economic_gate"]["passed"])
            and bool(shadow["forward_shadow_checks_passed"])
        ):
            raise ValueError("computed research/economic/shadow gates did not pass, even in test-only mode")
        if not self.test_mode:
            if any(not report["production_eligible"] for report in reports.values()):
                raise ValueError("release is pending: verified non-fixture evidence has not passed every gate")
            if acceptance["economic_gate"] is None or acceptance["economic_gate"]["status"] != "passed":
                raise ValueError("production release requires a computed cost-improved, non-inferior, risk-passing evaluation")
            if shadow["evidence_status"] != "passed":
                raise ValueError("production release requires observed forward-shadow evidence; status remains pending_forward")

        test_only_bundle = self.test_mode or any(report["fixture_only"] for report in reports.values())
        bundle = {
            "schema_version": "adaptive-mas.release-bundle.v2",
            "package_id": package_id,
            "package_sha256": stable_hash(package),
            "complete_runtime_package": package,
            "evidence_ids": dict(evidence_ids),
            "evidence_reports": reports,
            "evidence_status": {
                "engineering": "complete_compatible_bundle",
                "research_quality": (
                    "test_only_fixture_gate"
                    if test_only_bundle else "independently_accepted"
                ),
                "economic_value": (
                    "test_only_gate_calculated"
                    if test_only_bundle and acceptance["economic_gate"] and acceptance["economic_gate"]["passed"]
                    else "test_only_gate_failed"
                    if test_only_bundle else "cost_improved_effect_noninferior_risk_pass"
                ),
                "forward_evidence": shadow["evidence_status"],
                "scientific_claim": "not_established",
            },
            "test_only": test_only_bundle,
            "publication_reason": reason,
            "published_at": self._now(),
        }
        digest = stable_hash(bundle)
        release_id = release_id or f"r-{digest[:20]}"
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", release_id):
            raise ValueError("release_id must be a safe identifier")
        encoded = _json(bundle)
        now = self._now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM release_bundles WHERE release_id=?", (release_id,)).fetchone():
                connection.rollback()
                raise ValueError(f"release {release_id!r} already exists")
            connection.execute("INSERT INTO release_bundles VALUES (?,?,?,?)", (release_id, encoded, digest, now))
            connection.execute(
                "INSERT INTO production_pointer(singleton,release_id,updated_at) VALUES(1,?,?) "
                "ON CONFLICT(singleton) DO UPDATE SET release_id=excluded.release_id,updated_at=excluded.updated_at",
                (release_id, now),
            )
            details = _json({"package_id": package_id, "bundle_sha256": digest, "reason": reason})
            connection.execute(
                "INSERT INTO release_audit(action,actor_role,release_id,details_json,created_at) VALUES('publish',?,?,?,?)",
                (actor_role, release_id, details, now),
            )
            connection.commit()
        return {"release_id": release_id, "bundle_sha256": digest, **bundle}

    def _load_bundle(self, release_id: str) -> dict[str, object]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT bundle_json,bundle_sha256 FROM release_bundles WHERE release_id=?", (release_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"unknown release bundle {release_id!r}")
        bundle = json.loads(row["bundle_json"])
        if stable_hash(bundle) != row["bundle_sha256"]:
            raise ValueError("release bundle failed its integrity hash")
        if bundle.get("schema_version") != "adaptive-mas.release-bundle.v2":
            raise ValueError("release bundle schema is unsupported")
        if bundle.get("test_only") and not self.test_mode:
            raise ValueError("test-only evidence bundles cannot be loaded by a production store")
        package = self._load_package(str(bundle["package_id"]))
        if package != bundle["complete_runtime_package"] or stable_hash(package) != bundle["package_sha256"]:
            raise ValueError("release bundle does not contain the complete registered runtime package")
        reports = {
            name: self._verified_evidence(str(evidence_id))
            for name, evidence_id in bundle["evidence_ids"].items()
        }
        if reports != bundle["evidence_reports"]:
            raise ValueError("release evidence no longer matches the immutable bundle")
        return bundle

    def load_active(
        self,
        *,
        config: Mapping[str, object],
        training_state: Mapping[str, object] | None = None,
        training_seed_provenance: Mapping[str, object] | None = None,
        versions: Mapping[str, object] | None = None,
    ) -> tuple[dict[str, str], str, dict[str, object]]:
        with closing(self._connect()) as connection:
            pointer = connection.execute("SELECT release_id FROM production_pointer WHERE singleton=1").fetchone()
        if pointer is None or pointer["release_id"] is None:
            return seed_candidate(), "seed", {
                "status": "no_accepted_production_bundle",
                "procedural_memory": {
                    "status": "seed_defaults",
                    "loaded_from_persistent_store": False,
                    "updated_from_prior_task": False,
                },
            }
        release_id = str(pointer["release_id"])
        bundle = self._load_bundle(release_id)
        package = bundle["complete_runtime_package"]
        if stable_hash(dict(config)) != package["config_sha256"]:
            raise ValueError("production release bundle is incompatible with the requested configuration")
        runtime = package["runtime_context"]
        if training_state is not None and stable_hash(dict(training_state)) != stable_hash(runtime["training_state"]):
            raise ValueError("production release bundle is incompatible with the active training state")
        if training_seed_provenance is not None and stable_hash(dict(training_seed_provenance)) != stable_hash(runtime["training_seed_provenance"]):
            raise ValueError("production release bundle is incompatible with training seed provenance")
        if versions is not None and stable_hash(dict(versions)) != stable_hash(runtime["versions"]):
            raise ValueError("production release bundle is incompatible with data/model versions")
        implementation = source_package_identity()
        if package["implementation_identity"] != implementation:
            raise ValueError("production release bundle is incompatible with the current implementation")
        candidate = validate_candidate(package["candidate"])
        version = {
            **bundle,
            "version_id": package["candidate_version_id"],
            "origin": "verified_complete_release_bundle",
            "restored_package": package,
        }
        return candidate, str(package["candidate_version_id"]), version

    def rollback(
        self,
        release_id: str,
        *,
        config: Mapping[str, object],
        actor_role: str,
        reason: str,
        training_state: Mapping[str, object] | None = None,
        training_seed_provenance: Mapping[str, object] | None = None,
        versions: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self._check_release_actor(actor_role)
        if not reason.strip():
            raise ValueError("rollback requires a reason")
        if release_id != "seed":
            bundle = self._load_bundle(release_id)
            package = bundle["complete_runtime_package"]
            if stable_hash(dict(config)) != package["config_sha256"]:
                raise ValueError("rollback target release bundle is incompatible with the requested configuration")
            runtime = package["runtime_context"]
            for label, actual, expected in (
                ("training state", training_state, runtime["training_state"]),
                ("training seed provenance", training_seed_provenance, runtime["training_seed_provenance"]),
                ("data/model versions", versions, runtime["versions"]),
            ):
                if actual is not None and stable_hash(dict(actual)) != stable_hash(expected):
                    raise ValueError(f"rollback target release bundle is incompatible with {label}")
        now = self._now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT release_id FROM production_pointer WHERE singleton=1").fetchone()
            previous_id = previous["release_id"] if previous else None
            target_id = None if release_id == "seed" else release_id
            connection.execute(
                "INSERT INTO production_pointer(singleton,release_id,updated_at) VALUES(1,?,?) "
                "ON CONFLICT(singleton) DO UPDATE SET release_id=excluded.release_id,updated_at=excluded.updated_at",
                (target_id, now),
            )
            details = _json({"from_release_id": previous_id or "seed", "to_release_id": release_id, "reason": reason})
            connection.execute(
                "INSERT INTO release_audit(action,actor_role,release_id,details_json,created_at) VALUES('rollback',?,?,?,?)",
                (actor_role, release_id, details, now),
            )
            connection.commit()
        restored = None
        if release_id != "seed":
            loaded, version_id, active = self.load_active(
                config=config,
                training_state=training_state,
                training_seed_provenance=training_seed_provenance,
                versions=versions,
            )
            restored = {
                "candidate_sha256": _sha256_text(_json(loaded)),
                "candidate_version_id": version_id,
                "complete_package_sha256": active["package_sha256"],
            }
        return {
            "from_release_id": previous_id or "seed",
            "to_release_id": release_id,
            "reason": reason,
            "audit_recorded": True,
            "restored_bundle": restored,
        }

    def status_summary(self) -> dict[str, object]:
        with closing(self._connect()) as connection:
            pointer = connection.execute("SELECT release_id,updated_at FROM production_pointer WHERE singleton=1").fetchone()
        if pointer is None or pointer["release_id"] is None:
            return {"status": "seed", "release_id": "seed", "reason": "no accepted production bundle"}
        bundle = self._load_bundle(str(pointer["release_id"]))
        package = bundle["complete_runtime_package"]
        return {
            "status": "test_only" if bundle["test_only"] else "production",
            "release_id": pointer["release_id"],
            "candidate_version_id": package["candidate_version_id"],
            "candidate_sha256": package["candidate_sha256"],
            "package_sha256": bundle["package_sha256"],
            "evidence_status": bundle["evidence_status"],
            "updated_at": pointer["updated_at"],
        }

    def audit_history(self) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT event_id,action,actor_role,release_id,details_json,created_at FROM release_audit ORDER BY event_id"
            ).fetchall()
        return [
            {
                "event_id": int(row["event_id"]),
                "action": str(row["action"]),
                "actor_role": str(row["actor_role"]),
                "release_id": row["release_id"],
                "details": json.loads(row["details_json"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def evaluation_status(self) -> dict[str, object]:
        """Return compact registry status for the local workbench."""
        with closing(self._connect()) as connection:
            protocols = connection.execute(
                "SELECT protocol_id,protocol_sha256,declared_at FROM evaluation_protocols ORDER BY declared_at"
            ).fetchall()
            evidence = connection.execute(
                "SELECT evidence_id,run_id,protocol_id,partition_name,package_id,report_json,created_at "
                "FROM evidence_artifacts ORDER BY created_at"
            ).fetchall()
            packages = connection.execute(
                "SELECT package_id,package_sha256,created_at FROM package_artifacts ORDER BY created_at"
            ).fetchall()
        return {
            "protocols": [
                {"protocol_id": row["protocol_id"], "protocol_sha256": row["protocol_sha256"], "declared_at": row["declared_at"]}
                for row in protocols
            ],
            "packages": [
                {"package_id": row["package_id"], "package_sha256": row["package_sha256"], "created_at": row["created_at"]}
                for row in packages
            ],
            "evidence": [
                {
                    "evidence_id": row["evidence_id"],
                    "run_id": row["run_id"],
                    "protocol_id": row["protocol_id"],
                    "partition": row["partition_name"],
                    "package_id": row["package_id"],
                    "created_at": row["created_at"],
                    "evidence_status": json.loads(row["report_json"]).get("evidence_status"),
                    "research_quality_passed": json.loads(row["report_json"]).get("research_quality_passed"),
                    "economic_gate": json.loads(row["report_json"]).get("economic_gate"),
                    "forward_shadow_checks_passed": json.loads(row["report_json"]).get("forward_shadow_checks_passed"),
                }
                for row in evidence
            ],
        }
