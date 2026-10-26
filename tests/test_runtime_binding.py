from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from adaptive_mas.agents.harness_prompts import seed_candidate
from adaptive_mas.runtime_binding import (
    _path_digest,
    build_workflow_runtime_binding,
    validate_workflow_runtime_binding,
)
from adaptive_mas.release_governance import HarnessReleaseStore
from adaptive_mas.workflow_snapshot import materialize_workflow_snapshot


def _workflow_fixture(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    inputs = {
        "baostock_run_dir": root / "data" / "raw" / "baostock" / "cache",
        "config": root / "configs" / "daily.toml",
        "public_cache_dir": root / "data" / "raw" / "akshare",
        "reuse_research_json": root / "experiments" / "reuse.json",
        "training_seed": root / "data" / "processed" / "seed.json",
    }
    for key, path in inputs.items():
        if key in {"baostock_run_dir", "public_cache_dir"}:
            path.mkdir(parents=True, exist_ok=True)
            (path / "cached.json").write_text(f"{key}-fixture", encoding="utf-8")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{key}-fixture", encoding="utf-8")
    request = {key: str(path) for key, path in inputs.items()}
    return request, request


def test_workflow_binding_rejects_changed_config_directory_and_binding_hash(tmp_path: Path) -> None:
    request, paths = _workflow_fixture(tmp_path)
    binding = build_workflow_runtime_binding(
        request,
        project_root=tmp_path,
        runtime_package={"package_id": "seed", "package_sha256": None},
    )

    assert validate_workflow_runtime_binding(
        binding, project_root=tmp_path, request_paths=paths,
    ) == binding["binding_sha256"]

    config_path = Path(paths["config"])
    config_path.write_text("changed config", encoding="utf-8")
    with pytest.raises(ValueError, match="config changed"):
        validate_workflow_runtime_binding(binding, project_root=tmp_path, request_paths=paths)
    config_path.write_text("config-fixture", encoding="utf-8")

    changed_cache = Path(paths["public_cache_dir"]) / "new.json"
    changed_cache.write_text("new cached input", encoding="utf-8")
    with pytest.raises(ValueError, match="public_cache_dir changed"):
        validate_workflow_runtime_binding(binding, project_root=tmp_path, request_paths=paths)

    tampered = dict(binding)
    tampered["runtime_package"] = {"package_id": "different", "package_sha256": "f" * 64}
    with pytest.raises(ValueError, match="content hash"):
        validate_workflow_runtime_binding(tampered, project_root=tmp_path, request_paths=paths)


def test_workflow_directory_binding_ignores_test_csv_without_opening_it(tmp_path: Path, monkeypatch) -> None:
    request, _ = _workflow_fixture(tmp_path)
    cache_dir = Path(request["public_cache_dir"])
    test_csv = cache_dir / "test.csv"
    test_csv.write_text("first", encoding="utf-8")
    test_csv.write_text("different and intentionally unread", encoding="utf-8")
    original_open = Path.open

    def reject_test_csv_open(path: Path, *args, **kwargs):
        if path.name.lower() == "test.csv":
            raise AssertionError("test.csv must never be opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_test_csv_open)
    first = _path_digest(cache_dir)
    second = _path_digest(cache_dir)

    assert first == second
    with pytest.raises(ValueError, match="test.csv"):
        _path_digest(test_csv)


def test_workflow_directory_binding_rejects_symlink_targets(tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "cache"
    directory.mkdir()
    (directory / "linked.json").write_text("target data", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def mark_link(path: Path) -> bool:
        return path.name == "linked.json" or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", mark_link)
    with pytest.raises(ValueError, match="unbound symbolic link"):
        _path_digest(directory)


def test_workflow_file_digest_streams_without_read_bytes(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "large.bin"
    payload = b"bound-data" * 200_000
    target.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    def reject_read_bytes(_path: Path) -> bytes:
        raise AssertionError("workflow binding must hash input files incrementally")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    assert _path_digest(target) == ("file", expected)


def test_workflow_snapshot_restores_queued_inputs_and_rejects_snapshot_tampering(tmp_path: Path, monkeypatch) -> None:
    request, _ = _workflow_fixture(tmp_path)
    store_root = tmp_path / "harness_versions"
    release_store = HarnessReleaseStore(store_root, project_root=tmp_path)
    artifact_names = (
        "training_seed", "training_state", "calibration", "model_versions", "data_versions", "tool_schema",
    )
    registered = release_store.register_package({
        "candidate_version_id": "fixture-runtime-v1",
        "candidate": seed_candidate(),
        "config": {"prediction_horizon_sessions": 3},
        "runtime_context": {
            "training_state": {"mature_rows": [1]},
            "training_seed_provenance": {"trained_through": "2025-01-01"},
            "versions": {"config_sha256": "a" * 64},
        },
        "artifacts": {
            name: {"version": f"{name}-v1", "payload": {"fixture": name}}
            for name in artifact_names
        },
    })
    runtime_package = {
        "package_id": registered["package_id"],
        "package_sha256": registered["package_sha256"],
    }
    snapshot = materialize_workflow_snapshot(
        request,
        project_root=tmp_path,
        snapshot_root=tmp_path / "service_artifacts" / "snapshot-task" / "workflow_snapshot",
        runtime_package=runtime_package,
        runtime_package_store_root=store_root,
    )
    binding = build_workflow_runtime_binding(
        request,
        project_root=tmp_path,
        runtime_package=runtime_package,
        materialization=snapshot,
    )
    snapshot_paths = {
        key: str(descriptor["path"])
        for key, descriptor in snapshot["inputs"].items()
    }
    copied_config = Path(snapshot_paths["config"])
    original_config = Path(request["config"])
    assert all(len(str(descriptor["sha256"])) == 64 for descriptor in snapshot["inputs"].values())
    assert len(str(snapshot["runtime_registry_sha256"])) == 64

    original_config.write_text("changed after task submission", encoding="utf-8")
    assert validate_workflow_runtime_binding(
        binding, project_root=tmp_path, request_paths=snapshot_paths,
    ) == binding["binding_sha256"]
    assert copied_config.read_text(encoding="utf-8") == "config-fixture"
    copied_source_root = Path(snapshot["source_root"])
    assert (copied_source_root / "scripts" / "run_harness_decision.py").is_file()

    monkeypatch.setattr("adaptive_mas.workflow_snapshot._runtime_environment", lambda: {})
    with pytest.raises(ValueError, match="Python runtime environment differs"):
        validate_workflow_runtime_binding(
            binding, project_root=tmp_path, request_paths=snapshot_paths,
        )
    monkeypatch.undo()

    copied_config.write_text("tampered queued snapshot", encoding="utf-8")
    with pytest.raises(ValueError, match="materialized input failed integrity verification: config"):
        validate_workflow_runtime_binding(
            binding, project_root=tmp_path, request_paths=snapshot_paths,
        )
    copied_config.write_text("config-fixture", encoding="utf-8")
    copied_entrypoint = copied_source_root / "scripts" / "run_harness_decision.py"
    copied_entrypoint.write_text(copied_entrypoint.read_text(encoding="utf-8") + "\n# tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="workflow source snapshot failed integrity verification"):
        validate_workflow_runtime_binding(
            binding, project_root=tmp_path, request_paths=snapshot_paths,
        )
