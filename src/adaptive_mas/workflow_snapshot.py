"""Local, content-verified materialization for queued offline workflows."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import sys
import uuid
from contextlib import closing
from collections.abc import Mapping
from pathlib import Path

from adaptive_mas.package_identity import source_package_identity
from adaptive_mas.research_provenance import stable_hash
from adaptive_mas.runtime_binding import _file_digest, _path_digest


_INPUT_KEYS = (
    "baostock_run_dir",
    "config",
    "public_cache_dir",
    "reuse_research_json",
    "training_seed",
)
_SNAPSHOT_MANIFEST = ".workflow_snapshot.json"
_CHUNK_BYTES = 1024 * 1024


def _copy_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as source_stream, destination.open("xb") as target_stream:
        for chunk in iter(lambda: source_stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
            target_stream.write(chunk)
    return digest.hexdigest()


def _copy_input(source: Path, destination: Path) -> tuple[str, str]:
    kind, source_digest = _path_digest(source)
    if kind == "file":
        copied_digest = _copy_file(source, destination / source.name)
        materialized_path = destination / source.name
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
            if child.name.lower() == "test.csv":
                continue
            if child.is_symlink():
                raise ValueError(f"workflow input directory contains an unbound symbolic link: {child.relative_to(source)}")
            if child.is_file():
                _copy_file(child, destination / child.relative_to(source))
        copied_digest = _path_digest(destination)[1]
        materialized_path = destination
    if copied_digest != source_digest:
        raise ValueError(f"workflow input changed while its queued snapshot was being materialized: {source.name}")
    return kind, copied_digest


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise ValueError(f"workflow input is not a readable file or directory: {path.name}")
    total = 0
    for child in path.rglob("*"):
        if child.name.lower() == "test.csv":
            continue
        if child.is_symlink():
            raise ValueError(f"workflow input directory contains an unbound symbolic link: {child.relative_to(path)}")
        if child.is_file():
            total += child.stat().st_size
    return total


def _runtime_environment() -> dict[str, object]:
    packages = sorted(
        [str(distribution.metadata.get("Name", "")).casefold(), str(distribution.version)]
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    return {
        "python_version": sys.version,
        "python_implementation": sys.implementation.name,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_base_prefix": str(Path(sys.base_prefix).resolve()),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "installed_distributions": packages,
    }


def _resolve_inside(path: str | Path, root: Path, label: str) -> Path:
    raw = Path(path)
    lexical = Path(os.path.abspath(raw if raw.is_absolute() else root / raw))
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"workflow {label} must stay inside the configured project root") from exc
    current = lexical
    while current != root:
        if current.is_symlink():
            raise ValueError(f"workflow {label} cannot traverse a symbolic link")
        if current == current.parent:
            raise ValueError(f"workflow {label} must stay inside the configured project root")
        current = current.parent
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"workflow {label} must stay inside the configured project root") from exc
    return resolved


def materialize_workflow_snapshot(
    request: Mapping[str, object],
    *,
    project_root: str | Path,
    snapshot_root: str | Path,
    runtime_package: Mapping[str, object],
    runtime_package_store_root: str | Path,
) -> dict[str, object]:
    """Capture source, input files and the pinned release registry in an immutable task folder."""
    root = Path(project_root).resolve()
    final_root = _resolve_inside(snapshot_root, root, "snapshot")
    if final_root.exists():
        raise ValueError("workflow snapshot path already exists")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary sibling name compact. Snapshot copies include nested
    # source paths, and a descriptive suffix can push valid Windows workspace
    # paths past the legacy MAX_PATH boundary before the final rename.
    staging_root = final_root.parent / f".ws-{uuid.uuid4().hex[:24]}"
    staging_root.mkdir(parents=False, exist_ok=False)
    implementation_identity = source_package_identity()
    package_root = Path(__file__).resolve().parent
    source_repo_root = package_root.parent.parent

    try:
        input_sources = {
            key: _resolve_inside(str(request[key]), root, f"{key} input")
            for key in _INPUT_KEYS
            if isinstance(request.get(key), str)
        }
        if set(input_sources) != set(_INPUT_KEYS):
            raise ValueError("workflow request is missing a required input for snapshotting")
        source_registry_root = _resolve_inside(runtime_package_store_root, root, "runtime package registry")
        source_registry_db = source_registry_root / "release_governance.sqlite3"
        if source_registry_db.is_symlink() or not source_registry_db.is_file():
            raise FileNotFoundError("pinned runtime package registry database is unavailable")
        source_paths = [
            source_repo_root / str(relative)
            for relative in implementation_identity["implementation_files_sha256"]
        ]
        if any(path.is_symlink() or not path.is_file() for path in source_paths):
            raise ValueError("workflow implementation contains an unavailable or linked source file")
        required_bytes = sum(path.stat().st_size for path in source_paths)
        required_bytes += sum(_path_size(path) for path in input_sources.values())
        required_bytes += source_registry_db.stat().st_size
        if shutil.disk_usage(final_root.parent).free < required_bytes:
            raise OSError("insufficient free space to materialize the queued workflow snapshot")

        staging_source = staging_root / "source"
        staging_source.mkdir()
        source_files: dict[str, str] = {}
        for relative, expected_digest in sorted(implementation_identity["implementation_files_sha256"].items()):
            source = source_repo_root / str(relative)
            destination = staging_source / str(relative)
            copied_digest = _copy_file(source, destination)
            if copied_digest != expected_digest:
                raise ValueError(f"workflow implementation changed while snapshotting: {relative}")
            source_files[str(relative)] = copied_digest

        input_records: dict[str, dict[str, object]] = {}
        for key in _INPUT_KEYS:
            raw = request.get(key)
            if not isinstance(raw, str):
                raise ValueError(f"workflow request is missing its {key} input")
            original = input_sources[key]
            staging_path = staging_root / "inputs" / key
            kind, digest = _copy_input(original, staging_path)
            materialized_path = staging_path / original.name if kind == "file" else staging_path
            input_records[key] = {
                "source_path": str(original),
                "path": str(final_root / materialized_path.relative_to(staging_root)),
                "kind": kind,
                "sha256": digest,
            }

        staging_registry_root = staging_root / "runtime_package_store"
        staging_registry_root.mkdir()
        staging_registry_db = staging_registry_root / source_registry_db.name
        with closing(sqlite3.connect(source_registry_db)) as source_connection, closing(sqlite3.connect(staging_registry_db)) as target_connection:
            source_connection.backup(target_connection)
        from adaptive_mas.release_governance import HarnessReleaseStore

        copied_store = HarnessReleaseStore(
            staging_registry_root,
            project_root=staging_source,
        )
        if runtime_package.get("package_id") == "seed":
            if runtime_package.get("package_sha256") is not None:
                raise ValueError("seed runtime package cannot have a package hash")
        else:
            package = copied_store._load_package(str(runtime_package.get("package_id")))
            if stable_hash(package) != runtime_package.get("package_sha256"):
                raise ValueError("pinned runtime package changed while its registry was being materialized")
        registry_digest = _file_digest(staging_registry_db)

        runtime_environment = _runtime_environment()
        body = {
            "schema_version": "adaptive-mas.workflow-materialization.v1",
            "snapshot_root": str(final_root),
            "implementation_identity": implementation_identity,
            "implementation_files": source_files,
            "source_root": str(final_root / "source"),
            "inputs": input_records,
            "runtime_package": dict(runtime_package),
            "runtime_package_store_root": str(final_root / "runtime_package_store"),
            "runtime_registry_sha256": registry_digest,
            "runtime_environment": runtime_environment,
            "runtime_environment_sha256": stable_hash(runtime_environment),
        }
        body["snapshot_sha256"] = stable_hash(body)
        (staging_root / _SNAPSHOT_MANIFEST).write_text(
            json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_root, final_root)
        return json.loads(json.dumps(body, ensure_ascii=False, sort_keys=True))
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


def verify_workflow_materialization(
    materialization: Mapping[str, object],
    *,
    project_root: str | Path,
) -> None:
    """Verify every captured byte and the compatible local Python environment before replay."""
    root = Path(project_root).resolve()
    raw_snapshot_root = Path(str(materialization.get("snapshot_root", "")))
    if raw_snapshot_root.is_symlink():
        raise ValueError("workflow snapshot root cannot be a symbolic link")
    snapshot_root = _resolve_inside(raw_snapshot_root, root, "snapshot")
    manifest_path = snapshot_root / _SNAPSHOT_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("workflow materialization manifest is missing or linked")
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dict(materialization) != stored:
        raise ValueError("workflow materialization descriptor differs from its stored manifest")
    body = {key: value for key, value in stored.items() if key != "snapshot_sha256"}
    if stored.get("snapshot_sha256") != stable_hash(body) or stored.get("snapshot_sha256") != materialization.get("snapshot_sha256"):
        raise ValueError("workflow materialization manifest failed its content hash")
    if body.get("schema_version") != "adaptive-mas.workflow-materialization.v1":
        raise ValueError("workflow materialization schema is unsupported")
    if body.get("implementation_identity") != materialization.get("implementation_identity"):
        raise ValueError("workflow implementation snapshot differs from its queued manifest")
    source_root = _resolve_inside(str(body["source_root"]), root, "source snapshot")
    if source_root != snapshot_root / "source":
        raise ValueError("workflow source snapshot is outside its immutable task directory")
    source_files = body.get("implementation_files")
    if not isinstance(source_files, Mapping):
        raise ValueError("workflow source snapshot has no file manifest")
    identity = body.get("implementation_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("implementation_files_sha256") != dict(source_files)
    ):
        raise ValueError("workflow source file hashes differ from their implementation identity")
    for relative, expected in source_files.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("workflow source snapshot contains an invalid relative path")
        raw_file_path = source_root / relative_path
        if raw_file_path.is_symlink():
            raise ValueError(f"workflow source snapshot contains a symbolic link: {relative}")
        file_path = _resolve_inside(raw_file_path, root, "source file")
        try:
            file_path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError("workflow source file is outside its immutable code snapshot") from exc
        if not file_path.is_file() or _file_digest(file_path) != expected:
            raise ValueError(f"workflow source snapshot failed integrity verification: {relative}")
    inputs = body.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(_INPUT_KEYS):
        raise ValueError("workflow materialized inputs are incomplete")
    for key, descriptor in inputs.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"workflow materialized input descriptor is invalid: {key}")
        raw_input_path = Path(str(descriptor.get("path", "")))
        if raw_input_path.is_symlink():
            raise ValueError(f"workflow input snapshot cannot be a symbolic link: {key}")
        input_path = _resolve_inside(raw_input_path, root, f"{key} snapshot")
        try:
            input_path.relative_to(snapshot_root / "inputs")
        except ValueError as exc:
            raise ValueError(f"workflow {key} is outside its immutable task directory") from exc
        kind, digest = _path_digest(input_path)
        if kind != descriptor.get("kind") or digest != descriptor.get("sha256"):
            raise ValueError(f"workflow materialized input failed integrity verification: {key}")
    raw_registry_root = Path(str(body["runtime_package_store_root"]))
    if raw_registry_root.is_symlink():
        raise ValueError("workflow runtime package registry cannot be a symbolic link")
    registry_root = _resolve_inside(raw_registry_root, root, "runtime package registry")
    if registry_root != snapshot_root / "runtime_package_store":
        raise ValueError("workflow runtime package registry is outside its immutable task directory")
    registry_db = registry_root / "release_governance.sqlite3"
    if registry_db.is_symlink() or not registry_db.is_file() or _file_digest(registry_db) != body.get("runtime_registry_sha256"):
        raise ValueError("workflow runtime package registry failed integrity verification")
    runtime_package = body.get("runtime_package")
    if materialization.get("runtime_package") != runtime_package:
        raise ValueError("workflow pinned runtime package differs from its materialization")
    if _runtime_environment() != body.get("runtime_environment"):
        raise ValueError("workflow Python runtime environment differs from the queued snapshot")
    if stable_hash(_runtime_environment()) != body.get("runtime_environment_sha256"):
        raise ValueError("workflow Python runtime environment failed its identity hash")
