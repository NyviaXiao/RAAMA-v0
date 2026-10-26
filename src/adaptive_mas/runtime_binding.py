"""Immutable identity checks for queued harness decision inputs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from adaptive_mas.package_identity import source_package_identity
from adaptive_mas.research_provenance import stable_hash


_BINDING_SCHEMA = "adaptive-mas.workflow-runtime-binding.v1"
_INPUT_KEYS = (
    "baostock_run_dir",
    "config",
    "public_cache_dir",
    "reuse_research_json",
    "training_seed",
)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> tuple[str, str]:
    if path.name.lower() == "test.csv":
        raise ValueError("test.csv cannot be bound as a workflow input")
    if path.is_file():
        return "file", _file_digest(path)
    if not path.is_dir():
        raise ValueError(f"workflow input is not a readable file or directory: {path.name}")
    entries = []
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.name.lower() == "test.csv":
            continue
        if child.is_symlink():
            raise ValueError(f"workflow input directory contains an unbound symbolic link: {child.relative_to(path)}")
        if child.is_file():
            entries.append({
                "path": child.relative_to(path).as_posix(),
                "sha256": _file_digest(child),
            })
    return "directory", stable_hash(entries)


def build_workflow_runtime_binding(
    request: Mapping[str, object],
    *,
    project_root: str | Path,
    runtime_package: Mapping[str, object],
    materialization: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    inputs: dict[str, object] = {}
    if materialization is not None:
        if materialization.get("implementation_identity") != source_package_identity():
            raise ValueError("materialized implementation differs from the source being queued")
        if materialization.get("runtime_package") != dict(runtime_package):
            raise ValueError("materialized runtime package differs from the queued package pin")
        materialized_inputs = materialization.get("inputs")
        if not isinstance(materialized_inputs, Mapping) or set(materialized_inputs) != set(_INPUT_KEYS):
            raise ValueError("workflow materialization does not include every required input")
        for key in _INPUT_KEYS:
            descriptor = materialized_inputs[key]
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"workflow materialized input descriptor is invalid: {key}")
            path = Path(str(descriptor.get("path"))).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError("workflow snapshots must stay inside the configured project root") from exc
            inputs[key] = dict(descriptor)
    else:
        for key in _INPUT_KEYS:
            raw = request.get(key)
            if not isinstance(raw, str):
                raise ValueError(f"workflow request is missing its {key} input")
            path = Path(raw).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError("workflow inputs must stay inside the configured project root") from exc
            kind, digest = _path_digest(path)
            inputs[key] = {"path": str(path), "source_path": str(path), "kind": kind, "sha256": digest}
    body = {
        "schema_version": _BINDING_SCHEMA,
        "implementation_identity": source_package_identity(),
        "inputs": inputs,
        "runtime_package": dict(runtime_package),
        "materialization": dict(materialization) if materialization is not None else None,
    }
    return {**body, "binding_sha256": stable_hash(body)}


def validate_workflow_runtime_binding(
    binding: Mapping[str, object],
    *,
    project_root: str | Path,
    request_paths: Mapping[str, str | Path],
) -> str:
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if (
        binding.get("schema_version") != _BINDING_SCHEMA
        or binding.get("binding_sha256") != stable_hash(body)
    ):
        raise ValueError("workflow runtime binding failed its content hash")
    if binding.get("implementation_identity") != source_package_identity():
        raise ValueError("workflow implementation changed after this task was queued")
    inputs = binding.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(_INPUT_KEYS):
        raise ValueError("workflow runtime binding has an unsupported input manifest")
    root = Path(project_root).resolve()
    materialization = binding.get("materialization")
    if materialization is not None:
        if not isinstance(materialization, Mapping):
            raise ValueError("workflow materialization descriptor is invalid")
        if materialization.get("runtime_package") != binding.get("runtime_package"):
            raise ValueError("workflow materialized runtime package differs from the queued package pin")
        from adaptive_mas.workflow_snapshot import verify_workflow_materialization

        verify_workflow_materialization(materialization, project_root=root)
    for key in _INPUT_KEYS:
        descriptor = inputs[key]
        actual = Path(request_paths[key]).resolve()
        if not isinstance(descriptor, Mapping) or Path(str(descriptor.get("path"))).resolve() != actual:
            raise ValueError(f"workflow {key} path differs from the queued snapshot")
        try:
            actual.relative_to(root)
        except ValueError as exc:
            raise ValueError("workflow inputs must stay inside the configured project root") from exc
        kind, digest = _path_digest(actual)
        if descriptor.get("kind") != kind or descriptor.get("sha256") != digest:
            raise ValueError(f"workflow {key} changed after this task was queued")
    return str(binding["binding_sha256"])
