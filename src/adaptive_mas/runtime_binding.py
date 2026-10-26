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
) -> dict[str, object]:
    root = Path(project_root).resolve()
    inputs = {}
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
        inputs[key] = {"path": str(path), "kind": kind, "sha256": digest}
    body = {
        "schema_version": _BINDING_SCHEMA,
        "implementation_identity": source_package_identity(),
        "inputs": inputs,
        "runtime_package": dict(runtime_package),
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
