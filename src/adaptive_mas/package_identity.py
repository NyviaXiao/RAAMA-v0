"""Stable hashes for the runtime source, tool/evaluator contract and dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()


def source_package_identity() -> dict[str, object]:
    package_root = Path(__file__).parent
    repo_root = package_root.parent.parent
    source_files = set(package_root.rglob("*.py"))
    entrypoint_files = {
        repo_root / "scripts" / "run_harness_decision.py",
        repo_root / "scripts" / "run_harness_evolution.py",
    }
    dependency_files = (repo_root / "pyproject.toml", repo_root / "uv.lock")
    schema_files = set((repo_root / "schemas").rglob("*.json"))
    all_files = sorted(
        [path for path in source_files | entrypoint_files | schema_files if path.is_file()]
        + [path for path in dependency_files if path.is_file()]
    )
    hashes = {
        path.relative_to(repo_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in all_files
    }
    tool_path = "src/adaptive_mas/agents/harness.py"
    evaluator_path = "src/adaptive_mas/harness_evolution.py"
    dependency_hashes = {
        path.name: hashes[path.relative_to(repo_root).as_posix()]
        for path in dependency_files if path.is_file()
    }
    return {
        "implementation_files_sha256": hashes,
        "implementation_package_sha256": _stable_hash(hashes),
        "tool_schema_sha256": hashes[tool_path],
        "evaluator_sha256": hashes[evaluator_path],
        "dependency_lock_sha256": _stable_hash(dependency_hashes),
    }
