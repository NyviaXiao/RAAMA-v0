from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from adaptive_mas.runtime_binding import (
    _path_digest,
    build_workflow_runtime_binding,
    validate_workflow_runtime_binding,
)


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
