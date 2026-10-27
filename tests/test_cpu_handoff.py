from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_mas.cpu_handoff import (
    THREAD_ENVIRONMENT,
    evaluate_resource_limits,
    inspect_cpu_environment,
    inspect_linux_cgroup,
    parse_cpu_env_file,
    parse_cpu_max,
    parse_cpuset,
    parse_finite_cgroup_limit,
)


def test_cgroup_cpu_parsers_handle_linux_formats() -> None:
    assert parse_cpu_max("2000000 100000") == 20.0
    assert parse_cpuset("0-3,5,8-9") == {0, 1, 2, 3, 5, 8, 9}
    assert parse_finite_cgroup_limit("180000000000", "memory.high") == 180_000_000_000


@pytest.mark.parametrize("value", ["max 100000", "0 100000", "1000 0", "1", "1 2 3"])
def test_cpu_quota_parser_rejects_unbounded_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_cpu_max(value)


@pytest.mark.parametrize("value", ["", "1,,2", "3-1", "1-2-3", "a"])
def test_cpuset_parser_rejects_invalid_ranges(value: str) -> None:
    with pytest.raises(ValueError):
        parse_cpuset(value)


def test_resource_matrix_passes_only_inside_cpu_memory_swap_and_task_budgets() -> None:
    valid = {
        "cgroup_path": "/raama.slice/workbench.service",
        "cpu_quota_cores": 20.0,
        "cpuset_cpus": 20,
        "memory_high_bytes": 180_000_000_000,
        "memory_max_bytes": 200_000_000_000,
        "memory_swap_max_bytes": 0,
        "tasks_max": 4096,
    }
    assert evaluate_resource_limits(valid)["passed"] is True
    for field, excessive in (
        ("cpu_quota_cores", 20.01),
        ("cpuset_cpus", 21),
        ("memory_high_bytes", 180_000_000_001),
        ("memory_max_bytes", 200_000_000_001),
        ("memory_swap_max_bytes", 1),
        ("tasks_max", 4097),
    ):
        result = evaluate_resource_limits({**valid, field: excessive})
        assert result["passed"] is False
        assert any(not value for value in result["checks"].values())
    assert evaluate_resource_limits({**valid, "cgroup_path": "/user.slice/other.service"})["passed"] is False


def test_cpu_environment_matches_template_and_hides_gpu_devices() -> None:
    root = Path(__file__).resolve().parents[1]
    template_path = root / "deploy/systemd/raama-cpu.env"
    template = parse_cpu_env_file(template_path)
    assert {key: template[key] for key in THREAD_ENVIRONMENT} == THREAD_ENVIRONMENT
    environment = {
        **THREAD_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
    }
    result = inspect_cpu_environment(environ=environment, env_file=template_path)
    assert result["passed"] is True
    assert result["visible_gpu_devices"] == 0
    assert result["provider_requests"] == 0
    assert inspect_cpu_environment(
        environ={**environment, "DOCLING_DEVICE": "cuda"},
        env_file=template_path,
    )["passed"] is False


def test_linux_cgroup_reads_caps_from_raama_slice_and_effective_cpuset_from_leaf(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    slice_path = root / "raama.slice"
    leaf_path = slice_path / "workbench.service"
    leaf_path.mkdir(parents=True)
    for filename, value in {
        "cpu.max": "2000000 100000",
        "memory.high": "180000000000",
        "memory.max": "200000000000",
        "memory.swap.max": "0",
        "pids.max": "4096",
    }.items():
        (slice_path / filename).write_text(value, encoding="ascii")
    (leaf_path / "cpuset.cpus.effective").write_text("0-19", encoding="ascii")
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/raama.slice/workbench.service\n", encoding="ascii")

    result = inspect_linux_cgroup(cgroup_root=root, membership_file=membership)

    assert result["passed"] is True
    assert result["values"]["cpu_quota_cores"] == 20
    assert result["values"]["cpuset_cpus"] == 20


def test_linux_cgroup_rejects_inherited_unbounded_slice_caps(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    slice_path = root / "raama.slice"
    leaf_path = slice_path / "workbench.service"
    leaf_path.mkdir(parents=True)
    for filename, value in {
        "cpu.max": "max 100000",
        "memory.high": "max",
        "memory.max": "max",
        "memory.swap.max": "0",
        "pids.max": "max",
    }.items():
        (slice_path / filename).write_text(value, encoding="ascii")
    (leaf_path / "cpuset.cpus.effective").write_text("0-19", encoding="ascii")
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/raama.slice/workbench.service\n", encoding="ascii")

    with pytest.raises(ValueError, match="finite"):
        inspect_linux_cgroup(cgroup_root=root, membership_file=membership)
