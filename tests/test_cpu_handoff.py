from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_mas.cpu_handoff import (
    CPU_ENVIRONMENT,
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
        "slice_cpuset_cpus": 20,
        "memory_high_bytes": 180_000_000_000,
        "memory_max_bytes": 200_000_000_000,
        "memory_swap_max_bytes": 0,
        "tasks_max": 4096,
    }
    assert evaluate_resource_limits(valid)["passed"] is True
    for field, excessive in (
        ("cpu_quota_cores", 20.01),
        ("cpuset_cpus", 21),
        ("slice_cpuset_cpus", 21),
        ("memory_high_bytes", 180_000_000_001),
        ("memory_max_bytes", 200_000_000_001),
        ("memory_swap_max_bytes", 1),
        ("tasks_max", 4097),
    ):
        result = evaluate_resource_limits({**valid, field: excessive})
        assert result["passed"] is False
        assert any(not value for value in result["checks"].values())
    assert evaluate_resource_limits({**valid, "cgroup_path": "/user.slice/other.service"})["passed"] is False


def test_cpu_environment_matches_template_and_hides_gpu_devices(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    template_path = root / "deploy/systemd/raama-cpu.env"
    template = parse_cpu_env_file(template_path)
    assert {key: template[key] for key in CPU_ENVIRONMENT} == CPU_ENVIRONMENT
    result = inspect_cpu_environment(
        environ=CPU_ENVIRONMENT,
        env_file=template_path,
        device_root=tmp_path,
        platform_name="linux",
    )
    assert result["passed"] is True
    assert result["accessible_gpu_device_node_count"] == 0
    assert result["gpu_mask_policy"]["passed"] is True
    assert result["gpu_device_probe"]["performed"] is True
    assert result["provider_requests"] == 0
    assert inspect_cpu_environment(
        environ={**CPU_ENVIRONMENT, "DOCLING_DEVICE": "cuda"},
        env_file=template_path,
        device_root=tmp_path,
        platform_name="linux",
    )["passed"] is False


def test_missing_gpu_masks_fail_even_when_device_probe_finds_no_device(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    result = inspect_cpu_environment(
        environ=THREAD_ENVIRONMENT,
        env_file=root / "deploy/systemd/raama-cpu.env",
        device_root=tmp_path,
        platform_name="linux",
    )
    assert result["gpu_mask_policy"]["passed"] is False
    assert result["gpu_device_probe"]["performed"] is True
    assert result["accessible_gpu_device_node_count"] == 0
    assert result["passed"] is False


def test_device_probe_is_reported_separately_from_passed_gpu_masks(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    node = tmp_path / "nvidia0"
    node.write_bytes(b"synthetic device node")
    monkeypatch.setattr(Path, "is_char_device", lambda path: path.name == "nvidia0")

    result = inspect_cpu_environment(
        environ=CPU_ENVIRONMENT,
        env_file=root / "deploy/systemd/raama-cpu.env",
        device_root=tmp_path,
        platform_name="linux",
    )

    assert result["gpu_mask_policy"]["passed"] is True
    assert result["gpu_device_probe"]["accessible_device_nodes"] == ["nvidia0"]
    assert result["accessible_gpu_device_node_count"] == 1
    assert result["passed"] is False


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
    (slice_path / "cpuset.cpus.effective").write_text("0-19", encoding="ascii")
    (leaf_path / "cpuset.cpus.effective").write_text("0-19", encoding="ascii")
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/raama.slice/workbench.service\n", encoding="ascii")

    result = inspect_linux_cgroup(cgroup_root=root, membership_file=membership)

    assert result["passed"] is True
    assert result["values"]["cpu_quota_cores"] == 20
    assert result["values"]["cpuset_cpus"] == 20
    assert result["values"]["slice_cpuset_cpus"] == 20
    assert result["project_wide_membership_verified"] is False


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
    (slice_path / "cpuset.cpus.effective").write_text("0-19", encoding="ascii")
    (leaf_path / "cpuset.cpus.effective").write_text("0-19", encoding="ascii")
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/raama.slice/workbench.service\n", encoding="ascii")

    with pytest.raises(ValueError, match="finite"):
        inspect_linux_cgroup(cgroup_root=root, membership_file=membership)


def test_linux_cgroup_rejects_wide_slice_even_if_process_leaf_is_narrow(tmp_path: Path) -> None:
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
    (slice_path / "cpuset.cpus.effective").write_text("0-127", encoding="ascii")
    (leaf_path / "cpuset.cpus.effective").write_text("0-19", encoding="ascii")
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/raama.slice/workbench.service\n", encoding="ascii")

    result = inspect_linux_cgroup(cgroup_root=root, membership_file=membership)

    assert result["values"]["slice_cpuset_cpus"] == 128
    assert result["values"]["cpuset_cpus"] == 20
    assert result["checks"]["slice_effective_cpuset_within_20_cpus"] is False
    assert result["checks"]["process_effective_cpuset_within_20_cpus"] is True
    assert result["passed"] is False
