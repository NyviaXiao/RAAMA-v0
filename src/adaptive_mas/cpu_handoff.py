"""Linux cgroup-v2 preflight for the bounded CPU-only RAAMA handoff."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping


CPU_CORE_LIMIT = 20.0
MEMORY_HIGH_LIMIT = 180_000_000_000
MEMORY_MAX_LIMIT = 200_000_000_000
TASKS_MAX_LIMIT = 4096
THREAD_ENVIRONMENT = {
    "RAAMA_PROCESS_THREADS": "1",
    "RAAMA_MAX_HTTP_THREADS": "4",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "DOCLING_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "DOCLING_DEVICE": "cpu",
}
CPU_ENVIRONMENT = {
    **THREAD_ENVIRONMENT,
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "none",
}
LOCKED_SMOKE_TESTS = (
    "tests/test_cpu_handoff.py",
    "tests/test_harness_evolution.py",
    "tests/test_harness_state.py",
    "tests/test_financial_rag_live_agent.py",
    "tests/test_financial_rag_evaluation.py",
    "tests/test_industrial_rag.py",
    "tests/test_industrial_rag_temporal.py",
)


def parse_cpu_max(value: str) -> float:
    fields = value.split()
    if len(fields) != 2 or fields[0] == "max":
        raise ValueError("cpu.max must contain a finite quota and period")
    quota, period = (int(field) for field in fields)
    if quota <= 0 or period <= 0:
        raise ValueError("cpu.max quota and period must be positive")
    return quota / period


def parse_cpuset(value: str) -> set[int]:
    if not value.strip():
        raise ValueError("cpuset.cpus.effective is empty")
    cpus: set[int] = set()
    for part in value.strip().split(","):
        if re.fullmatch(r"\d+", part):
            cpus.add(int(part))
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", part)
        if match is None:
            raise ValueError("cpuset contains an invalid CPU range")
        first, last = (int(item) for item in match.groups())
        if last < first:
            raise ValueError("cpuset contains a reversed CPU range")
        cpus.update(range(first, last + 1))
    if not cpus:
        raise ValueError("cpuset.cpus.effective is empty")
    return cpus


def parse_finite_cgroup_limit(value: str, name: str) -> int:
    value = value.strip()
    if value == "max" or not re.fullmatch(r"\d+", value):
        raise ValueError(f"{name} must be a finite nonnegative integer")
    return int(value)


def parse_cpu_env_file(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("CPU environment template contains an invalid assignment")
        values[key] = value.strip().strip("\"'")
    return values


def evaluate_resource_limits(values: Mapping[str, Any]) -> dict[str, Any]:
    cpu_quota = values.get("cpu_quota_cores")
    cpus = values.get("cpuset_cpus")
    slice_cpus = values.get("slice_cpuset_cpus")
    memory_high = values.get("memory_high_bytes")
    memory_max = values.get("memory_max_bytes")
    memory_swap = values.get("memory_swap_max_bytes")
    tasks_max = values.get("tasks_max")
    cgroup_path = str(values.get("cgroup_path", ""))
    checks = {
        "raama_slice_membership": "raama.slice" in cgroup_path.split("/"),
        "finite_cpu_quota_within_20_cores": isinstance(cpu_quota, (int, float)) and 0 < cpu_quota <= CPU_CORE_LIMIT,
        "slice_effective_cpuset_within_20_cpus": isinstance(slice_cpus, int) and 0 < slice_cpus <= CPU_CORE_LIMIT,
        "process_effective_cpuset_within_20_cpus": isinstance(cpus, int) and 0 < cpus <= CPU_CORE_LIMIT,
        "memory_high_within_180_gb": isinstance(memory_high, int) and 0 < memory_high <= MEMORY_HIGH_LIMIT,
        "memory_max_within_200_gb": isinstance(memory_max, int) and 0 < memory_max <= MEMORY_MAX_LIMIT,
        "swap_disabled": memory_swap == 0,
        "tasks_max_within_4096": isinstance(tasks_max, int) and 0 < tasks_max <= TASKS_MAX_LIMIT,
    }
    return {"checks": checks, "passed": all(checks.values())}


def inspect_linux_cgroup(
    *,
    cgroup_root: str | Path = "/sys/fs/cgroup",
    membership_file: str | Path = "/proc/self/cgroup",
) -> dict[str, Any]:
    root = Path(cgroup_root).resolve()
    membership = Path(membership_file).read_text(encoding="utf-8").splitlines()
    path_line = next((line for line in membership if line.startswith("0::")), None)
    if path_line is None:
        raise RuntimeError("the process is not in a unified cgroup-v2 hierarchy")
    cgroup_path = path_line[3:]
    directory = (root / cgroup_path.lstrip("/")).resolve()
    if not directory.is_relative_to(root):
        raise RuntimeError("the process cgroup path escapes the cgroup root")
    components = [part for part in cgroup_path.split("/") if part]
    try:
        raama_index = components.index("raama.slice")
    except ValueError as error:
        raise RuntimeError("the process is not in the raama.slice subtree") from error
    slice_directory = root.joinpath(*components[: raama_index + 1]).resolve()
    if not slice_directory.is_relative_to(root):
        raise RuntimeError("the raama.slice path escapes the cgroup root")
    # Limits must be configured on the owning slice. Reading only the leaf can
    # mistake cgroup-v2's inherited `max` values for an uncapped process.
    cpu_quota = parse_cpu_max((slice_directory / "cpu.max").read_text(encoding="ascii"))
    cpus = parse_cpuset((directory / "cpuset.cpus.effective").read_text(encoding="ascii"))
    slice_cpus = parse_cpuset((slice_directory / "cpuset.cpus.effective").read_text(encoding="ascii"))
    values = {
        "cgroup_path": cgroup_path,
        "cpu_quota_cores": cpu_quota,
        "cpuset_cpus": len(cpus),
        "slice_cpuset_cpus": len(slice_cpus),
        "memory_high_bytes": parse_finite_cgroup_limit((slice_directory / "memory.high").read_text(encoding="ascii"), "memory.high"),
        "memory_max_bytes": parse_finite_cgroup_limit((slice_directory / "memory.max").read_text(encoding="ascii"), "memory.max"),
        "memory_swap_max_bytes": parse_finite_cgroup_limit((slice_directory / "memory.swap.max").read_text(encoding="ascii"), "memory.swap.max"),
        "tasks_max": parse_finite_cgroup_limit((slice_directory / "pids.max").read_text(encoding="ascii"), "pids.max"),
    }
    return {
        "values": values,
        **evaluate_resource_limits(values),
        "scope": "current process cgroup and its owning raama.slice only",
        "project_wide_membership_verified": False,
    }


def probe_accessible_gpu_device_nodes(
    *,
    device_root: str | Path = "/dev",
    platform_name: str | None = None,
) -> dict[str, Any]:
    """Probe accessible Linux GPU device nodes independently of environment masks."""
    runtime_platform = sys.platform if platform_name is None else platform_name
    if runtime_platform != "linux":
        return {
            "performed": False,
            "reason": "Linux device-node probe is unavailable on this platform",
            "accessible_device_nodes": None,
        }
    root = Path(device_root)
    candidates = [*root.glob("nvidia[0-9]*"), *root.glob("dri/renderD*")]
    accessible = sorted({
        str(path.relative_to(root)).replace("\\", "/")
        for path in candidates
        if path.is_char_device() and os.access(path, os.R_OK | os.W_OK)
    })
    return {
        "performed": True,
        "reason": None,
        "accessible_device_nodes": accessible,
    }


def inspect_cpu_environment(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path = "deploy/systemd/raama-cpu.env",
    device_root: str | Path = "/dev",
    platform_name: str | None = None,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    template = parse_cpu_env_file(env_file)
    template_checks = {
        key: template.get(key) == expected
        for key, expected in CPU_ENVIRONMENT.items()
    }
    runtime_checks = {
        key: environment.get(key) == expected
        for key, expected in CPU_ENVIRONMENT.items()
    }
    cuda_value = environment.get("CUDA_VISIBLE_DEVICES", "")
    nvidia_value = environment.get("NVIDIA_VISIBLE_DEVICES", "").casefold()
    gpu_checks = {
        "cuda_devices_hidden": "CUDA_VISIBLE_DEVICES" in environment and cuda_value.strip().casefold() in {"", "-1", "none", "void"},
        "nvidia_devices_hidden": "NVIDIA_VISIBLE_DEVICES" in environment and nvidia_value in {"none", "void"},
    }
    gpu_probe = probe_accessible_gpu_device_nodes(device_root=device_root, platform_name=platform_name)
    checks = {
        **{f"template_{key}": value for key, value in template_checks.items()},
        **{f"runtime_{key}": value for key, value in runtime_checks.items()},
        **{f"gpu_mask_{key}": value for key, value in gpu_checks.items()},
        "gpu_device_probe_performed": gpu_probe["performed"],
        "no_accessible_gpu_device_nodes": gpu_probe["performed"] and not gpu_probe["accessible_device_nodes"],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "gpu_mask_policy": {"checks": gpu_checks, "passed": all(gpu_checks.values())},
        "gpu_device_probe": gpu_probe,
        "accessible_gpu_device_node_count": (
            len(gpu_probe["accessible_device_nodes"]) if gpu_probe["performed"] else None
        ),
        "scope": "process mask environment and accessible NVIDIA/DRM character device nodes under /dev; not a full GPU-runtime or project service/container inventory",
        "provider_requests": 0,
    }
