"""Linux cgroup-v2 preflight for the bounded CPU-only RAAMA handoff."""

from __future__ import annotations

import os
import re
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
    memory_high = values.get("memory_high_bytes")
    memory_max = values.get("memory_max_bytes")
    memory_swap = values.get("memory_swap_max_bytes")
    tasks_max = values.get("tasks_max")
    cgroup_path = str(values.get("cgroup_path", ""))
    checks = {
        "raama_slice_membership": "raama.slice" in cgroup_path.split("/"),
        "finite_cpu_quota_within_20_cores": isinstance(cpu_quota, (int, float)) and 0 < cpu_quota <= CPU_CORE_LIMIT,
        "effective_cpuset_within_20_cpus": isinstance(cpus, int) and 0 < cpus <= CPU_CORE_LIMIT,
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
    values = {
        "cgroup_path": cgroup_path,
        "cpu_quota_cores": cpu_quota,
        "cpuset_cpus": len(cpus),
        "memory_high_bytes": parse_finite_cgroup_limit((slice_directory / "memory.high").read_text(encoding="ascii"), "memory.high"),
        "memory_max_bytes": parse_finite_cgroup_limit((slice_directory / "memory.max").read_text(encoding="ascii"), "memory.max"),
        "memory_swap_max_bytes": parse_finite_cgroup_limit((slice_directory / "memory.swap.max").read_text(encoding="ascii"), "memory.swap.max"),
        "tasks_max": parse_finite_cgroup_limit((slice_directory / "pids.max").read_text(encoding="ascii"), "pids.max"),
    }
    return {"values": values, **evaluate_resource_limits(values)}


def inspect_cpu_environment(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path = "deploy/systemd/raama-cpu.env",
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    template = parse_cpu_env_file(env_file)
    template_checks = {
        key: template.get(key) == expected
        for key, expected in THREAD_ENVIRONMENT.items()
    }
    runtime_checks = {
        key: environment.get(key) == expected
        for key, expected in THREAD_ENVIRONMENT.items()
    }
    cuda_value = environment.get("CUDA_VISIBLE_DEVICES", "")
    nvidia_value = environment.get("NVIDIA_VISIBLE_DEVICES", "").casefold()
    gpu_checks = {
        "cuda_devices_hidden": cuda_value.strip().casefold() in {"", "-1", "none", "void"},
        "nvidia_devices_hidden": nvidia_value in {"", "none", "void"},
    }
    checks = {
        **{f"template_{key}": value for key, value in template_checks.items()},
        **{f"runtime_{key}": value for key, value in runtime_checks.items()},
        **gpu_checks,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "visible_gpu_devices": 0 if all(gpu_checks.values()) else "not_hidden",
        "provider_requests": 0,
    }
