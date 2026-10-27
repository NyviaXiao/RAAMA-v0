"""Verify the CPU-only RAAMA host contract; optionally run a frozen offline smoke suite."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.cpu_handoff import (  # noqa: E402
    LOCKED_SMOKE_TESTS,
    inspect_cpu_environment,
    inspect_linux_cgroup,
)


def _revision() -> str | None:
    if shutil.which("git") is None:
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _run(command: list[str], environment: dict[str, str]) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-offline-smoke",
        action="store_true",
        help="after all resource checks pass, verify the lock offline and run the focused synthetic regression suite",
    )
    parser.add_argument("--output", type=Path, help="write the sanitized JSON handoff report to this path")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": "raama.cpu-only-handoff-check.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revision": _revision(),
        "platform": platform.platform(),
        "provider_requests": 0,
        "network_access_requested": False,
        "gpu_inference_requested": False,
        "project_wide_membership_verified": False,
    }
    errors: list[str] = []
    try:
        report["cgroup_v2"] = inspect_linux_cgroup()
    except (OSError, RuntimeError, ValueError) as error:
        report["cgroup_v2"] = {"passed": False, "error": str(error)}
        errors.append("cgroup_v2")
    try:
        report["cpu_environment"] = inspect_cpu_environment(
            env_file=ROOT / "deploy/systemd/raama-cpu.env",
        )
    except (OSError, ValueError) as error:
        report["cpu_environment"] = {"passed": False, "error": str(error)}
        errors.append("cpu_environment")
    else:
        if not report["cpu_environment"]["passed"]:
            errors.append("cpu_environment")
    if report["cgroup_v2"].get("passed") is not True:
        errors.append("cgroup_v2")

    if args.run_offline_smoke and not errors:
        uv = shutil.which("uv")
        if uv is None:
            report["offline_lock_check"] = {"exit_code": 127, "stderr_tail": "uv is required for the offline handoff check"}
            errors.append("offline_lock_check")
        else:
            env = dict(os.environ)
            env.update({
                "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "none",
                "DOCLING_DEVICE": "cpu",
                "TOKENIZERS_PARALLELISM": "false",
            })
            report["offline_lock_check"] = _run(
                [uv, "lock", "--check", "--offline"], env,
            )
            if report["offline_lock_check"]["exit_code"] != 0:
                errors.append("offline_lock_check")
            else:
                smoke_env = dict(env)
                smoke_env.update({
                    "RAAMA_PROCESS_THREADS": "1",
                    "RAAMA_MAX_HTTP_THREADS": "4",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "BLIS_NUM_THREADS": "1",
                    "DOCLING_NUM_THREADS": "1",
                })
                report["offline_smoke"] = _run(
                    [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *(
                        str(ROOT / path) for path in LOCKED_SMOKE_TESTS
                    )],
                    smoke_env,
                )
                if report["offline_smoke"]["exit_code"] != 0:
                    errors.append("offline_smoke")
    elif args.run_offline_smoke:
        report["offline_smoke"] = {"skipped": True, "reason": "resource or CPU-only environment checks failed"}

    report["failed_checks"] = sorted(set(errors))
    report["status"] = "passed" if not errors else "failed"
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
