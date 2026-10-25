"""Small process-local thread limits for CPU-bound research components."""

from __future__ import annotations

import os


MAX_PROJECT_THREADS = 20
DEFAULT_PROCESS_THREADS = 1
DEFAULT_HTTP_THREADS = 4


def process_thread_limit() -> int:
    """Return a bounded per-process BLAS/model thread count.

    The shared-host aggregate limit is enforced by the delegated project cgroup;
    this setting prevents each process from multiplying that allocation.
    """
    raw = os.environ.get("RAAMA_PROCESS_THREADS", str(DEFAULT_PROCESS_THREADS))
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_PROCESS_THREADS
    return max(1, min(MAX_PROJECT_THREADS, value))


def http_thread_limit() -> int:
    raw = os.environ.get("RAAMA_MAX_HTTP_THREADS", str(DEFAULT_HTTP_THREADS))
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_HTTP_THREADS
    return max(1, min(8, value))


def apply_thread_limits(*, apply_torch: bool = False) -> int:
    """Set common numerical-library limits before importing inference libraries."""
    threads = process_thread_limit()
    value = str(threads)
    for variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
        "DOCLING_NUM_THREADS",
    ):
        os.environ[variable] = value
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        pass
    else:
        threadpool_limits(limits=threads)

    if apply_torch:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.set_num_threads(threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # PyTorch allows this setting only before its first parallel operation.
                pass
    return threads
