# RAAMA CPU-only handoff check (2026-10-27)

This handoff provides a fail-closed preflight and an optional offline regression run for a Linux CPU host. It does not provision cgroups, choose host CPUs, start project services, download models, or contact a provider. The Windows development machine cannot verify the target Linux slice; the target-host run is still required before calling the resource contract verified.

## Required target setup

Run the checker from a process whose unified cgroup-v2 membership is under the exact `raama.slice` path component. Configure the `raama.slice` itself with no more than 20 CPU-equivalents (`CPUQuota=2000%`), `MemoryHigh=180000000000`, `MemoryMax=200000000000`, `MemorySwapMax=0`, and `TasksMax=4096`. Give the slice an administrator-assigned `AllowedCPUs` set with at most 20 CPUs. The checker reads slice limits from the slice cgroup and reads `cpuset.cpus.effective` from the current process cgroup, so inherited `max` values at a child cannot be mistaken for a bounded setup. Do not assume or claim particular CPU IDs.

Load the supplied `deploy/systemd/raama.slice` and `deploy/systemd/raama-cpu.env` templates through the host's normal systemd administration process. The environment file limits numerical and Docling threads, caps workbench request threads at four, selects CPU Docling, and disables tokenizer parallelism. The checker requires these values in both the template and the running process environment, and fails if CUDA or NVIDIA devices are visible.

## Target-host command

From the repository root, with the process already placed in the configured slice:

```sh
python scripts/verify_cpu_handoff.py --run-offline-smoke --output .tmp/raama_cpu_handoff_20261027.json
```

The script first checks the cgroup-v2 membership, slice limits, effective CPU set, runtime thread environment, and hidden GPU devices. Any missing, unbounded, or excessive resource limit stops the run before tests start. On a passing preflight, it checks the locked dependencies with `uv lock --check --offline`, then runs the named local regression tests with provider-independent CPU settings. It makes no provider request and does not enable network access or GPU inference.

The JSON output records the revision, checks, and test exit status. Review it together with the host's systemd unit and cgroup configuration; a successful test suite alone does not establish service-level performance, PostgreSQL/HA readiness, or completion of RAAMA plan stages A–G.

## Current evidence

- The cgroup parser and resource matrix passed 15 focused local tests on Windows, including synthetic bounded and unbounded cgroup-v2 trees. Python compilation and `git diff --check` passed. The full preflight was also invoked on Windows and correctly failed closed because Linux cgroup-v2 files and the runtime slice environment are absent; it skipped the offline smoke suite. The report is `.tmp/raama_cpu_handoff_windows_preflight_20261027.json`.
- The real Linux `raama.slice` limits and effective CPU set have not been observed from the target host. The complete CPU-only handoff script has therefore not passed on that host, and the CPU resource contract remains unverified.
- Browser acceptance of the Chinese workbench remains pending. The earlier computer-use request to access Microsoft Edge was explicitly denied, so only loopback HTTP health and query behavior were checked.
- No provider was called for this handoff. Existing Qwen reservation limits remain in force.
