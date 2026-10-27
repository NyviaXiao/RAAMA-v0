# RAAMA CPU-only handoff check (2026-10-27)

This handoff provides a fail-closed preflight and an optional offline regression run for a Linux CPU host. It does not provision cgroups, choose host CPUs, start project services, download models, or contact a provider. The Windows development machine cannot verify the target Linux slice; the target-host run is still required before calling the resource contract verified.

## Required target setup

Run the checker from a process whose unified cgroup-v2 membership is under the exact `raama.slice` path component. Configure the `raama.slice` itself with no more than 20 CPU-equivalents (`CPUQuota=2000%`), `MemoryHigh=180000000000`, `MemoryMax=200000000000`, `MemorySwapMax=0`, and `TasksMax=4096`. Give the slice an administrator-assigned `AllowedCPUs` set with at most 20 CPUs. The checker reads slice limits and the slice's `cpuset.cpus.effective`, then separately reads the current process cgroup's effective CPU set. A narrower process leaf is allowed; it cannot conceal a slice wider than 20 CPUs. Do not assume or claim particular CPU IDs.

Load the supplied `deploy/systemd/raama.slice` and `deploy/systemd/raama-cpu.env` templates through the host's normal systemd administration process. The environment file limits numerical and Docling threads, caps workbench request threads at four, selects CPU Docling, disables tokenizer parallelism, and explicitly sets `CUDA_VISIBLE_DEVICES=` and `NVIDIA_VISIBLE_DEVICES=none`. The checker requires those masks in both the template and running process environment. Separately, on Linux it probes accessible character device nodes matching NVIDIA or DRM render devices under `/dev`; any accessible node fails the preflight even when masks are set. The environment-mask result is not reported as a device count. The device-node count covers this process only; it is not a full GPU-runtime inventory.

## Target-host command

From the repository root, with the process already placed in the configured slice:

```sh
python scripts/verify_cpu_handoff.py --run-offline-smoke --output .tmp/raama_cpu_handoff_20261027.json
```

The script first checks the cgroup-v2 membership, slice limits, both effective CPU sets, runtime thread environment, explicit GPU masks, and accessible NVIDIA/DRM character device nodes. Any missing, unbounded, excessive, or accessible device node stops the run before tests start. On a passing preflight, it checks the locked dependencies with `uv lock --check --offline`, then runs the named local regression tests with provider-independent CPU settings. It makes no provider request and does not enable network access or GPU inference.

The JSON output records the revision, checks, device-node probe, and test exit status. Its cgroup scope is the current process plus the owning `raama.slice`; it does not inventory membership for every RAAMA service or container, and always reports project-wide membership as unverified. Review it together with the host's systemd unit and cgroup configuration; a successful test suite alone does not establish service-level performance, PostgreSQL/HA readiness, or completion of RAAMA plan stages A–G.

## Current evidence

- The cgroup parser, resource matrix, explicit GPU-mask policy, and separate Linux device-node probe passed 18 focused local tests on Windows, including synthetic bounded/unbounded cgroup-v2 trees and a 128-CPU slice with a 20-CPU process leaf. Python compilation and `git diff --check` passed. The full preflight was also invoked on Windows and correctly failed closed because Linux cgroup-v2 files, the runtime slice environment, and the Linux device probe are absent; it skipped the offline smoke suite. The report is `.tmp/raama_cpu_handoff_windows_preflight_20261027.json`.
- The real Linux `raama.slice` limits and effective CPU set have not been observed from the target host. The complete CPU-only handoff script has therefore not passed on that host, and the CPU resource contract remains unverified.
- Browser acceptance of the Chinese workbench remains pending. The earlier computer-use request to access Microsoft Edge was explicitly denied, so only loopback HTTP health and query behavior were checked.
- No provider was called for this handoff. Existing Qwen reservation limits remain in force.
