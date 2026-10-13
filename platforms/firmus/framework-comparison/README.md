# Firmus framework experiments through the team Actions pipeline

The current SGLang condition matches ASPIRE run `37714793539` at concurrency 48,
P2TP4/D2TP4 EP1, FP8 E4M3 KV, pool 131072, chunk/context 8192, FA3 attention,
standard prefill CUTLASS, and round-robin prefill routing. It reuses the pinned
base SGLang image without the ASPIRE custom kernel patches and disables EAGLE.
Model loading still uses Firmus's shared model path; measured serving uses GPU
resident weights. ASPIRE stages the same model revision to node-local storage.
The client uses official 60/120/30 timings and records actual server settings
before measurement. This SGLang condition is unprofiled; profile/reference modes
are rejected before provisioning. vLLM retains the earlier reference path.

## Earlier reference and profile controls

The following records describe the earlier six-worker controls and their commits.

This branch copies the working `matched-settings-comparison` scripts. The remote
`readable-baseline` and `matched-settings-comparison` directories remain unchanged.
Worker launch arguments, GPU layout, client arguments, and 20/60/15-second timings
are preserved. These are reference-07 comparisons, not official-duration submissions.

Edit `configs/firmus-framework.json`: `framework` is `sglang` or `vllm`, and
`concurrency` initially equals 8. Commit each condition separately and push with
`[cluster:firmus] [nodes:2]` in the commit title. A push without a cluster tag runs
validation only. Both controls request both nodes and all 16 GPUs, so they may be
queued together but cannot execute concurrently. Their worker homes are isolated.
For manual dispatch, this branch defaults to Firmus only; leave the other cluster
checkboxes disabled. Experiment conditions should still be tracked by commits.

The two images are reused from the working baseline, with pinned sizes and SHA-256
hashes checked before Slurm submission. The SGLang image also runs the same AIPerf
client for both frameworks. The vLLM image includes the previously applied CUDA
compiler PATH fix. Runtime experiments do not need an AWS rebuild. This branch
rejects `[require-build]`: source/container experiments need a separately built
image and an updated pin. `[profile]` captures every worker of the selected framework (seven for the current vLLM layout).
The separate [vLLM image build on AWS](../../../research/vllm-container-build.md)
produces a Docker archive and a new Firmus SquashFS without overwriting these
baseline images or submitting a benchmark.

Each run gets `/scratch/home/ACCOUNT/qwen/runs/pd-<Actions ID>-<attempt>.<suffix>`.
Only the paths change: worker/router/Slurm logs go in `logs/`, generated worker code
and scripts/configuration/provenance in `source/`, and raw AIPerf files and
`score.json` in `results/`. `profile/` is empty for normal runs. vLLM's known working
TensorRT-LLM JIT caches are copied into independent worker homes; their starting
hashes are recorded. This is a prepared-cache serving comparison, not cold startup.

For normal benchmark runs, the client is unchanged. A post-benchmark parser validates the exported workload,
excludes warmup records, checks token lengths and errors, and computes the joint
TTFT <= 3000 ms / mean ITL <= 20 ms fraction from raw requests. It uses the existing
repository score calculation, marks `official_submission=false`, and does not
treat missing the latency target as a scheduler failure. `run-success` means the
reference benchmark completed and its export was validated, not that it qualified
for an official submission or passed the competition accuracy task.

The existing Firmus scheduler monitor and artifact collector are used. Artifacts
keep the name `qwen-firmus-pd-<Actions ID>-<attempt>`. The unchanged archive job moves
them to `/srv/qwen-artifacts/<Actions ID>/<attempt>/` on the
team home server, writes `SHA256SUMS`, then deletes the temporary GitHub artifact.
The per-run Firmus directory is removed afterward. Failure logs follow the same
archive path; only this workflow's own scheduler job is cleaned up.

BigQuery is not a step in the current Firmus workflow. This branch preserves that
behavior; it does not invent a separate metrics publishing path.

The committed SGLang and vLLM worker, router and client commands were compared with
their remote working scripts. They match after substituting the run/output paths.
The proxy is byte-identical, SHA-256
`4cb4e23e0b4b48afc4a4af7b6fa29ddbc1a21cde21bdb1979a50010614077de0`.

The first integration controls are Actions `37562274684` (SGLang, commit `c7bb031`)
and `37562605777` (vLLM, commit `caa006f`). A subsequent metadata correction keeps
the standard `model.json` and container hash fields and makes `cache_source` null
for SGLang. In the first SGLang control's original provenance that field names the
comparison directory even though the provisioner only copies caches for vLLM;
it must not be interpreted as evidence of a prepared SGLang cache. Original run
artifacts are retained as generated by their committed scripts.

## Nsight Systems capture

With `framework: sglang` or `framework: vllm`, push a commit tagged `[cluster:firmus] [nodes:2] [profile]`.
The baseline image and all serving/workload settings are reused. Provisioning downloads
NVIDIA's Nsight Systems 2025.3.2.474 Debian package, verifies SHA-256, and extracts
only its CLI runtime into this run's temporary directory. No system installation
or image rebuild is performed; workers mount the tool read-only.

After AIPerf warmup and five seconds of benchmark traffic, `/start_profile` activates
CUDA profiling on P4/D2 concurrently. SGLang receives the `CUDA_PROFILER`
activity; vLLM uses `--profiler-config '{"profiler":"cuda"}'` and an empty
POST body, with `VLLM_WORKER_MULTIPROC_METHOD=spawn` for Nsight. These extra
settings are enabled only in profile runs. vLLM stops five seconds after all acknowledgments.
SGLang captures through the AIPerf benchmark and grace completion marker before
calling `/stop_profile`: a short window can miss idle prefill workers, and stopping
Nsight stalls the workers while reports flush. Each worker records CUDA graph node kernels, NVTX,
and OS runtime; CPU sampling and Torch CPU/GPU profiler are disabled. Partial starts
receive best-effort stop before workers are cleaned up. Reports, API responses,
actual launch arguments and timestamped capture logs survive benchmark failure.

Success additionally requires six nonempty `.nsys-rep` files and CUDA kernel events
on all sixteen GPUs, checked through temporary SQLite exports in the allocation.
`profile/validation.json` retains the per-worker GPU and kernel counts; successful
SQLite exports are removed after validation to avoid duplicating the native reports.
The instrumented score is not a performance adoption measurement.

The Actions summary records status, topology, concurrency, reference timings,
requests, good fraction, goodput, score, Nsight report/GPU validation and the canonical
archive location. Failed submitted runs also get a summary of whatever was collected.

For Dynamo topology optimization, this capture is the fixed reference control:
P4 × TP2 / D2 × TP4 EP4, concurrency 8. Nsight shows kernel/communication costs
for this configuration; it does not itself select replica counts or TP. Dynamo's
Profiler needs engine latency/throughput/capacity measurements for the candidate
TP configurations, and candidate topologies still need uninstrumented E2E checks
with the task's TTFT/ITL constraints. The current job uses the reference NIXL proxy;
Dynamo frontend/worker integration is a separate experiment.

## Dynamo PD experiments

`codex/firmus-vllm-dynamo` and `codex/firmus-sglang-dynamo` select their backend in
`configs/firmus-dynamo.json`. Dynamo 1.5.1 and each official runtime's amd64 OCI
digest are pinned. The vLLM runtime contains vLLM 0.28.0 / NIXL 1.3.2; the SGLang
runtime contains SGLang 0.5.18 / NIXL 1.4.0. Engine versions differ from the native
controls and must be included when interpreting comparisons.

Push `[require-build] [cluster:firmus] [dynamo] [nodes:2]` to validate the official
runtime on AWS, preserve its environment for Enroot, and submit the scheduler job.
For subsequent runtime changes, record the completed build's Actions ID as
`container_run_id` in the configuration and omit `[require-build]` to reuse it.
Build artifacts survive benchmark failures.

The first uninstrumented control keeps P4 × TP2 / D2 × TP4 EP4 and the reference-07
client unchanged. An etcd service discovers workers across both nodes; Dynamo's
frontend uses the TCP request plane and ZMQ event plane, and NIXL transfers KV
between prefill and decode. Worker system-health checks and a chat request precede
AIPerf. Worker arguments, metrics, logs, results and image provenance follow the
existing artifact archive path and Actions summary. These controls establish a
working Dynamo baseline before candidate TP or worker-count experiments.

## Short vLLM profiling runs

For vLLM, `[profile]` runs only a diagnostic client: c104, ISL4000/OSL200,
20 seconds warmup, 30 seconds measurement and up to 30 seconds grace, with
2,048 synthetic entries. It omits the normal 60/120/30 benchmark pass and collects
five seconds of CUDA/NVTX activity from all seven workers. The profile branch
requests 14 minutes instead of 17. Native reports and diagnostic requests live
under `profile/`; `results/` holds server configuration, with no official score.
Provenance records `measurement_mode=profile-only`; terminal validation requires
the diagnostic export, then validates all 16 GPU traces on the login runner.
Use existing uninstrumented benchmarks for performance comparison. A profile
completion marker does not imply an official benchmark or accuracy result.
