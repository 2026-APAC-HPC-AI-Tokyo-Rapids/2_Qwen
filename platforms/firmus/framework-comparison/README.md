# Firmus framework experiments through the team Actions pipeline

This branch copies the working `matched-settings-comparison` scripts. The remote
`readable-baseline` and `matched-settings-comparison` directories remain unchanged.
Worker launch arguments, GPU layout, client arguments, and 20/60/15-second timings
are preserved. These are reference-07 comparisons, not official-duration submissions.

Edit `configs/firmus-framework.json`: `framework` is `sglang` or `vllm`, and
`concurrency` initially equals 8. Commit each condition separately and push with
`[cluster:firmus] [nodes:2]` in the commit title. A push without a cluster tag runs
validation only. Run the frameworks sequentially to avoid resource contention.

The two images are reused from the working baseline, with pinned sizes and SHA-256
hashes checked before Slurm submission. The SGLang image also runs the same AIPerf
client for both frameworks. The vLLM image includes the previously applied CUDA
compiler PATH fix. Runtime experiments do not need an AWS rebuild. This branch
rejects `[require-build]` and `[profile]` for Firmus rather than silently ignoring
them: source/container experiments need a separately built image and an updated
pin, and GPU profiling has not been implemented for these copied scripts.

Each run gets `/scratch/home/ACCOUNT/qwen/runs/pd-<Actions ID>-<attempt>.<suffix>`.
Only the paths change: worker/router/Slurm logs go in `logs/`, generated worker code
and scripts/configuration/provenance in `source/`, and raw AIPerf files and
`score.json` in `results/`. `profile/` is empty for normal runs. vLLM's known working
TensorRT-LLM JIT caches are copied into independent worker homes; their starting
hashes are recorded. This is a prepared-cache serving comparison, not cold startup.

The client is unchanged. A post-benchmark parser validates the exported workload,
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
