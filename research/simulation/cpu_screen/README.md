# Bounded CPU serving screen (2026-10-03)

This isolated branch starts at serving-control commit `642e97d`. Each YAML manifest has a separate local commit with the condition and motivation. No condition was pushed, and no GPU/scheduler job was submitted. Raw outputs are retained outside Git at `/tmp/qwen-cpu-screen-20261003`; `results.csv`, `results.json`, `results-table.md`, `verification.json`, and `environment-and-source.json` are the review entrypoints.

The replay is synthetic 4000-input/200-output, fixed concurrency 40, four TP4/EP1 workers (16 modeled GPUs). It uses measured target-only BF16 KV/attention timing and a fixed transfer fit. It does **not** represent the deployed EAGLE3 FP8-KV control. `fallback_policy: deny` remains mandatory. The external FPM SHA256 is `d185ab5430d90031a5b8f6e92a5234e989d278e86a79e3ee3970cf586eaf3215`; its identity requires the exact read-only collected model path in the manifests.

## Correct SGLang prefill fields

The public compiler emits generic `max_num_batched_tokens`, but native SGLang at AISimulate `992dddc7ee2bad8fd523e7db99eea4ac64254766` ignores that field for admission. `crates/core/src/engine/scheduler/sglang/config.rs` uses nested `sglang.max_prefill_tokens` (default 16384) and `sglang.chunked_prefill_size` (default 8192); `prefill.rs` uses both remaining budgets. Therefore the initial `p2d2-t*` generic-budget comparisons are inert-knob diagnostics, not evidence that larger chunks cannot help.

`native_run.py` supplies these existing native fields at the Runner's existing deployment seam, without modifying installed simulator/source/timing data. Every native run retains exact `prediction/execution-spec.json`. The native 8192/s8 control reproduces all request-level aggregate control metrics exactly. `verification.json` confirms each executed native condition and 4000 completed requests.

Example replay (output directory must not exist):

```bash
/tmp/qwen17-simulation/venv/bin/python research/simulation/cpu_screen/run.py research/simulation/cpu_screen/native-p1d3-t32768-s8.yaml /tmp/qwen-cpu-review-p1d3-32k --fpm /tmp/qwen17-simulation/qwen16-dense-fpm/fpm_forward_perf.parquet
```

The runner sets `AIC_ALLOW_UNLISTED_VERSIONS=1` in its CPU child process. Native manifests use a narrow extra `native_sglang` section consumed by this runner; do not pass them directly to the stock CLI. They are not deployment configuration changes.

## Interpretation and data gaps

P1/D3 is the user's intended 4-GPU prefill / 12-GPU decode layout. Its original 8192 raw directory retains the historical name `p1d3-negative`, but it is assessed as a real candidate. A 32768 chunk with sequence cap 8 rescues the synthetic joint latency gate; throughput remains far below the target-only P2/D2 control. Under this model, one prefill worker must reduce measured B8/full-4000-prompt latency from 1087 ms to below approximately 647 ms to match the actual serving control's request rate, before remaining overhead. Existing FPM cannot invent a backend-specific speedup.

Unconstrained P3/D1 original-pool replay exceeds collected decode KV totals (129368 > 128512), while its doubled-pool replay reaches an unmeasured B36 (>32). These are explicit unsupported shapes, not scored runs. Sequence caps 27/29 are separately labeled external admission diagnostics: a 512-token reservation bounds a 131072 pool at 29 synthetic prompts initially and 27 with full 200-token outputs. They do not implement SGLang's reservation/preallocated PD queue semantics. Readmissions in some original-pool diagnostics materially alter tail ITL; `results.json` records them. Doubled-pool capacity is an assumed input, not a device-memory validation.

The all-request table includes the initial transient. The separate steady diagnostic counts completions between 10% and 90% of virtual duration, retaining their full request latency. Neither window is official AIPerf measurement. Joint SLA uses each completed request's TTFT <=3000 ms and mean ITL <=20 ms, with a fraction >=0.90; native global ITL percentiles may describe a different sample basis, so table percentiles are recomputed from request records. Actual forward-batch size is unavailable from the whole-FPM detail provider; admission cohorts are not treated as forward batches.

The empirical single-D pool sensitivity at `/tmp/qwen-cpu-screen-20261003/empirical-pool-sensitivity.json` pairs two actual observations with a smooth affine latency assumption and an optimistic constant-ITL bound. It is explicitly not a validated predictor and is not injected into FPM. CUDA graph rungs include 32 and 40, so smooth values have no claim to rung-level precision. Pool superiority is not robust across these assumptions.

The source copies/provenance, initial model-path identity rejection logs, unsupported shape logs, and all successful raw request/detail reports remain in the external output directory. This screen supplies prioritization evidence only. Actual EAGLE3 FP8-KV prefill timing and scheduling coverage would be needed before changing the GPU layout or claiming serving performance.
