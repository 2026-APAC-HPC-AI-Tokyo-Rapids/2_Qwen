# Measured Firmus result (PP4 round 34)

Actions run: 41767886175, attempt 1. Slurm job: 28660, COMPLETED 0:0, 00:14:52.
Qwen commit: 50773368654b5a36497c1d69a1bb0e544eaee66e, the round 34 submit commit (`[prefill-pp4]` serving of image 3a47b757 with its in-job gates).
Image build: 41729881283 (Qwen 3f9124fd8afb28e34c71288b6dc31e5c490c7f60, SGLang fc275674); SHA-256 3a47b757bb88f6bc14ad5f151e8405f645007f25f381f631d88046d9bd1b7b04; 12296638464 bytes.
Good output tokens/s: 10054.492606928183 (goodput 50.272463034640914 requests/s times 200).
Good request fraction: 0.9821569556292437; 6333 requests, zero errors, zero OSL mismatches.
The run is eligible under the official good-request rule; this is not the organisers' accuracy check.

This is the measured round 34 tuning run the package was frozen from, not a standalone verification of this package. Same-image runs whose FC1 first-K gate rejected the kernel scored 9998.67 and 10015.98. Its four in-job gates all adopted their candidates; their `selection.json` records are kept as evidence for the pinned env files in `../../config/`:

- `prefill-fa3-pair/selection.json`: adopted `SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe+m192n96-pair` on all twelve Prefill GPUs.
- `prefill-fa3-prepack/selection.json`: selected P3 (adopted); the final value is `n192-direct-o-varlen-pipe+m192n96-pair-prepack-v3`.
- `decode-greedy/selection.json`: adopted `SGLANG_DECODE_GREEDY_VERIFY=1` on all four Decode GPUs at vocabulary 151936.
- `prefill-megamoe-fc1-first-k/selection.json`: adopted `DG_SM90_PREFILL_FC1_FIRST_K=1` on all twelve Prefill GPUs (byte equality on every case, 1200-call endurance, health and the pooled timing rule).

`decode-greedy-dispatch.txt` shows the fused greedy verify kernels on the Decode ranks, and `prefill-r32-dispatch-{0,1,2}.txt` show every Prefill stage prewarming and dispatching the FC1 first-K kernel. After this run the package adds one runtime overlay, `34-pp4-direct-logprob.patch` (the logprob relay for the direct PP4 Prefill loop), as the last SGLang overlay.

Only curated text evidence is included. Each file is unchanged from the run's output. Omitted: `results/results/profile_export.jsonl` (raw event stream), the AIPerf working directory `results/results/artifacts/`, `eagle-acceptance.json`, `pp4-kernel-settings.json`, the per-rank gate records and the GPU telemetry CSVs.
