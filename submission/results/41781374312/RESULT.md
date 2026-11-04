# Standalone run of this package on Firmus

Actions run: 41781374312, attempt 1. Slurm job: 28673, COMPLETED 0:0, 00:10:37 (READY after 265 s).
Image build: 41729881283; SHA-256 3a47b757bb88f6bc14ad5f151e8405f645007f25f381f631d88046d9bd1b7b04.
Good output tokens/s: 10038.107744896623 (goodput 50.19053872448311 requests/s times 200).
Good request fraction: 0.981835413046912; 6331 requests, zero errors, zero OSL mismatches (output lengths 199 to 201).

This run used this package exactly as shipped, with no tuning step: `run/prepare.sh` staged the run directory and `sbatch scripts/submit.sbatch` ran the single job. The job script, preparation script and helpers were the versions before the readability rewrite described in `../../README.md`; the rewrite keeps the same commands, flags and environment.

- `source/prefill-kernels.env` and `source/decode-kernels.env` are byte-identical to `../../config/`.
- `source/sglang-overlay.json` and `source/deep-gemm-overlay.json` list the 34 SGLang and 14 DeepGEMM overlays applied, in order, with SHA-256 values equal to `../../patches/overlay/`, on the image above.
- `decode-greedy-dispatch.txt` shows the fused greedy verify kernels on the Decode ranks; `prefill-r32-dispatch-{0,1,2}.txt` show each of the twelve Prefill stages prewarming and dispatching the FC1 first-K kernel.
- `greedy-parity.json` is the post-benchmark greedy probe: all eight responses carry 64 token logprobs, and the Prefill logs show no error (requests with `logprobs: true` go through overlay 34).

Only curated text evidence is included; each file is unchanged from the run's output. Omitted: `results/profile_export.jsonl` (raw event stream), the AIPerf working directory `results/artifacts/`, and the rest of the run's `source/` records.
