# Measured Firmus result

Actions run: 41573354672, attempt 1. Slurm job: 28478, COMPLETED 0:0, 00:13:22.
Qwen commit: 14c3e861d52185f4a2753620213caf629c5d0400 (image built and served in the same run).
Image build: 41573354672 (Qwen 14c3e86, SGLang fc275674); SHA-256 a001f90142b571380894e9532e467c57b831397c9e2e9d613d9e04004c87def9; 12298305536 bytes.
Good output tokens/s: 10021.598269468013 (goodput 50.10799134734007 requests/s times 200).
Good request fraction: 0.9819620253164557; 6320 requests, zero errors, zero OSL mismatches.
The repository-local official-workload validator accepted this run; it is not the organisers' accuracy check.

This is the measured round 25 tuning run, not a standalone verification of this package.
Its three in-job gates all adopted their candidates; their `selection.json` records are kept here as evidence for the pinned env files in `../../config/`:

- `prefill-fa3-pair/selection.json`: adopted `SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe+m192n96-pair` on all twelve Prefill GPUs.
- `prefill-fa3-prepack/selection.json`: selected P3 (adopted); the final `prefill-kernels.env` value it left is `n192-direct-o-varlen-pipe+m192n96-pair-prepack-v3`.
- `decode-greedy/selection.json`: adopted `SGLANG_DECODE_GREEDY_VERIFY=1` on all four Decode GPUs at vocabulary 151936.

`decode-greedy-dispatch.txt` is the raw file this run wrote; its grep pattern cut the line at `g` (the bracket `[^\r]` excludes `\` and `r`). The submitted job uses the corrected `.*` pattern from `main`. `logs/decode-0.log` holds the full line on each of the four Decode ranks: `Decode greedy verify using fused kernels greedy_tiles/greedy_finish: rows=4 vocab=151936 stride=151936`.

Only curated text evidence is included. The raw archive of this run is `/srv/qwen-artifacts/41573354672/1/` on the team home server (checksums in its `SHA256SUMS`). Omitted here:

- results/results/profile_export.jsonl (30046914 bytes; raw event stream, over 20 MB)
- results/results/artifacts/ (AIPerf working directory, including the 227995065-byte inputs.json; outside the curated list)
- results/results/eagle-acceptance.json, pp4-kernel-settings.json and the per-rank gate records under prefill-fa3-pair/, prefill-fa3-prepack/ and decode-greedy/ (diagnostics outside the curated list)
- logs/logs/gpu-telemetry-*.csv and their stderr logs (telemetry, removed from the package)
