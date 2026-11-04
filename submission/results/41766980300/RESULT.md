# Standalone run of this package with the logprob overlay

Actions run: 41766980300, attempt 1 (temporary standalone check workflow). Slurm job: 28658, COMPLETED 0:0, 00:10:47.
Qwen commit: b5b4b9765d18d36f295b68ee785cbe52e0e49d63, this package with `33-pp4-direct-logprob.patch` and the readability rewrite of `prepare.sh`, `submit.sbatch` and `worker.sh`.
Image: build 41573354672; SHA-256 a001f90142b571380894e9532e467c57b831397c9e2e9d613d9e04004c87def9; 12298305536 bytes.
Good output tokens/s: 10016.32823236973 (goodput 50.08164116184865 requests/s times 200). Good request fraction: 0.9816281279695914; 6314 requests, zero errors, zero OSL mismatches, eligible.
The repository-local official-workload validator accepted this run; it is not the organisers' accuracy check.

The job became READY after 264 s and wrote `run-success` (`logs/slurm-28658.log` ends with `AIPERF ok`). All four Decode ranks used the fused greedy verify kernels (`decode-greedy-dispatch.txt`).

`greedy-parity.json` is the post-benchmark greedy probe: eight fixed prompts through the gateway, 64 tokens each, every response with 64 token logprobs. The Prefill logs contain no logprob error, so overlay 33 served the logprob requests that stopped Prefill in run 41694678252.

The later happy-path rewrite of the scripts (commit 0b07356) was not run on Firmus; `test_scripts_do_what_the_measured_scripts_did` checks that it launches the same steps and servers with the same settings.

Only curated text evidence is included. The raw archive is `/srv/qwen-artifacts/41766980300/1/` on the team home server (checksums in its `SHA256SUMS`); every file here equals its archived copy. Omitted: `results/results/profile_export.jsonl` (raw event stream) and the AIPerf working directory `results/results/artifacts/`.
