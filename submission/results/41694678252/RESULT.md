# Standalone run of this package: highest measured score

Actions run: 41694678252, attempt 1 (temporary standalone check workflow). Slurm job: 28589, FAILED 1:0, 00:13:05.
Qwen commit: 00a631e98d9a66c0911b4978263608253d9140c2, this package before `33-pp4-direct-logprob.patch` was added.
Image: build 41573354672; SHA-256 a001f90142b571380894e9532e467c57b831397c9e2e9d613d9e04004c87def9; 12298305536 bytes.
Good output tokens/s: 10025.33529068414. Good request fraction: 0.9817922735908803; 6316 requests, zero errors, zero OSL mismatches, eligible.
This is the highest good output tok/s the team measured on Firmus. The repository-local official-workload validator accepted it; it is not the organisers' accuracy check.

The job prepared the package with `prepare.sh`, ran `submit.sbatch` unchanged and became READY after 302 s (`logs/slurm-28589.log`). All four Decode ranks used the fused greedy verify kernels (`decode-greedy-dispatch.txt`; four lines in `logs/decode-0.log`).

The job ended FAILED after the score was written and validated. The post-benchmark greedy probe sends requests with token `logprobs`. The team's direct PP4 Prefill loop did not relay the last stage's logprob outputs, so `process_batch_result_disagg_prefill` raised `AttributeError: 'NoneType' object has no attribute 'next_token_logprobs'` on prefill-0 and, after the router retried there, on prefill-2 (tracebacks in `logs/prefill-0.log` and `logs/prefill-2.log`). The probe timed out after 120 s (`logs/greedy-probe.log`) and `run-success` was not written. The official AIPerf workload sends no logprob requests, so the benchmark was unaffected. Overlay 33 fixes the relay; requests without logprobs take the unchanged path.

Only curated text evidence is included. The raw archive is `/srv/qwen-artifacts/41694678252/1/` on the team home server (checksums in its `SHA256SUMS`); every file here equals its archived copy. Omitted: `results/results/profile_export.jsonl` (raw event stream) and the AIPerf working directory `results/results/artifacts/`.
