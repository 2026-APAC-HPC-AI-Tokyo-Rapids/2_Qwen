# Firmus official workload evidence: Slurm 17005

This directory preserves evidence from GitHub Actions run `34726983099`, artifact `10309090452`, Firmus Slurm job `17005`, and source commit `b8568107a8414cfb8f8994e7bec013c01279f488`.

The fixed AIPerf 0.12.0 workload completed its 60-second warmup, 120-second benchmark, and 30-second grace phases. It processed 558 requests with zero errors, a good request fraction of 1.0, goodput of 4.536230620411887 requests/s, and a score of 907.2461240823774 output tokens/s.

Slurm recorded `FAILED 1:0` after the workload because the parser in the source commit incorrectly required the average observed output length to equal exactly 200. `score-original.json` preserves that rejection. The pinned AIPerf tolerance accepted the observed 200..201 output-length distribution with `osl_mismatch_count` equal to zero. `score.json` is the replay result from the corrected parser and records `official_validation.valid` as true. This evidence does not relabel the original Slurm state.

Replay the validation from the repository root:

```bash
python3 scripts/parse-aiperf-result.py submission/results/17005/profile_export_aiperf.json /tmp/score-17005.json --concurrency 16 --require-eligible --command submission/results/17005/aiperf-command.txt --provenance submission/results/17005/aiperf-provenance.txt
diff -u submission/results/17005/score.json /tmp/score-17005.json
```

`logs/` contains the AIPerf, router, four prefill, two decode, environment, prewarm, server-step, and Slurm logs recovered from the artifact. `source/` contains the recorded runtime configuration and provenance. `SHA256SUMS` covers every evidence file except the manifest itself.
