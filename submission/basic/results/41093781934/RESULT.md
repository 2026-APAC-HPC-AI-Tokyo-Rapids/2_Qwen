# Measured Basic result

Actions run: 41093781934, attempt 1. Slurm job: 27709, COMPLETED 0:0, 00:10:35.
Qwen commit: 0cd50aea3d4d3075d59d4a73a4c98ac584608051 (runtime settings only). `configs/firmus-basic.json`, the worker launcher and the image recipe are byte-identical to this package, apart from the optional registry cache of `scripts/build-basic-container.sh`; `prepare.sh` and `submit.sbatch` were later rewritten for readability with the same recorded behaviour (see `../../README.md`).
Image build: 41046995676; SHA-256 d6ea5d95d6d76ab87ea55aeb70a052f04dbba8f9bbd515c3ece394690b2a1058; 16417116160 bytes; official `lmsysorg/sglang` v0.5.23 (SGLang 297e54fd105e974cdf401e5b4d20253a22c16f90), no source patch.
Good output tokens/s: 6339.309798565987 (goodput 31.696548992829936 requests/s times 200).
Good request fraction: 0.9791252485089463; 4024 requests, 0 errors, 0 OSL mismatches.
The repository-local official-workload validator accepted this run; it is not the organisers' accuracy check.

Files: the official AIPerf export, command and provenance, the local score, the worker server info and sanity chat (this directory), every worker, router, AIPerf and Slurm log (`logs/`), and the complete public launch argv/env of each worker (`launch/`). Each file is unchanged from the run's output. Omitted: `results/results/profile_export.jsonl` (raw event stream), the AIPerf working directory `results/results/artifacts/` and the diagnostic `eagle-acceptance.json`.
