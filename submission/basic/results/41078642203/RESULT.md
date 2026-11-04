# Measured Basic result

Actions run: 41078642203, attempt 1. Slurm job: 27682, COMPLETED 0:0, 00:10:04.
Qwen commit: 40001baee63a6bb2f9a305ee3f04dbfe6f0b5717 (runtime settings only). `../../config/firmus-basic.json` and the image recipe are byte-identical to this package, apart from the optional registry cache of the build script; the job scripts and the worker launcher were later rewritten for readability with the same recorded behaviour (see `../../README.md`).
Image build: 41046995676; SHA-256 d6ea5d95d6d76ab87ea55aeb70a052f04dbba8f9bbd515c3ece394690b2a1058; 16417116160 bytes; official `lmsysorg/sglang` v0.5.23 (SGLang 297e54fd105e974cdf401e5b4d20253a22c16f90), no source patch.
Good output tokens/s: 6321.115669372811 (goodput 31.605578346864053 requests/s times 200).
Good request fraction: 0.9768599154018413; 4019 requests, 0 errors, 0 OSL mismatches.
The run is eligible under the official good-request rule; this is not the organisers' accuracy check.

Files: the official AIPerf export, command and provenance, the local score, the worker server info and sanity chat (this directory), every worker, router, AIPerf and Slurm log (`logs/`), and the complete public launch argv/env of each worker (`launch/`). Each file is unchanged from the run's output. Omitted: `results/results/profile_export.jsonl` (raw event stream), the AIPerf working directory `results/results/artifacts/` and the diagnostic `eagle-acceptance.json`.
