# Qwen3 submission

`build/setup.sh` invokes the repository's pinned Buildah recipe. It requires a local mirror named `5_SGLang_Src.git` and publishes a verified Enroot-compatible SquashFS image. `build/build.log` is the complete log from the successful fresh build in GitHub Actions run 34681001728; `build/build-test.log` records the package and runtime checks executed inside that image.

On a Firmus login node, submit the official two-node run with:

```bash
CONTAINER_IMAGE=/path/to/sglang.sqsh sbatch run/submit.sbatch
```

The script uses `/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8` by default. It starts four TP2 prefill workers on the first node, two TP4/EP4 decode workers on the second node, and one SGLang PD router backed by NIXL. AIPerf 0.12.0 runs the fixed official 60/120/30 second workload through the single OpenAI-compatible gateway. Logs, the raw `profile_export_aiperf.json`, and `score.json` are written below `results/logs/$SLURM_JOB_ID`.

Set `PROFILE=true` only for a separate diagnostic run. This records one-second GPU telemetry and must not be treated as the performance result.
