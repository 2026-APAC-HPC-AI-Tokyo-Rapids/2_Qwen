# Qwen3 submission

`build/setup.sh` invokes the repository's pinned Buildah recipe. It requires a local mirror named `5_SGLang_Src.git` and publishes a verified Enroot-compatible SquashFS image. `build/build.log` is the complete log from the successful fresh build in GitHub Actions run 34681001728; `build/build-test.log` records the package and runtime checks executed inside that image. `build/provenance.json` records the verified SquashFS and OCI digests, source commits, and build tool versions from that run.

On a Firmus login node, submit the official two-node run with:

```bash
CONTAINER_IMAGE=/path/to/sglang.sqsh sbatch run/submit.sbatch
```

The script uses `/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8` by default. It starts four TP2 prefill workers on the first node, two TP4/EP4 decode workers on the second node, and one SGLang PD router backed by NIXL. It resolves each compute hostname to IPv4 and supplies the same addresses to the router, health checks, and SGLang's internal control plane through `SGLANG_HOST_IP`. It also limits `UCX_NET_DEVICES` to active HCAs whose link layer is InfiniBand, excluding the two Ethernet HCAs present on Firmus nodes. AIPerf 0.12.0 runs the fixed official 60/120/30 second workload through the single OpenAI-compatible gateway. Logs, the raw `profile_export_aiperf.json`, and `score.json` are written below `results/logs/$SLURM_JOB_ID`.

Set `PROFILE=true` only for a separate diagnostic run. This records one-second GPU, CPU, NIC, and InfiniBand telemetry below the run's `profile` directory and must not be treated as the performance result.

The run records the exact AIPerf command and runtime version/tokenizer revision in `aiperf-command.txt` and `aiperf-provenance.txt`. Official validation checks these artifacts against the fixed workload before accepting input-length observations of 4000 or the pinned client's prefix-compensated 3986 tokens. Other input lengths, output lengths other than 200, and good request fractions below 0.90 fail validation. Raw observations and scoring remain unchanged; `score.json` also records `official_validation`. See the [input-length investigation](../research/aiperf-isl-validation.md) for the source and diagnostic evidence.

`analysis/profiling_data/16928` contains the recovered telemetry and aggregate notes from a separate concurrency-16 diagnostic run. That allocation reached the Firmus time limit during AIPerf warm-up and has no benchmark score; it is retained only to document the observed low utilization and transfer stall.
