# Qwen3 submission

`build/setup.sh` invokes the pinned Buildah recipe from this repository. The repository root, including `containers/` and `scripts/build-sglang-container.sh`, is part of the build input; `submission/` alone is not a standalone container-build context. The recipe checks out SGLang commit `fc275674a97d78553b22597ee72c8fd48f5420f4` and installs the versions fixed by the container files. It requires root, Buildah, Skopeo, Umoci, SquashFS tools, Git, GitHub CLI, and jq. `GIT_MIRROR_DIR` may contain a local bare mirror named `5_SGLang_Src.git`; otherwise the recipe clones the repository with GitHub CLI.

From the repository root, build an Enroot-compatible SquashFS with a writable temporary directory, output directory, and OCI cache repository:

```bash
sudo env BUILD_CACHE_REPOSITORY=registry.example/team/qwen-build-cache BUILD_DIR=/var/tmp/qwen-build CONTAINER_ARCHIVE_DIR="$PWD/submission/build/output" GITHUB_RUN_ATTEMPT=1 GITHUB_RUN_ID=100001 GITHUB_SHA="$(git rev-parse HEAD)" GIT_MIRROR_DIR=/path/to/git-mirrors submission/build/setup.sh
```

Choose a numeric build ID that does not collide with an existing output. The command publishes `submission/build/output/100001.sqsh` and `100001.container-sidecar.json`; the OCI cache repository must be writable by Buildah. `build/build.log` is the complete container-build log from the successful build job in GitHub Actions run 34681001728; the overall workflow later failed in a cluster execution job. `build/build-test.log` records the package and runtime checks executed inside that image. `build/provenance.json` records the verified SquashFS and OCI digests, source commits, and build tool versions from that build.

On a Firmus login node, submit the official two-node run with:

```bash
CONTAINER_IMAGE=/path/to/sglang.sqsh sbatch run/submit.sbatch
```

The script uses `/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8` by default. It starts four TP2 prefill workers on the first node, two TP4/EP4 decode workers on the second node, and one SGLang PD router backed by NIXL. It resolves each compute hostname to IPv4 and supplies the same addresses to the router, health checks, and SGLang's internal control plane through `SGLANG_HOST_IP`. It also limits `UCX_NET_DEVICES` to active HCAs whose link layer is InfiniBand, excluding the two Ethernet HCAs present on Firmus nodes. AIPerf 0.12.0 runs the fixed official 60/120/30 second workload through the single OpenAI-compatible gateway. Logs, the raw `profile_export_aiperf.json`, and `score.json` are written below `results/logs/$SLURM_JOB_ID`.

Set `PROFILE=true` only for a separate diagnostic run. This records one-second GPU, CPU, NIC, and InfiniBand telemetry below the run's `profile` directory and must not be treated as the performance result.

The run records the exact AIPerf command and runtime version/tokenizer revision in `aiperf-command.txt` and `aiperf-provenance.txt`. Official validation checks these artifacts against the fixed workload before accepting input-length observations of 4000 or the pinned client's prefix-compensated 3986 tokens. Other input lengths, output lengths other than 200, and good request fractions below 0.90 fail validation. Raw observations and scoring remain unchanged; `score.json` also records `official_validation`. See the [input-length investigation](../research/aiperf-isl-validation.md) for the source and diagnostic evidence.

`analysis/profiling_data/16928` contains the recovered telemetry and aggregate notes from a separate concurrency-16 diagnostic run. That allocation reached the Firmus time limit during AIPerf warm-up and has no benchmark score; it is retained only to document the observed low utilization and transfer stall.
