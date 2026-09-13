# Qwen3 submission

`build/setup.sh` invokes the pinned Buildah recipe from this repository. The repository root, including `containers/` and `scripts/build-sglang-container.sh`, is part of the build input; `submission/` alone is not a standalone container-build context. The recipe checks out SGLang commit `fc275674a97d78553b22597ee72c8fd48f5420f4` and installs the versions fixed by the container files. It requires root, Buildah, Skopeo, Umoci, SquashFS tools, Git, GitHub CLI, and jq. `GIT_MIRROR_DIR` may contain a local bare mirror named `5_SGLang_Src.git`; otherwise the recipe clones the repository with GitHub CLI.

From the repository root, build an Enroot-compatible SquashFS with a writable temporary directory, output directory, and OCI cache repository:

```bash
sudo env BUILD_CACHE_REPOSITORY=registry.example/team/qwen-build-cache BUILD_DIR=/var/tmp/qwen-build CONTAINER_ARCHIVE_DIR="$PWD/submission/build/output" GITHUB_RUN_ATTEMPT=1 GITHUB_RUN_ID=100001 GITHUB_SHA="$(git rev-parse HEAD)" GIT_MIRROR_DIR=/path/to/git-mirrors submission/build/setup.sh
```

Choose a numeric build ID that does not collide with an existing output. The command publishes `submission/build/output/100001.sqsh` and `100001.container-sidecar.json`; the OCI cache repository must be writable by Buildah. `build/build.log` is the complete container-build log from the successful build job in GitHub Actions run 34681001728; the overall workflow later failed in a cluster execution job. `build/build-test.log` records the package and runtime checks executed inside that image. `build/provenance.json` records the verified SquashFS and OCI digests, source commits, and build tool versions from that build.

On a Firmus login node, submit the official two-node run with:

```bash
CONTAINER_IMAGE=/path/to/sglang.sqsh TOKENIZER_PATH=/path/to/Qwen3-235B-A22B-Instruct-2507-ac9c66cc sbatch run/submit.sbatch
```

The script uses `/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8` by default. `TOKENIZER_PATH` must point to a pre-staged snapshot of `Qwen/Qwen3-235B-A22B-Instruct-2507` revision `ac9c66cc9b46af7306746a9250f23d47083d689e`; the compute job does not access the network. Before allocating model memory, the script verifies the model and all 24 shard metadata revisions, the index shard set, the Qwen3 MoE FP8 configuration, and the exact tokenizer-file hashes. It copies the tokenizer into the mounted run cache, starts four TP2 prefill workers on the first node, two TP4/EP4 decode workers on the second node, and one SGLang PD router backed by NIXL. It resolves each compute hostname to IPv4 and supplies the same addresses to the router, health checks, and SGLang's internal control plane through `SGLANG_HOST_IP`. It sets `UCX_NET_DEVICES` to all ten active Firmus HCAs: eight InfiniBand and two Ethernet. After all workers and the router are ready, AIPerf 0.12.0 immediately runs the fixed official 60/120/30 second workload through the single OpenAI-compatible gateway. Logs, the raw `profile_export_aiperf.json`, and `score.json` are written below `results/logs/$SLURM_JOB_ID`.

Set `PROFILE=true` only for a separate diagnostic run. This records one-second GPU, CPU, NIC, and InfiniBand telemetry below the run's `profile` directory and must not be treated as the performance result.

The run records the exact AIPerf command and runtime version/tokenizer revision in `aiperf-command.txt` and `aiperf-provenance.txt`. The local validator checks those artifacts and the raw export against the fixed official workload before accepting the pinned client's cache-bust input-length distribution. For the configured OSL of 200, it also requires internally consistent `min`, `max`, `count`, and `sum` observations within AIPerf 0.12.0's pinned 190–210 tolerance and an `osl_mismatch_count` of zero. Good request fractions below 0.90 fail validation. Raw observations remain unchanged, and the score remains goodput multiplied by the configured 200 tokens; `score.json` records the local validation result under `official_validation`. See the [token-length investigation](../research/aiperf-isl-validation.md) for the source and diagnostic evidence.

`analysis/profiling_data/16928` contains the recovered telemetry and aggregate notes from a separate concurrency-16 diagnostic run. That allocation reached the Firmus time limit during AIPerf warm-up and has no benchmark score; it is retained only to document the observed low utilization and transfer stall.

`results/17005` contains the complete service logs and raw AIPerf export from the fixed official 60/120/30-second workload on Firmus Slurm job 17005. The workload completed 558 requests with zero errors, a good request fraction of 1.0, goodput of 4.536230620411887 requests/s, and a score of 907.2461240823774 output tokens/s. The original wrapper exited with code 1 because its OSL check incorrectly required an exact average of 200; `score-original.json` preserves that rejection, while `score.json` records the corrected parser replay with `official_validation.valid` set to true. `results/17005/RESULT.md` documents the distinction and the replay command.
