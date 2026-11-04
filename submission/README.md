# Qwen3 PD Bonus submission — Firmus, Tokyo Rapids

**Classification: Bonus.** This package contains team source and kernel changes. The separately tuned, unmodified SGLang Basic entrypoint is [basic/README.md](basic/README.md), with two selected c160 measurements averaging 6330.21 good output tok/s. Basic's public image and serving path are separate from the patches in this package.

The package is the PP4 round 34 configuration, measured at **10054.49 good output tok/s**: Actions [41767886175](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41767886175), Slurm 28660, COMPLETED 0:0 in 14:52, 6333 requests, zero errors, good request fraction 0.98216, eligible. That is the team's highest score on Firmus and **1.59×** the team's Basic mean of 6330.21 on the same 16 H200 GPUs. Its text evidence, including the four in-job gate records behind the pinned env files, is in `results/41767886175/`. This is one measured run, not a repeatability claim: two runs of the same image in which the new FC1 first-K gate rejected its kernel scored 9998.67 and 10015.98, and round 25 (the previous configuration, without that kernel) scored 10002.75 to 10025.34.

The package adds one runtime overlay to that run, `patches/overlay/sglang/34-pp4-direct-logprob.patch`. The team's direct PP4 Prefill loop did not relay the last stage's logprob outputs to the other stages, so a request with `logprobs: true` stopped the Prefill scheduler. The package's greedy probe found this on the round 25 configuration (Actions [41694678252](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41694678252), which scored 10025.34 before the probe). The overlay relays them with each result, as the upstream PP loop does. On round 25 it was then verified in Actions [41766980300](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41766980300) (10016.32, eligible, every probe response with 64 token logprobs, no Prefill error). Requests without logprobs, including the official AIPerf workload, take the unchanged path.

The package as shipped, including overlay 34, was then run standalone: Actions [41781374312](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41781374312), Slurm 28673, COMPLETED 0:0 in 10:37, **10038.11 good output tok/s**, 6331 requests, zero errors, good request fraction 0.98184, eligible; every greedy probe response carried 64 token logprobs. Its evidence is in `results/41781374312/`.

Common baseline: **1,832 good output tok/s** on Firmus 2×8 H200 (official task `5. Common Baselines.md`, job 25013). The baseline uses different serving settings.

## Topology and serving settings

The submission uses all 16 H200 GPUs on two nodes (from `config/serving.json` and `results/41767886175/server-info-*.json`):

- **Prefill: three workers, each TP1 × PP4 × EP1** on four GPUs (one pipeline stage per GPU, layer partition 24/24/24/22, at most two micro-batches, async batch depth 1, chunked prefill 8192, max 232 running requests, MegaMoE with the DeepGEMM runner, no CUDA graph, no overlap schedule). The last, lightest stage is moved to each worker's slowest GPU at the 700 W cap by `run/order-firmus-pp4-gpus.py` with the measured clock table `config/firmus-gpu-sm-clock.json` (in the measured run prefill-1 and prefill-2 used visible order `0,1,3,2`, prefill-0 kept `0,1,2,3`).
- **Decode: one worker, TP4/DP4/EP4** with DP attention and DP LM head, MegaMoE, max 192 running requests, 20 extra slots, polling interval 1, CUDA graph batch sizes 1–48.
- Node 0 hosts prefill-0 and prefill-1; node 1 hosts prefill-2 and decode-0. The router uses manual Prefill assignment with minimum live load (`--prefill-policy manual --assignment-mode min_load`).
- Client concurrency 232, EAGLE3 steps/draft tokens **3/4** with the draft loaded with FP8 weight quantization, **FP8 E4M3 KV**, gigatoken tokenizer backend.

The kernel choices are fixed in `config/prefill-kernels.env` and `config/decode-kernels.env`, byte-identical to the final env files the workers of run 41767886175 sourced after its four in-job gates (Prefill FA3 pair, Prefill FA3 prepack P3, Decode greedy verify, Prefill MegaMoE FC1 first-K) all adopted their candidates. The gates, selection screens, GPU telemetry, EAGLE acceptance collector and profiling branches are removed; the gate records are kept as evidence in `results/41767886175/`. Every Prefill stage prewarms the adopted FC1 first-K kernel before readiness, and the job records each worker's prewarm and dispatch lines in `results/prefill-r32-dispatch-{0,1,2}.txt`.

## What is here

- `build/`: build entrypoint, public-source mirror reconstruction, three-commit Git bundle, and build 41729881283's original Buildah log and image sidecar.
- `run/`: login-node preparation (`prepare.sh`), the single Slurm serving/benchmark job (`submit.sbatch`) and the worker launcher it runs in the container (`worker.sh`), the official AIPerf command and the local scorer (`official-aiperf.sh`, `parse-aiperf-result.py`), the draft config adapter (`prepare_eagle.py`), the PP4 GPU-order helper (`order-firmus-pp4-gpus.py`) and the post-benchmark greedy probe (`greedy-probe.py`).
- `config/`: serving settings, the pinned role env files and the PP4 GPU clock table.
- [patches/README.md](patches/README.md): every image patch, kernel source and runtime overlay, with hashes, application points and numerical classifications (bit-identical, FP-order, rounding-placement or no numeric effect, with the basis for each).
- `results/41767886175/`: curated evidence of the round 34 run the package was frozen from.
- `results/41781374312/`: curated evidence of the standalone run of this package.
- [analysis/README.md](analysis/README.md): the optional profiling data, the round 24 Nsight capture of the 10k configuration, with its relation to this configuration, its findings and the file list with checksums.

This folder is self-contained: `build/source/` holds every image build input (`containers/`, `kernels/`, `patches/`, `scripts/`, `configs/`) as byte-identical copies of the team repository at the commit the measured image was built from, so the image can be rebuilt, prepared and run from this folder alone. Basic has its own build inputs under `basic/build/source/`. No private SGLang mirror, team credentials, CI system or team-specific runtime directory is needed.

The scripts in `run/` were rewritten for this package from the ones the measured runs used, for readability and down to the steps of a successful run. Branches for earlier configurations, argument and input checks, traps, timeouts, fallbacks and diagnostics were removed; the worker launcher, a heredoc inside `submit.sbatch` before, is `worker.sh`. The commands, flags and environment the workers and the benchmark receive are unchanged:

- The team's test harness runs `prepare.sh` and `submit.sbatch` with every cluster command stubbed and requires what the scripts of run 41694678252 (round 25) do on this configuration: the same srun steps and argv, the same `launch_server` argv and environment for every worker, the same ordered HTTP calls, the same image extractions and patch applications (without the separate `git apply --check`), and the same run-directory files.
- `official-aiperf.sh` runs the same official AIPerf command, now directly instead of through a wrapper that only stopped a failed client start. `parse-aiperf-result.py` computes the score from the export and no longer re-validates the command; on the exports in `results/` and `basic/results/` it reproduces every value it reports of the recorded `score.json`.
- `prepare_eagle.py` makes the same two config changes and the same records, without its input checks. `order-firmus-pp4-gpus.py` applies the same rule. `greedy-probe.py` sends the same requests.

## Build

Use a root Linux build host with Buildah, Skopeo, Umoci, SquashFS tools (`mksquashfs`, `unsquashfs`), Git, jq, Python 3, GNU coreutils and tar. Buildah storage must be on local ext4 or XFS. The build uses `chroot` isolation, checks the OCI layout through Skopeo/Umoci, and produces an Enroot-compatible SquashFS. It does not require a GPU.

Network access is needed to GitHub, PyPI/files.pythonhosted.org, flashinfer.ai, the PyTorch package hosts, Ubuntu/NVIDIA package repositories, and the Rust toolchain/crates registries. `build/source/containers/Dockerfile` uses **Docker Hub `docker.io/nvidia/cuda@sha256:5a2d3b02eb7412847d051d0f2b0f0a5031057a0172d9ca78743cc41cfc5d037f`**, rather than an NGC image.

From the top-level folder of the submission (the parent of `submission/`), create a new mirror directory and choose a new numeric build ID:

```bash
export GIT_MIRROR_DIR=/var/tmp/qwen-submission-mirrors
submission/build/sglang-mirror.sh
sudo env BUILD_DIR=/var/tmp/qwen-submission-build CONTAINER_ARCHIVE_DIR="$PWD/submission/build/output" GITHUB_RUN_ATTEMPT=1 GITHUB_RUN_ID=100001 GIT_MIRROR_DIR="$GIT_MIRROR_DIR" submission/build/setup.sh
```

`setup.sh` runs `build/source/scripts/build-sglang-container.sh` with `build/source/` as its input directory (made a Git work tree first, since the script records file blob IDs) and labels the image with Qwen `3f9124fd8afb28e34c71288b6dc31e5c490c7f60`, the commit those inputs come from.

The mirror script shallow-fetches public SGLang `0d6bef6b6d2f4fa03187fb4d02773ad6e1256adb`, whose tree is `9ea4c7300d38bd9e024330b45bf5fc9cbf70f97d`. That tree matches the team mirror boundary `6b3506869d4840b18678de19d18d751a34f9b737`, recreated from the supplied raw commit object. The bundle imports three publication commits and produces HEAD `a60ed3656341ffbc37d5ddd73e7935b65fab08f4`, pinned by the standalone builder. The internal mirror-maintenance workflow was removed from all three commits. The runtime `python/sglang` tree and `python/uv.lock` are byte-identical to recorded source `fc275674a97d78553b22597ee72c8fd48f5420f4`; measurement provenance retains that original source ID. The legacy bundle filename identifies the recorded source from which it was exported. Serving changes remain the separate text patches listed in `patches/README.md`. `BUILD_CACHE_REPOSITORY` is optional.

Build outputs are `submission/build/output/100001.sqsh` and `100001.container-sidecar.json`. The recorded measured image is build **41729881283**, built from Qwen `3f9124fd8afb28e34c71288b6dc31e5c490c7f60`: **12296638464 bytes**, SHA-256 **3a47b757bb88f6bc14ad5f151e8405f645007f25f381f631d88046d9bd1b7b04** (`build/provenance.json`). Every file in `build/source/` is byte-identical to that commit's copy; the one runtime overlay added since (`patches/overlay/sglang/34-pp4-direct-logprob.patch`) is applied at preparation, not in the image. `build/build.log` is that build's raw Buildah log (all 57 steps built without a cache hit). Bit-for-bit reproducibility of a fresh container archive has not been established.

How the recorded image was built differs from the commands above only in the host and the source mirror. The team's CI archived the Qwen checkout together with a full `git clone --mirror` of the team's private SGLang repository, started a temporary EC2 m7i.2xlarge with Ubuntu 24.04 (with Buildah, Skopeo, Umoci, SquashFS tools, Git, jq and zstd installed), set `BUILD_DIR`, `BUILD_STORAGE_ROOT=/var/lib/containers/storage`, `CONTAINER_ARCHIVE_DIR`, `GITHUB_*`, `GIT_MIRROR_DIR` (the private mirror) and `BUILD_CACHE_REPOSITORY` (an ECR repository, written to but never hit by this build), and called the same `build-sglang-container.sh` that `build/setup.sh` executes. Both mirrors check out the same `fc275674` commit.

For a new build, in the organisers' deployment copy, record that build's image integrity fields before preparation; this changes no serving parameter:

```bash
jq --slurpfile built submission/build/output/100001.container-sidecar.json '.images.sglang = ($built[0].container_artifact | {path:"/path/to/100001.sqsh",sha256,size_bytes})' submission/config/serving.json >submission/config/serving.json.tmp
mv submission/config/serving.json.tmp submission/config/serving.json
```

## Prepare and run

Copy `submission/`, the built image and its sidecar to storage accessible to the dedicated account on Firmus. On the login node, preparation requires **unsquashfs, git, jq, python3, sha256sum, uvx**, GNU coreutils and network access to Hugging Face for the pinned public draft. Slurm, Pyxis/Enroot and the site GPU/InfiniBand drivers are required for execution. The image, model and new run directory must be readable on both allocated nodes; all generated overlays and caches are in that shared run directory.

`prepare.sh RUN_DIR IMAGE MODEL TOKENIZER_SNAPSHOT` creates RUN_DIR, records the image's and tokenizer files' SHA-256 values and the model reference, stages the tokenizer cache, downloads the pinned draft and declares its SGLang architecture and dtype (`prepare_eagle.py`), applies the 34 SGLang and 14 DeepGEMM numbered overlays to packages extracted from the image, installs the pinned role env files and the GPU clock table, and prints the submission command. Use a pre-staged tokenizer snapshot of `Qwen/Qwen3-235B-A22B-Instruct-2507` revision `ac9c66cc9b46af7306746a9250f23d47083d689e`.

For the recorded image and site snapshots, from the top-level folder of the submission:

```bash
submission/run/prepare.sh "/scratch/home/$USER/qwen-submission-001" /scratch/home/ACCOUNT/qwen/containers/41729881283.sqsh /scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8 /scratch/home/ACCOUNT/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e
cd "/scratch/home/$USER/qwen-submission-001"
sbatch scripts/submit.sbatch
```

For a rebuild, substitute the new image's shared path and a tokenizer snapshot readable by the organisers' account. The model is the published `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, revision `e156cb4efae43fbee1a1ab073f946a1377e6b969`. Preparation happens before allocating GPUs; the compute job uses offline caches.

The single job requests partition `hpcai`, two nodes, eight GPUs per node, and 17 minutes. It launches the workers and router, waits for readiness, records each worker's server info, sends one non-dataset connection warm-up chat through the gateway, freezes the startup heap of every server process (`/freeze_gc`), then runs the unchanged official AIPerf 0.12.0 command (60/120/30 seconds, seed 2026, closed-loop c232). After the score exists it records the Decode greedy verify and Prefill FC1 first-K dispatch lines, captures server info again, and runs the greedy probe. There is no profiling option.

Allow **about 11–13 minutes** for the job, excluding preparation and queue time. The standalone run of this package took 10:37 (READY after 265 s, then the 60/120/30 s benchmark and the greedy probe); earlier standalone runs of the previous configuration took 10:47 and 13:05. Cold startup and compilation time can vary.

## Outputs

Outputs are in `RUN_DIR/logs/`, `RUN_DIR/results/` and `RUN_DIR/source/`. `logs/slurm-<job>.log` ends with `AIPERF ok` and `run-success` exists on success. The raw official metrics are `results/profile_export_aiperf.json`; **good output tok/s = `goodput.avg × 200`**. `results/score.json` records that value as `good_output_tokens_per_second`, eligibility (good request fraction of at least 0.90), the request, error and OSL mismatch counts, and the input and output length statistics. This scorer is not an organiser accuracy check. `aiperf-command.txt`, `aiperf-provenance.txt`, server-info files, the sanity answer, `decode-greedy-dispatch.txt` (one line per Decode rank: fused kernels or fallback, with vocabulary and stride), `prefill-r32-dispatch-{0,1,2}.txt` (each Prefill stage's FC1 first-K prewarm and kernel), draft evidence and overlay hashes remain available for review. Preserve these outputs before deleting the run directory.

`results/greedy-parity.json` is the greedy probe: eight fixed prompts sent through the same gateway after the official benchmark, with `temperature 0`, `max_tokens`/`min_tokens` 64, `ignore_eos` and token `logprobs`. It changes no AIPerf input and runs after `score.json` exists. It exists so the team can compare outputs before and after a change; it is not an accuracy measurement, and the organisers run the full-set accuracy check.

## Draft declaration

The sole draft is public [nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507), revision `ff0f0c0999f6c86b21dfd2f5803eebc4ac712224`: **572,538,880 parameters, BF16, 1,145,943,008 weight bytes** (CC-BY-4.0). `run/prepare_eagle.py` changes only the architecture selector and explicit dtype declaration; weights are unchanged. The downloaded weight SHA-256, recorded in `source/draft-provenance.json`, is `bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86`. The workers load it with `--speculative-draft-model-quantization fp8` (`speculation.draft_quantization` in `config/serving.json`); each worker's `results/server-info-*.json` reports `speculative_draft_model_quantization: "fp8"`. No draft is trained or fine-tuned by this package.
