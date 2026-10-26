# Qwen3 PD submission — Firmus, Tokyo Rapids

The measured best run is **9120.51 good output tok/s**, Actions [40225743220](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40225743220), Slurm 26232: 5802 requests, zero errors, good request fraction 0.97708. This is one measured run, not a repeatability claim. Its text evidence and checksums are in `results/40225743220/`.

Standalone re-run of this package: **9132.03 good output tok/s**, Actions [40273247322](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40273247322), Slurm 26298, COMPLETED 0:0 in 9m13s: 5801 requests, zero errors, good request fraction 0.97776. It ran exactly `run/prepare.sh` followed by `sbatch scripts/submit.sbatch` on the recorded image; evidence is in `results/40273247322/`.

Common baseline: **1,832 good output tok/s** on Firmus 2×8 H200 (official task `5. Common Baselines.md`, job 25013). The baseline uses different serving settings.

The submission uses all 16 H200 GPUs: **P3 TP4/DP4/EP4 Prefill + D1 TP4/DP4/EP4 Decode**, concurrency 240, MegaMoE, EAGLE3 steps/draft tokens **3/4**, and **FP8 E4M3 KV**. Node 0 hosts two Prefill workers; node 1 hosts the third Prefill worker and Decode. The router uses manual Prefill assignment with minimum live load. `config/serving.json` and the single `run/submit.sbatch` pin the serving configuration. The measured kernel choices are fixed in role-specific env files before worker launch; the tuning screens and diagnostics have been removed.

## What is here

- `build/`: build entrypoint, public-source mirror reconstruction, three-commit Git bundle, and build 40197369083's original Buildah log and image sidecar.
- `run/`: login-node preparation, one Slurm serving/benchmark job, unchanged AIPerf wrapper and local scorer, and unchanged EAGLE preparation helper.
- `config/`: serving settings and static Prefill expert placement.
- [patches/README.md](patches/README.md): every image patch, kernel source and runtime overlay, with hashes, application points and numerical classifications (unestablished classes are marked `to confirm`).
- `results/40225743220/`: curated evidence of the measured tuning run.
- `results/40273247322/`: curated evidence of the standalone re-run of this package.

The whole accompanying repository is required for rebuilding: `containers/`, `scripts/build-sglang-container.sh`, repository patches and `kernels/` are build inputs. Once an image exists, `submission/` is sufficient for preparation and execution. No private SGLang mirror, team credentials, GitHub Actions, or team-specific runtime directory is needed by the organisers.

## Build

Use a root Linux build host with Buildah, Skopeo, Umoci, SquashFS tools (`mksquashfs`, `unsquashfs`), Git, jq, Python 3, GNU coreutils and tar. Buildah storage must be on local ext4 or XFS. The build uses `chroot` isolation, checks the OCI layout through Skopeo/Umoci, and produces an Enroot-compatible SquashFS. It does not require a GPU.

Network access is needed to GitHub, PyPI/files.pythonhosted.org, flashinfer.ai, the PyTorch package hosts, Ubuntu/NVIDIA package repositories, and the Rust toolchain/crates registries. `containers/Dockerfile` uses **Docker Hub `docker.io/nvidia/cuda@sha256:5a2d3b02eb7412847d051d0f2b0f0a5031057a0172d9ca78743cc41cfc5d037f`**, rather than an NGC image.

From the repository root, create a new mirror directory and choose a new numeric build ID:

```bash
export GIT_MIRROR_DIR=/var/tmp/qwen-submission-mirrors
submission/build/sglang-mirror.sh
sudo env BUILD_DIR=/var/tmp/qwen-submission-build CONTAINER_ARCHIVE_DIR="$PWD/submission/build/output" GITHUB_RUN_ATTEMPT=1 GITHUB_RUN_ID=100001 GITHUB_SHA="$(git rev-parse HEAD)" GIT_MIRROR_DIR="$GIT_MIRROR_DIR" submission/build/setup.sh
```

The mirror script shallow-fetches public SGLang `0d6bef6b6d2f4fa03187fb4d02773ad6e1256adb`, whose tree is `9ea4c7300d38bd9e024330b45bf5fc9cbf70f97d`. That tree matches the team mirror boundary `6b3506869d4840b18678de19d18d751a34f9b737`, recreated from the supplied raw commit object. The bundle imports three publication commits and produces HEAD `a60ed3656341ffbc37d5ddd73e7935b65fab08f4`, pinned by the standalone builder. The internal mirror-maintenance workflow was removed from all three commits. The runtime `python/sglang` tree and `python/uv.lock` are byte-identical to recorded source `fc275674a97d78553b22597ee72c8fd48f5420f4`; measurement provenance retains that original source ID. The legacy bundle filename identifies the recorded source from which it was exported. Serving changes remain the separate text patches listed in `patches/README.md`. `BUILD_CACHE_REPOSITORY` is optional.

Build outputs are `submission/build/output/100001.sqsh` and `100001.container-sidecar.json`. The recorded measured image is build **40197369083**, built from Qwen `cbca2b30514dae898921867ae47c83db4acea8ab`: **12272513024 bytes**, SHA-256 **1f247668933c7252d3dfbe8f6c23897d354f095d56842788d4415a85d7dc5e55**. The serving source, kernel and dependency inputs match that build; registry-cache use is now optional. Bit-for-bit reproducibility of a fresh container archive has not been established.

For a new build, in the organisers' deployment copy, record that build's image integrity fields before preparation; this changes no serving parameter:

```bash
jq --slurpfile built submission/build/output/100001.container-sidecar.json '.images.sglang = ($built[0].container_artifact | {sha256,size_bytes})' submission/config/serving.json >submission/config/serving.json.tmp
mv submission/config/serving.json.tmp submission/config/serving.json
```

## Prepare and run

Copy `submission/`, the built image and its sidecar to storage accessible to the dedicated account on Firmus. On the login node, preparation requires **unsquashfs, git, jq, python3, sha256sum, uvx**, GNU coreutils and network access to Hugging Face for the pinned public draft. Slurm, Pyxis/Enroot and the site GPU/InfiniBand drivers are required for execution. The image, model and new run directory must be readable on both allocated nodes; all generated overlays and caches are in that shared run directory.

`prepare.sh RUN_DIR IMAGE MODEL TOKENIZER_SNAPSHOT` takes four required positional arguments and fails if RUN_DIR already exists. It verifies image bytes, the official FP8 checkpoint revision/configuration and all 24 shards, stages the tokenizer cache, downloads/verifies the draft, and applies the numbered overlays to packages extracted from the image. It prints the submission command. Use a pre-staged tokenizer snapshot of `Qwen/Qwen3-235B-A22B-Instruct-2507` revision `ac9c66cc9b46af7306746a9250f23d47083d689e`.

For the recorded image and site snapshots, from the repository root:

```bash
submission/run/prepare.sh "/scratch/home/$USER/qwen-submission-001" /scratch/home/ACCOUNT/qwen/containers/40197369083.sqsh /scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8 /scratch/home/ACCOUNT/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e
cd "/scratch/home/$USER/qwen-submission-001"
sbatch scripts/submit.sbatch
```

For a rebuild, substitute the new image's shared path and a tokenizer snapshot readable by the organisers' account. The model is the published `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, revision `e156cb4efae43fbee1a1ab073f946a1377e6b969`. Preparation happens before allocating GPUs; the compute job uses offline caches.

The single job requests partition `hpcai`, two nodes, eight GPUs per node, and 17 minutes. It launches the workers and router, waits for readiness, captures server info, sends one non-dataset connection warm-up chat through the gateway, then runs the unchanged official AIPerf 0.12.0 command (60/120/30 seconds, seed 2026, closed-loop c240). There is no profiling option.

Allow **about 10 minutes** for the job, excluding preparation and queue time. The standalone re-run took 9m13s (Slurm accounting): readiness after 264 s, then the warm-up request and the official AIPerf phases. Cold startup and compilation time can vary.

Outputs are in `RUN_DIR/logs/`, `RUN_DIR/results/` and `RUN_DIR/source/`. `logs/slurm-<job>.log` ends with `AIPERF ok` and `run-success` exists on success. The raw official metrics are `results/profile_export_aiperf.json`; **good output tok/s = `goodput.avg × 200`**. `results/score.json` records that value as `good_output_tokens_per_second`, eligibility and the repository-local official-workload checks. This scorer is not an organiser accuracy check. `aiperf-command.txt`, `aiperf-provenance.txt`, server-info files, the sanity answer, draft evidence and overlay hashes remain available for review. Preserve these outputs before deleting the run directory.

## Draft declaration

The sole draft is public [nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507), revision `ff0f0c0999f6c86b21dfd2f5803eebc4ac712224`: **572,538,880 parameters, BF16, 1,145,943,008 weight bytes** (CC-BY-4.0). `run/prepare_eagle.py` changes only the architecture selector and explicit dtype declaration; weights are unchanged. The downloaded weight SHA-256 is checked against `bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86`. No draft is trained or fine-tuned by this package.
