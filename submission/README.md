# Qwen3 experimental submission on Firmus

**Submitted result: 10054 good output tok/s (measured: 10054.49).** This is the measured round 34 result, Actions [41767886175](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41767886175), Slurm **28660**, on **two nodes / 16 H200 GPUs**. The job completed with exit 0:0 in 14:52: 6333 requests, zero errors, good request fraction **0.98215696**. [Raw metrics and logs](results/41767886175/RESULT.md) accompany the submission.

**Classification: Bonus.** This configuration includes source patches and custom kernels. For the unmodified SGLang 0.5.23 Basic and the submission overview, start with [SUBMISSION.md](SUBMISSION.md).

## Results and evidence

| Record | Good output tok/s | Good request fraction | What it establishes |
| --- | ---: | ---: | --- |
| Submitted experimental result, run 41767886175 | **10054.49** | 0.98215696 | Highest observed score; all four kernel candidates selected |
| Packaged configuration with logprob relay, run 41781374312 | 10038.11 | 0.98183541 | Standalone preparation and serving completed; eight probe responses included token logprobs |
| Unmodified Basic, runs 41078642203 / 41093781934 | 6321.12 / 6339.31 | 0.97685992 / 0.97912525 | Two measurements of the fixed c160 configuration; mean 6330.21 |

The experimental configuration achieved **1.59× the Basic mean** on the same hardware. The configurations differ in code, Prefill parallelism, concurrency and draft precision. Against the organisers' **1832 tok/s** reference (job 25013), the result is **5.49×**.

The **10038.11 package verification run** used the kernel choices selected in the submitted run, plus `patches/overlay/sglang/34-pp4-direct-logprob.patch`. That patch relays logprobs in the direct PP4 Prefill loop. The official workload does not request logprobs; its path is unchanged. The standalone job completed 0:0 in 10:37 (Slurm 28673).

The submitted score comes from one run. Two runs of the same image scored 9998.67 and 10015.98 after their FC1 first-K gates rejected the candidate; earlier round 25 results ranged from 10002.75 to 10025.34. Because their selected kernels differ, these runs do not isolate the effect of FC1 first-K.

[EXPERIMENTS.md](EXPERIMENTS.md) maps the presented experiments to their configurations, scripts and logs. [patches/README.md](patches/README.md) explains each change and its numerical behavior. See [validation scope](SUBMISSION.md#validation-scope) for the checks performed and the remaining accuracy assessment.

## Topology and serving settings

The submission uses all 16 H200 GPUs on two nodes (from `config/serving.json` and `results/41767886175/server-info-*.json`):

| Setting | Prefill | Decode |
| --- | --- | --- |
| Workers | Three, four GPUs each | One, four GPUs |
| Parallelism | TP1 × PP4 × EP1, layers 24/24/24/22 | TP4/DP4/EP4, DP attention and DP LM head |
| MoE | MegaMoE, DeepGEMM runner | MegaMoE |
| Request capacity | 232 per worker | 192, plus 20 extra slots |
| Scheduling | At most two micro-batches, async batch depth 1, chunked prefill 8192, no overlap schedule | Polling interval 1 |
| CUDA graphs | Disabled | Batch sizes 1–48 |

Node 0 hosts prefill-0 and prefill-1; node 1 hosts prefill-2 and decode-0. The router assigns Prefill requests by minimum live load (`--prefill-policy manual --assignment-mode min_load`). Client concurrency is 232. Both roles use EAGLE3 with 3 steps / 4 draft tokens, FP8 draft weights, FP8 E4M3 KV and the gigatoken tokenizer backend.

`run/order-firmus-pp4-gpus.py` places the last, lightest Prefill stage on each worker’s slowest GPU at the 700 W cap, using `config/firmus-gpu-sm-clock.json`. In the measured run, prefill-1 and prefill-2 used visible order `0,1,3,2`; prefill-0 kept `0,1,2,3`.

`config/prefill-kernels.env` and `config/decode-kernels.env` fix the choices selected by the submitted run’s four in-job checks: Prefill FA3 pair, Prefill FA3 prepack P3, Decode greedy verify and Prefill MegaMoE FC1 first-K. Both files are byte-identical to the final environments used by that run. The selection records are in `results/41767886175/`.

The delivered job uses these fixed choices without tuning or profiling. Every Prefill stage prewarms FC1 first-K before readiness; `results/prefill-r32-dispatch-{0,1,2}.txt` records the prewarm and dispatch lines.

## Files

| Path | Contents |
| --- | --- |
| `build/` | Build entrypoint, source reconstruction and bundle, image inputs, original Buildah log and sidecar |
| `run/prepare.sh`, `run/submit.sbatch`, `run/worker.sh` | Login-node preparation, Slurm job and worker launcher |
| `run/official-aiperf.sh`, `run/parse-aiperf-result.py` | Official benchmark command and local scorer |
| `run/prepare_eagle.py`, `run/order-firmus-pp4-gpus.py`, `run/greedy-probe.py` | Draft config adapter, GPU placement and post-benchmark probe |
| `config/` | Serving settings, pinned role environments and GPU clock table |
| [patches/README.md](patches/README.md) | Patch and kernel inventory with hashes, application points and numerical arguments |
| `results/41767886175/` | Submitted result and kernel-selection records |
| `results/41781374312/` | Standalone package verification |
| [EXPERIMENTS.md](EXPERIMENTS.md), `experiments/` | Presented experiments and historical evidence |
| [analysis/README.md](analysis/README.md) | Optional round 24 profile, findings and checksums |

`build/source/` contains all image build inputs, copied byte-for-byte from the measured build’s source commit. Basic has separate inputs under `basic/build/source/`. The package can be built and run without access to the team’s private SGLang mirror, credentials or CI system.

The packaged scripts were simplified after the GPU runs while retaining the measured commands, flags and environment. CPU fixtures compare worker launch arguments, environment and the retained serving and benchmark steps with the recorded versions; the final scripts have not been run on GPUs.

## Build

Use a root Linux build host with Buildah, Skopeo, Umoci, SquashFS tools (`mksquashfs`, `unsquashfs`), Git, jq, Python 3, GNU coreutils and tar. `BUILD_DIR` must be a new path on local ext4 or XFS. The wrapper creates it and uses `${BUILD_DIR}/storage` exclusively for this build; `CONTAINER_ARCHIVE_DIR` must be outside it. The build uses `chroot` isolation, checks the OCI layout through Skopeo/Umoci, and produces an Enroot-compatible SquashFS. It does not require a GPU.

Network access is needed to GitHub, PyPI/files.pythonhosted.org, flashinfer.ai, the PyTorch package hosts, Ubuntu/NVIDIA package repositories, and the Rust toolchain/crates registries. `build/source/containers/Dockerfile` uses **Docker Hub `docker.io/nvidia/cuda@sha256:5a2d3b02eb7412847d051d0f2b0f0a5031057a0172d9ca78743cc41cfc5d037f`**, rather than an NGC image.

From the package root, create a new mirror directory and choose a new numeric build ID:

```bash
export GIT_MIRROR_DIR=/var/tmp/qwen-submission-mirrors
build/sglang-mirror.sh
sudo env BUILD_DIR=/var/tmp/qwen-submission-build CONTAINER_ARCHIVE_DIR="$PWD/build/output" GITHUB_RUN_ATTEMPT=1 GITHUB_RUN_ID=100001 GIT_MIRROR_DIR="$GIT_MIRROR_DIR" build/setup.sh
```

`setup.sh` runs `build/source/scripts/build-sglang-container.sh` with `build/source/` as its input directory (made a Git work tree first, since the script records file blob IDs) and labels the image with Qwen `3f9124fd8afb28e34c71288b6dc31e5c490c7f60`, the commit those inputs come from.

The mirror script shallow-fetches public SGLang `0d6bef6b6d2f4fa03187fb4d02773ad6e1256adb`, whose tree is `9ea4c7300d38bd9e024330b45bf5fc9cbf70f97d`. That tree matches the team mirror boundary `6b3506869d4840b18678de19d18d751a34f9b737`, recreated from the supplied raw commit object. The bundle imports three publication commits and produces HEAD `a60ed3656341ffbc37d5ddd73e7935b65fab08f4`, pinned by the standalone builder. The internal mirror-maintenance workflow was removed from all three commits. The runtime `python/sglang` tree and `python/uv.lock` are byte-identical to recorded source `fc275674a97d78553b22597ee72c8fd48f5420f4`; measurement provenance retains that original source ID. The legacy bundle filename identifies the recorded source from which it was exported. Serving changes remain the separate text patches listed in `patches/README.md`. `BUILD_CACHE_REPOSITORY` is optional.

Build outputs are `build/output/100001.sqsh` and `100001.container-sidecar.json`. The recorded measured image is build **41729881283**, built from Qwen `3f9124fd8afb28e34c71288b6dc31e5c490c7f60`: **12296638464 bytes**, SHA-256 **3a47b757bb88f6bc14ad5f151e8405f645007f25f381f631d88046d9bd1b7b04** (`build/provenance.json`). Every file in `build/source/` is byte-identical to that commit's copy; the one runtime overlay added since (`patches/overlay/sglang/34-pp4-direct-logprob.patch`) is applied at preparation, not in the image. `build/build.log` is that build's raw Buildah log (all 57 steps built without a cache hit). Bit-for-bit reproducibility of a fresh container archive has not been established.

The recorded build used a temporary Ubuntu 24.04 EC2 m7i.2xlarge host and the team’s private SGLang mirror. The supplied mirror script reconstructs a public snapshot with identical runtime Python and dependency lock from public sources and the included bundle; the original measured commit remains in the provenance. The original build environment and steps are recorded in `build/build.log` and `build/provenance.json`.

For a new build, update the image integrity fields in the deployment copy of `config/serving.json` before preparation:

```bash
jq --slurpfile built build/output/100001.container-sidecar.json '.images.sglang = ($built[0].container_artifact | {path:"/path/to/100001.sqsh",sha256,size_bytes})' config/serving.json >config/serving.json.tmp
mv config/serving.json.tmp config/serving.json
```

## Prepare and run

Copy this folder, the built image and its sidecar to storage accessible to the dedicated account on Firmus. On the login node, preparation requires **unsquashfs, git, jq, python3, sha256sum, uvx**, GNU coreutils and network access to Hugging Face for the pinned public draft. Slurm, Pyxis/Enroot and the site GPU/InfiniBand drivers are required for execution. The image, model and new run directory must be readable on both allocated nodes. Stage the complete model checkpoint as regular files; external shard symlinks are not supported. The job mounts the model read-only at its host path; overlays and the DeepGEMM cache are in that shared run directory. Triton and Inductor use node-local `/tmp` caches.

`prepare.sh RUN_DIR IMAGE MODEL TOKENIZER_SNAPSHOT` stages the run directory before GPU allocation. It records image and tokenizer hashes, stages the tokenizer cache and pinned draft, applies 34 SGLang and 14 DeepGEMM overlays, installs the role environments and GPU clock table, and prints the submission command. Use a pre-staged tokenizer snapshot of `Qwen/Qwen3-235B-A22B-Instruct-2507` revision `ac9c66cc9b46af7306746a9250f23d47083d689e`.

For the recorded image and site snapshots, from this folder:

```bash
run/prepare.sh "/scratch/home/$USER/qwen-submission-001" /scratch/home/ACCOUNT/qwen/containers/41729881283.sqsh /scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8 /scratch/home/ACCOUNT/qwen/framework-comparison/sources/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e
cd "/scratch/home/$USER/qwen-submission-001"
sbatch scripts/submit.sbatch
```

For a rebuild, substitute the new image's shared path and a tokenizer snapshot readable by the organisers' account. The model is the published `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, revision `e156cb4efae43fbee1a1ab073f946a1377e6b969`. Preparation happens before allocating GPUs; the compute job uses offline caches.

The single job requests partition `hpcai`, two nodes, eight GPUs per node, and 17 minutes. It launches the workers and router, waits for readiness, records each worker's server info, sends one non-dataset connection warm-up chat through the gateway, freezes the startup heap of every server process (`/freeze_gc`), then runs the unchanged official AIPerf 0.12.0 command (60/120/30 seconds, seed 2026, closed-loop c232). After writing `score.json`, it marks benchmark completion, extracts kernel dispatch lines and runs the greedy probe. A diagnostic failure is logged without changing the completed benchmark’s exit status. Server info is captured once, before the benchmark. There is no profiling option.

Allow **about 11–13 minutes** for the job, excluding preparation and queue time. The package verification run (41781374312, before the script simplification) took 10:37 (READY after 265 s, then the 60/120/30 s benchmark and the greedy probe); earlier standalone runs of the previous configuration took 10:47 and 13:05. Cold startup and compilation time can vary.

## Outputs

Preserve `RUN_DIR/logs/`, `RUN_DIR/results/` and `RUN_DIR/source/` before deleting the run directory. `AIPERF ok` in `logs/slurm-<job>.log` and `run-success` mark completion of the official benchmark and scorer. Check `score.json` separately for SLA eligibility. The probe has a 60-second overall timeout and is skipped when less than 90 seconds remain or Slurm does not provide the allocation end time. Later diagnostic warnings do not invalidate the benchmark result; inspect `logs/greedy-probe.log` and the dispatch files for diagnostic completion.

| Output | Contents |
| --- | --- |
| `results/profile_export_aiperf.json` | Official metrics; **good output tok/s = `goodput.avg × 200`** |
| `results/score.json` | Score, SLA eligibility (good request fraction ≥ 0.90), request/error/OSL mismatch counts and length statistics |
| `aiperf-command.txt`, `aiperf-provenance.txt` | Benchmark command and provenance |
| Startup server-info files and warm-up chat response | Worker configuration and connection check |
| `decode-greedy-dispatch.txt` | Selected or fallback kernels, vocabulary and stride, one line per Decode rank |
| `prefill-r32-dispatch-{0,1,2}.txt` | FC1 first-K prewarm and kernel dispatch for all Prefill stages |
| Draft provenance and overlay manifests | Downloaded weight hash and applied patch hashes |

`results/greedy-parity.json` is the greedy probe: eight fixed prompts sent through the same gateway after the official benchmark, with `temperature 0`, `max_tokens`/`min_tokens` 64, `ignore_eos` and token `logprobs`. It changes no AIPerf input and runs after `score.json` is written. Use it to compare outputs before and after a change. It is a short diagnostic probe; full-set accuracy is assessed separately by the organisers.

## Draft declaration

The sole draft is public [nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507), revision `ff0f0c0999f6c86b21dfd2f5803eebc4ac712224`: **572,538,880 parameters, BF16, 1,145,943,008 weight bytes** (CC-BY-4.0). `run/prepare_eagle.py` changes only the architecture selector and explicit dtype declaration; weights are unchanged. The downloaded weight SHA-256, recorded in `source/draft-provenance.json`, is `bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86`. The workers load it with `--speculative-draft-model-quantization fp8` (`speculation.draft_quantization` in `config/serving.json`); each worker's `results/server-info-*.json` reports `speculative_draft_model_quantization: "fp8"`. No draft is trained or fine-tuned by this package.
