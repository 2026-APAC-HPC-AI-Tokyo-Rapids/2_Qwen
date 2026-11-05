# Qwen3 PD Basic submission — Firmus, Tokyo Rapids

**Classification: Basic.** This is the submission the organisers re-run. It serves the official FP8 checkpoint with the unmodified public SGLang **0.5.23** release (upstream commit `297e54fd105e974cdf401e5b4d20253a22c16f90`) on Firmus, 2 nodes / 16 H200. Only built-in options, environment variables and deployment settings are tuned. No source or kernel patches or runtime overlays are applied. The team's submitted experimental result is **10054.49 good output tok/s**, documented separately in the [Bonus package](../README.md).

**Selected run to re-run:** Actions [41093781934](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41093781934), Slurm **27709**, **6339.31 good output tok/s**, good request fraction **0.97912525**, concurrency **160**. Use `basic/prepare.sh` and `basic/submit.sbatch` with `basic/config/firmus-basic.json`.

The two c160 runs averaged **6330.21 good output tok/s**, with zero request errors:

| Actions run | Slurm job | State | Good output tok/s | Good request fraction | Requests | Evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| [41078642203](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41078642203) | 27682 | COMPLETED 0:0, 10:04 | 6321.12 | 0.97686 | 4019 | [results/41078642203/](results/41078642203/RESULT.md) |
| [41093781934](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41093781934) | 27709 | COMPLETED 0:0, 10:35 | 6339.31 | 0.97913 | 4024 | [results/41093781934/](results/41093781934/RESULT.md) |

The otherwise identical c144 control scored 5970.02 and 5961.67; c192 failed the SLA gate (good request fraction 69.42%). c160 was the best configuration measured in this campaign.

Common baseline: **1,832 good output tok/s** on Firmus 2×8 H200 (official task `5. Common Baselines.md`, job 25013), with different serving settings.

## Selected runtime

- **Topology**: three Prefill workers and one Decode worker (P3/D1), four GPUs each. Public flags `--tp-size 4 --attn-dp-size 4 --ep-size 4`, that is attention TP1/DP4 and MoE EP4.
- **Kernels and memory**: FA3 attention, FP8 E4M3 KV cache, page size 64, the release's own MegaMoE with the DeepGEMM runner, memory fraction 0.90.
- **Routing**: both roles use `--load-balance-method total_requests --load-snapshot-publish-interval 1`.
- **Prefill**: chunked prefill 16384 globally (4096 per DP rank), `--prefill-max-requests` not set, KV pool 32768 tokens per rank, global request cap 256. No CUDA graph.
- **Decode**: KV pool 393216 tokens per rank, global request cap 128 (32 per rank), CUDA graph batch sizes 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28 and 32.
- **Speculative decoding**: EAGLE3 with steps 3, draft tokens 4, top-k 1. The target verifies every token.

[worker.py](worker.py) generates each worker’s launch arguments and environment from [config/firmus-basic.json](config/firmus-basic.json). That configuration is byte-identical to the one used by both measured runs; their worker records are in `results/<run>/launch/`. The packaged scripts were simplified afterwards. CPU fixtures compare worker launch arguments, environment and the retained serving and benchmark steps with those records and the original scripts. The final scripts have not been run on GPUs.

## Draft declaration

The sole draft is public [nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507), revision `ff0f0c0999f6c86b21dfd2f5803eebc4ac712224`: **572,538,880 parameters, BF16, 1,145,943,008 weight bytes** (CC-BY-4.0). Basic serves it in BF16, without load-time quantization. `run/prepare_eagle.py` changes only the architecture selector and explicit dtype declaration in its config; weights are unchanged, and the downloaded weight SHA-256 recorded in `source/draft-provenance.json` is `bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86`. No draft is trained or fine-tuned.

## Build

The image is the official `lmsysorg/sglang` v0.5.23 linux/amd64 image, pinned by digest `sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5` in [build/source/containers/basic.Dockerfile](build/source/containers/basic.Dockerfile). The recipe installs AIPerf 0.12.0 and `sglang-router` 0.3.2 in a separate `/opt/aiperf` environment, leaving the inference packages unchanged. It also adds a provenance manifest and an image self-check. Serving uses the gateway binary shipped in the official image.

Build on a root Linux host with Buildah (`chroot` isolation), Skopeo, Umoci, SquashFS tools, Git and jq. From the package root (the folder holding `SUBMISSION.md`):

```bash
sudo env BUILD_DIR=/var/tmp/qwen-basic-build CONTAINER_ARCHIVE_DIR="$PWD/basic/build/output" GITHUB_RUN_ID=100001 GITHUB_RUN_ATTEMPT=1 basic/build/setup.sh
```

`setup.sh` runs `build/source/scripts/build-basic-container.sh` with `build/source/` (byte-identical copies of the team repository's Basic build inputs) as its input directory.

`BUILD_CACHE_REPOSITORY` is optional (a writable OCI registry for Buildah layer caching). Outputs are `100001.sqsh` and `100001.container-sidecar.json`. The measured image is build **41046995676** from Qwen `aa48ceeca33afc27c2995713d55d4a4c65d02fc1`: **16417116160 bytes**, SHA-256 **d6ea5d95d6d76ab87ea55aeb70a052f04dbba8f9bbd515c3ece394690b2a1058**. `build/build.log` is that build's raw Buildah log and `build/provenance.json` its sidecar. Since that commit, `build-basic-container.sh` changed only to make the registry cache optional; the Dockerfile and image check are unchanged. Bit-for-bit reproducibility of a fresh archive has not been established.

A new build has a different artifact identity. Copy `config/firmus-basic.json` and set `container_build_run_id`, `container_source_commit`, `container_sha256` and `container_size_bytes` from the new sidecar; keep the published pins unchanged to reproduce the existing measurement.

## Prepare and run

`prepare.sh` copies the worker, job script and shared AIPerf/draft helpers into the run directory. On the Firmus login node, stage these inputs on storage readable by both allocated nodes:

- The image and its sidecar, side by side as `BUILD.sqsh` and `BUILD.container-sidecar.json`.
- `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, revision `e156cb4efae43fbee1a1ab073f946a1377e6b969`: all 24 shards and metadata.
- A tokenizer snapshot of `Qwen/Qwen3-235B-A22B-Instruct-2507`, revision `ac9c66cc9b46af7306746a9250f23d47083d689e`: `config.json`, `generation_config.json`, `merges.txt`, `tokenizer.json`, `tokenizer_config.json` and `vocab.json`.

Preparation downloads the pinned public draft and sets its SGLang architecture and dtype through `run/prepare_eagle.py`.

```bash
bash basic/prepare.sh /scratch/home/ACCOUNT/qwen-basic-run /shared/BUILD.sqsh /shared/Qwen3-235B-A22B-Instruct-2507-FP8 /shared/tokenizer/ac9c66cc9b46af7306746a9250f23d47083d689e basic/config/firmus-basic.json
cd /scratch/home/ACCOUNT/qwen-basic-run
sbatch --export=PATH,HOME,USER,LOGNAME scripts/submit.sbatch
```

`prepare.sh` creates the run directory, links the image, writes the effective configuration with the image's pins, stages the tokenizer cache and the draft, and copies the job scripts. The single Slurm script (partition `hpcai`, two nodes, eight GPUs per node, 17 minutes) starts the P3/D1 workers and the public gateway, waits for readiness, records every worker's server info (release, layout and KV pools), sends one connection sanity chat, and then runs the unchanged official AIPerf command (60/120/30 seconds, seed 2026, closed-loop c160). AIPerf runs from the isolated `/opt/aiperf` environment. The partition and resource lines are for Firmus.

## Outputs

Check the job's terminal state and exit code, then `results/score.json`: `eligible`, the good request fraction and request errors. **Good output tok/s = `goodput.avg × 200`** from `results/profile_export_aiperf.json`. `run-success` marks a job that ran to the end; the server info is recorded again after the benchmark. See [validation scope](../SUBMISSION.md#validation-scope) for the separate accuracy assessment. Preserve `source/`, `logs/` and `results/` of the run directory before deleting it.
