# Tokyo Rapids with RIKEN: Qwen3 submission

**Submitted experimental result: 10054 good output tok/s (measured: 10054.49)** on Firmus, two nodes / 16 H200 GPUs. The submitted run is **Actions 41767886175 / Slurm 28660**. This folder contains the required Basic submission, the experimental configuration and its execution evidence.

| Entry | Configuration | Good output tok/s | Role |
| --- | --- | ---: | --- |
| **Basic** | Public SGLang 0.5.23, P3/D1, c160 | **6339.31**, run 41093781934 / Slurm 27709 | Selected run for the organisers to re-run |
| Basic repeat | Identical configuration, run 41078642203 / Slurm 27682 | 6321.12 | Two-run mean: **6330.21** |
| **Submitted experiment (Bonus)** | Patched SGLang, PP4 Prefill, c232, round 34 | **10054.49**, run 41767886175 / Slurm 28660 | Submitted result and source/kernel work |
| Package verification | Fixed round 34 choices plus logprob relay | 10038.11, run 41781374312 / Slurm 28673 | Supporting check; does not replace the submitted result |
| Organisers' reference | Common baseline, job 25013 | 1832 | Different serving settings |

The Basic and experimental configurations serve the published `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` checkpoint with the official AIPerf workload. The score is `goodput.avg × 200`, and at least 90% of requests must meet both latency limits. Model accuracy is assessed separately from this throughput score.

The experimental configuration achieved **1.59× the Basic mean** on the same 16 H200 GPUs. This comparison covers the complete configurations: their code, Prefill parallelism, concurrency and draft precision differ.

The organisers' 3 November clarification defines Basic as the official software release and places code changes in the experimental Bonus path.

## Reproduction entrypoints

Paths below are relative to this `qwen/` folder. Follow each README for the exact model, tokenizer, image and draft pins and the build-host and login-node prerequisites.

| Step or evidence | Basic | Submitted experiment |
| --- | --- | --- |
| Read first | [basic/README.md](basic/README.md) | [README.md](README.md) |
| Image build | `basic/build/setup.sh` and `basic/build/source/` | `build/sglang-mirror.sh`, `build/setup.sh` and `build/source/` |
| Prepare on login node | `basic/prepare.sh`, `basic/config/firmus-basic.json` | `run/prepare.sh`, `config/serving.json`, pinned role env files |
| Single allocation: serving then official benchmark | `basic/submit.sbatch` | `run/submit.sbatch` |
| Build log and image provenance | `basic/build/build.log`, `basic/build/provenance.json` | `build/build.log`, `build/provenance.json` |
| Submitted metrics and logs | [basic/results/41093781934/RESULT.md](basic/results/41093781934/RESULT.md), plus the repeat | [results/41767886175/RESULT.md](results/41767886175/RESULT.md) |
| Package verification | CPU orchestration and recorded launch checks | [results/41781374312/RESULT.md](results/41781374312/RESULT.md), plus CPU orchestration checks after the readability rewrite |
| Exact code changes and numerical arguments | No source patch | [patches/README.md](patches/README.md), with text patches and kernel sources |
| Presented experiments and historical evidence | [EXPERIMENTS.md](EXPERIMENTS.md) | Same index, with scripts, configuration and logs per run |
| Optional profile | — | [analysis/README.md](analysis/README.md), `analysis/profiling_data/` |
| Draft declaration | Basic README | Experimental README |

Both build methods use the inputs included in this folder. Recorded hashes identify the measured images. A fresh build may produce a different archive hash; copy its integrity fields from the new sidecar into the deployment configuration.

## Validation scope

[patches/README.md](patches/README.md) lists 143 build and runtime inputs, with application points, hashes and numerical arguments. The submitted run used 33 SGLang and 14 DeepGEMM runtime overlays. The delivered package adds SGLang overlay 34 to relay logprobs.

The submitted run passed all four kernel-selection checks on their tested shapes and devices. The package verification run returned eight 64-token responses with token logprobs. These checks cover local numerical behavior and serving integration. They do not establish that the full stack is bit-identical to stock SGLang. Changes to floating-point order and rounding can affect generated tokens; the team has not run the organisers’ full-set accuracy check.

The delivered Basic and Bonus scripts were simplified after the measured runs. The current wrappers isolate build storage, and the jobs distinguish benchmark completion from optional diagnostics. CPU fixtures compare worker launch arguments, environment and the retained serving and benchmark steps with the recorded versions; the final scripts have not been run on GPUs.

## Reading the presentation

The shared deck is `../presentation/Tokyo Rapids with RIKEN Presentation.pptx`. Qwen main slides are **38–67**; supporting slides **69–82** contain additional evidence. Slides **96 onward** are explicitly archived drafts. [EXPERIMENTS.md](EXPERIMENTS.md) explains the historical package references in the supporting slides and identifies the current configuration.
