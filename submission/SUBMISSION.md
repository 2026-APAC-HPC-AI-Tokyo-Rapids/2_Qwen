# Tokyo Rapids — Qwen3 PD submission (Firmus)

All results are on Firmus, 2 nodes / 16 H200, with the official `Qwen3-235B-A22B-Instruct-2507-FP8` checkpoint and the unchanged official AIPerf command.

| Tier | Configuration | Good output tok/s | Role |
| --- | --- | ---: | --- |
| **Basic** | Public SGLang 0.5.23, options only, P3/D1, c160 | **6330.21** (mean of 6321.12 and 6339.31) | The submission to re-run and rank |
| **Bonus** | Team source and kernel changes, PP4 Prefill, c232 (round 34) | **10054.49** | Experimental work, judged on proof |
| Common baseline | Organisers' reference settings | 1,832 | Job 25013 |

The Bonus configuration is **1.59×** the team's own Basic mean on the same machine (10054.49 / 6330.21).

## Where each deliverable is

Paths are relative to `repository/`, a snapshot of the team repository. The whole repository is included because both stacks are rebuilt from files in it (`containers/`, `scripts/`, `patches/`, `kernels/`, `configs/`).

| Deliverable | Basic | Bonus |
| --- | --- | --- |
| Overview and reproduction | [submission/basic/README.md](repository/submission/basic/README.md) | [submission/README.md](repository/submission/README.md) |
| Software stack build method | `containers/basic.Dockerfile`, `scripts/build-basic-container.sh` | `submission/build/setup.sh`, `submission/build/sglang-mirror.sh` |
| Build log and image record | `submission/basic/build/` | `submission/build/build.log`, `submission/build/provenance.json` |
| Single Slurm job script | `submission/basic/submit.sbatch` (prepared by `submission/basic/prepare.sh`) | `submission/run/submit.sbatch` (prepared by `submission/run/prepare.sh`) |
| AIPerf logs and `profile_export_aiperf.json` | `submission/basic/results/41078642203/`, `41093781934/` | `submission/results/41767886175/` (measured run), `41781374312/` (standalone run of the package) |
| Patches as text | None (no source patch) | `submission/patches/` with [the inventory](repository/submission/patches/README.md) |
| Draft declaration | Basic README, "Draft declaration" | Bonus README, "Draft declaration" |
| Profiling data (optional) | — | `submission/analysis/` and `submission/analysis/profiling_data/` |
| Presentation slides | `presentation/team_presentation.pptx` | (same deck) |

## Bonus proof (README Addendum R3)

1. **What changed**: every image patch, kernel source and runtime overlay is listed with its SHA-256 and application point in `submission/patches/README.md`; the runtime overlays are copied verbatim under `submission/patches/overlay/`.
2. **Why it is the same math**: the same inventory gives each change's numerical class (bit-identical, FP-order, no numeric effect) with its basis. Rows whose class the records do not establish are marked `to confirm` rather than guessed.
3. **Evidence**: the organisers run the full-set accuracy check. The Bonus job also runs a fixed greedy probe after the official benchmark, written to `results/greedy-parity.json` of its run directory (`submission/results/41781374312/greedy-parity.json` for the standalone run), so outputs can be compared before and after a change.
4. **Speedup against our Basic**: 10054.49 against 6330.21 good output tok/s, both on Firmus 2 × 8 H200.
