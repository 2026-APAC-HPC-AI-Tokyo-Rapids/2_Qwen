# Tokyo Rapids — Qwen3 PD submission (Firmus)

All results are on Firmus, 2 nodes / 16 H200, with the official `Qwen3-235B-A22B-Instruct-2507-FP8` checkpoint and the unchanged official AIPerf command.

| Tier | Configuration | Good output tok/s | Role |
| --- | --- | ---: | --- |
| **Basic** | Public SGLang 0.5.23, options only, P3/D1, c160 | **6330.21** (mean of 6321.12 and 6339.31) | The submission to re-run and rank |
| **Bonus** | Team source and kernel changes, PP4 Prefill, c232 (round 34) | **10054.49** | Experimental work, judged on proof |
| Common baseline | Organisers' reference settings | 1,832 | Job 25013 |

The Bonus configuration is **1.59×** the team's own Basic mean on the same machine (10054.49 / 6330.21).

## Where each deliverable is

Paths are relative to this folder, which is self-contained: both images are rebuilt from the build inputs copied into it (`build/source/` for Bonus, `basic/build/source/` for Basic).

| Deliverable | Basic | Bonus |
| --- | --- | --- |
| Overview and reproduction | [basic/README.md](basic/README.md) | [README.md](README.md) |
| Software stack build method | `basic/build/setup.sh` with `basic/build/source/` | `build/sglang-mirror.sh`, `build/setup.sh` with `build/source/` |
| Build log and image record | `basic/build/` | `build/build.log`, `build/provenance.json` |
| Single Slurm job script | `basic/submit.sbatch` (prepared by `basic/prepare.sh`) | `run/submit.sbatch` (prepared by `run/prepare.sh`) |
| AIPerf logs and `profile_export_aiperf.json` | `basic/results/41078642203/`, `41093781934/` | `results/41767886175/` (measured run), `41781374312/` (standalone run of the package) |
| Patches as text | None (no source patch) | `patches/` with [the inventory](patches/README.md) |
| Draft declaration | Basic README, "Draft declaration" | Bonus README, "Draft declaration" |
| Profiling data (optional) | — | `analysis/` and `analysis/profiling_data/` |
| Presentation slides | Team deck, submitted alongside this folder | (same deck) |

## Bonus proof (README Addendum R3)

1. **What changed**: every image patch, kernel source and runtime overlay is listed with its SHA-256 and application point in `patches/README.md`; the runtime overlays are copied verbatim under `patches/overlay/`.
2. **Why it is the same math**: the same inventory gives each change's numerical class (bit-identical, FP-order, rounding-placement, no numeric effect) with its basis, read from the patch text and the recorded evidence. No change approximates a function, lowers a number format below the checkpoint's FP8 blocks, or skips a computation whose result is used.
3. **Evidence**: the organisers run the full-set accuracy check. The Bonus job also runs a fixed greedy probe after the official benchmark, written to `results/greedy-parity.json` of its run directory (`results/41781374312/greedy-parity.json` for the standalone run), so outputs can be compared before and after a change.
4. **Speedup against our Basic**: 10054.49 against 6330.21 good output tok/s, both on Firmus 2 × 8 H200.
