# Round 12: pair-only M192/N96 FA3 implementation

Base `7b1dd68`; branch `perf/pp4-r12-fa3m192-20261031`. Implementation and local CPU checks only. No workflow dispatch, SSH, GPU job or image build. The default config leaves `prefill.fa3_pair_gate=false`; only a later user serving commit enables it.

## Why this scope

The round-12 survey (`origin/perf/pp4-r12-survey-20261031`, sections 3.1 and 5) found the archived all-shape screen in run 41151283746. All twelve 3986+4027 cells saved 17.858–27.255 us, but a single-L4027 cell saved only 4.546 us against the unchanged 5 us margin. The aggregate retained `n192-direct-o-varlen-pipe`, and candidate endurance never ran. This change narrows runtime scope instead of relaxing that margin.

Native code and precision-reference validation are ported from komoriyuta's 821f409/fd8a66c lineage on the read-only `origin/perf/pp4-throughput-bundle-20261031`. Its unrelated graph/sparse changes, all-shape selector wiring and image are not ported. The newest QKV gate at b9a47b2/2ff978d supplies the timeout, per-GPU fragments, all-twelve aggregate and atomic env-publication pattern.

At L4001, the survey's logical K+V tile loads fall 1.07761→0.70189 GB/sequence and padded tensor work 275.87→269.53 GFLOP; KV visits increase 21924→28560. Useful work and unique input bytes remain unchanged; useful causal QK+PV at two L3994 is about 522.85 GFLOP. Actual HBM traffic is unknown, not zero. NVLink traffic is unchanged. The measured pair saving projects 0.429–0.654 ms per 24-layer stage, not per 48 calls. A conditional serving expectation is 0.4–0.6 ms, with zero gain on rejection. Use work-removal factor about 1, with no extra clock multiplier; the old screen's power/clock response cannot establish normal-serving gain. There is no new GPU result or official score.

## Files and build path

- `kernels/fa3-verify/prefill-candidates.json`: append native `m192n96-direct-o-varlen-pipe`, variant 8416, N96, numeric reference to the served row. Existing IDs/order remain unchanged.
- `kernels/fa3-verify/prefill-geometry.patch`: M192 geometry only behind variant bit 8192, varlen scheduler and direct-O support.
- `kernels/fa3-verify/build.py`: apply this patch after the existing chain; `generate_prefill_dispatch` emits the distinct translation unit, including unchanged `prefill_variant.cu`/`verify.h`, and appends it to the companion dispatch table/manifest.
- `patches/sglang/prefill-fa3-pair.patch`: forward host sequence lengths from the FA3 backend and select a native row per call. Cached lookup retains both control and candidate, avoiding JSON reads when pair/single calls alternate.
- `scripts/check-firmus-decode-kernels.py`: reuse existing FP32 sampled-row reference, thresholds, changed-input replay and paged-KV checks; add the ported served-row numeric reference and an opt-in FP32 Q/K-norm producer for this gate. Existing selectors keep their original producer default.
- `scripts/select-firmus-prefill-fa3-pair.py`: screen, serving-path endurance, CPU aggregate and atomic env update.
- `platforms/firmus/framework-comparison/sglang.sbatch`, `scripts/provision-firmus-framework.sh`, `scripts/prepare-firmus-pp4-settings.py`, config: default-off gate before model loading, staging/archive and baseline provenance. Other settled gates remain skipped.
- `qwen-experiment-metrics/tests/test_prefill_fa3_pair.py` plus six existing related test files: dispatch, aggregate/refusal, shell timeout continuation, native/build wiring and updated candidate/overlay inventories.

**Fresh image required:** this is native FA3 source, not a runtime-only flag. AWS input preparation copies the entire committed checkout (`aws-container.yml`), `build-sglang-container.sh` copies the complete `kernels/fa3-verify` directory, and Dockerfile copies it to `/opt/apac/fa3-verify` and runs `build-fa3-verify.sh`. The new patch and JSON are included by these directory copies; the build script explicitly applies the patch and emits/compiles the new native TU. The private `apac_fa3_private` namespace and separate TU preserve the stock FA3 extension. The build's manifest hashes native inputs and binary. Runtime overlay staging applies the config's ordered SGLang patches.

## Dispatch contract

The only new serving setting is `SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe+m192n96-pair`. On the existing eligible fixed-Prefill path, use M192/N96 iff there are exactly two varlen sequences and **each length is in inclusive 3900–4096**. Every other fixed geometry, missing lengths, single request or 3/4 requests retains the served M128/N192 pipe native row. Calls outside the pre-existing fixed path keep their original ordinary FA3 fallback. The decision uses `forward_batch.extend_seq_lens_cpu`, with no CUDA tensor read, device sync, timing, model-output or benchmark-state decision. Decode is unaffected.

`prepare-firmus-pp4-settings.py` continues to initialize the control env. Enabling the boolean does not select the candidate; only successful aggregation may publish the composite value. Its baseline settings record flags the optional selection accurately; the actual decision is in the gate JSON and final env.

## Gate and budget

Three four-GPU workers run concurrently on disjoint groups before model load. Each rank checks 3994+3994, 3986+4027, 3986+3986, 4027+4027, 4096+4096, 3900+4096 and 4096+3901. The latter two extend ragged coverage to the inclusive lower/upper bounds. These representative cells do not exhaust every integer pair in the dispatch interval.

For every cell, use the current FP32 Q/K norm/RoPE producer, existing prepared-control/code/scale and served-row checks, sampled FP32 causal reference with relative L2 ≤0.05 and maximum scaled error ≤0.25, and two changed-input replays. Time producer plus FA3 using 12 calls per graph sample, ten rounds each of interleaved control→candidate and candidate→control order. Require at least **5 us/call** saved in **each order and pooled medians**, on every shape and every GPU. Archive raw samples and observed order deltas. No new noise assumption or lowered margin.

A locally eligible GPU then runs the composite serving `prefill_fp8.attention` path at 3994+3994. Its initial output must equal the direct candidate and pass FP32 reference; after **1200 back-to-back eager calls**, recheck FP32 and exact bytes of both FP8 O and FP32 scales against its own initial output. Equality to the different control tiling is not required.

Timing retains both pre-capture closures and their output/metadata storage for the graph lifetimes. Scale-byte equality first materializes the column-major scale view contiguously; a real-Torch CPU regression checks this layout and the 1200-call count.

CPU aggregation revalidates all twelve distinct fresh H200 records, both timing orders, all shapes/numerics, 1200 calls, direct-dispatch equality and final bytes. Missing, stale, malformed, nonfinite, late, failed, under-margin or timed-out evidence leaves the existing env file byte-for-byte unchanged. Only success atomically replaces its single control assignment, retaining all other env lines. A timeout is a refusal, not a job failure; a shell fixture verifies continuation to control serving.

Estimated **60–85 s serial wall**, including container start, imports/JIT, numerics and endurance; unmeasured on H200. Each complete srun is bounded by **150 s + 2 s kill grace**, including launch, with a common **start+155 s** aggregate acceptance deadline and rank elapsed strictly below 150 s. The bound was first 88 s; it was widened before the first run because a timeout would spend the build and the run without an answer. Native FA3 is already AOT-compiled. No broad old sweep or second endurance phase is added. Round 10 used 9:25 of its 17-minute allocation; this estimate is not an allocation-duration guarantee.

## CPU evidence

`fa3-prefill-pair-cpu-validation-20261031.json` preserves exact compile commands, patch/input hashes, full ptxas logs, static opcode counts and SASS hashes. Fresh hash-verified upstream FA3/CUTLASS archives, exact production patch order, same CUDA 13.4 toolchain and sm_90a for all three builds; image toolchain remains CUDA 13.0.88.

| Native row | registers | spill stores / loads | stack | dynamic shared bytes | static SASS instructions |
|---|---:|---:|---:|---:|---:|
| Control before geometry patch | 168 | 84 / 116 B | 40 B | 166912 | 5504 |
| Control after geometry patch | 168 | 84 / 116 B | 40 B | 166912 | 5504 |
| M192/N96 | 128 | 40 / 28 B | 16 B | 124928 | 4336 |

The **entire control SASS dump including encoded control words is byte-identical**, SHA-256 `737b75e9c3f34e25d698374e4b908d392b0a1d07ea36d120e57d8539d5b45bc7`. No C75xx diagnostic. Resource-only host calls return both shared sizes within 227 KiB without launching a kernel. Static counts are whole instantiated kernels, not dynamic work or a speed prediction. Sum-chain scheduling is omitted: no separate dependency-depth/PV-lifetime proof was attempted in this bounded implementation. Candidate 2 was not attempted.

All 30 runtime SGLang overlays passed sequential check/apply to a task-local clone of the prepared image source. The pre-existing blank-at-EOF warning in the contiguous-KV overlay remains. `check-firmus-pp4-cpu.py` verified the archived round-10 manifest's 23 baked patch hashes and passed all source/real-CPU-Torch checks using Torch 2.11.0 and NumPy; no GPU/CUDA initialization. Full image SHA/extraction/staged-source comparison cannot run locally: the pinned SquashFS is absent and the new image has intentionally not been built. Mandatory full provisioning gate remains enabled.

Final validation: full suite via `uv run --project qwen-experiment-metrics --python 3.12 --with 'numpy>=1.26,<3' pytest -o addopts='' -q --durations=10 qwen-experiment-metrics/tests` passed **1210 tests, 81 skipped, 25 subtests passed**, 477.52 s. Final targeted gate tests, including the later buffer-lifetime and column-major regressions, passed **47/47** with `uv run --project qwen-experiment-metrics --python 3.12 --with torch==2.11.0 --with 'numpy>=1.26,<3' python -m pytest -o addopts='' -q qwen-experiment-metrics/tests/test_prefill_fa3_pair.py`. Ruff across metrics/scripts/containers, touched Python compilation, sbatch/provision shell syntax and diff checks passed. Existing fixture expectations were updated for the extra overlay/optional-gate boundary. Task-owned compiler/source clones and temporary diagnostics were removed; no remote resources or jobs were created.

## Later user serving commit: exact changes

1. Set **`prefill.fa3_pair_gate` from `false` to `true`** in `configs/firmus-prefill-pp4.json`. Keep all other performance settings and overlay/native inputs as committed. Do not manually change the initial FA3 env pin or set the raw native row name; the aggregate owns adoption.
2. Request a **fresh build** and ordinary two-node Firmus PP4 serving in the user-created commit using the repository's build/cluster/PP4 tags. Do not reuse the old image for this native addition.
3. **Keep the checked-in round-10 image pin and all matching `container_*` fields together for that build request.** They currently agree with `check-firmus-pp4-cpu.py`'s accepted `e7b09add…3f62` pin. With fresh-build mode, `provision-firmus-framework.sh` rewrites `images.sglang.{path,sha256,size_bytes,provenance}` and `container_{build_run_id,sha256,size_bytes,source_commit}` in the runtime config from the new build outputs. The CPU gate verifies the new sidecar SHA and source commit against that serving commit. Do not fabricate a future SHA/path/run ID or change the gate's accepted historical pin. Round 10 run 41150387984 used this same rewrite path; ced41e5 pinned its successful image afterwards. The prior revert 26353da explains why an arbitrary other image pin is rejected.
4. After terminal serving/result checks, record the new image's actual run ID, SHA, size, source commit and provenance together if it is to become a reusable pin. Inspect `results/prefill-fa3-pair/selection.json`, per-GPU fragments, final Prefill env/worker logs, official parser and archived outputs before claiming adoption or improvement.

Further unimplemented levers, ranked by the survey: (1) sum-only scheduling with actual SASS dependency/PV-lifetime proof; (2) separately validated V-layout producer/consumer work (outside this task); (3) new tile/communication designs only after proving a new work/byte contract. Their speculative ceilings are not additive to this row's measured saving.
