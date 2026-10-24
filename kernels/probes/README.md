# Genkai one-GPU probes

Run **Genkai 1 GPU kernel probe** from GitHub Actions, selecting the commit/branch to test. The `benchmark` input selects a preparation script under `kernels/probes/`; the common runner allocates one GPU and executes the resulting `source/probe`. The script may compile native CUDA or prepare a container command. FC1 is the default. Leave `candidate_patch` empty for a control/control check, or supply one repository-relative patch path such as `patches/deep-gemm-overlay/example.patch`. The workflow does not trigger the serving workflows or an image build.

The default `prepare-fc1.sh` measures Prefill FC1 only. It extracts the actual math loop from the pinned serving header plus its configured overlays. Candidate patches must change only that FC1 body; edits to FC2, loaders, resource declarations, or other files are rejected before submission. Routing, NVLink, SwiGLU, combine and serving are outside this probe. Its time is not a prediction of serving throughput.

The login node extracts headers from the already deployed Genkai copy of `configs/firmus-framework.json`'s SGLang image and compiles with CUDA 12.9.1 / GCC 13. It checks the sidecar identity, image size and extracted base-header hash, rather than rescanning the entire image. Results appear in the Actions summary; no artifact archive is uploaded.

PJM receives `b-batch`, `gpu=1`, and a three-minute limit. There is no node or MPI allocation request. Three processes each check occupancy, changed-input graph replays, full bit equality, and a sampled independent arithmetic reference, then take 30 interleaved timings per variant. A failed process stops the job. After terminal-state confirmation, the workflow removes its temporary run directory. No Firmus or Aspire job is submitted.

## Decode calibration against Firmus

Select either benchmark below, with **`candidate_patch` empty**, on the commit
containing these files. Run them separately through the existing workflow;
neither preparation script builds an image or submits a job itself.

| benchmark | Scope |
| --- | --- |
| `kernels/probes/prepare-calib-dense.sh` | M128, QKV N9216/K4096 and O N4096/K8192 |
| `kernels/probes/prepare-calib-fa3.sh` | B32 TARGET_VERIFY, Q4, 64 Q heads / 4 KV heads / D128 |

The current image `39932105942` has the required DeepGEMM host override and
FA3 companion. Its host override requires M≤128 and 132 SMs; the companion
requires B32–36. These calibration probes use **M128 and B32 only**. M160,
B40 and p128split2/3 are excluded. The container has CUDA13.0 / Torch cu130;
loading the host's CUDA12.9.1 module does not recompile the container's AOT
FA3 binary. Interpret this as an H100/container calibration, and keep the
H200 power/clock and four-GPU serving differences in view.

Dense reuses `select-firmus-decode-dense.py`'s preparation, CUDA graph capture,
timing and alternating order: four prequantized weight/input sets, 20 calls
per graph, 25 samples per turn, two blocks with reversed candidate order.
Each candidate turn is paired with stock. QKV candidates are stock,
BM128/BN80/stages7, BM64/BN144/stages5, BM128/BN144/stages5,
BM128/BN80/stages7/multicast2-on-A, and Triton64/128/K128/stages3/warps4.
O uses its Firmus selector geometries: stock, BM128/BN32/stages10,
BM64/BN64/stages9, BM128/BN64/stages8,
BM64/BN64/stages13/multicast2-on-A, and the same Triton candidate.
All non-multicast configurations use multicast1/on-A0.
The BM128/BN144/stages5 QKV row is diagnostic: the supplied Firmus result
for that configuration was M160, so it has no supplied M128 reference.

FA3 reuses `select-firmus-decode-fa3.py`'s 94-layer graph and paired CUDA-event
timing: four KV sets, shared live sequence metadata, one scheduler preparation
per 94 layers, 20 alternating stock/candidate pairs per block, two blocks.
Candidates are stock, control96, p128s2, t64s2 and t64s3. Stock samples are
also retained separately for each candidate; its displayed aggregate pools
those paired samples. Times are **per layer**, including amortized scheduler
preparation. This follows B4 metadata sharing rather than 94 scheduler calls.

Both probes reuse the existing FP32/stock error tolerances and changed-input
graph checks. A candidate that fails its numeric check makes the calibration
fail; no failed candidate is silently omitted. Results include median, p10,
p90, candidate/stock ratio (below 1 is faster), and both block ratios. Raw
sample and numeric-check JSON goes to the PJM/Actions log. The summary prints
the candidate table; no new artifact archive is uploaded.

Calibration uses one process with a **165-second timeout**, retaining the
three-minute PJM limit and reserving 15 seconds for job setup/exit. FC1 keeps
its original three 30-second trials. The two measurement blocks provide the
within-process repetition; repeat the workflow manually to check run-to-run
stability before drawing a calibration conclusion.

Timing budget (estimate, **not GPU-validated**): dense's timed graph work is
about 0.3–0.7 seconds using the supplied Firmus latencies; FA3's timed work is
about 1.5 seconds at 50 µs/layer. Imports, Enroot setup, cold JIT, numerical
references and graph capture dominate. Budget roughly 30–120 seconds for
dense and 20–90 seconds for FA3, within the 165-second process limit; cold
JIT completion on Genkai remains to be checked by the parent workflow.

For comparison, Firmus run 39901974117 at M128 had QKV stock 17.0 µs,
BM128/BN80/stages7 15.2 µs (0.894×), BM64/BN144/stages5 16.5 µs (0.971×),
multicast candidates 15.5–20.2 µs and Triton 30.8–31.6 µs. O stock was
15.2 µs, other DeepGEMM candidates 15.1–19.6 µs and Triton 50.3 µs.
FA3 B32 stock was 49.5 µs, control96 50.6 (1.022×), p128s2 46.1 (0.931×),
t64s2 54.4 (1.099×) and t64s3 53.2 (1.075×). Compare rankings and ratios
with these same-shape references; do not treat H100 absolute times as
Firmus serving throughput predictions or infer a serving win from one run.

CPU protocol/aggregation fixtures: `python3 kernels/probes/test-calibration.py`.

## Calibration against Firmus, 2026-10-24

Runs [40043191081](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40043191081) (dense, M128) and [40043193928](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40043193928) (FA3 TARGET_VERIFY, 32 requests, 94-layer graph) measured cases whose Firmus H200 results were already known from the in-job selections of runs 39901974117 / 39970713547. Ratios are candidate / stock (median).

| case | candidate | Firmus H200 | Genkai H100 |
| --- | --- | ---: | ---: |
| QKV M128 | BM128/BN80/7 (Firmus winner) | 0.894 (15.2 / 17.0 µs) | 1.003 (23.7 / 23.6 µs) |
| QKV M128 | BM64/BN144/5 | 0.971 | 0.985 |
| QKV M128 | BM128/BN80/7, multicast A | 0.912 | 0.993 |
| QKV M128 | Triton 64x128 | 1.81 | 1.42 |
| O M128 | BM128/BN32/10 | 1.02 | 1.03 |
| O M128 | BM128/BN64/8 | 1.22 | 1.08 |
| O M128 | multicast A | 1.27 | 1.04 |
| O M128 | Triton 64x128 | 3.31 | 2.41 |
| FA3 B32 | p128s2 (Firmus winner) | 0.931 (46.1 / 49.5 µs) | 1.010 (71.1 / 70.0 µs) |
| FA3 B32 | control96 | 1.022 | 1.019 |
| FA3 B32 | t64s3 | 1.075 | 1.052 |
| FA3 B32 | t64s2 | 1.099 | 1.056 |

What the comparison shows:
- Genkai ranks the clear losers correctly (Triton, the N64 TMA FA3 tiles, BM128/BN64 for O): every candidate that is 5% or more slower on Firmus is also slower on Genkai.
- It misses both Firmus winners. They are memory-path improvements for Decode-sized (memory-bound) shapes. On H100, HBM is 3.35 TB/s against 4.8 TB/s on H200, and the stock kernels are already about 1.4x slower in absolute terms, so the same change does not help.

Use the one-GPU probe as a correctness gate (bit equality, changed-input graph replay, boundary shapes) and to drop candidates that are clearly slower. Do not use it to reject a candidate that is neutral or slightly slower on Genkai. Decide performance on Firmus: the in-job selection in the serving job, or the one-node kernel bench. Compute-bound Prefill kernels have not been calibrated here.
