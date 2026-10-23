# Genkai one-GPU probes

Run **Genkai 1 GPU kernel probe** from GitHub Actions, selecting the commit/branch to test. The `benchmark` input selects a preparation script under `kernels/probes/`; the common runner allocates one GPU and executes the resulting `source/probe`. The script may compile native CUDA or prepare a container command. MegaMoE math is the default. Leave `candidate_patch` empty for a control/control check, or supply one repository-relative patch path such as `patches/deep-gemm-overlay/example.patch`. The workflow does not trigger the serving workflows or an image build.

The default `prepare-megamoe-math.sh` extracts the Prefill FC1 and FC2 math loops from the pinned serving header plus its configured overlays. Each phase continuously processes 32 experts with 1024 received tokens each (3072 FC1 tiles, 4096 FC2 tiles), using the serving TMA stage layout and register budgets. Candidate patches must change only those math bodies; changes to loaders, resource declarations or other paths are rejected before submission. The probe writes FP32 outputs for validation. Routing, NVLink, SwiGLU, combine and serving are outside this probe, so its time is not a prediction of full MegaMoE or serving throughput.

The login node extracts headers from the already deployed Genkai copy of `configs/firmus-framework.json`'s SGLang image and compiles with CUDA 12.9.1 / GCC 13. It checks the sidecar identity, image size and extracted base-header hash, rather than rescanning the entire image. Results appear in the Actions summary; no artifact archive is uploaded.

PJM receives `b-batch`, `gpu=1`, and a three-minute limit. There is no node or MPI allocation request. Three processes each check occupancy, changed-input graph replays, full bit equality, and a sampled independent arithmetic reference, then take 30 interleaved timings per variant. A failed process stops the job. After terminal-state confirmation, the workflow removes its temporary run directory. No Firmus or Aspire job is submitted.

MegaMoE findings on 2026-10-24 (H100, math probe only):

| Candidate | Paired result | Decision |
| --- | --- | --- |
| Three-stage loop unroll | FC1 −1.65% to +0.36%; FC2 +2.25% to +5.11% latency | Reject; [run 40003839043](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40003839043) |
| Earlier consumed-stage release, both phases | FC1 −2.08% to −4.64%; FC2 −0.45% to +0.50% | Keep only FC1; [run 40004596361](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40004596361) |
| Earlier release, two-warpgroup FC1 only | 950.624 → 912.992, 927.744 → 904.352, 926.112 → 900.480 µs (−2.52% to −3.96%) | Candidate for full validation; [run 40005113228](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40005113228) |

The retained `patches/deep-gemm-bench/prefill-fc1-early-release.patch` waits for the current WGMMA and releases its TMA stage before waiting for the next stage's scales. It changes no arithmetic or resource allocation: all output bits match, sampled reference error is below 0.002, registers remain 168, local memory is zero, and dynamic shared memory remains 225472 bytes. The unchanged FC2 control in the final run varied by −0.02% to +2.74%, so treat the speedup magnitude as provisional. Six FC1 process trials across the two release runs improved, but routing, handoff, remote scatter and whole-serving effects still require four-rank validation. This patch is not enabled in the serving configuration.
