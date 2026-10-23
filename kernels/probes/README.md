# Genkai one-GPU probes

Run **Genkai 1 GPU kernel probe** from GitHub Actions, selecting the commit/branch to test. Leave `candidate_patch` empty for a control/control check, or supply one repository-relative patch path such as `patches/deep-gemm-overlay/example.patch`. The workflow does not trigger the serving workflows or an image build.

The current probe measures Prefill FC1 only. It extracts the actual math loop from the pinned serving header plus its configured overlays. Candidate patches must change only that FC1 body; edits to FC2, loaders, resource declarations, or other files are rejected before submission. Routing, NVLink, SwiGLU, combine and serving are outside this probe. Its time is not a prediction of serving throughput.

The login node extracts headers from the already deployed Genkai copy of `configs/firmus-framework.json`'s SGLang image and compiles with CUDA 12.9.1 / GCC 13. It checks the sidecar identity, image size and extracted base-header hash, rather than rescanning the entire image. All extracted headers, applied patches, generated CUDA, build log, toolchain versions, executable and hashes are saved in the Actions artifact.

PJM receives `b-batch`, `gpu=1`, and a three-minute limit. There is no node or MPI allocation request. Three processes each check occupancy, changed-input graph replays, full bit equality, and a sampled independent arithmetic reference, then take 30 interleaved timings per variant. A failed process stops the job. After terminal-state confirmation and artifact upload, the workflow removes its temporary run directory. No Firmus or Aspire job is submitted.
