# Genkai one-GPU probes

Run **Genkai 1 GPU kernel probe** from GitHub Actions, selecting the commit/branch to test. The `benchmark` input selects a preparation script under `kernels/probes/`; the common runner allocates one GPU and executes the resulting `source/probe`. The script may compile native CUDA or prepare a container command. FC1 is the default. Leave `candidate_patch` empty for a control/control check, or supply one repository-relative patch path such as `patches/deep-gemm-overlay/example.patch`. The workflow does not trigger the serving workflows or an image build.

The default `prepare-fc1.sh` measures Prefill FC1 only. It extracts the actual math loop from the pinned serving header plus its configured overlays. Candidate patches must change only that FC1 body; edits to FC2, loaders, resource declarations, or other files are rejected before submission. Routing, NVLink, SwiGLU, combine and serving are outside this probe. Its time is not a prediction of serving throughput.

The login node extracts headers from the already deployed Genkai copy of `configs/firmus-framework.json`'s SGLang image and compiles with CUDA 12.9.1 / GCC 13. It checks the sidecar identity, image size and extracted base-header hash, rather than rescanning the entire image. Results appear in the Actions summary; no artifact archive is uploaded.

PJM receives `b-batch`, `gpu=1`, and a three-minute limit. There is no node or MPI allocation request. Three processes each check occupancy, changed-input graph replays, full bit equality, and a sampled independent arithmetic reference, then take 30 interleaved timings per variant. A failed process stops the job. After terminal-state confirmation, the workflow removes its temporary run directory. No Firmus or Aspire job is submitted.
