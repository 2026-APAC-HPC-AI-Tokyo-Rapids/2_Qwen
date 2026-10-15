# Private SM90 provider: first executable image gate

Current status (2026-10-15): native CPU image38693955483 and finite four-GPU gate38699779914 have passed independent verification. See README.md for exact evidence, timing limitations and the decision to audit the DP4 architecture before a full serving port. The original preparation notes below are historical; they are not current gate status.

## Concrete local evidence

- `audit_namespace.py` verified the exact public x86_64 sgl-deep-gemm0.1.5 wheel (4,814,485 bytes, SHA256 e5e9f92cdf0515fe878bd81131bc25d2984e00c005824d2932af01a5e0d75261), all11 relevant Python files byte-identical to the existing SGLang image, original SM90 header, and exact vLLM resolver source. Actual complete package Python import under apac_sm90.deep_gemm passed with explicit Torch/TVM stubs; native wrapper arguments matched the existing API. This is Python wiring evidence, not native ABI evidence.
- `install.py` actually unpacked that wheel into a disposable private destination, applied Komori's exact existing4517-byte late-topk/FP32 patch (SHA256462d68230da3ab9a1e69ca85d9fa804b8edaf6a77bf0975726611a08fda6622b), and verified patched header5479807104bf1aeed4cc3ecc8ce57d2a835d68457f9cf4962ef3f15cc176e8d2 plus the untouched native library hash. No pip install or global dependency change.
- Negative controls reject corrupt wheel before destination creation, unknown package VERSION, changed native-library bytes, and changed precision header before loading the provider. Actual private loader imported the staged Python files with explicit stubs, did not publish top-level deep_gemm, and repeated loads returned the same private module. Results: cpu-staging-smoke.json.
- Pinned Ruff0.16.2 format/lint, Python AST parsing and Bash syntax passed. A first Windows smoke attempt correctly rejected a CRLF-transformed precision header; per-command `git -c core.autocrlf=false -c core.eol=lf apply` fixed the temporary installer behavior. No global Git setting changed.
- Live existing vLLM image metadata shows apache-tvm-ffi0.1.11, exactly the new private wheel's required version. Torch2.13.0+cu130/vLLM0.30.0 also match the native checker pins. No dependency upgrade is needed.

## Minimal image integration to review

Keep the current vLLM base OCI pin and dense-draft repair. The following is the complete proposed Dockerfile addition, after those existing steps. Every RUN command occupies one physical line. The wheel is unpacked privately; its parent directory is NEVER added to PYTHONPATH/sys.path and no top-level deep_gemm is installed.

```dockerfile
COPY containers/firmus-sm90-megamoe /opt/apac/firmus-sm90-megamoe
RUN curl -fsSL https://files.pythonhosted.org/packages/bb/36/93c38fd790f7c7ea8b19f0c4980603c64ba65cb4d712f638d2d225ceeef4/sgl_deep_gemm-0.1.5-py3-none-manylinux2014_x86_64.whl -o /tmp/sgl_deep_gemm-0.1.5.whl && python3 /opt/apac/firmus-sm90-megamoe/install.py /tmp/sgl_deep_gemm-0.1.5.whl /opt/apac/sm90-provider && rm /tmp/sgl_deep_gemm-0.1.5.whl
RUN bash /opt/apac/firmus-sm90-megamoe/check_cpu_image.sh
```

The existing build-context copier must copy that one directory. Record its source files and exact reused precision patch in the existing embedded build provenance. The existing actual-image checker should require the resulting /opt/apac/sm90-megamoe-cpu-gate.json, exact provider hashes/version and unchanged ordinary vLLM DeepGEMM resolution; do not call an earlier stub smoke result a successful native image check. Root should run one require-build-only commit without any cluster tag to establish this gate, leaving the current diagnostic job untouched.

`check_cpu_image.sh` creates only an owned temporary libcuda.so.1 symlink to CUDA's stub library for the CPU builder process and removes it on exit. It does not alter image-wide LD_LIBRARY_PATH or leave a driver stub installed on the serving path. `check_cpu_image.py` imports actual Torch, actual vLLM-vendored DeepGEMM2.8.0, and the actual private TVM-FFI native library. It checks exposed native SM90 functions, the real default block128 recipe, exact source/dependency hashes, no top-level namespace collision, and re-resolves ordinary dense DeepGEMM after private import. It then compiles the same three SM90 template geometries already used by Komori, using nvcc only. It does not allocate CUDA tensors, initialize a distributed group or execute a GPU kernel.

The literal recipe default was verified in the existing source and actual Python stub import: `(128,128,128)`. Whether both native libraries can coexist/import in this exact image with the real CPU stub remains the next honest build pass/fail. Missing/changed versions, API/header/source, import failure, changed dense provider or compile failure must fail the build; no fallback or unpinned dependency replacement is added.

### Files and scope

- contract.py: wheel/header/native/source hashes and guards.
- install.py: private extraction plus exact reused precision patch; rejects an existing destination.
- provider.py: explicit private package load without global deep_gemm alias or sys.path insertion.
- check_cpu_image.sh / check_cpu_image.py: real native CPU import and finite SM90 CUDA compilation.
- sm90-late-topk-fp32.patch: byte-for-byte existing Komori patch, no new math.

The loader is an integration aid, not an active vLLM backend. No serving adapter or MoE backend selection is changed by these files. No MegaMoE GPU/benchmark is authorized by merely adding them to an image.

## Finite four-GPU gate interface after native image PASS

Implement a separate diagnostic script only after the CPU image gate passes. Run via traceable Actions/Slurm, four local H200 ranks, one finite execution; no scheduler call inside the harness. Input contract is the private provider root plus existing exact native fused-MoE reference. Fixed test set:

- Global TP4 token rows M={1,4,7,32,97,128,288,324,384,768}; padded contiguous row shards of ceil(M/4), preserving original row IDs and trimming only after gather.
- Exact local weights FP8 E4M3 w13[32,3072,4096], w2[32,4096,1536], original FP32 block128 scales, 128global experts/top8, group128 first-activation quantization, group64 FC2 activation quantization, BF16 terms and late FP32 weighted combine. Reuse Komori's independent reference and tolerance rather than inventing relaxed accuracy gates.
- Fixed route patterns: random across ranks, all-hot experts on one owner, empty/tail ranks, masked padding. All ranks must enter every collective; record local live counts and global row ownership.
- Warm/JIT and allocate symmetric buffers before capture. Capture at fixed addresses, then four changed-input replays with alternating idle/live ranks and masked IDs. Assert no stale/nonfinite output and compare gathered output to the independent reference. Do not treat a successful first replay as sufficient.
- Timed full path includes split/copy, vLLM-equivalent gate/topk routing, predispatch, fused kernel and output all-gather; compare the full native EP4 route/reduction path for identical input/weights. Report GEMM alone only as a component. Fixed20warmups/100timediterations per representative non-reference timing shape, CUDA events with synchronization outside timed sections, no growing knob sweep.
- JSON output includes source/image hashes, GPU/rank topology, exact shapes and aligned capacity, symmetric workspace bytes, correctness errors, replay checks, full-path median/p95 and per-component times. There is no official throughput score from this diagnostic.

Capacity192 is only ceil(768/4) before the native token-alignment requirement; query the native alignment and account for actual warmup/dummy shapes. The workspace must fit alongside the existing decoder KV allocation without silently reducing it. Stop on math/replay/ABI/memory failure or no useful full-path advantage. Only then consider a decoder-only serving adapter and one clean official benchmark; retain attentionTP4/DP1, dense draftTP4/EPfalse, all six prefills and original NIXL topology.
