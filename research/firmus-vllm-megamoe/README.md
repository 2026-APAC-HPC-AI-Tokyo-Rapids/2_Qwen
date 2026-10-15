# vLLM SM90 MegaMoE feasibility audit

Owner: aizakovzky vLLM adapter track under issue 21; Komori retains SGLang MegaMoE and SLA repair. This audit did not submit a build or GPU job, modify serving/runtime files, or change a teammate branch. Inspected on 2026-10-15 JST. Source evidence is from cached compute bb77e51f79789c4861a4f0262d6a777443f98821 and selected public package files in the exact existing images.

## Decision

A bounded decoder-only port is plausible while retaining attention TP4, DP1, native dense draft TP4, and the existing NIXL KV topology. DP4 attention is required by Komori's SGLang adapter guard, not an intrinsic requirement established by the native MegaMoE API. This is an implementation proposal, not GPU compatibility or performance evidence.

The smallest useful first step is dependency isolation plus a four-GPU row-sharded numerical/timing harness. Do not first convert the vLLM server to four independent attention-DP engines. That would add scheduler, routing, KV-transfer, draft and memory confounds before learning whether the kernel helps.

## Evidence and concrete integration boundaries

1. The patched SGLang Qwen guard requires SM90, H4096, I1536, 128 experts, top8, MoE TP1/EP4, attention DP4, FP8 block128, and no TBO/EPLB/redundant experts. Decode also requires DP-local dense TP1, DP LM head, EAGLE3/topk1/four draft-token rows. Those are its adapter's tested conditions; they do not prove vLLM TP4 or spec2 support. Source: patches/sglang/qwen3-sm90-megamoe.patch at bb77e51.

2. Native entry points accept per-rank hidden rows plus GLOBAL expert IDs, use the EP process group and symmetric buffer, and return combined output only for the originating rank's rows. No attention object, scheduler, NIXL object or SGLang ForwardBatch enters fp8_mega_moe. All four ranks must enter the collective even when a rank has no live tokens. Submitting every replicated TP4 row on every rank would create four independent copies of the same routed work. Correct row ownership is essential.

3. Pinned vLLM Qwen already contains the needed high-level row-split/gather shape: Qwen3MoeSparseMoeBlock.forward, qwen3_moe.py lines 223-234, calls sequence_parallel_chunk before experts and TP all-gather after, then trims padding. models/utils.py lines 1047-1065 pads to a TP multiple and takes rank-contiguous rows. However ParallelConfig.use_sequence_parallel_moe lines 715-729 requires data_parallel_size > 1. Current attention TP4/DP1 therefore DOES NOT enable this path. A narrow backend-specific predicate or equivalent local adapter is required; a flag-only switch is insufficient. Do not enable generic all-to-all followed by MegaMoE dispatch as that would dispatch twice.

4. MegaMoE fuses dispatch, expert math and combine. It cannot be substituted as an ordinary TritonExperts GEMM. The integrated vLLM method must own its complete dispatch/combine and return an already-reduced local row shard. MoERunner._maybe_reduce_final_output lines 480-514 otherwise all-reduces an unreduced EP result; the new backend must correctly report output_is_reduced and skip both generic naive dispatch/combine and late EP all-reduce. Qwen's TP all-gather then reconstructs the full token rows once. Gate/top-k must use vLLM's current normalized routing semantics, not import SGLang's router.

5. Raw vLLM EP4 checkpoint shapes match native MegaMoE inputs before backend conversion: w13 FP8 E4M3 [32,3072,4096], scales FP32 [32,24,32]; w2 FP8 [32,4096,1536], scales [32,32,12]. Fp8MoEMethod.process_weights_after_loading/_setup_kernel, fp8.py lines 700-795, currently converts weights to the selected kernel format. The MegaMoE adapter must hook before that conversion and interleave only L1 weight gate/up in groups of 8; preserve original FP32 block128 scales and L2 weight. Do not reinterpret converted/E8M0 scales as original block scales. Disable fallback after interleaving because ordinary kernels would see the wrong L1 layout.

6. Komori's patch moves routing weight application to AFTER FC2 and accumulates weighted BF16 FC2 terms in FP32. This precision contract is part of reuse, not an optional optimization. Existing numerical gate checks exact Qwen local shapes, empty/tail ranks, hot experts/capacity, masked routes and changing graph inputs with relative L2 <0.025 and max scaled error <0.08. Reuse those cases and add TP4 row-shard/gather, spec2 verify shapes, zero-live-rank and padded-row cases; compare against independent FP32 GEMMs and the same activation quantization. Native late combine ordering need not be bitwise identical to the existing Triton backend; accuracy and official task validity remain gates.

7. Dependency blocker: vLLM embeds DeepGEMM 2.8.0 in vllm.third_party.deep_gemm; it exports fp8_fp4_mega_moe but not the required SM90 fp8_mega_moe, mega_moe_pre_dispatch_sm90 or transform_weights_for_mega_moe_sm90 APIs, and its image listing has no sm90_fp8_mega_moe.cuh. The SGLang image uses sgl-deep-gemm 0.1.5 and apache-tvm-ffi==0.1.11. Installing it as top-level deep_gemm would change ALL existing vLLM DeepGEMM consumers: vllm.utils.deep_gemm._import_deep_gemm explicitly prefers that top-level package. This would confound unchanged prefill/dense behavior.

8. Proposed dependency solution: stage the pinned sgl-deep-gemm wheel under a private namespace such as apac_sm90_deep_gemm, preserving vendored vLLM DeepGEMM as the selected prefill/dense provider. Inspected runtime Python imports in __init__, mega, cuda_helpers, utils and testing are relative; top-level callable aliases use __name__, so source-level isolation is plausible. Do not simply monkeypatch sys.modules['deep_gemm']. The tvm-ffi module/symmetric-memory ABI, two libraries in one process, JIT headers/cache ownership, exact wheel/hash and ordinary DeepGEMM resolution must be checked in the CPU image build before GPU allocation. If private namespace import fails or changes the vendored provider, stop and repair the bounded dependency layer before integrating the engine.

9. Weight memory: each rank's raw expert weight elements remain unchanged between native EP4 and MegaMoE EP4. Interleaving should replace/share the stored L1 tensor after loading instead of retaining an unnecessary second full copy. Symmetric workspace is extra and must be measured against the existing 8192-block decoder KV allocation. Use the decoder's actual maximum row-shard capture shape to choose bounded capacity (e.g. ceil(768/4)=192 before native alignment), not SGLang's 4096 capacity by habit. Kernel alignment, empty cases and warmup/init shapes must be verified before fixing that cap. Do not shrink the official KV allocation silently to make the kernel fit.

10. CUDA graphs and EAGLE: preallocate/shared symmetric buffers and packed weights before capture; warm/JIT all required geometries before capture. No host count reads or per-step collective setup. Different ranks must replay the same collective order with fixed addresses and masked padding. Keep the repaired dense draft EP=false/TP4. Changing target MoE row distribution must not mutate draft parallel config or verification row order. Capture success is not proof of replay share or correct changed-input replay.

11. NIXL: retaining attention TP4 and server DP1 preserves the currently verified KV sharding and gateway endpoint arrangement. That is the reason to prefer row-sharded MoE rather than copy SGLang attention DP4. It is an inference from the adapter boundary, not a new successful NIXL runtime test. Startup, seven server exports, target/draft graphs, KV transfer and official benchmark must all pass before declaring the port usable.

## Bounded next implementation

- Build gate only after source patch exists: private SM90 package, exact patched header and package pins, CPU import/API/source guards; assert ordinary vLLM dense/P MoE still resolve vendored DeepGEMM 2.8.0. Preserve repaired SpeculativeConfig.
- One finite four-H200 diagnostic through tagged Actions: reuse existing numerical harness with row slicing/gather, all target-verify sizes needed by spec2, changed-input captured replays, all-rank empty/tail/hot routes, and measured workspace. Measure full split + routing + predispatch + MegaMoE + output all-gather, not GEMM alone. Include comparable native MoE for the same tensors inside the diagnostic, not another whole benchmark sweep.
- Only after correctness, graph and memory gates pass: integrate decoder-only complete MoE pipeline with pre-load weight hook and correct already-reduced semantics. Keep all six TP2 prefills and their ordinary DeepGEMM provider untouched.
- Then one clean official c108/spec2 benchmark against E4, noprofile, same memory/cache and workload, explicit unavoidable changed factors documented. Verify parser/provenance/server-info/archive/accounting/run-success and all cleanup as usual.

Stop/redirect if ABI isolation fails, numerical/changed-input graph checks fail, full MoE pipeline fails to beat native for representative verify batches, or symmetric workspace forces material KV/topology compromise. Do not promise 7k from the 6.4k raw SGLang result: that run failed eligibility. No performance conclusion is made by this audit.

## Source provenance

SGLang image: /scratch/home/ACCOUNT/qwen/containers/38491386685.sqsh, SHA256 67a08ed37d06100e1b130f5d46a1af385346b550893dbfb76b6449c7c2ff1eb1, build2e7e6f4a29cbb1439ebab88c5a037595330f8f15, public sourcefc275674a97d78553b22597ee72c8fd48f5420f4. Archived provenance read at /srv/qwen-artifacts/38491386685/1/source/source/. The image is reused by bb77e51 and the live image size12224233472 matches archived metadata; full12GB hash was not recomputed in this CPU audit.

vLLM image: /scratch/home/ACCOUNT/qwen/vllm-containers/38650006335.sqsh, SHA256 d1b022670a5291cbc3ceff8a65048ea7572706c7ba7ccb217fe1d014bfc48aca from completed repaired-image verification. Selected source hashes independently read from this image in this audit:

| Source path within package | SHA256 |
|---|---|
| vllm/model_executor/models/qwen3_moe.py | 3a0dae7a1b0140bcbf1f7903135c72e5e44a8174c7fade026b77659c8377d7ee |
| vllm/model_executor/models/utils.py | 570a2eb26c7274c9790b6ff42e3fd30403ad6534c68d256f4d867d22e2924919 |
| vllm/config/parallel.py | f012ace89e3fb730a467f729f602679bc145c49f0d10721811c9d818570e7b2b |
| vllm/model_executor/layers/fused_moe/runner/moe_runner.py | 05893188402fe37229bf72d75088d66193c417d5d502daafd3a55e1bfedaa60c |
| vllm/model_executor/layers/fused_moe/routed_experts.py | 68c48e5df0ea4a7cc1f04436b7b9153eb53802ecd9bb55d38db8738508daac6d |
| vllm/model_executor/layers/quantization/fp8.py | 947540b443c15e9e5b4c08ff07f7a7c0e8d8f0d8494f3964c0172a772a779fe5 |
| vllm/utils/deep_gemm.py | 5401bbf2556f072f210817888c3e13918773b6b91fdcb67870a235cf205fb73e |
| vllm/utils/import_utils.py | 3236d7a25f93d4989f977d7146f497cdf377ce5d40386baedd1b45f7bfa85f08 |
| vllm/third_party/deep_gemm/__init__.py | 6505093c596a2393849f36619594e677931c6c5c0d302c9fdf4d71917ca9d10e |
| sglang/srt/layers/moe/mega_moe_sm90.py | 70e8782065a5958eff4a3bbd6986b2a02a8515aab34993d741f8a56658b31e8e |
| sglang/srt/layers/moe/mega_moe.py | fb299d42fae12c78c04cb1760806196f47933b74f0a552826967b77902b0d012 |
| deep_gemm/__init__.py (sgl0.1.5) | b33e89deacdce241f01f5070d321918f5e5480e3e6d3af569678d4192db4f2a7 |
| deep_gemm/mega/__init__.py (sgl0.1.5) | a3fa9f4ab614df8af29a0fcd8e1571e9e05cb1e9f2c02bbb77f4c730264d0339 |
| deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh (patched) | 5479807104bf1aeed4cc3ecc8ce57d2a835d68457f9cf4962ef3f15cc176e8d2 |

All read-only source extractions used verified owned /tmp/firmus-megamoe-* or /tmp/firmus-vllm-mega-read.* directories and were removed in finally blocks. No user source/image/run directory was modified. No CUDA import/kernel execution was performed by this audit.

## Private-provider CPU image gate prepared

The user explicitly authorized proceeding with both tracks. audit_namespace.py/cpu-isolation-audit.json verifies the exact published x86_64 sgl-deep-gemm0.1.5 wheel SHA e5e9f92cdf0515fe878bd81131bc25d2984e00c005824d2932af01a5e0d75261 and eleven runtime Python files against the existing SGLang image. Real pinned Python modules import under apac_sm90.deep_gemm with explicit Torch/TVM stubs; wrapper argument wiring and the exact vLLM resolver are checked, including a top-level-package negative control that displaces vendored2.8. This is Python wiring evidence only; no native ABI/GPU was loaded by that audit.

containers/firmus-sm90-megamoe privately stages the exact wheel under /opt/apac/sm90-provider, reuses Komori's unchanged precision patch SHA462d68230da3ab9a1e69ca85d9fa804b8edaf6a77bf0975726611a08fda6622b and verifies patched header5479807104bf1aeed4cc3ecc8ce57d2a835d68457f9cf4962ef3f15cc176e8d2 plus native libraryab8dc0accf34adbd64037202c917250e1c084fe5417f086805036cef99ef2959. Private root is not put on sys.path; a specific loader installs only the apac_sm90 namespace. No global pip/dependency change, no top-level deep_gemm package, and no vLLM serving backend change. Existing base provides apache-tvm-ffi0.1.11/torch2.13.0+cu130/vLLM0.30.0.

The actual Docker CPU gate imports both native libraries using a temporary libcuda stub, requires the original vLLM provider object before/after, checks the native SM90 entry points, and compiles the same three finite SM90 geometries used by Komori. The report is embedded in the built image and rechecked by the existing image checker; source/helper blob IDs are added to the existing manifest. Preserve the dense-draft EP repair. GPU launch, symmetric buffers, math, memory, changed-input graph replay and speed remain separate unpassed gates. Serving/client image pins are unchanged.

Root independently validated positive exact-wheel private installation, header/native/source hashes, and rejection of existing destination, corrupt wheel and modified package. Ruff across containers/scripts, Python AST and shell syntax pass in a self-cleaning local LF snapshot. A local stdin terminator error occurred after the checks had passed; corrected wrapper replay exited0. Temporary dependencies/wheel/snapshot were removed, no global host install. FullCI must pass before the one require-build-only AWS image gate. This build has no cluster tag and launches no H200 job; do not label it a MegaMoE serving implementation or performance result.
