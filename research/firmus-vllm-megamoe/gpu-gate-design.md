# Finite four-H200 MegaMoE gate

Entrypoint: `torchrun --standalone --nproc_per_node=4 scripts/megamoe/gpu_gate.py --output <run>/results/gate.json`. Root owns Actions/Slurm integration, image/source provenance, archive and cleanup. This script never submits jobs, loads a checkpoint/tokenizer, starts a server or changes the serving configuration.

The actual private image import/three-cubin build passed separately; the new GPU harness has **not run on a GPU**. CPU evidence is gpu-gate-cpu-check.json:160 fixed host row/replay plans, contiguous ownership without duplication, mask/tail/idle coverage, unsupported-plan rejection, original Komori quantizer/dequantizer AST identity, call order/argument wiring for actual pipeline methods with explicit sentinels, and summary math. Pinned Ruff0.16.2 and Python AST/CLI checks pass. These checks do not demonstrate a successful native GPU constructor, CUDA capture, numerical kernel or collective.

## Compared paths

Native: replicated BF16 synthetic gate GEMM and pinned vLLM fused_topk -> actual MoEPrepareAndFinalizeNoDPEPModular.prepare -> actual TritonExperts.apply with preallocated exact native workspace shapes -> native NoOP finalizer -> ordinary GroupCoordinator custom all-reduce. This uses the same full-route Triton class and invalid-expert defaults as serving, rather than functional fused_experts, whose assignment helper has a different ignore-invalid override. It omits engine scheduling/chunk-manager setup and model-wide attention/EAGLE, so it is a MoE pipeline measurement.

Mega: contiguous token-row split/copy -> BF16 synthetic gate GEMM and the same native fused_topk -> existing private SM90 predispatch -> exact existing patched native fp8_mega_moe -> output all-gather through the same GroupCoordinator. All ranks participate even with no live rows. Target attention is not converted to DP4. There is no second generic dispatch or all-reduce.

The coordinator's name is deliberately `tp`: pinned CudaCommunicator disables custom all-reduce for other group names. The harness requires an active custom communicator and `should_custom_ar` support for every native output shape, rather than silently timing another backend. Symmetric-memory/FlashInfer ordinary all-reduce and E8M0 are disabled explicitly; a conflicting inherited value fails. The native MegaMoE kernel's own symmetric workspace remains active as required.

## Precision and route validation

The Mega reference reuses Komori bb77e51 scripts/check-firmus-megamoe.py's original equations: FP32 independent GEMMs, E4M3 group128 first activation, group64 second activation, BF16 FC2 terms, routing weights applied once after FC2, FP32 combine. SourceSHA9f7e3fe8c173f11f099eee7b0eb02eb038e90a729011c721d9b8a7bb8893eea3. RelativeL2<0.025 and maximum-scaled-error<0.08 are unchanged; masked rows must be exactly zero. No tolerance was increased.

Native Triton has a distinct contract: FC1 stores BF16 before the fused float SiLU/Mul + group128 second-activation quantization, router weights are applied before BF16 FC2 output storage, local weighted terms reduce to BF16, then ordinary cross-rank reduction. Its separate independent reference includes these rounding stages. This is tolerance-based verification, not bitwise emulation of Triton/WGMMA/SiLU/collective accumulation order. The Mega and native references are intentionally separate rather than hiding their known precision difference.

Global replicated gate GEMM and row-sharded gate GEMM can round differently near top-k boundaries. Each graph's routes and weights must first exactly match an eager call at that same GEMM shape for current input data, catching stale replay routing. Then both outputs are compared to independent references using their actual routing. Global native-versus-Mega ordered route-slot differences, changed-weight counts and maximum/mean weight differences are recorded on every replay without a new acceptance threshold. Subsequent official task/model validation is still required before claiming model equivalence. The gate weights are synthetic, not an actual checkpoint's trained router.

## Fixed scope and measurements

Global rows1,4,7,32,97,128,288,324,384,768; random, all-hot-owner, empty/tail, and masked patterns; four changed-input replays per case. Global inputs remain at stable addresses and ownership is contiguous ceil(M/4) rows per rank with masked padding. No CPU count is read from a device buffer in the execution pipeline. Expected numeric references recompute outside timing; they include current exact gate shapes and actual padded inputs.

The native API aligns Mega capacity from requested192 rows per rank; the returned aligned capacity and symmetric buffer byte count are recorded. This does NOT validate larger decoder warmup/draft-prefill/initialization shapes; those remain future serving-integration checks. Fixed numerical cases replace the old SGLang4096-per-rank capacity stress because this gate is specifically bounded to the proposed decoder verify row range. It is not a general prefill-capacity validation.

Four representative row sizes128,288,324,768 receive20warmups and100alternated paired uninstrumented graph-replay samples. Report full-path CUDA event median/p95 using maximum rank per paired iteration, raw rank samples, and paired host wall time. Separate eager component measurements include routing/prepare/native math/reduce or split/routing/predispatch/Mega math+combine/gather; these component events are instrumentation and are not engine critical-path attribution. A valid negative timing comparison is a completed measurement, not an experiment failure.

All four visible H200s are required and ranks0..3 plus names/UUIDs are reported. Available UUIDs must be distinct. CUDA_VISIBLE_DEVICES, NVIDIA_VISIBLE_DEVICES and SLURM_JOB_GPUS are recorded; root's scheduler wrapper must expose only the four allocated devices. Source hashes guard exact native MoE/router/communication files. Private provider guards are reused from the verified image.

Peak allocated/reserved memory, free memory, aligned capacity and symmetric workspace are standalone measurements. No complete target/draft model or decoder KV cache is loaded, so actual decoder KV coexistence remains unmeasured. `useful_for_serving` stays null even if timings improve. No official throughput score is produced or inferred.

## Output and failure contract

Final JSON schema is `vllm-sm90-megamoe-gpu-gate/v1`; completed/correctness_passed/graph_passed become true only after all cases and timings finish. `world_size=4`, `gpus` has ranks0..3, `official_benchmark_score=null`; negative timing remains successful. `timing_advantage` only indicates whether all four sampled median ratios favored Mega. This does not establish official goodput or adoption.

Rank0 saves incomplete progress after each case; on exception it records failure type/message and keeps completion flags false, then raises so torchrun/Actions can collect the initiating traceback. No retry/fallback or accuracy relaxation. Root should archive terminal source/result/logs and verify canonical checksums before deleting only this owned run directory.


## Mask representation correction after the first actual GPU numerical checks

Run38696933235/23677 passed nativeABI/CUDA symmetric-buffer construction plus all four changed-input replays on all four ranks for M1 random and hot patterns. It then failed native comparison during the next empty_tail case (the partial failure file did not record the exact replay); Mega's check evaluated first and passed. Native output was nonzero on masked rows, relativeL2=1.785/maxscaled2.373, not a tolerance-boundary discrepancy. The harness had copied Mega's -1 input-route encoding into the native Triton assignment path. Native expert-map -1 marks a nonlocal expert after assignment, which is distinct from feeding negative global top-k IDs before assignment; unassigned intermediate slots can retain stale data.

The comparator now keeps the native router's valid global IDs and masks the corresponding weights to zero. Mega retains its explicit -1 encoding. Both represent the same independently checked weighted contributions, including exactly zero all-empty/padded rows. Do not clear outputs after execution, skip empty-rank collective work, change GEMM/reference equations or relax tolerances. Each backend's encoding runs inside its measured graph, so native zero-weight work remains in timing. Graph-versus-eager checks use the same backend representation; route-rounding statistics exclude explicitly masked slots.160 host layout/mask plans exercise effective weighted-output equality and valid native IDs before another GPU release. Failure messages now include backend/rows/pattern/replay context without extra timed instrumentation. This is a harness input-contract correction, not an adopted serving/kernel optimization or proof of complete GPU correctness.
