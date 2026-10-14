# Compact local-expert routes: CPU contract before a vLLM kernel change

Owner: aizakovzky, issue21 comments6158960204 /6159096057 /6159268174. This is the distinct vLLM-native decoderEP4 packing track. No source/image/GPU job change is made here. It does not implement teammate NVLS, Nsight capture, SGLang MegaMoE or draft-model experiments.

## What the reference establishes

compact_routes_reference.py compares independent full-top8/zero-filled and compact-local-route formulations. Frozen fixtures contain128experts,4ranks,top8,H128/intermediate128,128-element activation quantization groups; these small CPU dimensions are not the official Qwen kernel shapes. Fixture weight bytes/scales are generated once and unchanged; no real checkpoint is loaded or requantized. Activation reference uses BF16 round-to-nearest-even, finite E4M3FN codes, per-row group scales, SiLU, routed weights before BF16 output cast, original route-slot combine order and the same rank-reduction order in both paths.

30cases cover M1/7/32/97/288 with random routes,allroutesonrank0,expert-boundaryIDs,maskedrows,allinactive,andchangedroutes.120rankcomparisons and30rank-reductioncomparisons agree bit-for-bit under this CPU arithmetic. Packed activation FP8 bytes and scales match the corresponding full-reference rows. All254finite FP8 bit patterns roundtrip and126positive midpoint ties follow RNE. Unused fixed-capacity payload rows are poisoned with NaN; correct outputs must not read them. Per-rank capacity remains M*top8; hot-rank cases fill that whole capacity. Source/model CUDA rounding, group-quant implementation, graph replay and organizer accuracy remain unverified.

## Device metadata contract

For each bucket of physical M rows, accept topk expert IDs/weights and actual valid-row metadata from the pinned engine. The fixture valid mask is a contract to audit; do not assume the engine already provides this exact interface. AttentionTP4 remains unchanged; local expert ownership follows the native EP4 map.

- original_route_id = token_id*top8 + route_slot. Preserve this identity and the original routing weight.
- Device-only local predicate combines actual row validity and expert ownership. Emit local route count,local expert counts and a stable expert-grouped permutation. No host scalar extraction or GPU-to-CPU count transfer is allowed in the decode loop.
- Keep original token ID,original route ID,local expert ID,and packed row ID distinct. Do not reuse compact row indices as token IDs.
- Inverse route map has shape M/top8, with -1 for absent/nonlocal/inactive routes. Reinitialize this map every invocation; an empty rank must not reuse an old count or route map.
- Physical route capacity is M*top8, because all8routes can belong to one rank. Padded expert alignment needs additional bounded capacity: at most M*top8 + local_experts*(BLOCK_SIZE_M-1), checked against the selected kernel's exact allocation convention. This is an upper bound, not a native CUDA allocation assertion.
- Kernel launches can use static bucket capacity and device-side live counters/masks. Padding lanes must never affect group scales,weighted outputs or another token. No unwritten payload may be read by activation,quantization or combine.

## Integration boundaries requiring actual kernel work

1. Quantize the original hidden input per row/group once, preserving the existing E8M0-off scale policy. Packing gathers already quantized input rows and their matching scales; it must not re-estimate scales across routes or tokens.
2. Existing alignment can ignore nonlocal experts, but merely changing that flag is insufficient. The current up GEMM writes C by original route ID. A compact implementation needs separate original-input lookup and packed-output lookup, with local expert alignment and correct padded-lane masking.
3. Apply SiLU/block-FP8 quantization only to live packed rows. Retain quantization group boundaries and numerical policy; changing weight precision or E8M0 is outside this candidate.
4. Down GEMM consumes packed activation rows and their scales, applies the original routing weights in the same stage, and emits packed BF16 contributions.
5. Gather/accumulate contributions through the inverse map in original top8slot order, skipping -1 entries. Preserve the existing EP reduction semantics and its communication backend. A faster FP32 atomic-scatter combine would alter summation order and is a separate numerical change, not part of this first contract.

The CPU reference sorts and allocates on the host deliberately; those operations are not a serving implementation. It cannot be inserted into the GPU decode loop. A CUDA/Triton implementation would need device-side count/prefix/permutation kernels and corresponding GEMM address changes in the consuming repository, a traceable new build, and GPU numerical/capture validation through Actions.

## Benefit arithmetic and limits

Under perfectly balanced routing,eachEP4rank owns about1/4ofroute rows. Relative to naive full-route EP4,compact gate/up/activation work touches about1/4as many rows;relative to existing TP4/EP1,the4timeswiderexpert dimension offsets that row reduction for activation element count. Wider GEMMs may still be more efficient. Compact down/combine payload can avoid nonlocal route-output traffic, but end-to-end savings depend on actual kernel time and route skew. Hot-rank fixtures have no local-route row reduction; processing all weight bytes is not automatically cheaper. Do not interpret these counts as a4xspeedup, lower reserved memory, or guaranteed7k throughput.

## Gates before shared-queue GPU use

- Consume the owners' current profile/NVLS/MegaMoE findings and check fresh issue21 ownership. Do not duplicate their capture or GPU measurements.
- Establish target decode batch/expert histogram or a clearly bounded shape range. Identify whether wider GEMM or full-route zero/activation/combine is an exposed cost; aggregate incomplete trace shares are not enough.
- Complete implementation-level GPU fixtures covering native128-block FP8 scales,BF16outputs,emptyranks,hotexperts,multiplechangedreplays,maxcapacity and graph buckets. The CPU contract is a prerequisite,not a substitute.
- Then run at most one distinct official candidate versus the documented E8M0-off control, keeping NVLS state/backend/proposals/concurrency/precision fixed. A major source/kernel change requires a new image and build provenance. No manual sbatch or blind parameter sweep.
- Official accuracy/precision requirements remain separate from the local workload/SLA parser. Preserve raw output,attemptcorrect archive/checksums/terminal/accounting/cleanup.