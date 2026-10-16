# What Komori changed on the way to 6344.96 good TPS

This is a history audit, not a monotonic speedup claim. Read exact Git commits,
normal benchmark artifacts and effective worker exports rather than treating the
final launch flags as the whole optimization. The numbers below are separate
normal60/120/30 passes preceding each profiling pass, not active-profile TPS.

## Measured sequence

| Actions / source | Main stage / changes | c | Good TPS | Raw TPS | Good fraction | Mean ITL, ms | p95 TTFT, ms |
|---|---|---:|---:|---:|---:|---:|---:|
|38110812353 / f7484ae|P MegaMoE DP4/EP4; D still TP4/EP1; repaired heterogeneous KV staging|256|92.64|5825.44|1.5903%|28.057|5912.505|
|38164365415 / 2751028|D execution cap128 separated from transfer waiting slots, c192, staging/page-release and legacy TP4 SiLU/quant latency fixes|192|1743.97|4729.23|36.8762%|19.861|5817.604|
|38491386685 / bb77e51|D attention-DP4/EP4 MegaMoE, native total-request DP routing/per-iteration publication, nonblocking KV source-rank lookup|192|4747.55|6464.11|73.4441%|15.591|4220.602|
|38538953568 / 9d252ce|New native outer P router/load-reservation lifetime and ready-result handoff; test DeepEP-normal P with contiguous DeepGEMM; D remains MegaMoE|192|4211.22|6360.83|66.2046%|15.157|3278.972|
|38664373784 / e6e295bd|Keep new routing/handoff image, restore only P MegaMoE backend|192|6344.96|6611.00|95.9748%|15.749|2901.299|

Only the last row meets the90% good-request eligibility threshold. Good TPS is
stored goodput_requests_per_second times configured OSL200, including for
ineligible rows; the local gated-score0 is not measured throughput0. Source and
container/client changes, concurrency changes and different scored/drain
durations prevent interpreting the entire table as isolated single-factor gains.
There is one observation per row, with no estimate of run-to-run variation.

Audit evidence: `/srv/qwen-artifacts/<run>/1/SHA256SUMS`, exactly one
canonical suffix each for `results/results/score.json`,
`results/results/profile_export_aiperf.json`, `source/source/provenance.json`
and four `results/results/server-info-<worker>.json` files. All35 selected input
files match their manifest hashes. Some runs retain an extra
`qwen-firmus-pd-<run>-1/` directory prefix. This checks these35 analysis inputs,
not all native-report bytes in all five archives. Read provenance explicitly:
all five have `normal_benchmark_before_profile=true`.

## Strongest comparison and its limits

38538953568 and38664373784 use the same serving **and client** image SHA256
`8df47c11d7ede66fd19f12c7012a9ab5c97b13387a2655fcbce9f13e1c334110`,
12249612288 bytes. The source diff changes P's backend and updates an old image
pin; the actual38538953568 provenance confirms it already ran the fresh image,
not the older pin still present in its pre-build config file. Comparing selected
static fields in all four actual server exports finds only
`moe_a2a_backend: deepep -> megamoe` on P0/P1/P2 and no selected D difference.
Other selected fields include DP/EP/TP, attention-DP/LM-head mode, native routing,
token budgets, request caps, KV/page/cache settings, speculation, attention and
graph toggles, dtype and context. This is stronger backend evidence than comparing
arbitrary points in the earlier campaign, but still not a repeated trial.

The two MegaMoE/MegaMoE observations,38491386685 and38664373784, show raw TPS
6464.11 ->6611.00 while good fraction rises73.44% ->95.97%; mean ITL does not
improve. Thus final goodput is consistent with reduced TTFT tail as well as
retained backend speed. The intervening DeepEP trial shows that routing/handoff
changes did not guarantee an eligible result with a slower P backend. Their
individual routing, publication and handoff contributions are not isolated.

## What carries over to the current vLLM work

| Historical lesson | Current implementation / remaining evidence |
|---|---|
| Use measured full-stage kernel execution and retain MegaMoE on both roles | Modified MegaMoE/PDL is mandatory; the numerical adapter gate passed64 records. Dense DeepGEMM is a separate projection path, not a MoE fallback. |
| Bound D execution independently of queued KV-transfer requests | D32/rank and native waiting-for-remote-KV handling; maintain the matched cache budget rather than globally lowering client concurrency. |
| Avoid stale/skewed rank and worker load | Outer P least-live reservation lasts through the entire response body; explicit P rank reservations rotate ties. D uses native waiting/running/in-flight accounting. These are framework-specific implementations, not proof of identical native SGLang queue budgets. Inspect per-rank latency/distribution before declaring the historical imbalance reproduced or solved. |
| Ready output should not wait behind another long P forward | Current native MP nonconsuming queue readiness and early publication were validated through full serving, but did not demonstrate an improvement. SGLang's CUDA-copy-event readiness and vLLM's worker-RPC readiness are different boundaries; source similarity alone does not prove the same handoff latency. |
| Fix real transfer ownership / lifetime errors | Native Push completion telemetry accounts for all5857 warmup+measurement transfers in38872232338; no failure/expiry/preemption events and final caches empty. Do not copy obsolete heterogeneous TP1->TP4 staging work into the current TP1->TP1 path. |
| Do not port improvements targeting a superseded kernel path | Legacy TP4 Triton SiLU/quant and TP reduction fixes are not automatically applicable inside the current fused MegaMoE. P native compilation targets separately recorded surrounding work. Compile-only candidate38917392565 was cancelled before allocation after the subsequent native Push-order finding; source7295527 batches the changes and requires a fresh image. |

The useful pattern is to retain both service capacity and reliable handoff/load
behavior, using official TTFT/ITL and eligibility to assess the whole deployment.
The current vLLM gap remains unlocalized; neither this history nor similar
ITL establishes that the GPU or the scheduler alone is responsible. No extra
SGLang run, duplicate GPU experiment or new settings follow from this history
audit. Follow Issue21 for the active vLLM experiment and Issue24 for the separate
SGLang work.

## Concrete follow-up from the handoff comparison

Inspecting the actual pinned vLLM GPU wrapper exposed a later boundary than
engine-RPC response publication: completed Push block metadata binds before a
new forward but reaches the background writer only in post-forward. The guarded
source7295527 change submits completed prior P blocks and D registrations before
unrelated compute, fencing prior P target/draft stores on the current stream
first. CPU execution of the actual methods verifies submission order; it does
not measure its TTFT contribution or certify GPU transfer correctness. The
compile-only queued run38917392565 /24104 was cancelled before allocation,
archived with19 checksums and cleaned. Build38923173999 supplies the batched
source repair; use its independently verified image before any replacement GPU
submission. See configuration-comparison.md for the exact guards and evidence.
