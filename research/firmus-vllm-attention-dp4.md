# Native vLLM attention-DP4 candidate

**Final result:** rejected. Official score0; raw4788.51tok/s;70.89%good. ITL caused839 misses versus30TTFT misses. Exact parser/rawrecords/77canonicalchecksums/exports/provenance/accounting/GitHub artifact removal and workflow cleanup verified; own terminal staging directly confirmed absent. Continue the separate full modified-MegaMoE recreation; no native tuning rerun.

Owner: aizakovzky, issue21 vLLM structural track. Prepared2026-10-15. Komori retains SGLang MegaMoE/SLA; Nattyu3's basic-track tuning and Katayama's profiling are separate. No teammate branch/job is changed.

## Decision and comparison

The completed MegaMoE standalone gate38699779914 proved finite numerical/graph compatibility, but complete-path median savings were only3.74% at288 global rows and6.39% at324 rows against our regressed nativeEP4 architecture. It did not establish a useful full serving port or7k. Rather than promote that small-gain TP4-preserving adapter, test the larger attention-parallelism change using native vLLM kernels first. This is one performance candidate, not a new baseline sweep or profile.

Primary comparator: native decoderEP4 clean38652678784,4516.47 qualified/4998.14 raw tok/s,90.36% good. Adoption target: safer E4 vLLM38488532230,5265.31 qualified/5330.18 raw,98.78% good. Team architectural reference: Komori SGLang38664373784,6344.96 qualified/6611.00 raw,95.97% good. Single runs/configuration differences do not isolate kernel causality.

The hypothesis is that TP1 attention and dense draft remove decoder TP reductions and change the attention geometry beneficially. Native allgather/reducescatter for experts, four synchronized schedulers, load imbalance and fragmented admission can offset the gain. No speedup prediction or bottleneck proof follows from source inspection.

## Exact candidate

| Item | Previous nativeEP4 | Candidate |
|---|---|---|
| Hardware | Two nodes,16H200 | Same |
| Prefill | Six TP2/EP1 workers,DeepGEMM | Unchanged |
| Decoder attention / dense EAGLE | TP4,DP1 | TP1,DP4 |
| Decoder experts | Native Triton,TP1/EP4 | Same native Triton,TP1/EP4 |
| Expert communication | Replicated rows plus native reduction | Native allgather_reducescatter acrossDP4 |
| D KV budget |8192 blocks ×64 =524288 tokens |2048 blocks ×64 ×4DP =524288 aggregate |
| D admission |256 sequences |64 ×4DP =256 aggregate |
| D scheduled token ceiling |8192 |2048 ×4DP =8192 aggregate |
| D graph coverage |FULL_AND_PIECEWISE,53 mapped buckets through768tokens |Same mode,29 mapped buckets through192tokens perDP rank |
| Client/model |c108,official60/120/30,FP8 target,BF16 Nebius EAGLE3/spec2 |Unchanged |
| Other settings |FA3,FP8E4M3 KV,page64,8192 context,.90 memory,D prefixoff,E8M0off |Unchanged |

Same exact native serving image38650006335, SHA256 d1b022670a5291cbc3ceff8a65048ea7572706c7ba7ccb217fe1d014bfc48aca; same pinned client4b919a7465f1647f4eddc677393cef00f2f390a8544b52a5f6b2456dce940d62. No provider/kernel/source dependency changes or image rebuild. This candidate does not use MegaMoE. Existing dense-draft EPfalse repair is retained. Decoder internal DP load balancing uses one public API endpoint; existing P proxy/routing is unchanged.

## CPU evidence and runtime gates

`firmus-vllm-megamoe/check_dp4_cpu.py` executed AST-selected exact native-image functions without Torch/vLLM imports. Seven source hashes and results are in `dp4-cpu-contract.json`. Native engine-core assignment yields distinct decoder_dp0..3 IDs; actual NIXL scheduler assignment yields ports5700..5703. Native TP mapping for target and draft attention groups at PTP2→DTP1 reads both producer ranks with distinct head slots and one consumer notification. Native speculative metadata function gives equal factors for matched P/D configurations. Target and draft each have four KV heads; block64 stays equal, so the known heterogeneous-block restriction is avoided.

Exact native source also contains idle-rank dummy passes and target/EAGLE DP batch synchronization. The draft factory defaults are not evidence that the target has DP1: the proposer coordinates with the target VllmConfig. Existing dense-draft EP correction stays at the native call site. These source checks do not demonstrate GPU collectives, graph execution, copied KV contents or trained-model output equivalence.

Nominal target94+draft1 FP8 attention payload is12750684160bytes/GPU for either TP4×8192blocks or TP1×2048blocks. This is a payload calculation only: padding, cache groups, dense replication, graph pools, native workspaces and allocator overhead require runtime validation. Do not equate the logged available-KV budget with measured tensor allocation or silently shrink the aggregate pool to fit.

Cheap validation must pass actual shell launch/role construction, runtime-config guard positive and corrupted DP/cache/admission/backend/dense-draft cases, full CPU tests, Ruff, shell syntax and Actions definitions before one tagged push. Profiler launch plumbing was removed from this candidate; an explicit true/invalid PROFILE value fails before worker launch. General historical profile validators remain unchanged and strict.

Before interpreting performance, verify four DP engine/rank placements and native kernels, target and draft captures on all four ranks, seven API server exports, P/D NIXL handshake/actual request completion, no errors/mismatch/cancellation and official parser eligibility. Archive attempt-correct raw JSON/JSONL, exports, metrics, provenance/accounting and run-success; independently replay parser/check hashes, then verify workflow/GH/staging cleanup. If it fails, identify the first fatal and make only a focused repair. Never duplicate or manually sbatch.

Validation: full CPU372 passed, one existing skip,22 subtests; Ruff, shell syntax, YAML and actionlint with ShellCheck passed. The first local full-suite attempt exposed an incomplete test snapshot and missing temporary zstd executable (not candidate failures); the completed snapshot/temporary tool rerun passed. One runtime-only tagged compute push is being released; exact Actions/scheduler provenance must be verified before mapping. Runtime memory/KV/graph/throughput remain unproven.


## Released run

Exact compute2a231009b6607c8ab17dfb5d5b2496cfa444466a created [Actions38716553408 attempt1](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38716553408). Full CI116360534655 and provisioning116360815347 succeeded; execution116362060724 is waiting for the scheduler. Exact source/run/attempt/profilefalse/benchmark/official/both image identities and seven staged Git blobs were independently verified before mapping Command/WorkDir to `/scratch/home/ACCOUNT/qwen/runs/pd-38716553408-1.4gms6V038B` and its scripts/submit.sbatch.

Slurm23738 was submitted2026-10-15 10:00:02UTC(19:00:02JST); last check PENDING Priority, zero GPU runtime. No captures, readiness or performance result yet. Canonical expected `/srv/qwen-artifacts/38716553408/1/`. Never duplicate or clean staging before terminal canonical checks. Separate Luna max heartbeat follows this exact run every10minutes and escalates meaningful changes to Sol. Issue21comment6189486835 records the mapping. Previous gate23684 and profile23572 remain closed and cleaned. Original checkout's four submission/build deletions are preserved.


## Terminal result: correct execution, rejected performance

Actions38716553408attempt1 /compute2a231009 /Slurm23738 ran on g1/g4 from2026-10-15 14:04:56 to14:17:04UTC(23:04:56–23:17:04JST),12m08, FAILED1:0. Seven workers ready after422s; all seven server-info exports match P6TP2EP1/c108/spec2/P58graphs and DTP1DP4EP4/nativeTriton/perrank2048blocks64/admission64/batch2048/FULL_AND_PIECEWISE29mappedbuckets. All four decoder ranks log completed target/draft capture passes; capture/config does not measure replay share. Actual official60/120/30 benchmark completed,4384 total JSONL records including2961 measured,0requesterrors,0OSLmismatch,was_cancelledfalse.

| Metric | Native DP4 result |
|---|---:|
| Official score |0, ineligible |
| Raw output throughput |4788.508031953611 tok/s |
| Good requests |2099/2961 (70.88821344140493%) |
| TTFT mean / p99 |571.3148749780479 /2908.097104000023 ms |
| ITL mean / p99 |19.47604366509797 /21.91144158492462 ms |
| TTFT misses / ITL misses / both |30 /839 /7 |

First fatal is the official parser eligibility gate `good_request_fraction is below 0.90`; no preceding worker/proxy fatal was found. Later scheduler cleanup signals are teardown, not the cause. Exact compute-pinned parser independently replays identical score.json and identical required rejection. Independent raw per-request SLA classification reproduces2099good and the miss counts. Server-side low TTFT alone does not make this configuration eligible: ITL misses dominate. Compared with E4 best5265.31qualified/5330.18raw and nativeEP44516.47qualified/4998.14raw, this single measured DP4 configuration is rejected. Do not claim an isolated architectural/framework effect or infer that MegaMoE shares the same outcome.

Canonical `/srv/qwen-artifacts/38716553408/1/`: all77 SHA256SUMS entries independently pass; exact source/run/attempt/profilefalse/official/server+client images, raw JSON/JSONL, seven exports and accounting present. No run-success expected. Actions collection/upload/summary/terminal staging cleanup succeeded; archive116474788676success and GitHub artifacts empty. Direct read-only check confirms exact terminal staging absent. No native tuning rerun: human clarified full Komori modified-MegaMoE architecture recreation as primary. That distinct adapter is now CPU-image verified and has one finite correctness/graph gate before full serving.
