# Matched configuration: Komori 6344.96 TPS versus vLLM MegaMoE

Reference: SGLang `e6e295bd`, Actions 38664373784 / Slurm 23571. vLLM attempted baseline: `26b3631`, Actions 38781401340 / Slurm 23872 (failed during decoder graph capture, before readiness or benchmark traffic). Source of actual SGLang settings is the canonical `source/source/server-info-*.json` plus worker startup logs; the prior vLLM launch 38775681156 reached and exported the same serving configuration but sent no benchmark traffic.

| Dimension | SGLang reference | vLLM reproduction / remaining difference |
|---|---|---|
| Hardware and topology | 2 nodes, 16 H200; three 4-GPU prefill groups and one 4-GPU decode group | Same; two P groups on head, one P and D on tail |
| Attention / experts | Attention TP1, DP4 and EP4 per group | Native TP1/DP4/EP4 with 32 linear experts per rank |
| MoE implementation | Modified SM90 MegaMoE; FP32 late routing-weight combine | Same kernel patch; vLLM method owns dispatch, expert math and combine, bypassing duplicate native communication |
| Target / draft precision | Official published FP8 target; BF16 Nebius EAGLE3 | Same revisions and draft conversion hash; original FP32 block scales; E8M0 disabled |
| Client | AIPerf 0.12.0, concurrency/workers 192, ISL4000/OSL200, 60/120/30, seed2026, first-turn cache bust | Same official command and successful registration/record-processor settings |
| P KV allocation | 131072 tokens per DP rank; page64 | 2048 blocks ×64 per DP rank |
| D KV allocation | 393216 tokens per DP rank; page64 | 6144 blocks ×64 per DP rank |
| Chunk / admission | P4096 tokens and48 requests per rank; D2048 and32 | Same numerical budgets. Scheduler implementations and chunk admission differ |
| Attention / KV | FA3 / FP8 E4M3 | FA3 / FP8 E4M3; framework-specific KV grouping and transfer metadata |
| EAGLE3 | Three steps, topk1, four draft-token rows | Three speculative proposals and up to four target verification tokens; native verification, draft scheduling and acceptance implementation |
| Graphs | P eager; D22 request-size buckets through32 | P target/draft eager; D native FULL_AND_PIECEWISE, same request buckets mapped to four verification tokens through128 |
| Prefix cache | P radix cache enabled; D disaggregated radix cache disabled | P native prefix cache; D prefix cache disabled. Implementations differ; official cache bust preserved |
| Outer P routing | Minimum live load, retain reservation through body completion | Least in-flight with rotating ties, reservation held through body consumption |
| Inner DP routing | SGLang `total_requests` | vLLM native waiting/running/in-flight accounting with KV-pressure penalty; not an identical dispatch policy |
| Prefill handoff | SGLang ready-result fast path before another long forward | vLLM native scheduler/NIXL response path; no literal scheduler-patch translation |
| LM head / dense draft | P `enable_dp_lm_head=false`, TP4 group; D DP-local LM head and dense TP1 | Native TP1 replicates these weights per DP rank; P sharding differs |
| NIXL | Native SGLang disaggregation | Native vLLM producer/consumer connector and HTTP proxy; both use NIXL but handoff/ownership differ |

The vLLM adapter requires the production 4096-row symmetric workspace, observed as 1608529184 bytes per rank during the prior full-model startup. It is shared across 94 serialized MoE layers. Matching cache token budgets does not imply matching total GPU-memory accounting or identical scheduler overlap.

Performance must be reported from normal official passes only. Profile-only 20/30/30 runs diagnose kernels and communication and cannot supply official throughput. Numerical synthetic tests establish bounded adapter correctness, not the organizers' full-set accuracy score; their stated threshold is 99% of an unchanged official-FP8 reference score.

The user confirmed that no separate organizer accuracy procedure/data is shared for this work. Preserve the published checkpoint, original scales and kernel arithmetic; do not claim the organizer accuracy threshold was measured. The active diagnostic uses the installed native adapter, independent reference and changing-input CUDA Graph replays at the production workspace capacity before full-model startup. Its results must be checked independently of serving readiness.
