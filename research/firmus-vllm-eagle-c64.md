# Firmus vLLM recreation of the optimized SGLang c64 run

Comparator: SGLang Actions [37888931369, attempt 2](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37888931369/attempts/2), commit `23463ed2036fb9706d40c7b0b4c4744f235b0e51`, Slurm 22058. Its archived official result is 3450.893144456974 good output tokens/s, 2173 requests, 97.8831% good, mean TTFT 1686.710 ms and ITL 9.530885 ms. Archive: `/srv/qwen-artifacts/37888931369/2/` on archive-host.

This experiment changes the serving framework while retaining the model and draft revisions, hardware class, worker packing and official client. It reuses the previously verified vLLM 0.30.0 image; no container rebuild, profiler allocation, repeated baseline, or PR is required. All computation is submitted by the tagged GitHub Actions push. The allocation has the normal 17-minute Firmus limit.

| Setting | SGLang comparator | vLLM recreation |
|---|---|---|
| Hardware/topology | 16 H200, four TP2 prefills on node0, two TP4 decoders on node1 | Same layout; actual assigned nodes recorded in logs |
| Expert parallelism | EP1 everywhere | Expert parallelism disabled everywhere |
| Target | Qwen3-235B-A22B-Instruct-2507-FP8, revision e156cb4efae43fbee1a1ab073f946a1377e6b969 | Same shared weights, BF16 activations |
| Draft | Nebius EAGLE3, ff0f0c0999f6c86b21dfd2f5803eebc4ac712224, BF16 | Same checked weights and compatible architecture selector |
| Speculation | 3 steps, topk1, 4 verification positions including bonus | eagle3 with 3 proposed tokens; native linear proposal and verification |
| KV / context | FP8 E4M3 / 8192 | Same precision and context |
| Cache capacity | max_total_tokens 131072 | block_size16 times 8192 GPU blocks; runtime group layout remains framework specific |
| Prefill chunk / memory | 8192 / 0.90 | max_num_batched_tokens8192 with chunking / 0.90 |
| Admission limits | prefill64, decode256 | Same max_num_seqs per role |
| Attention | FA3 | FLASH_ATTN with flash_attn_version3 |
| Symmetric memory | Enabled on prefills, disabled on decoders | Native vLLM SymmMem selected on prefills; competing custom/FlashInfer all-reduce disabled there; disabled on decoders |
| MoE | Prefill CUTLASS with SGLang shuffle/routing patches; decoder auto | Prefill flashinfer_cutlass supports H200 block-FP8; decoder auto |
| Graph buckets | Committed decode batch-size list, 1..512 | Same numeric list passed to vLLM; capture units and speculative padding differ between engines |
| Routing / transfer | SGLang router, round-robin prefill, NIXL | Existing vLLM proxy, round-robin prefill/decode, NixlConnector |
| Client | AIPerf0.12.0; concurrency64; 60/120/30s; ISL4000, OSL200; seed2026 | Same image, tokenizer revision and official command/parser |

SGLang's four source/kernel patches cannot be copied into vLLM. Native scheduler, graph construction, draft execution, routing, fused kernels and communication can still differ. An observed result compares these two concrete configurations, not an isolated causal proof about frameworks or a general winner from one run.

Before allocating GPUs, the exact argument schemas were checked inside the pinned vLLM filesystem on a CPU platform. Shell syntax, static checks and 19 CPU launch/parser/provenance tests passed. Actions runs the full validation suite before provisioning and submission. All six effective server configurations and launch commands are archived before the workload runs. Completion requires the official parser result, terminal scheduler state, raw exports and canonical archive, followed by workflow cleanup.

## Result

Pending submission and execution. No performance claim is made until the official result is collected.
