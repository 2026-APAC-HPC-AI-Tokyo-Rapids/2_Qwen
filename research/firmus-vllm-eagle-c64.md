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
| Graph buckets | Committed decode batch-size list, 1..512; prefill graphs also captured | Same numeric decoder list; prefill graph capture disabled after the startup failure below. Capture units and speculative padding differ between engines |
| Routing / transfer | SGLang router, round-robin prefill, NIXL | Existing vLLM proxy, round-robin prefill/decode, NixlConnector |
| Client | AIPerf0.12.0; concurrency64; 60/120/30s; ISL4000, OSL200; seed2026 | Same image, tokenizer revision and official command/parser |

SGLang's four source/kernel patches cannot be copied into vLLM. Native scheduler, graph construction, draft execution, routing, fused kernels and communication can still differ. An observed result compares these two concrete configurations, not an isolated causal proof about frameworks or a general winner from one run.

Before allocating GPUs, the exact argument schemas were checked inside the pinned vLLM filesystem on a CPU platform. Shell syntax, static checks and 19 CPU launch/parser/provenance tests passed. Actions runs the full validation suite before provisioning and submission. All six effective server configurations and launch commands are archived before the workload runs. Completion requires the official parser result, terminal scheduler state, raw exports and canonical archive, followed by workflow cleanup.

## Result

Actions [38070007994](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/38070007994), experiment commit `fb0845131e3dcc3ba987d1607bd52a7adeff0519`, submitted Slurm **22463**. Validation and provisioning passed. At submission the job was pending `Priority`; no performance claim is made until the official result is collected.

The preceding Actions run 38069819876 stopped in CPU validation because two old tests expected the former baseline profiler launcher. It submitted no scheduler job. Those tests now check that this optimized launcher rejects profiling before starting workers; the active EAGLE3 argument tests and generic profiler-wrapper test remain.

Provisioning retried a transient SSH interruption before submission. The unused staging directory was verified against the run provenance and active Slurm command, then removed. Only one compute job was submitted. The active directory is `/scratch/home/ACCOUNT/qwen/runs/pd-38070007994-1.Wxuwei5g1M`.

The inherited Actions summary formatter has a stale topology label for this new configuration. Effective worker commands and server-info artifacts are authoritative. The reporting-only correction identifies the optimized c64 EAGLE3 layout as P4TP2/D2TP4 EP1; its 15 CPU tests passed. It does not alter the submitted job.

Background follow-up `firmus-vllm-eagle3-comparison` checks this same job and its Actions collection every five minutes, remaining quiet while unchanged. Its scope includes collecting the official result, completing this comparison and stopping after completion; it does not authorize additional controls or optional profiling. The foreground read-only polling process was stopped when this follow-up was registered.

### First allocation: startup failure, no benchmark result

Slurm 22463 ran on g1/g4 from 2026-10-10 22:18:36 to 22:23:34 (scheduler timestamps), then failed with exit 1:0 after 4m58s. The first fatal error is `CUDA error: invalid argument` in `logs/logs/prefill-2.log`, during target PIECEWISE CUDA graph capture at 46/51 buckets. The worker died at 06:23:12 +08:00; later `RuntimeError: cancelled` and other workers' cancellation messages are shutdown consequences, not the first error. No official client, score, server-info snapshots, or run-success marker was produced.

Target and draft weights loaded successfully. Logs confirm EAGLE3 auxiliary layers (2,47,91), prefill FLASHINFER_CUTLASS FP8 MoE, decoder TRITON FP8 MoE, and prefill symmetric-memory environment enabled. This does not establish which CUDA kernel or communicator caused the capture error. Other prefill workers progressed past capture before cleanup.

Actions collection and cleanup succeeded despite the benchmark failure. Canonical logs/source archive: `/srv/qwen-artifacts/38070007994/1/` on archive-host; its SHA256SUMS verified. The active Firmus run directory was removed by workflow cleanup.

The necessary replacement disables CUDA graph capture only on prefills using `--compilation-config '{"cudagraph_mode":"NONE"}'`. Torch compilation, symmetric memory, target/draft precision, worker layout, cache capacity, decoder graph buckets, and official client remain as above. This bypasses the observed failing stage; it is not a claim that the underlying CUDA fault is diagnosed. SGLang's successful prefill graph capture makes this a material remaining comparison difference. Before replacement submission, the pinned vLLM CPU CLI schema validated this setting; shell syntax, 34 reference/profile CPU tests and Ruff passed. No image rebuild or optional experiment is submitted.
