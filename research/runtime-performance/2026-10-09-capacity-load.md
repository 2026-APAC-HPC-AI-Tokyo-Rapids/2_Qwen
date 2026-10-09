# H100 / H200 capacity and load validation

2026-10-09. Targets remain ASPIRE 2A+ H100 6000 and Firmus H200 10000 good output tok/s, under the official 4000/200 workload and joint TTFT/ITL SLA. The user requested validation on both clusters and bundled conditions to conserve allocation points. This campaign therefore skips pool-only c48 and intermediate load runs. Pool and concurrency effects cannot be separated from these comparisons.

## Conditions and provenance

| Platform | Baseline | Candidate | Actions |
|---|---|---|---|
| ASPIRE H100, 2 nodes / 16 GPU | main fb7dfe6, P/D pool131072, c48 | 8be82d7611f2155e7341ece484cf473488c6a7b5, P131072 / D196608, c64 | [37872490282](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37872490282) |
| Firmus H200, 2 nodes / 16 GPU | e481de9, P/D pool131072, c48 | f0849435d406759e53e1923eb369882c26f88b3d, P131072 / D262144, c96 | [37872430145](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37872430145) |

Both keep P2TP4/D2TP4, EP1, EAGLE3 steps3/draft4, FP8 KV, the prefill kernel combination and symmetric memory, existing decode graph buckets, and the official 60/120/30 client. Firmus Prefill max-running follows its client concurrency and therefore becomes96; Decode max-running remains256. The requested image is the existing four-patch image with SHA256 `182fbb6df7f941aa688a393dfc181bb269f3112a6d34c33f6007ddc5d09f0a42`. Actual selected image must be verified in each collected provenance.

Before push, runtime shell syntax, Ruff, worker-argv/official-parser tests and each checkout's full GPU-independent test suite passed. Historical raw fixtures were preserved; experiment-specific expected arguments were updated. No source/container rebuild or profiling is requested.

## Latest Firmus baseline audit

[37868180304](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37868180304), commit e481de9, Slurm21951: COMPLETED 0:0, allocation elapsed9m18s. Official score3170.205291 good tok/s, 1940/1950 good requests (99.4872%), TTFT mean1291ms/p99 2393ms, ITL mean8.504ms/p99 10.473ms. The commit's parser replay equals the saved score and all48 archived file hashes match. Raw is `/srv/qwen-artifacts/37868180304/1/`.

Full8192-token Prefill log samples show median input throughput32384.925/32540.14 tok/s per worker and queue median2/1. Decode log samples show running-request median16/15, maximum26/25, and accepted-length median2.21/2.22. These are log-sample statistics including warmup/tail, not time-weighted profiling-phase estimates. They motivate checking Prefill queueing alongside Decode admission, not treating spare VRAM as proof of a pool bottleneck. Post-initialization free memory is approximately68.47GiB on Prefill and75.22GiB on Decode.

The preceding37812345274/Slurm21893 failed because the image lacks jq (worker exit127); e481de9 moved parsing to the host and the successful baseline verifies that fix end to end. It is not an engine performance failure.

## ASPIRE budget

`myprojects -p 59010014` snapshot09/10/2026-09:40:02: grant64000, used47009.672, balance16990.328, In Doubt0 SU, expiry2026-11-15. Rate128 SU/GPU-hour. Preserve6144 SU for final validation; initial exploration wave cap4096 SU. This new H100 allocation has the existing30-minute wall limit, so reserve at most1024 SU before submission. Previously queued217513/217516 each reserve1024 SU and are unrelated; do not stop them. Balance less existing2048, this new1024 and protected6144 leaves7774.328 SU before further exploration. Recheck accounting after terminal state; never subtract prepaid In Doubt twice.

Old project50000108 shows21398.148 SU but expired2025-10-31 and is excluded. No assumption of extension or transferable credit is made.

## Results and decision

Submitted handles: ASPIRE PBS217936 (`pd-37872490282-1.9JM2r5J7SM`), Firmus Slurm21957 (`pd-37872430145-1.D5gv5XmQFS`) and follow-up21963 (`pd-37873721300-1.lgsZ9dvYv1`). Both clusters' selected image sidecars match the requested SHA; ASPIRE runtime provenance remains pending its start. Official full-set accuracy is a separate acceptance gate and has not been run in this capacity/load campaign.

ASPIRE remains queued as of the later10:40 accounting snapshot: balance16990.328 SU/In Doubt0, no estimated start or execution vnode. Queue comments changed from insufficient memory to insufficient free nodes. Its terminal state, runtime parameters, artifacts and score remain pending. Do not claim performance success from Actions submission alone. Repeat promising candidates before adoption; no candidate is merged into main based on a single unpaired run.

### Firmus first candidate: SLA rejection

Slurm21957 ended FAILED1:0 after9m45s on g1/g4. AIPerf completed2006 requests with no request errors, but only38 met joint SLA (1.894317%). The scorer correctly rejected the run below90% and its eligible score is0. Raw output3191.425 tok/s and input63605.817 tok/s are not a successful score. TTFT mean4178.624ms/p99 5365.583ms; ITL mean8.541793ms/p99 10.670413ms. Effective decode concurrency averaged27.126 despite client c96. During the steady log window, P queue averaged27.18/23.47 per worker, D running batch16.34/15.50 and D queue0. P full8192-token service log rates were approximately32.05k/32.40k input tok/s. These observations point toward insufficient Prefill supply rather than exhausted Decode capacity. They do not identify a particular GEMM kernel as the cause. The previous c48 baseline was a different allocation, so this is not a controlled single-factor gain/loss estimate.

The archived raw directory `/srv/qwen-artifacts/37872430145/1/` has47 files, all SHA256 verified. Per-commit parser replay reproduced the saved rejection, the official command/export were valid, and runtime cleanup completed. The Actions failure represents the SLA gate, not a build/load/request error.

A second candidate, commit0ef0943b99f7b9a4cf08d33c76adc4d00ffa0db8, keeps Dpool262144/c96 and increases only P chunk8192→16384 (max-prefill-tokens16384 explicitly, same as the existing default). D chunk8192 and P pool131072 remain. [Actions37873721300](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37873721300) uses the same image. This is a targeted follow-up to the observed TTFT failure, not another pool-only sweep. CPU shell, argv, parser, Ruff and full tests passed before push. No additional ASPIRE allocation has been submitted.

### Firmus P chunk follow-up: SLA rejection

Slurm21963 ended FAILED1:0 after9m56s on g1/g4; Actions is terminal failure with successful artifact recovery and runtime cleanup. Server-info verifies P chunk16384, max-prefill16384, P pool131072/max-running96, D pool262144/max-running256/chunk8192, TP4/EP1, FP8 KV and unchanged EAGLE3. Image SHA is unchanged.

2048 completed requests, zero errors,29 good requests (1.416016%): eligible score0. TTFT mean4245.9ms/p99 5527.6ms, ITL mean7.5534ms/p99 10.3864ms, raw input/output65181.9/3270.5 tok/s. AIPerf effective decode concurrency mean24.58. Replaying the run's parser reproduced the saved rejection.

Across377 Prefill log batch samples per worker, new-token means16126.7/16137.2, max16384; queue means19.97/17.29. Logged input rate means43280/43206 tok/s are batch-sample means, not the achieved end-to-end input rate or a time-weighted saturation capacity. Decode batch-sample means15.93/16.82, queue0. These sample statistics must not be equated with AIPerf's request-weighted effective decode concurrency. The larger chunk did not restore SLA; no adoption or further blind concurrency escalation follows.

Raw archive: `/srv/qwen-artifacts/37873721300/1/`; all47 entries in SHA256SUMS verified after terminal state. Theoretical/simulator reassessment and next candidate gates are in [simulation-parallelism.md](2026-10-09-simulation-parallelism.md).
