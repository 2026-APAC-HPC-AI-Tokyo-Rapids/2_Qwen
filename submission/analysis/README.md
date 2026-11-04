# Profiling data (optional deliverable)

The profile is Actions [41423181527](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41423181527) (PP4 round 24, Slurm 28300): one Nsight Systems 2025.3.2 capture of every Prefill and Decode worker of the 10k configuration, taken in a profile-only pass (no score) on image `c7804efb3180440281a2ec1ebcff19a092786641abbcf6d70b76f9777f565bff`. It is the only full-stack capture of the 10k configuration; no later round improved the score, so it was not repeated for this package.

## Relation to the submitted configuration

The image inputs of round 24 (Qwen `599c093`) and of the submitted image build 41729881283 (Qwen `3f9124f`) differ in two changes: the Decode greedy verify vocabulary fix (`patches/sglang/decode-greedy-verify.patch` and the greedy argmax shape check in `kernels/fa3-verify/wrapper.cpp`) and the round 34 FC1 first-K MegaMoE specialization (`patches/deep-gemm-host/sm90-prefill-fc1-first-k.patch`, with its runtime overlays and `DG_SM90_PREFILL_FC1_FIRST_K=1`). Otherwise the serving configuration differs only in `profile_only`, and the package adds `34-pp4-direct-logprob.patch`, which requests without logprobs do not reach. Therefore:

- **Prefill traces** show the submitted Prefill kernel stack except the entry of each MegaMoE FC1: round 34 peels its first K128 chunk, which its gate measured at about 16–25 µs per call faster with byte-identical outputs. Both runs' Prefill gates selected the same `SGLANG_PREFILL_FA3_VERIFY=n192-direct-o-varlen-pipe+m192n96-pair-prepack-v3`, and the rest of the Prefill settings are the same.
- **Decode trace** predates the working greedy verify fusion. The capture shows the 151936-column FP32 cast, dense penalty and plain argmax that round 25 replaced with the fused `greedy_tiles/greedy_finish` kernels. Everything else in the Decode step is the submitted stack.

The capture is 8 s per worker with a short 20/30/90 s profile client, so it is not an official score. The Slurm job ended FAILED only because its in-allocation validator required every Prefill forward to be a two-request pair; 4–7 one-request forwards beside about 70 pairs per worker failed that check after the reports were written. All four reports were collected, and the validator was relaxed afterwards (Qwen `7757eb8`).

## Findings

The per-stage Prefill numbers below were derived from the reports with `analyze_prefill_profile.py` (in this directory).

- **Prefill** (12 GPUs, PP4 × 3 workers): kernels fill 98–99% of the 112–116 ms two-request pair cycle on every stage, so launch latency does not set the cycle. Per cycle on worker 0 PP0: MegaMoE 63.4 ms, QKV/O GEMM 23.4 ms, FA3 18.9 ms, MoE input 2.9 ms, writer/pack 2.1 ms, norm 1.5 ms. The GPUs sit at 685–691 W with SM clocks of 1410–1455 MHz, 12–15% below the same GPUs in the isolated gate, so the 700 W power cap sets the clock.
- **Decode** (4 GPUs, TP4/DP4/EP4): the EAGLE3 step is 38.5 ms on average (MegaMoE 22.9, attention 7.3, dense GEMM 4.5, draft 1.3, sampling 0.2 ms), with 0.66 ms of GPU idle and no exposed launch gaps.
- **Limit**: Prefill. In the scored round 22 run, completions were 52.41 req/s against 52.40 req/s of Prefill capacity derived from the pair cycle.

## Files

`profiling_data/` holds the Nsight reports (111 MB) and their companion files from that run, and `profiling_data.sha256` lists each with its SHA-256. To verify them, from this directory:

```bash
(cd profiling_data && sha256sum -c ../profiling_data.sha256)
```

| File in `profiling_data/` | Content |
| --- | --- |
| `profile/profile/nsys-prefill-{0,1,2}.nsys-rep`, `nsys-decode-0.nsys-rep` | Nsight Systems reports, one per worker (CUDA and NVTX trace, CUDA graph node trace) |
| `profile/profile/nsys-*.nsys-command.argv`, `nsys-*.argv` | Full nsys command line and the unchanged serving command of each worker |
| `profile/profile/gpu-map-*.csv` | GPU UUID and PCI bus of each worker's CUDA devices, in `CUDA_VISIBLE_DEVICES` order |
| `profile/profile/nsys-*.gpu-metrics.txt`, `nsys-version.txt`, `interpretation.txt` | GPU counter request, Nsight version, capture window |
| `logs/logs/gpu-telemetry-g1.csv`, `gpu-telemetry-g4.csv` | NVML SM clock, power and throttle reasons per node at 100 ms, UTC |
| `logs/logs/profile-capture.log`, `profile-validation.log` | Capture start/stop UTC timestamps and the in-allocation validator output |
| `source/source/capture-firmus-profile.sh` | The capture script the job ran |

To re-derive the Prefill cycle breakdown, export a report with `nsys export --type=sqlite` and pass it to `analyze_prefill_profile.py` together with the matching `gpu-map` CSV, `profile-capture.log` and the telemetry CSVs.
