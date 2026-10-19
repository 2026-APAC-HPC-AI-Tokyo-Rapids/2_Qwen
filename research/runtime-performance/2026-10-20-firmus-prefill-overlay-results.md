# 2026-10-20 Firmus Prefill：overlay serving の結果（8258.34）

## 結論

main の image（317ab7c7、2d294f0 の build。SGLang と DeepGEMM の chain は 7589c59 の a8ed4592 と同じで、gigatoken を追加。Firmus では未使用）に、build を経ずに次を overlay した構成が **8258.34**（main 7999.37 比 +3.2%）。

- SGLang（`sglang_overlay_patches`）：`add-rmsnorm-group-quant`（Prefill EXTEND 2048 行以上に限定）と `prefill-attention-contiguous-kv`（vectorized prologue と、prefix のない extend 用の連続 KV FA3 varlen 経路）
- DeepGEMM header（`deep_gemm_overlay_patches`）：FC2 activation scale の per-128 化（利用者承認済み。checkpoint は `weight_block_size [128,128]`・`activation_scheme dynamic` で、活性値の粒度を定めていない）と、FC2 staging の XOR swizzle

## serving の比較（いずれも c192、Decode 上限 128、profile 付き、単回）

| run | 構成 | score | TTFT 平均 / p90 | ITL 平均 | Decode 実行中 / rank |
| --- | --- | ---: | --- | ---: | ---: |
| [39114949128](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39114949128) | main（a8ed4592） | 7999.37 | 1868 / 2005 ms | 13.9 ms | 28.6 |
| [39198642036](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39198642036) | Prefill bundle の build（MegaMoE ping-pong を含む） | 7493.00 | 2224 / 2416 ms | 13.6 ms | 24.8 |
| [39210912916](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39210912916) | 317ab7c7 + SGLang overlay | 8164.84 | 1737 / 1910 ms | 14.1 ms | 29.1 |
| [39225913979](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39225913979) | 上記 + DeepGEMM FC2 per-128 + swizzle | **8258.34** | 1659 / 1825 ms | 14.3 ms | 29.5 |

Prefill の kernel（prefill-0、1 回あたり）：MegaMoE 1704（main）→ 1728（SGLang overlay）→ 1682 µs（per-128 + swizzle）、FA3 Prefill 492 → 420 µs、QK-norm/RoPE prologue 152 → 134 µs、入力 norm + quant 31.4 + 19.9 → 47.0 µs。Decode MegaMoE は 194 µs で不変。

## kernel bench の要点（1 node、4 GPU、a8ed4592 の header に overlay）

- [39206818509](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39206818509)：prologue の head row を position・cos/sin より先に load する案は byte 一致だが 4k 行で 102.9 → 111.2 µs。不採用。
- [39213738748](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39213738748)：4 math WG の WGMMA 発行 rotation（mma-order）は prefill-balanced-4096 で +6.0%。FC2 の MMA drain wait を 60 → 12 µs/CTA に減らしても K loop は縮まず、4 WG の convoy は律速でない。Prefill split timeline（per CTA）では FC1 K loop 686 µs（約 1330 clk/K block、MMA 理論 1024）、FC2 K loop 510 µs（約 2000 clk/K block）で、FC2 が遅い相。
- [39217809231](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39217809231)：FC2 per-128 で FC2 K loop 485 → 342 µs/CTA、drain wait 59 → 8 µs。ただし FC2 の NVLink scatter が 79 → 160 µs 前後に伸び、balanced では −0.6〜−1.0%、全 expert local の場合は −4〜−5%。FC2 相は combine の BF16 転送（rank あたり約 200 MB）で頭打ちになる。FC1 TMA store drain の繰り延べは 24 → 22 µs で効果が小さく不採用。

## 次の方向

- MegaMoE FC2 相は NVLink 律速に移った。同一 token の rank 内 expert 出力を FP32 で先に合算して BF16 1 本で送る（remote 転送の約半減）、または combine 送信を FC1 相にも分散する。
- ping-pong が負けた理由（turn を `wait<0>` 後に渡して pipe が空く、SMEM 帯域 125%）は explore 報告と [ping-pong ノート](2026-10-19-megamoe-prefill-pingpong.md) を参照。

## FC2 interleave（kernel bench [39332619830](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39332619830)、7a179ad）

FC2 tile を ready になり次第 FC1 と交互に流し、FC1 を rank 共有 counter で動的に配る案（header のみ、出力は control と同一の見込み）。gate は通過したが prefill-balanced-4096 で control（sf128 + swizzle）1.575 ms に対し 1.701 ms（+8%）、skew-4096 は +8.5%。不採用。

timeline（per CTA、rank 1）：FC1 K loop 686 → 804 µs、FC1 store 27 → 34 µs、FC2 scatter 170 → 204 µs。FC1 と FC2 の重みが同時に L2 を使うことと、FC1 が scatter と競合することで FC1 が遅くなり、scatter 自体も短くならなかった。scatter 時間は FC2 相に集中しているかどうかにかかわらず CTA あたり 160〜200 µs で、帯域より math warp による 1 行ずつの remote store 発行が律速の可能性がある。次の候補は scatter の非同期 bulk copy 化（math warp を解放）。rank 内 expert 出力の事前合算は symmetric buffer の追加（host 変更と image 再 build）が必要。

## EAGLE3 steps 2 / draft 3（serving [39336644243](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39336644243)、ba7b370）

main（8258.34）の構成から EAGLE3 を steps 3 / draft 4 → 2 / 3 に変えただけの run。image の MegaMoE decode guard が draft 4 固定だったため（1 回目 [39333825784](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39333825784) は Decode 起動時に停止）、guard を `in (3, 4)` にする 1 行（2026-10-17 の b996afd と同じ）を SGLang overlay に追加した。

| | main 39225913979 | steps 2 / draft 3 |
| --- | ---: | ---: |
| score | 8258.34 | 8238.82 |
| accept 長（bonus 込み） | 2.21 | 2.01 |
| ITL 平均 | 14.3 ms | 14.6 ms |
| TTFT 平均 / p90 | 1659 / 1825 ms | 1608 / 1792 ms |
| Decode 実行中 / rank | 29.5 | 29.7 |

Decode step は約 31.6 → 29.3 ms（ITL × accept）と約 7% 短いが、accept 長の低下でほぼ相殺し、Prefill 律速のため score は同等（−0.2%、run 間のばらつき ±3% の範囲）。不採用で steps 3 / draft 4 のまま。
