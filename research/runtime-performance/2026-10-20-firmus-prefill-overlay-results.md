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

## FC2 scatter の非同期 bulk copy 化（kernel bench [39352299232](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39352299232)、0056a46）

FC2 epilogue の行 scatter を、lane ごとの 16 B remote store から、1 行 256 B の `cp.async.bulk`（TMA store 1D、宛先は `sym_buffer.map()`）に置き換え、完了待ちを次 tile の staging 上書き前と combine 前の grid sync の直前に移した案（header のみ、出力は control と同一の見込み、gate 通過）。

| case | control（sf128 + swizzle） | bulk scatter |
| --- | ---: | ---: |
| prefill-balanced-4096 | 1.564 ms | 1.614 ms（+3.2%） |
| prefill-local-4096 | 1.440 ms | 1.371 ms（−4.8%） |
| prefill-skew-4096 | 1.709 ms | 1.764 ms（+3.3%） |

local（remote 転送なし）では発行コストが減って速くなるが、balanced では FC2 scatter（前 tile の copy 待ちを含む）が CTA あたり 170〜176 → 180〜217 µs と縮まず、kernel は遅くなった。FC2 相は NVLink 帯域で頭打ちであることがこれで確定し、不採用。転送量そのものを減らす rank 内事前合算（host 変更を伴う）を進める。

## Combine の rank 内事前合算（serving [39355994503](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39355994503)、82a7f54、branch perf/firmus-prereduce-20261020 は削除）

expert rank 上で同じ token の expert 出力を FP32 で合算し、BF16 1 行にして送る案。remote combine は試算で rank あたり 201 → 91 MB。DeepGEMM host を image 内で再 build し（symmetric buffer に約 1.25 GB/GPU の staging・counter・合算行の領域、`DG_SM90_PRE_REDUCE=0` で無効、Decode worker は 0）、header は overlay で追加した。gate は通過し、公式判定も有効だが **7775.63**（main 8258.34 比 −5.8%）。

| | main 39225913979 | pre-reduce |
| --- | ---: | ---: |
| Prefill MegaMoE / call | 1682 µs | 1904 µs（+13%） |
| TTFT 平均 / p90 | 1659 / 1825 ms | 2030 / 2179 ms |
| ITL 平均 | 14.3 ms | 13.8 ms |
| Decode MegaMoE | 194 µs | 192 µs |

送信量の削減より、local staging の書き込み（約 233 MB）と読み出し（約 147 MB）、行ごとの acq_rel counter、FC2 K loop に入った spill（20/20 → 112/128 B）の増加が大きかった。不採用。host 再 build の配線は d2fd691 と 82a7f54 の履歴に残る。

FC2 の NVLink 律速に対し、header で試せる手（interleave、bulk copy、事前合算）はいずれも負けた。MegaMoE の次の改善は、FC1 K loop（理論 1024 clk に対し約 1330 clk / K block）の効率か、combine を BF16 のまま減らせない以上、通信と計算の重なりを kernel の外（micro-batch など）で作る方向になる。

## Micro-batch による通信と計算の重ね（試算のみ、実装せず）

Prefill の 1 forward を 2 micro-batch に分け、片方の MegaMoE（FC2 の NVLink scatter）ともう片方の attention 等を重ねる案を、main の profile の stage 時間から試算した。kernel 内の固定費（dispatch、grid sync、NVLink barrier）が 2 回分に増え、tile が小さくなって FC1/FC2 の K loop 効率も下がるため、試算は中央値 −1.2%（p10 −3.3%、p90 +0.9%）で、良くなる条件も狭い。No-Go。FC1 が大きく縮んで FC2 通信の比率が上がった場合に再検討する。

## FC1 A operand の 2 CTA cluster TMA multicast（kernel bench [39367977620](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39367977620)、7d7e5d3、branch experiment/firmus-megamoe-fc1-multicast-kb-20261020 は削除）

Prefill の FC1 で、同じ token tile（A）を読む隣接 2 CTA を `__cluster_dims__(2,1,1)` の cluster にし、A を片方の TMA から両方の smem へ multicast する案（header のみ、Decode の SASS は不変、Prefill は 96 regs・spill 20/20 で main と同じ）。L2 からの読み込みは CTA・K block あたり 48 → 40 KiB。gate は通過。

| case | control（sf128 + swizzle） | multicast |
| --- | ---: | ---: |
| prefill-balanced-2048 | 0.8417 ms | 0.8473 ms（+0.7%） |
| prefill-balanced-4096 | 1.5611 ms | 1.5644 ms（+0.2%） |
| prefill-local-4096 | 1.3923 ms | 1.4055 ms（+0.9%） |
| prefill-skew-4096 | 1.7114 ms | 1.7162 ms（+0.3%） |

timeline（prefill-balanced-4096、rank 中央値）では ML1FullWait が 70〜73 → 78〜82 µs、ML1Loop が 679〜696 → 689〜708 µs と逆に増えた。cluster の 2 CTA が smem の空き（empty barrier）を互いに待つようになり、L2 読み込み削減の効果より大きかった。FC1 K loop の遅れは TMA の供給ではなく math 側（WGMMA の smem operand 読み込みと 4 math WG の発行間隔）で決まっていると判断し、不採用。

次の候補は math WG を 2 つ × m64n256 にして B operand の smem 読み込みを減らす再構成だが、math WG 内の `trap`（nvlink barrier の timeout と combine の assert）が setmaxnreg を無効にしているため、その除去と host 側のスレッド数変更（image 再 build か host overlay）が前提になる。

## math 2 warpgroup × 64×256（header のみ、RS m64n128 ×2）（kernel bench [39374402355](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39374402355)、aaedc3e、branch experiment/firmus-megamoe-2wg-kb-20261020 は削除）

FC1 K loop の不足（約 1400 clk / K block、MMA は 1024 clk）が SMEM の operand 読み出し（4 WG × SS m64n128 で K block あたり 96 KiB + TMA 書き込み 48 KiB）によるという仮説を試した。host の 640 thread 起動では kernel 全体の register 上限が 96 で m64n256 の WGMMA が ptxas に拒否される（C7602）ため、math WG 0/1 が setmaxnreg で 200 register を持って 64×256 の accumulator を持ち、A を ldmatrix で register に 1 回読んで RS m64n128 を 2 本発行する形にした（operand 読み出し 80 KiB / K block）。WG 2/3 は GEMM 中 24 register で待機し、combine は 16 warp のまま。gate 通過、Decode の SASS は不変。

| case | control | 2 WG |
| --- | ---: | ---: |
| prefill-balanced-2048 | 0.8403 ms | 0.8395 ms（−0.1%） |
| prefill-balanced-4096 | 1.5795 ms | 1.5615 ms（−1.1%） |
| prefill-local-4096 | 1.3987 ms | 1.3674 ms（−2.2%） |
| prefill-skew-4096 | 1.7091 ms | 1.7035 ms（−0.3%） |

この run の control は前回（39367977620、1.5611 ms）より遅く、clock も 1980 → 1620 MHz と揺れているので、balanced の差は誤差の範囲。timeline（prefill-balanced-4096、rank 中央値）の決定的な結果は、ML1Loop が 679〜716 → 684〜703 µs とほとんど変わらず、clock 差（1.616 → 1.683 GHz）で割り戻すと K block あたりの clk はむしろ約 3% 増えたこと。ML2Loop も clk 換算で同等だった。一方、epilogue は warp が半分になって ML1Epi が 64〜67 → 87〜89 µs、ML2Epi が 26〜28 → 37〜38 µs に伸び、ML1MmaWait は 7 → 18 µs に増えた（kernel は 1552 → 1567 µs）。

SMEM 読み出しを 17% 減らしても K loop は縮まず、**FC1 K loop の不足は SMEM 帯域では説明できない**。残る候補は、K block ごとの `wgmma.wait_group 0` → FP32 rescale の直列部（convoy）と、WGMMA の発行間隔。不採用。

## math 2 warpgroup × 本物の m64n256 SS（host 切り替え、kernel bench [39382077358](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39382077358)、4260933、image 39379223361）

DeepGEMM host（`_C.so`）を v0.1.5 から再 build し、`DG_SM90_PREFILL_2WG=1` のときだけ Qwen Prefill の 128×256 tile を math 256 thread（全体 384 thread、kernel 全体の register 上限 168）で起動する image を作った（913f7bb、run 39379223361、sha256 652aaeef）。header は本物の m64n256k32 SS WGMMA を 2 WG で発行し、待機 WG も register の受け渡しもない（168 regs、spill 0、max R154）。変数なしの image は従来どおりで、image variant の時間は前回までと同等だった。gate は全 case 通過。

| case | control | 2 WG host |
| --- | ---: | ---: |
| prefill-balanced-1024 | 0.4738 ms | 0.4595 ms（−3.0%） |
| prefill-balanced-2048 | 0.8395 ms | 0.8285 ms（−1.3%） |
| prefill-balanced-4096 | 1.5552 ms | 1.5486 ms（−0.4%、候補の clock は 1680 MHz、control は 1890〜1980） |
| prefill-local-4096 | 1.3925 ms | 1.3436 ms（−3.5%） |
| prefill-skew-4096 | 1.7111 ms | 1.6833 ms（−1.6%） |

timeline（prefill-balanced-4096、control は rank 2 だけが遅く（MLastL1End 945 µs）kernel 1639 µs になったので、rank 0/1/3 で比べる。clock は 1.652 / 1.642 GHz でほぼ同じ）:

| 指標（CTA 中央値） | control | 2 WG host |
| --- | ---: | ---: |
| ML1Loop | 674〜677 µs | 619〜631 µs（−7%） |
| ML2Loop | 380〜385 µs | 331〜344 µs（−12%） |
| ML1MmaWait | 7 µs | 68〜74 µs |
| ML1Epi | 63 µs | 97〜101 µs |
| ML2Epi | 25〜26 µs | 38 µs |
| ML2Scatter | 165〜181 µs | 186〜204 µs |
| MLastL2End | 1388〜1401 µs | 1381〜1394 µs |
| kernel | 約 1550 µs（前回の control） | 1520 µs |

本物の SS m64n256 で、初めて K loop が縮んだ（FC1 −7%、FC2 −12%）。header 版の RS m64n128 ×2 で縮まなかったのは、ldmatrix と RS の経路の分だったと考えられる。ただし epilogue・scatter を担当する warp が 16 → 8 に半減した分（ML1Epi +35 µs、ML2Epi +12 µs、ML2Scatter +20 µs）が K loop の利得をほぼ相殺し、kernel 全体は −0.4〜−3.5%（timeline で約 −2%）にとどまる。serving 1 回のばらつき（±3%）では判定できない幅なので、epilogue 側を詰めてから serving で確認する。branch perf/firmus-megamoe-2wg-host-20261020 と image 39379223361 は残す。

## 2 WG の FC1 epilogue 改善と WGMMA 交互発行（kernel bench [39429313746](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39429313746)、da53188、image 39379223361）

候補 `fc1-2wg-host-epi-turn`（2 WG host 版の上で、FC2 は NVLink 律速のため FC1 だけを変更）:
- FC1 epilogue：SwiGLU を分岐なしで計算する（行の有効判定は amax と store だけ）。FP8 staging は warp ごとの 144 B pitch で bank conflict をなくし、16 B の global store で直接書く（WG barrier、TMA store、その完了待ちを廃止）。
- FC2 開始可否の通知：warp 単位の arrival を次 tile の K block 2 で出す。CTA の最後の tile は FC2 block を取る直前に出す。
- FC1 WGMMA：2 WG が named barrier 8/9 で交互に発行する（turn は発行時点で渡す）。

gate 通過。出力 hash は 12 case すべてで control、2 WG host と一致した。

| case | control | 2 WG host | epi-turn |
| --- | ---: | ---: | ---: |
| prefill-balanced-1024 | 0.4803 ms | 0.4671 | 0.4576（−4.7%） |
| prefill-balanced-2048 | 0.8497 | 0.8374 | 0.8221（−3.2%） |
| prefill-balanced-4096 | 1.5666 | 1.5443 | 1.5217（−2.9%） |
| prefill-local-4096 | 1.4331 | 1.3441 | 1.3596（−5.1%） |
| prefill-skew-4096 | 1.7435 | 1.7088 | 1.7038（−2.3%） |

timeline（prefill-balanced-4096、CTA 中央値）:

| 指標 | 2 WG host（1.622 GHz） | epi-turn（1.550 GHz） |
| --- | ---: | ---: |
| ML1Epi | 98〜103 µs | 45〜52 µs |
| ML1Store | 29 µs | 2 µs |
| ML1MmaWait | 67〜69 µs | 15〜17 µs |
| ML1Loop | 631〜652 µs | 697〜797 µs（clk 換算で約 +5%） |

epilogue の改善は狙いどおり効いた（FC1 epilogue + store が約 −80 µs/CTA）。一方、交互発行は K loop を悪化させた。次は交互発行を外した epilogue のみの版を出す。

## epilogue のみの版と、FC2 scatter の dispatch warp への移設（kernel bench [39439372069](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39439372069)、image 39379223361）

`fc1-2wg-host-epi` は epi-turn から交互発行を外したもの。`fc1-2wg-host-epi-scatter` はその上で、FC2 の NVLink scatter を dispatch warp（2 warp）に移したもの（Codex が実装、64 KiB の staging 1 面、full/empty mbarrier、END 通知）。gate はどちらも通過し、出力 hash は 12 case すべてで control と一致した。

| case | control | epi | epi-scatter |
| --- | ---: | ---: | ---: |
| prefill-balanced-1024 | 0.4806 ms | 0.4553（−5.3%） | 0.4638 |
| prefill-balanced-2048 | 0.8564 | 0.8217（−4.1%） | 0.8252 |
| prefill-balanced-4096 | 1.5724 | 1.5157（−3.6%） | 1.5170 |
| prefill-local-4096 | 1.4133 | 1.3095（−7.3%） | 1.3734 |
| prefill-skew-4096 | 1.7384 | 1.6679（−4.1%） | 1.6989 |

timeline（prefill-balanced-4096）では、epi-scatter で math 側の ML2Scatter は 0 になった。しかし ML2Loop が 360 → 510〜535 µs、ML2FullWait が 30 → 140〜200 µs に増え、MLastL2End は epi と同程度だった（kernel 1525 µs 対 epi 1505 µs）。FC2 区間は NVLink 律速のままで、送信を別 warp に移しても math が FC2 stage を待つだけになった。scatter 移設は不採用（branch perf/firmus-megamoe-2wg-scatter-20261020 は削除）。

epi が最良なので、serving で確認する（run 39447708936、perf/firmus-megamoe-2wg-host-20261020、image 39379223361、Prefill の worker にだけ `DG_SM90_PREFILL_2WG=1`）。
