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
- ping-pong が負けた理由（turn を `wait<0>` 後に渡して pipe が空く、SMEM 帯域 125%）は explore 報告と ping-pong ノート（当時の資料 `2026-10-19-megamoe-prefill-pingpong.md`、現行ツリーには非同梱） を参照。

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

## Prefill の batch を 8192 token/rank に倍増（serving [39379601121](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39379601121)、4559d2d、branch experiment/firmus-prefill-chunk8192-20261020 は削除）

`chunked_prefill_size` を 16384 から 32768 にし、MegaMoE の token 上限も 8192/rank にした（f0fed01 の配線）。他は main と同じ。

| | main 39225913979 | chunk 8192 |
| --- | ---: | ---: |
| score | 8258.34 | 8202.54（−0.7%） |
| good requests | 5121 / 5205 | 5088 / 5172 |
| TTFT 平均 / p99 | 1659 / 3856 ms | 1948 / 3954 ms |
| ITL 平均 / p99 | 14.3 / 16.6 ms | 13.0 / 16.0 ms |
| prefill-0 の batch | 4032 token が主 | 8064 token が 1369 回、4032 token が 234 回 |
| prefill-0 の input throughput（batch 行の平均） | 14332 token/s | 14162 token/s |

batch の大部分は 2 要求（8064 token）になったが、Prefill の token あたり処理速度は上がらなかった（−1.2%）。1 forward あたりの固定費（MegaMoE の dispatch と末尾の同期など）は、4k token でもすでに forward 時間に比べて小さく、行数を倍にしても償却の効果が出ない。TTFT が約 0.3 s 延び、score は誤差の範囲で下がったので、不採用。

## 2 math warpgroup ＋ per-warp FC1 epilogue の serving（[39447708936](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39447708936)）：**8356.18**、main に採用

構成は main（8258.34）と同じで、次の 3 点だけが違う。
- image：39379223361（913f7bb の build。v0.1.5 の `_C.so` を host patch `sm90-prefill-2wg.patch` 付きで再 build したもの）
- DeepGEMM overlay：`sm90-prefill-2wg-host-epi.patch` を追加
- Prefill の worker：`DG_SM90_PREFILL_2WG=1`

| | main 39225913979 | 2 WG + epi |
| --- | ---: | ---: |
| score | 8258.34 | **8356.18（+1.2%）** |
| good requests | 5121 / 5205 | 5185 / 5260 |
| TTFT 平均 / p99 | 1659 / 3856 ms | 1562 / 3610 ms |
| ITL 平均 / p99 | 14.3 / 16.6 ms | 14.5 / 16.9 ms |
| prefill-0 の input throughput（batch 行の平均） | 14332 token/s | 14648 token/s（+2.2%） |

kernel bench での MegaMoE −3.6%（balanced-4096）が、Prefill の処理速度 +2.2% と TTFT −6% として現れた。出力は kernel bench で control と bit 一致している。

main に入れる際、image の選択（`scripts/select-qwen-container.sh` は first-parent をたどる）が 913f7bb の build を見つけられるよう、branch 側に main を merge し、main をその merge commit へ fast-forward した。
## 2 WG ＋ hot-4 複製の serving（[39550627795](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39550627795)、fd122bc）：8366.32

kernel bench [39550917343](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39550917343) では、2 WG header の物理 140 gate が PASS し、出力は control と bit 一致した。実測 routing（層 43/62/7）の MegaMoE は、control・現行配置に対して −11.5% / −11.4% / −9.4% で、2 つの利得はほぼ足し算になった。

| | main bc4e5c7（2 WG） | 2 WG ＋ 複製 |
| --- | ---: | ---: |
| score | 8356.18 | 8366.32（+0.1%） |
| TTFT 平均 / p99 | 1562 / 3610 ms | 1551 / 3591 ms |
| prefill-0 input throughput | 14648 token/s | 14808 token/s |
| prefill-0 の `#queue-req` 平均 / 0 の割合 | 1.68 / 24.6% | 1.30 / 41.7% |
| Decode running / rank | 29.9 | 29.8 |

（8258 run は queue 2.88 / 0 の割合 6.2%。）Prefill の速度が上がるほど、Prefill の待ち行列が空になる時間が増えた。score は closed loop の concurrency 192 と、Decode の 1 周期（ITL 14.5 ms × 200 token ≈ 2.9 s）でほぼ決まる。TTFT（約 1.55 s）が少し縮んでも、cycle 全体（約 4.45 s）への効きは小さい。

concurrency はチームが宣言する値（task rules）。Prefill に余力ができたので、次は c208 にして Decode を上限（128 = 32/rank）近くまで埋める。Decode の slot は max_running 128 ＋ extra × DP で concurrency を覆う必要があるので、`decode_extra_slots` を 16 → 20 にする。

## 2 WG ＋ hot-4 複製、concurrency 208（[39622940739](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39622940739)、0037d97）：8352.94

| | c192（39550627795） | c208 |
| --- | ---: | ---: |
| score | 8366.32 | 8352.94 |
| good fraction | 98.54% | 98.30% |
| TTFT 平均 / p99 | 1551 / 3591 ms | 1915 / 3908 ms |
| ITL 平均 | 14.5 ms | 14.5 ms |
| prefill-0 queue 平均 / 0 の割合 | 1.30 / 41.7% | 1.70 / 35.7% |
| Decode running / rank（平均） | 29.8 | 30.0 |

c208 の decode-0 の running は、上限 32 の時間が 60.7%、31 が 23.7% で、Decode はすでに上限で詰まっている。concurrency を増やした 16 要求は Decode の slot 待ちに並ぶだけで、TTFT が約 0.36 s（16 ÷ 44 req/s）延びた。

現構成の上限は、Prefill（3 instance、余力は小さい）と Decode（cap 128 = 32 req/rank × 4 verify token）の両方がほぼ同時に詰まる点にある。Decode cap は DeepGEMM の SwapAB 上限（128 token/rank）で決まっている。次の伸びしろは次の 2 つ。
- Decode の容量を上げる：SwapAB-144 の host patch（d2fd691 の `sm90-swap-ab-144.patch`）を host series に追加して cap 144 にする。ただし、36 req/rank での FA3 decode の悪化（39190318626 で +77%）を同時に解く必要がある。
- Decode step（ITL 14.5 ms）そのものを速くする。

複製は、この状態では score に出ない（Prefill 側の余力を増やすだけ）。branch perf/firmus-2wg-replica4-20261021 に残し、Decode の容量を上げるときに再評価する。

## FA3 decode（TARGET_VERIFY）の 33 要求/rank を超えたときの崖（kernel bench [39692814746](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39692814746)、ec47c09、Codex 調査）

**原因：**
- FA3 は 132 CTA の persistent scheduler で動く（`tile_scheduler.hpp:548-550`）。仕事の単位は「要求 × kv head 4」で、32 要求なら 128、33 要求なら 132 で 1 巡に収まる。34 要求以上は 2 巡目に 4〜12 単位だけが残る。
- `num_splits` は上限の指定にすぎず、prepare の heuristic が実効値を 1 に戻す（`flash_prepare_scheduler.cu:140-158`）。以前の knob sweep が効かなかったのはこのため。

| 要求/rank | default | sched-meta ＋ splits | split を実効値で固定（meta 事前計算）の最良 |
| --- | ---: | ---: | ---: |
| 32 | 50.6 µs | 47.6 | 57.2（split 2） |
| 33 | 50.4 | 48.0 | 58.5（split 2） |
| 34 | 89.1 | 84.2 | 73.8（split 3） |
| 35 | 90.2 | 85.2 | 74.2（split 3） |
| 36 | 90.1 | 85.2 | 75.4（split 3） |

split を固定すると 36 要求で −16% になるが、理想（約 57 µs）には遠く、split の combine の費用が大きい。

Decode の step（約 32 ms ＝ ITL 14.5 ms × accept 2.2）の内訳の目安は、MegaMoE 18 ms（token 数にほぼ依存しない）と FA3 4.5 ms。そのうえで：
- 36 要求/rank（split 3）：FA3 が約 +2.6 ms で step 約 +9%、要求 +12.5%。差し引き +3% 前後。
- 33 要求/rank：FA3 の追加は 0 で step はほぼ不変、要求 +3.1%。差し引き +3% 前後。

**Decode の上限は 132（33 要求/rank）を先に試す。** SwapAB-144 の image で 132 token が SwapAB に入る。FA3 の変更は不要。

## Decode 上限 132（33 要求/rank）＋ SwapAB-144 image ＋ 2 WG ＋ hot-4 複製（serving [39699383079](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39699383079)、531c78a、image 39693217189）：8155.07、不採用

| | main（2 WG、cap 128、c192） | 2 WG ＋ 複製、c192 | cap 132、c196 |
| --- | ---: | ---: | ---: |
| score | 8356.18 | 8366.32 | **8155.07** |
| ITL 平均 / p99 | 14.5 / 16.9 ms | 14.5 / 16.9 ms | 15.3 / 17.9 ms |
| Decode running / rank | 29.9 | 29.8 | 30.7（33 が 61%） |
| accept 長 | 2.2 | 2.2 | 2.21 |
| prefill queue 0 の割合 | 24.6% | 41.7% | 52.2% |

Decode の running は +3% になったが、ITL が +5.8% 延びて score は −2.4%。decode-0 の profile（kernel-summary の平均時間、cap 128 の run 39550627795 との比較）：
- DeepGEMM の dense FP8 GEMM（`sm90_fp8_gemm_1d2d`、QKV と O）：15.8 → **21.6 µs（+37%）**。M = 33 × 4 = 132 行が BLOCK_M 128 の 1 tile を超え、2 tile 目が 4 行だけになる。
- Decode MegaMoE（SwapAB）：199.2 → 207.0 µs（+3.9%）。
- FA3 verify（cutlass device_kernel）：50.8 → 50.8 µs（33 要求までは 1 巡、予想どおり）。
- 1 step あたり dense GEMM で約 +1.1 ms、MegaMoE で約 +0.7 ms、計約 +1.8 ms（+6%）となり、ITL の増加と合う。

Decode の 128 行（32 要求 × 4 検証 token）は、dense GEMM・MegaMoE SwapAB・FA3 の 3 つで同時に tile 境界になっていて、上限を少し上げるだけでは step が割高になる。Decode の容量を増やすには、128 行を超える範囲で dense GEMM の tile の選び方（BLOCK_M、SwapAB）を直すか、Decode step そのものを速くする必要がある。branch perf/firmus-swapab144-20261022 と image 39693217189 は残す。

## Decode step の削減 B1・B4・B6（B5 なし）（serving [39764967207](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39764967207)、a9a6ba6、image 39711569005）：**8481.99**、採用

B1・B4・B5・B6 をまとめた run 39759800897（9f93944）は 7582.76 だった。B5（`DG_SM90_1D2D_SMALL_M=1`）を有効にすると、Decode の dense FP8 GEMM の 128 行に BLOCK_M 256 / BLOCK_N 32（QKV）・80（O）が選ばれ、1 呼び出しが 17.3 → 44.3 µs、14.3 → 30.5 µs に延びた（約 +4.1 ms/step）。そこで、Decode にこの環境変数を渡すのをやめた。B5 の host patch（`sm90-1d2d-small-m.patch`）は image に入ったまま opt-in で、変数がなければ元の heuristic になる。それ以外の条件は 9f93944 と同じ（hot-4 複製、cap 128、c192、EAGLE3 3/4）。

| | main（2 WG） | 2 WG ＋ 複製（39550627795） | B1/B4/B5/B6（39759800897） | **B1/B4/B6（39764967207）** |
| --- | ---: | ---: | ---: | ---: |
| score | 8356.18 | 8366.32 | 7582.76 | **8481.99** |
| TTFT 平均 / p99 | — | 1551 / 3591 ms | 1700 / 3663 ms | 1539 / 3407 ms |
| ITL 平均 / p99 | 14.5 / 16.9 ms | 14.54 / 16.91 ms | 16.06 / 18.66 ms | **14.33 / 16.74 ms** |

decode-0 の kernel-summary を、MegaMoE の 1 呼び出し（＝ 1 層）あたりに直した比較（39550627795 → 39764967207）：

| kernel | 前 | 後 | 担当 |
| --- | ---: | ---: | --- |
| `flash::prepare_varlen_num_blocks_kernel` | 3.65 µs（1.03 回/層） | 0.30 µs（0.09 回/層） | B4（scheduler metadata を step で共有） |
| FA3 verify（cutlass device_kernel） | 52.4 | 51.7 | B4 |
| 名前なし kernel（QK-norm/RoPE など） | 14.4 ＋ 1.3 | 13.0 ＋ 0 | B6 |
| `_router_triton_kernel` | 3.04 | 3.53 | B6（router 後処理の融合で増えた） |
| MegaMoE | 199.2 | 200.8 | B1（39759800897 では 198.4。差は run の揺れの範囲で、効果は確認できない） |
| dense FP8 GEMM（QKV ＋ O） | 31.5 | 31.9 | B5 なし（元の config） |
| 合計（上位 kernel） | 338.4 | 334.5 | 約 −0.37 ms/step（31.81 → 31.44 ms） |

score は main 比 +1.5%、同じ構成の参照（39550627795）比 +1.4%。step の短縮（−1.2%）に、ITL p99 と TTFT p99 の改善が加わった。B5 は Decode の M=128 では逆効果なので使わない。B1 は効果が見えないが、Decode の SwapAB 経路にだけ効く overlay で害もないため、まとめて残す。

## Decode round 2：job 内の比較計測（serving [39881882325](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39881882325)、bb91047、image 39875650446）：8501.98、どの候補も選ばれず

branch perf/firmus-decode-r2-20261023。build（5451914、run 39875650446）で DeepGEMM host に Decode 用の dense GEMM config 指定（`DG_SM90_1D2D_DECODE_CONFIG`）と MegaMoE の L2 先読み stage 数（`DG_SM90_DECODE_L2_PREFETCH_STAGES`）を足した。serving の job の中で、モデル起動前に 4 GPU で候補と現行を交互に測り、明確に速いものだけを Decode に渡す形にした（17 分の上限に収めるため、gate と Prefill の起動に重ねた。profile なし）。結果は `results/decode-selection/`。

| 候補 | 結果 | 判断 |
| --- | --- | --- |
| router の gate GEMM を Triton 1 kernel に（cuBLAS split-K 6.9 µs/層の置き換え） | 94 層で **+314〜318 µs 遅い**（全 4 GPU、2 ブロックとも）。数値検査は通過 | 不採用 |
| MegaMoE の L2 先読み（最初の FC1 block、K8 以降 8/16 stage） | cold-L2、balanced と実 routing 代用 3 層で、8 stage は **+1.8〜3.7 µs/層 遅い**、16 stage は +2.7〜7.6 µs 遅い。無効時 203.7 µs | 不採用 |
| dense GEMM（DeepGEMM 8 config、Triton 2、FlashInfer 1） | DeepGEMM と Triton の全候補は数値検査を通過。FlashInfer 候補が TensorRT-LLM の JIT cache（`~/.tensorrt_llm`、BeeGFS 上）の rename に失敗し、selector 全体が停止して**計測なし** | 未判定 |

score は 8501.98（ITL 14.35 / 16.72 ms、TTFT p99 3411 ms）。どの候補も選ばれていないので、構成は main（3642c64）と同じであり、8481.99 との差は run 間の揺れ（+0.2%）として扱う。READY は 437 s（前回 439 s）で、選択の重ね合わせは起動を遅らせなかった。

L2 先読みは、token の dispatch を待つ間の HBM の空きを使う狙いだったが、実際には遅くなった。先読みが dispatch の通信や最初の stage の通常 load と競合しているか、L2 に残らないと見られる（未確認）。Decode MegaMoE の残り（約 200 µs/層、下限 約 126 µs）を縮めるには、別の手が要る。dense GEMM は FlashInfer 候補を外し、候補ごとの失敗で selector を止めない形にすれば、次の serving で同じ仕組みのまま測れる。

## FA3 verify の候補と dense GEMM の測り直し（serving [39901974117](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39901974117)、b60f2e3、image 39899235870）：**8576.66**、採用

branch perf/firmus-decode-fa3v-20261023。build（89936a3、run 39899235870）で、固定 source（sgl-attn f89bc23、CUTLASS 57e3cfb）から TARGET_VERIFY 専用の FA3 の specialization だけを集めた companion `/opt/apac/fa3-verify`（2.4 MB）を image に足した。既存の `flash_ops.abi3.so` は変えていない。serving の job の中で、モデル起動前に候補と今の kernel を交互に測った（`results/decode-selection/`）。FA3 は HEAD ノードの空き 4 GPU で 21 秒、dense は Decode の 4 GPU で 48 秒。

FA3 verify（1 層、94 層 graph、4 GPU の中央値、b32 で選択）：

| 要求/rank | 今の FA3（B4 あり） | p128s2（N128、non-TMA、2 stage） | t64s3（N64、paged TMA、3 stage） |
| --- | ---: | ---: | ---: |
| 32 | 49.5 µs | **46.1** | 53.2 |
| 33 | 50.4 | 47.0 | 53.6 |
| 34 | 87.6 | 80.8 | 88.4 |
| 36 | 88.6 | 79.9 | 89.3 |

- p128s2 が全 GPU・全ブロックで 3.2〜3.7 µs/層 速く、選ばれた（`SGLANG_DECODE_FA3_VERIFY=p128s2`）。
- paged TMA を使う N64（t64s2 54.4、t64s3 53.2）は遅かった。page のずれではなく、N block の数（44 → 66）が増えた分の反復の費用が上回った。
- split 2/3 ＋ 専用 combine は数値検査に落ちた（t64split2 は出力の不一致、t64split3 は空 split の出力）。採用候補から外れ、原因は未調査。
- 同じ設定を companion で compile し直した control96 は、wheel の FA3 より 1.3 µs 遅い（50.6 µs）。toolchain の差と見られる。
- 34〜36 要求の 2 巡目の段差は残る（p128s2 でも 80 µs 前後）。上限を 128 より上げる根拠にはならない。

dense GEMM（1 呼び出し、M=128）：

| 候補 | QKV（N9216/K4096） | O（N4096/K8192） |
| --- | ---: | ---: |
| stock（BM64） | 17.0 µs | 15.2 µs |
| BM128/BN80/7 stage | **15.2**（選択） | — |
| BM128/BN32/10 stage | — | 15.5 |
| BM64 で stage だけ変更 | 16.5 | 15.1（差 0.1、規則未満で stock のまま） |
| TMA multicast あり | 15.5〜20.2 | 16.2〜19.6 |
| Triton block FP8 | 30.8〜31.6 | 50.3〜50.5 |

選ばれたのは QKV だけ（`DG_SM90_1D2D_DECODE_CONFIG=QKV:128,80,7,1,0`、1.8 µs/呼び出し短縮）。

合計の見込みは、FA3 が約 −0.32 ms/step、QKV が約 −0.17 ms/step。

| | main（39764967207） | 再測定（39881882325） | **今回（39901974117）** |
| --- | ---: | ---: | ---: |
| score | 8481.99 | 8501.98 | **8576.66** |
| ITL 平均 / p99 | 14.33 / 16.74 ms | 14.35 / 16.72 ms | **14.20 / 16.48 ms** |
| TTFT p99 | 3407 ms | 3411 ms | 3410 ms |

同じ構成の 2 run の差は 0.24% で、今回は両方の平均（8492）より +1.0%。kernel の短縮の見込み（約 −0.49 ms/step、+1.6%）よりやや小さいが、方向は合う。job は 15:44 で、17 分の上限内に収まった。
