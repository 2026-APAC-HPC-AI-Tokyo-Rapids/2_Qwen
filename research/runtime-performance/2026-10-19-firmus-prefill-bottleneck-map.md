# 2026-10-19 Prefill のボトルネック地図：これまでの探索と、効く書き直しの方向

現行 main（7589c59、7999.37）の throughput は Prefill の供給で決まっている（Decode の平均 running 116 < cap 128、ITL 不合格 0）。8500 には Prefill 1 要求あたりの GPU 時間を約 288 → 271 ms（約 −6%）にする必要がある。本稿は、10 月初めからの Prefill の探索を profile の根拠とともに整理し、いま何が律速で、どの方向の書き直しが効くかをまとめる。数値の出所は各節に示す。

## 1. 構成の変遷と Prefill 効率

| 時期 | Prefill の構成 | GPU あたり持続 prefill | 律速 |
|---|---|---:|---|
| 10/3〜10/13 | TP4（H100）/ TP2（H200）、MoE は TP 内（CUTLASS / Triton） | H100 TP4 約 7.3k、H200 TP2 約 8.8〜10.75k tok/s | TP の all-reduce と MoE の重複処理 |
| 10/15〜 | DP attention（attention TP1）＋ EP4 MegaMoE（dispatch/GEMM/combine を 1 kernel に融合） | 約 13.5k tok/s（7999.37 の run、40.7 req/s × 3986 token ÷ 12 GPU） | MegaMoE の GEMM 効率と GPU の電力制限 |

## 2. 第 1 期（TP 構成）の profile と各施策の結果

profile（[2026-10-09-prefill-supply-rebalance.md](2026-10-09-prefill-supply-rebalance.md) §4、[2026-10-10-dp-attention-deepep-plan.md](2026-10-10-dp-attention-deepep-plan.md) §0）：Prefill の kernel 時間のうち TP all-reduce 23%、MoE 補助（全 65536 route の shuffle/combine/sort）19%、MoE GEMM 28%、FP8 dense MFU 約 16%。

| 施策 | 狙ったボトルネック | 結果 | 理由 |
|---|---|---|---|
| NCCL NVLS（Prefill のみ、[10/4](2026-10-04-prefill-nvls-campaign.md)） | all-reduce の帯域 | +0.26%（単 pair、未確認、不採用） | 主な all-reduce は SGLang の custom all-reduce（multicast 有効）が担っており、NCCL 側の NVLS 選択が効く範囲が小さかった |
| NCCL symmetric memory（Prefill のみ、[10/7](2026-10-07-prefill-nccl-symmetric-memory.md)） | 64 MB 級 all-reduce の遅延と SM 占有 | **+2.74% / +2.90%**（逆順 2 pair で符号一致、採用） | rank0 で 9885 回 `AllReduce [Symmetric]`、64 MB は `RSxLDMC_AGxSTMC`（multimem）。TTFT −9.5%、ITL +2.2% |
| MoE routing 補助の置換（[10/5](2026-10-05-prefill-routing-campaign.md)） | MoE 補助 | helper 3% の範囲で採用 | 全 route 走査を tile histogram + prefix scan に |
| H200 Prefill TP2（P6×TP2 / D1×TP4） | all-reduce 係数と GEMM 幅 | +28%（4429.9、当時最良） | 通信係数 1.5 → 1.0、GEMM 幅が倍 |
| DP8 attention ＋ DeepEP EP8 | TP 分割そのもの | 6.9k tok/s/GPU で TP2 より遅く不採用 | DeepEP normal mode の dispatch/combine と layout 変換の費用 |
| TensorRT-LLM | エンジン差 | 8〜9k tok/s/GPU で同水準 | 分割方式が同じなら効率も同じ |
| reduce-scatter ＋ FP8 all-gather、2 micro-batch 通信 overlap | all-reduce | 未実装 | 構成転換で前提が消えた |

結論：TP 分割そのもの（all-reduce と、各 rank が全 route を扱う MoE）が律速だった。第 2 期の DP attention ＋ MegaMoE はこれを構造的に取り除いた。attention は TP1 で all-reduce が消え、MoE の通信は MegaMoE の kernel 内（NVLink pull と combine）に融合されたので、**NVLS や symmetric memory のような NCCL 側の施策は現構成では作用する対象がない**（7999 の profile の Prefill 上位 kernel に NCCL は現れない）。

## 3. 現構成の profile（7999.37、run 39114949128、prefill-0、1 step ≈ 282 ms、稼働 97.9%）

| kernel | 1 層 | 1 step | 割合 | 性質 |
|---|---:|---:|---:|---|
| MegaMoE（BLOCK 128×256、3 stage） | 1704 µs | 160 ms | 57% | GEMM 区間は compute 律速、FC1 は throttled peak の約 54〜58%、FC2 は約 44% |
| FA3 prefill（FP8 Q、FP8 paged KV、page 64、causal 4k） | 492 µs | 46 ms | 16% | 約 545 TFLOPS 相当 |
| dense GEMM（QKV、O、DeepGEMM FP8） | 228 µs × 2 | 43 ms | 15% | ピークに近い |
| QK-norm＋RoPE＋FP8 Q＋FP8 KV store | 152 µs | 14 ms | 5% | 約 130 MB の読み書きで 0.9 TB/s。帯域なら約 45〜50 µs |
| RMSNorm（31.4 µs × 2）、group quant（19.9 µs × 2）、elementwise 等 | 約 0.14 ms | 13 ms | 5% | 帯域律速の小 kernel |
| EAGLE3 draft extend（BF16 nvjet） | — | 3.7 ms | 1.3% | |

kernel 外の損失：Prefill GPU は電力上限 700 W（仕様上限）で 1.47〜1.67 GHz に落ち、GPU ごとに差がある（[timeline note](2026-10-18-megamoe-kernel-timeline.md) §4.3）。MegaMoE の combine 前で速い rank が遅い rank を最大 119 µs 待つ。DP rank の idle forward が rank-forward の 4.5〜5.2%、そのうち他 rank に待ちがあるのに遊ぶ分が 1.6〜3.1%（[tier1 note](2026-10-18-firmus-runtime-tier1.md)）。開始時一斉投入の TTFT 不合格 84 件（1.7%）。

## 4. ボトルネックごとの原因と、効く書き直しの方向

### 4.1 MegaMoE の GEMM 効率（57%、最大のレバー）

原因（timeline、W5/W2/W3 の解析）：
- 各 k-block で WGMMA の FP32 累積を CUDA core で promote（scale 乗算）する処理が WGMMA と直列になる。累積器が 1 本しかないので次の WGMMA と重ねられない。
- 4 math warpgroup の epilogue が同時に走り、その間 Tensor Core が空く。FC2 は k-block ごとに半分ずつ 2 group で発行する。
- launch が 640 thread なので register の上限が 96。2 本目の累積器（+64 register）を持てない。FMA 化した promote を試すと spill 1.8 KB で全 WGMMA が直列化した。
- loader は空き stage を待ち続けており、データ供給は律速ではない。stage を増やす余地も SMEM にない（4 stage で 275 KB）。

効く方向：**consumer warpgroup の ping-pong 化（CUTLASS の pingpong schedule 相当）と register 予算の再配分**。
- dispatch/epilogue 専用 warp を減らして thread 数を下げ（例：640 → 384〜512）、math warpgroup あたりの register を 168 前後まで上げる。
- 2 本の累積器、または 2 つの math warpgroup が交互に MMA と promote/epilogue を担当する構成にして、promote と epilogue を Tensor Core の稼働と重ねる。DeepGEMM の通常の SM90 FP8 GEMM（1d2d）がピークの 7 割前後を出しているのはこの構造による。
- host library の geometry（thread 数、warp 割り当て）と kernel 本体の両方の書き直しになる。精度は同じ FP8 入力・FP32 累積で、加算順序だけが変わる。
- 見込み：GEMM 区間 1.5 ms の効率を 58% → 70% にできれば MegaMoE −15%、Prefill −8〜9%。これだけで 8500 の条件（−6%）を満たす。規模は最大で、リスクも最大。

### 4.2 電力制限（kernel 外、全 kernel に効く）

原因：700 W で clock が 1.47〜1.67 GHz に落ちる。演算あたりのデータ移動が多いほど電力を食う。
効く方向：設定では変えられない。同じ計算をより少ないメモリ移動で行う変更（融合、再読み込みの削減）は、時間短縮に加えて clock の回復にも効く。4.4、4.5 の融合系はこの意味でも有利。

### 4.3 FA3 prefill（16%）

原因（[FA3 再検討ノート](2026-10-18-firmus-fa3-reconsideration.md)）：FP8 Q と paged FP8 KV（page 64）で causal 4k を回している。runtime knob（num_splits、pack_gqa、事前 scheduler metadata）は改善の根拠が弱い。
効く方向：**no-prefix extend の連続 KV 経路**。request-aligned chunk により Prefill の要求は 1 chunk・prefix なしになったので、paged KV を読み直さず、RoPE kernel が書く FP8 K/V を連続 buffer にも出して FA3 の varlen（非 paged）経路に渡す。精度は同じ FP8。効果は測定でしか決まらない（paged と連続の 1 回の比較で判定できる）。

### 4.4 QK-norm/RoPE/FP8 kernel（5%）

原因：152 µs/層で帯域の 1/4。W4 の token-warp 版は KV store を取り込んだが時間は縮まなかった（141 → 152 µs）。launch 形状と occupancy、head ごとの依存 load の直列が残っている可能性が高い。
効く方向：帯域律速に寄せる書き直し（1 CTA が複数 token × 全 head を vectorized load、cos/sin と norm weight を shared に置く）。−9〜10 ms/step（−3.4%）。4.3 の連続 K/V 出力もこの kernel に足すのが自然。

### 4.5 RMSNorm と group quant（5%）

効く方向：入力 RMSNorm と QKV 用 group quant の融合を Prefill（行数の多いとき）に限定して使う（8016 の run で Prefill −4.9 µs/層、Decode では悪化）。post-attention 側は MegaMoE の pre-dispatch が BF16 を読むので融合対象外。効果は −0.2% 程度と小さい。

### 4.6 MegaMoE の末尾・同期（数%）

原因：combine（86〜88 µs、HBM 律速）、rank 内の遅い CTA 待ち（tail patch で 12〜20 µs に短縮済み）、rank 間の clock 差待ち（最大 119 µs）。
効く方向：flat NVLink barrier を Prefill geometry に限定（8016 の run で Prefill MegaMoE −1.2%、Decode は atomic 集中で +3.3% だったため外す）。combine と最後の FC2 の重ね合わせは hang リスクが高く 1〜2% 程度。

### 4.7 DP rank の idle forward と handoff（供給 1.6〜3.1%）

原因：他 rank に待ちがあるのに遊ぶ forward。tie rotation では解消しなかった。負荷 snapshot が in-flight の bootstrap 要求を数えていない可能性。
効く方向：DP の割り当てに使う負荷を「queue＋bootstrap 中＋計算中」の要求数にする scheduler 側の修正。原因の特定が先。

### 4.8 効かない、または対象外

- NVLS、NCCL symmetric memory、reduce-scatter＋all-gather、micro-batch 通信 overlap：現構成に NCCL all-reduce がない。
- MegaMoE の stage 増、cluster/multicast：SMEM 不足、kernel に実装なし（[geometry note](2026-10-18-megamoe-geometry-and-redundancy.md)）。
- 冗長 expert：静的配置で rank 負荷の max/mean は 1.034 まで均衡済みで、−0.2〜0.3%。
- dense GEMM：ピークに近い。

## 5. 優先順位（8500 = Prefill −6%）

| 順 | 方向 | Prefill の見込み | 規模・リスク | 判定方法 |
|---:|---|---:|---|---|
| 1 | QK-norm/RoPE kernel の帯域化（＋連続 FP8 K/V 出力） | −3.4% | 小〜中 | CPU 検証＋確認 run |
| 2 | FA3 の no-prefix 連続 KV 経路 | 不明（数%〜） | 中 | paged/連続の 1 回比較が必要 |
| 3 | flat barrier と norm＋quant 融合の Prefill 限定 | −0.8% | 小（branch に実装あり） | 確認 run |
| 4 | DP 割り当ての負荷計数の修正 | +1.6〜3%（供給） | 中（原因特定から） | ログ解析 |
| 5 | MegaMoE の ping-pong 化と register 再配分 | −8〜9% | 大 | kernel bench → 確認 run |

1〜4 を合わせて −6% 前後、5 が成功すれば単独で −8〜9%。komoriyuta の [7999 → 8500 ノート](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/blob/perf/firmus-7999-gigatoken-20261018/research/runtime-performance/2026-10-18-firmus-7999-to-8500.md) も 1 と 2 を次の候補に挙げているので、着手前に分担を決める。
