# 2026-10-17 Firmus の Prefill／Decode 同時高速化の方針

対象は Firmus H200×16 の現行最良 6521.98 good output tok/s（Actions 38945855851、`a6acf5f` の kernel chain、P3/D1、c192、Decode 上限 32/rank）。目標は komoriyuta の記録（[2026-10-17-decode-loop-hoist.json](2026-10-17-decode-loop-hoist.json)）と同じ 7200 とし、必要な改善幅は goodput +10.4%。本稿は raw 出力（`/srv/qwen-artifacts/38945855851/1/`、38664373784）の AIPerf 記録、worker ログ、保存済み kernel-summary、[2026-10-16 の nsys 再解析](2026-10-16-megamoe-nsys-reanalysis.md)、komoriyuta の ablation 記録（`research/kernel-ablation/2026-10-17-decode/`）から、段ごとの律速と施策を並べたもの。探索ではなく、根拠のある施策を束ねて少数回で判定する。

## 1. 現状の収支

| 項目 | 値 | 出所 |
|---|---:|---|
| raw output / request throughput | 6875 tok/s / 34.4 req/s | AIPerf |
| 公式 score（goodput × 200） | 6521.98 | score.json |
| SLA 不合格 4279 件中 | TTFT 開始一斉投入 114（2.7%）、ITL 定常 82（1.9%）、TTFT 定常 26（0.6%） | per-request 記録 |
| Prefill 12 rank の供給 | rank あたり 2.9 req/s、DP0 は 78% の秒で queue あり | prefill ログ |
| Prefill DP rank 間の処理率 | 5.9 / 5.8 / 5.3 / 4.5 seq/s | prefill ログ |
| Decode running / rank（上限 32） | 29.2、req slot・KV pool は未飽和 | decode ログ |
| ITL 平均 / 定常 p99 | 17.3 / 20.1〜20.6 ms | AIPerf |
| ITL と全体 running N の関係 | ITL ≒ 6.2 ms + 0.089 ms × N（N=75 で 14.2、126 で 17.4） | 要求ごとの回帰 |

raw 6875 に対して score 6522 なので、SLA 不合格だけで 5.2% を失っている。throughput は Prefill 供給（12 × 2.9 req/s）で決まり、Decode は ITL が 20 ms に貼り付く位置まで埋まっている。二つの段は釣り合っており、片方だけ速くしても score は動かない。

## 2. 段ごとの GPU 時間内訳（kernel-summary、worker-0 の device 0）

### Prefill（4096 token/rank の 1 step ≒ 320 ms、GPU 稼働 96%）

| kernel | 1 layer あたり | 割合 | 備考 |
|---|---:|---:|---|
| MegaMoE `sm90_fp8_mega_moe_impl` | 2.17 ms | 63.5% | 旧イメージ 2.44 ms から −11%。1.24 TFLOP/layer/rank → 約 570 TFLOPS、H200 dense FP8 peak の約 58% |
| FA3 prefill（cutlass device_kernel） | 0.46 ms | 13.7% | 4k causal、FP8 Q/KV |
| dense GEMM `sm90_fp8_gemm_1d2d`×2 | 0.44 ms | 12.8% | QKV / O |
| 融合 QK-norm＋RoPE＋FP8 Q | 0.13 ms | 3.9% | kernel chain で追加 |
| その他（elementwise、quant、router） | 約 0.2 ms | 約 6% | |

MegaMoE は grid 132 / 640 thread / 225.6 KB shared で 1 CTA/SM（再解析 §A）。peak の 58% は融合 kernel 内の all-to-all と同期を含む値なので、残り余地は kernel 内部の待機にある。

### Decode（N=126、1 step ≒ 38 ms、GPU 稼働 86%）

| kernel | 1 step あたり | 割合 | 備考 |
|---|---:|---:|---|
| MegaMoE | 22.7 ms | 63% | 0.2415 ms/call。expert 重み 56.8 GB/rank の読み出し床は 11.8 ms。旧イメージと同値（Decode tile patch は中立） |
| FA3 decode | 4.6 ms | 13% | 49 µs/layer、verify 4 token |
| dense GEMM（16.8＋14.1 µs） | 2.9 ms | 8% | |
| 小 kernel（elementwise、copy、router、splitK reduce） | 約 2.5 ms | 7% | |
| kernel 間の空き | 約 5 ms | 14% | `get_next_batch_to_run` 20 ms のうち GPU と重ならないのは約 2.7 ms。CUDA 欠損警告つきの値 |

ITL の回帰から、step 時間は固定分 13.7 ms（= 6.2 × 2.21）と要求比例分 0.197 ms × N に分かれる。N=126 では比例分 24.8 ms が固定分を上回る。つまり H200 の Decode は重み読み出しではなく、行数に比例する MoE 計算・attention・量子化・verify の費用が支配的で、kernel 側の改善余地は「行あたり費用」にある。

## 3. TTFT と concurrency の構造

閉ループでは Prefill 側に滞在する要求数 = c − Decode running = 192 − 117 = 75。内訳は Prefill rank あたり queue 0.9＋bootstrap 待ち 1.0＋計算中 1.0 の計 35、Decode の prealloc 待ち 9、残り約 31 は router 到着から Prefill 受付、dp-rank 問い合わせ、KV 転送のどこかにあり、worker ログの計数に現れない。この約 31 要求（約 0.9 s）が c を押し上げ、開始時一斉投入の不合格（114 件 ≒ 3 秒 × 34.4 req/s を超えて並んだ分）を増やしている。c を下げると Prefill の queue が切れて供給が落ちるので、c の調整ではなく handoff の遅延を削るのが筋。

## 4. 施策と期待値

### Tier 1：既存イメージのまま、runtime だけで 1 run

1. **Prefill の DP dispatch を round_robin にする。** ASPIRE で rank 間差 37% → 2% に揃い、Firmus でも DP0 と DP3 の差は 24%。4 rank が同じ step を進む構成では、遊んだ rank の分がそのまま損失で、見込みは Prefill 供給 +5〜9%（34.4 → 36〜37.5 req/s）。Decode 側は `total_requests` のまま。
2. **Decode 上限と ITL の整合。** 供給が増えると Decode は上限 32/rank（N=128）に張り付き、回帰では ITL 平均 17.6、p99 20.5 ms で ITL 不合格が増える。上限 30/rank（N=120）なら ITL は −0.5〜0.7 ms、p99 は 19.6〜19.9 ms で不合格はほぼ消え、Decode 容量は 120 / 3.4 s ≒ 35 req/s。供給 36〜37 req/s にはやや足りないので、Tier 2 の Decode 短縮がない間は上限 31 を基準にし、ITL 不合格と raw の収支で決める。
3. **c は 192 のまま。** §3 の通り、下げると Prefill が飢える。handoff が短くなった分だけ下げる。
4. **handoff 遅延の計測（GPU 不要）。** router ログの request_id ごとの時刻と prefill/decode ログを突き合わせ、§3 の約 31 要求がどの状態にいるかを特定する。候補は `firmus-dp-rank-query` patch の問い合わせ（prefill address ごとに 1 件ずつの HTTP）、Decode の prealloc 周期、bootstrap handshake。0.3 s 削るごとに in-flight が約 10 減り、同じ Prefill 飽和を c −10 で保てる。開始一斉投入の不合格は c にほぼ比例する。

期待値：raw 7200〜7400、SLA 損失 5.2% → 3% 前後、score 6950〜7150。

### Tier 2：kernel（新イメージ、komoriyuta の作業と接続）

5. **Decode MegaMoE の行あたり費用。** step の 63%。komoriyuta の `register-redux` と `warp-redux-r176` は CPU コンパイル・SASS 検査済みで GPU 未実行（`results.json`）。21c2927 の失敗は量子化器の barrier と scale cache のどちらが原因か分離できていないので、GPU で測るなら baseline＋候補 1 条件ずつの matched run にする。7200 に必要なのは「同じ batch で Decode cycle −9.4%」または「MoE 単独で −17%」（loop-hoist 記録の計算）。
6. **Prefill MegaMoE の tile/pipeline。** 2.17 ms/layer、peak の 58%、1 CTA/SM。残り余地は kernel 内の待機なので、代表 shape（4096 token × 4 rank、expert 偏りあり）の hardware counter が要る。現行の nsys は driver 13.3 非対応で CUDA イベントが欠け、GPU metrics も取れていない。driver 13.3 対応の Nsight を image に入れ、ncu の単独 replay が分散 persistent kernel で成立するかを先に確認する（再解析の TODO）。
7. **Decode の非 MoE 部分。** FA3 decode 4.6 ms（split-KV 掃引は済みで約 1 ms の余地）、dense GEMM 2.9 ms、小 kernel 2.5 ms、空き 5 ms。空き 5 ms は欠損警告つきの値なので、欠損のない収録を取ってから `get_next_batch_to_run` の Gloo all_gather（12 ms、87% は GPU と重なる）や polling 周期を触る。

Tier 1 と Tier 2 の Decode 短縮 −10% が揃えば、上限 32 のまま ITL p99 が 20 ms を割り、Prefill 供給 37 req/s が全部 goodput になって 7200 に届く計算になる。Prefill kernel の改善はその先の伸びしろ。

### 採らない・後回し

- **P/D 比の変更。** Prefill と Decode が同程度に改善する限り P3/D1 が釣り合う。TP4 単位では P2/D2 しかなく、Prefill 供給が 2/3 になる。
- **投機設定。** steps4/draft5 は中立（campaign summary §12）。topk>1 は verify 行数を増やし、行あたり費用が支配的な現状では不利。
- **c の単独調整。** §3 の理由で Prefill 飽和を崩す。

## 5. 検証計画（Firmus は fair share で 1 run の待ちが数時間）

| run | 変更 | 判定 |
|---|---|---|
| A | round_robin（Prefill）＋ Decode 上限 31、c192、現行イメージ e29dfde… | Prefill の queue 分布が rank 間で揃い供給 ≥ 36 req/s、ITL 不合格 ≤ 30 件、score > 6522 |
| B | A ＋ komoriyuta の Decode 候補 1 条件（新ビルド）、上限 32 | Decode cycle −5% 以上で ITL p99 < 20 ms、score 7000 超 |
| 事前 | router ログからの handoff 遅延分解（GPU 不要） | §3 の約 31 要求の所在を特定し、A の c を決める |

各 run は条件ごとの commit から Actions で投入し、結果の正本は Actions summary と NAS の score.json とする。
