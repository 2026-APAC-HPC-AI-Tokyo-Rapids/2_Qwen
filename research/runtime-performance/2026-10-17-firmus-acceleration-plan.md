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

### 3.1 handoff の分解結果（router ログと worker カウンタの突き合わせ）

router.log は request_id ごとに開始と終了（latency 付き、1 秒刻み）しか残さず、終了は patch 済み router の `try_join!(prefill body, decode headers)` の完了、すなわち Decode の最初のトークン時点に等しい（client TTFT − 41 ms、±1 ms）。worker カウンタを Little の法則（滞在数 / 34.4 req/s）で時間に直すと、TTFT 2.04 s は次のように埋まる。

| 段 | 滞在数 | 時間 | 備考 |
|---|---:|---:|---|
| Decode: dp-rank 問い合わせ＋prealloc（`#prealloc-req`） | 9.5 | 0.28 s | Prefill 側の登録を待つ |
| Prefill: scheduler の受信待ち（計数なし） | — | 約 0.17 s | 340 ms iteration の半分 |
| Prefill: bootstrap 待ち（`#bootstrap-req`） | 12.0 | 0.35 s | handshake が iteration 途中に届き、次の境界で取り出す |
| Prefill: waiting queue（`#queue-req`） | 6.7 | 0.20 s | |
| Prefill: 起動済み batch 内（`#new-seq`） | 22.1 | 0.64 s | 3986 token の要求が 4096 token chunk を 2 回またぐ |
| Prefill: KV 送信中（`#inflight-req`） | 12.1 | 0.35 s | 実転送は短く、1 iteration ごとの poll で量子化 |
| Decode: running slot 待ち（Decode 行末の `#queue-req`） | 16.8 | 0.49 s | KV 到着済み、上限 32/rank が空くのを待つ |
| router 返却 → client 最初の byte | — | 0.041 s | Linux の delayed-ACK/Nagle と整合する固定値。router の TCP_NODELAY を確認する価値あり |

「ログに現れない約 31 要求」は、起動済み batch 内の 22 と Decode の slot 待ち 16.8 だった。含意は二つ。

- **Firmus も Decode の slot 上限が同時に縛っている。** running は 84% のサンプルで 31〜32 で、KV が届いた要求が rank あたり 4 件 slot 待ちをしている。したがって run A（上限 31、Prefill round_robin）の raw の伸びは Decode 側で頭打ちになり、改善は主に ITL 不合格の解消（約 1.9%）と占有率の向上（29.2 → 30.5/31）で、score の見込みは 6950〜7150 ではなく **約 6850** に下方修正する。Decode の step 時間短縮（Tier 2）なしには raw は伸びない。
- **Prefill は handshake 待ちで約 10% 空転している。** batch 起動の 55% は waiting queue が空で bootstrap 待ちの要求だけがある状態で起き、GPU 稼働は約 90%。bootstrap 待ちは iteration 境界に量子化された 0.35 s で、GPU を使わない。`--optimistic-prefill-attempts`（handshake 完了前に forward を始める）か、handshake を iteration 途中でも取り込む変更で、Prefill 供給 +5〜10% が見込める。安全性（出力不変、patch との整合、失敗時の挙動）は別途ソース確認のうえで判断する。

Decode の上限を 40/rank に上げる案は、ITL の傾き（0.089 ms/要求）から p99 が 23 ms 近くになり不合格が急増するので採らない。

### 3.2 optimistic prefill のソース確認（fc275674 ＋ 本リポジトリの patch）

- `--optimistic-prefill-attempts N` は runtime 引数で、再ビルド不要。到着時に `create_sender` が同期で `/register_dp_rank` を済ませ（`common/conn.py:1028-1066`）、`pop_bootstrapped`（`prefill.py:419-432`）が handshake 未完了でも試行回数 < N なら waiting queue へ入れて同じ iteration で forward する。KV は自分の page に保持し、`pending_bootstrap` のあいだ送信を止める（`prefill.py:740,797,1037,1074`）。handshake が届いた時点で `finalize_bootstrap`（`:322-339`）が decode 側の page を受け取り、まとめて送る。decode の prealloc が先、転送が後という順序は保たれる。
- chunk 境界で handshake 未完了かつ handshake 済みの要求が待っていれば譲り（`:1043-1048`）、試行回数 < N なら waiting queue 先頭へ、達していれば bootstrap queue 末尾へ戻す（`:1241-1271`）。3986 token の要求は 2 chunk なので **N=2** が妥当で、N=1 だと譲った要求が末尾へ回る。
- 失敗時は `handle_bootstrap_failure`（`:965-996`）で KV を解放し HTTP 500。NIXL 側に bootstrap timeout はなく、parked 要求は decode が handshake するか abort するまで KV を持つ。decode の req slot は 49/rank × 4 = 196 ≥ c192 なので、prealloc 待ちで parked が溜まる構造ではない。
- 本リポジトリの patch（dp-rank-query、prefill-ready-handoff、staged-eagle-kv、nixl-expired-chunks、chunk-cache-page-release、kernel 系）はこの経路と衝突しない。staged-eagle-kv が要求する chunk_id の連番は、pending 中に何も送らず retry でも sender を再利用するので保たれる。
- 出力は同じ重み・同じ kernel・同じ KV で意味的に不変だが、batch 構成と chunk 分割位置が変わるので bit 一致は保証されない（round_robin や到着揺らぎと同種）。判定は公式 parser で行う。
- 動作確認の指標: 起動ログの `optimistic_prefill_attempts=2`、Prefill batch 行の `#optimistic-req > 0`、`#bootstrap-req` がほぼ 0、「waiting queue 空かつ bootstrap 待ちあり」の起動割合が 55% から 0 に近づくこと。失敗の兆候: `Prefill bootstrap failed`、`finalize` の AssertionError、NIXL の `has no transfer metadata`、Prefill の `#inflight-req` と token usage が増え続ける。
- bootstrap を iteration 途中で取り込む別の手段はこの版にない（`follow_bootstrap_room` は登録を省くが dispatch が実質ランダムになる）。

run C = run A ＋ `--optimistic-prefill-attempts 2`（Prefill のみ、イメージ不変）とする。

## 4. 施策と期待値

### Tier 1：既存イメージのまま、runtime だけで 1 run

1. **Prefill の DP dispatch を round_robin にする。** ASPIRE で rank 間差 37% → 2% に揃い、Firmus でも DP0 と DP3 の差は 24%。4 rank が同じ step を進む構成では、遊んだ rank の分がそのまま損失で、見込みは Prefill 供給 +5〜9%（34.4 → 36〜37.5 req/s）。Decode 側は `total_requests` のまま。
2. **Decode 上限と ITL の整合。** 供給が増えると Decode は上限 32/rank（N=128）に張り付き、回帰では ITL 平均 17.6、p99 20.5 ms で ITL 不合格が増える。上限 30/rank（N=120）なら ITL は −0.5〜0.7 ms、p99 は 19.6〜19.9 ms で不合格はほぼ消え、Decode 容量は 120 / 3.4 s ≒ 35 req/s。供給 36〜37 req/s にはやや足りないので、Tier 2 の Decode 短縮がない間は上限 31 を基準にし、ITL 不合格と raw の収支で決める。上限は CUDA graph の bucket 列と一致させる必要がある。SGLang は decode batch を `bisect_left` で次の捕捉済み bucket へ padding するので（`model_executor/runner/base_cuda_graph_runner.py`）、現行の `[…, 28, 30, 32]` のまま上限 31 にしても 32 の graph で実行され、GPU の仕事量は 32 と変わらない。6521.98 run でも running 31 のサンプル 142 件は 32 graph で動いていた。上限 31 なら bucket 列の末尾を 32 → 31 に置き換える（graph 1 本分の捕捉時間と数十 MB だけの差）。2 のべき乗であることに依存する箇所は KV pool、req slot、MegaMoE の行数（8 の倍数に丸める）にはない。
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

## 6. 結果（2026-10-17 夜）

### run A：Prefill round_robin ＋ Decode 上限 31/rank（Actions 39019907035、`24fa557`、参照イメージ e29dfde…）

最初の投入（39018316844、`d93ebab`）は Decode の `dist_init_port 33000` が自プロセスの別 socket に取られて起動失敗した。Firmus の一時ポート範囲は 32768–60999 で、33000 だけが範囲内だった。`a047c15` で Decode の dist-init を 29000 へ移し（main）、`24fa557` で再投入した。

| 指標 | 参照 6521.98（38945855851） | **run A** | run B（A ＋ warp-redux-r176、39018872189） |
|---|---:|---:|---:|
| 公式 score | 6521.98 | **6336.27**（−2.8%） | 6105.73（−6.4%） |
| raw tok/s / req/s | 6875 / 34.4 | 6728 / 33.6 | 6558 / 32.8 |
| 良好率 | 94.9% | 94.2% | 93.1% |
| 不合格 TTFT 一斉投入 / TTFT 定常 / ITL 定常 | 114 / 26 / 82 | 108 / 93 / 47 | 111 / 15 / 157 |
| TTFT 平均 / p90 | 2.04 / 2.39 s | 2.20 / 2.68 s | 2.20 / 2.60 s |
| ITL 平均 / p99 | 17.33 / 20.25 ms | 17.09 / 20.04 ms | 17.76 / 20.71 ms |
| Decode running / rank（上限） | 29.2（32） | 28.4（31） | 28.4（31） |
| Prefill DP rank 別 seq 数 | 5.9 / 5.8 / 5.3 / 4.5 seq/s | 1015 / 1043 / 1044 / 1039 | 911 / 948 / 939 / 921 |
| Prefill queue あり割合（DP0） | 78% | 34% | 20% |
| GPU 数値ゲート / 公式 validation | PASS / valid | PASS / valid | PASS / valid |

読み取り:
- **round_robin は意図通り効いた**（rank 間差 24% → 3% 以内）。ただし Decode 律速なので throughput には現れず、Prefill の queue が減っただけ。害はない。
- **上限 31 は逆効果。** Decode slot が 128 → 124 になった分 throughput が −2.3%、待ちが増えて定常 TTFT 不合格が 26 → 93 件。ITL 不合格は 82 → 47 に減ったが補えない。上限 32 に戻す。
- **run B の kernel（warp-redux-r176）は棄却。** A と同じ runtime で同じ batch（N≈120）の ITL が 17.1 → 17.8 ms（+3.8%）、1 step で約 +1.5 ms。棄却済みの 21c2927 と同規模の悪化で、数値ゲートと公式 validation は通るが遅い。`perf/firmus-decode-warp-redux-20261017` は patch を外して記録のみ残す。register-redux も register 常駐を共有するので、komoriyuta と相談なしには投入しない。

### run A と参照の Decode step（ITL × accept）は一致する

run A は N=113.6 で 37.6 ms、参照は N=116.8 で 38.3 ms。傾き 0.197 ms/要求で補正すると一致し、round_robin と上限変更は Decode の step 時間を変えていない。したがって次に効くのは step 時間そのものの短縮だけで、runtime で残る手段は **投機長の短縮**（verify 行数を減らす）である。

### run D の根拠：steps 2 / draft 3

参照 run の step は固定分 13.7 ms（重み読み出し＋draft 3 回など）と要求比例分 0.197 ms × N に分かれ、N≈117 では比例分 23 ms が支配的。比例分は MoE 行数・attention の query 行数・dense GEMM の M に比例し、いずれも要求あたり verify 4 token（draft 3 ＋ bonus 1）で決まる。draft を 3 token（steps 2）にすると行数 −25%、draft forward −1 回で、step は約 36.7 → 31〜32 ms。受理長は 4 draft で 2.21（位置ごとの受理率 ≒ 0.57 の幾何級数）なので 3 draft では約 2.08（−6%）。ITL = step / accept は 16.6 → 15.2 ms（−8%）の見込み。greedy verify は target の出力と一致するので精度は変わらない（campaign summary §12 の steps4/draft5 は中立、steps2/draft3 は未測定と記録済み）。

- 上限は 32/rank に戻し、round_robin は残す（throughput に中立、Prefill 側の待ちは減る）。上限は D の ITL を見てから上げる（ITL p99 が 18 ms 台なら 34/rank）。
- 期待: ITL 不合格 ≒ 0、Decode 容量 128 / (200 × 15.2 ms) ≒ 39 req/s で Prefill 律速（34〜37 req/s）に戻り、raw 7200 前後、score 7000 前後。受理長が 1.9 まで落ちれば ITL は現状並みで中立。
- 判定: ITL 分布、accept len（Decode ログ）、score。不合格なら speculation を 3/4 に戻す。

### run D の投入記録（2 回とも GPU 計測に至らず）

- 39032066280（`019fc60`）: 配備段階で失敗。`scripts/provision-firmus-framework.sh` の EAGLE3 ガードが steps 3 / draft 4 を固定していた。main `0984122` で「draft = steps + 1」の SGLang 側の不変条件に置き換え、同時に見つかった shard 検査の無効化（`.weight_map | values | unique` が object でエラーになりプロセス置換内で握り潰されていた）も修正した。
- 39033000278（`afab218`、Slurm 24313、2 分 51 秒）: Decode worker が起動時に `Qwen SM90 MegaMoE requires FP8 block128 H4096/I1536/top8, DP4/EP4, Prefill or DP-local EAGLE3 Decode, without TBO/EPLB` で停止。`patches/sglang/qwen3-sm90-megamoe.patch` の Decode 側ガードが `speculative_num_draft_tokens == 4` を要求している。つまり steps 2 / draft 3 は現行イメージでは runtime だけでは通らず、少なくとも SGLang patch の変更（＝イメージ再ビルド）が要る。ガードだけの制約か、kernel 側に 4 token 前提（buffer サイズ、graph capture の token 数、tile 選択）があるかを監査中。実 kernel の GPU ゲートは 1〜65 行の任意行数を通しているので、ガードだけである可能性が高いが、確認してから進める。

### run D の結果（Actions 39035989837、`b996afd`、新イメージ＝guard 緩和のみ、header 0e34a083… 不変）

| 指標 | 参照 6521.98 | run A（round_robin、上限31） | **run D（round_robin、steps 2 / draft 3、上限32）** |
|---|---:|---:|---:|
| 公式 score | 6521.98 | 6336.27 | **6320.39**（−3.1%） |
| raw tok/s / req/s | 6875 / 34.4 | 6728 / 33.6 | 6728 / 33.6 |
| 良好率 | 94.9% | 94.2% | 93.9% |
| 不合格 TTFT 一斉投入 / TTFT 定常 / ITL 定常 | 114 / 26 / 82 | 108 / 93 / 47 | 107 / 91 / 58 |
| 定常 TTFT p50 / p99 | 1.94 / 2.68 s | 2.09 / 3.08 s | 1.97 / 3.20 s |
| ITL 平均 / p99 | 17.33 / 20.25 ms | 17.09 / 20.04 ms | 17.64 / 20.08 ms |
| accept len | 2.21 | 2.20 | **2.00** |
| 1 step（ITL × accept）、N≈117 | 38.3 ms | 37.6 ms（N=114） | **35.3 ms** |
| Decode running / rank | 29.2 | 28.4 | 29.3 |

読み取り:
- **投機短縮は不成立。** step は 38.3 → 35.3 ms（−8%）と短くなったが、見込みの −14% に届かず、受理長は 2.21 → 2.00（−9.5%、見込み −6%）と大きく落ちた。ITL は 17.33 → 17.64 ms で悪化。要求比例分 0.197 ms のうち verify 行数に比例するのは半分程度で、残りは要求あたりの固定費（attention の KV 読み出し、scheduler）と見るべき。steps 3 / draft 4 に戻す。guard の緩和（3 か 4 を許可）は無害なので残してよいが、それだけのために再ビルドはしない。
- **round_robin は定常 TTFT の裾を悪化させる。** 2 run とも定常 p99 が 2.68 → 3.1〜3.2 s に伸び、定常 TTFT 不合格が 26 → 91〜93 件。rank 別の queue 平均は揃っているので偏りではなく、到着した rank の瞬間的な待ちを見ずに割り当てることによる分散の増加と見られる。`total_requests` は最も空いた rank を選ぶので裾が短い。throughput は Decode 律速で変わらない。**total_requests に戻す。**
- 参照構成（`a6acf5f` ＋ 上限 32、total_requests、steps 3 / draft 4、c192）は Firmus の runtime 設定として局所最適で、今回試した runtime 変更（上限 31、round_robin、投機短縮）と kernel 候補（warp-redux-r176）はすべて中立か悪化だった。

### 残る runtime 候補（いずれも TTFT 不合格 3.3% の削減が対象）

1. **concurrency を 192 → 176 程度に下げる。** c192 では KV 到着済みで Decode slot 待ちの要求が常時約 17 あり、これは TTFT を 0.49 s 押し上げるだけで throughput に寄与しない（§3.1）。c176 なら定常 TTFT 不合格（26 件）がほぼ消え、一斉投入分も 114 → 約 100。slot の空転が増える分は 1% 未満の見込み。
2. **optimistic prefill（N=2）。** Prefill 側滞在を bootstrap 待ちの約 12 要求分減らし、同じ占有率をより低い c で保てる。一斉投入時の Prefill 処理も速くなる。安全性は §3.2 の通り。
3. router → client の 41 ms（TCP_NODELAY）。

合計で +1.5〜3%（6620〜6720）が上限で、7200 には Decode step の短縮（kernel）が不可欠。Decode MegaMoE は step の 63%、Prefill MegaMoE は FP8 peak の 58% で、komoriyuta の 2 候補は両方とも遅くなった。
