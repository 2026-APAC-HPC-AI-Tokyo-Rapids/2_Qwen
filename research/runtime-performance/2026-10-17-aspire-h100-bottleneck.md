# 2026-10-17 ASPIRE H100 の律速分析と P2/D2 への再配分

Firmus の最速記録 6521.98 good output tok/s（Actions 38945855851、`a6acf5f`、P3/D1、c192、SM90 kernel chain）を ASPIRE 2A+ へ持ち込む前に、ASPIRE の現行最良 3895.05（Actions 38796892855、`6d8d718`、P3/D1、c112）がどこで律速しているかをログから定量化した。結論は「H100 では Decode 1 worker が slot 上限・KV pool・ITL の三つに同時に当たっており、Prefill 12 GPU は半分遊んでいる。kernel chain だけ入れ替えても Decode 側は速くならないので、GPU を Decode へ戻して Prefill 律速にし、Prefill を速くする施策を重ねる」である。探索ではなく、根拠のある変更を一括で入れて少数回で検証する方針に従う。

数値はすべて `/srv/qwen-artifacts/<run>/1/` の raw 出力から集計した。集計スクリプトは本文中の式で再現できる単純なものなので、リポジトリには残さない。

## 1. H100 と H200 の同一構成比較

| 指標 | ASPIRE H100 P3/D1 c112（38796892855） | Firmus H200 P3/D1 c192（38945855851） |
|---|---:|---:|
| 公式 score / raw output tok/s | 3895.05 / 4020 | 6521.98 / 6875 |
| Decode running 平均 / rank 上限 | 17.4 / 20 | 29.2 / 32 |
| running＋転送中（req slot 使用） | 22 が 72%、cap は 20+2+1 | 38〜43、cap は 32+16+1 |
| Decode KV pool 使用率 | 0.85（106496 token/rank） | 0.44（393216 token/rank） |
| Decode pool 確保後の空き VRAM | 3.4 GB（graph 2.31 GB を引くと約 1.2 GB） | 38.4 GB |
| ITL 平均 / p99（SLA 20 ms） | 16.8 / 19.9 ms | 17.3 / 20.3 ms |
| TTFT 平均 / p99（SLA 3 s） | 2.12 / 3.55 s | 2.04 / 5.12 s |
| SLA 不合格の内訳（TTFT / ITL、warmup 含む） | 140 / 52 | 未集計 |
| Prefill rank に queue があった秒の割合（DP0） | 5% | 78% |
| Prefill rank の処理率（queue ありの秒） | 約 4.0〜4.5 seq/s | 約 5.9 seq/s（1 要求 ≒ 2 seq） |

Decode のメモリ内訳（server-info）は weight 61.72 GB、draft 2.23 GB、KV 9.55 GB、CUDA graph 2.31 GB。H100 80 GB では KV pool を増やす余地がない。

## 2. Decode 側の三つの上限

1. **req slot 上限。** SGLang `fc275674` の `pool_configurator._get_num_req_slots` は Decode の slot 数を `max_running_requests + disaggregation_decode_extra_slots + 1` とする。ASPIRE は rank あたり 20+2+1 で、ログの running＋`#transfer-req` は 22 に貼り付いている（491 サンプル中 352）。転送中（Prefill 完了待ち＋KV 受信）の要求が平均 3.9 slot を占めるので、running は 17.4 で止まる。Firmus は extra 16 で slot は縛っていない。
2. **KV pool。** 同じ running 17.4 で token usage 0.85。admission は `available − 512 × active` を要求 token と比べるため、extra slot を増やしても pool が先に詰まる。
3. **ITL。** 要求ごとの ITL を、その要求の生存区間に重なる Decode ログの running 合計に回帰すると、傾きは 0.12 ms / 要求（rank あたり +1 で +0.47 ms）。running 合計 58 の区間では 14.7 ms、72 で 16.8 ms。rank あたり 20 まで埋めると平均 +1 ms、p99 は 21 ms 近辺になり、不合格が増える。H100 の HBM 帯域では expert 重み 56.8 GB/rank の読み出しだけで 1 step 約 17 ms を占め、H200 の約 12 ms より床が高い。

したがって P3/D1 のまま slot や pool を触っても、ITL が上限で得られる改善は数%にとどまる。

## 3. kernel chain は Decode を速くしていない

Firmus の 38664373784（旧イメージ）と 38945855851（kernel chain）を比べると、raw は 6611 → 6875（+4%）だが、Decode running 27.5 → 29.2、ITL 15.75 → 17.33 ms で、1 step あたり（ITL × accept 2.21）は 34.8 → 38.3 ms に伸びている。Decode MegaMoE の平均 kernel 時間は nsys 再解析で 0.242 ms/call（旧）、`2a6a096` の記録で 0.247 ms/call（新）と変わらない。改善は Prefill 側（TTFT p1 1894 → 1278 ms、queue 短縮）から来ており、Firmus は Prefill 律速だった（Prefill DP0 の 78% の秒に queue、Decode slot は未飽和）。

ASPIRE P3/D1 は Decode 律速なので、kernel chain を入れ替えるだけでは効果が出ない。Prefill 律速の配置に変えて初めて、Firmus で確認された Prefill 側の改善が効く。

## 4. P2/D2 への再配分

Prefill 8 GPU（node 0）、Decode 8 GPU（node 1）。worker 内は TP4/DP4/EP4 MegaMoE、EAGLE3、FP8 KV のまま。Decode worker の pool・running 上限・graph 列・memory fraction は 38796892855 と同一で、メモリ前提を変えない。

- Decode は 8 rank × 20 slot に増え、供給 22〜24 req/s では rank あたり約 8 要求、ITL は 14 ms 前後になる（§2 の回帰の下側）。ITL 不合格はほぼ消える。
- Prefill は 8 rank になり、Firmus の飽和実測 2.9 req/s/rank（kernel chain）を H100 の演算性能（FP8 peak は同じ、重み読み出し差は 1 forward 約 6 ms）で割り引くと、worker あたり 11〜12 req/s、合計 22〜24 req/s（raw 4400〜4800 tok/s）が上限になる。
- 以前の P2/D2 MegaMoE 実行 38791490269（c40、旧イメージ、Decode pool 57344）は 2320.67 で完走しており、2 Prefill + 2 Decode の router 経路は動作済み。c40 では TTFT 平均 1.13 s（最小 0.29 s）、ITL 11.4 ms と、どちらも余っていた。

### 同時に入れる Prefill 側の施策

1. **kernel chain イメージ**（SHA-256 `e29dfde4…`、Actions 38939081379）。Firmus の Prefill 律速で +4% raw の実測がある。
2. **Prefill の DP dispatch を round_robin にする。** `total_requests` dispatch では、ASPIRE の Prefill DP0〜DP3 の処理率が 2.52 / 2.32 / 2.19 / 2.02 seq/s、Firmus では 5.9 / 5.8 / 5.3 / 4.5 seq/s と、DP0 に偏る。DP attention と MegaMoE は 4 rank が同じ step を進めるので、空いた rank の分は capacity の損失になる。公式ワークロードは全要求が同じ長さなので、`data_parallel_controller.round_robin_scheduler` の単純な巡回で均等になる。Decode は `firmus-dp-rank-query` patch が Prefill 側 rank を問い合わせるため、Prefill の dispatch 方法に依存しない。Decode は適応的な `total_requests` のまま。
3. **concurrency 88。** 閉ループでは TTFT ≒ (c − Decode 中の要求数) / throughput。Decode 時間約 2.7 s、供給 20〜24 req/s で c=88 なら TTFT は 1.0〜1.7 s に収まり、供給が 24.8 req/s を超えない限り Decode を飢えさせない。c112 のままだと供給 20 req/s のとき TTFT 2.1 s、p99 が 3 s を越える。

### 期待値と判定

- raw 4400〜4800 tok/s、良好率 97% 前後で score 4300〜4650（現行 3895 に対し +10〜19%）。
- 不合格側のリスクは TTFT のみで、Prefill capacity が 20 req/s を下回った場合に p99 が 3 s を越える。ITL 側のリスクはなくなる。
- 1 回の実行で判定する。Prefill 律速なら TTFT と `#queue-req` が立ち、Decode は running 8 前後で推移するはずで、ログでその通りかを確認する。結果の正本は Actions の step summary と保管庫の `score.json`、本文の表は更新しない。

## 5. 採らなかった案

- **P3/D1 のまま extra slot 6・reserve 320。** running は 20 近くまで上がるが §2 の傾きで ITL p99 が 21 ms 付近になり、良好率が 90% を割る可能性がある。max_running 19 に抑えると raw +5% と ITL 不合格 +2〜3% で相殺される。
- **Decode の EP8（node 1 を 1 worker TP8/DP8/EP8）。** 重み読み出しが半減して ITL の床は下がるが、MegaMoE の compiled geometry は `kNumRanks == 4` に特化しており、H100 での実行実績がない。
- **投機長の変更（steps2/draft3）。** step 時間は減るが accept 長 2.21 も下がり、ITL は改善しない。steps4/draft5 は 2026-10-09 に不合格を確認済み。
