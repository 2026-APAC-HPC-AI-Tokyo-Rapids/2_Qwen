# 2026-10-26 Firmus 9120 構成の再プロファイルと、要求ごとの段階時間

main の Firmus 構成（提出物と同じ 9120.51 構成、image 40197369083）を 2 回測った。実験 branch は `experiment/firmus-profile-9120-20261026`（4a9a953 に main 8c18eab を `--no-ff` で merge）。main 自身の first-parent 履歴は cbca2b3 に届かず、reuse run が古い image を選ぶため、9120 系列を first parent にした。

| run | commit | Slurm | 状態 | score | 追加の計測 |
|---|---|---|---|---:|---|
| [40310899466](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40310899466) | 1739629 | 26380 | FAILED 1:0、17:09 | 9118.01（0.97708、5802 要求） | Nsight。Actions は待ち中に取り消したが job は直前に開始していたため、成果物は手で回収した |
| [40313004834](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40313004834) | e45acd5 | 26385 | TIMEOUT、17:02 | 9129.28（0.97777、5802 要求） | `--enable-request-time-stats-logging`（両 role） |

どちらも 17 分の上限に profile pass の最後で当たった。通常 pass の score は profile pass の前に書かれている。9120.51 / 9132.03 / 9118.01 / 9129.28 の 4 回で、差は 0.15% 以内。正本は `/srv/qwen-artifacts/<run>/1/`。40310899466 は Actions の取り消し時点の途中分に、job 終了後に回収した logs / results / source / profile と slurm-accounting を足し、SHA256SUMS を作り直した。

## 1. Prefill 反復間の空きは消えた（Nsight、prefill-0 の 4 GPU）

| | 40071892077（10-24、async admission 前） | 40310899466 |
|---|---:|---:|
| 1 反復（94 MegaMoE 周期） | 255.0 ms | **249.9 ms** |
| 50 µs 超の空き | 4.8 ms | **0**（最大 0.02 ms） |
| 50 µs 以下の起動間隙 | 2.2 ms（約 860 回） | 2.0 ms |
| GPC clock（反復中の平均） | 中央値 1437 MHz | 1370〜1400 MHz |

[§9 の host 待ち](2026-10-24-broad-ceiling-review.md)は `prefill-async-admission.patch` で全部消えた。空きが消えた分だけ電力が上がり、clock が下がっている。残る 2.0 ms は Prefill が eager 起動のための起動間隙。

反復の最後の約 6.6 ms（2.6%）は、SM Active が 97% から 90% に落ちる。これは空きではなく、EAGLE3 draft extend（1 層の draft を 4032 token 全部に通して draft KV を作る）である。BF16 GEMM が約 3.9 ms（fc、attention 射影、MLP。FLOP からの見積りと一致）、draft FA3 が 0.95 ms。draft の lm_head は最後の token だけで、0.14 ms。

## 2. Decode の内訳（Nsight、decode-0 GPU 0）

1 step 38.2 ms（10-24 の cap 160 で 37.0 ms。cap 176 で rank あたりの要求が増えた）。MegaMoE 21.5 ms（227 µs/回）、FA3 verify 7.0 ms（71 µs/回）、dense GEMM 4.1 ms、kernel 稼働率 98.2%。

## 3. 要求ごとの段階時間（40313004834）

両 role の `ReqTimeStats` を bootstrap_room と entry_time で突き合わせた（9750 組）。

定常（profiling 開始後 20〜100 s、3831 要求、ms）：

| 段階 | p50 | p90 | p99 |
|---|---:|---:|---:|
| Prefill bootstrap | 250.7 | 252.6 | 254.3 |
| Prefill queue | 998.6 | 1005.9 | 1253.5 |
| Prefill forward 開始 → Decode が KV 到着を検知 | 281.7 | 298.9 | 310.9 |
| Decode が KV を検知 → Prefill が要求を解放 | 466.5 | 483.9 | 492.4 |

- **KV 転送そのものは約 30 ms**（1 反復 250 ms を引いた残り。388 MB 前後で 12 GB/s 程度、Decode の検知待ちを含む）。TTFT 1.8 s の中では小さい。
- Prefill 側の `transfer_speed` 1.46 GB/s（= 250 ms）と bootstrap 250 ms は、Prefill scheduler が 1 反復に 1 回しか確認しないことによる量子化である。閉ループでは待ち行列が吸収するので、定常の throughput には効かない。
- Prefill は Decode が KV を受け取った後も約 470 ms（約 2 反復）その KV page を保持している。
- Decode の forward 開始からクライアントの最初の token までは、件数曲線の横ずれで約 0.22 s（p10 0.18、p90 0.24）。`stream_interval` は 1。原因（overlap による出力の 1 step 遅れ、detokenizer、tokenizer manager 1 worker、router の SSE 中継）は未分解。

## 4. profiling 開始時の一斉投入

AIPerf は warmup の要求を全部終えてから、profiling 開始時に 240 要求を同時に出す。9132.03 の run では SLA 違反 129 件のうち 121 件がこの最初の 10 s の TTFT 違反だった。累積件数（開始からの秒）：

| t | Prefill 到着 | bootstrap 完了 | Prefill forward 開始 | Decode KV 検知 | Decode forward 開始 | クライアント最初の token |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 31 | 3 | 2 | 0 | 0 | 0 |
| 0.50 | 169 | 14 | 6 | 5 | 5 | 0 |
| 0.75 | 240 | 73 | 21 | 15 | 15 | 3 |
| 1.00 | 240 | 111 | 42 | 25 | 25 | 3 |
| 1.50 | 240 | 182 | 66 | 49 | 49 | 10 |
| 2.00 | 240 | 240 | 82 | 73 | 73 | 61 |
| 3.00 | 240 | 240 | 121 | 112 | 112 | 109 |

- 遅れの主因は KV 転送ではなく **bootstrap**。240 要求の handshake が 2 s かけて完了し、最初の 0.75 s は 12 rank の Prefill が全力で動けていない（全力なら 0.75 s までに約 36〜48 件が forward を始める）。
- 加えて、Decode の forward 開始からクライアントまでの遅れが立ち上がりで 0.25〜0.5 s ある。
- 両方を解消した場合の上限は、3 s 以内の最初の token が 109 から約 125〜135 件、score で +0.3〜0.5%。

## 5. 以前の EAGLE draft FP8 試験（記録漏れの補記）

2026-09-29 に ASPIRE（c40、P2/D2）で BF16 draft を `--speculative-draft-model-quantization fp8` で読み込んだ（759227b、run [36538766257](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/36538766257)）。score 2468.02 で、BF16 対照 2473.60（run 36390750421）と差がなく不採用。Actions の failure は成果物回収の失敗で、benchmark は完走している。現在の Firmus では、draft extend の BF16 GEMM 約 3.9 ms/反復が対象になる。FP8 化の上限は Prefill −0.8%、電力上限下で実効はその半分程度。受理長への影響は未測定。

## 6. 開始時の TTFT 不合格の原因（40313004834 の再解析）

最初の 240 要求のうち、3 s 以内に最初の token が届いたのは 109。以下の模型（Decode の KV 検知時刻 + 遅れ 0.227 s）はこの 109 を再現する。

**(a) 容量で決まる分（約 108 件、避けられない）。** 240 × 4032 token を 12 rank が 1 反復 250 ms で 1 要求ずつ処理する。全 rank が 0.31 s に最初の KV を出したとしても、3 s 以内に収まるのは rank あたり 10〜11 件（計 120〜132）である。

**(b) 立ち上がりの遅れ（約 11 件）。** 各 rank は最初の 1 件を出した後、0.25 s ごとに規則正しく KV を出している（間隔の中央値 0.25〜0.26 s）。損失は最初の 1 件の遅れに集中している。最初の KV は 0.31 s（P0DP0、P1DP3、P2DP0）、0.55〜0.62 s（7 rank）、0.82 s（P0DP3、P2DP3）。しかも遅れた P0DP3 と P2DP3 は最多の 23 件を受け持っていた。
- Prefill scheduler が新しい要求を受け取るのは、loop 1 周に 1 回だけである。DP attention で 4 rank が lockstep なので、空いている rank も 1 反復 250 ms ごとにしか受け取らない。scheduler への到着（entry_time）は 0.0〜0.05 s、0.15〜0.2 s、0.4〜0.45 s、0.65 s の塊に分かれる。
- 開始直後は front-end（各 server の tokenizer manager 1 worker が 4000 token の prompt を順に処理）が要求を少しずつしか渡せない。最初の反復は、1 rank にしか要求がない状態で始まる。
- tie 回転（`SGLANG_DP_TOTAL_REQUESTS_ROTATE_TIES=1`）は有効だが、最初の塊は 1 rank に 7 件偏った。
- 全 rank が最初の反復から動けば、3 s 以内は 109 から 120 になる。

**(c) router が最初の token を約 0.22 s 止めている（約 12 件、全要求の TTFT にも効く）。** router の request_id は AIPerf の x_request_id と同じで、5802 要求すべてで client TTFT − router latency は 0.6 ms（p90 1.4 ms）である。つまり client の最初の token は、router が応答を返し始めた時刻そのものである。`patches/sglang-router/prefill-load-lifetime.patch` は `tokio::try_join!(prefill_response, decode_response)` の prefill 側で、`process_prefill_response` により **Prefill の応答 body を最後まで読む**。upstream は両方の `send()`（header）だけを join していた。

Prefill の応答が終わるのは、Prefill scheduler が KV 送信の完了に気付いたときである。それは次の loop、つまり 1 反復（250 ms）後になる（Prefill 側の transfer 時間がちょうど 250 ms になる理由と同じ）。Decode は KV 到着の約 30 ms 後に最初の token を出しているのに、router が Prefill の完了まで流さない。差は 0.22 s（定常 p10 0.18、p90 0.24）で、上の模型の遅れと一致する。0.05 s にすれば、3 s 以内は 120 から 132 になる。

**対策の候補。**
1. router：Prefill の load guard を保持したまま Prefill body の消費を別 task に移し、Decode の header が来た時点で stream を返す。Decode が KV 受信前に拒否した場合に Prefill を取り消す処理は残す。全要求の TTFT が約 −0.2 s。router は image 内の binary なので build が要る。
2. Prefill：worker 全体が空いている状態から要求が来たとき、最初の forward を数十 ms だけ待ち、全 DP rank に要求が入ってから始める。定常では待ち行列があるので発動しない。runtime overlay で済む。
3. 1 と 2 を合わせて開始時の不合格を約 23 件減らす。score で約 +0.4%（不合格 131 → 約 108）。残りは (a) の容量の限界である。
