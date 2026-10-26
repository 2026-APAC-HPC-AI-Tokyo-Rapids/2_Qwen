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
