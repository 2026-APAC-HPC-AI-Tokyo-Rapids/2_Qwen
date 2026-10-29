# 2026-10-29 PP4 Prefill の profile 解析：遅さは pipeline の結合にあり、kernel は EP4 より速い

komoriyuta の PP4 試行（branch `perf/prefill-pp4-local-moe-20261029`、Prefill を 3 本の PP4/TP1/EP1 pipeline、層 24/24/23/23、microbatch 2 要求、`pp-async-batch-depth 1`）は、serving [40767107134](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40767107134) で供給 36.7 req/s（EP4 の 47.8 の 77%）、TTFT p50 3.4 s、score 0 だった。profile 専用 run [40796263174](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40796263174)（Slurm 27186、nsys 4 本 + torch CPU trace 16 本、`/srv/qwen-artifacts/40796263174/1/profile/`）の prefill-0 を stage ごとに分けて読んだ。解析 script は作業用（nsys の sqlite export を kernel 種別で union、GPU metrics、CPU trace の record_function 集計）。

## 1. 結論

- **kernel 単体は EP4 より速い**（同じ約 1.5 GHz）。stage 1 つの 1 microbatch（2 要求、8064 token、23〜24 層）の計算は **109〜113 ms**。EP4 の 2 要求相当 125.5 ms より **10% 短い**。
- **遅いのは pipeline の結合**。stage の周期は 164 ms で、stage 0〜2 は 1 microbatch あたり **52〜56 ms を NCCL の SendRecv kernel の中で待つ**（計算と重ならない）。stage 3（最終）は待ちがない代わりに、**microbatch ごとに MegaMoE 1 回が 37〜66 ms 止まる**。
- MegaMoE が止まる原因は、同じ GPU に常駐した NCCL の SendRecv kernel。MegaMoE は SM 数ぶんの CTA を同時常駐させて grid 同期する persistent kernel で、NCCL の CTA が SM を占めると残りの CTA が載らず、常駐済みの CTA が barrier で回り続ける（停止中 SMs Active 100%、SM Issue 1.6〜2.5%、Tensor Active 1〜3%。通常の MegaMoE は Issue 39%、Tensor 60%）。
- 結合を外して層を均せば周期は **約 113〜115 ms** になり、Prefill 供給は 3 pipeline × 2 / 0.114 ≈ **52.6 req/s（EP4 比 +10%）** と見込める。10k 計画の Prefill 側の要件（−7〜8%）を満たす。Decode は変わらないので、score は Decode 容量の上限（約 9,500〜9,600）で止まる。Decode −3.6〜5% は別途要る。

## 2. 測定（prefill-0、capture 約 5.5 s、stage あたり 34 microbatch、GPC clock 1.50〜1.52 GHz）

stage ごとの 1 microbatch（中央値、ms）。「NCCL のみ」は SendRecv kernel が動き計算 kernel が無い時間。

| stage | 周期 | 計算 kernel の union | NCCL のみ | 空き | MegaMoE | FA3 | QKV+O |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0（24 層） | 164.2 | 113.5 | 51.8 | 0.5 | 64.7 | 19.8 | 21.3 |
| 1（24 層） | 165.9 | 112.5 | 53.2 | 0.5 | 64.1 | 19.7 | 21.1 |
| 2（23 層） | 165.1 | 108.7 | 55.7 | 0.5 | 62.1 | 18.9 | 20.5 |
| 3（23 層 + lm_head + draft） | 164.1 | 163.4 | 0.3 | 0.5 | **109.4**（うち停止 37〜66） | 20.1 | 20.5 |

- stage 3 の MegaMoE は 23 回のうち 22 回が 2.5〜2.8 ms で、1 回だけが 37〜66 ms。停止する層は microbatch 開始から 5〜8 ms 後に始まる NCCL SendRecv（別 stream、次の microbatch の hidden の受信）とちょうど重なり、長さも一致する（例：NCCL 42.6 / 58.7 / 37.0 ms に対し MoE 45.2 / 61.5 / 36.5 ms）。capture 中の停止は 31 回、合計 1.74 s（capture の 32%）。
- stage 3 の lm_head の後、次の microbatch までは 9.2 ms（draft の FP8 GEMM 3.2、lm_head 1.2、draft FA3 1.0、1d2d GEMM 0.9 ほか）。
- stage 0〜2 の NCCL 待ちは計算の末尾（約 110 ms）に始まる。stage 3 の NCCL は計算の冒頭に始まり計算と重なる。

CPU（scheduler thread、record_function）：

| 区間 | PP0 | PP2 | PP3 |
|---|---:|---:|---:|
| `run_batch`（eager の起動、23〜24 層） | 25.0 ms | 24.8 | 27.8 |
| `pp.wait_result`（`d2h_event.synchronize()`） | **127.3** | 77.8 | **126.0** |
| `recv_res_dict_from_prev_stage` | 0.6 | 0.6 | 0.6 |
| その他（prepare、bootstrap、transfer、release、mlp_sync） | 各 0.1〜1.8 | 同 | 同 |

scheduler は 1 周期の約 77% を `pp.wait_result` で止まっている。この待ちは upstream の `event_loop_pp_disagg_prefill` の構造による：各 rank は microbatch i の forward を起動する前に、出力の ring（最終 rank → rank 0 → 1 → 2 → 3 の順に 1 周期に 1 hop）から microbatch i+1 の出力 dict を受け取り、D2H の完了を待つ。出力は最終 rank の forward が終わるまで出ないので、**全 rank の起動が最終 rank の forward 完了に揃い、pipeline に流れる microbatch が実質 1 つになる**。komoriyuta の patch は EAGLE の relay と計測を足しているが、この loop の順序は upstream のまま。

kernel の効率（同じ clock、1 行あたり）：

| kernel | PP4（EP1、8064 行） | EP4（40310899466、3994 行 × 2 換算） | 比 |
|---|---:|---:|---:|
| MegaMoE | 2.65 ms（0.329 µs/行） | 1.513 × 2 = 3.03（0.379 µs/行） | **−13%** |
| FA3 target | 0.819 | 0.436 × 2 = 0.872 | −6% |
| QKV + O | 0.473 + 0.411 = 0.884 | (0.251 + 0.219) × 2 = 0.940 | −6% |
| 1 stage の計算（2 要求） | 109〜113 | 125.5 | **−10%** |

MegaMoE の −13% は、FC2 出力の NVLink 転送と rank 待ちが無いぶん（EP4 の §2.1 で 87〜107 + 12〜75 µs/層）と、M が 2 倍で wave の端数が減るぶん。combine（78〜81 µs/層）はまだ残っていると見られる。

## 3. 性能を出すための実装（効果順）

1. **出力 ring を forward の起動経路から外す。** 各 stage が forward を起動する条件は「前 stage から hidden が届いた」だけにする。出力（token id、request の完了情報）は NCCL の ring ではなく、既にある pyobj の経路（CPU）で最終 rank から直接各 rank へ送り、届いたものから `process_batch_result` する。`d2h_event.synchronize()` を起動前に置かない。EAGLE の hidden・topk は最終 rank でしか使わないので ring に載せない。判定：`pp.wait_result` が消え、GPU の NCCL のみの時間が 1 microbatch あたり 1 ms 未満。
2. **hidden の転送を NCCL kernel から copy engine に変える。** 送り側は最後の層の直後に `cudaMemcpyPeerAsync`（IPC handle か torch の symmetric memory）で相手の受信 buffer へ書き、flag を立てる。受け側の forward stream は `cuStreamWaitValue32` 相当で flag を待つ。SM を使わないので MegaMoE と干渉しない。NCCL を残す場合は、送り側が pyobj で「送った」と知らせてから recv を post し、SendRecv kernel の常駐を 1 ms 未満に抑える。判定：nsys で Prefill GPU に 2 ms を超える `ncclDevKernel` が無いこと、stage 3 の MegaMoE が全層 2.5〜2.8 ms であること。
3. **各 stage に microbatch を 2 つ先まで積む**（`pp-async-batch-depth 2`、loop size 6）。eager の起動は 25〜28 ms/microbatch で GPU の 113 ms より短いが、1 つしか積まないと 1 と 2 の後でも起動の遅れが露出する。KV の確保量は microbatch 1 つぶん増える。
4. **層の配分を 24/24/24/22 にする。** 最終 stage の lm_head + draft は 9.2 ms で約 2 層ぶん。今の 23 層 + 9 ms = 118 ms が最長 stage になる。配分後は全 stage 約 113〜115 ms。
5. **EP1 専用の MoE kernel**（1〜4 の後）。symmetric buffer と combine を外し、FC2 の epilogue で top-k の重み付き和を取る。expert あたり平均 504 行（EP4 の約半分）なので端数 tile を block_m 64 にする。MegaMoE −5〜8%、stage −3〜5% の見込み。
6. **確認**：全 12 Prefill GPU が約 690 W（電力上限）に張り付くこと（今回の serving は 590〜629 W）。stage 0〜2 の SMs Active が 95% 超（今回 65〜71%）。周期 ≤ 120 ms。TTFT は pipeline の 4 段で 1 要求あたり約 0.46 s になるので、DES（scratchpad の des.py）で c215 の TTFT 分布と開始時の fill を確認してから serving に出す。

## 4. 代替

PP2 × EP2（各段を TP1/DP2 attention/EP2）は、段数が半分で 1〜3 の実装が楽になり、FC2 の remote 比率も 3/4 → 1/2 に下がる。ただし MegaMoE の −13% のうち通信ぶんは半分しか取れない。PP4 の 1〜4 が難航した場合の退路とする。

## 5. 結果：結合を外した PP4 は 8,952.69（不採用）

§3 の 1〜4（出力 ring を起動経路から外す、hidden の転送を forward stream 上の PyNccl へ、`pp-async-batch-depth 2`、層 24/24/24/22）と concurrency 235 を branch `perf/prefill-pp4-decoupled-20261029`（overlay `pp4-direct-result.patch`、`pp4-proxy-instream.patch`、image は 40742736653 のまま）に実装し、serving [40825417205](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40825417205)（Slurm 27268、`[profile]` 付き）で 1 回測った。score は **8,952.69**（eligible、good 97.39%）で、main の 9,235.28（[40630160781](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40630160781)、EP4 c215）を 3.1% 下回る。assert（stage 間の microbatch 不一致、publisher の失敗）は出ていない。

| | EP4 c215（40630160781） | PP4 c235（40825417205） |
|---|---|---|
| 完了要求（124 s） | 5,811 | 5,708（−1.8%） |
| TTFT 不合格（全て開始 10 s 以内） | 95 | 139 |
| ITL 不合格 | 4 | 10 |
| TTFT p50 / p90 / p99（ms） | 1,257 / 1,483 / 3,726 | 1,730 / 1,901 / 4,688 |
| ITL 平均 / p99（ms） | 15.89 / 18.79 | 16.14 / 18.99 |
| Decode #running-req（定常、4 DP 合計） | 162.5 | 160 |
| Decode #transfer-req / #prealloc-req | 38.9 / 10.7 | 73 / 0 |
| Decode gen throughput（tok/s） | 9,707 | 9,544 |
| Prefill #queue-req / #bootstrap-req / #inflight-req（3 worker 合計） | 17.2 / 0 / 24.0 | 11 / 15 / 25 |
| EAGLE 受理長（bonus 込み） | 2.209 | 2.208 |
| Prefill GPU 電力・SM clock | 692〜694 W、1.47〜1.53 GHz | 687〜691 W、1.45〜1.51 GHz |

profile pass（prefill-0、44 microbatch、stage ごとに MegaMoE を層数で区切る）：

| stage（層） | 周期 中央値（p10〜p90） | 計算 | NCCL のみ | 空き |
|---|---|---|---|---|
| 0（24） | 124.3（120.7〜128.0） | 118.3 | 3.5 | 0.5 |
| 1（24） | 124.3（120.8〜128.0） | 118.9 | 4.1 | 0.4 |
| 2（24） | 124.6（121.4〜127.6） | 120.2 | 2.6 | 0.4 |
| 3（22 + lm_head + draft） | 124.1（120.6〜129.4） | 115.3 | 8.5 | 0.6 |

- **結合は外れた。** 周期は 164 → 124 ms、NCCL kernel と MegaMoE の重なりは 0、MegaMoE は全層 2.77〜2.82 ms（最大 3.5 ms）で停止は消えた。Prefill の 12 GPU は全て電力上限に張り付き、空きは 1 microbatch あたり 0.5 ms。PP0 の 2 要求 batch はログ上も毎秒 8 回（約 123 ms）。
- **ただし電力上限で計算が伸びた。** 24 層の計算は 113.5 → 118〜120 ms。§2 の 109〜113 ms は電力 590〜629 W（上限未満）での値で、上限に張り付くと約 5% 遅くなる。Prefill 供給は 3 × 2 / 0.124 ≈ **48.4 req/s で EP4（47.8）の +1%** にとどまった。§1 の「+10%」は電力上限を織り込んでいなかった。
- **遅延が concurrency を食う。** Decode の 1 要求あたりの生成速度は同じ（59.7 tok/s/要求）で、受理長も同じ。差は Decode にいる要求数だけで、PP4 は c を 20 上げても 160 にとどまった。残りは Prefill の pipeline 内（4 段 × 約 124 ms）と bootstrap 待ち（3 worker 合計 15 要求、Little の法則で約 0.33 s）にいる。bootstrap 待ちは PP の合意（stage 0 → 3 の積集合と、最終 stage から lag 2 で戻る合意）で、1 回あたり数周期かかる。EP4 では 0。TTFT p50 は 0.47 s 長い。
- **開始時の TTFT 不合格が 44 件多い。** pipeline を埋める遅延と bootstrap 合意の遅れが、開始直後の 235 要求に上乗せされた。
- DES の予測（9,217〜9,351）が外れたのは、周期を 113〜115 ms と置いたこと（実測 124）と、bootstrap 合意の遅延を模型に入れていなかったため。

### 5.1 PP4 を続ける場合の残り

1. **bootstrap 合意の短縮。** stage 0 が自分の bootstrap 完了だけで batch を決め、後段はその rid 列に従う（後段の bootstrap は KV 送信の前に済めばよい）。TTFT −0.3 s 程度、Decode の要求数 +14 程度。
2. **hidden の転送を copy engine へ。** in-stream の PyNccl が stage 0〜2 で 2.6〜4.1 ms、stage 3 で 8.5 ms 露出している。周期 −3〜8 ms（−3〜6%）。
3. **EP1 専用 MoE**（§3 の 5、image の再 build が要る）。stage −3〜5%。
4. concurrency 240。

1〜4 が全部効いても、Prefill 供給は EP4 の +5〜10%、遅延は EP4 より長いままなので、score は Decode 容量の上限（約 9,500）で頭打ちになる。10k には Decode 側の短縮が別に要る。
