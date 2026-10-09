# Prefill供給律速とP/D再配分

2026-10-09。目標はASPIRE H100で6000、Firmus H200で10000 good output tok/s（他チームの報告値）。本稿は既存runのログから律速段を定量化し、その結果に基づくP3/D1大pool試行を記録する。

## 1. goodputはPrefill GPU数でほぼ決まっている

4000/200 workloadでは1 good output tokenあたり20 input tokenのprefillが必要。cache-bustはrequestごとに一意な`[rid:<12hex>]`をprompt先頭へ置く（AIPerf 0.12.0 `timing/strategies/cache_bust.py`）ため、request間のprefix cacheは効かない。

Prefillログの`input throughput`は1バッチごとの瞬間値で、その単純平均は待ちの長いバッチを軽く数えて過大になる。queueが3以上ある飽和区間で、ログ上のバッチ新規token総数を区間長で割った持続速度を求めた（1秒分解能、`prefill_rate.py`相当の計算）。

| run | 構成 | 持続Prefill tok/s/worker | ログ値の単純平均 |
|---|---|---:|---:|
| ASPIRE 37872490282 | H100 TP4、chunk8192、c64 | 29273 / 28951 | 31023 / 31507 |
| Firmus 37868180304 | H200 TP4、chunk8192、c48 | 31892 / 31955 | 44866 / 51067 |
| Firmus 37872430145 | H200 TP4、chunk8192、c96 | 32114 / 32215 | 35090 / 36350 |
| Firmus 37873721300 | H200 TP4、chunk16384、c96 | 32953 / 32953 | 43744 / 40827 |

GPUあたりではH100約7.3k、H200約8.1k input tok/s。chunk16384は持続速度を約2.6%しか上げず、以前「43k tok/s」と読めた値は平均の取り方による見かけの値である。

これより、good output上限は概ね `P GPU数 × GPUあたりprefill速度 / 20 × good率` となる。

| | P8（現行） | P12 |
|---|---:|---:|
| H100（7.3k/GPU） | 約2900（実測2853–2865） | 約4300 |
| H200（8.1k/GPU） | 約3250（実測3170） | 約4900 |

一方Decodeは現行P2/D2で1 workerあたりrunning約17、平均ITL約10ms（SLA 20ms）と大きく余裕がある。c96のFirmus試行でもD runningは約16のままで、TTFTだけが4秒超へ伸びてSLA不適格となった。

**含意。** 現行のGPUあたりPrefill効率のままでは、P12にしてもH100 6000・H200 10000に届かない。必要なGPUあたり効率はH100 P12で約10.2k（+40%）、H200 P12で約16.7k（+105%）、H200 P14（D TP2）で約14.3k（+75%）。従って施策は(1) P/D配分をPへ寄せる、(2) GPUあたりPrefill効率を大きく上げる、の2本立てになる。同じ4K/200負荷を対象とするNVIDIA DynamoのHopper向けdisagg recipeもP6×TP2/D1×TP4EP4（P12/D4）を採っている（数値は未公開）。ただしTP2はH100 80GBにtarget重みが載らないため、H100ではTP4単位でP3/D1になる。

## 2. 旧P3/D1が負けた理由

旧P3/D1（4ac2cf6、Actions 37092369218、c40、D pool131072）は1863.6 good tok/sだった。単一Dのrunningは平均25.2・最大28、prealloc待ちは平均11.8、token usageは約0.9で、Dのadmission poolが上限に達していた。Prefill 3 workerのqueueは0で、Pは遊んでいた。つまりこのrunは配分ではなくD poolの上限を測っており、P3/D1自体の棄却根拠にはならない。

## 3. 今回の試行：P3/D1、D pool393216

runtimeのみの変更で、既存image lineageを再利用する。node1のGPU0–3をprefill worker2へ移し、GPU4–7に単一のTP4/EP1 decode workerを置く。decode poolは393216 token（512 token予約込みで約87 request）。c64 runのD rankはpool196608でgraph後13.12GB空きがあり、追加分はFP8 target KV約4.4GB＋draft KV約0.4GB。

| 条件 | commit | Actions |
|---|---|---|
| P3/D1、D pool393216、c72 | ed8c534 | [37915507286](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37915507286) |
| 同、c88 | 1948ae0 | [37915544198](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37915544198) |

事前の不確定要素は、単一DのITLがrunning約60でどこまで伸びるか。保存済みH100 decodeログではrunning 8–15で約10.4ms、24–31で約13.6msで、それ以上のbatchは実測がない。

結果は次節に追記する。

## 4. GPUあたりPrefill効率の改善候補

P/D再配分だけでは目標に届かないため、次の律速はGPUあたりPrefill効率になる。2026-10-04のPrefill profile（37135273516、symmetric memory・融合shuffle・tiled routing採用前）では、P rankの5秒窓のkernel時間4.56秒の内訳は概ね次のとおりだった。

| 区分 | 時間 | 比率 |
|---|---:|---:|
| MoE grouped GEMM（up 715 + down 578） | 1293 ms | 28% |
| TP all-reduce（BF16 ring LL） | 1043 ms | 23% |
| MoE補助（input/scale shuffle 332、combine 324、sort/problem size 155、quant 45） | 約855 ms | 19% |
| dense QKV/O GEMM | 352 ms | 8% |
| FA3 attention | 330 ms | 7% |
| fused add RMSNorm | 280 ms | 6% |

このうちsymmetric memory（+2.7%）、shuffle融合（+1.3%）、tiled routing（+2.5%）は採用済みで、上表の一部をすでに削っている。残りの候補を、想定効果の大きい順に並べる。

1. **Prefill MoEのEP化（TP4 group内、CUTLASS local expert）。** TP4のMoEでは各rankが全65536 routeをshuffle/combineし、intermediate幅384の細いGEMMを回している。同じ4 GPU内でAttention TP4を保ったままMoEをEP4にすると、各rankのroute処理は約1/4、GEMMは幅1536（downのKが384→1536）になる。MoE補助と GEMM非効率の合計（約2.1秒中）から見て、GPUあたり+15〜25%の余地がある。通信はEP部分和のall-reduceで現行と同量。expert偏りが損失要因。チームメイトが`codex/native-cutlass-ep`で準備中のものがこれにあたる。既存Triton EPはglobal route数でkernelとbufferを確保し、無効routeにもzero書込みをするため、この利益は得られない。
2. **all-reduceのreduce-scatter＋FP8 all-gather化（Prefillのみ）。** o_proj後とMoE後のall-reduceをreduce-scatterにし、add+RMSNormとgroup128 FP8量子化をshard上で一度だけ行ってから、FP8値とscaleをall-gatherする。後段のQKV GEMMとMoEはもともとこの同じFP8量子化値を入力にしているため、計算内容は変わらない（加算順序の違いのみ）。通信量は約25%減、RMSNormと入力量子化は1/4になる。見積もりは+8〜12%。固定sourceの`enable_attn_tp_input_scattered`はMLA専用（`q_lora_rank`必須、EAGLE3不可）なので、Qwen3には新規実装が必要。
3. **Prefillの2 micro-batch通信overlap。** 8192 tokenバッチを2分割し、片方のall-reduce中にもう片方を計算する。all-reduce（現在約20%前後）の大半を隠せる可能性があるが、NCCLのSM占有とGEMM小型化で相殺される。SGLangのTBOはDeepEP前提なので新規実装になる。
4. **H200のみ：Prefill TP2。** 通信係数が1.5→1.0になり、GEMM幅が倍になる。チームメイトのrun 37888931369（P4×TP2/D2×TP4、c64）で効果が分かる。H100では重み約111GiB/rankとなり載らない。

H100で6000には「P3/D1が単一DのITLを満たす」ことに加え、上記1〜2の合計で+40%程度が必要になる。H200で10000は、P12でGPUあたり約2倍の効率（P14/D2では約+75%）が必要で、上記の積み上げだけでは届かない可能性が高い。

## 5. 結果

### c72（ed8c534、Actions 37915507286、PBS218108）

dgx004/dgx026、19:14 SGT開始、walltime12分17秒、Exit0。archive 全SHA256一致、image `182fbb6d…`、commit ed8c534。server-infoでP×3がTP4/pool131072/CUTLASS/symmetric memory、DがTP4/pool393216/auto MoEであることを確認した。D rankのgraph capture後の空きは8.67GB。

| 指標 | 値 |
|---|---:|
| good output tok/s | **3313.0** |
| good率 | 97.61%（2093 request、error 0） |
| TTFT 平均 / p90 / p99 | 769 / 1253 / 3196 ms（>3s 23件） |
| ITL 平均 / p90 / p99 | 17.18 / 18.77 / 20.22 ms（>20ms 28件） |
| D running 平均 / 最大 | 61.6 / 68（prealloc待ち0、retract 0） |
| D gen tok/s（ログ平均） | 3499、accept len 2.23 |
| P queue | 3 workerとも平均0 |

同時期のc48基準（約2853–2865、別node対）より約+16%。matched pairではないため改善率は暫定。

律速はPからDへ移った。Pはqueueが空で、GPUあたりの持続prefillは約5.4k tok/sにとどまる。単一DのITLはrunningに対してほぼ線形に伸びた（running約17で10.2ms、約62で17.2ms、1 requestあたり約0.155ms）。1ステップに換算すると1 requestあたり約0.35ms。verify 4 token分のMoE・射影計算量（rankあたり約43 GFLOP/request/step）を小さいMで回していることと整合するが、kernel時間での確認はしていない。

ITL 20msの制約下で、単一D TP4が出せるgood outputは概ね3500–3700 tok/s。P3/D1はD律速、P2/D2はP律速で、TP4単位ではその中間の配分がとれない。次の改善対象は、大batchでのDのrequestあたりコスト。

### c88（1948ae0、Actions 37915544198、PBS218109）

dgx004/dgx026、walltime12分29秒、PBS Exit0、archive全SHA256一致。AIPerfはerror 0で完走したが、good率67.50%でSLAゲート不適格（score 0）。Actionsのfailureはparserの`--require-eligible`による。

| 指標 | 値 |
|---|---:|
| TTFT 平均 / p50 / p90 | 2037 / 779 / 6732 ms（>3s 484件） |
| ITL 平均 / p90 / p99 | 17.16 / 19.78 / 21.79 ms（>20ms 165件） |
| D running 平均 / 最大 | 58.6 / 76、transfer待ち26.6 |
| P0 / P1 queue | 0 / 0 |
| P2 queue 平均 / 最大 | 13.8 / 28 |

TTFT不合格の主因はP2（Dと同じnode1のGPU0–3）の飽和だった。P2の持続prefillは21.4k tok/sで、node0のTP4 workerの通常値（約29k）より約26%遅い。c72でもP2はinflight 2.9（他1.7）、バッチあたりtoken 5800（他4400）と、同じ兆候があった。原因候補はノード内のPD KV転送経路（UCX/NIXL）、NVSwitch・CPUの競合、GPU配置だが、ログからは特定できない。round-robinは各Pへ均等に振るため、遅いP2にだけqueueが溜まる。

ITLは平均がc72と同じでもp90が19.78msへ伸び、上側の裾が厚くなった。単一DのITLの余裕はc72付近でほぼ尽きている。

### c72、Decode CUTLASS（b04c6ba、Actions 37925814284、PBS218175）

c72（ed8c534）と同じdgx004/dgx026。変更点はDのMoE runnerをauto（Triton）からnative CUTLASSにしたことだけ。walltime12分19秒、archive全SHA256一致、server-infoでDのCUTLASSを確認した。AIPerfはerror 0で完走したが、good率78.26%でSLAゲート不適格（score 0）だった。

| 指標 | c72 Triton（ed8c534） | c72 CUTLASS（b04c6ba） |
|---|---:|---:|
| D running 平均 | 61.6 | 63.4 |
| ITL 平均 / p90 | 17.18 / 18.77 ms | 18.85 / 20.67 ms |
| ITL >20ms | 28 | 414 |
| D gen tok/s | 3499 | 3296 |
| good tok/s | 3313.0 | 0（不適格） |

running差（+1.8）による伸びは、上記の傾きで約0.3ms分にすぎない。補正後もCUTLASSは約1.4ms（約8%）遅い。running約60（verify約250 token）でも、Decodeは既存Tritonの方が速い。**不採用。**

## 6. 判断と次の候補

- **暫定最良はP3/D1、D pool393216、c72の3313.0 good tok/s。** 採用には、main（P2/D2 c48）との同一node対・順序反転のmatched pairと反復が必要（各約420 SU）。主催者accuracyは配置変更では計算内容が変わらないが、別ゲートとして残る。
- **P3/D1の律速は単一DのITL。** 傾きは1 requestあたり約0.155ms。原因候補はverify 4 tokenぶんの計算で、MoEだけでなくQKV/O、attention、LM headも含む。対策のためには、running約60でのDecode kernel profileが次に必要（profileは通常benchと分ける）。CUTLASSは否定された。残る候補はTriton MoE configのM≈250向け調整、verify token数（steps2/draft3）、Dのall-reduce融合範囲の確認。
- **Dと同居するPは約26%遅い（21.4k vs 29k tok/s）。** round-robinではこのworkerにqueueが偏り、c88でTTFT不合格の主因になった。同居の原因を特定するか、遅いworkerへの配分を減らすrouting（inflight数ベース）が必要。
- **GPUあたりPrefill効率**（§4）は依然として、H100 6000・H200 10000のための最大の課題。teammateのCUTLASS EP（Prefill MoE EP化）と、Firmus TP2 Prefill（run 37888931369）の結果を待って優先順位を更新する。
- **Firmus（チームメイト担当）への提案。** H200でもP2/D2はP律速（3170）。H200のDはHBM帯域が大きいので、P3/D1（D pool拡大）での単一DのITLはH100より低く出る可能性が高く、最初に試す価値がある。

今回のASPIRE消費は、精算済みの2本で419.27 SUと426.10 SUだった。CUTLASS runは約420 SUで、本稿の記入時点では未精算（myprojects: balance 14714.772 SU、In Doubt 1024）。
