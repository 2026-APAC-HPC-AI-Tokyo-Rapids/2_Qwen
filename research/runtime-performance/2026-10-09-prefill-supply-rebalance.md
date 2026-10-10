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

## 7. 単一Decodeのkernel内訳（running約63）

f84ea21（ed8c534と同一のserving条件に、Decodeのみの区間Nsightを追加）。Actions 37931615219のattempt 1はdgx034で`/dev/infiniband/rdma_cm`のenroot mountに失敗し、SGLang起動前に終了した（PBS218198、walltime3分15秒）。9月のdgx016と同じノード起因の事象で、条件を変えずに1回だけ再実行した。attempt 2（PBS218203、dgx021/dgx027、walltime12分28秒）でreport（60MB、archive全SHA256一致）を取得した。

AIPerfの計測開始から20秒後にDecodeの`/start_profile`（CUDA_PROFILER）を開き、21:22:59–21:23:06 SGTの約3秒を記録した。stop後のreport書き出し（約48秒）でDecodeが止まり、ITL p99が276msに伸びて計測runはSLA不適格になった。このscoreは計測の副作用であり、比較には使わない。記録区間直前のD runningは60〜70。reportはlogin nodeで`nsys export`によりSQLiteに変換し、rank0の66 target forwardを集計した。計測下のstep間隔は46.4ms、kernel合計は43.6ms/stepだった（通常runの推定約38ms/stepより、Nsight分だけ長い）。

| kernel | ms/step | calls/step | 所見 |
|---|---:|---:|---|
| `fused_moe_kernel`（Triton） | 18.48 | 188 | up 117.5µs/層は全expert FP8重み403MBを約3.4TB/sで読む帯域上限。down 77.3µs/層は201MBで約2.6TB/s（約78%） |
| FA3 attention（`FlashAttnFwdSm90`＋combine＋prepare） | 7.50 | 97 | 1 callでKV約66MB（63 request×1.05MB）を約0.97TB/s。HBMの約28%しか出ていない |
| FlashInfer AR＋residual＋RMSNorm（oneshot lamport） | 5.64 | 187 | 252 token×4096 BF16（約2MB）を約30µs |
| dense入力の`per_token_group_quant` | 3.02 | 188 | 約2MBの入力に16µs |
| DeepGEMM dense（QKV/O） | 2.16 | 188 | |
| その他（FP8 copy 0.74、moe_sum 0.71、通常AR 0.63 など） | 約6 | | |

**結論。** MoEは重み読出しの帯域上限にあり、batchに依存しない固定費である。running 1件あたりの増分（約0.35ms/step）のうち最大のものはFP8 KVのattentionで、約0.12ms/request/stepが帯域効率28%で消費されている。残りはtoken数に比例するall-reduce・量子化・小kernel群。

固定sourceで確認したこと：

- FA3は`page_size % kBlockN == 0`かつ`seqlen_q×GQA比 > kBlockM`のときだけpaged KVのTMA経路を使う（upstream `hopper/flash_api.cpp`の`get_pagedkv_tma`）。verifyは4×16=64で、FP8 hdim128のkBlockM=128を下回る。従って`page_size`を変えても非TMA経路のままで、効果は不確実。PDはPとDの`page_size`一致も要求する（`disaggregation/common/conn.py`）。
- AR融合のoneshot/twoshotは呼び出し側で指定されず（`use_oneshot=None`）、FlashInfer内部で決まる。252 tokenでもoneshotが選ばれていた。

**Decode改修候補（すべてimage再buildを伴う）。**

1. verify用FP8 decode attentionの帯域効率改善（28%→70〜80%なら約4ms/step短縮、batch比例の傾きもほぼ半減）。候補は、GQA 16×4 tokenを1 CTAにまとめるdecode専用kernel、FA3のTMA paged KV経路の強制（heuristicの見直し）、FlashInferのFP8 decode/XQA系。
2. AR融合を一定token数以上でtwoshotにする（約2ms/stepの見込み）。
3. dense入力のgroup128 FP8量子化をAR＋RMSNorm融合の出力側へ移す（約2.5ms/stepの見込み）。丸めは同じ値・同じgroupで行う。

1〜3で約8ms/step（通常runの約20%）が見込め、running 63でのITLは約17.2ms→約14ms。ITL 18msで単一Dが扱えるrunningは約67→約90へ増える。P3/D1の上限はDecode側の約3.6kから、Prefill側（3 worker、約4.1–4.3k）へ移る。導出スクリプトと集計は`/home/USER/qwen-artifacts/research-runtime/20261009-p3d1-decode-profile/`（SHA256SUMS付き）。

## 8. Decode verifyのSM90 XQA化（不採用、計算結果が壊れた）

§7で特定したFP8 attentionの低帯域を、FlashInfer 0.6.15のXQA kernel（SGLang `trtllm_mha` backendがSM90で使う経路）で置き換えようとした。

| commit | Actions / PBS | 結果 |
|---|---|---|
| 6b7df28（新patchと再build） | 37939137941 / 218213 | ServerArgsで停止。SM90では`trtllm_mha`をprefillに使えない（3分45秒） |
| 526019a（prefill fa3、decode trtllm_mha、`--speculative-attention-mode decode`） | 37945511358 / 218219 | XQAのJITとgraph captureは通過。最初のverifyで`HybridAttnBackend`に`max_context_len`がなく停止 |
| 662561f（`max_context_len`委譲を追加して再build） | 38012213131 / 218281 | 配信は動いたが、出力が壊れた |

662561fでは、dgx007/dgx013、walltime12分33秒、archive全SHA256一致。good率0%、accept len 1.01、ITL平均39.9ms。8件のgreedy probeはすべて崩れた文になった（例「The sky的 sky appears sky appears blue字体 blue …」、baselineは正常な文）。draftの提案がほぼ受理されず、出力も別物になったので、XQA経由のverify attentionが誤った値を返している。計算内容を変える変更は規則上採用できない。

固定sourceの状態としては、SM90の`trtllm_mha`は複数token verifyでXQAに必須のspec-dec mask（`xqa.py`: "Mask is required for speculative decoding"）を渡しておらず、上流でも検証されていない経路と考えられる。patchではFlashInferのテスト生成関数と同一のmaskを渡した。それでも結果が壊れたので、残る候補はspec-decでのseq_lens・位置の解釈、page 64でのKV layout、BF16 Q×FP8 KVのscale換算。kernel単体の数値比較なしに実機で試行を重ねることはしない（1回のキュー待ちが数時間）。

3commitはrevertし、branchの設定はed8c534（P3/D1、D pool393216、c72、FA3、page size 1）と同一に戻した。revertで削除したpatchは、git履歴の6b7df28と662561fに残る。Decode attentionの帯域改善は、GPU 1基で既存FA3と出力を照合できるkernel単体の検証環境を用意してから再開するのが妥当。

ASPIRE消費（myprojects 59010014）：本ノート全体で精算済み約2.2k SU、662561fの約0.43k SUは記入時点で未精算（balance 13347.163、In Doubt 1024）。

## 9. Decode AR融合のtwoshot化（不採用）

a37febb（新patch `decode-allreduce-twoshot.patch`を含めて再build）。§7で5.6ms/stepだったFlashInfer TRT-LLM AR＋residual＋RMSNorm融合は、FlashInfer 0.6.15のB200向けheuristicにより、TP4では1024 tokenまでoneshot（lamport）になる。この選択を、`SGLANG_FLASHINFER_AR_ONESHOT_MAX_TOKENS=128`を超える呼び出しでtwoshotに切り替えた。Decodeのみ。workspace初期化には手を入れていない。

Actions 38016612848、PBS218289、dgx007/dgx013、walltime12分07秒、archive全SHA256一致。Decodeのargvで環境変数の適用を確認した。

| 指標 | c72 oneshot（ed8c534、dgx004/026） | c72 twoshot（a37febb、dgx007/013） |
|---|---:|---:|
| D running 平均 | 61.6 | 63.0 |
| ITL 平均 / p90 | 17.18 / 18.77 ms | 18.19 / 19.92 ms |
| ITL >20ms | 28 | 183 |
| good率 | 97.61% | 89.89%（不適格） |
| accept len | 2.23 | 2.23 |

greedy probeは8/8でc72と同一で、数値は保たれている。running差による伸び（約0.2ms）を除いても、ITLは約0.8ms（約5%）悪い。node対が異なるのでnodeの個体差を切り分けられないが、改善は確認できない。H100 NVLink上の約2MB・TP4では、twoshotの2段同期がoneshotの読出し量削減を上回ると考えられる。**不採用。** patchとconfigはrevertし、branchの設定をed8c534へ戻した。

ASPIRE（myprojects 59010014）：記入時点でbalance 12918.790、In Doubt 1024（本runの未精算分）。本ノートの全試行で約3.1k SU。

## 10. Firmus H200：P6×TP2 / D1×TP4（採用候補）

ユーザー承認のもと、チームメイトのFirmus経路（`codex/firmus-sglang-dynamo` 23463ed、P4×TP2/D2×TP4 c64）を親にして、別branch `research/firmus-p6tp2-d1tp4`（ff58ed8）で1条件だけ実行した。チームメイトのbranchは変更していない。

親run 37888931369は3450.9 good tok/s（97.9%）、平均ITL 9.5ms、D runningは約18/worker。TP2 prefillはworkerあたりログ中央値約21.5k tok/s（GPUあたり約10.75k、H200 TP4の約8.1kより約33%高い）だった。H100と同じく、Decodeの余裕をPrefillへ回す配置とした。

条件：TP2 prefillを6 worker（head nodeに4、tail nodeに2）、TP4 decodeを1 worker（tail node、`--max-total-tokens 524288`）、c104。image（37886841020、sha256 `4b919a74…`）、EAGLE3、FP8 KV、P kernel bundleは親と同じ。

Actions 38020231725、Slurm 22269（g1/g4、COMPLETED 0:0、11分32秒）。archive全SHA256一致。server-infoでP×6がTP2/pool131072/max-running104/CUTLASS/symmetric memory、DがTP4/pool524288であることを確認した。

| 指標 | P4×TP2/D2×TP4 c64（37888931369） | **P6×TP2/D1×TP4 c104（38020231725）** |
|---|---:|---:|
| good output tok/s | 3450.9 | **4369.4** |
| good率 | 97.88% | 93.57%（2879 request） |
| TTFT 平均 / p90 | 1687 / — ms | 840 / 1356 ms（>3s 38件） |
| ITL 平均 / p90 | 9.53 / — ms | 17.86 / 19.53 ms（>20ms 153件） |
| D running（ログ平均 / 最大） | 約18 / worker | 57.1 / 101 |
| P queue（ログ平均） | — | head P0–3 約0.16、tail P4/5 1.33 |

親より+26.6%、TP4 P2/D2 c48基準（3170、37868180304）より+37.8%。いずれも単回・別allocationの比較で、matched pairではない。

H100 ASPIREと同じ構図で、律速はDecodeのITLに移った（平均17.9ms、p90 19.5ms）。Prefill側は、head nodeのworkerがほぼ待ちなし、decodeと同居するtail nodeの2 workerに軽い待ちがある。Firmusでのさらなる改善点は、単一Decodeのrequestあたりコスト（§7のattention帯域）と、同居prefillの遅さの原因特定。cの微調整（例：96）は、ITL違反を減らす代わりに並行数も減るので、取り得る幅は小さいと見込む。
