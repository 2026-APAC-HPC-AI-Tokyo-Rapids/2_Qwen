# カーネルの理論値と改善台帳

更新日 2026-10-28。時間と仕事量の正本は `kernel-profile-40310899466-*.json` と `kernel-theory-model.json`。この文書は `python3 research/runtime-performance/render-kernel-theory.py --write` で再生成する。計算・解析のみでGPU jobは起動しない。

実測費用、理論上必要な仕事量、削る対象、過去の結果を同じ単位で追う。最大の未解明箇所はPrefill MegaMoE内部の計算と出力転送、および現在のN192 FA3の依存待ち。小kernelの融合を重複計上しない。

## 比較条件

対象はrun40310899466、image40197369083、H200、3P1D、EP4、94層。通常scoreと別のprofile passからprefill-0/decode-0 GPU0を再抽出した。両端2周期を外し、同じ94 MegaMoE周期内の全kernelを保持する。PDLで周期境界を跨ぐ仕事は外側の窓だけでclipし、kernel時間の和と重複区間のunionを別々に記録する。元nsys-repのSHA256はarchive manifestと一致。

このprofileはBF16 draft時代のもの。後続のFP8 draft採用後run40480817670、router run40528222753のkernel時間として扱わない。新しい通常runのscoreと古いprofileのkernel差を因果関係で結ばない。

Prefill代表shapeは3994行（norm_quantの実gridは3994/3995）。旧4032行仮定を訂正。Decodeは144〜176行の混在で、以下の理論欄は176行・44要求・KV4100の条件模型。shape別の新しい実測とは異なる。

## 理論値の定義

GEMMはFMAを2 FLOPとし、`2MNK`。MoEはFC1とFC2を分け、有効行数とtile丸め後の行数も分ける。Tensor Coreのdense FP8は8192 FLOP/SM/cycle、BF16は4096。132 SMとの積から計算し、旧 `1979×clock/1.98` は使わない。公称FP8 1979 TFLOPSとHBM4.8 TB/sは[NVIDIA仕様](https://www.nvidia.com/en-us/data-center/h200/)、SM単位の算出は[Hopper解説](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)と[NVIDIA技術回答](https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727)に基づく。

同窓GPC rawのworker内平均をHzと解釈した場合、Prefill約1.35〜1.38GHz、Decode約1.65〜1.73GHz。表の演算欄はP1.35〜1.40GHz/D1.65〜1.74GHzの明示的な条件計算。NsightのMHzラベルとraw値の桁が不一致で、kernel別clockとdevice対応は未確定。公称値の床も機械計算に保持する。

独立資源の必要時間は `max(F/P, B_HBM/W_HBM, B_NVLink/W_NVLink, 命令/発行率)`。因果関係のある直列区間だけを足す。表の『論理I/O換算』は各bufferの論理byteを4.8TB/sで割った参考値であり、L2 hit・再読出し・同期を含むHBM実測値でも、厳密なkernel下限でもない。理論との差をそのまま削れる時間としない。

## 現在の実行状態

2026-10-28、ユーザー指示で旧Firmus kernel benchを停止。Actions40563900060を取消し、停止専用workflow40567082710でSlurm26800のCANCELLED・経過0秒・割当0台・残存runtime削除を確認した。この停止済みbenchのGPU測定は0回。旧benchとは別に通常serving試験を完了し、今回FC2候補の改善は認められなかった。

kernel benchはASPIRE H100 EP4を優先する。従来のFirmus移行条件の目安はASPIRE通常serving公式6500 output tok/sで、kernel単体の改善率では代替しない。最新ASPIREはc112の6081.54（run40274854588、BF16 draft、FA3 control）。6500には約6.88%必要。c104の5925.07はFA3 n192だったため、差をconcurrencyだけの効果としない。今回の通常Firmus結果は以下に記録する。

ASPIRE EP4経路はcommit396f1fbで実装済み。run40569440787のLinux全検証は成功したが、接続中継WIN-5IM4VJ0SFOSへのSSH timeoutで準備前に停止した。PBS投入・GPU測定・runtime作成は0。中継は10/27 23:36 JST以降offlineで、直接ASPIRE接続もtimeout。このASPIRE benchは再投入していない。

旧通常run40652220352はGPU前CIで4件の設定期待値testに失敗。GPU測定0・runtime作成なし。ASPIRE固定7本とFirmusのFC2追加、D5a後のFC2順序、凍結submissionとの差分を3test fileで明示した。修正commitbe5c453のローカル検証は80 passed・1 skipped（既存source mirror未配置）、Ruffとdiff検査はPASS。実行設定・FC2候補・submission実体は変更していない。

ユーザーの追加指示『いやfirmusで、benchじゃなくて、普通に投入して欲しいなって』により、今回の通常Firmus FC2試験はASPIRE6500条件より優先して明示許可された。ASPIRE6500は未達のまま。通常serving run40653531712（commitbe5c453）は投入済み、記録時点の状態はsuccess。kernel benchは実行せず、9235.28のc215・FP8 draft・Prefill開始整列を保持しFC2 patchだけを追加した。FC1候補は非active。

通常run40653531712は公式9234.34、reference9235.28比−0.0102%で実質横ばい。平均TTFT1345.50→1329.99ms、ITL15.890→15.939ms、全応答4507.64→4501.79ms。公式validation/eligibility合格・error0、Slurm26960 COMPLETED0:0・14分15秒、artifact回収とruntime削除を確認。設定はFC2追加だけで、実際のPrefill/Decode選択envも一致した。単回の別run比較なので微小差の因果は断定しない。採用済み改善として数えず、同候補の自動再投入を止める。詳細は `megamoe-fc2-serving-40653531712.json`。

ユーザー目標は公式good output 10000超。9234.34から約8.29%必要。raw outputも9403.98なので、SLA違反の解消だけでは10000に届かない。追加GPU計測・subagent再開は行わず既存routingを解析する。

直近の完了Firmus reference run40630160781（840b40f、c215、FP8 draft、Prefill開始整列）は公式9235.28 good output tok/s。imageと7本のDeepGEMM overlayのSHAは本候補の既存controlと全一致した。作業baseをこのcommitへ進め、c215等を保持するFC2追加設定の比較基準とする。FC2 bundleはGitに存在するが9235のserving設定では未有効。旧6f0164dの5本を再適用したheaderともbyte一致するため、新しいFC2手法とは呼ばない。同bundleの旧servingは8922.77（旧8854.80比+0.77%の単回比較）、個別/結合kernel benchの記録は見つからない。bulk/fragment/transpose等の過去の棄却案とは異なり、それらの再測定は不要。今回の9235への追加試験は9234.34（−0.0102%）で上積みなし。詳細は `megamoe-fc2-history-and-9235-compatibility.json`。

FC1にもscale読取アドレスの共通化候補を作成した。約509万組のpointer一致、6 geometryのfresh CPU compileを確認。Prefill128/140のFC1静的spanは290→277/280→267命令（整数address-family各−10）、168 registers・spill0を維持。その他4 geometryは命令encoding一致。動的命令数や速度改善率ではなく、GPUでの数値一致・速度は未確認。候補patchはactive bench inventoryに含めない。詳細は `megamoe-fc1-cpu-validation-20261028.json`。

Prefill FA3 N192のepilogueを、accumulatorから直接group-128 FP8へ量子化する候補 `n192-direct-o`（variant 32）を作成した。BF16 smem staging、quad SHFLの列並べ替え、named barrier2回、staged layoutのBF16展開とgroup-8 shuffleを除き、consumer warpあたりM-tileごとの実行命令は808→343（CUDA 13.4、3994行で1回7.44M warp命令・smem131MB減）。定常KV loopは597命令/tileのまま、registers・spillは不変、他のgeometryのSASSは一致。出力codeとscaleはn192とbyte一致する設計で、in-job選択のGPU gateがn192とのbyte一致と速度を判定する。見込みは単体−4〜−6%、電力上限下−2〜−5%でFA3目標10%に届かない。GPU未測定・未有効化。詳細は `fa3-prefill-direct-o-cpu-validation-20261029.json`。

PP4（run40825417205）では2要求のmicrobatchが固定経路に入らず、prepared N224（899.7µs/回＋prepare 4.1µs）で動いていた。固定schedulerを1〜4要求の連結順へ広げたvariant 64の `n192-varlen` と `n192-direct-o-varlen` を追加し、PP4ではこの2行だけを3長と3986+4027の組でin-job選択する（controlは組ではprepared N224）。数値gateは要求ごとのn192とのbyte一致を要求する。定常KV loopは不変、spillはproducer側で72/132→96/124B、他kernelのSASS本体は一致。見込みは124ms周期あたり電力上限下で−0.7〜−1.6ms（direct-o併用）。draft extend（stock paged N160、BF16出力）は同じ64/4/128だが出力・KV形式が異なり、stage3は律速でないため未配線。GPU未測定。詳細は `fa3-prefill-varlen-cpu-validation-20261029.json`。

## Prefill の収支

17周期、周期平均 **250.991 ms**、kernel union **248.908 ms**、kernelのない区間 **2.083 ms**。92個のkernel名とgeometryの組を全て保持。重なったkernelの時間和を周期時間へ足し直さない。

| kernel | 実測 ms/周期 | 実測 µs/回 | 演算だけの床 µs/回 | 論理I/O換算 µs/回 |
|---|---:|---:|---:|---:|
| MegaMoE | 142.233 | 1513.11 | 796.7–826.2 | 未算定 |
| FA3 target | 41.023 | 436.41 | 172.7–179.1 | 14.9 |
| QKV GEMM | 23.636 | 251.45 | 199.2–206.6 | 26.6 |
| O GEMM | 20.596 | 219.10 | 177.1–183.6 | 20.6 |
| QKnorm RoPE | 6.224 | 66.22 | 未算定 | 23.9 |
| draft等 BF16 GEMM | 3.845 | 549.27 | 未算定 | 未算定 |
| add RMSNorm | 3.187 | 31.87 | 未算定 | 27.3 |
| norm quant | 2.855 | 31.72 | 未算定 | 24.0 |
| pre dispatch | 1.537 | 16.35 | 未算定 | 10.5 |
| router gate | 1.408 | 14.98 | 5.5–5.7 | 未算定 |
| その他 未モデル化 | 1.315 | 3.29 | 未算定 | 未算定 |
| routing topk | 0.589 | 6.26 | 未算定 | 未算定 |
| draft FA3 | 0.516 | 516.25 | 未算定 | 未算定 |

µs/回はその分類の実測平均。shape混在・draft混在の行に単一shapeの床を掛けて周期全体の『余地』を作らない。

## Decode の収支

134周期、周期平均 **38.343 ms**、kernel union **37.665 ms**、kernelのない区間 **0.678 ms**。501個のkernel名とgeometryの組を全て保持。重なったkernelの時間和を周期時間へ足し直さない。

| kernel | 実測 ms/周期 | 実測 µs/回 | 演算だけの床 µs/回 | 論理I/O換算 µs/回 |
|---|---:|---:|---:|---:|
| MegaMoE | 21.526 | 229.00 | 28.2–29.8 | 125.8 |
| FA3 target | 6.738 | 71.68 | 12.6–13.3 | 38.5 |
| その他 未モデル化 | 2.583 | 6.77 | 未算定 | 未算定 |
| QKV GEMM | 2.309 | 24.57 | 7.1–7.4 | 8.7 |
| O GEMM | 1.769 | 18.81 | 6.3–6.6 | 7.6 |
| add RMSNorm quant | 0.832 | 9.24 | 未算定 | 1.1 |
| verify split merge | 0.427 | 4.54 | 未算定 | 2.7 |
| QKnorm RoPE | 0.420 | 4.46 | 未算定 | 未算定 |
| add RMSNorm | 0.361 | 3.47 | 未算定 | 未算定 |
| routing topk | 0.337 | 3.59 | 未算定 | 未算定 |
| draft FA3 | 0.284 | 94.63 | 未算定 | 未算定 |
| pre dispatch | 0.205 | 2.18 | 未算定 | 未算定 |
| router gate | 0.006 | 28.41 | 未算定 | 未算定 |

µs/回はその分類の実測平均。shape混在・draft混在の行に単一shapeの床を掛けて周期全体の『余地』を作らない。

## MegaMoEの既存内部計装と今回の変更

run40057117293、image39932105942、prefill-real43-replica4-4096。代表callのWG0、CTA中央値（µs）。rawとparserのhashは`megamoe-phases-40057117293.json`に保存。両WGのmedian/maxも同JSONに保持。現行servingとは別条件。

| 区間 | rank0 | rank1 | rank2 | rank3 |
|---|---:|---:|---:|---:|
| 最初のstage待ち | 22.8 | 24.2 | 27.2 | 28.5 |
| FC1 mainloop | 653.9 | 668.4 | 689.0 | 682.8 |
| うちFC1 stage待ち | 61.2 | 61.7 | 63.0 | 63.5 |
| FC1 SwiGLU/FP8/store | 47.4 | 48.6 | 50.1 | 49.6 |
| FC2 mainloop | 342.2 | 339.2 | 349.9 | 345.2 |
| うちFC2 stage待ち | 40.0 | 36.9 | 36.1 | 35.6 |
| FC2 BF16/address/scatter | 186.7 | 198.3 | 200.8 | 208.7 |
| combine前grid/NVLink待ち | 75.1 | 49.8 | 11.8 | 16.2 |
| combine/cleanup | 80.6 | 80.3 | 78.1 | 79.3 |

通常bench中央値1407.1µs、計装bench中央値1450.8µs、別の8-call dumpのcalls2–8中央値1423.9µs。これらを同じ測定値にしない。CTA中央値の和もcritical pathではない。stage待ちはmainloopの内数。

FC2出力の変換・アドレス・store発行はepilogueに入り、完了待ちはgrid/NVLink同期に混ざる。転送の単独所要時間は未確定。localケースとの差もcache/仕事分布が変わるため純NVLink時間にはならない。

| rank | 同call SM GHz | FC1 compute床 µs | FC2 compute床 µs | FC1 useful/peak÷loop | FC2 useful/peak÷loop |
|---|---:|---:|---:|---:|---:|
| 0 | 1.6531 | 453.0 | 226.5 | 69.3% | 66.2% |
| 1 | 1.6498 | 461.2 | 230.6 | 69.0% | 68.0% |
| 2 | 1.6027 | 478.0 | 239.0 | 69.4% | 68.3% |
| 3 | 1.6186 | 478.5 | 239.2 | 70.1% | 69.3% |

この比率はpadding/setup/stage待ち/FP32 promotionを含むCTA中央値との診断比で、Tensor pipe実測稼働率ではない。旧73–75%を訂正するが、残差全部を短縮可能と主張しない。

このfixtureのremote率は53.40%。FC2 partialは平均268.44MB/rank、remote平均は143.34MB/rank。combineの論理read+writeは301.99MB/rankで、4.8TB/s換算62.9µs。combineのFP32 FMAは0.268GFLOP/rankで、memory以外の変換・同期も残る。実転送量のcounterではない。source別expert countsがない旧runのrank別remote量とpadding量は未確定として残す。

今回の候補：6f0164dの5つのFC2出力改善が別branchのまま未統合だった。アドレスを行単位で先に計算し、行の有効性分岐・shared address計算を減らし、metadata loadをBF16 packingに先行させる。BF16 payload、combine加算順、同期は保持。転送量削減とは呼ばない。

旧run40116925551はscore8854.80→8922.77だったが、単回・別run比較で因果効果は未確立。現行D5aとの組合せをcontrol/candidate/control-repeatで一度検証する。無関係なQKV候補は再実行しない。現行controlの代表routingとlocalだけに別passのtimelineを付け、同時にsource-rank×expert countsを保存する。このcountsは集約済みlayer分布から生成したfixtureの値であり、本番source別routingではないため本番配置改善の根拠に流用しない。候補timelineはCPU compileでspill増加（store32B/load80–84B）が出たため実行しない。通常版は双方168 registers、stack/spill 0、Prefill命令slotは7512→7200 / 8072→7768、Decode3形状のencodingは完全一致。

### 通信は律速ではないのか

既存run40071892077/40313004834のSQLiteには、NVLink request/response別のUser/Protocol Data Throughput %が残っていた。403130の全4 CUDA GPUがMegaMoE実行中の共通区間で、TX Requests User Data平均18.11–18.35%、p95 31%、最大37%。同窓のTensor Active平均50.96–52.99%、DRAM read15.08–15.31%、write6.77–6.82%。詳細とraw hashは`megamoe-communication-metrics.json`。

これは約1msごとのサンプルで、約200µsのFC2出力区間や約80µsのcombineを分解できない。sampleの集計期間はkernel外を跨ぎ得る。typeIdとCUDA deviceIdの直接対応も未取得なので、下位bitから推測せず全GPU共通区間で比較した。

[NVIDIA定義](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#available-metrics)のNVLink bytesはsample期間の最大byteに対する割合だが、このrequest/response細分metricを足して総利用率やGB/sへ換算しない。低い全体平均は短い転送区間の飽和を否定しない。raw link bytes、L2 traffic、PC stallは未取得。今回のaddress変更は減らせる仕事が具体的な候補であり、『通信ではなく命令が主因』という結論ではない。

## 仕事量と改善の対応

### Prefill MegaMoE

FC1=4*T*topk*H*I, FC2=2*T*topk*H*I。

有効演算 1206.148 GFLOP/回。全EP rank均衡の平均仕事量。rank別負荷/padding/通信は未確定。

追う差：GEMM外のBF16出力・combine・通信、tile padding、rank待ち。完全均衡のFLOP下限との差全量を削減可能としない。

次の作業：通常FC2 run40653531712は9234.34、9235.28比−0.0102%で改善なし。既存本番routing40621030371の配置探索もtrain2.316msに対しheld-out0.103msで採用条件未達、追加GPU投入なし。10000超には+8.29%必要。小さいaddress変更とこの配置案は再投入せず、FC1/FC2のmainloop・stage待ちやFA3で複数msを削る具体的な仕事量を特定する。

測定へ進む条件：今回の通常Firmus試験はユーザーが明示許可。既存EP4数値gateと公式servingの品質・SLA eligibility・goodput・完了時間/ITL/受理長で評価する。先行kernel benchやASPIRE6500条件を達成済みとは記録しない。9235.28との単回比較を因果的speedupと断定しない。

内部の仕事量（時間の分解とは別）：

| 区間 | 仕事量/層/rank | 追跡する分母 |
|---|---:|---|
| FC1 | 804.098 GFLOP | 同runのFC1時間、実rowとtile row |
| FC2 | 402.049 GFLOP | 同runのFC2時間、実rowとtile row |
| 全expert重み | 660.603 MB | HBM cold readなら137.6µs、実cache traffic未確定 |
| FC2 partial出力 | 261.751 MB | 全てlocal HBMなら54.5µs相当 |
| combine read＋output write | 294.470 MB | HBM換算61.3µs、cache/演算は別 |
| FC2 remote出力 | 261.751 MB × remote率r | 片方向450GB/s換算 581.7×r µs |

rはこのrunでは未取得。旧routing fixtureの48.6%を現在の実測値へ代入しない。通信が計算に重なる部分は足し算せず、追加bufferのread/writeも必ず勘定する。

### Prefill FA3 target

causal QK+PV=2*L*(L+1)*Qheads*D; M128/N192 PackGQA。

有効演算 261.424 GFLOP/回、tile演算 274.811 GFLOP/回（1.051倍）、論理I/O 71.572 MB/回。1 sequence/rank。論理I/Oはunique Q/K/V/O+scale/LSEのみ。KV tile再読出しはL2/DRAM配分不明。

追う差：Tensor Coreだけの床に加えてsoftmax・FP8変換・転置・同期がある。全I/OをHBMへ流す模型も、全てL2 hitとする模型も実測帯域ではない。

次の作業：現在のN192についてM/N tile数、有効演算とpadding、softmaxのexp/変換命令、KV再読出し、fill/drain依存を数える。N224 NCUの命令数を流用しない。次の候補は削減命令または削減tileをnative compileで示してから測る。

測定へ進む条件：現在のN192で命令/trafficの減少があり、半精度expや多項式という失敗案と異なる根拠を示す。数値誤差と公式出力品質を検証する。

N192の条件模型ではKV tile訪問21,840回、softmax対象536.740M要素。tile単位のKV読出しは1.073GB相当で、unique I/Oの71.6MBとは別物。L2 hit率を無視してどちらかをHBM実測量と呼ばない。tile Tensorだけの床は181.5〜188.3µs。

各tile要素にscalar exp2を1回実行する模型では、[CUDAの16結果/SM/cycle](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#arithmetic-instructions)からexpだけで181.5〜188.3µs。同じ条件のTensor時間とほぼ等しい。これらは重なり得るので足し算しない。追加rescale exp、変換、ALU、転置、依存latencyを加えた現在の命令モデルは未完成であり、436µsとの差を単に『expが遅い』とは判定しない。

### Prefill QKV GEMM

2*M*N*K; M=3994, N=9216, K=4096; tile=256x128。

有効演算 301.537 GFLOP/回、tile演算 309.238 GFLOP/回（1.026倍）、論理I/O 127.726 MB/回。FP8 A/B + BF16 Cの各1回I/O。scaleと再読出しは別。

追う差：tile padding、scale、epilogue。

次の作業：M256のpaddingとepilogueを確認。FP8入力化は実装済み。小さい残差を理由に独立した全体測定をしない。

測定へ進む条件：既存選択より速い同shapeの候補だけを他の検証に同乗。

### Prefill O GEMM

2*M*N*K; M=3994, N=4096, K=8192; tile=256x128。

有効演算 268.033 GFLOP/回、tile演算 274.878 GFLOP/回（1.026倍）、論理I/O 98.992 MB/回。FP8 A/B + BF16 Cの各1回I/O。scaleと再読出しは別。

追う差：tile padding、scale、epilogue。

次の作業：FA3出力FP8化は実装済み。M256 paddingとepilogueを確認。

測定へ進む条件：同shapeでの改善と全体の電力・clock変化を照合。

### Prefill QKnorm RoPE

BF16 packed QKV read + FP8 Q write + contiguous KV write + paged KV write。

論理I/O 114.516 MB/回。既に融合済み。trig/weight/metadata/reduction/変換を除く。

追う差：BF16丸め保存、reduction、trig、FP8変換、複数出力。

次の作業：Q/K正規化・RoPE・FP8 Q・contiguous/paged KV書出しの既存融合を前提に、変換/アドレス/転送の命令量を減らす。入力はH4096ではなくQKV9216。

測定へ進む条件：head4/paired-roundの再試行はしない。現在の演算順を守り、追加loadやregister圧を含めて比較。

### Prefill add RMSNorm

2 BF16 reads + 2 BF16 writes + weight once。

論理I/O 130.884 MB/回。target代表shape。100 callsにはdraft等も含むため全体床へ掛けない。

追う差：残差を含む4本のBF16 read/write。33MBだけを分母にしない。draft由来の呼出しも混在。

次の作業：post-attention normからgate用BF16とMegaMoE用FP8を一度に作るconsumer契約を検討する。gate用BF16出力は残す。

測定へ進む条件：pre_dispatchとの組でBF16再読出し1回を削る。追加書出し/同期で相殺されないこと。

### Prefill norm quant

2 BF16 reads + BF16 residual write + FP8 Q write + FP32 scales。

論理I/O 115.035 MB/回。既に融合。入力/出力がL2に残る場合、この量はHBM量と一致しない。

追う差：2 BF16 reads、BF16 residual write、FP8 outputとFP32 scales、reduction。

次の作業：input add+RMSNorm+FP8 quantは既に融合。これを再度『融合余地』に数えない。

測定へ進む条件：既存融合より少ない命令/trafficを示せる場合のみ。

### Prefill pre dispatch

BF16 X read + FP8 X write + FP32 scale + topk ids/weights read/write。

論理I/O 50.229 MB/回。T3994代表shape。padding/metadata初期化を除く。

追う差：入力quantとrouting metadataの境界。

次の作業：post-attention normがMegaMoE対称bufferへ直接FP8/scaleを書く経路を設計し、routing metadata部分を独立させる。

測定へ進む条件：norm側に融合して減るBF16 readは約33MB/層。これを12ms削減と誇張しない。

### Prefill router gate

2*T*H*128。

有効演算 4.188 GFLOP/回。小N GEMM。logitsのI/O dtype/scaleは未確定。

追う差：小N128 GEMM、起動、scale/epilogue。

次の作業：gateが必要とするBF16入力を保存する。norm/dispatch融合の正当性の制約として追う。

測定へ進む条件：計算または中間bufferを消せる変更だけを検討。

### Decode MegaMoE

FC1=4*T*topk*H*I, FC2=2*T*topk*H*I。

有効演算 53.150 GFLOP/回、論理I/O 603.980 MB/回。32 expertのFP8重みcold readだけ。activation/scale/通信を含まない。

追う差：cold-weight時間は125.8usだが、activation/通信/epilogueを含む下限ではない。

次の作業：604MB/rankの重み読出しと、実際のHBM/L2 bytesを対応づける。次にpaddingとdescriptor/同期の依存を確認する。

測定へ進む条件：stage数やprefetchの再掃引を避け、実転送/依存の減少を示す。

内部の仕事量（時間の分解とは別）：

| 区間 | 仕事量/層/rank | 追跡する分母 |
|---|---:|---|
| FC1 | 35.433 GFLOP | 同runのFC1時間、実rowとtile row |
| FC2 | 17.717 GFLOP | 同runのFC2時間、実rowとtile row |
| 全expert重み | 603.980 MB | HBM cold readなら125.8µs、実cache traffic未確定 |
| FC2 partial出力 | 11.534 MB | 全てlocal HBMなら2.4µs相当 |
| combine read＋output write | 12.976 MB | HBM換算2.7µs、cache/演算は別 |
| FC2 remote出力 | 11.534 MB × remote率r | 片方向450GB/s換算 25.6×r µs |

rはこのrunでは未取得。旧routing fixtureの48.6%を現在の実測値へ代入しない。通信が計算に重なる部分は足し算せず、追加bufferのread/writeも必ず勘定する。

### Decode FA3 target

QK+PV=4*M*KVlen*Qheads*D; KV bytes=B*KVlen*KVheads*D*2。

有効演算 23.645 GFLOP/回、論理I/O 184.730 MB/回。44 requests, verify4, KV4100の条件模型。KVを各1回読む場合。実profileは可変batch/長さ。

追う差：KV read、softmax、tile数/並列仕事量、split後merge。

次の作業：request別KV長・verify rows・split2のKV読出しとpartial buffer bytesを追う。

測定へ進む条件：過去に数値不一致だったmerge統合は、誤りの特定とCPU/source証拠を先に必要とする。

### Decode QKV GEMM

2*M*N*K; M=176, N=9216, K=4096; tile=128x144。

有効演算 13.288 GFLOP/回、tile演算 19.327 GFLOP/回（1.455倍）、論理I/O 41.714 MB/回。FP8 A/B + BF16 Cの各1回I/O。scaleと再読出しは別。

追う差：小Mのwave不足とtile padding。

次の作業：M128の末端paddingと実row分布を追う。

測定へ進む条件：平均shapeでなく各graph bucketの条件一致。

### Decode O GEMM

2*M*N*K; M=176, N=4096, K=8192; tile=64x96。

有効演算 11.811 GFLOP/回、tile演算 12.986 GFLOP/回（1.099倍）、論理I/O 36.438 MB/回。FP8 A/B + BF16 Cの各1回I/O。scaleと再読出しは別。

追う差：小Mとepilogue。

次の作業：M64/N96の末端paddingと実row分布を追う。

測定へ進む条件：他の検証に同乗する小さい改善候補。

### Decode add RMSNorm quant

2 BF16 reads + BF16 residual write + FP8 Q write + FP32 scales。

論理I/O 5.077 MB/回。既に融合。入力/出力がL2に残る場合、この量はHBM量と一致しない。

追う差：小MでHBMを埋められずreduction/launchが支配し得る。

次の作業：現行の融合でも約9usある。HBM床だけで評価せず、CTA数とreduction依存を数える。

測定へ進む条件：Prefillのwide-row結果をDecodeへ流用しない。

### Decode verify split merge

split2 FP32 partial O/LSE read + FP8 O/scale write。

論理I/O 13.111 MB/回。verify176行。実call shape混在、metadata/reductionを除く。

追う差：partial reduction、FP8出力scale、起動。

次の作業：FA3 split2のpartial FP32入出力を実shapeから勘定し、attention本体との合計で比較。

測定へ進む条件：merge単独の削除効果をattention本体の増加と相殺する。

## 優先順位と不足する証拠

1. Prefill MegaMoE：142ms/周期が最大。旧40057117293のFC1/FC2をrawから分解し、同callのclockで演算比を訂正した。上記表をこのserving profileへ差し引かない。未統合だったFC2 address bundleを現行D5aと組み合わせて検証する。source rank別remote行数が無ければ配置の得は未判定。hot4複製は既に有効。

2. Prefill FA3：41ms/周期。N192のtile仕事量と有効演算の差を分離した。Tensorだけの床との差にはsoftmax/FP8変換/転置/同期がある。旧N224 NCUのpipe比率を現在の内訳へ転用しない。

3. post-attention norm→MegaMoE入力：gate用BF16を残しつつFP8/scaleを直接書くと、BF16再読出し1回を削れる可能性がある。3994×4096×2=32.72MB/層、94層で3.08GB。4.8TB/s換算0.64ms/周期はbyteだけの参考値で、予測短縮ではない。12.3ms丸ごとの削減案ではない。

4. Decodeも同時に追う：MoEの重み604MB/層、verifyのKV読出しを別の上限として記録。Prefill供給だけを増やすとITLが悪化し得る。CPU起動間隙の除去だけで大幅改善とは見積もらない。

不足：現行FP8 draftのkernel profile、同runのMoE subphase、実HBM/NVLink bytes、rank別routing、N192のopcode/依存内訳、Decodeの各callのKV長。これらは0ではなく未確定。既存成果物で解ける項目から埋め、採否が変わらない測定は増やさない。

## 過去の結果

| 候補 | run | 状態と結果 |
|---|---|---|
| FC2 address/guards/metadata/pair bundle | 40116925551 | no_incremental_serving_improvement_on_current_base: Old unpaired8922.77 versus8854.80 did not transfer: current normal run40653531712 is9234.34 versus9235.28 (-0.0102%). No serving improvement observed. Do not repeat without new evidence. |
| Prefill independent WG | 40069880580 | rejected: +22-36% kernel latency; extra TMA/stage pressure |
| Prefill bulk scatter | 40095847738 | rejected: +1.3-4.9%; see broad-ceiling-review before any repeat |
| FP32 precombine | 39355994503 | rejected: serving -5.8%; added scratch traffic |
| hot8 replicas vs hot4 | 39462905249 | rejected: all three recorded layers slower; hot4 already adopted |
| combine direct store | 40459750245 | rejected: numerically exact; did not pass control-paired speed gate |
| QKV head4 | 40459750245 | rejected: 1.1-1.4% improvement below predeclared 3% gate |
| QKV paired rounding | 40463471079 | rejected: about5% slower |
| FA3 half2 exp | CPU compileのみ | rejected_without_gpu: native compile emits two MUFU; conversions/spills increase |
| FA3 polynomial exp | 40472152140 | rejected: numerical gate passed; about5.2% slower |
| router early first response | 40528222753 | not_adopted_for_goodput: TTFT -201ms; completion -0.9ms; ITL +1.006ms; score9161.67->9135.42. NIXL version confound. |

## 更新手順

- 候補を変える前にkernel名、shape/dtype、source/image、FLOP/bytes/命令の何を減らすか、過去の棄却理由を更新する。
- 未検証の改善幅は成功条件として記載し、実測欄には入れない。
- 実験後は同shapeのcontrol/candidate、clock/電力、数値品質、normal goodput・完了時間・ITL・受理長を追加する。失敗も残す。
- profileは通常benchと別に扱う。更新は既存のジョブに必要な観測を同乗させる判断を優先する。

9500は旧通常score9161.6688から約3.69%増が必要。仮にPrefillだけが律速し他条件が固定なら、251msから約242msへ約9ms削る規模になる。これは必要条件の逆算で、達成予測ではない。routerで先頭だけ約201ms早めても完了時間は約1msしか変わらなかったため、今後の採用は完了時間とgoodputで判定する。

## 再現

```sh
nsys export --type sqlite --output /tmp/prefill-0.sqlite /srv/qwen-artifacts/40310899466/1/profile/profile/nsys-prefill-0.nsys-rep
python3 research/runtime-performance/extract-kernel-profile.py /tmp/prefill-0.sqlite --role prefill --run 40310899466 > research/runtime-performance/kernel-profile-40310899466-prefill.json
nsys export --type sqlite --output /tmp/decode-0.sqlite /srv/qwen-artifacts/40310899466/1/profile/profile/nsys-decode-0.nsys-rep
python3 research/runtime-performance/extract-kernel-profile.py /tmp/decode-0.sqlite --role decode --run 40310899466 > research/runtime-performance/kernel-profile-40310899466-decode.json
python3 research/runtime-performance/render-kernel-theory.py --write
```

GPU計算なし。元nsys-repのhashはmodel JSON、export SQLiteと抽出scriptのhashはprofile JSONに保存。窓・呼出し数・重複・shape仮定を変更した場合は旧値へ上書きせず、差と理由をGitに残す。
