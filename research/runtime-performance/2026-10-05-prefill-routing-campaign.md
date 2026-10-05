# Prefill routing helperの部分置換

更新日2026-10-05 JST。ユーザーの実測高速化要求に対し、[独立設計review](2026-10-05-next-moe-routing-review.md)を実装へ進めた。既CUTLASS gridとstage4確認は事前gate未達のまま、GEMM候補を採用しない。既shuffle最終反復37177667936/PBS216140は後述の監視接続障害で未GPU実行のままworkflow cleanupが取消した。第四runの性能は未確認。現在Lunaは新routing-off control37319361445を監視する。

## 実装と条件

実験branch `experiment/prefill-routing-20261005`、source/build commit `9c5a5a371524304bbab960cd8d3d86d67de56adb`。expertごとの全route走査をB512のtile histogram、expert/tile整数exclusive prefix、tile-local shared cursor scatterへ置換する。独立CUDA 1 TU・3整数kernelのみ、MAX_JOBS=1。GEMMテンプレートを再compileせず、既wheel GEMM、FP8 scale128/quantize once、BF16中間丸め、FP32 top8集約、target verificationを維持する。P/Hopper/EP1、BF16入力、E128/top8/H4096/I384/T≤8192に限定し、default off。T0はwhole MoEを早期returnし、helper単独ではmetadataをresetする。

新imageでは私有cloneへの適用順をSiLU→draft selected-head→FP8/scale shuffle→routingの4patchに固定。不採用ES/tile patchと大きいnative tile extensionはbuild対象から撤去した。source cleanupを性能gainと混同しないため、以後のoff/on比較は同じ新imageを使う。D/既c48/FA3/FP8 KV/EAGLE3 BF16 draftは維持する。

旧profileのhelper計154.587ms/5.004391秒（3.09%）は全消去感度上限。論理IDs読取の削減はL2/HBM実trafficの削減量を証明せず、実改善は未測定。

## 検証と進行

274 CPU tests、routing/probe Ruff、bash syntax、actionlint、actual依存patch chainと固定source AST fixture、Docker呼出script実行bit、diff checks通過。独立review GO。CPU検査はnativecompile/GPU正しさの証明ではない。

[build-only Actions 37218171768](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37218171768)を投入。`[require-build] [cluster:aspire2ap] [build-only]`でGPU計算は起動しない。first compiler fatal、module import/source/.so manifest、terminal archiveをLunaが監視する。native compileとimage来歴はまだ未確認。

build成功後、別runtime commitで同imageのstock/tiledをscheduler経由1GPU・15分walltime・720秒timeoutの有限microprobeで測る。allocated GPU UUIDのみmount/監視、active clock/powerを1秒CSVに記録する。モデルstaging・公式score生成は行わない。

事前gateは8192 balanced/skewのwholeMoEそれぞれ中央値ratio≤0.995かつ20paired round中15勝以上。heldout4096skew/8192hot-only/65tailは中央値退行無し。全eager/graph finite/relativeL2≤1e-3、bitwise差診断、T32独立BF16/FP32 oracle envelope、map逆写像/offset/count、同graphでIDsを変えたreset、T0を必須にする。候補2・training2shape・heldout3shape・oracle1shape、最大404測定/40400timed replay、交互順序と連続≥200ms warmupで有限終了する。synthetic routeは実expert分布ではない。micro有望時だけ同新image・shuffleon・routing off/onの公式E2E比較条件と費用をrootへ報告する。自動採用・実行中探索は導入しない。

## helper build成功

37218171768はterminal success。source `d958b6ce7ea382c9ac42a8ebf8f4f6c4b68a781900129e15aa9fa650f2cb38ce`、build Python `ba9dd52211441c69529ce76d712a520c4204f71a3a76fa287a480a23e4094c8e` はcheckoutと一致。module `620e625331d74985538fbef04f077daf299b5af7eb9d97b2e25e2bd608666e1d` がlink/import済み。ptxasはscatter32/prefix28/histogram18 registers、各512B shared、1TU/3kernel/MAX_JOBS1を確認。EC2 elapsed1568秒、image12231192576 bytes、S3/全cluster archive成功、GPU/PBS無し。

条件のみの空commit `4b34df8` はpaths-ignore付きpush workflowを生成しなかった。GPU消費無し。PBS job/runtime名をrouting専用にした実runtime commit `f041fe1` で[finite 1GPU run37220486463](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37220486463)を起動した。source/image/gateは変更しない。公式E2E性能改善はまだ未確認。

## 初回probe起動失敗

37220486463/PBS216326はDGX039の1GPUで8秒、Exit1。最初のfatalはprobe main43がtorchより先にnative extensionをimportし、`ImportError: libc10.so: cannot open shared object file`。build時はtorchを先にimportしてmodule検査が通過しており、probe import順を最小修正する。新build不要。JSON/性能値は生成されなかった。image正本SHA `182fbb6df7f941aa688a393dfc181bb269f3112a6d34c33f6007ddc5d09f0a42`、raw正本 `qwen-artifacts/37220486463/1` の13hash一致、runtime/self-job cleanupとhome archival成功。初回probeは性能比較に使用しない。

## 有限probe成功と公式比較開始

Torch先行importへ修正した `1781bc9` / [37221346131](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37221346131) はActions/PBS216329成功、wall6:58、DGX038のallocated1GPU、正本14hash一致・runtime cleanup/homearchive成功。事前gateを独立再計算してPASS、全wholeMoE eager/graph bitwise一致・finite・relativeL2=0、map/reset/T0/oracle完備。

|shape|helper stock→tiled ms|wholeMoE stock→tiled ms|whole短縮|勝数|
|---|---|---|---|---|
|8192 balanced|.107728→.032630|1.329936→1.249666|6.0356%|20/20|
|8192 skew|.102588→.031703|1.354264→1.277219|5.6891%|20/20|
|4096 skew heldout|.057165→.020120|.806483→.755452|6.3277%|20/20|
|8192 hot heldout|.102524→.031538|1.293173→1.227552|5.0744%|20/20|
|65 tail heldout|.010822→.008815|.259699→.257781|.7385%|20/20|

helper約69–70%とwhole約6%を区別し、GEMM単独TF/s改善とは呼ばない。合成routing・同run paired比較であり実expert分布/E2E gain未証明。active util>0 258 clock samplesはSM1770–1980MHz、1秒samplerではvariant別clock完全一致を証明しない。CSVtimezoneはallocation日時とのUTC+8推定対応。旧profileからの3.09%消去感度をこの別GPU/別測定windowの厳密gain上限にはしない。

同image182fbb6dで [routing-off公式control37222559966](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37222559966) / `4cdeca6` を投入。Pshuffle1/routing0、D/c48/fixed026030/topology/precision維持、microhook/profile無し。off terminal監査後routing1のみを変更して候補を順次比較する。新build不要、通常pair約24分/6.4GPUh、queue別。公式score・8greedy・TTFT/ITL/Pqueue/Dacceptlen/全hash/cleanupまで確認し、単pairで採用を断定しない。

## 公式controlの待機中接続障害と再投入

37222559966/PBS216333は約12時間Q。最後の正常観測はQ、source/rawにはstime/resources_used/Exit_statusもscoreも無く、GPU実行無し。first failureは2026-10-05 06:37:26Zの監視SSH timeout124。06:41:10Zの再timeout、06:47:24Zに連続接続障害budget300秒を超えてmonitor終了。Q待ち全体deadline/PBS execution walltimeの失敗ではない。その後cleanupが自分の未実行PBSをqdel、raw回収時run-success欠損は二次障害。正本9hash一致、postcleanupPBS F/terminated、runtime削除済み。公式性能値無し。

旧shuffle37177667936/PBS216140も同じ接続timeout124→cleanupによる未起動取消、stime/Exit/walltime無し、正本9hash一致。最終shuffle反復は未検証のまま、既3runの所見を広げない。ユーザー/agentによる健全jobの手動取消は行っていない。

最小infra修正main1988b2d：最後の確定PBS stateがQの場合だけ連続接続障害budgetを1800秒へ、R/unknownは既存300秒、各SSH180秒/PBS30分walltime維持。Q待機そのものの上限は追加しない。Q124/255障害→復旧R→再124でbudget300へ戻るfixtureをmain65940aaで明示確認、monitor7/contract15 testsとRuff/bash通過。runtime-onlyでnewbuild不要。

[新公式off control37319361445](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37319361445) / `77dab97` を同image182fbb6d・Pshuffle1/routing0・c48/D/fixed026030維持で再投入。Luna collab監視を復帰できたためR/score/terminalでSolをfollowupする。旧CLI監視は古いQ時点で終了しており終端引継ぎを果たさなかったので、現在の状態判断には使用しない。現在liveは新offのみ、公式onはまだ未投入。Goalは未完了、off terminal/全監査後on→公式比較まで継続する。

### 再投入のscheduler照合

新off37319361445のPBSは216741、runtime `pd-37319361445-1.RVNTC8TmPY`。Q/aiq2、固定026030各8GPU/112CPU/1880GB、comment `Not Running: Not enough free nodes available`。qstat-fにestimated.start_time/ETAは表示されていない。runtime provenanceの77dab977・image182fbb6d・PROFILEfalseを照合。通常tomokazu shellとscheduler surfaceを区別し、`ssh -o BatchMode=yes archive-host "~/actions-runner/shell/aspire2p-shell '/opt/pbs/bin/qstat -u ACCOUNT'"` でuidACCOUNT/optpbs/scratchを確認した。Lunaへこの具体経路を渡しQ5分/R60秒監視を再確認。観測経路の不一致をjob失敗/再投入理由にしていない。

## 同image公式onを先にqueueへ投入（10月6日）

長い固定pair待ちを再び挟まないため、off終端後にonを投入する従来の順序を変更し、既定のoff/on一組だけを先にqueueへ置く。runtime条件commit `934636c5e7417a603bdd0239096cf32177a6a781` / [on Actions37389659143](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37389659143) / PBS216810、runtime `pd-37389659143-1.vJDVyvtpkw`。Pの`SGLANG_PREFILL_TILED_ROUTING=1`と対応fixture/commentだけを変更し、Pshuffle1、D、c48、precision、profilefalseを固定した。27 integration tests/bash syntax/diff checksを再確認した。

onのASPIRE image sidecarとruntime provenanceでimage182fbb6d、size12231192576、buildsource9c5a5a3、SGLangfc275674、commit934636c5を照合し、offとserving JSONがbyte一致することを確認。EC2 build/S3/mirror preparationはskippedで新build無し。PBS216810もoff216741と同じ固定026/030各8GPU・112CPU・scheduler1880GB、place scatter:excl、wall30分を要求してQ。二組を同時計算せず、同じexclusive pairで順次実行となる。ただしschedulerの実行順序は保証しない。既計画の2run予算内で追加反復無し、健全off取消無し。onのqueue投入だけでoffとの優劣や精度を判定しない。

Lunaが両Actions/PBSを監視し、R/terminal時にSolが独立監査する。off失敗時は最初のfatalを解析し、有効対照が無いonの性能を断定しない。terminal、repository parser（公開規則準拠で主催者提供ではない）、TTFT/ITL、8greedy、actual4worker argv、全rawhash、runtime消去を両runで確認する。整数routingのmicrobitwise期待を維持するが、E2E logprob exact差はtoken位置・数値差・原因の診断として記録し、ユーザーが許容する丸め変更を自動不採用条件にしない。主催者の完全accuracy評価と単pairの採用判断は別途未確認。
