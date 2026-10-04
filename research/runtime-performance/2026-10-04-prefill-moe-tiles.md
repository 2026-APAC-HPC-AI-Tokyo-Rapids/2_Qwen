# Prefill MoE kernel最適化の続行（2026年10月4日）

ユーザーのカーネル最適化続行指示に従い、既存precisionのままSM90 CUTLASS tile/cluster/scheduleを限定比較する。実装branch `experiment/prefill-down-tiles-20261004`、成功同imageoff対照b557fcdを親にする。組織mirrorは読取専用、consumer-owned extensionとsourcepatchで実装。候補はdefaultoff・P/Hopper/EP1/BF16出力・QwenTP4 shape限定、D/EAGLE3 draft/FP8KV/top8/targetverify不変。production small expanded rows<=2048・T0・形状外はwheelを保持する。

## down＋weighted reduction融合を優先しなかった理由

T8192/top8/H4096のdown BF16中間は512MiB。stockは512MiB store＋combineで512MiB read＋最終64MiB write（計1088MiBの論理traffic）。down nativeはFP32 accumulatorをBF16へ丸めてから保存し、combineはBF16 router weightとともにFP32でtop8を加算する。標準CUTLASS TMA epilogueのaffine grouped row-major storeはc_mapによる非affine scatterとexpert間8-way競合をそのまま表現できない。

FP32 atomic融合はBF16丸めをepilogue内に残して精度を維持できる可能性があるが、8回のFP32 read-modify-writeが論理2GiB、128MiB accumulatorのzero＋final read/BF16 writeが320MiBとなる。実HBM量はcache次第でこの論理上限と異なるが、512MiB中間を消すだけで帯域改善を保証できない。token-owned persistent kernelは競合を避ける一方、expert別weight再利用を崩す。今回これらはCPU実現性評価で留め、将来不可能と棄却せず、短いK384のepilogue/scheduleに適した既存mainloopのgeometryを有限比較する。

## 最小native構成と有限候補

固定SGLang fc275674のSM90 launcher/helper、CUTLASS57e3cfbを用いる。現在buildはsglang-kernel wheelをdownloadするため、AOT sourceのpatchだけではnative変更がnoopになる。必要launcher/helperだけをtorch CUDAExtensionでbuild時sm_90a限定compileし、prebuilt.soを通常Python importする。compute nodeでcompileしない。dtype/shape/physicalstride/device/metadata連続性をguard、workspaceはget_workspace_sizeと照合、packed stride/layoutはcompile-time sizeof検査。FP8重み/活性、FP32 scales/accumulator、BF16 identity epilogueを保持し新しい低精度化を加えない。native source・buildrecipe・CUTLASS revision・.so SHA256を保存し、probeで実modulehashを検証する。

| 条件 | M×N×K tile | cluster | schedule | 対象 |
|---|---|---|---|---|
| stock clone |128×128×128|1×2|cooperative|up/down|
| down C1 |128×128×128|1×1|cooperative|K384/N4096|
| down N256 |128×256×128|1×1|cooperative|K384/N4096|
| down pingpong |64×128×128|2×1|pingpong|K384/N4096|
| up pingpong |64×128×128|2×1|pingpong|K4096/N768|

全候補は既存CUTLASS mainloop/APIと同じprecision。up64は元H20LargeKのgeometryをH100の限定shapeで検証するもので、M padding/weight再利用とAtraffic/occupancyの回帰を実測する。downもGEMM計算そのものの変更ではない。stock cloneをwheelと先に照合し、compiler/library差をtile gainへ混同しない。期待gainは未確認、理論peakからの差を回収可能なgainとみなさない。

## 一buildの測定と品質条件

同allocation採点前にtile probe＋既存paired FP8/scale row-shuffle融合probeを有限取得し、その後全candidateoffの通常c48対照を公式採点する。runtime探索やmode自動採用は行わない。tile probeはT8192 balanced/skew、T4096 skew、T32独立oracle（small native強制は外部probeのみ）。wheel/clone/down3/up1を各global graph captureし、5round×20replayをrotate/reverseして中央値・範囲を保存する。eager/host wallを分離。modelの実expertM分布ではない合成routecountsを明記する。

stock clone large shapeはrelativeL2<=1e-3を必須とし、bitwise一致は診断併記。candidateはrelativeL2/maxabs/maxrel/finite、T32独立BF16/FP32参照に対するbaseline envelope、graphの同検査を保存する。relativeL2>1e-3は品質未承認の注意flag、0.1閾値は粗大誤りscreenに過ぎない。丸め順序差を許容しつつ、明白な品質悪化は採用不可。GPU illegal access/未知fatal後に測定を続けず、can_implement不成立の候補だけは理由を記録する。局所で有望なgeometryのみ追跡conditioncommitで固定し、同newimageoff/on公式E2E＋必要反復、実routingとgreedy品質を確認する。

CPU218tests、actualP/Dargvfixture、Ruff/bash/5patch ASTと独立reviewでcheap gateを先に実施。localnvccが無いためnativecompile・can_implement・CUDAgraph・実品質/速度はまだ未確認、初回remotebuild/probeで検証する。raw/派生正本とIssue17へrun識別子・採否・cleanupを追記し、投入だけで完了としない。

## 初回build失敗と最小修正

実装ffccc09/run[37172308998](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37172308998)は2026-10-04 02:54:51 UTC、固定fc275のprivate cloneでfp8.py:224のpatch適用checkに失敗。Docker/native compile未到達、PBS未投入、GPU消費なし。複数patchを一括`git apply --check`すると後続tileが依存するES追加contextを元sourceへ照合し失敗した。前CPU検証は逐次check/applyでありactualscriptと異なる手順を検査していた。offset変更ではなく、actualscriptを各patch順のcheck→applyへ修正。固定FCで旧失敗と修正版全5patch/AST成功を双方再現し、actualloopを抽出して依存5patchへ実行するfixtureを追加した。

別途、直接Docker RUNする新compiler recipeのmode100644漏れも100755へ修正し、全COPY対象shellの実行bit検査を追加。219CPUtests/Ruff/bash/diffと独立review通過。修正[6ea0e6b](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/6ea0e6b38272f2d5cc4fe1eb85edda0bb52b54ea)を前runterminal後pushし、再build[37172734078](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37172734078)を開始した。候補・serving条件は不変、native/GPU結果は未確認。

## N256 compile不成立の除外

修正版37172734078はsource installまで成功し初めてnativecompileへ到達したが、WideN mode2だけが固定CUTLASS `fp8_accumulation.hpp` のAccumulator/scaleB同形assertに違反。instantiationではaccumulator `((2,2,32),1,1)`、scaleB `((2,2,(16,2)),1,1)`でnested形状が異なる。ScaleNsPerTile>1の分岐は存在するため、128scaleでN256が一般に不可能という結論ではなく、このstockScaleConfig/MMA64x256の既存layout組合せがcompile不可という限定判断である。[固定CUTLASS accumulation](https://github.com/NVIDIA/cutlass/blob/57e3cfb47a2d9e0d46eb6335c3dc411498efa198/include/cutlass/gemm/collective/fp8_accumulation.hpp)、[blockscale mainloop](https://github.com/NVIDIA/cutlass/blob/57e3cfb47a2d9e0d46eb6335c3dc411498efa198/include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp)参照。

新scale256への変更やlayout改造を追加せず、[77e059e](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/77e059e489bce0b9273af6f736b61f32da1aa334)でmode2のnative/Python選択/manifest/probeを除外。上表N256はcompile不成立の履歴であり、実測対象はstock clone＋down C1＋down/up pingpongの5variant（wheel含む）へ減った。残configはN128を保持しFP8scale128/FP32acc/BF16outputを変更しない。219CPUtests・actual5patch/AST・独立review通過後、前runterminal/PBS無しを確認し再build[37173471321](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37173471321)開始。残configのcompile/graph/性能は未確認、GPU未消費。

再build37173471321はnativecompile/link/importに成功。actualnvccはsm90a/CUTEpacked/CUTLASStensorcore/O3/C++17、sourceSHA17fb2b4ed82cc4e5a2f80790fc1587e5fdc0b77752fd8f924e1f645f29801a41が77e059eと一致、moduleSHA5acc92c40938a30990a93c957a2cd8169f554c983955315c94c0e754c0765496。新imageSHA6842ee6576d80d26ee239ce6ed25519402ecdf9f5b9f2cc13b658f63bf3b64ff/12231323648 bytes、sidecar77e059e+fc275一致。PBS216125は固定DGX026/030で開始（03:35:52 UTC）、profilefalse。局所probe/通常off E2Eの結果は未確認。

## 初回局所結果（通常対照のterminal待ち）

# 37173471321 有限probeの独立所見

live read-only JSONを解析。terminal canonical rawのhash照合はaudit担当。公式off E2Eは進行中であり、以下は採用性能ではない。

stock cloneはT8192balanced/skew・T4096skewでwheelとのeager/graph relative L2=0、bitwise同一。graph中央値比は0.999859/0.999779/1.001329。移植/ビルド差は今回のfixtureでほぼない。全4shape×5variantsでgraphと精度gate成立、candidate numerical attention flagなし。

| tile候補 | T8192 balanced対wheel時間差 | T8192 skew | T4096 skew |
|---|---:|---:|---:|
| down C1 | -0.174% | +0.174% | +0.305% |
| down Pingpong | +4.513% | +0.892% | +1.918% |
| up Pingpong | +6.035% | +5.166% | +5.726% |

downC1の五round paired差はbalanced[-0.248,+0.076]%、skew[-0.063,+0.471]%で一貫する勝者ではない。Pingpongは全large shapeのpaired roundsが正（遅い）。T32はnative large geometry強制のoracleでproductionはsmallM wheelを保持するため、serving速度の比較対象にはしない。全MoEgraph時間でありup/down個別GEMM改善率には逆算しない。tile candidate E2Eを追加する根拠なし。

paired shuffleはT8192 stock中央値0.203475ms→fused0.167149ms、17.853%短縮（36.326us/call）。stock範囲[0.203357,0.203814]、fused[0.167069,0.167434]msと全五sample分離。T4096は22.428%、T32は36.095%短縮。rawbytepatternsとquantize-once inputの値/scale bitwise一致、graph capture/replay、T0 empty契約が通過。測定順はstock五sample後fused五sampleでtileのrotate/reverseとは異なり、順序交絡の限界は残るが差はsample内の揺らぎより十分大きい。

旧profile1569callsへ単純適用すると約56.996ms（active窓1.139%）の局所削減感度。ただしisolated cached copyと実MoEのcache/occupancy/メモリallocationは異なり、instrumented profileと通常E2Eも異なる。これをscoreの期待gainとは言わない。

**推奨:** tile全offを保持し、同newimageのoff対P-only fused shuffle enableという単要因の非profile公式E2Eへ進む。局所勝者はshuffleのみ。採用は公式parser/品質/greedy/来歴と必要な反復の確認後、単runで改善を確定しない。

root判断によりtile candidate E2Eは見送り。shuffleのみP環境flag1、tile/ES/NVLSoff・D/c48固定の同image候補をCPU準備、actualargvを含む59targetedtestsと独立review通過。offterminal全監査後だけ追跡commitpushする。採点前finiteprobehookは候補で除去し条件差に記録。既存helperには分岐発火logが無いためactualPenv/sourceguard/batchshape照合を確認し、直接kernel名による選択証明はnonprofile未観測という限界を保持する。

## off監査とshuffleonly E2E投入

37173471321はActions/PBS216125成功（11:56）、正本54hash/公式制約parser再実行一致・eligible、score2681.82392469891/good98.9202%、errors/cancel0・TTFT不適格18件。実worker4設定、同image/source、2probe liveと正本byte一致、前offとの8greedy token/logprob完全一致を確認。runtime削除/qstat空。これは新build全off対照でありtile/shuffle改善の値ではない。

[8505eae](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/8505eae300dbb643d189e8f4401a8b5c47349e0e)でP2workersの`SGLANG_PREFILL_FUSED_FP8_SHUFFLE=1`だけを有効にし、同image再利用の[37175480138](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37175480138)を投入。D/c48/TP4EP1DP1PP1/fixed026+030/profilefalseとtile/ES/NVLSoffを保持。既に完了した採点前finiteprobehookは除去した条件差を記録。worker actualcommandをlaunch前に保存する2linesを追加、bench observer/source変更なし。offの環境command記録は欠落しtrackedsource/fixture/workerinfo/defaultoffから裏付けるという限界がある。219CPUtestsと独立review通過、候補E2Eが有望なら同imageoff/on追加pairで確認、局所gainを採用値にしない。

追加runtime探索screenではcontext8192によりgraphmaxは8192 clampされchunk16384だけでは16384graphを作れず、追加buffer/peakも未確認。c44→48でPqueue増/Dactiveほぼ不変だったためc52の根拠は弱い。現在はchunk16K/c52を混ぜず、融合の明確なE2E改善後だけ別単要因候補として再検討する。

## 初回shuffle E2E独立比較

# shuffle on/off 初回E2E独立比較

同image off37173471321対on37175480138。正本rawをread-only比較。source来歴・全hash・terminal・parser監査はaudit担当。非profile公式c48、P-only fusedshuffle以外precision/role/tileESoff等を固定。offは起動前finiteprobeあり、onでは除去しargv証跡を追加した条件差がある。

| 指標 | off | on |
|---|---:|---:|
| official good tok/s |2681.8239|2715.4992 (+1.2557%)|
| total output tok/s |2711.0573|2744.7800 (+1.2439%)|
| requests / jointbad |1667 /18|1685 /18|
| good fraction |98.9202%|98.9318%|
| TTFT mean / p95 / p99 ms |1565.04 /1825.80 /3109.34|1510.34 /1734.51 /3065.85|
| ITL mean / p95 / p99 ms |9.7168 /11.3216 /12.0871|9.7989 /11.3875 /12.1011|
| client effective P / D concurrency |21.2149 /26.2125|20.7281 /26.7627|
| sampled P queue pooled |1.7875|1.6377|
| sampled D acceptlen |2.22980|2.22669|
| sampled full8192 batch |95.2381%|96.1353%|

TTFT平均3.495%低下、Pqueue約8.38%減、実効D並列約2.10%増はP待ちの改善と整合する。平均ITL0.844%増は同時decode増と両立するが因果確定ではない。jointbad両18、errors/cancel0、OSL mismatch0。greedy8件すべてchoices（content/tokens/logprobsを含む）がoff/on一致、prompt hash一致。独立タスク精度評価の代わりにはしない。

Pqueueは各worker1.776/1.799→1.612/1.663と両側低下。D log active15.103/15.112→15.419/15.581、queue0。転送待ちreq7.936→7.742は複合待ち件数でネットワーク時間ではない。log統計は無加重sampleで時間重み付きoccupancyではない。Pinput-throughput平均は片側だけ上昇し反対側は低下、ログで孤立したサービス時間の明確改善を確定しない。

**判断:** microのbitcopy品質と17.85%copy短縮、初回E2Eのscore/TTFT/Pqueueの方向は有望。同image off/on一対の反復は合理的。ただし初回1pairだけでは採用を確定しない。Pサービスkernel直接測定もなく、c52を混ぜない。反復が中立・逆方向なら初回+1.256%だけで改善としない。候補tileは全off維持。

## main採用の依存範囲（反復前の案、変更未実施）

shufflepatch単独のcleanFC適用はCPUで成功し、形式上SiLU/ES/tileへ依存しない。既存quantを一度だけ行ったFP8値/FP32scale bitsをgatherするTritonhelperであり、SM90 nativeextensionは不要。tile用module/buildrecipe/選択patchを撤去してもshuffleの意味は保てるが、image/sourceが変わるため新buildと通常E2Eが必要で、現imageの測定値を新imageの保証にできない。

現mainは未patch source/c40、測定baselineはSiLU融合/drafthead選択/c48/Dgraph13,15等を含むため、shuffleだけmain onにして同測定条件と扱わない。丸ごとbranch mergeも不採用ES/tile/native/probeを持込むため不適切。最小費用案は測定済branch/imageを実行・再現正本として保持し、反復成功ならその条件での限定採用を文書化。mainでsource再構成する時はbaselineの採用前提を個別整理し、SiLU/drafthead（必要ならc48/Dgraph）とshuffleのみへ縮小、ES/tile/nativeextension/probeを除去する。次の別理由sourcebuild機会にまとめれば追加buildを抑えられるが、新imageoff/on通常E2Eは省略できない。反復前にactualmain変更は行わない。

初回onは52hash/parser/settings/PBS0/11:55・actualPflag1とDにPenv無し・8greedy対off token/logprob一致・runtime削除/qstat空を確認。Actions全体success、非対象Genkai選択のみSSH133.5.4.132:22切断→tarnotarchiveexit2の独立障害を記録。反復off[7c1bbb7](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/7c1bbb75108aadeb041d6cb6f1ef91e38dc97170)/[37176596065](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37176596065)を同image・no probe/actualargv双方保持の条件で順次投入、反復onはoffterminal監査後に別条件commitで実施する。

反復off37176596065は全52hash/parser/actualPflag0・D不変/8greedy/PBS2161360（11:57）/runtime削除・qstat空監査通過、公式2677.3603631583346（初回off比-0.166%）。続く反復on[62d9f91](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/62d9f91c8694fb6e0a79d3f812c10628f5a7cda8)/[37177667936](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37177667936)を同imagefixedpair/c48/no probeで順次投入。branchheadはon条件へ戻し、今後の運用候補正本はexperiment/prefill-fp8-shuffle-repeat-20261004@62d9f91（反復結果未確定）であり、off7c1bbb7は対照として区別。mainruntime/sourceは保留、次sourcebuildで不要ES/tile/module/probeを撤去する方針をrootが採用した。
