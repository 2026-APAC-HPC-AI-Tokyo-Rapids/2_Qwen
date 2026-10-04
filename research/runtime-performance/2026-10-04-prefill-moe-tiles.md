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
