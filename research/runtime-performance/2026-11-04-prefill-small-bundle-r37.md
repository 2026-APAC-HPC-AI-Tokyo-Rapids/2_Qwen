# Round 37：FC1 first-K の固定化と小案 bundle の配備停止

FC1 first-K を固定設定へ移した。A/B/C の r36 CPU probe は新規 compile で再現したが、**3案の通常 Firmus serving 投入準備は未完了**。A の独立 flag を、同じ dense template・既存 launch 条件のまま JIT source に渡すには、image 内の DeepGEMM host extension の変更が必要だった。「image rebuild が要るなら理由を書いて止める」という今回の指示に従い、bundle の配備を停止した。B/C だけの試験へ変更していない。新候補の production overlay、独立 flag、in-job gate、r37 prewarm/dispatch 回収は追加していない。

親 main は `6813d21`、作業 branch は `perf/pp4-r37-small-bundle-20261104`。image pin は `3a47b757bb88f6bc14ad5f151e8405f645007f25f381f631d88046d9bd1b7b04`、build 41729881283 のまま。SSH、Actions 操作、Slurm/PJM 投入、GPU 実行、image build、submit commit は行っていない。mirror、`~/.codex`、`/tmp/qwen-*`、他 worktree は変更していない。

## 確認した根拠と停止理由

AGENTS.md、r36、r32〜34、r29 の runtime overlay の記録、tracker/model と入力 profile JSON、関連 script/patch/test、commit 本文、退避済み Codex 会話と r32/r36 CPU proof を読んだ。Actions の証拠は既存の `/srv/qwen-artifacts/41767886175/1/results/results/` の selection/dispatch と研究記録を読み取り専用で確認した。外部 Actions API は使っていない。

`stage-qwen-overlays.sh` は、image からコピーした SGLang package と DeepGEMM site-packages を patch して mount する。B の `rope.cuh` とその JIT host dispatch、C の Python/Triton はこの範囲に収まる。一方、A の dense JIT source を作る `SM90FP8Gemm1D2DRuntime::generate_impl` は `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp` にあり、既に host extension へコンパイルされている。これは site-packages 内の device header ではない。

実 API の `fp8_gemm_nt` は `fp8_fp4_gemm_nt` の alias で、Python binding に任意の template flag、macro、epilogue 型を渡す口はない。`generate_impl` は compiled dimensions、geometry、既存 epilogue 型を列挙して同じ source を作り、新しい environment flag を読まない。FC1 が独立に切り替わるのは r32 の **host patch** が flag を読み、JIT source 先頭に macro を入れる実装を既存 image が含むためである。dense には対応する実装がない。

さらに compiler は name/signature/flags/code から runtime cache key を作り、IncludeParser は最初に読んだ header hash を process 内で保持する。env を変えるだけ、header を書き換えるだけ、外部 nvcc wrapper で定義を足すだけでは、同じ resident process に同じ template の control/candidate を安全に共存させられない。`compiled_dims`、SM 数、geometry を flag の符号として使うと、r36 が監査した template/compile 条件まで比較要因に加わる。これらの迂回や、新しい独自 JIT loader は導入しなかった。

r36 の「header/Python/JIT の範囲なら image rebuild 不要」は条件付きの記述であり、独立 flag の host 経路まで既存 image が持つという意味ではない。A を指定の同一 template 条件で配備するには host extension を変更した fresh image が必要。今回これを作らずに停止した。**今回実際に行った FC1 固定化には build は不要**である。停止は A の数値上の棄却を意味しない。

## FC1 の固定化

`prepare-firmus-pp4-settings.py` の `PP4_SELECTED` に `DG_SM90_PREFILL_FC1_FIRST_K=1`、`PP4_SELECTED_RUNS` に run 41767886175 を追加した。`source/` と `results/` の `pp4-kernel-settings.json` に固定値と根拠 run が残り、Prefill env にだけ有効値が入る。Decode env へは渡さない。

根拠の r34 selection は全12 UUIDで44 case×4中間/最終成分、FP32 reference、1200 call、pooled 速度条件を通過して adopted=true、worker exit code は3本とも0。12 serving process の ready 前 prewarm と candidate dispatch も各worker4行ずつ確認した。公式 score は10054.49。これは既存 run の結果で、今回の新しい serving 測定ではない。

config の FC1 gate key、sbatch の Round 32 block、専用 selector module と staging、legacy selector の専用 CLI を削除した。同時に faulthandler と main returning の一時診断を削除した。廃止 gate 専用の CPU tests/fixtures も整理した。kernel/host patch 自体と ready 前の実 weight・共有bufferによる7988行 prewarm、1 process 1回の serving dispatch log、その既存 `results/prefill-r32-dispatch-{0,1,2}.txt` 回収は維持した。既存の pinned PP4 selections の手順と同じく、採用済み kernel を毎job再選択しない。

## CPU 再現結果と各候補の扱い

CUDA 13.4.92、sm_90a、O3/C++20、Triton 3.7.0 の ASTSource/GPUTarget(cuda,90,32)、pointer alignment 16 byte、onepass の既存 fusion=true で compile した。GPU/driver を用いて kernel を load/launch していない。以下は **r36 audit probe の再現**であって、production flag を実装した結果ではない。control と probe の全10本の encoded SASS は、r36 保存値とそれぞれ完全一致した。したがって命令内訳と path の数値も再現する。image 内 CUDA13.0.88 の生成結果や GPU byte 一致を証明するものではない。

| 候補 | 新規 CPU compile の結果 | 資源と code size | 今回の扱い |
|---|---|---|---|
| A dense QKV/O 初回 promotion | 入口zero初期化 QKV68/O67→0、steady chunk は両方1534→1533。初回は `fma.rn(scale,accum,+0)` | 168 registers、spill0。QKV2376→3888 slot・38016→62208 byte、O2336→3840 slot・37376→61440 byte | CPU削減を再現。独立flagのhost経路にfresh imageが必要なので配備停止、serving未採用 |
| B Q/K norm・RoPE fixed shape | Q64/KV4/stride18432/D128/BF16、int32/int64とも全体472→464。通常Q−4/K−9/V−11 slot | 48 registers、spill0、7552→7424 byte | CPU削減を再現。runtime overlay自体は可能だがbundle停止により未配備、serving未採用 |
| C onepass amax/448 | static loop656→627、通常issued636→619、全体4128→4104 slot。gain側は元の除算を維持 | 132 registers、dynamic shared172544 byte、LOCAL0、66048→65664 byte | CPU削減と有限数全数比較を再現。runtime overlay自体は可能だがbundle停止により未配備、serving未採用 |

B の通常pathは int64 Q258→254/K297→288/V164→153、int32 Q256→252/K295→286/V164→153。C は r36 と同じ3 FMAと +Inf 選択の source を compile した。CPU hardware FMA と IEEE 除算で `float32(1e-10)..max finite` の **1,352,931,585 pattern、差異0**、reciprocal と refinement 後の値はともに `0x3b124925`。全数比較 source と log を proof archive に保持した。有限数の証明と、source の明示 +Inf 選択、GPU FTZ/NaN・下流出力の未検証を区別する。

A の対象は QKV(7988,9216,4096) と O(7988,4096,8192)、BM256/BN128/BK128、3stage、2 math WG、cluster2/A multicast、E4M3入力/FP32 promotion/BF16出力。FLOPは603073806336/536065605632、論理一回I/Oは218734080/166482944 byteで不変。B の主要read/write約234.032 MB、C のgate有効8.376025088 GFLOPも不変。実HBM/NVLink byte、GPU時間、serving scoreは未測定で、ゼロとして埋めない。過去のM-tail分離、norm FMA、共有reciprocalの棄却を再開する変更ではない。

新候補の production overlay は追加していないので、配備されるA/B/CのsourceとJIT条件は親mainのまま。これは「新しいflag offのproduction生成物を検証済み」という主張ではない。FC1は既存の採用済みheader/host/prewarm patchを一切変更せず、有効値の決定経路だけを固定した。

## gate 設計と予算の境界

旧 FC1 の worker95秒・aggregate105秒を解放した。新gateは**実装・投入していない**。将来の設計条件は、Decode greedy後、全12 Prefill GPU、候補ごとの独立採否、実shape/実weightまたは保存snapshotと境界caseの中間・最終byte一致、入力/bufferを変えたchanged-stateを含む1200call、forward/reverse各12round、公開済みhealthy結果による判定である。pooledはまずGPU内で両順sampleをまとめ、その12GPU中央値の節約>0、両順それぞれの中央値>0、正のGPU9/12以上、どのGPUも−1%より悪化しないこと。Aは同じresidentにQKV/O両方を置いて確認する。Bはwriterだけでなくpaged/contiguous/native Vと後段attentionまで検査する。採用後のready前実shape prewarmと各process1回のdispatch、r37 dispatchファイル回収も未実装である。

105秒を仮の総予算に置くと、実行条件 `SECONDS + 105 + 630 <= 1020` の最終開始は285秒。r34の開始249秒なら984秒で余裕36秒、r33の257秒なら992秒で余裕28秒。HEADの起動遅延33秒に対し95秒worker内で本体に残るのは約62秒。timeoutの2秒grace後にもaggregate等へ8秒ある。これは旧gateの予算枠の計算であり、新3候補のcompile/実検査がその中に収まる証拠ではない。未測定の新gate時間を仮定して有効化しなかった。新しい採否分岐・予算実行testを完了したとも扱わない。

## 検証と証拠

git履歴へ依存しない pinned image source fixtureによる実prewarm/dispatchテストと、native FC1 schedule harnessを維持した。固定値の両manifest一致、Prefill/Decodeの分離、gate/診断/stagingの廃止、残すdispatch回収を新テストで確認した。host generator/API/compiler/include cacheの読み取り専用sourceをhash付きfixtureへ残し、停止判断の根拠を履歴やmirrorの配置なしで調べられるようにした。

CPU結果の正本は `prefill-small-bundle-r37-cpu-validation-20261104.json`。新規cubin/SASS/command/resource/全数比較source/logは `prefill-small-bundle-r37-cpu-proof-20261104.tar.gz`（1,194,735 byte、SHA256 `7aeb149fac53f0dcc14373e321d2ab1e1be9779a24faa1bdf1fcfeb51e9ad347`）に保持する。入力r32/r36 archiveは読み取り専用。展開source・compiler cache・一時診断は作業完了時に削除する。

指定のPATH設定後、完全pytestは終了コード0（1951 passed、90 skipped、872秒）、Ruffも終了コード0。最初の完全実行では旧FC1=0を期待する既存testが1件失敗し、固定値1へ更新後に単独testと完全実行を通した。shell/Python構文、diff検査、image pin/production overlay一覧の不変、archive内全10本のSASS hashを確認した。最終のpytest/Ruff logもproof archiveに保持する。展開source・compiler cache・一時診断を整理し、GPU・scheduler・Actions jobは作成していない。
