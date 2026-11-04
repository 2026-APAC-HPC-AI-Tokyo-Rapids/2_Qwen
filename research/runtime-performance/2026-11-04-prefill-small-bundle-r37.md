# Round 37：FC1 first-K の固定化と小案 bundle の配備停止

継続後の現在の状態：下記「継続作業」節で A/B/C の production overlay・独立 flag・統合 gate・ready前準備と回収を実装した。最初の停止記録は経緯として残す。新規 image build と GPU 採否・通常 serving は次回の投入で確認する。

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

## 継続作業：host overlay を含む production bundle と統合 gate

前節の停止後、ユーザーが fresh image rebuild を許可したため、親 `531adf9` から同じ branch で A/B/C の production 実装を完成させた。この節が継続後の状態である。今回も SSH、Actions 操作、scheduler 投入、GPU 実行、image build は行わない。次回の投入は require-build を付けた別の submit commit から **build 1回＋通常 serving 1回**にまとめる。本 commit は skip-ci の実装 commit であり、image pin は build 41729881283 / `3a47b757…` のまま。CPU 検証を GPU 採用や serving の改善と扱わない。

### 実装と flag

| 候補 | 独立 flag | production 経路 |
|---|---|---|
| A | `DG_SM90_PREFILL_DENSE_FIRST_K` | `patches/deep-gemm-host/series` の末尾に source generator patch、PP4 の DeepGEMM device overlay 末尾に constexpr guard 付きの初回 promotion を追加 |
| B | `SGLANG_PREFILL_QKNORM_FIXED` | 元の FP32 writer はそのまま保持。Q64/KV4、BF16 row stride 18432 の実 host check 後だけ fixed writer を dispatch |
| C | `SGLANG_PREFILL_ONEPASS_DIV448` | 元の onepass JIT 関数はそのまま保持。on では RN 3 FMA と明示的 +Inf 選択による div448 専用 JIT 関数を選ぶ |

初期値はすべて0。`prepare-firmus-pp4-settings.py` が Prefill の env と両 manifest に記録する。Decode へは渡さず、既存 flag を中継する環境変数は追加しない。既存 FC1 first-K=1 と過去の採用設定は維持する。

A の host patch は r32 と同じ利用側 overlay series を使い、off の生成文字列を元のまま返す。on だけ source 先頭に macro を置くため、同 resident・同 compiled_dims=nk・同 geometry で control/candidate の JIT cache key が分かれる。device 側は K-major SF、M未定義、group1、QKV N9216/K4096 または O N4096/K8192、BM256/BN128/BK128、各 swizzle128、3stage、TMA128/math256、cluster2/A multicast、SM132、Normal、Identity epilogue の2 template に限定する。BF16 出力は元の static assert も維持する。他 template は元の K loop をそのまま使う。初回 FMA は乗算に変えず、positive zero を addend にした `__fmaf_rn` とする。stage/phase、M wave、TMA release、全 K 積は保存する。

B では追加 include と fixed kernel の挿入によって元の device assert の行番号まで変わる問題を CPU compile が検出した。元の source 行を `#line` で保持して解消した。C では既存関数に constexpr 引数を加える試作が SASS 一致でも cubin を変えたため撤去し、元の JIT 関数と signature を保持する最終構成にした。近似演算、粗い量子化、gain 側の除算変更、reduction/同期の変更はない。

### CPU の production 不変確認

CUDA 13.4.92、O3/C++20、sm_90a の **42 compile**（A対象2＋対象外10＋Bのint32/int64、各 control/off/on）を実施。同じ source/header path と同じ compile 条件で比較し、14構成すべて control と off の encoded SASS **および cubin 全 byte が一致**した。A の対象外10構成は macro on でも両方一致した。対象外は compiled M、N、K、group、BM、BN、stage、multicast数/向き、SM数の変更を含む。これは全 template の総当たりではなく、constexpr guard の各軸と代表構成を照合した検証である。

A 対象 on は QKV 38016→62208 byte、O 37376→61440 byte。B は両 IdType とも7552→7424 byte。資源は r36 と同じ168/48 registers、spill0。別途 Triton 3.7.0、GPUTarget(cuda,90,32)、pointer alignment16、fusion=true で C の production 関数を3 compile し、off は SASS と cubin 全体で一致、on は66048→65664 byte、shared172544、132 registers、LOCAL0 を確認した。host generator は source fixture から実関数を抽出した CPU C++ harness で36条件を実行し、off の生成文字列一致と on の macro 以外の文字列一致を検査する。

これらは device artifact の一致である。変更した host extension や、追加 kernel を含む JIT host module 全体の binary が旧 image と同じという主張ではない。CUDA13.0 の fresh image、GPU の FTZ/NaN、実出力 byte、時間、通常 serving の正しさは次回の in-job gate と公式 parser で確認する。C の有限数全数比較の根拠は前節の正本を維持する。

### in-job gate の検査と公開結果

`small_bundle_r37_gate` を PP4 config に追加した。Decode greedy の後、モデル起動前に3 worker を並列に開始し、**各 worker に torchrun は1回、4 resident process、全12 GPU**とする。A/B/C は同じ process に常駐し、候補ごとに JSON を atomic publish する。r34 と同じく、srun/timeout の終了コードは記録のみで採否に使わない。候補別の欠測・不健康・期限超過・byte不一致はその候補を不採用にする。後続候補の失敗や interpreter の終了遅延で、既に公開した候補の成果を無効にしない。

最初の model layer の Q/K/V/O の FP8 weight と FP32 scale、Q/K norm、post-attention norm、router weight を safetensors から読み取り専用で取り出す。大きな expert weight はロードしない。tensor 名と実 byte の SHA256、overlay source の SHA256、image、UUID、worker/rank、時刻を公開する。入力は shape ごと・役割ごとの固定 seed の専用 buffer とし、並行 compile の thread 順序で値が変わらない。元 model や serving の weight を変更しない。

- A：M7936/7988/7989/8192、QKV と O を同 resident で比較。両 BF16 projection の出力と、その FP8 consumer operand/scale を中間・最終として byte 比較する。
- B：M31/32/33/7988/7989×int32/int64。Q、paged K/V、contiguous K/V に加え、後段の通常 attention output/scale、7988行では実 serving と同じ別 kernel の native V prepack と P/P3 attention output/scale も byte 比較する。
- C：M1/63/64/65/7988/7989/8192、cap8192。residual、live FP8入力、FP32 scale、paddingを含む全top-k ID/weightを byte 比較する。これは onepass operation の全公開出力の検査であり、expert GEMM までを毎回再実行する gate ではない。

各 case は initial、boundary、changed、restored の4状態を検査し、復帰時には保存した初期出力とも byte 比較する。boundary は zero/signed zero、tiny、wide、NaN/Inf、cache slot0、位置端、部分 tile を含む。byte の検査は uint8 view で行い、浮動小数の許容誤差は使わない。

耐久は **各候補それぞれ1200 call**。initial600とchanged600、候補の別出力buffer2個を交互に使用し、各 call 後に GPU checker で全公開中間/最終出力の byte を control snapshot と比較する。control は状態ごとに再利用する。各 buffer の初期状態への復帰も確認する。候補間で耐久回数を共有しない。A の1callには QKV/O の両方を含む。

通常 kernel を CUDA event で測り、forward/reverse 各12 round、各 sample8 call。候補の GPU 内で両順 sample をまとめてから、12 GPU の節約中央値を求める。採用には pooled・forward・reverse の12GPU中央値がすべて正、正のGPUが9以上、最悪GPUが−1%以内を要求する。A は各GPUの対応する QKV/O sample を足して判定し、さらに QKV または O が単独でどれかの GPU/order/pooled で−1%超悪化すれば拒否する。別 run の時間や clock を合成しない。

### 95/105秒の枠と55秒の本体設計

開始条件は厳密に `SECONDS + 105 + 630 <= 1020`。worker95秒、kill grace2秒、aggregate105秒を使い、公開時刻も開始+95秒以内でなければ不採用。HEAD の約33秒起動を加えると、本体55秒で88秒、worker内余裕7秒。過去249秒開始なら984秒、257秒なら992秒、最終開始は285秒となる。

今回のローカル CPU cold compile は同時3本の条件で、A の対象control/candidateの合計が約30秒、Bの単一device instantiationが各3.4〜3.8秒、Cのcold variantが約3秒だった。CのPython scalarは1、16の倍数、その他の3種で production JIT key が分かれるため、全候補を直列に cold compile する設計は55秒を使い切る。このコストを隠さず、A の compile/byte検査中に、同じ process 内の2 CPU threadでBの実JIT moduleとCの両実JIT関数×3 scalar keyを先に warmup compile する。Cの warmup は launch しない。初期化で使う専用GPU buffer以外にGPU計算を並行させず、両 future を **耐久・計時前に必ず join** する。compile失敗も候補ごとに記録する。

本体の計画配分は、import/実weight snapshot/provenance 6秒、重ねた cold compile 30秒、境界/中間/attention byte検査8秒、各1200耐久と両順12roundの合計8秒、公開/処理余白3秒で約55秒。これはローカルCUDA13.4と既存kernel時間を使った **条件付きの予算見積もり**であり、CUDA13.0上の実測ではない。全候補を保持し、round数・候補別1200call・境界caseは削っていない。次回に未解決なのは、cold host module compileと同時4rankでの実時間がこの枠に収まるかである。期限を緩める retry/fallback はなく、間に合わない候補は未採用のまま serving を開始する。

### serving の準備、dispatch、回収

採用 flag だけを各実 Prefill process の post-load transform/KV dtype 設定後、pool/graph/ready前に7988行で1回実行する。A は実 QKV/O weight、B は実 norm/trig、C は実 post-norm/router weightを使う。専用bufferとRNG forkを使い、同期成功後だけ prewarm done を出す。実 forward の完了後に、候補ごと・processごと1回だけ dispatch log を出す。後続 call は set の判定だけで logger/import/function call を繰り返さない。aiperf/公式 score の後、r32 と同じ grep の方法で `results/prefill-r37-dispatch-0.txt`、`-1.txt`、`-2.txt` へ回収する。

実装のCPUテストは履歴非依存のimage/host/control source fixtureを使う。独立採否、全12rankの欠測、不健康、期限、wrong image/run/UUID、byte、1200/changed/buffer/restored、round欠損/非有限値、source差、全速度分岐、A片側損失、Slurm再割当、prewarmの実行順/once、Prefill/Decode分離、予算、stage/回収を検査する。image内にjqがない既存制約に合わせ、pinはsbatch側でparseしてwrapperのargvで渡す。

FLOP、論理I/O、dtype、全K積、NVLink payloadは前節の模型を維持する。実HBM/NVLink byte、新GPU時間、serving scoreは未測定。source/mirror/環境の所有境界は変えず、worktree内のtomokazu所有ext4にだけ一時展開して検証した。

継続作業の最終検証：指定の完全pytestは終了コード0（2031 passed、90 skipped、約883秒）、Ruffも終了コード0。最初の全体実行は中断せず完走し、既存の末尾overlay/flag一覧/optional gateを固定した期待9件を更新してから完全再実行した。patch空行contextの空白を正規化し、source不変のfixture再適用も通した。CPU正本は `prefill-small-bundle-r37-production-cpu-validation-20261104.json`、raw生成物・source/command/resource・初回/最終ログは `prefill-small-bundle-r37-production-cpu-proof-20261104.tar.gz`（3,347,134 byte、SHA256 `be07a222a7e70875a2e84df6a7eccea53163f289d09279fee13a1325ca9568f2`）。syntax/diff検査は終了0、image pin不変。一時展開・compiler cache・診断ファイルを削除し、今回作成したGPU/scheduler/Actions jobはない。

### Round 37 の結果（run 41792751530）

build image `5018d322ff7b5884520300052af6f3cd1d5594e210bee85e62cd0495f417d94a`（source 6d2cb17）。公式 score は **10053.96**（main 10054.49 と同等）、TTFT 超過 111 件はすべて計測先頭5秒以内で従来どおり。FC1 first-K の常時有効化は serving で `using candidate` と ready 前の prewarm done を全 Prefill process で確認した。

小候補 A/B/C は 12 GPU 全てで gate 起動後約6秒に失敗し、kernel の正しさ・速度について何も測れていない。原因はどちらも gate script 側である。

- A：checkpoint の dense `weight_scale_inv` は BF16（例: q_proj [64,32]）で、snapshot がそれをそのまま DeepGEMM へ渡し `layout.hpp:68` の `sfa/sfb == kFloat` assert に当たった。production は FP32 parameter へ copy してから使う。
- B/C：`prepare_compilers` の thread 内 import と main thread の import が交差し `_DeadlockError`（`sglang.kernels.ops.moe` / `.quantization`）。

修正は gate script だけで image を変えないため、次回は build せずにこの image を pin して再投入する。95秒枠内に実際の cold compile が収まるかは引き続き未測定。
