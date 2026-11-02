# Round 23：V tile writer W、基準見直し後は CPU admission 合格・gate 待ち

現在の実装・判定は末尾の「同日追補」に記す。以下は commit 0570798 の初回棄却記録を保存したもの。

## 初回判定の記録

2026-11-02。基点 main `7343357`、branch `perf/pp4-r23-vwriter-20261102`。**W は不合格。gate、config boolean、candidate ID、production patch、build 入力、image pin は追加・変更しない。** CPU 正本は `prefill-v-writer-r23-cpu-validation-20261102.json`。診断 source と compiler の raw 証拠は指定 scratch の archive に残す。

## 判定と証拠の範囲

同一 kernel launch の V 分岐内に置いた shared 配列も kernel 全体の資源になる。旧 FP32 writer は static shared **0 B**、W は **24,576 B**で、Q/K CTA にも後者が予約される。CUDA 13.4 の cuobjdump resource usage では W は driver 予約 1,024 B を含む **25,600 B/CTA**。Q/K の branch が shared 命令・barrier を実行しなくても予約量は変わらない。「Q/K 側の shared memory を悪化させない」という先行条件を満たせないため棄却した。dynamic shared に移しても launch 単位の予約であり、この問題の解消にはならない。

Q/K の source body をそのまま複製した診断でも、W 内の Q/K の encoded SASS は旧 writer と一致しなかった。例えば int32 の最初の norm FMA は旧 PC `0x0410` の `FFMA.FTZ R23, R2, R2, RZ`、W は PC `0x3a60` の `FFMA.FTZ R20, R2, R2, RZ`。命令 word はそれぞれ `0x0000000202177223` と `0x0000000202147223`。int64 でも差を確認した。これは register 割当て・schedule の差の証拠であり、出力不一致を測ったという意味ではない。選択した FFMA/FADD/FMUL は双方85命令だが、個数一致を算術・出力の byte 証明にはしない。

| CUDA 13.4 / sm_90a、両 index 型 | registers | stack / spill store / spill load | static shared | CPU occupancy model |
|---|---:|---:|---:|---:|
| 旧 FP32 writer | 48 | 0 / 0 / 0 B | 0 B | 5 CTA/SM |
| 診断 W、Q/K と V を同じ kernel に収容 | 48 | 0 / 0 / 0 B | 24,576 B | 5 CTA/SM |

両 compile に C75xx 診断なし。register と model occupancy は悪化していない。shared だけを根拠に occupancy が低下したとは言わない。CUDA 13.4 の `cuda_occupancy.h` を CPU 実行し、sm90、65,536 registers/SM、228 KiB shared/SM、256 threads、48 registers、driver 予約1 KiBという明示した条件で計算した。register 割当ては双方12,288/CTA、shared は1,024→25,600/CTA、制限要因は register。実GPUの occupancy、L1 carveout や時間は未測定。

**実image c7804efb の CUDA は13.0.88。今回の13.4.92はCPUコンパイルによる代替証拠だけである。** GPU、image build、SSH、Actions操作、Slurm/PJM投入は行っていない。

## 設計、CTA と barrier の契約

`AGENTS.md`、astra23/report.md 全文とその独立 oracle、round12 pair、round13 scatter、round14 prepack、round17 V3、round22、tracker/modelと歴史的profileの適用範囲、writer導入commit本文、保存済みround14/15/17のCodex最終報告を確認した。Actionsの過去経過・GPU結果は保存済みresearch/fixture/analysisから確認し、remote APIは呼んでいない。提供された場所の round22 Actions raw directory はこの環境にはなかった。別runのtimelineやclockで欠測を補わない。

現行 P の経路は `prefill-fa3-vsum.patch` の host pair dispatch から、旧FP32 writer → contiguous V → prepack → native FA3 41184。要求長はhostの既存 metadata、cu は既存 forward metadata、image は attention instance が保持する。single は既存 N192 と旧 writer のまま。この buffer/dispatch 契約を調べたが、棄却したため実行経路の tensor allocation や呼出しは変更していない。

診断 W は以下の構造とした。

- Q/K は従来の `blockIdx.x * 32 + threadIdx.x / 8`、1 head/CTA、256 threads、8 lanes/row、旧FP32 norm → RoPE → FP8とstoreの source bodyを保存する。PDL wait の後に CTA 一様な V 分岐だけを前置する。
- V は96 token × 1 head × 1 request。256 threads が16要素の BF16 を各3回読み、既存 `qkv_prefill_fp8` helperで同じFP8 bytesへ変換する。paged Vへ同じ値を1回だけ書き、slot 0は書かない。有効rowの入力loadとcache storeだけをmaskし、paddingは正のzero byteにする。
- FP8 contiguous shared 12,288 Bへstaging → CTA barrier → production逆mapでgatherしてnative shared 12,288 Bへstaging → CTA barrier →各thread 3回のaligned uint4 global store。両barrierは有効なV CTAの全256 threadsが通る。padding threadも途中returnしない。Q/K CTAと余剰CTAはこの領域へ入らない。CTA間のshared参照・barrierはない。各出口でPDL triggerを維持する。
- 既存2次元Q/K gridをそのまま保つため、V用の行をgridの末尾に足す。3994+3994なら `dim3(250,70)`、Q/K 17,000 CTA、V 有効336 CTA、V行の余剰164 CTA、合計17,500 CTA。oracleの17,336は余剰なしの理想的な仕事量で、診断launchのCTA数とは区別する。gridを1次元に詰め直してQ/Kのindex算術を変える案は採らなかった。

V分岐のSASSは128-bit global storeが6本（paged3本＋native3本）、CTA barrierが2本。shared stagingは残る。旧scatter writerの散在byte storeを再導入したものではなく、過去の約17.4 µs／lane変更案約7.6 µsの悪化を再試験したものでもない。ただし同一kernelのshared資源は分離できず、今回の必須条件に抵触する。

## CPU の全byte照合

astra23/layout_census.py は変更せず、指定r23 scratch内のcopyを実行した。production prepackのinverse helperと16-byte gather式はC++で実行し、独立oracleのforward mapと**全12,288座標**で全単射・逆写像一致を確認した。

既存7 pairすべてで、全request/head/tile/byte、末尾paddingを検査した。元入力と変更入力2通りの計3入力、実QKVのBF16 row stride18,432 BとV offset17,408 Bを反映したaddress patternを使った。量子化後に同じ値が多いことによる見逃しを避け、全座標を識別する64-bit address tagでも別に照合した。

paged V は64-token page、offset配置、page permutation、逆順の疎なslot配置で照合した。32/64/96/128境界、request境界、末尾をslot 0へ向け、reserved slotと未使用slotのsentinel保持、各有効byteのstoreが1回であること、int32/int64のlocation一致を確認した。7 pair × 3入力 × 3 cache配置＝63ケース。native VのCPU検査は21入力、byteとaddress tagの双方で一致した。

BF16 **65,536 bit pattern全部**を正確にFP32へ展開し、既存helperのhost実行、旧writerの16要素lane単位repair分岐、CUDA13.4 headerのhost NOSAT変換、独立したE4M3有限値表によるRNEとbyte照合した。別の全単射permutationを相方にしたmixed float2の上位byteも全65,536パターンで一致。NaNのsign/payloadのcanonical化、±0、BF16 subnormal、Inf、overflow、464 midpoint、NOSAT補正を含む。これはCUDA headerのCPU側分岐と配置模型の証拠で、GPUのhardware変換や実際のcache writeの検証ではない。

未変更の旧writer i32/i64、旧native diagnostic writer、その他のrope JIT行とpackの**13 entryすべて**で、追加前後のinstruction wordとcontrol wordがbyte一致した。production入力43ファイルは基点mainとbyte一致。**FA3 control/P/P3は今回再コンパイルしていない。** admission失敗後の行追加を止め、全native source/build入力を無変更にしたことと、新たなSASS証明を混同しない。

## byte の勘定と見込み

3994+3994、Q64/KV4/D128、BF16入力・E4M3出力。以下はV部分だけのlogical global byteで、HBM実測ではない。

| 項目 | 旧writer＋pack | Wの設計 |
|---|---:|---:|
| BF16 V入力 | 8,179,712 B | 8,179,712 B |
| contiguous FP8 V write＋pack read | 8,179,712 B | 0 B |
| paged V write | 4,089,856 B | 4,089,856 B |
| native image write、padding込み | 4,128,768 B | 4,128,768 B |
| 合計 | 24,578,048 B | 16,398,336 B |

削減の設計値は**8,179,712 B/層とpack launch 1回**。native storeのlogical sector要求129,024/代表pair、shared read/write16,515,072 Bは維持する。Q/K、FA3全causal項の522,847,416,320 FLOP（tile込み535,881,056,256）、O、paged KV payload、NVLink payloadは変えない。V変換は同じで、有用な数学の省略はない。実HBM/NVLink byte、clock、相互作用・所要時間は未測定でnullとする。

旧writer＋packと旧writerの差5.261 µs/層を比較上の予算とした、2〜5 µs/層・全workerで24層なら48〜120 µs/周期・約+4〜11 tok/sという値は**admission前の条件付き計画値**にとどまる。今回の棄却から性能利得は主張しない。pack単独4.2 µsを丸ごと通常serving利得としない。

## gate、時間、serving 引き継ぎ

先行条件不合格なので、依頼どおりW gate・W config・IDを追加しない。Pの完全gate、P3の84 cell規則、UUID集合による集約、single/FP32/replay/1200回とtimeout時のserving継続は現行のまま。WとP3を組み合わせる実行経路はない。`test_prepare_keeps_control_and_records_optional_gate` は既存のpair/prepack/v3とdecode greedyの4 booleanをすべて切り替えており、W booleanがないため追加対象はない。

W用の12 GPU × 7 pair composite、代表全GPU2 µs、全shapeの回帰1 µs以内、正逆中央値正という固定基準は実装・測定していない。これらを緩めたgateや、測っていないW+P3の採用経路も作らない。GPU byte比較・FP32 attention・O・1200回耐久は今回のCPU照合で代替できない。

現行群のworker150秒・集約155秒は変更しない。W追加時間は**0秒**。round22のREADY511秒、Slurm13:13＝793秒に対し、`793 + 0 = 793秒`、17分1,020秒まで**227秒**。これは過去runの観測値に対する計算で、将来のwall上限を保証しない。候補を足していないため新たなWの時間見積もりやtimeout blockは不要。既存worker/aggregate timeoutのshell fixtureは指定の全体suiteで検証する。

**ClaudeがW serving commitで変更する箇所はない。** Wをtrueにするbooleanは存在せず、実装は出荷しない。production sourceが変わらないため、この棄却記録のためのimage buildやpin更新は不要。別案を進めるには今回の資源・SASS条件を満たす新しいCPU証拠から始める。

## 検証と片付け

repo直下、CPython3.13.12、**TMPDIRは既定のまま**で指定コマンドを完走した。

`uv run --project qwen-experiment-metrics --with 'numpy>=1.26,<3' --with pytest python -m pytest -q -p no:cacheprovider qwen-experiment-metrics/tests`

**exit 0、1,491 passed、90 skipped**。二重quietでsummaryが出ないため、collect-onlyの1,581件とprogress markerの1,491 pass・90 skipを照合した。subtest数・所要秒数は補わない。90 skipは現行mainと同じ追加依存等の既存検査。dispatch、gateの採用・拒否、P優先、UUID集合、worker/集約timeout後のserving継続、prepareの全boolean切替、wiring/build入力は既存suiteのまま検証した。W専用GPU gateの実装や成功を意味しない。

`uv run --project qwen-experiment-metrics ruff check qwen-experiment-metrics scripts containers` もexit 0、All checks passed。JSON構文、production入力43ファイルのmainとのbyte一致、git whitespace検査も合格。

CPU raw証拠は指定 `scratchpad/codex-scratch/r23/cpu-proof.tar.gz`、**716,654 B**、SHA256 **`bf1c3cb77d682e3122c99a53ee0731339966322e52dd89de2127117dcdaf26c3`**。診断source、元header、before/after object・SASS・compiler/resource log、compile command、BF16 byte表、C++ helper/inverse、独立oracleの未変更copy、照合script・結果を含む。全27 entryの名前集合、manifestの26 entryすべてのsize/SHA256を再検証した。includeの直接参照8 headerもpinned upstream fc275674とbyte一致した。

loose source・object・helper executable・byte表・一時script・collect-only logと今回生成したworktree cache/venvは削除し、archive・最終suite/Ruff logと元のexec.logだけを指定scratchに残す。productionに診断処理はない。SSH、Actions操作、Slurm/PJM投入は0件で、今回作成した残留scheduler jobも0。他のbranch/main/mirror、他作業のjobには触れていない。変更ファイルは本note、CPU JSON、tracker、modelの4ファイルだけである。

## 同日追補：受け入れ基準の見直しと W gate の実装

**現在の判定は CPU admission 合格、GPU gate 待ち。** 上記の初回棄却と `prefill-v-writer-r23-cpu-validation-20261102.json` は当時の基準・事実として保存する。依頼側の再判断により、「shared memory 自体の非増加」「W 内の Q/K encoded SASS の一致」を要求から外した。同一 launch の shared 予約増加や register 名・schedule の差だけでは、occupancy・精度・性能の退行を証明しないためである。新基準は occupancy 模型の維持、共通算術と同じ compile flag、実 GPU byte 比較、composite の固定速度判定とする。変更した基準を過去の実験結果に遡って適用したり、GPU の合格に読み替えたりはしない。

追補の CPU 正本は `prefill-v-writer-r23-revised-cpu-validation-20261102.json`。初回 archive を読み取り、指定 scratch 内で source と検証を再構成した。source repository、他 round の archive は読み取り専用。必読 note、tracker/model、歴史的 profile の対象、保存済み astra23 と前回 Codex final/exec 記録を参照した。Actions の根拠は保存済み note/raw の範囲であり、新たな Actions API 操作は行っていない。

### CPU admission と精度契約

採用した証明方法は **同一 inline 関数と同一 compile flag**。新 overlay `prefill-v-writer.patch` は旧 FP32 kernel の PDL wait/trigger を除く body を、そのまま `qknorm_rope_prefill_fp32_body<IdType>` に移す。旧 writer と W の Q/K CTA はこの同じ forceinline 関数を呼ぶ。body は旧 source と文字列の byte まで一致し、FP32 norm の FMA/reduction、RoPE、量子化と NOSAT repair、lane、address/store を保存する。W の V 分岐は CTA 一様で、そこから戻った Q/K CTA は旧 writer と同じ blockIdx/threadIdx を使う。SASS の FP 依存グラフ比較を行ったという主張ではない。

旧/W は同じ `_jit_rope_fp8_q_module` の translation unit に登録し、JIT flags を変更しない。pinned SGLang の `jit/utils/arch.py` の default は C++20、O3、relaxed constexpr と target。FMA は同じ nvcc default true で、片側だけ fmad/ftz/fast-math を切り替える仕組みはない。初回と同じ use_fast_math 付き診断に加え、これを外した default JIT 相当の組でも before/after をコンパイルした。param を const reference で渡した最初の抽出案は旧 writer の SASS を変えたため使わず、**値渡しの inline body**で旧 writer の encoded instruction/control word を保存できた。これは実装段階の不採用案で、GPU 測定はない。

| CUDA 13.4.92 / sm_90a、i32/i64、両 flag 組 | 旧 writer | W |
|---|---:|---:|
| registers | 48 | 48 |
| stack / spill store / spill load | 0 / 0 / 0 B | 0 / 0 / 0 B |
| static shared | 0 B | 24,576 B |
| allocated shared、driver 1 KiB 込み | 1,024 B | 25,600 B |
| register 制約 | 5 CTA/SM | 5 CTA/SM |
| thread/warp 制約 | 8 CTA/SM | 8 CTA/SM |
| shared 制約 | 228 CTA/SM | 9 CTA/SM |
| architecture block 制約 | 32 CTA/SM | 32 CTA/SM |
| 最終 occupancy 模型 | **5 CTA/SM** | **5 CTA/SM** |

CUDA occupancy header の CPU 計算による。どちらも register 割当て 12,288/CTA が制約となる。W の barrier 制約は 64 CTA/SM で制限要因にならない。shared の増加自体は事実であり、実 GPU の carveout/性能は未測定。C75xx はなし。**実 image の CUDA 13.0.88 の build/実行とは区別する。**

旧 writer、native 診断 writer、decode 等と pack の既存 **13 entry は、両 flag 組の before/after で encoded SASS 一致**。加えて pinned FA3/CUTLASS archive の SHA を確認し、現行の production patch chain を fuzz 0 で適用し、control N192、P、P3 を再コンパイルした。全 encoded word が round17 の各保存 SASS と一致した。全 `kernels/fa3-verify` 入力も基点 main と byte 一致で、native row/ID は追加していない。P3 の既存 4/4 B spill は変更していない。新規 W の spill 0 と、既存 P3 の資源を混同しない。

独立 `layout_census.py` の未変更 copy と CPU 全 byte 検査を再実行した。全 12,288 座標、7 pair の native/padding/address tag 各 3 入力、paged V の 63 ケース、BF16 全 65,536 bit pattern と mixed pair 上位 byte が一致。production W の V branch は前回診断と同一で、inverse は production pack と同一であることも文字列で確認した。新 CPU test は実際の W inverse を C++ 実行し、独立 forward map と全座標を照合する。これらは GPU hardware conversion、Q/K、attention、scale、O の byte 一致を代替しない。

### 実行経路と buffer

新しい overlay は現行の ordered overlays の末尾。Q/K の body と既存 2D grid を保存し、V は前回と同じ 96 token × 1 head × 1 request、256 thread、shared 2 配列、CTA barrier 2 回、連続 16-byte native store 各 thread 3 回とする。paged V は同じ変換値を 1 回 store し、slot 0 を書かない。全 padding byte は 0、余剰 CTA は shared/barrier に入らず PDL trigger 後に戻る。代表 pair の launch は 17,500 CTA、そのうち Q/K 17,000、有効 V 336、余剰 V 164。

host の既存 `use_fixed_prefill` と `pair_candidate` の成立範囲だけで W を使う。2 request、各 3900〜4096、fixed/native P の条件を満たした場合に限り、V image と旧 contiguous K を割り当て、W を 1 回呼ぶ。判定前の V は contiguous K の shape/stride stand-in とし、W 以外でだけ contiguous V を割り当てる。W ではその中間 V allocation/write と pack launch を外し、attention に渡す V 引数は K と同じ shape の stand-in。native consumer は別の image を読む。image は既存の attention instance に保持する。single、範囲外 pair、非 fixed 経路では旧 allocation/writer を使う。host の選択で CUDA tensor の値を読まない。

### Gate の固定条件と採用規則

P/P3 の既存 150 秒 worker、155 秒集約は変更しない。**P 完全 gate と既存 P3 判定・env 公開が終わってから**、別の W group を起動する。P 不成立なら W を投入しない。これにより W の timeout/失敗は公開済み P/P3 を無効にしない。W の集約でも prior selection の P 全検査を再確認する。

3 組 × 4 rank、計 12 Prefill H200。既存 7 pair は 3994+3994、3986+4027、3986+3986、4027+4027、4096+4096、3900+4096、4096+3901。各 GPU/shape で P と W の **writer → P FA3 → O** を測る。FA3 は同じ row ID15 / variant41184。10 rounds、graph 内 12 calls、正順と逆順を interleave する。writer 単体時間を採否に使わない。

固定した判定は次のとおり。GPU 実行前に定義し、今回の実装後に測定値から緩和していない。

- 3994+3994 は 12 GPU の全員で、正順・逆順・pooled の短縮が各 **2 µs 以上**。
- 全 7 shape × 12 GPU で、正順・逆順・pooled の回帰が各 **1 µs 以下**。片順の回帰を pooled で隠さない。
- 84 cell の短縮の中央値が、正順・逆順のそれぞれで **厳密に正**。
- 各 shape の元入力と変更 2 入力で、P→W と W→P の両順を replay し、**Q/K、paged K/V、native image、attention、scale、O を全 byte 比較**。比較前に各出力を別々に poison し、W が contiguous V を書かないことも検査。page/tile 境界、request 境界、slot 0 を含む location の並べ替えを入力ごとに使う。host forward map と独立 GPU transpose dump の比較も維持する。
- 旧 FP32 causal reference の relative L2 ≤0.05、max scaled error ≤0.25 を全 3 入力で維持。single 3994/4096/3584 は旧 writer、旧 N192 の producer/output bytes と FP32 reference を検査し、速度条件を加えない。
- serving dispatch の **1200 回 eager composite**、初回/最終 FP32、attention/scale/O の安定性と direct dispatch 一致を検査し、終了後の producer/image/output を P とも全 byte 比較する。

W+P3 は測らず、serving でも組み合わせない。W は P と同じ consumer を使うことで一つの主要因に限定し、追加の 84 cell と耐久を節約する。W 成功時は W+P、W 不成立時は直前の P3 または P、P 不成立時は従前 control。既存 P3 の 84 cell 規則は変更しない。W 成功で P3 を上書きする規則であり、W が P3 より速いことまでは主張しない。

集約は worker/rank の固定対応でなく、**prior P selection と W の UUID 集合の一致**を要求する。12 個の重複しない UUID、allocation/run/image/native binary、finite samples、全 shape/reference/single/endurance を再検査した後だけ env の単一 assignment を atomic replace する。欠落、古い fragment、byte 失敗、閾値未達、時間切れなら元の env の byte を保つ。selection JSON/raw/log は `results/prefill-fa3-v-writer/` と既存 ART log 配下に残り、既存 results/log 回収経路に乗る。

### 時間予算と 17 分枠

前回 P/P3 group は起動込み 150 秒、集約 155 秒が上限。W は別枠で **60〜75 秒を条件付きで見積もり**、worker は全 srun 起動込み共通 start+95 秒、kill 猶予 2 秒、集約は start+100 秒まで。見積もりは P の過去 rank 約36秒に対し、W の両順 byte replay と import/JIT を余分に見込んだもの。実 H200 測定ではない。全形状・耐久を時間に合わせて削らない。

round22 の READY511秒、Slurm793秒を基準にすれば、W 見積もり追加で READY571〜586秒、Slurm853〜868秒。**上限の単純加算は 793+100=893秒、1020−893=127秒の余裕**。既存 P/P3 群の上限そのものは増やさず、W group 100秒だけを追加する。さらに起動時点で `1020−SECONDS >= 100+90+660` を満たす場合だけ W を開始し、後続 Decode gate 最大90秒と serving/回収660秒を確保する。遅い startup では W を skip し、直前の選択で続行する。この条件は READY と終了時刻の保証ではなく、allocation 内の保守的な予算管理である。

shell fixture は W worker timeout、aggregate timeout、残予算不足、P 不成立のすべてで、公開済み P3 env が byte 不変のまま serving に到達することを実行確認する。P/P3 の timeout fixture も維持する。

### 配線、後続 serving commit、未確認事項

`prefill.fa3_v_writer_gate` は **false** で commit する。prepare は prepack gate を前提とし、選択記録にも W boolean を残す。`test_prepare_keeps_control_and_records_optional_gate` は pair、prepack、v3、W、Decode greedy の全 5 boolean を切り替える。candidate table の既存 ID はすべて維持する。

後続で Claude が変更するのは、この boolean を true にする serving commit と、fresh build 後に得られた実 image pin 一式。**source 変更なので新規 image build が必要**。本作業では c7804efb の `images.sglang` / `container_*` pin を変えず、実在しない将来の SHA を作らない。build 用 workflow は committed checkout を入力にし、runtime overlay は config の末尾 patch として staging/CPU source gate に渡る。harness は既存 provision の script/source copy、sbatch の追加 group、既存 results/log archive に接続した。SSH/Actions/Slurm の起動や image build は本依頼では禁止されており、実行していない。

削減する仕事、byte 勘定、条件付きの 2〜5 µs/層・48〜120 µs/周期・約+4〜11 tok/s は初回 note と同じ。新しい kernel 時間や serving score ではない。GPU Q/K ほか全 byte、13.0.88 の resource/occupancy、composite gate、1200 回耐久、通常 serving の公式 parser/eligible/score は未確認で null。採否が変わる未確定点は「shared 予約を伴う W が実 GPU で byte 一致し、P 比の composite 閾値を全 cell で満たすか」。通常 serving の性能結論は同条件の反復で判断する。

### 追補の検証・片付け

repo 直下で指定コマンドの全体 suite は **CPython 3.13.12、1,578 passed、100 skipped、exit 0**。TMPDIR は未設定、既定 /tmp のまま、fixture 差し替えや一時 plugin は使っていない。collect-only の1,678件と progress marker を照合した。100 skip は追加依存を要する既存90件と、新しい実Torch検査10件。実Torch2.11.0+cu130を使う追加のCPU targeted検査はPython3.12.12で **307 passed、skip 0、exit 0**（W/pair/prepackの3 file）。Wの各byte不一致を注入する9ケースを含む。指定Ruffもexit 0。ordered patch check/applyと適用結果の全4 fileのbyte比較、Python AST、CUDA compile/SASS、JSON、shell構文、git whitespaceも合格。

追補 raw は指定 scratch の `cpu-proof-revised.tar.gz`、**2,345,429 B**、SHA256 **`e0c522a8bc1dd324bd7b58c0e84db9f9bff8fa80bd4f393d5bd488c9db95c61d`**。最終overlay/source、旧source、default/diagnosticのobject・SASS・logとcommand、FA3 control/P/P3の旧/新SASS・object・log、host byte/occupancyと独立oracleを含む。全61 entryを照合し、manifestの60 entryすべてのSHA/sizeを検証した。初回archiveのSHA `bf1c3cb7…26c3` は不変。

変更は13 file：新しい `patches/sglang/prefill-v-writer.patch`、selector、sbatch、prepare、provision、config、pair/prepackの既存2 test、新しい `test_prefill_v_writer.py`、本note、新CPU JSON、tracker、model。native FA3/table/build.pyとimage pinは不変。不要な展開source・object・診断script・worktree cache/venvは削除し、raw archiveと検証logを残す。SSH/Actions/Slurm/PJM/GPU/image buildは0件で、作成した残留jobも0件。他作業の資源には触れていない。
