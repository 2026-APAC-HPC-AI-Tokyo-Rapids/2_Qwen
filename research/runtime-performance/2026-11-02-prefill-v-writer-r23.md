# Round 23：V tile writer W は CPU admission で棄却

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
