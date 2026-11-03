# Round 28: Prefill small bundle with independent in-job gates

実装 branch は `perf/pp4-r28-gemm-tail-20261103`、base main は `8d7848f`。実装 checkpoint は `3ae0b61`、契約・fixture修正は `67eb56a`。config の image pin `a001f901` は維持した。**GPU、SSH、Actions 操作、Slurm 投入はしていない。全候補は GPU 判定待ちで、性能改善・公式 score の新しい結果はない。** gate の key は true、serving の候補 flag はすべて 0 から開始する。

## 入力・比較の境界

指定の `codex-scratch/r28/astra29-final.md` を全文読んだ。同内容の roofline note、tracker/model、round24 profile JSON、round9 dense、pair/prepack/P3、round26/27 の記録、main と remote round27 branch の実装・commit 本文を確認した。会話の参照元は提供された依頼と保存分析・round27 記録であり、未提供の履歴を取得したとは扱わない。Actions の実行経過は退避済み run `41622902468` の Slurm/rank log を参照した。API での Actions 操作は行っていない。

対照は main の PP4（3 worker × PP4、H200、24/24/24/22 層）、image a001f901、既存 C/P/P3・greedy の選択経路。round24 の実 clock 1.410 GHz、113.61 ms/代表周期、QKV/O 23.47 ms、FA3 18.95 ms、norm_quant 約1.37 ms/22回は既存 profile の値であり、今回の通常 serving 実測ではない。GPU によって gate が選ぶ C/P/P3 を固定して C2 の対照とし、C1/C3 と組み合わせて効果を推測しない。

## C1: QKV/O の M tail を別 launch に分離

対象は FP8 1d2d の QKV `(7988,9216,4096)`、O `(7988,4096,8192)`、FP32 K128 scale/promote・BF16 出力。`prefill-small-bundle-r28.patch` は `prefill_fp8.linear` に、`SGLANG_PREFILL_GEMM_TAIL_SPLIT=1` のときだけ 7936 行と52行を同じ stream へ順番に launch する経路を追加する。入力・column-major scale は view、出力は同じ buffer の相互に交わらない行。7936 は256/128/16の倍数であり、scale の既存 column stride は維持する。weight、K の累積順と scale 適用、DeepGEMM host/header/mainloop を変更しない。7988 以外は現行の一回呼出し。

実 host heuristic の CPU 実行で次を確認した。dtype サイズと heuristics runtime だけを CPU 用に置き換え、SM90 selector 本体を実行している。CUDA descriptor encoding や GPU load の証拠ではない。

| launch | BM/BN/BK | stage | cluster | math threads | shared bytes |
|---|---|---:|---:|---:|---:|
| QKV body7936 / control7988 | 256/128/128 | 3 | 2、A multicast | 256 | 216448 |
| O body7936 / control7988 | 256/128/128 | 3 | 2、A multicast | 256 | 216576 |
| QKV tail52 | 64/80/128 | 11 | 1 | 128 | 216320 |
| O tail52 | 64/32/128 | 16 | 1 | 128 | 205312 |

CPU で実 `get_swizzled_block_idx` 本体を実行し、control/body/tail の6796 tileについて全座標の一意な網羅を確認した。body の隣接 CTA は同じ A/M tile と対になる N tile を持つ。tail は cluster1 で multicast 相手を待たない。別 launch の間に barrier/peer の状態を引き継がない。通常 mainloop 内に分岐を足して6.3%悪化した round9 の形は持ち込まない。

有効 FLOP/層は QKV 603073806336、O 536065605632。論理一回通過 I/O は218734080/166482944 byte（A、scale、B、scale、出力）で、実 HBM byte ではない。新 tail の QKV は N9280 まで padding する。削る MMA は QKV 14461960192、O 12884901888 FLOP/層、24層で約0.6563 TFLOP。1.410 GHz の1524.70 TFLOP/sで約0.4305 msという仕事量換算であり、追加 launch・B再読出し・critical waveを含む時間改善ではない。NVLink の処理・payload は変更しない。実 HBM/NVLink byte と net gain は未測定。

同一launchでBM256とBM64を切り替えるには、TMA box/SMEM配置、math WG数、stage arrival数の異なる経路を共存させる必要があり、通常mainloopのSASS不変を保証する小変更に収まらないため見送った。既存 kernel を組み合わせる分離案を control と比較する gate に載せる。QKVだけ良くてもOが閾値に達しなければC1全体を採らない。

## C2: pair をまたいだ longest-first

`prefill-pair-global-order.patch` は FA3 scheduler に既定 false の constexpr を追加し、variant bit1048576 のときだけ2要求を一つの降順 M-block 列へ割り当てる。長い側の余分な block を先に処理し、その後は同じ block 番号の2要求を交互にする。head の interleave、QK/PV、softmax、scale、output address、producer/consumer の handoff はそのまま。single と pair 範囲外の host dispatch は変えない。

既存 C8416/P41184/P3 172256 に対応する3つの AOT row を ID18/19/20 に追加し、元の ID0..17 は維持する。`SGLANG_PREFILL_FA3_PAIR_GLOBAL_ORDER=1` のときだけ、現在選ばれた pair row 名へ `-global-order` を付ける。P/P3 の native V と精度選択は維持する。

実 CUDA mapping body の CPU 実行で、3900..4096 の全38809 length pair、計103582600 tileが一度ずつ処理され、KV step数が非増加であること、終端 sentinel を確認した。CI用テストも実 patch の body を抜き出して実行する。

代表3994×2の有効522847416320 FLOP、tile実行535881056256 FLOP、56784 KV訪問/層は不変。論理 K+V load 約1.396 GB/層、unique K+V約8.18 MBも仕事量としては不変で、実 HBM byte は未測定。NVLink payload を変えない。削る対象は要求間の長い tile の偏りのみ。既存模型の0..0.032 ms/周期は条件付きで、今回の期待利得・測定値ではない。L2 localityやscheduler命令の増分はgateで判定する。候補の見送りはしていない。

## C3: residual store の幅を2要素へ変更

`prefill-norm-store-r28.patch` は既存 `prefill-fp8-integration.patch` 自体の SHA を変えず、全既存 overlay の後で `norm_quant` へ既定 false の `STORE_PAIR` を加える。Prefill env は `SGLANG_PREFILL_NORM_STORE_PAIR`。BF16 store 用の値だけを warp の xor1 shuffle で交換し、偶数列の lane が隣接2要素を32-bit storeする。未丸めFP32 h、sum-of-squares reduction、rsqrt、weight、BF16丸め、group128の量子化を変えず、`enable_fp_fusion=False` を維持する。round27 FMAは持ち込まない。既存 Decode wide norm が導入する `os` を使い、Decode の関数は変更しない。

対象は7988/7989/3994/2049×4096。全65536 BF16 patternをxor1の相手とpairにし、little-endian 32-bit packingが元の2個の16-bit storeと一致することをCPUで確認した。laneの連続列配置はローカルcompileのTTGIRで確認した。実imageのcompiler配置と出力の一致はGPU gateで確認する。

7988行の論理一回通過I/Oは230062592 byte（X/R read、R/Q/scale write、weight一回）。source上の通常FP32演算は24641 op/行、計196832308 op、別にrsqrtが1回/行。abs、max/min比較、変換はこのFLOP計数に含めない。いずれも候補で不変。実 HBM/NVLink byte は未測定で、通信処理は追加しない。

**今回のローカルcompileでは、分析の131 MB/呼出しのSMEM往復を消す証拠は得ていない。** residual は既にscalar global storeであり、候補はその幅・lane割当だけを変える。SMEM命令数は変わらず、shuffleが増える。仕事量削減や0.32 msの回収を主張しない。要求されたstore layout候補として厳しいbit/speed gateの後ろへ置く。

## CPU / SASS の正本

sourceはDeepGEMM host revision `1c8138a59b3df5dac50b76b730834ed57982fdfb` と現行overlay、SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4` と現行overlay、FA3 `f89bc2306632d1ec5f97b014dded4254f5b4a907`、CUTLASS `57e3cfb47a2d9e0d46eb6335c3dc411498efa198`。CUDAは `/usr/local/cuda-13.4` の13.4.92、target sm_90a。GPU/driverを実行せず、encoded instruction/control wordをbyte列にして比較する。imageのCUDA13.0.88とは別の証拠。

- C1: DeepGEMM host/header/generated templateは変更なしであり、flagなしの全instantiationは構造上同一。さらにnative shapeと実際のpinned Decode overrideを含む12個のunique templateをbefore/offで二重compileし12/12一致。bodyはregister168、tailのQKV/Oは248、spill store/load 0/0、C75xxなし。tail encodingは23808/15616 byte。全sourceの全template引数の組合せをcompileしたという意味ではない。
- C2: FA3の既存live Prefill15 rowはbefore/offのencoded SASSが15/15一致。新3 rowはregister128、C75xxなし。Cはencoding69376→71936 byte、spill40/28→44/32 B、Pは70784→73216 byte、spill0/0、P3は71680→74112 byte、既存と同じspill4/4 B。C2にspill0のadmission条件はなく、実機のbyte/speed gateで拒否可能。GPU loadと速度の証拠ではない。
- C3: Triton3.6.0のASTSource/GPUTarget(cuda,90,32)、8warps、fusion false、CUDA13.4 ptxasでcompile。offはSASSのSHA `ed96832eff63d23ec9d7f2169dafe59e42f70da24de2a5a0ea3564f04b9a8881` が一致。before/candidateはREG64/64、LOCAL0/0、shared1024/1024、LDL/STLなし。FFMA92/FMUL81/FADD58は同じ。STG33/LDS20/STS23も同じ、SHFL90→106、encoding15872→17152 byte。

要約は `prefill-small-bundle-r28-{cpu,cuda,fa3,triton}-validation-20261103.json`。入力・実body・compile script・SASS・terminal logの再現用控えは指定scratch内のCPU proof archiveにまとめる。kernel-theory-model.jsonは変更しない。archiveは `round28-cpu-proof-20261103.tar.gz`、SHA256 `e41e9af6a5c3c21a74272a75ad9e5dcbc85f53e109a3cc4eac675a7b484fdfa4`。

## Gate・budget・失敗範囲

`select-firmus-prefill-r28.py` と `firmus_prefill_r28.py` が3候補の共通処理を持つ。greedy後・model起動前に3 worker×4GPUを並行起動する。round26/27と同様、同一allocationのpair gateにある12 UUIDの集合を照合し、重複、別run、別image/source、別C/P/P3 pin、非H200、欠測、失敗、期限超過を採用しない。先行pairが性能で不採用でも12 UUIDの証拠が揃えばC1/C3を評価できる。C2は現行pair rowが存在しない場合だけ独立に拒否する。

全候補・全shapeで control対candidateの byte一致、入力v0/v1/v0復帰、変更入力のgraph replay、2出力bufferで1200回の通常launchと復帰を要求する。C1はQKV/O別に24回graphの平均µs/層を1sampleとする。C2は既存pair7shape、12回/sample、C3は4行shape、8回/sample。C3のresidual reset copyは両variantの計時に同じように含め、状態蓄積を避ける。各order10sample、forwardとreverseを両方取る。

| 候補 | 絶対閾値、1呼出し換算 | 相対閾値 | 理由 |
|---|---:|---:|---|
| C1 | 2 µs、24回blockでは48 µs | 0.5% | 約450..525 µsのdenseに対しrelativeの約2.2..2.6 µsが主に効く |
| C2 | 1 µs | 0.5% | pair約790 µsなら約4 µsを要求し、微小な並べ替えnoiseを採らない |
| C3 | 0.5 µs | 0.5% | norm約60 µsなら絶対閾値約0.8%が効き、短いshapeにも同じ下限を置く |

**各GPU・各shape・各orderの中央値がmax(絶対閾値,0.5%)以上改善する場合だけ採用する。** pooledだけで救済しない。一つのshapeの悪化はその候補を拒否する。各候補は完了したrank evidenceを個別に保存し、後続候補のtimeoutで完了済み候補を撤回しない。norm→FA3→denseの順で短いgateを先に完了させる。workerが停止して残る候補が欠測なら、その候補は現行のまま。新しいretry/fallback/lockは加えない。

workerの共通絶対期限90 s、kill grace2 s、集約を含む共通絶対期限100 s。最終shape完了後も90 sを検査し、aggregateは90 sより遅いfinished_atを受理しない。`SECONDS + 100 + 660 <= 1020` のときだけ始める。SECONDS260なら開始可、261なら不可を実shell blockで検査し、worker/aggregate timeout時にenvが変わらないことを確認した。過去round27は910 s、READY614 s、norm正常rank約8 sだったが、今回のcold JIT・shape追加の所要時間保証には使わない。固定期限で660 sのserving予約を維持する。

## Cache、dispatch、E2E

gateは `TRITON_CACHE_DIR=/tmp/qwen-prefill-r28-${SLURM_JOB_ID}-${worker}/triton` を明示し、home/BeeGFSへ書かない。同じjob別workerのnode-localへTVM/native DeepGEMM cacheも置く。既存jobで観測されたshared-cache dlopen障害を避けるためで、新しい中継envやretryは作らない。servingは既存 `TRITON_CACHE_DIR=/tmp/sglang-$name/triton` をそのまま使い、既存のDG cacheも維持する。

採用flagはPrefillのenvにだけ1を設定し、Decodeへ渡さない。worker.shは3flagの継承をunsetしてからPrefillだけ結果envをsourceする。実際にdispatchしたprocessは候補ごとに一度 `Prefill r28 gemm-tail using ...`、`Prefill r28 fa3-pair-order using ...`、`Prefill r28 norm-store using ...` をwarning logへ出す。score.json確認後、各workerの全行を `results/prefill-r28-dispatch-{0,1,2}.txt` に保存する。rank prefixを落とすgrep -oやsortによるprocess間の重複除去はしない。何もなければno-dispatchの行を明記する。今回の実serving dispatch logは未取得。

Actions→image build/provision→ordered runtime overlay→必須PP4 CPU gate→Slurm内のpair/prepack/greedy/r28 gate→通常serving→既存公式parser/score→dispatch/成果物回収という既存経路を維持する。BigQuery資格情報・計算ノードの通信経路は変更しない。

## Buildと投入担当への引継ぎ

C1はSGLang Python runtime overlayのみでDeepGEMM header/host変更なし。C3はSGLang runtime overlayのTriton JIT。`stage-qwen-overlays.sh` は既存imageのpackageをcopyし、順にpatchしてbind mountするので、**C1/C3単独ならhost/AOT/image buildは不要**。C2はFA3 headerとAOT tableを変えるため、**今回のbundle全体はfresh imageが必須**。初回checkpointのcommit本文がSGLang overlayもfresh imageの理由とした点は、この経路確認で訂正した。

Claudeの投入commitでは3gate trueと既存image pinを維持し、round28のjobを指す追跡可能な変更をcode/config等のworkflow対象pathに含める。空commitやresearch/Markdownのみの変更はpushのpath filterで除外される。先頭のskip-ciを外して、require-build、cluster:firmus、nodes:2、prefill-pp4の各tagを使う。profileは付けない。runtimeのconfig image/provenanceは既存provisionが新buildへ更新するため、投入前に古いpinを架空の新SHAへ手修正しない。mainへのmergeは行わず、terminal状態・gate12GPU・公式parser・dispatch・raw成果物を確認してから判断する。

## 検証と片付け

指定されたrepo直下のCI形式pytestとruffのterminal log・終了コードを指定scratchへ記録する。最終結果は追記する。SSH/Actions/Slurmを開始していないので今回作った残留scheduler jobはない。自分のcompile/検証processのterminalを確認し、展開したsource/cache/生成物を再現用archiveへ整理して不要な診断ファイルを削除する。他作業のファイル・job・branchは変更しない。

最終検証（実装 `67eb56a`、以降は文書・証拠のみ）：

| 検証 | 終了コード |
|---|---:|
| `uv run --project qwen-experiment-metrics --with 'numpy>=1.26,<3' --with pytest python -m pytest -q -p no:cacheprovider qwen-experiment-metrics/tests` | **0** |
| `uv run --project qwen-experiment-metrics ruff check qwen-experiment-metrics scripts containers` | **0** |
| 既存6失敗の関連fileとr28のfocused regression | 0 |
| r28の90秒期限内/超過、shell予算境界、実overlay/割当テスト | 0 |
| sbatch/provisionの`bash -n`、`git diff --check` | 0 |
| native heuristic、CUDA dense、FA3、Triton、実scheduler CPU proof | 0 |

全体pytestはrepo直下、project既定のCPython3.13.12、TMPDIR上書きなし、指定どおりnumpyとpytestをuvに追加した。既存のsource/GPU条件付きskipは残る。初回は6つの固定候補表/設定/worker fixtureで終了1となり、明示的な期待値を新規候補/flagへ更新した。90秒期限の境界を締めた際、修正前に始めた自分の途中pytestだけを止めて終了143を保存し、最終コードに対して上記全体コマンドを再実行して終了0を得た。全runのterminal logとexitファイルはscratchへ保存した。

再現archiveはSHA256とtar一覧を検証済み。自分が作った展開source、compiler cache、重複log、生成/移行用scriptなど59項目を削除し、再現用archive・説明・terminal log・exitだけを残した。提供された分析と自動会話logは保持した。CPU compile/最終pytestはterminal、今回のActions/Slurm/PJM投入・重複job・他branch/mainへのpushはない。GPUのbyte/耐久/速度、通常serving dispatch、公式parser/scoreは投入担当が確認する未実施項目である。
