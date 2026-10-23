# Decode 4候補の直接ソース再レビュー（2026-10-23）

ユーザーの依頼に基づく実行前レビュー。ジョブ投入・GPU実行・pushは行わない。初回は直接ソースを点検し、後続ではCPU検査、実CUDAコンパイル、読み取り専用のSSHによるコンテナ取得まで追加した。今回の結論はソースの条件付き整合性に関するもので、GPU数値一致、全interleaving、CUDA graph replay、性能を保証しない。

## 対象と方法

image header `ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1` に `configs/firmus-framework.json` の5 overlayを順番に適用した基準headerを使用。基準SHA256は `77ee7caea6e5736803b6034f83575f9a846d30c54ce4970ced7c65450a9be497`。Git worktreeの外に作った一時directoryで適用し、patchのsilent skipを避けてhashを照合した。

| 案 | レビュー開始時のcommit | 候補header SHA256 |
| --- | --- | --- |
| A sparse dispatch | `5becac9` | `360d066b26f472acb2747d6ddf2c4248a021bdcf44ab5122f84a617d2a2025e8` |
| B FC1 register | `fd5a14e` | `fa4a86e57417b92dd7dfd25b30b4c7c1f38ac04ac4264656c8876779a58378b2` |
| C independent WG | `04de958` | `952e0eb1f476455bb48cabd193017797505c5a039783cdfdd65e97551a299b5e` |
| D streaming combine | `1cf40e4` | `b41bdda2ef87c05aa4e3363a26d8be2b4943258d19da9bf5284559f4b7644189` |

実適用後のCUDA本体と、host選択、scheduler、workspace、grid/NVLink barrier、PTX wrapperを追った。B・Cは実装担当とは別のAstraがレビューし、Dは親担当と別のAstraが独立にレビューした。Aでは既存compileのSASSも参照した。今回、新しいGPU実行やCUDAコンパイルはしていない。

## 指摘と判断

### C: 独立WGと互換性のないテンプレート設定の拒否が必要

独立WGのloaderはqueueごとの動的task列を処理する。一方、`kSplitPhaseHotPath=true` ではmath側が `kSpecializePhase` の分岐を先に選び、共有の静的schedulerを処理する。この組み合わせは、同じstageに対するloaderとmathのtask/phase対応を壊す。

同様に、`kL2ArrivalCounter=true` はFC1の加算カウンタ通知と独立loaderのbit-mask待機を食い違わせる。`kReuseAccumAsFinal=true` はSwapAB用のfragment計算より前の別計算経路を選び、独立WGのepilogueが読む配置と合わない。

現行hostはQwen Decodeでこれら3 flagを全てfalseにするため、現在の実験条件からは到達しない。ただしkernelのテンプレート契約として拒否していなかったため、独立WGモードでは3 flagがfalseであることを `DG_STATIC_ASSERT` で要求する制約を追加した。独立WGが無効な既存modeには制約を加えない。適用後の検査結果はCの候補別ノートにも記録する。

修正はCのローカルcommit `194c25e` に保存した（未push、main未merge）。修正後header SHA256は `c40d4c141aea45bb17a864ae8d7088a97363f12d0188b2584e558203aacb11ad`。実patchのassert式を通常C++に抽出して16通りを構文検査し、現行独立WGの1通りとopt-outの8通りは成功、非対応の7通りはそのassertで失敗した。今回の変更はcompile時の制約のみで、CUDAカーネル本体の再compileは行っていない。

独立queue間のstage/ring/output分離、stage解放の4 warp、TMA store boxとhost shared-memory量の一致も確認した。schedulerのexpert/phase/wave状態を追う別CPUモデルは、wave=1/2/4/8/16/32を含む102条件で独立したtask列と一致した。これもCUDAの実行ではない。

### A: 新たな具体的不具合は未検出

- low-wordの保守的上限は `4224 * 8 * 4 = 135168`。32-bitからhigh-wordへのcarryは起きない。high-word設定前のuint64変換もある。
- ゼロ件CTAはsource entry自体を持たず、省略した予約のoffsetを使用しない。全paddingでもSM0は全expertのreadyを公開する。
- grid同期後にcountを読み、high-wordを132として公開する。non-SwapAB rankも132を公開し、受信側の期待値528と合う。既存SASSでもgrid acquire完了→CTA barrier→count load→system publicationの順だった。
- 既存math完了barrier、dispatch/math join、counter cleanup、最終rank barrierは変更しておらず、新たなreplay状態はない。

### B: 新たな具体的不具合は未検出

- xor4/8/16によるmax集約はtokenを変えず、両WGの64出力列を集約する。xor4/8によるFP8 packも元の列順へ対応する。
- scratchのamaxはfloat index `[0, 512)`、inverse-SFは `[512, 576)`。FP8 tileは別のbyte offset 16384以降で、重ならない。
- 3 barrierでpartial amax→SF→FP8を順序付ける。前taskのTMA store waitを次taskの第1barrier前に通り、scratchは前taskの読取り完了後に再利用する。
- valid Mの1〜64、shuffle先、書込み所有、初期化済み領域の読取りを追加列挙した。NaN/Inf/signed-zeroについてsource上の演算手順に矛盾は見つからなかった。

### D: 二重レビューで新たな具体的不具合は未検出

- scatter writer全員のsystem fence→全math barrier→leaderのsystem acq_rel RMWに対し、consumerはremote acquire→warp同期→BF16読取りとなっている。
- 前回修正済みのinactive M WGもpublication barrierへ参加する。有効Mが存在すればleader WGはactiveで、通知経路を通る。
- tile寄与 `valid_m * (BLOCK_N / 128)` のchunk内合計は `expert_recv_count * 4`。BM64/128、BN128/256/512で一致する。
- streaming consumerは自分のpull完了後に入り、GEMMのshared bufferやproducer roleを保持しない。全FC1がFC2より前のtask indexにあり、循環待ちを作る依存は見つからなかった。
- math完了rank barrier→ローカルjoin→consumer完了rank barrier→cleanupの順。共通NVLink phase counterを役割間で同時使用せず、他rankが参照中のexpert countを消さない。
- 各token/chunkの所有は一意。32 lane × 8 BF16 pairで512列を処理し、slot昇順のFP32 FMAと最後のBF16丸めを保つ。

## GPUなしの再検査と限界

- A: 210 reservationケース＋64 mixed/idle-rank・replayケース、計274ケースが成功。
- B: 既存CPUテスト2件成功。追加のsource対応列挙でもvalid M 1〜64を確認。
- C: 実assert式のhost compile 16通り、既存CPU配置・task checkerが成功。変更patchの適用、Ruff、差分の空白検査も成功。
- D: 既存CPUテスト16件成功。

上記PythonテストはCUDAの実行ではない。特にDのpublicationモデルは「データが通知より前に公開される」ことを前提にしており、GPU memory orderingの証明ではない。Bのテストも任意のencoded byteの配置検査で、実際のSF helperやFP8命令を実行しない。Pythonの数値等価比較だけではsigned-zeroのbit一致を証明しない。

現行hostの対象条件で、今回新たに実行結果を壊す具体的な箇所は見つからなかった。ただし、これは問題が存在しない証明ではない。特にC/Dの非同期実行・GPU間可視性・変更入力のgraph replayと、Cのhost extensionを含む新規buildは未検証のままである。

## 後続確認: 数値gate自体のmixed-geometry参照を修正

ユーザーから、キュー待ち後の初期整合性チェック失敗を避けたいとの指摘を受け、`scripts/check-firmus-megamoe.py` の参照計算も確認した。従来は `max(counts) >= 256` によって全rankのFC2 activation量子化をSF128へ切り替えていた。しかし各rankが計算するlocal expertのgeometryは、そのrankの出力tensorの行数 `y.size(0)` に依存する。

既存の `mixed-geometry-graph` 入力 `[7, 0, 145, 256]` では、参照側のgroupが全rankで128になっていた。本来は `[64, 64, 64, 128]` である。旧sourceから選択式を抽出してCPUでこの食い違いを再現した。これはgateの比較対象に余分な量子化差を混入させるもので、実際に許容誤差超過が起きたと実測したわけではない。

参照groupを各rankで実際に確保する出力行数から選ぶように修正し、idle Prefillの256 masked rowsも同じ経路で扱う。許容誤差は変更しない。gateの既存JSON行へ使用したgroupを追加した。実sourceの選択式を使ったGPU不要回帰ケースを5件追加し、mixed geometry、255/256境界、idle Prefill、all idle、SwapAB 144を確認する。

kernel-bench fixtureは29件成功・任意依存の3件skip。Ruff・diff checkも成功。mainのローカルcommit `b52ea30` をA/B/C/Dへcherry-pickし、それぞれ `ee9a2a1` / `73bc3af` / `a5bc797` / `0fc30c5`。4候補の修正対象2ファイルがmainと同一であること、各worktreeがcleanであることを確認した。未push。

この修正でGPU数値検証を代替できるわけではない。特にCの新host extension buildと、B/Cの生成GPU命令による量子化値の一致、C/Dの同期・replayは未検証のまま。ジョブ投入・GPU実行・pushはしていない。

## 後続の投入前検査（同日、GPU実行なし）

先の「CUDAコンパイル・host build未検証」は、以下の結果に置き換える。GPU上の数値・同期・replay検証は引き続き未実施。

- 数値gateの参照group修正に、別の変数上書き不具合があった。確保する行数 `rows` をexpertループの `torch.where` の戻り値が上書きし、`torch.empty` へtensorを渡す構造になっていた。`output_rows` に分離して `1e9291a` で修正した。回帰検査は選択式だけでなく、実sourceのexpertループから出力確保まで実行する形へ強化した。
- kernel-benchのcontrol/candidate双方で、servingと同じ `DG_SM90_PREFILL_2WG=1`、`SGLANG_MEGAMOE_IDLE_PREFILL_GEOMETRY=1` を渡す。masked idle Prefillケースも実行対象にする。
- 4候補は出力要素の演算を保つ設計なので、数値gateに加えてcontrolとの出力hash一致を要求する。他の算術変更候補までこの制約を拡大しない。replay未確認・非決定的結果・case欠落・空結果も拒否し、controlのbench/replay不成立時に候補だけOKと表示しない。これらは `789e91c` で保存し、4候補へ反映した。
- 共通kernel-bench fixture 48件、Ruff、Bash構文、差分検査が成功。今回はtorch/numpyを導入したCPU環境を使い、任意依存によるskipはない。

### 実CUDAコンパイルとhost ABI

`/tmp/qwen-preflight-20261023/four-candidate-compile/` に生ログ、command、cubin、SASS、manifestを保持する。CUDA nvcc 13.0.88、実CUTLASS/DeepGEMM依存を使用し、現在のgateの128/140 expertsについてcontrol 10、A 10、B 10、C 12、D 10の計52特殊化が成功した。hostが実際に生成したtemplate引数を使用しており、CUDA命令のstub置換はない。通常/partial wave、SwapAB/nonSwap、Prefill 2/4WG、Cの6-stageとopt-outを含む。変更されない経路のSASS比較22件も一致した。

Cの6-stage Decodeは168 registers、spill store/load各8 bytesで、zero-spill目標は未達。これは性能上の課題であり、compile失敗ではない。A/B Decodeはspillなし。実測latencyは取得していない。

Cは実v0.1.5 sourceと4本のhost patchから、リポジトリの `containers/rebuild-deepgemm-host.py` のbuild関数でTVM-FFI `_C.so` をビルドした。実torch 2.11.0、apache-tvm-ffi 0.1.11、sgl-deep-gemm 0.1.5とのimportが成功した。driver stubに対するdevice-count照会の警告はあるが、GPU kernelは実行していない。extension SHA256は `60aebb872919fd3a25d3fa9d98115b6ee895d2debc41e605b77eb042d1b9bacc`。正本は `/tmp/qwen-preflight-20261023/host-build-result.json` とbuild log。これは本番コンテナ全体の新規build成功を意味しない。

### 数値境界とhost/kernel契約

Bの `scripts/check-decode-fc1-arithmetic.py` は実C++の演算式を抽出し、公式CUDA host FP8変換と実CUTLASS min/max/absを使用する。144196量子化値について、全有限E4M3丸め中点と上下1 FP32 ULP、signed zero、subnormal、scale下限、NaN/Inf等を検査した。非NaN中間値、scale、最終packed FP8は対照と一致し、独立したnearest-value/ties-to-even参照とも一致した。NaN中間値のpayload/符号差は明示記録し、同一視して隠していない。`expf`/reciprocalはhost surrogateなのでGPU固有命令の証明ではない。結果と変異検査は `/tmp/qwen-fc1-host-arithmetic-20261023/`、実装・説明はBのcommit `4b978d6`。

Cの `scripts/check-sm90-independent-wg-host.py` は実host heuristic/JIT文字列生成を432条件で実行し、17種類のtemplateについて実kernel constexprのSMEM・threads・7 TMA boxes・25引数との契約を確認した。CPU上のdescriptor/launch stubによる契約検査であり、CUDA実行ではない。結果はCの `research/runtime-performance/2026-10-23-independent-wg-host-contract.json`、commit `48cccef` に保存した。

raw idle rankについて補足する。現在のgateはzero live tokensでも出力を1行確保し、hostはSwapABを選ぶ。0行のnonSwapだとした以前の口頭説明は誤りだった。masked idle Prefillでは128 experts時256行、140 experts時280行でPrefillを選ぶ。上記compile/host検査は実際の出力行数を使用している。

### Cの投入順序

現行pinのimage `39899235870` にはCのhost変更がない。このままCのGPU jobを投入してはならない。C branchのprovision guardはhost patch SHA、TVM-FFI ABI、実 `_C.so` SHAを照合し、旧imageをscheduler投入前に拒否する（`e3fd55e`）。新規container build、成功したartifactへのconfig pin更新、provisionの照合成功の順序が必要。新imageができる前にhashやartifact IDを仮置きしない。

この順序を可能にするため、既存workflowで既に用意されていたFirmus build-only経路をtag validatorでも許可した（main `4919de0`、4候補へ反映済み）。21 CPU fixtureで実workflow条件を評価し、`[require-build][build-only][cluster:firmus]` がcontainerだけを選び、serving/GPU jobを選ばないことを確認した。新buildの配備は失敗許容stepなので、Actions全体成功だけで判断せずimage/sidecarの存在・hash・size・source commitを確認してからpinを更新する。

静的点検・CPU検査・compileで判明した不具合は修正したが、C/DのGPU間同期、入力を変えたCUDA graph replay、実GPU丸め、公式score、126 us達成は今回の結果だけでは確認できない。GPU待ちキュー前に可能な検査と、実GPUで初めて確認できる事項を区別する。

## 実イメージを使用した最終provision検査

取得した `39899235870.sqsh` は12,270,067,712 bytesで、実provisionerのSHA256照合がconfigの `3b3d4a8d46671f6264204b35ea797e3f34aad91e6ec806ae252ece77ec22f8c8` と一致した。各branchからscripts/configs/platforms/patchesをGit外のローカルdirectoryへcopyし、configのimage pathだけを取得先へ変更して、実 `scripts/provision-firmus-kernel-bench.sh` を最初から最後まで実行した。scheduler scriptは生成だけで実行しない。

| 候補 | 検査対象commit | 実provision結果 |
| --- | --- | --- |
| A | `ea79ef3` | exit 0、候補header SHAは52特殊化compileの対象と一致 |
| B | `6967826` | exit 0、候補header SHAは52特殊化compileの対象と一致 |
| C | `d3860eb` | exit 1、実旧imageのhost patch不足をguardが検出。意図した投入前拒否 |
| D | `dc8304e` | exit 0、候補header SHAは52特殊化compileの対象と一致 |

正本は `/tmp/qwen-preflight-20261023/real-provision-results.json` と同directory下の `real-provision/`（各log、生成script、provenance、variants、適用後source）。これはfixtureの展開mockではなく、実unsquashfs・実image・実patchを使用した検査。SSHはimage/sidecarの読取りのみ。

compileに使用した依存についても、実imageのDeepGEMM header 56 files、CUTLASS/cute 802 filesがすべてhash一致した。compile用include treeにだけ存在するtimeline headerは今回の非instrumented候補からincludeされない。実imageのtorch 2.11.0、apache-tvm-ffi 0.1.11、sgl-deep-gemm 0.1.5も確認した。numpyは実image 2.3.5、CPU fixture環境2.5.4で差があるが、今回のhost拡張はnumpy C ABIを使用しておらず、数値境界checkerもnumpy計算に依存しない。

実imageから抽出したtorch・tvm_ffi・deep_gemmに候補 `_C.so` を組み合わせ、driver stubでABI importを再確認した。3 packageとも実image抽出先からimportされ、両entrypointが存在することを確認した（`image-abi-result.json`、`image-abi-import.log`）。旧image自体は変更していない。この検査を新imageの完成とは扱わない。

また、Cの実Dockerfileが呼ぶ `containers/check-sm90-independent-wg.py` 全体をローカルで実行し、5 compile・対照SASS一致・直列化warningなし・結果JSON生成までexit 0を確認した（`container-compile-result.json`、`container-compile.log`）。assert追加後headerでも既存tableのSASS値と一致し、8/8-byte spillは引き続き未解消。

### 残る投入条件

A/B/Dは現pinで投入前のCPU検査とprovisionを完走した。Cは新規imageの完成が必須で、現pinのGPU数値gateへは進めない。既存builderは固定CUDA OCIからBuildah chrootで全レシピを構築し、Skopeo/Umoci検証後にSquashFSを生成する。旧SquashFSを入力とする差分更新経路はない。`_C.so`だけ差し替えた再packでは埋込host記録とmanifestが古くなり、image検査を満たさない。今回のためだけに別のimage構築経路は追加しない。

次にCで実行すべきものは既存のCPU build-only jobであり、成功したartifactとsidecarを照合して4つのpin値を更新し、provisionが成功した後にGPU数値gateへ進む。ジョブ投入禁止を維持しているため、新imageは未作成。全4案が現在のconfigのままGPU jobを投入可能だという結論にはしていない。
