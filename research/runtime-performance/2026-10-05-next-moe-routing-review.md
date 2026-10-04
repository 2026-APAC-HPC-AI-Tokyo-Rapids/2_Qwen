# 次のMoE部分置換の独立review

更新日2026-10-05 JST。ユーザーは方式を限定せず実測高速化を優先し、全面再実装も許容するが既存小修正で達成できれば優先する。今回はCPU調査・read-only reviewのみ、新build/GPU投入無し。[有限gridと独立確認](2026-10-04-cutlass-moe-grid.md)の両候補gate未達は維持する。既shuffle反復37177667936/PBS216140の健全なQ・監視を変更せず、terminal結果が到着したら正本監査を優先する。

## 優先度と費用

第一候補はprepare_moe_inputの部分置換。固定fc275674 `python/sglang/kernels/aot/csrc/moe/prepare_moe_input.cu:13–45,87–107` はexpert128 CTAそれぞれがR=8T全IDsを走査し、countとmap生成で繰り返す。[P profile](2026-10-04-current-prefill-profile.md)はcompute_arg_sorts114.717ms＋compute_problem_sizes39.870ms、合計154.587ms/active5.004391秒＝3.09%を測定した。39.870msはproblem_sizes単独でありoffset/zeros込みではない。全消去の上限であって予測gainではなく、旧image20c3a383のprofileと現在shuffleonのsource差も区別する。

route tileごとのhistogram→整数prefix→scatterへ置換すればIDs論理読取はT8192で64MiBから0.5MiBへ減るが、元IDsはL2で共有され得るのでHBM削減量と同一ではない。warp-aggregated expert-global cursorはhot expertのCTA間競合を増やす恐れがある。partialcountからexpert内tile-prefixを作り、scatterはtile-local shared cursor＋expert/tile offsetへ書く案を比較すべき。B512ならpartialcount/tileoffset各64KiBで済み、大きいGEMM mainloop再実装を要しない。

第二候補はcombineの小さいblock/vector調整。固定source259–319はBF16値・BF16重みをFP32でslot順に加算してBF16保存する。launcherはScalarType enumのsizeofを使用する。[PyTorch2.11の公式定義](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/headeronly/core/ScalarType.h)はint8_tであり、H4096で256threads、kernel実vec8は2反復になる。既Nsight SQLiteの全6392 combine eventsもblock256/grid8192/register32/shared0を確認した。128/256/512の有限比較は可能だが512へ変えれば速いとは判断しない。P0 active323.798ms/1568call＝206.50us、512MiB read＋64MiB writeの論理量から約2.925TB/s。H1003.35TB/sの単純モデルならtraffic不変の短縮上限は約13%call/約0.82%active窓に留まり、HBM counter未取得・cache効果もある。

down＋combine完全融合やinput gather直接GEMMは大きい対象だが、TMA非affine map・8-way出力所有権・weight reuse・FP32累積/BF16中間丸めが障害。全面再実装を排除しないが、routing部分置換の成立と実測を先に確認し、阻害箇所が特定されてから範囲を広げる。

## 精度・map・graphの必須契約

FP8 quantize-once/scale128・FP32 GEMM accumulator・BF16 down中間・BF16 router weight・FP32 top8 sum・target verificationを保持する。expert内行順は既存atomicでも非決定的だが、同一counts/offsets、a_mapとc_mapの対応、全validroute一度だけ書込み、同じmapで値とscaleを並べることが必須。順序変更が全出力bitwise/relativeL2/oracleにどう影響するかはGPUで確認する。

productionはP/Hopper/EP1/E128/top8/H4096/I384に限定しdefaultoff。-1等は元helperが無視してもcallerはtorch.emptyのmapを全R shuffleへ使うため、wholeMoEの安全対応を保証しない。EP1入力のvalid0..127を前提としinvalidはhelper-only CPU invariantsに分離、未初期化mapを後段へ渡さない。__match_any_syncのmaskはvalidlane/tailで一致させる。T0のhelper no-launchだけでは後段GEMMのstale metadataを防げないためMoE全体skip契約を確認する。

partialcounts/tileprefixは毎graph invocationで全対象要素を更新、scatter cursorは毎回reset、expert offset/countは整数で範囲とtotalRを確認する。buffer形状はCPU metadataから固定allocationし、productionのGPUcounts.item()/hostsynchronizeは追加しない。

## buildと有限測定

新source/.soはnewimageが必要であり66925d03へ外部binaryを差し込んでsameimage対照とは扱わない。既CUDAExtension/Actions/OCI/Enroot基盤は再利用し、最小helper TU/bindingだけをcompileすればよい。不要な16 GEMM template再compileを必須依存にしない案を検討するが、不要module/patch撤去でもimage変更なので同newimageのoff/on E2Eは必要。既SiLU/drafthead/shuffle・c48・D条件の前提を保存する。

CPUはcounts/prefix/offsettotal/map範囲・uniqueness/inverse/各expert区間ID、balanced/skew/hot-only/zeroexpert/T32/4096/8192/tailを検査。GPUは同一短い1GPU allocationでstock・warp-global版・tile-prefix版のstandalone helperとwholeMoEをfinite graph/順序反転/連続warmup/allocatedGPU clock下で比較し、small dense oracle・finite/L2/greedy gateを分離する。新image一buildに候補をまとめ、trialごとのcompile/E2Eは行わない。実routing histogramは未取得なので合成局所勝者だけで採用しない。有望幅が実測されたものだけ同newimage固定pairの公式parser off/on比較へ進む。今回の調査ではそのbuild/GPUは投入していない。
