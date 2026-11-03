# Round 31：Firmus PP4 Prefill MegaMoE の phase 別設計調査

**結論：既存証拠から、3 ms／周期以上の正味短縮が見込める未棄却設計はない。** これは物理的に改善不可能という結論ではない。通常版の内部 phase 時間と round 24 本番の expert 別 token 数は保存されておらず、25.59 ms の差を実測時間として各 phase に配賦することはできない。以下では未確定を残し、source の仕事量、保存実 routing の正確な padding、条件付き模型を分ける。新規の具体案は小さい入口特殊化 1 件に限定し、棄却済み案の再投入は提案しない。

## 条件と証拠

親は main `19ea97e`、作業 branch は `research/megamoe-prefill-design-20261104`。AGENTS.md、astra29/astra30 の最終分析、tracker/model/profile JSON、関連 research、コミット本文、保存された Actions 由来ログを確認した。GPU・SSH・Actions・Slurm は使用していない。

- 対象：`T7988 = 3994×2、E128、top8、H4096、I1536、EP1`。BF16 入出力、FP8 E4M3 の FC1/FC2 積、FP32 scale・carried accumulator、FC1 出力の group128 FP8、FC2 の BF16 partial、FP32 weighted combine。1 周期は **24 call／24 層**であり、1 call が 24 層を処理するわけではない。
- round 24 の `/srv/qwen-artifacts/41423181527/1/profile/profile/nsys-prefill-0.nsys-rep` を指定 scratch の SQLite へ展開した。report SHA256 は `376914ea0cff6103100ad888bc823877b7a64cb66032eb5cbb5e4b234ae73c37` で archive manifest と一致。
- 既存解析と同じ内側窓、相対 `1426516182..8442926294 ns`、PP0/GPU0、24 MegaMoE 開始点で区切る周期を使用。writer grid が `250×72` の pair に限り、norm_quant の 7989 行を含む周期を除外した。**53 周期・1272 call、MegaMoE の 24 call 合計の中央値 63.560586 ms、周期中央値 113.605003 ms**を再現した。request ID に基づく forward 境界ではない。
- 同じ run の GPU UUID map と telemetry を再照合した kernel 稼働中の周期別 SM clock 中央値は **1.410 GHz**。FP8 dense peak は `132×8192×1.410 GHz = 1524.695 TFLOP/s`。別 run の clock は代入しない。
- round 24 の image は `c7804efb`、現設定は `a001f901`。保存 DeepGEMM overlay manifest の **13 本すべてが現 config と順序・SHA256 とも一致**する。r26 proof archive の control source を scratch へ再現し、最終 header SHA256 `fbfaa6be207b26dd7127aae406204b44668b4341ff5ce9e2928510c932e57852` を確認した。保存 CUDA 13.4/sm_90a の同じ instantiation の SASS encoding は **157696 byte、SHA256 `d582cf2059b10cd17523ecf1bcb70135ca5ca7174defa29caf9ad8345f6015ad`**。168 launch registers、math role は setmaxnreg 224、spill なし。これは image の CUDA 13.0.88 の機械語や GPU 動作を新たに検証した証拠ではない。
- profile の launch は **132 CTA、384 thread、dynamic SMEM 228560 byte**。4 stage、lean、split accumulator、tail-pair 無効。EP1 のため expert 間 NVLink 通信は対象外。PP/KV 転送は別の仕事である。

round 24 は profile-only run で、capture/client status は成功だが、保存 validation は「writer grid does not identify a steady two-request forward」で失敗している。全 capture を pair とみなす検査では single 混在を扱えなかった。今回は明示した 53 周期を抽出しており、この失敗を通常 serving の公式 score 成功へ読み替えない。現 image の通常 serving 内部時間を取り直した調査でもない。[round 24](2026-11-02-pp4-profile-r24.md)

## Phase ごとの実測、資源床、差

source は次の構造である。dispatch が expert count・pool を作り、到着した block から FC1 を開始する。FC1 の SwiGLU・amax・quant・store 後に arrival を公開し、FC2 の A loader が ready を待って動的に block を取得する。CTA 間では FC1 と FC2 が重なり得る。最後に grid 同期と workspace の self-signal、dispatch との同期を経て combine する。**dispatch、FC1、FC2 の CTA 時間の中央値を足して全 kernel 時間にはできない。**

`R=T×8=63904` とすると、有効積は FC1 `4RHI`、FC2 `2RHI`。FMA は 2 FLOP。以下はすべて **ms／24 層周期**である。

| phase | round 24 の内部実測 | 有効仕事の資源床・参考換算 | 実測−床 |
|---|---:|---|---|
| dispatch | 未確定 | 主要論理 I/O 7.288 GB、4.8 TB/s 換算 **1.518**。厳密な HBM 床は未確定 | 未確定 |
| FC1 mainloop | 未確定 | 有効 **38.597 TFLOP、Tensor 床 25.314**。FC1 epilogue 出力を含む主要論理 I/O 47.572 GB、同換算 9.911 | 未確定 |
| SwiGLU・amax・group128 quant | 未確定 | 有効 SwiGLU 出力 98156544 個／層。既存 EX2 と sigmoid RCP の計 47.115 億結果／周期、16 結果／SM／clock のサービス床 **1.582**。reduction・変換・store は別途必要 | 未確定 |
| FC2 mainloop・BF16 partial store | 未確定 | 有効 **19.298 TFLOP、Tensor 床 12.657**。主要論理 I/O 34.325 GB、同換算 7.151 | 未確定 |
| combine | 未確定 | BF16 partial 523.502 MB＋BF16 出力 65.438 MB＋top-k weight 0.256 MB／層。計 14.141 GB／周期、同換算 **2.946**。厳密な HBM 床は未確定 | 未確定 |
| 同期・待ち、fill/drain | 未確定 | FC1→FC2 acquire、stage full/empty、grid 同期、最後の CTA など。必要時間・除去可能時間とも未確定 | 未確定 |
| **MegaMoE 全体** | **63.561** | **有効 Tensor 床 37.972** | **25.589、効率 59.74%** |

I/O は主要 buffer の一回通過模型であり、metadata の全アクセス、再読出し、L2 hit/writeback を再現した実 HBM byte ではない。weight は全 128 expert が使われる場合で、本番の active expert 数は未確定。FC1 の I/O 欄に SwiGLU/quant 後の pool 出力が入っているため、同じ byte を epilogue 欄へ再加算しない。Tensor・SFU・転送は重なり得るので、表の床・換算時間を合計しない。

nsys はこの内部を 1 本の persistent kernel として記録しており、phase 境界、expert counter の値、各 TMA の実 byte を記録していない。r19 の細分 trace は **通常版の約4.4〜4.7倍**になり、細分時間は採否に使えない。旧 EP4 trace の combine 78〜81 µs、FC2 scatter 182〜209 µsなどを現在の EP1 へ流用しない。kernel 間 idle 約0.474 ms／周期も、kernel 内部の25.589 msとは別である。[r19](2026-11-01-prefill-megamoe-diag-r19.md)、[旧 timeline の訂正](2026-10-24-prefill-megamoe-timeline.md)

## Routing と tile 量子化を実分布から数える

正本は既存 `kernels/probes/megamoe-local-ep1-routing.npz`、SHA256 `1907184191e49a6dcc1431cdadafe9bfef2eb90bf4ad3f58407f6ed32a136b52`。`logical_ids` は `(2 sample, 3 layer, 4 old EP rank, 4001 token, 8)`、layer は `11,22,3`。全行で expert ID の範囲と top8 の非重複を検査した。

この保存 capture は旧構成の実 routing であり、**round 24 の本番 routing ではない**。従来の CPU 検査と同じく、24 個の単要求分布と、rank `0+1`／`2+3` を連結した12個の二要求相当分布を全数計算した。後者は現在の scheduler が実際に組んだ pair とは主張しない。r20 の本番 capture 追加 pass は15秒期限で未完了である。[r20](2026-11-01-prefill-megamoe-sched-r20.md)

expert ごとの行数を `n_e` とすると、

```text
R = Σ n_e
B128 = Σ ceil(n_e / 128)
Q = count(1 <= n_e mod 128 <= 64)
R128 = 128 B128
R64 = Σ 64 ceil(n_e / 64) = 128 (B128 - Q/2)
padding FLOP = (R64 - R) × 6HI
```

pool は M128 単位だが、空の後半 WG は MMA を省く。**現行の Tensor padding は M128 丸めではなく M64 丸め**。異なる expert は異なる weight を使うため、M tile を expert 境界の向こうの token で埋めることはできない。

次表は12分布すべて。有効行は各64016、active は非空 expert 数、中央値・最大は expert ごとの行数。追加 Tensor 床はその分布を24層に反復した **1.410 GHz の条件計算**である。

| sample/layer/old rank pair | active | token数 中央値／最大 | CV | B128 | Q | R64 | padding率 | 追加 Tensor 床 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0/11/01 | 110 | 34／7938 | 2.636 | 575 | 77 | 68672 | 7.273% | 2.767 |
| 0/11/23 | 112 | 39／7922 | 2.513 | 575 | 84 | 68224 | 6.573% | 2.500 |
| 0/22/01 | 106 | 26／7974 | 2.251 | 572 | 75 | 68416 | 6.873% | 2.614 |
| 0/22/23 | 108 | 45／7972 | 2.167 | 571 | 74 | 68352 | 6.773% | 2.576 |
| 0/3/01 | 114 | 149.5／7965 | 1.829 | 573 | 80 | 68224 | 6.573% | 2.500 |
| 0/3/23 | 116 | 184.5／7962 | 1.764 | 576 | 84 | 68352 | 6.773% | 2.576 |
| 1/11/01 | 110 | 22／7959 | 2.705 | 578 | 85 | 68544 | 7.073% | 2.691 |
| 1/11/23 | 110 | 23／7945 | 2.646 | 580 | 86 | 68736 | 7.373% | 2.805 |
| 1/22/01 | 103 | 28／7975 | 2.258 | 567 | 69 | 68160 | 6.473% | 2.462 |
| 1/22/23 | 106 | 34／7975 | 2.230 | 576 | 83 | 68416 | 6.873% | 2.614 |
| 1/3/01 | 112 | 130.5／7971 | 1.869 | 570 | 74 | 68224 | 6.573% | 2.500 |
| 1/3/23 | 112 | 154.5／7973 | 1.796 | 569 | 71 | 68288 | 6.673% | 2.538 |

24単要求分布の M64 padding は **11.772〜15.771%**、active expert は94〜115。小 batch へ割るだけでは padding 比率が増える。二要求相当の expert CV は1.764〜2.705と大きいが、現行 FC1 の `12×M block` item を132 CTAへ round-robin する模型の **最大／平均 MMA 量は1.014〜1.048**にとどまる。hot expert の token 数だけから同じ倍率の CTA 待ちを推測してはいけない。FC2 の atomic claim 後の割当と ready 到着時刻は、この分布だけでは確定しない。

既存 gate の historical は別の組み方で、`logical_ids[:,0,0,:3994,:].reshape(7988,8)`。旧 sample 2個を各3994行へ切ったものである。全128 expert の行数は既存 r20 CPU JSON と完全一致し、次を得た。

| historical の指標 | 値 |
|---|---:|
| 有効行／active expert | 63904／108 |
| expert 行数の最小／中央値／p90／最大 | 0／21／1232.4／7947 |
| B128／Q／65〜127行の tail 数 | 571／75／31 |
| R128／R64 | 73088／68288 |
| 現行の padding | **4384行、6.860%** |
| padding の追加 Tensor 床 | **FC1 1.737＋FC2 0.868＝2.605 ms／周期** |
| FC1 MMA 担当量の最小／平均／最大 | 47.5／48.5／50 full-tile単位 |
| hypothetically 全expertをM16丸めにした行数 | 64816、現行からの Tensor 仕事削減は **2.063 ms**相当 |

最後の M16 は実装・速度予測ではない。r26 の N16/N32 swap-tail はこのうち2576行を除いたが、発行数と wait が増えて実時間ではほぼ相殺された。

balanced fixture も有効63904行だが、499／500行の分布なので R64 は65536、padding の床は0.970 ms。historical との差は **1.635 ms の追加 Tensor 仕事**。r19 の計装なし完全 kernel では historical が balanced より約130〜140 µs／層、24層で約3.1〜3.4 ms遅かったが、これを「routing を並べ替えれば3ms回収」とは解釈できない。balanced はモデルの routing を変えた対照であり、変更できない有効演算がある。padding、半tileの固定費、ready、clock の寄与は未分離である。

したがって、保存分布で padding の全積を理想的に消す Tensor 予算は2.46〜2.80 ms。これは本番分布の上限の証明でも、完全 kernel の短縮上限でもない。追加 rescale・epilogue の無駄も変わり得るが、その正味短縮は未確定である。

## Scale 依存に残る差の規模

現行 split の定常 K128 block は、active math warp 当たり **QGMMA 8本、accumulator FMUL 128本**。各 half の依存は「前の同じ half の完了→64個の独立 FMUL→次の QGMMA」。64段の直列 FMUL ではない。FC1 は32 K blockを8個ずつ、FC2 は12個を一つの chunkとして処理する。weight/activation scale が K ごとに異なるため、最後の scale 一個へまとめることはできない。

旧 H100 probe の split は1272〜1278 clock／K128、Tensor サービス床は1024 clockだった。同じ比が H200 でも成り立つと仮定した場合、historical の padding 込み Tensor 床40.577 msに対する超過の規模は **9.827〜10.065 ms／周期**になる。これは異なる GPU・run の **条件付き模型**であり、round 24 の FC1/FC2 の実測内訳ではない。25.589 msから差し引いて残りを epilogue や待ちに割り当てない。[consumer source/SASS 解析](2026-11-02-prefill-megamoe-consumer-review.md)

3 ms は平均 **125 µs／call、MegaMoE全体の4.72%**である。この模型の超過を対象にしても、約30%を追加費用なしで減らす必要がある。既採用の split をもう一度利得に数えたり、FMUL の本数だけでその目標を達成した扱いにはできない。

## 過去案の棚卸し

旧 EP4 の結果と現 PP4/EP1 の結果を区別した。表の過去結果は、今回測った値ではない。

| 案 | 結果・棄却理由 |
|---|---|
| FC2 動的取得、連続 load combine、local EP1、4stage/lean/split | **既採用**。動的取得や同期省略を新規の削減に数えない。[10/18 tail](2026-10-18-megamoe-prefill-tail.md) |
| independent WG／ping-pong | 旧EP4で **+22〜36%**。stage 減少と B TMA 重複が epilogue overlap を上回った。[timeline](2026-10-24-prefill-megamoe-timeline.md) |
| FC1 early release | 同じ旧EP4で **+1.1〜2.0%**。H100 単体 probe の利得は H200 へ移らなかった。同上 |
| WGMMA 発行交互化、WG chunk 境界ずらし、FC2 分割 | 交互化は K loop +5%、境界ずらし +2.3〜4.3%、FC2 2 chunk +2.9%。待ちだけ減らしても総時間が増加。[split](2026-10-30-prefill-megamoe-split-accumulator.md)、[LDSM](2026-10-30-prefill-megamoe-split-ldsm.md) |
| bulk scatter | 旧EP4で **+1.3〜4.9%**。TMA loadとstoreの競合でstage待ちが増え、partialの書出しも短縮しなかった。[ceiling review](2026-10-24-broad-ceiling-review.md) |
| direct FC2 store | EP1で **+1.6%、約43µs／層**。store要求倍増、combine bank conflict、store queue の待ち。[EP1 accounting](2026-10-30-prefill-megamoe-ep1-accounting.md) |
| FC1 A gather | EP1で **+44%、約1150µs／層**。重なっていた pool copy を消し、loader に大量の cp.async を載せた。同上 |
| tail split／tail-pair | split は2回margin未達。pair は r3で **約+10%**、r4も不採用。2 stage 消費と制御費用。[r3](2026-10-30-pp4-round3-9668.md)、[r4](2026-10-30-pp4-round4-9832.md) |
| split-LDSM | 単体模型で改善しても完全 kernel は **+1.6%**。候補から撤去。[shared RCP note](2026-10-30-prefill-megamoe-shared-rcp.md) |
| shared reciprocal | 合計約0.3%速いが、全GPU・全routingで600回目が自身の初回とbyte不一致。**耐久不合格**。[r7](2026-10-30-pp4-round7-timeout.md) |
| combine＋次層norm融合 | 24セル中22で遅い。buffer往復削減の見積もりは完全 kernel の改善にならなかった。同上・shared RCP note |
| FP32 precombine／combine direct store | 旧EP4で前者は serving **−5.8%**、scratch増。後者は数値一致でもpaired速度gate未達。[model](kernel-theory-model.json) |
| FC1 expert順 greedy割当 L | **全12 GPU・全3routingで+38〜166µs／層**。割当計算と再利用悪化が最大負荷低下を上回った。[r20](2026-11-01-prefill-megamoe-sched-r20.md) |
| N16/N32 swap-tail | 44 case・12GPUの中間byte一致と1200耐久は合格。MMA約3.8%削減でも historical **−0.6〜+0.6%**、hot最大+2.8%。発行約5.4%増とwait増。[r26](2026-11-03-prefill-megamoe-swap-tail-r26.md) |
| polling backoff 64ns | byte一致・1200耐久合格だが historical **+1.10〜2.26%**。到着検知の遅延が勝った。[r27](2026-11-03-prefill-backoff-normfma-r27.md) |
| scale先読み移動／合成scale持越し／inline PTX集約 | CPUで依存短縮なし、既に compiler が再利用、または60/80B spillと命令増。[r13](2026-11-01-prefill-megamoe-dep-r13.md)、[consumer review](2026-11-02-prefill-megamoe-consumer-review.md) |
| FFMA型 promotion・別 accumulator | 旧形状で1856B spillとWGMMA直列化。現行も1要素1FMULで、FFMA化だけでは命令数が減らない。[mainloop](2026-10-18-megamoe-prefill-mainloop.md) |

## 新規に具体化できる設計：FC1 初回 K128 の入口特殊化

**3 ms級の採用候補ではない。** 以下の1件だけを小案として記録する。棄却済みの scale先読み移動・合成scale再利用・swap-tail とは、削る仕事が異なる。

- **削る仕事：** FC1 の最初の K128 は accumulator がゼロ、倍率が1だが、外側に4 chunkのloopがあるため、現 SASS の `0x2090..0x2cf0` に128本の accumulator FMUL が残る。最初の K128 を loop の外へ出し、各halfの最初の K32 QGMMAだけ scale-d=0、残りのK32は通常accumulateとする。最初のゼロ初期化と identity rescale を削る。残り31 K128の carried scale、store、arrival は維持する。**各chunk先頭を全部省くのではない。** FC2 は既に `0xc920/0xca80` の QGMMA が `RZ, !UPT` で初期化を吸収しており、FC2 分の利得を加えない。
- **量的根拠：** historical の `68288×3072×24 = 5034737664` 個の scalar FMUL、32 lane換算で157335552 warp命令相当を対象とする。128 FP32結果／SM／clock での命令サービス量は **0.211 ms／周期**。ゼロ初期化も同じ要素数分のissueを全て消せるなら、追加の命令予算も約0.211 ms相当。ただしMOVの実発行率、隠れている時間、compile後の置換は未測定で、**約0.42 msは楽観的な命令量換算、正味短縮の予測ではない**。既存実測の短縮はない。3 msを裏付ける依存短縮の根拠もない。
- **精度：** FP8/BF16/FP32 の形式・scale・group128・全top8・全K積を保持する。モデルの計算を省略せず、初期値ゼロに対する恒等処理のみを除く。有限値では同じ結果を意図するが、signed zero、Tensor Core内部の初期累積、NaN/Inf、ftzをCPUだけで保証しない。中間と最終のbyte gateを要求する。
- **変更と規模：** DeepGEMM `impls/sm90_fp8_mega_moe.cuh` の split入口と host側候補selector。利用側の `patches/deep-gemm-bench/prefill-first-k/sm90-prefill-first-k.patch`、`patches/deep-gemm-host/sm90-prefill-first-k.patch`、`scripts/select-firmus-prefill-megamoe.py` と配備・gate検査が想定対象。数百行規模の小〜中変更。実装するなら source変更なので **fresh build が必要**。今回は作成しない。
- **主なリスク：** 最初のchunkの別展開によるcode増、register割当・spill、ptxasによるWGMMA直列化、stage/phaseと遅延publishの境界、初回JIT/load。定常loopの命令増で小さい削減が容易に消える。
- **CPU gate：** 現CUDA13.0.88と代替13.4で実instantiationをコンパイルし、初回の命令除去、通常loopの非増加、他geometryのSASS不変、spill増なし・C75xxなし、stage/arrival exact-onceを確認する。CPUで正の差がなければ終了する。
- **将来のin-job gateと耐久：** 完全P gateを先に終え、残り時間だけで全12 Prefill GPUのUUIDを照合。r26相当の44caseでFC1 FP8・scale・FC2 BF16 partial・最終BF16をbyte照合し、独立FP32参照の既存閾値も維持する。非計装の完全MegaMoEを同一resident processで両順12round計時し、各GPU・各routing・両順の中央値で `max(5µs, 0.5%)` を超えることを要求する。6routing・2入力・2bufferを切り替えて1200call、途中の比較と元入力への復帰を確認する。timeout・欠測・不一致は不採用。3ms候補と呼ぶにはさらに **平均125µs／call以上**が必要で、小さいselector margin通過では代用できない。
- **ready前準備と最終検証：** 実装を進める場合は採用後のserving各process・実shapeでready前に初回実行を済ませる。r30で別processのgate/cacheだけでは不足すると確認済み。公式窓を変えず、同条件の通常servingを複数回、公式parser・eligibility・品質・TTFT/ITLとともに確認する。[r30](2026-11-04-prefill-qkv-tail-prewarm-r30.md)

## 判断と完了確認

**3 ms／周期以上が見込める設計は、今回の証拠では「ない」。** 3msは指定の換算でscore約+260の感度に相当するが、公式scoreの達成予測ではない。新規小案の正味時間も未確定なので、3ms目標に対する実装・build・GPU検証計画は開始しない。

判断を変える証拠は、通常版への影響を定量評価できるphase測定、同じ本番forwardのrouting、または定常scale依存を追加spill・発行・waitなしで大幅に短縮する具体的SASSである。現存データにはこれらがなく、CPUのみという今回の制約内では取得できない。未確定をゼロにしたり、旧EP4/H100の区間を合成して25.589msを埋めることはしない。

検証は、保存reportのhash照合、SQLiteのread-only解析、53周期の再現、同runのclock照合、13overlayの順序/hash照合、header/SASS hash照合、全36routing分布とhistorical/境界式のCPU数え上げ。Pythonはuv経由。変更は本noteとtrackerの1段落のみで、実行コード・config・model JSONを変更していない。展開SQLite・source・SASS等の中間物は片付け、指定scratch `codex-scratch/mm31` に再現手順・script・集計JSON・logだけを保持する。本作業由来のjobはなく、既存jobには触れていない。
