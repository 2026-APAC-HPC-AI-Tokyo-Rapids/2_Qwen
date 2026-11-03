# Prefill の kernel 区分ごとの理論床との差（2026-11-03、round 27 の後、分析のみ）

- **作成：** gpt-6-astra の分析（入力は round 24 の nsys、run 41423181527、worker 0・PP0、代表 53 周期と、round 27 の telemetry、run 41622902468）。Claude が記録した。
- **扱い：** 理論床・条件付きの模型・実測を区別している。round 24 は image c7804efb の profile 専用 run なので、現 image の serving で測り直した値ではない。

以下は分析の全文である。

**最大の差は MegaMoE です。実 clock 1.410 GHz で、有効 FP8 演算の床は約37.97 ms、実測は63.56 ms／周期、差は約25.59 msです。次は FA3 の約10.72 msです。**

ただし、**この差はそのまま削減可能時間ではありません。現存データから正味の期待値を score +20 以上と裏付けられる未試行候補は、今回は「なし」**と判断します。残差をすべて電力・Tensor peak・HBMだけで説明し切れたわけでもありません。特に、融合 MegaMoE の内部時間と kernel 別の実 HBM byte は、指定された記録では確定できません。

以下、理論床・実測・条件付き模型を分けて示します。

**1. 今回の比較条件と、1周期の仕事量**

round 24 の `nsys-prefill-0.nsys-rep` を SQLite に展開し、capture の内側 **2026-11-02 10:12:07.731〜10:12:14.747 UTC** を再集計しました。

保存 source と現 main を照合すると、DeepGEMM overlay は13本すべて一致、SGLang overlay は31本一致し、相違は Decode greedy のみでした。ただし、round 24 は image `c7804efb` の profile 専用 run であり、現 image `a001f901` の通常 serving 実測ではありません。

| 項目 | 確定した内容 |
|---|---|
| chunk の設定上限 | `chunked_prefill_size=8192` |
| 定常の実入力 | 原則 **3994 token × 2要求＝7988 token／forward** |
| ログの8064 token | 64-token page 単位の計数。`4032×2`。GEMM の実 M ではない |
| pair | 独立した2要求を同じ microbatch で処理すること |
| PP分割 | **24／24／24／22層**。全94層 |
| 今回の比較単位 | worker 0・PP0で、7988 tokenを24層処理する仕事量 |
| FA3の系列長 | **3994と3994の独立した causal attention**。長さ7988の attention ではない |

したがって、1周期は **24回の MegaMoE、24回の QKV、24回の FA3、24回の O**です。pairだから48層分、という数え方にはなりません。

周期境界は既存解析と同じ「24回ごとの MegaMoE 開始点」であり、request IDに対応する厳密なforward境界ではありません。60周期から single混在と、`norm_quant` が7989行だった周期を除いた**53周期**を今回の代表集計に使いました。

| 区分 | 依頼文の既存集計 | 今回の代表7988行集計 |
|---|---:|---:|
| MegaMoE | 63.41 | **63.56** |
| QKV＋O | 23.42 | **23.47** |
| FA3 | 18.89 | **18.95** |
| MoE入力 | 2.94 | **2.94** |
| writer／pack | 2.14 | **2.15** |
| norm | 1.54 | **1.54** |
| その他の小kernel | 0.20 | **0.20** |
| 周期 | 113.41 | **113.61** |

単位は ms／周期です。各欄は個別の中央値なので、単純加算して周期を再構成しません。

設定と周期定義は、[PP4設定](configs/firmus-prefill-pp4.json)、[round 24解析記録](research/runtime-performance/2026-11-02-pp4-profile-r24.md)、[今回の集計結果](削除済みの一時集計)に対応します。

**2. 電力上限下の peak と、床の定義**

H200の132 SM、dense FP8の **8192 FLOP／SM／clock**、BF16の4096を使います。FMAは2 FLOPです。

\[
P_{\rm FP8}(f)=132\times8192\times f,\qquad
P_{\rm BF16}(f)=P_{\rm FP8}(f)/2
\]

| SM clock | FP8 dense peak | BF16 dense peak |
|---:|---:|---:|
| 1.380 GHz | 1492.25 TFLOP/s | 746.13 TFLOP/s |
| 1.395 GHz | 1508.47 TFLOP/s | 754.24 TFLOP/s |
| 1.410 GHz | **1524.70 TFLOP/s** | **762.35 TFLOP/s** |
| 1.425 GHz | 1540.92 TFLOP/s | 770.46 TFLOP/s |

Tensor peakがclockに比例するという仮定です。公称1979 TFLOP/sをCUDA-core最大clockの1.98 GHzで割る補正は使いません。[NVIDIA Hopperの仕様説明](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

HBMは公称 **4.8 TB/s**を使います。SM clockに比例させません。これは持続実効帯域の実測値ではありません。[NVIDIA H200仕様](https://www.nvidia.com/en-sg/data-center/h200/)

clockの来歴は次のように分けました。

- **主表：同じround 24のworker 0・PP0、kernel稼働中の周期別中央値1.410 GHz。**
- 指定されたround 27 telemetryでは、通常測定開始20秒後から終了までの **2026-11-03 12:39:52.762〜12:41:32.762 UTC** を再集計。g1の各GPUは100 sample、clock中央値1395〜1425 MHz、平均電力691.77〜692.80 W、memory clockは全件3201 MHz、throttle reasonは全件`0x4`でした。
- g1・GPU0の中央値は1395 MHzです。この値をround 24時間へ組み合わせる結果は、後述の**別runを組み合わせた感度計算**として示します。

床は原則、

\[
t_{\rm floor}=\max(F/P,\ B_{\rm HBM}/W_{\rm HBM},\ \text{その他の資源下限})
\]

です。重ねられるTensor処理と転送の床を足しません。

また、**指定profileには区分別の実DRAM byteを確定する情報がありません。** 以下のbyte欄はsourceから数えた「主要bufferを一回ずつ通過させる論理I/O模型」です。L2 hit、再読出し、writebackがあるため、実HBM byteや厳密なHBM下限と同一視しません。

**3. 区分別の仕事量、理論床、実測**

代表shapeは以下です。

- MegaMoE：`T7988、E128、top8、H4096、I1536、EP1`
- QKV：`M7988 × N9216 × K4096`
- O：`M7988 × N4096 × K8192`
- FA3：`2×L3994、Q64/KV4、D128、GQA16、causal`

Tensor演算量は、

\[
\begin{aligned}
F_{\rm FC1}&=4(T\cdot8)HI\\
F_{\rm FC2}&=2(T\cdot8)HI\\
F_{\rm QKV}&=2T\cdot9216\cdot4096\\
F_{\rm O}&=2T\cdot4096\cdot8192\\
F_{\rm FA3}&=2\text{要求}\times64\text{ heads}\times
\frac{3994\cdot3995}{2}\times4\cdot128
\end{aligned}
\]

としました。

| 区分 | Tensor演算量／周期 | 主要論理I/O／周期 | HBM換算時間 | Tensor床、1.410 GHz |
|---|---:|---:|---:|---:|
| FC1 | FP8 **38.597 TFLOP** | 約47.57 GB | 9.91 ms | **25.31 ms** |
| FC2 | FP8 **19.298 TFLOP** | 約34.33 GB | 7.15 ms | **12.66 ms** |
| dispatch | Tensor演算なし | 約7.29 GB | 1.52 ms | 対象外 |
| combine | Tensor演算なし | 約14.14 GB | 2.95 ms | 対象外 |
| QKV | FP8 **14.474 TFLOP** | 約5.25 GB | 1.09 ms | **9.49 ms** |
| O | FP8 **12.866 TFLOP** | 約4.00 GB | 0.83 ms | **8.44 ms** |
| FA3 QK＋PV | FP8 **12.548 TFLOP** | 約3.44 GB | 0.72 ms | **8.23 ms** |
| MoE入力内のrouter | BF16 **0.201 TFLOP** | 融合kernel全体で約5.61 GB | 1.17 ms | 0.264 ms |

FC1／FC2のweight byteは全128 expertを読む場合です。空expertがあれば減少します。round 24の実routingは保存されていないため、有効expert数を確定値として埋めていません。

MegaMoEの主要bufferは1層あたり、FP8 weight最大2.416 GB、dispatch pool 261.75 MB、FC1出力98.16 MB、FC2 BF16 partial 523.50 MB、最終出力65.44 MBです。表は各段階の読出し・書出しを数えています。

**FC2がBF16 partialを出すことと、BF16 Tensor演算であることは別です。FC1／FC2とも積はFP8です。** scale処理、SwiGLU、softmax、正規化、combineは主にFP32です。例えばcombineの必要な8項加重和は、8乗算＋7加算として約0.0118 TFLOP／周期ですが、Tensor peakで評価する処理ではありません。

実測との比較は次のとおりです。効率は「有効Tensor床÷実測時間」です。

| 区分 | 実測 ms／周期 | 有効Tensor床 | 効率 | 実測−床 |
|---|---:|---:|---:|---:|
| **MegaMoE全体** | **63.56** | **37.97** | **59.7%** | **25.59** |
| **FA3** | **18.95** | **8.23** | **43.4%** | **10.72** |
| **QKV** | **12.59** | **9.49** | **75.4%** | **3.10** |
| **O** | **10.88** | **8.44** | **77.5%** | **2.44** |

MegaMoEの効率にはdispatch、epilogue、combineなどの時間も含まれます。**FC1単独・FC2単独の効率をこの比率から逆算することはできません。**

FC1／FC2／dispatch／combineは一つのpersistent kernelに融合されています。round 24のnsysでは内部phaseの実時間が分離されず、round 19の細分計装も通常版を4.4〜4.7倍に遅くしています。したがって、**4区分の個別実測時間・個別効率は未確定**です。[round 19の測定限界](research/runtime-performance/2026-11-01-prefill-megamoe-diag-r19.md)

「その他」は、Tensor床よりもmemory・scalar命令の模型が適します。

| その他の内訳 | 実測 | 一回通過I/O模型の時間 | 参考効率／差 |
|---|---:|---:|---:|
| MoE入力融合kernel | 2.94 ms | 約1.17 ms | 約40%／約1.77 ms |
| writer＋pack | 2.15 ms | 約1.19 ms | 約55%／約0.96 ms |
| `norm_quant` 22回 | 約1.37 ms | 約1.05 ms | 約77%／約0.32 ms |
| 残るnorm・小kernel | 約0.37 ms | 未確定 | 未確定 |

writerの三角関数表など小bufferの再利用、scalar命令・reduction費用はこのI/O模型に含め切っていません。これらの「参考効率」はHBM実測効率ではありません。

その他全体は約6.83 msで、主要I/O模型との差は概ね3 ms台です。ただし、**QKVとの差より確実に大きい、という精度の順位付けはできません。**

別runの1395 MHzをround 24時間に組み合わせると、床はMegaMoE 38.38、FA3 8.32、QKV 9.59、O 8.53 msになります。最大差の順位は変わりません。これはround 27で各kernelを測り直した結果ではありません。

**4. 大きな差の内訳**

**MegaMoE：約25.59 msの差**

有効演算だけの37.97 msに対して、現在の実装には次の費用があります。

- **64行単位のpadding。** poolはM128ですが、空の後半WGはMMAを省くため、演算行数は128行丸めではなく、expertごとの64行丸めです。
- historical fixtureでは有効63,904行に対して68,288行。これを今回のclockで計算すると、**paddingだけで約2.60 ms／24層の追加Tensor仕事**になります。ただしround 24の実routingではありません。
- **carried-scaleの依存。** 現行は各K128 blockで128個のaccumulator FMUL／active math warpを持ちます。「前のWGMMA完了→rescale→次のWGMMA」という依存が残ります。
- **SwiGLU・amax・FP8量子化、BF16 partial store、combine、到着同期、最後のtileの偏り。**

過去のH100模型ではsplit loopが1272〜1278 cycle、Tensorサービス床が1024 cycleでした。同じ比率がH200 servingにも成立すると仮定すると、padding込み演算床に対して約9.8〜10.1 ms相当の差になります。**これは規模の参考であり、round 24の内部時間として足し込んではいけません。**

combineの主要I/Oは約0.589 GB／層で、公称HBM換算では約123 µs／層、約2.95 ms／周期です。これもcache状態を含む実測ではありません。

既存の棄却は、この差を簡単には回収できない理由を示しています。

| 既存案 | 得られた証拠 |
|---|---|
| A gather | poolコピー削除よりloader負荷増が大きく、約44%悪化 |
| direct FC2 | SMEM往復を消してもstore要求増などで約1.6%悪化 |
| FC1割当L | 負荷分散の模型は改善したが、全12 GPUで悪化 |
| swap-AB tail | MMA約3.8%削減に対して発行・waitが増え、historicalではほぼ相殺 |
| polling backoff | 到着検知が遅れ、historicalで約1.1〜2.3%悪化 |
| consumer再設計 | 命令削減が成立しない、またはspill増でCPU段階棄却 |

したがって、**最大対象はMegaMoE全体ですが、最大原因を「dispatch待ち」や「FC1だけ」と確定する証拠はありません。** [consumer解析](research/runtime-performance/2026-11-02-prefill-megamoe-consumer-review.md)、[round 26結果](research/runtime-performance/2026-11-03-prefill-megamoe-swap-tail-r26.md)

**FA3：約10.72 msの差**

現行pairは **M192／N96、3 math WG、P3**です。

| 1層・2要求あたり | 値 |
|---|---:|
| 有効QK＋PV | 522.847 GFLOP |
| tile実行量 | 535.881 GFLOP |
| M-tile数 | 2664 |
| KV block訪問数 | 56,784 |
| 論理K＋V load量 | 1.396 GB |
| unique K＋V | 約8.18 MB |

1.396 GBは主としてL2→SMEMの論理load量です。これをHBM trafficと呼ぶことはできません。

- paddingによる追加Tensor床は、**約0.205 ms／周期**。
- 現行softmaxのscore EX2とrow rescaleを数えた発行量は約10.88億結果／層。
- 16結果／SM／clockの発行率を使うと、EX2サービス床は**約8.77 ms／周期**。
- Tensor床8.44 ms〔padding込み〕とEX2床8.77 msは重なり得るので、合計しません。[CUDA算術命令の発行率](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#arithmetic-instructions)

この資源床を使っても実測との差は約10.18 msです。残る対象は、max／sum依存、FP8変換、O rescale、Q受渡し、tile境界のfill／drain、storeなどです。現在の通常kernelでの細分時間は未取得です。

一方、単純な改善案は既に反証されています。

- sum-chain短縮S：FA3単体で約39 µs／層悪化。
- V先読みP2：bundleで約182〜192 µs／層悪化。
- softmaxだけの対角trim：平均約3.7 µs悪化。
- Q double-buffer、early claim、tree-maxも棄却済み。

また、sourceは**各要求内で既にlongest-first**です。二つの要求全体をlongest-firstへ並べ直す未試行案を、132 CTAの貪欲割当模型で確認しました。tile固定費をKV stepの0／2／4／8倍とした範囲では、周期換算の改善は**0〜約0.032 ms、score 0〜約+2.8**でした。大きな残差をwave末尾だけで説明する模型にはなりません。

**QKV／O：合計約5.54 msの差**

現在の設定は両方とも **BM256／BN128／BK128、3 stage、cluster 2・A multicast**です。

- QKV：2304 tile、18 wave。
- O：1024 tile、8 wave。
- 両方ともwave slot利用率96.97%。
- M7988を8192行まで実行するpaddingあり。N paddingはありません。

今回の同runのpair時間から、tile時間を `a＋b×K128 block数` とする二点模型を当てると、`a≈1.58 µs`、`b≈0.861 µs`になります。

| 差の項目 | QKV | O |
|---|---:|---:|
| M paddingのTensor仕事 | 約0.24 ms | 約0.22 ms |
| wave quantizationの追加床 | 約0.30 ms | 約0.27 ms |
| epilogue／次tile立上りの模型 | 約0.68 ms | 約0.30 ms |
| K loopの床超過に相当する模型 | 約1.87 ms | 約1.65 ms |

後ろ二つは直接計装した時間ではなく、同形状・同runの時間からの模型です。K loopは約1214 cycle／K128で、1024-cycle Tensor床に対して約84%の効率になります。

既存tile調査では、BM192／BN128やBM128／BN192はmainloop内spill、spillなしのBM128／BN144でもL2 loadとtile数が増えました。Oでは代替tileのwave効率も低下します。現在の合法なconfig空間に、明確に有利な選択は見つかっていません。[dense GEMM調査と棄却結果](research/runtime-performance/2026-10-31-prefill-dense-pp4.md)

**launchの隙間・同期**

今回の代表周期では、全streamでkernel／copyがない空白の中央値は**約0.474 ms**でした。大部分は既に投入済みの処理の依存待ちやtransfer待ちに分類され、露出したlate-launchの中央値は0です。

したがって、MegaMoEの25.59 msやFA3の10.72 msを、kernel間launch gapとして回収することはできません。kernel内部のbarrier時間も、この0.474 msとは別物です。

**5. 精度を保つ未試行案の評価**

以下の数値は、明記した条件下の予算・感度です。正味短縮の期待値が未確定なものを、実測改善として扱いません。score換算は指定の **1 ms／周期＝約+88**です。

| 候補 | 条件付き短縮とscore換算 | 規模・過去の棄却との関係 |
|---|---|---|
| **QKV/OのtailをK loop外で特殊化** | 不要192行のTensor仕事は合計**約0.431 ms、約+38相当**。ただしcritical waveが短くならなければ直接短縮にならず、正味期待値は未確定 | 中〜大。各tile入口で通常／tailを分け、通常mainloopを不変にする。round 9の「毎wave分岐で+6.3%悪化」と異なる構造が必要 |
| **QKV/Oのcluster 4化** | 同tileの理想multicast模型ではAのL2 loadをさらに減らせるが、Tensor仕事・SMEM fill・wave数は変わらない。正味ms／scoreは未確定 | 大。現行cluster 2専用のscheduler／barrier契約を変更。既存tile列挙の外側 |
| **FA3 M256／小Nの再設計** | K/V再読出し削減の可能性はあるが、KV step数、O rescale、register圧が増える。正味ms／scoreは未確定 | 大。現在のM192/N96からの単純なblock変更ではない。spillとpipeline寿命を先に証明する必要 |
| **FA3の要求間tile順を統一** | 今回の模型では**0〜0.032 ms、0〜+2.8** | 小〜中。算術を変えずbit一致を要求可能。今回の+20基準には不足 |
| **FA3の対角QK/PVを分岐なしで縮小** | 全padding Tensor仕事を除けても約**0.205 ms、約+18相当**。追加制御費・clock波及は未確定 | 大。旧softmax-only trimの再試行ではなく、WGMMA直列化を起こさない構造が必要 |
| **norm residual storeのlayout改善** | 22回のnorm_quantでI/O模型との差は約**0.32 ms、約+28相当**。+20にはその約71%を正味回収する必要があり、根拠不足 | 中。round 27のFMAとは別。削るのはSMEM並べ替えで、HBM byteは減らない |
| **独立chunk間のstream overlap** | 露出idle全消去の楽観予算でも約0.47 ms。ただし依存・資源・律速stageの条件があり、正味期待値は未確定 | 大。複数buffer、イベント、persistent kernelの資源配分変更が必要 |
| **PP stage内で2 chunkを交互実行** | 現状のkernelをそのまま交互投入するだけでは正の短縮を示せない。score見込みは未確定 | 大。workspace再利用、PP slot、KV送信まで含む変更 |

精度の条件は、tile順・stream順だけならbit一致、reductionやtilingを変える場合も既存dtype・量子化形式・全causal項を維持し、FP演算順の違いに限定することです。近似exp、低精度化、項の省略は候補に含めません。

**stream overlapが簡単に成立しない理由**

同じtokenの依存は、

`O → post-attention norm/router → MegaMoE → 次層input norm → QKV`

です。前層Oと次層normを別streamに置くだけでは並列になりません。token tile単位で進めるにも、normはそのtokenの4096特徴が揃う必要があります。

さらに、profileのlaunch resourceは次のとおりでした。

| kernel | threads × registers/thread | shared memory |
|---|---:|---:|
| QKV／O | 384 × 168 | 約216 KB |
| MegaMoE | 384 × 168 | 約229 KB |
| FA3 P3 | 512 × 128 | 約100 KB |

SMのregister枠は65,536個です。QKV／O／MegaMoEはほぼ全枠、FA3はlaunch情報上で全枠を使い、132 persistent CTAが全SMに配置されます。**独立した仕事でも、別streamだけでは通常のnorm CTAを同じSMに追加できません。**

chunk交互実行も、二つの7988-token batchを交互にする場合と、pairを3994-tokenずつに分割する場合を分ける必要があります。後者では、

- MoEのexpert別paddingが増え得る。
- weight再読出しが増え得る。
- 現sourceではsingle用FA3 M128/N192へ戻り、pair用M192/N96の利点を失う。

という追加費用があります。単純な「重ねれば速い」という見積もりは成立しません。

**6. 結論と、再開するなら必要なgate**

**今回のデータから期待値+20以上として推す候補は、なしです。**

これは「理論床に達した」「残差はすべて物理限界」という結論ではありません。数値で確定できる範囲は以下です。

- 1.410 GHzで主要4区分の有効Tensor仕事には、合計**約64.13 ms／周期**が必要。
- 公称dense peak相当の1.830 GHzなら同じ仕事は約49.41 ms。**実clockを使うことで床が約14.72 ms上がる**。これはclock条件の比較であり、電力制限を外せば実周期が14.72 ms短くなるという実測予測ではない。
- MegaMoEの主要I/O一回通過模型は約21.53 msで、Tensor床37.97 msより小さい。**全体をHBM帯域だけで説明できない。**
- FA3はTensorだけでなくEX2にも約8.77 msのサービス床があり、softmax依存・pipeline・epilogueの費用が残る。
- 最大残差はMegaMoEの約25.59 ms。ただし、その全量を分類する通常servingの内部計測はない。

未試行案の中でCPU検討を続けるなら、比較的具体的なのは **QKV `(7988,9216,4096)`／O `(7988,4096,8192)` の、K loop外でのtail特殊化**です。ただし現段階では推奨実装ではなく、採否を変える証拠待ちです。

再開条件は、通常tileのSASS不変、mainloop内分岐追加なし、spill増なし、multicast／barrier契約維持を先に示すこと。その後の速度gateでは、全12 GPU・両順計時で、24層相当の完全QKV＋O経路を**少なくとも約0.227 ms／周期、平均9.47 µs／層**短縮し、bit一致・耐久を通す必要があります。さらに同条件の通常serving repeatと公式parserで+20相当が残ることを確認する必要があります。既存の小さいkernel選択marginだけでは十分ではありません。

分析のみを実施しました。リポジトリ変更、commit、push、SSH、Actions／Slurm操作は行っていません。展開したSQLiteと大きな周期別JSONは削除済みです。