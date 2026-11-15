# Kernel最適化の検討チェックリスト

検討候補の一覧です。採用・不採用や実測の根拠は [kernel theory tracker](runtime-performance/kernel-theory-tracker.md) と個別の実験記録を参照してください。

# 1. カーネル起動・スケジューリング

| 項目                            | 最適化の観点                         |
| ----------------------------- | ------------------------------ |
| カーネル起動レイテンシ                   | 特に小さいカーネルで支配的になる               |
| カーネル起動回数                      | Fusionなどで削減                    |
| CUDA Graph対応                  | CPU側の起動オーバーヘッドを削減              |
| Graph Replayの効率               | メタデータ更新やGraph切替のコスト            |
| Graph Capture可能性              | 動的メモリ確保、CPU同期、非対応APIを避ける       |
| カーネルFusion                    | 中間テンソルの読み書きと起動を削減              |
| 過剰Fusion                      | レジスタ圧迫・Occupancy低下・並列性不足を防止    |
| Persistent Kernel             | カーネルを常駐させ、起動・スケジューリング負荷を軽減     |
| Programmatic Dependent Launch | 対応GPUで依存カーネルの起動を前倒し            |
| Cooperative Launch            | 複数CTAの協調実行                     |
| Kernel Dispatch               | Shape・Dtypeごとに適切な実装を選択         |
| Shape Specialization          | 既知の行列形状・Head Dimensionを利用      |
| Dynamic Shape対応               | 再コンパイルやGraph再生成を抑制             |
| Kernel Warmup                 | JIT・初期化・キャッシュ生成による遅延を分離        |
| JITコンパイル時間                    | 実行時コンパイルのコストとキャッシュ効率           |
| CPU側の同期                       | 不要なGPU待ち、stream synchronizeを排除 |
| GPU Stream                    | 独立処理を並列実行可能にする                 |
| Stream Priority               | レイテンシ重視の処理を優先                  |
| カーネル間の依存関係                    | 不必要な同期・シリアライズを回避               |

注意点： Fusionは常に有利とは限りません。例えばRMSNorm・量子化・GEMMを融合しても、レジスタ使用量が急増し、GEMM本体の性能を大幅に落とす可能性があります。

# 2. レジスタ・Occupancy・SM資源

| 項目                              | 最適化の観点                        |
| ------------------------------- | ----------------------------- |
| レジスタスピル                         | Local Memoryへの退避を抑制           |
| レジスタ使用量                         | Threadあたりのレジスタ数を制御            |
| Register Pressure               | 同時に生存する変数を削減                  |
| Live Range                      | 変数の生存期間を短縮                    |
| Occupancy                       | Resident Warps / CTAsを適切に確保   |
| Active Warps                    | メモリ・演算レイテンシの隠蔽                |
| Registers / SM                  | SM全体のレジスタ割当制約                 |
| Registers / Thread              | Thread単位の使用量制約                |
| Register Allocation Granularity | レジスタ割当の丸めによる無駄                |
| Shared Memory / CTA             | CTAの同時常駐数を制限                  |
| Threads / CTA                   | Block Sizeによる資源利用率変化          |
| Warps / CTA                     | Warp数と並列性の調整                  |
| CTA Residency                   | SMに同時配置可能なCTA数                |
| Instruction-Level Parallelism   | Thread内で独立命令を増加               |
| Thread-Level Parallelism        | Thread・Warp間で処理を分散            |
| Register Reuse                  | 中間結果の再利用                      |
| Register Bank Conflict          | 該当アーキテクチャでのレジスタバンク競合          |
| Register Fragmentation          | 割当単位や配置による資源の無駄               |
| Compiler Register Allocation    | コンパイラの割当結果をSASS/PTXで確認        |
| OccupancyとILPのトレードオフ            | Occupancyを下げてILPを増やす方が速いこともある |

ここで特に重要なのは、Occupancy最大化と性能最大化は同義ではないことです。

GEMMでは、レジスタを多く使ってデータ再利用・ILPを高める方が高速な場合があります。一方、メモリ待ちが多いDecodeカーネルでは、十分なWarpを確保した方が有利な場合があります。

# 3. メモリアクセス・メモリ階層

| 項目                          | 最適化の観点                          |
| --------------------------- | ------------------------------- |
| Global Memory Coalescing    | 隣接Threadのアクセスを連続化               |
| Memory Transaction数         | 不必要なメモリトランザクションを削減              |
| Memory Alignment            | ベクトル化・効率的なロードを可能にする             |
| Vectorized Load / Store     | 1命令で複数要素を転送                     |
| DRAM Bandwidth              | 実効帯域の最大化                        |
| L2 Cache Hit Rate           | 重み・KV Cache・メタデータの再利用           |
| L1 Cache Hit Rate           | 局所的なデータ再利用                      |
| Cache Line Utilization      | 読み出したキャッシュラインを有効活用              |
| Cache Thrashing             | 容量・連想度不足による追い出し                 |
| Cache Pollution             | 再利用しないデータによるキャッシュ汚染             |
| Cache Eviction Policy       | アクセス特性に応じたキャッシュ制御               |
| Cache Bypass                | 再利用の少ないデータに対して検討                |
| Memory Access Stride        | 大きなStrideによる転送効率低下              |
| Memory Layout               | 演算・ロード方式に適した配置                  |
| Transpose回避                 | レイアウト変更による余分なメモリ転送を削減           |
| Shared Memory Reuse         | DRAMから読み込んだデータを再利用              |
| Shared Memory Bank Conflict | 同一Bankへの競合アクセスを削減               |
| Shared Memory Padding       | Bank Conflict回避                 |
| Swizzled Layout             | Tensor CoreやShared Memory向け配置   |
| Shared Memory容量             | Tile Size・Occupancyとのトレードオフ     |
| Memory-Level Parallelism    | 多数のメモリ要求を同時発行                   |
| Outstanding Memory Requests | 同時処理可能なメモリ要求数                   |
| Memory Latency Hiding       | 演算や他Warp実行で待機時間を隠蔽              |
| Prefetch                    | 次のデータを先行ロード                     |
| Double Buffering            | 演算とロードを並列化                      |
| Triple / Multi Buffering    | より深いパイプライン                      |
| TMA                         | Hopper以降での効率的な転送                |
| 非同期Copy                     | Threadによる逐次ロードを削減               |
| メモリ割当                       | 実行中のAllocationやFragmentationを抑制 |
| 不要なZero Initialization      | 初期化が本当に必要か確認                    |
| In-place処理                  | 追加テンソルの確保・転送を削減                 |
| Intermediate Tensor削減       | 中間結果をHBMへ書き戻さない                 |
| Read/Write Amplification    | 実際に必要なデータ量に対する転送量               |
| TLB / Page Walk             | アドレス変換の負荷                       |
| DRAM Partition利用            | アクセス分布の偏りを抑制                    |
| メモリ転送方向                     | Read / Write帯域の特性を考慮            |

特にDecodeでは、重みやKV Cacheの読み出しに対するArithmetic Intensityが低くなりやすいため、理論FLOPSよりメモリ性能が支配的になることが多いです。

# 4. 計算・命令レベル

| 項目                 | 最適化の観点                             |
| ------------------ | ---------------------------------- |
| Tensor Core利用率     | MMA命令の効率的な発行                       |
| CUDA Core利用率       | 非MMA演算の実行効率                        |
| MMA Tile Size      | ハードウェアが得意な形状に合わせる                  |
| Warp-level MMA     | Warp単位の行列演算                        |
| WGMMA              | Hopper世代のWarpgroup MMA             |
| tcgen05 / TMEM     | Blackwell世代のTensor Core機構          |
| 命令スループット           | 各命令種別の発行上限                         |
| 命令レイテンシ            | 依存命令が実行可能になるまでの時間                  |
| 命令依存チェーン           | 長い依存関係を短縮                          |
| ILP                | 独立した演算を並行発行                        |
| FMA利用              | 乗算・加算の融合                           |
| 演算強度               | FLOPs / Byteを高める                   |
| Reciprocal近似       | 除算の代わりに逆数乗算を利用                     |
| Fast Math          | 精度要件内で近似演算を利用                      |
| Exp / Log / Tanh   | 特殊関数の実行コスト                         |
| SFU利用率             | Softmaxや活性化関数の特殊関数命令               |
| Integer Arithmetic | アドレス計算・量子化処理の最適化                   |
| Address Generation | ポインタ演算を削減                          |
| Loop Unrolling     | 分岐・ループ管理の負荷削減                      |
| 過剰Unrolling        | コードサイズ・レジスタ使用量の増加を抑制               |
| Branch Divergence  | Warp内の分岐不一致を削減                     |
| Predication        | 分岐とマスク命令の使い分け                      |
| Warp Shuffle       | Shared Memoryを使わないWarp内通信          |
| Reduction方式        | Tree Reduction・Warp Reductionなどの選択 |
| Atomic Operation   | 競合・シリアライズを削減                       |
| Warp-wide Vote     | ballot / match / reduce等の活用        |
| Subnormal処理        | FTZなどの動作と精度への影響                    |
| Conversion命令       | FP32↔BF16、FP8変換などを削減               |
| SIMD Packing       | 複数の低精度データをまとめて処理                   |
| 命令数                | SASS上の実命令数を削減                      |
| 実際の命令発行率           | 理論演算量だけでなくSchedulerの稼働を確認          |
| Instruction Cache  | 大きすぎるカーネルによるI-cache圧迫              |

NVIDIA HopperのFlashAttention-3は、TMAとTensor Coreの非同期性を利用したWarp Specialization、行列積とSoftmaxのインターリーブ、FP8処理などを組み合わせています。計算と転送を重ねるだけでなく、異なる演算パイプライン同士を重ねることが重要です。

[image](https://www.google.com/s2/favicons?domain=https://ai.meta.com\&sz=32)

Research - AI at Meta



# 5. 並列化・同期・パイプライン

| 項目                              | 最適化の観点                     |
| ------------------------------- | -------------------------- |
| Warp Specialization             | Warpごとにロード・演算など役割を分担       |
| Producer / Consumer分離           | データ供給と計算を独立させる             |
| Compute / Memory Overlap        | メモリ転送待ちを計算で隠蔽              |
| MMA / Softmax Overlap           | Tensor Coreと特殊関数処理を重ねる     |
| Communication / Compute Overlap | GPU間通信と計算の並行化              |
| Pipeline Depth                  | Stage数の最適化                 |
| Pipeline Bubble                 | 各Stageの待機時間を削減             |
| Barrier Overhead                | 同期回数と待機時間を削減               |
| Barrier Granularity             | CTA・Warp・Warpgroup単位で適切に同期 |
| Asynchronous Barrier            | 非同期処理の完了を効率的に管理            |
| Memory Fence                    | 必要最小限のメモリ可視性保証             |
| CTA間同期                          | Global Synchronizationを抑制  |
| Warp間通信                         | Shared MemoryやBarrierの負荷   |
| Warp内通信                         | Shuffleなどで高速化              |
| Load Imbalance                  | CTA間で処理量が偏らないようにする         |
| Tail Effect                     | 少数のCTAだけが残る状況を削減           |
| Wave Quantization               | CTA数とSM数の関係による端数損失         |
| Grid Size                       | GPU全体を使い切れるCTA数を確保         |
| Block Size                      | 並列性と資源使用量のバランス             |
| Split-K                         | K方向に分割して並列性を増加             |
| Split-K Reduction               | 分割後の集約処理のコスト               |
| Stream-K                        | タイル間で演算量を再配分               |
| Work Stealing                   | 動的な負荷分散                    |
| Persistent CTA                  | SM上で複数タスクを連続処理             |
| Cluster Execution               | 対応GPUでCTA Clusterを活用       |
| Cluster Shared Memory           | Cluster内のデータ共有             |
| PDL                             | 前段の完了前に次段の起動を進める仕組み        |

# 6. GEMM / GEMVの最適化

LLMではLinear Projection、QKV Projection、MLP、MoEのExpertなど、多くの演算がGEMMになります。

| 項目                         | 最適化の観点                            |
| -------------------------- | --------------------------------- |
| Tile Size (M, N, K)        | SM資源、再利用率、並列性を最適化                 |
| Tile Aspect Ratio          | 行列形状に適した縦横比                       |
| Tensor Core Tile Alignment | ハードウェアの命令形状に合わせる                  |
| Padding Overhead           | 端数処理に伴う無駄な演算を抑制                   |
| Small-M GEMM               | Decode時の小さいMに特化                   |
| GEMV Specialization        | Batchが極小の場合にGEMV実装を検討             |
| Skinny GEMM                | M/N/Kのいずれかが極端に小さい形状               |
| Grouped GEMM               | 複数GEMMをまとめてスケジューリング               |
| Batched GEMM               | 多数の小規模行列積を効率的に実行                  |
| Persistent GEMM            | 重み・タイルの再利用や起動負荷の削減                |
| Split-K GEMM               | 並列性の不足をK方向分割で補う                   |
| Stream-K GEMM              | CTAの負荷をより細かく分散                    |
| GEMM Epilogue Fusion       | Bias、Activation、Quantizationなどを融合 |
| GEMM Prologue Fusion       | 前処理・量子化などの融合                      |
| Weight Prepacking          | 推論時に効率的に読める重み配置                   |
| Offline Weight Transform   | 実行時のTranspose・変換を回避               |
| Weight Reuse               | 同一重みを複数トークンで再利用                   |
| Quantized GEMM             | INT8、FP8、FP4などのTensor Core活用      |
| Weight-only GEMM           | INT4/INT8重みをロード時に復元               |
| Dequantization Fusion      | 逆量子化中間テンソルを生成しない                  |
| Scale Loading              | 量子化スケールの転送・再利用                    |
| Zero-point補正               | 非対称量子化の追加演算を削減                    |
| Sparse GEMM                | 対応する疎構造を活用                        |
| Structured Sparsity        | 2:4などハードウェア対応形式の利用                |
| Tile Swizzling             | CTA配置を変更して局所性を改善                  |
| Arithmetic Intensity       | データ再利用によって演算密度を向上                 |
| Accumulator Precision      | FP32などの累積精度と実行効率                  |
| Epilogue Store             | Store配置とベクトル化を最適化                 |

特にDecode時のLinearは、バッチサイズが小さい場合、重みの読み出し帯域がボトルネックになりやすいです。

この場合、Tensor Coreの理論FLOPSを最大化することよりも、重みを高速に読み、少ない命令・少ないメモリ転送で結果を得ることが重要になります。

# 7. Attention固有の最適化

| 項目                            | 最適化の観点                      |
| ----------------------------- | --------------------------- |
| FlashAttention                | Attention行列をHBMに実体化しない      |
| Online Softmax                | ブロックごとにSoftmax統計量を更新        |
| QKᵀとPVの融合                     | 中間Attention行列の書き戻しを回避       |
| SoftmaxとMMAのオーバーラップ           | 異なる演算パイプラインを利用              |
| Attention Tile Size           | Q/K/Vタイルの再利用と資源消費           |
| Head Dimension Specialization | 64/128/256などの形状に特化          |
| GQA最適化                        | 複数Q HeadでK/Vを再利用            |
| MQA最適化                        | KV Headが少ない構造に特化            |
| Multi-head Parallelism        | Head単位の並列性を利用               |
| Split-KV Attention            | 長いKV列を分割して並列処理              |
| Split-KV Merge                | 部分Softmaxの安定した統合            |
| Decode Attention              | Queryが極小のケースに特化             |
| Prefill Attention             | 多数のQuery Tokenを同時処理         |
| Multi-token Attention         | Speculative Decode時の複数Query |
| Variable-length Attention     | 要求ごとに異なる系列長に対応              |
| Ragged Batch                  | Paddingせず可変長バッチを処理          |
| Causal Mask                   | 無効なAttention位置の演算を回避        |
| Sliding Window Attention      | Window外のKV参照を削減             |
| Sparse Attention              | 不必要なAttention Blockをスキップ    |
| Attention Sink                | 特定トークンを保持する方式に対応            |
| RoPE Fusion                   | 位置埋め込みの追加カーネルを削減            |
| ALiBi Fusion                  | Attention ScoreへのBiasを融合    |
| Logits Soft-capping           | モデル固有のSoftcapを効率的に処理        |
| Softmax精度                     | 最大値・指数和の安定した計算              |
| KV Cache Prefetch             | 次に参照するKVブロックを先読み            |
| KV Cache Reuse                | GQAやMulti-tokenでの再利用        |
| Paged Attention               | 非連続KVブロックを効率的に参照            |
| Page Table Lookup             | アドレス参照・間接アクセスの削減            |
| Page Size                     | 断片化・局所性・アクセス粒度を調整           |
| KV Layout                     | NHD/HNDなど適切な配置              |
| KV Quantization               | FP8/FP4等による転送量削減            |
| Quantized KV Decompression    | 展開コストと帯域削減効果の比較             |
| Attention Parallelism         | Head数やSequence長に応じたCTA分割    |
| Padding Mask処理                | 無効トークンへの演算を最小化              |
| Long-context対応                | 極長系列での並列分割・帯域効率             |
| Short-context対応               | 小規模Attentionの起動負荷・並列性       |

FlashInferも、Paged KV Cache、GQA、CUDA Graph、Tensor Core Decode、複数のAttention Backend、KV量子化などを分けて扱っています。実装上も、単一のAttentionカーネルであらゆる形状を処理するより、条件に応じて切り替える設計が重要です。

[image](https://www.google.com/s2/favicons?domain=https://docs.flashinfer.ai\&sz=32)

FlashInfer 0.7.2.post1 documentation



# 8. KV Cache・メモリ管理

Attentionカーネルだけでなく、その周辺のKV管理がDecode性能に直結します。

| 項目                        | 最適化の観点                  |
| ------------------------- | ----------------------- |
| KV Cache容量                | 同時処理できるSequence数を左右     |
| KV Cache Bandwidth        | Decode時のメモリ転送量          |
| KV Cache Page Size        | 内部断片化とアクセス効率のバランス       |
| KV Block Allocation       | 新規ブロック確保の負荷             |
| Block Table更新             | ページ参照情報を効率的に更新          |
| KV Fragmentation          | 未使用領域・断片化を削減            |
| KV Cache Compaction       | 必要に応じてメモリ配置を整理          |
| Prefix Sharing            | 共通PrefixのKV再利用          |
| Prefix Cache Hit Rate     | 実際の共有率を高める              |
| KV Cache Eviction         | 利用価値の低いブロックを適切に削除       |
| KV Cache Quantization     | 容量・帯域を削減                |
| Quantization Scale Layout | スケール読み出しを効率化            |
| KV Cache Write Fusion     | 新しいK/Vの生成と書き込みを統合       |
| KV Cache Write Coalescing | Append時の小規模書き込みを効率化     |
| KV Cache Copy             | Beam SearchやFork時のコピー削減 |
| Copy-on-Write             | 共有KVを必要時だけ複製            |
| KV Cache Offload          | GPU↔CPU転送と容量のトレードオフ     |
| KV Cache Prefetch         | Offloadされたブロックを先読み      |
| KV Cache Compression      | 低精度化以外の圧縮               |
| Mixed Precision KV        | 層・Headごとに精度を変える         |
| Layer-wise KV管理           | 層ごとのアクセス・寿命を考慮          |

# 9. 量子化・数値表現

| 項目                       | 最適化の観点                 |
| ------------------------ | ---------------------- |
| BF16 / FP16              | Tensor Core性能と数値範囲     |
| FP8 E4M3 / E5M2          | 精度・ダイナミックレンジの選択        |
| FP4 / NVFP4              | 高圧縮率とBlock Scalingへの対応 |
| INT8 / INT4              | 整数演算性能と復元コスト           |
| Weight-only Quantization | 重み転送量削減                |
| Activation Quantization  | Activationの低精度化        |
| KV Cache Quantization    | Attentionのメモリ帯域削減      |
| Per-tensor Scaling       | スケール管理の簡素化             |
| Per-channel Scaling      | 精度とスケール読み出しのバランス       |
| Per-group Scaling        | グループサイズの最適化            |
| Per-block Scaling        | ハードウェア対応形式への最適化        |
| Dynamic Quantization     | 実行時スケール算出の負荷           |
| Static Quantization      | 事前算出したスケールを利用          |
| Scale Fusion             | 乗算を既存演算へ吸収             |
| Dequantization Cost      | 低精度化による帯域削減効果との比較      |
| Quantize/Dequantize回数    | 往復変換を削減                |
| Mixed Precision          | 演算ごとに精度を変更             |
| Accumulation Precision   | FP32累積などの精度管理          |
| Rounding Mode            | 丸め方式と誤差の変化             |
| Saturation / Clipping    | 量子化範囲外の処理              |
| Outlier Handling         | 外れ値に起因する精度劣化を抑制        |
| Subnormal / FTZ          | 非正規化数の扱い               |
| Numerical Stability      | SoftmaxやReductionでの誤差  |
| Non-associativity        | 並列Reductionによる結果の差     |
| Determinism              | 実行順序による再現性の変化          |

量子化については、モデルサイズが半分になっても、推論時間が半分になるとは限らない点が重要です。
例えばINT4 Weight-only GEMMでは、HBM帯域を節約できても、逆量子化の命令数、Scale Load、Tensor Coreに入力するためのPacking/Unpackingが新たなボトルネックになり得ます。
このコンテストにおいては、量子化はかなり制限されているので、規約に十分に注意して適用してください。

# 10. MoE（Mixture of Experts）固有の最適化

| 項目                       | 最適化の観点                      |
| ------------------------ | --------------------------- |
| Expert Routing           | TokenをExpertへ効率的に振り分け       |
| Top-k Selection          | Expert選択の演算効率               |
| Token Sorting            | 同一Expertに属するTokenを集約        |
| Token Permutation        | Rearrange処理の転送量削減           |
| Token Unpermutation      | 元のToken順へ戻す処理               |
| Histogram / Prefix Sum   | Expert別Token数とOffsetの算出     |
| Expert Load Imbalance    | ExpertごとのToken数の偏り          |
| Grouped GEMM             | 複数Expertをまとめて計算             |
| Small Expert GEMM        | Token数が少ないExpertへの最適化       |
| Expert Padding           | Tile境界に合わせる際の無駄             |
| Expert Weight Layout     | Expertごとの重み配置               |
| Expert Weight Prefetch   | 必要なExpert重みを先読み             |
| Expert Parallelism       | ExpertのGPU間分散               |
| All-to-All通信             | TokenをExpert所有GPUへ転送        |
| All-to-AllとGEMMのOverlap  | 通信中に計算を進める                  |
| Token Dispatch Fusion    | Routingとデータ移動を融合            |
| Token Combine Fusion     | Expert出力の集約を融合              |
| Expert Replication       | 人気Expertの複製                 |
| Dynamic Expert Placement | 負荷に応じた配置変更                  |
| Expert Capacity          | 最大Token数・Paddingの制御         |
| Shared Experts           | 共有Expertの計算・通信を効率化          |
| Expert Cache             | Expert重みのキャッシュ              |
| Routing Skew             | 特定Expertへの集中によるTail Latency |

MoEではGEMMが速くなっても、SortingやAll-to-Allが律速になるケースがあります。

そのため、Expert GEMMだけでなく、RoutingからCombineまでを一連の処理として最適化する必要があります。

# 11. Multi-GPU・分散推論

| 項目                                | 最適化の観点                   |
| --------------------------------- | ------------------------ |
| Tensor Parallelism                | TPサイズとGEMM/通信のバランス       |
| Pipeline Parallelism              | Pipeline Bubbleを削減       |
| Expert Parallelism                | MoE ExpertをGPU間分散        |
| Data Parallelism                  | 複数リクエストの並列処理             |
| Context Parallelism               | 長いSequenceをGPU間分散        |
| Sequence Parallelism              | Sequence方向での処理分割         |
| All-Reduce                        | TPに伴う出力集約                |
| All-Gather                        | 分散テンソルの収集                |
| Reduce-Scatter                    | Reductionと分散を融合          |
| All-to-All                        | MoE Token交換              |
| Collective Fusion                 | 複数通信をまとめる                |
| Collective Latency                | 小さい通信での固定コスト             |
| Collective Bandwidth              | 大きい通信での実効転送帯域            |
| Communication/Computation Overlap | GEMMとCollectiveを重ねる      |
| CUDA-aware通信                      | GPUバッファ間の直接転送            |
| GPUDirect RDMA                    | 対応環境でのNIC↔GPU転送          |
| NVLink / NVSwitch                 | GPU間帯域とトポロジー             |
| PCIe Bandwidth                    | PCIe接続時の帯域制約             |
| PCIe Topology                     | Root ComplexやSwitch構造の影響 |
| NUMA Affinity                     | CPU・GPU・NICの配置を一致させる     |
| P2P Access                        | GPU間直接アクセスの可否            |
| GPU Memory Mapping                | GPU間アドレス空間の扱い            |
| Communication Granularity         | 大きい転送と小さい転送の使い分け         |
| Communication Chunking            | 分割通信によるOverlap           |
| Network Contention                | 複数通信の帯域競合                |
| NIC Affinity                      | GPUに近いNICの利用             |
| Synchronization Granularity       | 全GPU同期を最小化               |
| Straggler GPU                     | 遅いGPUに全体が引きずられる問題        |
| Pipeline Microbatching            | Bubbleとレイテンシのバランス        |
| Communication Buffer Reuse        | 通信バッファの再利用               |
| Compute Resource Reservation      | 通信カーネル用のSM余力を確保          |

ここでは、GPU間通信を計算と重ねる場合、両者が共有する資源にも注意が必要です。

例えば通信処理にSMを使用する場合、通信時間は隠蔽できても、その分GEMMが遅くなる可能性があります。また、通信と計算が同じHBM帯域を奪い合うこともあります。

したがって、単純なOverlap率より、Overlap後の総実行時間を最小化することが重要です。

# 12. 推論エンジンとの統合

カーネル単体で速くても、SGLangやvLLMに統合したときに同じ性能が出るとは限りません。

| 項目                            | 最適化の観点                          |
| ----------------------------- | ------------------------------- |
| Continuous Batching           | 動的に追加・終了するリクエストへの対応             |
| Batch Size変動                  | 実際のBatch Size分布に対する最適化          |
| Chunked Prefill               | PrefillとDecodeの共存               |
| Prefill/Decode Interleaving   | 異なる計算特性の処理を混在させる                |
| Prefill/Decode Disaggregation | 分離配置したGPU間のKV転送                 |
| Speculative Decoding          | Draft/Verifyでの効率的な処理            |
| Speculative Acceptance Rate   | 複数トークン処理の実効効率                   |
| Dynamic Token Count           | リクエストごとのToken数の変化               |
| CUDA Graph Batch Padding      | Graph対応のためのPadding損失            |
| Graph Pool                    | 複数Batch Size用Graphの保持コスト        |
| Graph Switching               | Graph切り替えに伴うオーバーヘッド             |
| Kernel Backend Selection      | FlashInfer / FlashAttention等の選択 |
| Autotuning                    | 形状ごとの最適実装の探索                    |
| Autotuning Overhead           | 探索・JITの起動時コスト                   |
| Runtime Dispatch Overhead     | 適切なKernelを選択する処理                |
| Memory Pool                   | Allocation/Freeの負荷削減            |
| Prefix Cache Interaction      | Prefix再利用時の実行形状変化               |
| LoRA Integration              | LoRA適用の追加演算・通信                  |
| CUDA GraphとLoRA               | 動的Adapter切替への対応                 |
| Sampling Overhead             | Top-k/Top-p/Softmaxなどの処理        |
| Token Synchronization         | Sampling結果をCPU側へ戻す負荷            |
| Scheduler Overhead            | CPU側のスケジューリング処理                 |
| Request Length Distribution   | 長短混在リクエストへの対応                   |
| Tail Latency                  | 一部の長いリクエストによる遅延                 |
| Multi-tenant Interference     | 同一GPUで複数処理が競合する影響               |

vLLMではPagedAttention、Continuous Batching、CUDA/HIP Graph、Speculative Decoding、Chunked Prefillなどを組み合わせています。カーネル性能とエンジン全体のスケジューリング性能を切り離さず評価することが重要です。

[image](https://www.google.com/s2/favicons?domain=https://docs.vllm.cc\&sz=32)

vLLM



# 13. ハードウェア特性・実行環境

| 項目                         | 最適化の観点             |
| -------------------------- | ------------------ |
| GPU Architecture           | SM世代ごとの命令・資源の違い    |
| SM Count                   | 総並列実行容量            |
| Warp Scheduler             | 命令発行の仕組みと制約        |
| Tensor Core Generation     | 対応命令と演算形式          |
| SMEM / L1容量                | タイル設計とデータ再利用       |
| L2容量                       | キャッシュされる重み・KVの規模   |
| HBM Bandwidth              | 実効的なメモリ帯域の上限       |
| Memory Clock               | メモリ律速時の性能への影響      |
| GPU Core Clock             | 演算・命令発行性能への影響      |
| Power Limit                | 電力制限時の動作クロック       |
| Thermal Throttling         | 温度上昇に伴う性能低下        |
| DVFS                       | 短いベンチマークでのクロック変動   |
| GPU Residency / MIG        | GPU資源分割による制限       |
| Concurrent Workloads       | 他のカーネルとの資源競合       |
| Driver Version             | ランタイム・コンパイラ・性能への影響 |
| CUDA Version               | 利用可能な機能・最適化        |
| PTX Version                | 新命令への対応            |
| Compiler Version           | 命令選択・レジスタ割当の変化     |
| JIT / Binary Compatibility | 対象SM向けコード生成        |
| Hardware-specific Tuning   | 世代別に最適な実装を分離       |

# 14. 性能測定・プロファイリング

個人的にはここを特に重視すべきだと思います。最適化前に「どこが律速なのか」を識別できないと、改善の方向を誤ります。

| 項目                           | 見るべき指標                      |
| ---------------------------- | --------------------------- |
| Kernel Duration              | カーネル実行時間                    |
| End-to-end Latency           | モデル全体の処理時間                  |
| Launch Overhead              | CPU起動処理・GPU側の起動コスト          |
| Achieved FLOPS               | 実効演算性能                      |
| Tensor Core Utilization      | Tensor Coreの稼働状況            |
| DRAM Throughput              | 実効HBM帯域                     |
| L2 Throughput                | L2アクセスの転送量                  |
| L1 Hit Rate                  | L1キャッシュの利用状況                |
| L2 Hit Rate                  | L2キャッシュの利用状況                |
| Global Memory Load/Store     | 実際の読み書きバイト数                 |
| Arithmetic Intensity         | FLOPs/Byte                  |
| Roofline Analysis            | Compute BoundかMemory Boundか |
| Achieved Occupancy           | 実際のOccupancy                |
| Theoretical Occupancy        | 資源制約から求めたOccupancy          |
| Active Warps                 | 実行可能なWarp数                  |
| Eligible Warps               | 命令を発行できるWarp数               |
| Issued Instructions          | 実際に発行された命令数                 |
| Issue Active                 | 命令発行スロットの利用率                |
| Warp Stall Reasons           | Warpが待機する原因                 |
| Long Scoreboard Stall        | 主に長レイテンシのメモリ依存              |
| Short Scoreboard Stall       | Shared Memory等の依存           |
| Barrier Stall                | 同期待機                        |
| MIO Throttle                 | MIOパイプライン側の圧力               |
| LG Throttle                  | Load/Store命令キューの圧力          |
| Register Usage               | Threadごとのレジスタ使用量            |
| Local Memory Traffic         | Spill等による転送                 |
| Shared Memory Bank Conflicts | Shared Memoryアクセス競合         |
| Instructions per Warp        | Warpごとの実行命令数                |
| Branch Efficiency            | 分岐の一貫性                      |
| Wave Count                   | SMで処理するCTAのWave数            |
| Tail Effect                  | 最終Waveで残る仕事量                |
| Clock Frequency              | 測定中の動作クロック                  |
| Power Draw                   | 電力消費                        |
| Energy / Token               | トークンあたりの消費エネルギー             |

Nsight Computeの公式ガイドも、Occupancyだけではなく、Eligible Warp数、Issue効率、Stall原因を調べることを推奨しています。Stall率が低下してもKernel Durationが短くならなければ、実際の高速化とはいえません。

[image](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com\&sz=32)

NsightCompute



## ベンチマーク設計で考慮すべき点

- Warmup後のSteady-state性能と初回実行を分離する。
- CUDA EventによるGPU実行時間とCPU側のWall-clock時間を区別する。
- CUDA Graph使用時・非使用時の両方を測定する。
- Batch Size、Context Length、Model Shapeを複数用意する。
- 単一カーネルだけでなく、実際のモデルに組み込んで測定する。
- Nsight Computeの計測オーバーヘッドやリプレイによる影響を考慮する。
- 平均値だけでなく、中央値、p95、p99、分散を確認する。
- コンパイル・JIT・Autotune時間を分離する。
- GPUのクロック、消費電力、温度を記録する。
- 実際のリクエスト分布に基づく性能も評価する。

# 15. 正確性・安全性・保守性

高速化のために演算順序や精度を変更する場合、次の点も重要です。

| 項目                        | 最適化の観点                       |
| ------------------------- | ---------------------------- |
| Numerical Error           | 最大絶対誤差・相対誤差                  |
| ULP Error                 | 浮動小数点表現上の誤差                  |
| Error Accumulation        | 長い系列・多数Layerでの誤差蓄積           |
| Softmax Stability         | Overflow/Underflow防止         |
| Reproducibility           | 同一入力での再現性                    |
| Batch Invariance          | Batch Sizeが変わっても結果が安定するか     |
| Race Conditions           | 並行Write/Readの競合              |
| Barrier Correctness       | 非同期命令・同期の正しい使用               |
| Uninitialized Memory      | 未初期化データの読み取り防止               |
| Out-of-bounds Access      | 端数Tileや不正Indexの処理            |
| Graph Replay Correctness  | メモリアドレス・Buffer寿命の整合性         |
| Dynamic Shape Correctness | Shape変化時の境界条件                |
| Quantization Correctness  | Scale、Zero Point、Packingの整合性 |
| Architecture Portability  | 世代別のFallback実装               |
| Compiler Portability      | コンパイラ更新に対する安定性               |
| Autotuning Robustness     | 探索した構成が正確に動くこと               |
| Maintainability           | 複雑化したカーネルの保守コスト              |

# 16. 最適化のトレードオフを整理する

高速化では、次のような対立が頻繁に発生します。

| 最適化                         | 得られる効果                   | 犠牲になりやすいもの             |
| --------------------------- | ------------------------ | ---------------------- |
| Tile Sizeを大きくする             | データ再利用率上昇                | レジスタ・SMEM・Occupancy    |
| Occupancyを高くする              | Latency Hiding           | Threadあたりの資源           |
| Fusionを増やす                  | Launch/Memory削減          | Register Pressure、並列性  |
| Pipeline Stageを増やす          | Transfer/Compute Overlap | SMEM容量                 |
| Split-Kを増やす                 | 並列性向上                    | Reduction Overhead     |
| 量子化する                       | メモリ帯域・容量削減               | Dequantization Cost、精度 |
| Persistent Kernelを使う        | 起動・スケジューリング負荷削減          | 他カーネルとの共存性             |
| Warp Specialization         | 非同期処理の効率化                | Warp配分・同期の複雑化          |
| Batch Sizeを増やす              | Throughput向上             | リクエストLatency           |
| CUDA Graphを使う               | 起動負荷削減                   | 動的Shape対応、Graphメモリ     |
| 通信をOverlapする                | 通信待機を隠蔽                  | SM・HBM帯域の競合            |
| KV Page Sizeを大きくする          | アドレス管理負荷・アクセス局所性改善の余地    | 内部断片化                  |
| Compile-time Specialization | 最適化・定数畳み込み               | コンパイル時間・バイナリサイズ        |

# 18. 特に重要だと考える項目 TOP 15

LLM推論全般を想定して、優先的に検証する価値が高いものを選ぶと、次の15項目です。厳密な重要度順位ではなく、最初に確認したい順番です。

1. Compute-bound / Memory-bound / Launch-boundの判定
2. カーネル起動回数と起動レイテンシ
3. 実効HBM帯域と実際の転送バイト数
4. Tensor Core / CUDA Coreの利用率
5. Tile Sizeとデータ再利用率
6. レジスタ使用量・スピル・Occupancy
7. Warp Stall Reasonsと命令依存
8. Pipeline設計とCompute/Memory Overlap
9. カーネルFusionによる利益と資源消費
10. CTA数・Wave Quantization・負荷分散
11. KV Cacheのレイアウトとアクセス効率
12. 量子化・逆量子化・Scale処理のコスト
13. CUDA Graph・動的Shape対応
14. GPU間通信・CollectiveのOverlap
15. 実ワークロードでのEnd-to-end性能
