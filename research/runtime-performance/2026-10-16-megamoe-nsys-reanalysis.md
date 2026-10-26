## 2026-10-16追記：配置済み最新版によるnative report再解析

### 使用ツール・再解析範囲

- `/opt/nvidia/nsight-systems/2026.3.2/bin/nsys`：2026.3.2.476。
- `/opt/nvidia/nsight-compute/2026.3.1/ncu`：2026.3.1.0。
- 元の4本を変更せず、nsysで全native reportをSQLiteへ再export。全4本成功。
- kernel件数はP0/P1/P2それぞれ121,703、D0は1,076,732。旧exportの保存summaryと一致し、更新で欠損イベントが復元されたわけではない。
- 全4 workers・16 GPUの起動geometry、active window内のkernel区間、16 Torch CPU tracesとCUDA同期APIの時刻を再解析。追加GPUジョブ・再収録は実施していない。
- `.ncu-rep`は存在しない。ncuの `--import .../nsys-prefill-0.nsys-rep --page session` は「The file is not an Nsight Compute profiler report」で終了した。**ncuでのhardware counter再解析は対象データ不足で実施できない。** Tensor Core利用率、HBM実効帯域、achieved occupancy、stall reasonsは依然未計測。

### A. 新たに特定した制約：MegaMoEが1 CTA/SMに制約されるresource構成

全P workersで同一geometry、Dは別geometryだった。起動metadataとTorch devicePropertiesのH200の値を照合した。

| 起動metadata | Prefill MegaMoE | Decode MegaMoE |
|---|---:|---:|
| grid | 132 CTA | 132 CTA |
| threads/CTA | 640 | 384 |
| registers/thread | 96 | 168 |
| dynamic shared memory/CTA | 225,600 B | 230,672 B |
| shared memory/SM 233,472 Bに対する割合 | 96.63% | 98.80% |
| 1 CTA時のresident threads / SM上限2048 | 31.25% | 18.75% |

static shared memoryは0。1 CTAだけでSM共有メモリのほぼ全量を使うため、同じkernelの2 CTAを同時常駐させられない。grid132は記録されたSM数132と一致する。上表のthread割合は**起動resourceに基づく上限計算であり、ncuのachieved occupancyではない**。warp specializationやpersistent kernelでは低いthread占有率が合理的なこともあり、「残るthread割合だけ高速化できる」という意味ではない。

**最適化対象を具体化:** 同期・メモリ待ちを他CTAで隠せない構成なので、現行tile・pipeline stage・shared-memory buffer・register圧の組合せを優先して調べる。shared memoryを少し削るだけでは2 CTAに増えず、SM容量の半分以下などのresource条件と仕事量の両方を満たす必要がある。単純にoccupancyを上げるためにtileを小さくするとGEMM効率が下がる可能性がある。候補はwhole-MoEの時間で判定する。

- [x] 現行P/Dの実際のblock/grid/register/shared-memory値を既存reportから特定。
- [ ] exact native sourceのtemplate引数とresource量を対応付け、変更候補を限定する。
- [ ] 代表shapeのncu診断でTensor Core活動・memory traffic・stallを測り、stage/warp構成のどこが制約か確認する。分散同期するkernelの単独replayが成立するかを先に確認し、他rankを止めるreplayを安易に適用しない。

### B. 全Prefillのactive windowでは、記録済みGPU処理が96%以上

収録側のsession epochとTorchのbaseTimeNanosecondsを用いてUTCを合わせ、全worker共通の `05:00:17.278964454Z〜05:00:22.297356859Z` にclipした。

| worker | GPU別 recorded kernel union / 5.018秒 |
|---|---:|
| Prefill-0 | 96.75〜96.84% |
| Prefill-1 | 96.31〜96.45% |
| Prefill-2 | 96.72〜96.99% |
| Decode-0 | 84.92〜85.62% |

Pのcomplete MegaMoE callsのGPU別平均は2.401〜2.467ms、Dは0.239〜0.242ms。平均では大きなrank偏りは見られないが、layer別expert skewや同一collective内の待機がないことは保証しない。

このprofileでは、Prefillの大きな改善はhost launch回数削減だけより、**実行中のMegaMoE等を速くする方向**を優先する根拠が強い。欠損警告があるためDecodeの約15%を丸ごとCPU idleとは解釈しない。

### C. CPU stageをCUDA API・GPU timelineと照合し、以前の解釈を修正

以下はactive windowに完全に収まるCPU spansだけの集計。境界をまたぐspanは除外した。

- P0のdraft_extendは各rank14 spans、平均188.74〜188.82ms。そのうち同じprocessのCUDA同期APIと重なる時間は平均185.95〜185.96ms、約98.5%。
- 具体例（P0 rank0の中盤span）：`cudaStreamSynchronize`が187.896ms。その同期中、同GPUでMegaMoE50 calls・126.564ms、attention等device_kernel50 calls・22.718ms、dense GEMM99 calls・21.364msが記録されている。**draft_extendというCPUラベルの中で、先行target GPU処理の完了を待っている。draft固有演算189ms、あるいは189msのGPU idleという解釈は不適切。**
- Dのget_next_batch_to_runはrankごと118〜119 complete spans、平均20.20〜21.25ms。CUDA同期APIとの重なりは平均5.66〜6.90ms。Gloo all_gatherは平均11.64〜13.92ms。
- GPUとの重なりも照合した。Dのget_next_batch_to_run 20.20〜21.25msのうち、同じrankの記録済みGPU kernel区間unionは17.50〜18.50ms。Gloo 11.64〜13.92msのうち10.08〜12.22ms（約87%）でもGPUが実行中だった。Gloo時間をGPUが停止したCPUオーバーヘッドと同一視しない。P0 draft_extendでも186.53〜186.58msはGPU kernelと重なる。
- 初回記載のD20.61〜37.27ms、Gloo12.39〜29.72msは境界をまたぐspanを含む全trace平均だった。**定常のrank偏り・削減可能時間の根拠には使わず、上記の窓内値へ訂正する。** これらのCPU時間にはGPU処理とのoverlapがあり、そのままE2Eから差し引かない。

### D. 欠損の原因・位置について言える範囲

新版exportでも、収録時のCUPTI13.1使用・driver13.3非対応・CUDA不完全収録警告は残った。今回確認を4 reports全16 kernel-recording ranksへ拡張し、全rankにCUDA欠損警告があることを確認した。

今回reportのDIAGNOSTIC_EVENTには、別run38490463905のような `Number of incomplete CUPTI events dropped` の件数・原因内訳がない。不完全収録警告のtimestampTypeは `HostTimestamp` で、失われたGPUイベントの発生時刻を示さない。**このデータから「欠損は終了境界だけ」「active window内は完全」とは証明できない。** 別runの欠損件数や原因を今回へ転用しない。

### 優先順位の更新

1. Prefill MegaMoEのresource構成とkernel内の待機を調べ、tile/pipelineの限定比較へ進める。GPUが動いている時間の効率改善を主対象とする。
2. 初期TTFTのqueue/forward/handoffを改善し、実際の処理量をSLA適格なgoodputへつなげる。通常scoreは6344.96を固定対照にする。
3. Decodeのhost同期は、今回の境界除外後の値とGPUとの重なりを基準に評価する。旧最大37msや未記録15%を削減目標にしない。
4. 欠損のない収録と代表shapeのhardware countersを、上記の仮説を判別するために取得する。最新版の導入自体を完了条件としない。

