# 2026-10-23 DecodeのDRAM Read 42.8%と、現行ソースに残る改善候補

## 結論

8576.654970637783 good output tok/sを記録した [run 39901974117](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39901974117) 自身のNsightレポートから、Decodeの **DRAM Read 42.8267%** を再現した。これはHBMボトルネックが全時間の42.8%を占めるという意味ではない。読出しinterfaceの活動率であり、カーネル内の待ち、計算、書込み、他のカーネルも含む時間平均である。

高速化余地はある。ただし `1 / 0.428 = 2.34倍` という速度上限の読み方はできない。カーネル実行区間の和集合は約96.3%で、CPUが長時間GPUを遊ばせている状態ではない。最も大きい対象は、stepの約59%を使うMegaMoE。今回のソース調査では、**Decode FC2の共有メモリ配置、SwapAB mainloopのdescriptor計算、Decode専用norm＋quantの形状**を、既試行の単純な先読み・pipeline深化と区別した次候補にする。

本稿は既存結果の再解析と設計調査。GPU高速化の実証、最適化の実装・採用、追加ジョブ投入は行っていない。

## 対象と正本

- 調査時のmain: `b165d1d`。実行commit: `b60f2e32d393baeacd31fba25e677ffe4304aa95`。
- image: `39899235870`、SHA256 `3b3d4a8d46671f6264204b35ea797e3f34aad91e6ec806ae252ece77ec22f8c8`。
- raw正本: `/srv/qwen-artifacts/39901974117/1/`。
- 通常score: `results/results/score.json`。official_validation.valid=true、要求エラー0、5386要求。
- 今回再解析: `profile/profile/nsys-decode-0.nsys-rep`、同じdirectoryの `kernel-summary.json`、`validation.json`、`torch/decode-0/*.trace.json.gz`。
- 実行時patch・設定: `source/source/{deep-gemm-overlay.json,sglang-overlay.json,sm90-megamoe-build.json,decode-kernels.env}`。
- Actionsログと保存Slurm accounting: Slurm 25593、COMPLETED、0:0、15分44秒。今回新規GPUジョブ0。

コミット本文には「profileなし」とあるが、実際には4 workerのNsight report、Torch trace、profile validation成功が保存されている。`logs/logs/profile-capture.log` にもcapture_active/stop_requestedが存在する。本文の予定ではなく保存された実行結果を採用した。通常scoreと後続のprofile passは別計測であり、profileのstep時間から通常goodputの改善量を直接確定しない。

Claude Code session `5abe09c0-a310-5471-9b41-b47ce68d0de8` の10月21–23日の説明、Codexの10月22–23日のL2先読み調査・実装報告、関連research、commit本文、最新Actionsログを照合した。以前の期待値と実測結果を区別する。

## 1. 42.8%の再現と時間の内訳

Nsight Systems 2026.3.2でSQLiteへexport。各GPUの最初のMegaMoE開始と最後のMegaMoE終了の共通範囲 **260621532–5604721343 ns（5.3441秒）** を用いた。report全体の平均にすると、capture開始前と停止・export側の尾を含み、Readは25–27%に下がる。GPU 1–3は終了処理のカーネルも後ろにあるため、全カーネルのmin/maxを各GPUで別々に使う方法も適切でない。

| Nsight metric | GPU 0 | GPU 1 | GPU 2 | GPU 3 | 4 GPU平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DRAM Read Bandwidth % | 43.3046 | 42.8771 | 43.5638 | 41.5613 | **42.8267** |
| DRAM Write Bandwidth % | 2.2197 | 2.2270 | 2.2259 | 2.2215 | 2.2235 |
| GR Active % | 99.9727 | 99.9725 | 99.9732 | 99.9747 | 99.9733 |
| SMs Active % | 86.3116 | 86.2987 | 86.2242 | 86.4851 | 86.3299 |
| SM Issue % | 28.0172 | 26.7189 | 28.2390 | 25.1654 | 27.0351 |
| Tensor Active % | 13.6940 | 13.1368 | 13.7837 | 12.4913 | 13.2765 |

[NVIDIAの指標定義](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics)ではRead/Writeは経過cycleに対するDRAM interfaceの読出し/書込み活動率。Readだけを総HBM効率や「HBMが原因のstall割合」と呼ばない。ReadとWriteを単純加算して万能な帯域利用率とすることも避ける。1 kHzのsampleは200 µs級MegaMoEの各内部区間を分離できないので、sampleの時刻だけで個々のkernelへ帯域を帰属させない。

各GPUでMegaMoEは15,792 call = 94層×168周。94 callごとの先頭から次の先頭までの167個の完全な周期を取り、GPUごとの時間を計算した。平均31.8256 ms/周、kernel区間の和集合は96.30–96.39%。以下は4 GPU平均のkernel duration集計であり、重複を許すため厳密なcritical-path分解ではない。

| 処理 | ms/周 | 周期時間に対する目安 |
| --- | ---: | ---: |
| MegaMoE | **18.807** | **59.1%** |
| FA3全体（verify companion・stock・draft等を含む） | 4.445 | 14.0% |
| dense FP8 QKV/O | 2.784 | 8.7% |
| QK-norm/RoPE | 0.843 | 2.6% |
| fused add RMSNorm | 0.622 | 2.0% |
| per-token group quant | 0.358 | 1.1% |
| router Top-K | 0.329 | 1.0% |
| MegaMoE pre-dispatch | 0.179 | 0.6% |
| その他（gate GEMM、draft GEMM、logits等） | 2.440 | 7.7% |

GPU 0の1 call平均はMegaMoE **199.951 µs**、p128s2 FA3 **45.542 µs**、QKV **14.939 µs**、O **14.618 µs**。最新のFA3/QKV改善が本番profileにも現れている。

約3.7%のkernel非実行時間を全て消しても、このprofileでの周期短縮は約1.17 msに限られる。しかもこの差はmemcpy・同期等も含むので「CPU launch overhead」と同一視しない。一方、稼働中のSM Issueが27%であることは依存・同期・小さい仕事等の調査動機になるが、単独では原因の確定にならない。

## 2. 最近の改善で既に解消した点／外れた仮説

詳細は [10月20日以降の実行記録](2026-10-20-firmus-prefill-overlay-results.md)、[SwapAB mainloop](2026-10-18-megamoe-decode-mainloop-pipeline.md)、[epilogue/dispatch](2026-10-18-megamoe-decode-epilogue-dispatch.md)、[Decode tail](2026-10-18-megamoe-decode-tail.md) を正本とする。

| 対象 | 現状・測定結果 | 今回の判断 |
| --- | --- | --- |
| N_SWAPのK loop外へのhoist | 約229–230→205–207 µs、採用済み | 同じ分岐削除を新案にしない |
| FC1 FP32 stagingのbank conflict | swizzle＋4 threads/token量子化、採用済み | 残るFC2とは分ける |
| combine並列化、動的CTA割当、FC1 store wait局所化 | imageのpatch seriesに存在 | 昔のtail時間を現行の未対処時間として使わない |
| B1次blockの先行claim | 現行overlayに存在、servingで独立した効果は未確認 | 重ねたL2先読みとは別変更 |
| B4 FA3 scheduler metadata共有／B6融合 | 39764967207でB5なし8481.99、採用 | 準備kernelの94回削除を再提案しない |
| K loopの複数block pipeline/unroll | chunk 2/1、1/2、2/2等も遅い。大unrollほど悪化 | 命令cache・register増を伴う同案を繰り返さない |
| MegaMoE L2先読み8/16 stage | 39881882325で約+1.8–7.6 µs/層の悪化 | 42.8%だけを理由に再投入しない |
| router gateのTriton置換 | 94層で約+314–318 µs | 単純な1 kernel化は外す |
| B5 small-M dense GEMM | 7582.76へ悪化。QKV 44.3/O 30.5 µs | 既定を戻した経緯を維持 |
| 最新FA3 p128s2／QKV BM128 BN80 | 39901974117で採用、8576.65497 | 現行対照に必ず含める |

10月18日の計装では、TMA loaderが空きstageを待ち、math warp側のK loop・epilogueが進行を制限していた。単純にstageを増やす根拠にはならない。現在までにtailやepilogueが変わっているので、その当時の内訳を現在の199.95 µsにそのまま割り振ることはしない。

## 3. 現行ソースの確認方法

`ssh firmus` のlogin nodeでimage内の `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh` だけを展開して読んだ。抽出header SHA256は **ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1** でmanifestと一致。保存されたoverlay順序の5 patchをローカルcopyへ適用し、全て成功した。

以下の `H:行番号` はこの再構成後のheader。原本mirrorを編集していない。主要な現行設定はM64/N128/K128、8 stages、2 math warpgroup、384 threads/CTA、132 CTA。profileのshared memoryは230,672 bytes、static register表示168。共有メモリだけでSMあたり1 CTAに制限される。

## 4. 優先候補A：FC2のSwapAB出力配置とscatter

**調査結果として最も具体的で、小さく独立に検証できる候補。速度は未測定。**

対象はH:2727–2758の `store_bf16` / `store_l2_swap_chunk` とH:2835–2863のscatter。FC1はswizzle済みだが、FC2のSwapAB分岐は次のrow-major配置を使う。

```cpp
smem_cd_l2[wg_offset + token * WG_BLOCK_N + col] = __float2bfloat16_rn(value);
```

DecodeのWG_BLOCK_N=64、warp内laneをlとすると、最初のstoreのtokenは `2*(l%4)`、colは `warp*16+l/4`。32-bit wordのbankは `((token*64+col)/2)%32`。warp 0の32 laneでは4 bankへ集中し、**各bankに4個の異なる32-bit word** が対応する。同じwordのBF16上下半分もあるので「8-way conflict」とは断定しない。実際のwavefront数は生成SASSとcounterで確認する必要がある。

比較する配置案は、SwapABだけで次の対応をstoreとscatter readの両方に入れること。

```cpp
physical_col = col ^ (((token / 2) % 4) * 16);
```

CPU上のアドレス列挙では16 bankに分散し、各bankの異なるwordは1個になる。全valid_m=1..64でrow内の全64列が全単射になること、scatterのaligned 4-BF16 vectorを連続・整列のまま保つことを確認した。BF16丸め、remote出力の並び、Top-K重みと和の順序は変える必要がない。

併せて、現在はscatterの各jで `workspace.get_token_src_metadata_ptr(...)` を読んでremote pointerを組み立てている。Prefill 2WGにはmetadataを先にloadしshuffleで取り出す実装が既にある（H:2800付近）が、Decodeは別の行所有配置で旧loopを通る。**metadata先読みはswizzleとは別commit**で比較する。既存compilerがloadをまとめている可能性とregister live range増をSASSで確認してから進める。

限界：10月18日のFC2 epilogueは約10.5 µs/CTAだったが、これは現在のcritical pathから全量除去できる時間ではない。初回判定は現行比2–5 µs/層程度の差を測れる精度を狙う。これは改善予測ではなく測定設計の目安。

## 5. 優先候補B：SwapAB mainloopのdescriptor計算を縮める

対象H:1939–1954の `run_swap_ab::issue`。各K blockの各WGMMAで `make_smem_desc` を二つ構成している。一方、Prefillにはbase descriptorを一度作り、16-byte単位のstage/K offsetを加える実装が既にある（H:1490付近、[該当patch](../../patches/deep-gemm/sm90-prefill-mainloop-prefetch.patch)）。

このアドレス計算部分だけをSwapABに移す候補。FC1の `issue → wait<0> → promote → empty arrive`、FC2のlow/high別FP32 promotion、scale・演算順は維持する。K loopの大unrollや追加accumulatorは導入しない。

もう一つの小候補は、N_SWAP specialization内のpromotion（H:1962付近）で、端数の最後の8-token chunk以外まで繰り返している `token < valid_m` 判定を整理すること。ただしcompilerが既に最適化している可能性があり、descriptor変更とは別に検証する。

先にnvcc/SASSでinteger命令数、branch、register、spillを比較する。現行と同じSASSならGPU実験をしない。過去にpipeline深化が悪化したため「演算とloadをもっと重ねれば速い」という理由だけでは採用しない。

## 6. 優先候補C：Decode用の小さいnorm＋quant

現行 [add-rmsnorm-group-quant.patch](../../patches/sglang/add-rmsnorm-group-quant.patch) はTARGET_VERIFYを明示的に除外する。理由も記録されており、2 rows/CTAの既存融合kernelは約126行で63 CTAしか作らず、**8.8 µs対非融合5.2 µs**で負けた。フラグを有効にするだけでは再び悪化する。

試すなら、input_layernorm→QKVの一経路に限定した **1 row/CTA・128行で128 CTA** の別geometryを作る。BF16への中間丸め、FlashInferと同じnorm reduction、FP32 group128 scale、residual更新を保つ。norm済みBF16をregister内で量子化し、中間のglobal store/loadとquant launchを消す。

最新profileでquant全体は0.358 ms/周、入力側1回分は概ねその半分。この候補で何msも短縮する見積もりは過大。元のnorm自体より遅くしないことが条件。post-attention normはgateとMegaMoEの両方が使うため、同じ実装をそのまま流用しない。

## 7. FA3は資源と仕事の分割を同時に変える必要がある

現在のp128s2はGPU 0で45.54 µs/call、109,568 bytes shared memory、256 threads、static register表示255。共有メモリ量だけを見ると2 CTA/SMが入りそうだが、それだけでは判断できない。

固定source `sgl-project/sgl-attn@f89bc2306632d1ec5f97b014dded4254f5b4a907` の `hopper/flash_fwd_kernel_sm90.h:86–89,351,403` を確認した。one-MMA-WGはload側56、math側256のregister budgetを使い、単純な合計でも `128*(56+256)=39936` registers/CTA。2 CTAは65,536 registers/SMを超える。dynamic register再配分があるため、static 255×256だけで結論を出すことも避ける。

さらに32 requests×4 KV heads=128仕事、132 CTAなので、occupancyだけ増やしても並列仕事が増えない。2 CTAを活かすにはKV splitまたはGQA groupの分割も必要。前者はcombine費用・空splitの正しさ、後者はKV再読出しを伴う。今回のsplit2/3候補は数値gateに落ちている。まず誤りを解消し、registerを減らしたtileと**仕事の分割を一組の設計**として評価する必要がある。stage数だけの再掃引は優先しない。

小さな残りとして、[現行dispatch](../../patches/sglang/decode-fa3-verify.patch) はbatch32–36のみcompanionを使う。profileにもstock FA3が残り、Torch traceのbatchは29–32。29–31へ対応範囲を広げる候補はあるが、scheduler metadata・固定NumWarps・数値gateの対応が必要。上限batch32の主要経路は既に速くなっており、期待規模は小さい。

## 8. HBM床と次の検証の順序

従来の重み量見積もり604 MB/rankを4.8 TB/sで割ると約126 µs。現行MegaMoE約200 µsとの差は約74 µs/層、94層で約7 ms。ただしこれは重みのcold readだけの楽観的な比較で、activation・通信・出力・同期・cache挙動を含む到達可能なkernel下限ではない。42.8%から残り57.2%をそのまま削れるわけではない。

実験へ進むなら次の順にする。

1. FC2 swizzle単独のcandidate。CPUの配置検証に加え、実際のSASSでstore/load、bank関連wavefront、register/spillを確認する。
2. 現行image＋全採用overlayを対照に、公式serving由来のDecode routing、balanced、29–32 requests相当の行数、idle/mixed geometry、changed-input graph replayを検査。通常benchとinstrumentationを別passにする。
3. 勝った場合のみmetadata先読み、descriptor計算削減をそれぞれ独立commitで測る。mainloopの大unrollやL2先読みとのbundleにはしない。
4. kernel benchはGitHub Actions→schedulerで実行し、交互順の複数block・複数runで確認。数値gate、4 rank最大時間、cold-L2と実routing、resource使用を記録する。
5. 最後に同一条件の公式AIPerf→result parser→成果物回収まで比較し、通常goodput・ITL/TTFT・accept長で判断する。source変更は新規build、性能改善はbranchで行い、main比の確認後に採用する。

過去の単体勝利がservingで消えた例があるため、kernel短縮率からgoodputを確約しない。まず「帯域カウンタを上げること」ではなく、**同じ出力・同じ仕事のcritical pathを短くすること**を目的にする。

## 9. 理論下限との距離と、候補ごとの改善規模（追補）

基準はprofile周期31.8256 ms、94層、通常goodput 8576.65497。以下の「成功時シナリオ」は測定前の実験予算・目標であり、統計的な期待値や達成予測ではない。効果0・悪化もあり得る。改善がcritical pathへそのまま乗り、batch・accept長が一定なら、1層あたりΔ µsの短縮は `94*Δ/1000 ms/周期`、Decode処理速度比は `31.8256/(31.8256-94*Δ/1000)` となる。

### 理論的な距離

- **MegaMoE全体**：32 expert/rankのFP8重みは `32*(2*1536*4096+4096*1536)=603979776 bytes`。H200の公称4.8 TB/sによる重み読出しだけの時間は125.83 µs。199.95 µsはその1.59倍、差74.12 µs。ただし、通信・activation・同期・出力を無視し、重みをHBMから1回読むモデル。この126 µsへ到達できるとは主張しない。[NVIDIA H200仕様](https://www.nvidia.com/en-au/data-center/h200/)
- **FC2配置変更**：アドレスモデルでは競合bankあたり4 distinct word→1。改善対象のshared storeに最大4倍のservice-rate差があるという話で、FC2 epilogue全体やMegaMoE全体の4倍ではない。10月18日のepilogue約10.5 µs/CTAには変換、同期、metadata、remote scatterも含まれ、現在の競合store部分の時間は未測定。[shared memory bankの仕様](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- **descriptor計算削減**：整数命令・依存鎖がどれだけ短くなるかのSASS確認が未実施なので、この処理固有の理論時間・占有時間はまだ算出できない。旧timelineのK loop約135 µs（計装補正の概算）対重み読出し126 µsの差約9 µsは参考値であり、現在のdescriptorだけで9 µsを削れるという根拠ではない。
- **input norm＋quant**：現行の概数はnorm約3.2＋quant約1.9=5.1 µs。normを同じ速度で行い、追加quantを完全に隠す理想なら約3.2 µs、差1.9 µs。ただし融合後もquant演算は必要。この3.2 µsは物理下限ではなく、現行norm速度を使った融合効果の楽観的な基準。約3.7 MBの最小論理I/OをHBM peakで割れば0.77 µsになるが、L2再利用・reduction・起動費用を無視するので、これを実現可能なkernel時間には使わない。

### 最初の実験で狙う規模

| 候補 | 成功時の仮置き短縮/層 | 短縮/周期 | Decode処理速度の換算 |
| --- | ---: | ---: | ---: |
| FC2 swizzle単独 | 0.5–2 µs | 0.047–0.188 ms | +0.15–0.59% |
| descriptor計算削減 | 1–3 µs | 0.094–0.282 ms | +0.30–0.89% |
| input norm＋quant（5.1→3.5–4.5 µsと仮定） | 0.6–1.6 µs | 0.056–0.150 ms | +0.18–0.47% |
| 3案が全て成功し、効果が加算できた場合 | 2.1–6.6 µs | 0.197–0.620 ms | **+0.62–1.99%** |

FC2の0.5–2 µsはshared storeの小範囲を改善する試験としての仮置きで、metadata先読みを含めない。前節の2–5 µsは測定設計の目安であり、予測値ではなかった。descriptorの1–3 µsは3案の中でも特に根拠が弱い目標値。SASSが変わらなければ0で、GPU実験をしない。normは既存融合版の8.8 µsという敗北実績があり、CTAを増やすだけで目標へ達する保証はない。

Decodeだけが律速し、Prefill供給・accept長・SLA適合率が不変という追加仮定なら、合計は約8630–8747 good tok/sに相当する。公式goodputの予測としては扱わない。現在のデータから「前回の3案で10%以上速くなる」と言える根拠はない。

### 大きい改善に必要な短縮

| MegaMoEが到達したと仮定する時間 | 現行約200 µsからの周期短縮 | Decode処理速度換算 | goodputが同率で伸びる場合の参考値 |
| --- | ---: | ---: | ---: |
| 180 µs | 1.875 ms | +6.3% | 約9114 tok/s |
| 170 µs | 2.815 ms | +9.7% | 約9409 tok/s |
| 160 µs | 3.755 ms | +13.4% | 約9724 tok/s |

180/170/160 µsは達成見込みではなく、必要な改善規模を示す逆算。前回の3案は小さく切り分けられることを優先しており、これだけで20–40 µs/層を削る設計にはなっていない。10%以上を狙うなら、FC1/FC2 epilogue・dispatch/collective・GEMM進行を含む現行版の内部timelineから、別途まとまったcritical-path短縮を見つける必要がある。

## 再解析の再現手順

```sh
nsys export --type sqlite --output /tmp/decode-39901974117.sqlite /srv/qwen-artifacts/39901974117/1/profile/profile/nsys-decode-0.nsys-rep
```

SQLiteの `TARGET_INFO_GPU_METRICS` でmetric名を確認する。このreportはmetricId 18=Read、19=Write、typeIdは281479271677952〜955。次のSQLは同じ共通区間でのGPU別平均を返す。

```sql
SELECT typeId, metricId, COUNT(*) AS samples, AVG(value) AS mean
FROM GPU_METRICS
WHERE timestamp BETWEEN 260621532 AND 5604721343
  AND metricId IN (2, 3, 4, 5, 18, 19)
GROUP BY typeId, metricId;
```

kernel時間は `CUPTI_ACTIVITY_KIND_KERNEL.demangledName = StringIds.id` で名前を取得し、deviceごとに解析する。MegaMoEを開始順に94個ずつ区切る。先頭〜168周目の先頭の167周期についてclipしたkernel durationを集計し、同じ区間のunionを別に計算する。CPU user_annotationのdurationをGPU step時間として足さない。

今回の変更はこの調査記録のみ。runtime/source mirror、実行設定、既存の未追跡ファイル、既存branch、他作業のジョブは変更していない。login nodeで作成した抽出用directoryは削除済み。
