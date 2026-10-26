# 2026-10-23 MegaMoEを126 µsへ近づける設計と、batch拡大との比較

## 判定

現行の約200 µsから126 µsへの短縮を実証した案はまだない。ただし、単純な先読み・stage増加以外に、dispatchの集計方法、FC1量子化の中間値、warpgroup間の仕事分担、FC2とcombineの依存関係を変える余地がある。126 µsを狙うなら、これらによって演算と通信・後処理を重ね、同時に重みの実効読出しを公称帯域近くまで引き上げる必要がある。

調査対象と現行profileの正本は [HBM/source audit](2026-10-23-decode-hbm-source-audit.md)。通常goodput 8576.65497のrun 39901974117、image 39899235870、MegaMoE 199.9507 µs、profile周期31.8256 ms・94層を基準とする。本稿は設計調査であり、実装・GPU検証・追加Actions投入は行っていない。以下の到達時間は条件からの逆算であり、達成予測ではない。

## 1. 126 µsの成立条件

32 expert/rankを全て使い、FP8重みをHBMから1回ずつ読むモデルでは、重み量は `W = 32*(3072*4096+4096*1536) = 603979776 bytes`。[H200の公称4.8 TB/s](https://www.nvidia.com/en-au/data-center/h200/)で `W/B = 125.829 µs`。

これは全kernelの到達可能な下限ではない。weight scale、activation、出力、通信・同期を含まない。逆に、cache hitや未使用expertがあればHBMから読む重みはWより少なくなる。偏りで同一expertに複数M tileが必要なら、論理的なweight loadは増える。実routingのHBM bytesを測らず、Wを実測trafficと呼ばない。

| 重みの持続読出し帯域と仮定 | 重みだけの時間 | 126 µsまでの余裕 |
| --- | ---: | ---: |
| 3.7 TB/s | 163.24 µs | −37.24 µs |
| 4.2 TB/s | 143.80 µs | −17.80 µs |
| 4.5 TB/s | 134.22 µs | −8.22 µs |
| 4.8 TB/s | 125.83 µs | **0.17 µs** |

過去run 39881882325の `results/results/decode-selection/l2-prefetch.json` には128 MiB読出しreferenceが保存されている。開始median 35.984 µs、終了36.288 µs、換算3.730/3.699 TB/s。実装は [bench-firmus-megamoe.py](../../scripts/bench-firmus-megamoe.py) の `hbm_read_kernel` / `hbm_reference`：int32 load、4096要素ごとのsum、checksum store、CUDA graph、4 rank最大時間。このreferenceはreduction等を含むため、3.7 TB/sを装置の上限とは扱わない。604 MBのTMA weight readそのものの限界は未計測。

設計予算として `T ≈ W_eff/B + E` と置く。Eは、帯域処理と重ならずcritical pathに残る仕事。別途trafficがあればW_effへ加える必要があり、この式も楽観側である。

- B=4.2 TB/s、E=10 µsなら153.8 µs。
- B=4.5 TB/s、E=5 µsなら139.2 µs。
- B=4.8 TB/s、E=5 µsでも130.8 µs。
- **126 µs、E=5 µsを同時に満たすには**、Wから最低23.18 MB（4.8 TB/s）、59.48 MB（4.5 TB/s）、95.78 MB（4.2 TB/s）を計測区間のHBM readから除く必要がある。追加trafficがあれば必要量は増える。

既存のL2先読み8/16 stageは悪化済み。前kernel中に読み込んでMegaMoEのdurationだけ短くしても、その前kernelが延びれば周期は短くならない。126 µsに届いたとの判定には、追加prepare・先読みを含む区間と全周期を併記する。

## 2. ソース上の4つの変更候補

対象headerは `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`。image内baseのSHA256は `ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1`、採用overlayはauditに記載した5本。baseの行番号とoverlay適用後の行番号を混同しないよう、以下は関数・処理名で指定する。

### A. Decodeのdispatch集計を小規模batch専用にする

`ROLE 1: DISPATCH` の現行処理は、各CTAが全128 expertについて `(1ull << 32) | local_count` をglobal atomicで足す。件数0でも実行するため、132 CTA×128 expert = **16896回/rank**。その後grid sync、SM0から件数公開、rank barrier、token/SF pullと進む。schedulerの `fetch_expert_recv_count()` も上位wordが `kNumSMs*kNumRanks` になることを待つ。

128 token/rankなら、実際のTop-K entryは128×8=1024個。この規模に対して全CTA参加の件数集計は大きい。

候補は、rank内の小さいprepare kernelまたは既存pre-dispatchを拡張し、1024 entryから件数とsource index listを作ること。source rank別のsegmentを使い、受信側は4 source rankの件数を集めてprefixを作る。132 CTAの予約・完了通知を4 rankのデータ公開に変える。新しいprepareの起動費用も含めて比較する。

必要条件：token/SF書込みとready公開のsystem-scope ordering、0 token rank、−1 padding、expert偏り、source indexの対応を保持する。件数が確定する前に現行のcompact pool indexを仮定してweight taskを開始してはいけない。まず最終cleanup barrierを残し、次replayへのworkspace再利用を安全に保つ。

旧timelineの起動→FC1開始約19 µs全てがこのatomicの費用ではない。token pullとrank依存も含むため、16896→少数という操作数削減を同じ倍率の時間短縮には換算しない。現行の前半を計測してから、起動→FC1を5–8 µs程度に抑えられるかを実験目標にする。

### B. FC1のFP32全量stagingを、register保持＋小さいamax交換へ変える

現行SwapAB FC1は `store_l1_swap_chunk` でSwiGLU結果をFP32 shared tileへ保存し、CTA barrier後に4 threads/tokenが読み直し、amax・FP8 quantを行う。swizzle改善は既に採用済みだが、**全量store→loadは残る**。[既存patch](../../patches/deep-gemm/sm90-decode-swap-ab-l1-quant.patch)

候補はSwiGLU値を元のaccumulator registerの空いた位置へ保持し、各warpの8 channel分の部分amaxだけをshared memoryで交換する。64 token×64 channel×4 bytes=16 KiBのFP32 stagingに対し、64 token×8 warp×4 bytes=2 KiBの部分amaxで表現できる。最終FP8 tileのstagingと公開順序は残す。

現在の2 math WGはそれぞれ32 channelを持ち、scaleは両方を合わせた64 channelのamaxから決まる。WGごとに独立scaleを作ってはいけない。expf、clamp、SF、FP8丸めを維持し、NaN等の扱いも既存gateで確認する。2 KiBは交換データ量の案であり、GPU時間8倍やbarrier全廃を意味しない。register live range増・FP8 pack時のshuffleが利得を消す可能性がある。

### C. 2 math WGを独立taskにして、epilogue中にも別WGを進める

現状は2 WGが同じN128 tileの半分ずつを担当し、shared stageを一緒に解放する。FC1 epilogueでも64-channel scaleを共有し、両WGが停止する。ここを、WG0がtask Aの量子化中にWG1がtask BのGEMMを行う構造に変える候補。

独立させるには1 WGが64-channel SF group全体、つまりgate/up合計128 channelを担当する必要がある。SwapABのM64 WGMMA fragmentを2本扱うため、単なるbarrier変更では済まない。A/B stageとfull/empty barrierをWGごとに分け、既存の2 load warpが2 queueを扱い、独立出力bufferを持つ設計が必要。

現行230672 bytes/CTAに二重bufferを足すだけではshared memoryに収まらない。各WG3 stage・計6 stageなどへの組替えが必要だが、stage減でload latencyを隠せなくなるリスクがある。さらに過去のコンパイルでは384-thread launchのregister上限168に縛られ、`setmaxnreg`の指定を224へ上げても上限は上がらなかった。[過去のmainloop実験](2026-10-18-megamoe-decode-mainloop-pipeline.md)

**最初の判定はコンパイルとSASS**：spill 0、WGMMA直列化warningなし、資源内に収まること。これを満たさなければ本案は棄却またはtile再設計。過去に失敗したK-loop内accumulator二重化・大unrollとは異なるが、それだけで成功すると判断しない。

### D. FC2を出力chunk順に並べ、完成したchunkからcombineする

現行は全FC2のscatter完了をrank間で確認してからcombineを始める。[combine並列化patch](../../patches/deep-gemm/sm90-combine-parallel.patch) はその後の並列度を改善しているが、この開始依存は残っている。

候補はFC2の仕事を「expert優先」から「出力512 channel chunk優先」に並べ、各tokenのTop-8の当該chunkが揃ったら、そのchunkだけcombineすること。4096 channelを8 chunkに分け、後半chunkのGEMM/scatterと前半chunkのcombineを重ねる。

one M tile/expertの場合、1 chunkは32 expert×4 N128 tile=128 task。132 SMに近い仕事量があり、全expertの先頭chunkを先に進める意味がある。FC2重みは201.3 MB/rank、1 chunkは約25.17 MBで、公称帯域なら約5.24 µs分。ただしこれは実際のchunk処理時間の予測ではない。

readyはtoken単位の大量atomicではなく、expert/pool-block/output-tile単位にする案。one M tile/expertなら1024 tile通知/rank、4 source rankへ公開するなら最大4096通知/rank。remote storeの完了とsystem-scope release/acquireを保証し、combineはslot 0→7のFP32和の順序を維持する。全rankの終了を待つ代わりに、必要な8出力の公開を待つ構造になる。

token pull完了後のdispatch warpをconsumerに転用できるかを検討する。ただし2 dispatch warpは現行8 math warpのcombineより少なく、consumerが遅ければ効果はない。producerの前進をconsumer待ちで止める循環依存を作らない。既存shared scratchはGEMMと時間分割で共用されているので、同時実行には専用bufferとbarrierが必要。

まず最終cleanup barrierを残す。ready再利用は明示的なresetまたはGPU上のepochで設計し、CUDA graph replayでの一意性を確認していない `%gridid` をgeneration番号には使わない。

## 3. 既試行との違いと優先順位

- flat NVLink barrierは全geometry版でもrun 39446543672で不採用。A/Dはbarrier primitiveの置換ではなく、集計の参加者と公開単位を変える案。
- FC2 scatterをdispatch warpへ移す案はPrefillでrun 39439372069により不採用。scatter表示時間が0になってもmainloopが360→510–535 µsへ延びた。Dはscatter担当の移動だけでなく、出力chunkの生成順とcombine開始条件を変える。
- L2 hint prefetch 8/16 stageはrun 39881882325で遅くなった。126 µsに必要なcache量を計算できても、先読みの効果の証明にはならない。
- K-loop software pipelineはhoist以外が不採用。CはWG間の独立task化だが、同じregister/code-size制約を受ける。

まずAとBを独立に検証し、Cはコンパイル成立性を先に確認する。Dは通知・bufferの設計量が大きく、現行の最後のFC2→終了の実測短縮可能時間を確認してから進める。各案の効果は重なるので、個別の旧timeline時間を足して74 µs削減と見積もらない。

さらに大きい案として4 CTA/expertのclusterでFC1中間値をDSMEMに保持する方法もある。64×1536 FP8は96 KiB/expertで4 CTAへ分割できるが、32 expert×4=128 CTAの負荷偏り、local sharedへの再配置、cluster同期が必要。重み604 MBは減らず、主効果は中間値の受渡し。最初の候補にはしない。[HopperのDSMEM・TMA・cluster occupancy仕様](https://docs.nvidia.com/cuda/archive/13.0.3/hopper-tuning-guide/index.html)

## 4. claude-note-20261023.mdとの照合

[Claudeのメモ](../../claude-note-20261023.md)はbatch拡大による費用分担を提案している。これは有効な別軸であり、126 µsが未達でもgoodputが上がる可能性がある。ただし次を補正する。

1. **cap132悪化の直接原因にdense GEMMがある。** run 39699383079ではFA3は50.8→50.8 µs、denseは15.8→21.6 µs/call、MegaMoEは199.2→207.0 µs。132行がdenseのBM128を超えたことが大きい。FA3・MegaMoEだけでなく、denseのM192形状も対象にする。[実行記録](2026-10-20-firmus-prefill-overlay-results.md)
2. **SwapABの144はtile容量でなく検証範囲の上限。** [host patch](../../patches/deep-gemm-host/sm90-swap-ab-144.patch)にも明記されている。現host heuristicは平均64 token/expert以上でgeometryを変更し、192 token/rank・4 rank・top8・128 expertなら平均48。従って192でもM64/2WGの候補に入る。まず同じtileで適用上限を延ばして調べることは可能で、tile変更が必須とはまだ言えない。ただしpool capacity・graph shape・偏ったroutingと複数M tileを検証する。
3. **192でMegaMoE 205 µsは未検証。** 平均expert行数は32→48に増え、N_SWAP、promote、epilogue、転送が増える。重み量がほぼ一定でも時間一定とは限らない。4 token対128 tokenの比較も、active expert数や形状が異なり、192への外挿根拠として弱い。
4. **FA3 split修正だけで69 µsにはならない。** 既存のsplit固定実験も34–36 requestsで73.8–75.4 µs。現companionは [Python guard](../../patches/sglang/decode-fa3-verify.patch) と [C++ wrapper](../../kernels/fa3-verify/wrapper.cpp) の双方でbatch32–36に制限され、48対応には拡張と数値gateが必要。combineの空split・未初期化partial・LSE sentinel・layoutをまず確認する。現在の [combine](../../kernels/fa3-verify/combine.cu) はweight 0でもpartialを読み掛けるので、NaN partialの扱いを確認すべきだが、それが既存failureの原因とは未確定。
5. **+15–25%とp99約19 msは予測として未確立。** メモの時間表から `1.5/(376/329)=1.3125` は再計算できる。しかし205/69/40/62 µsの入力値が仮定で、平均ITLの比例計算からp99は確定できない。Prefill queueが空の割合も、そのまま余剰処理能力にはならない。

192行案の検証目標をprofile周期で逆算すると、accept長とbatch充填率が同じなら、容量+15%には周期41.51 ms以下、+25%には38.19 ms以下が必要。これは現行31.8256 msに対する容量比の式であり、通常benchのITL/SLAを直接保証しない。

今回の126 µs調査をbatch192へ置き換えることはしない。goodput目標では両方を比較する価値があり、実装開始時は「128行固定の時間短縮」と「192行への拡大」を別commit・別比較にする。source変更はbranch・新規build、一度のbundle servingだけで採用しない。

## 5. 効果の換算と実験での到達判定

| MegaMoE到達時間の仮定 | 現行からの周期短縮 | Decode周期の処理速度比 | goodputが同率で伸びる場合のみ |
| --- | ---: | ---: | ---: |
| 160 µs | 3.755 ms | +13.4% | 約9724 tok/s |
| 150 µs | 4.695 ms | +17.3% | 約10061 tok/s |
| 140 µs | 5.635 ms | +21.5% | 約10422 tok/s |
| 126 µs | 6.951 ms | +27.9% | 約10973 tok/s |

通常goodputとprofile周期は別pass。Prefill、accept長、batch充填率、SLA適合率が変われば右列の比例は成立しない。

次のGPU検証では、まず現行版の「起動→最初のFC1」「FC1/FC2処理中」「最後のFC2→終了」を軽い別passで測り直す。HBM bytes/帯域は対象kernelに帰属できる別profileで取得し、可能なら同shape・同weight配置のTMA read基準も測る。1 kHzのNsight Systems sampleだけで200 µs kernel内部の帯域を推定しない。

測定は現行image＋採用overlayを対照に、通常の数値gate、実serving由来routing、changed-input graph replay、idle/mixed rank、skewを含める。コンパイル・CPU検査を先に行い、4 rank最大時間・複数run・交互実行で評価する。通常benchとprofileを分離し、公式serving/parser/artifact回収まで確認して採否を決める。

現時点で言えるのは、**150–160 µsは実効4.2 TB/s前後＋露出費用約10 µs、140 µsは4.5 TB/s前後＋約5 µsという設計条件であり、126 µsはさらにcache/真の先行重複またはほぼ全費用の隠蔽を要求する**、ということ。各条件を満たせるかは未測定である。今回追加したのは本設計記録のみで、runtime/source mirror、ジョブ、既存の未追跡ファイルは変更していない。
