# Decode案D: chunk順FC2とdispatch warpによるstreaming combine

実装のみ。push、Actions、scheduler、GPU実行は行っていない。性能改善・GPU数値一致・CUDA graph replayの実機成立は未検証であり、採用済みの最適化ではない。

候補は `patches/deep-gemm-bench/decode-stream-combine/stream-combine.patch`。既存kernel-bench provisionがserving overlay適用後に発見・JITする形式で、production configは変更しない。host `_C.so`、workspace allocation、shared-memory capacityの変更は不要。ソースを変更するため将来の実験にはcandidateの新規JITが必要。timeline markerは追加していない。

## 基準と適用順

- image: `39899235870`、image内header SHA256 `ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1`。
- `configs/firmus-framework.json` の5本を順序通り全て適用したheader SHA256: `77ee7caea6e5736803b6034f83575f9a846d30c54ce4970ced7c65450a9be497`。
- 本候補適用後: `b41bdda2ef87c05aa4e3363a26d8be2b4943258d19da9bf5284559f4b7644189`。
- mirrorとmainのソースは変更せず、専用worktreeの実験branchで作成した。

## 実装範囲

SwapABかつ全expertを1 waveで処理するQwen geometryで、動的claim ringの絶対task indexを `FC1全task → FC2の512-channel chunk → pool block → N128 tile` に変換する。FC1は既存と同じexpert優先順を保つ。各taskの算術・量子化・scatter配置は変えない。FC2の全前提FC1は、より小さいtask indexにある。

pullを終えた2 dispatch warpがcombineを行う。1 work itemは1 tokenの512 channelで、chunk優先でCTA/warpへ割り当てる。各laneは8個のfloat2 accumulatorを持ち、slot 0→7の順番で既存と同じFP32 `fmaf` を行い、最後にBF16へ丸める。負のexpert slotは飛ばし、全paddingは0出力。register-onlyの通常global load/storeなので、GEMM中のshared CD/stage/ring/barrierを再利用しない。dispatch register budgetを48→64に増やすが、総budgetは49,664で上限64,512以内、shared memoryは増えない。

### 全geometryのproducer protocol

混在rankで一方だけreadyを待つdeadlockを避けるため、Qwenのhidden=4096/intermediate=1536/ranks=4/Top8では、SwapABに限らず全geometryが通知とcollectiveに参加する。streaming consumerとchunk順schedulerの選択だけをSwapAB/全expert-waveに限定する。partial-waveやnonswapは従来combineを保つ。

未使用の `l1_topk_weights_buffer` 先頭を、local expert×8 chunkのuint32 counterへ転用する。32 expertなら1,024 bytes。元のpoolは全geometryで同じcapacityを使い、host側の追加allocationは不要。静的assertでpool容量と `512 % BLOCK_N == 0` を検査する。model形状と最大buffer capacityは既存のrank間契約通り同一であることが必要。

各FC2 tileは、scatterを完了した全math threadで `__threadfence_system()`、math-only CTA barrier、leaderの `atom.acq_rel.sys.global.add.u32` の順に公開する。partial Mの非active WGも早期return側で同じfence/barrierへ参加し、active WGだけが待ち続けることを避ける。leaderのWG0はvalid_m>0なら必ずactiveで、通知はCTAにつき1回である。加算値は `valid_m * (BLOCK_N / 128)`。全M block・全N tileのrelease RMWを同じcounterに連鎖し、consumerはremote counterを `ld.acquire.sys.global.u32` で読む。期待値はremote expert token count×4であり、BM64/128、BN128/256の混在でも一致する。lane 0のacquire後にwarp同期し、全laneが対応BF16を読む。

これはexpert×chunk単位のreadyで、tokenごとのatomicではない。複数M blockのexpertでは、そのexpertの全M blockが当該chunkを完了するまで待つ保守的な形である。各FC2 CTA tileが1回RMWを行い、counterはchunkに集約する。tokenごとの最短開始時刻や4rankへのready複製までは追わない。

### cleanup・初回・replay・循環待ち

初回allocationのzeroに依存せず、毎callのdispatch先頭で全ready counterをzeroする。既存のdispatch grid syncとbefore-pull NVLink barrierがresetをproducerより前へ順序付ける。未証明の `%gridid` epochは使わない。

producerはconsumer完了を待たずFC1/FC2を完走する。consumerは自分のpull完了後に待機を始めるため、未処理pullを保持したままproducerを待つ循環は作らない。既存math側before-combine rank barrierとdispatch/math joinを残す。そのjoin後に全geometryのdispatchでconsumer完了NVLink barrierを追加し、前後のgrid syncも行う。これにより、どのCTAもremote consumerが読んでいるexpert countを早期cleanupできない。

`comm::nvlink_barrier` はtagごとの独立stateではなく単一rank phase counterを共有する。そのため追加barrierをmath側barrierと並行に呼ばず、math barrier→join→consumer barrierの順にする。既存の最終cleanup barrierも残す。次replayのresetは前callのkernel完了後にしか始まらない。

## GPU不要検査

`qwen-experiment-metrics/tests/test_decode_stream_combine.py` の16ケースで、task全単射、chunk順、動的claimのuint32 wrap、FC1依存が先行すること、mixed BM/BNのcounter充足と全データ公開、初回dirty counter、変更入力・allpadding replay、出力全要素の一意所有、slot順、cleanup依存DAG、partial Mの非active WGを含む全threadのpublication barrier参加を確認した。既存kernel-bench fixtureも24件pass・3件skip、追加testのRuff検査はpass。CPUモデルはGPUメモリモデルやWGMMAの実証を代替しない。

`ssh firmus` のlogin nodeでCUDA 13.0 nvccによるCPU compileのみ実施した。共有抽出toolchainはread-only、自分のcompile directoryのみ作成・削除した。compiler optionsは次の通り（includeと入出力を加えて使用）。

```sh
nvcc --cubin --std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -O3 --ptxas-options=-v,-warn-spills,--register-usage-level=10 -gencode arch=compute_90a,code=sm_90a
```

共通instantiate値はmax_tokens=4224、hidden=4096、intermediate=1536、experts=128、topk=8、pool=141312、padded_sf_pool=2260992、dispatch/loader=64/64、SMs=132、ranks=4、activation clamp=inf、fastmath=true。通常wave=32、partialwaveだけ16。

| geometry | BM/BN/stages/math threads | control registers / stack / spill stores, loads | candidate registers / stack / spill stores, loads |
| --- | --- | --- | --- |
| SwapAB Decode | 64/128/8/256 | 168 / 8 B / 0,0 B | 168 / 0 B / 0,0 B |
| nonswap | 64/128/8/256 | 168 / 8 B / 0,0 B | 168 / 8 B / 0,0 B |
| Prefill2WG | 128/256/3/256 | 168 / 0 B / 0,0 B | 168 / 0 B / 0,0 B |
| Prefill512 | 128/256/3/512 | 96 / 24 B / 20,20 B | 96 / 24 B / 20,20 B |
| SwapAB partialwave | 64/128/8/256 | 168 / 8 B / 0,0 B | 168 / 8 B / 0,0 B |

Prefillはreuse_accum=true、arrival_counter=true、fullsync=false、512-threadのみepilogue budget=112。他はfalse/false/true、budget=0。全てsplit_phase=false。全instantiateがcompile成功し、WGMMA serialization warningなし。Prefill512のspillは対照にも存在する。Decode候補のSASSで `MEMBAR.SC.SYS`、`ATOMG.E.ADD.STRONG.SYS`、`LDG.E.STRONG.SYS` を確認した。これは順序付け命令が消えていない確認であり、実機correctnessの証明ではない。

## 残る判定

GPU数値gate、mixed-rank geometry、idle rank、変更入力graph replay、公式routing、通常bench、公式serving/parserの確認は未実施。consumerが2warpしかないこと、各tileのsystem fence、expert全M完了待ち、追加collective、task decodeのprefix走査が利得を相殺する可能性がある。126µsへの到達やmainに対する優位性は主張しない。ジョブ投入は今回の指示で禁止されており、このbranchの実装とCPU検査までを成果物とする。
