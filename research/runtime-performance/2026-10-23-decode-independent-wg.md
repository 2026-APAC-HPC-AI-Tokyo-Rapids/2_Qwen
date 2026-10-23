# Decode案C: 独立math warpgroupの実装（GPU未検証）

## 状態

`perf/firmus-decode-independent-wg-20261023` に独立task/queue/outputの実装を置いた。現行image 39899235870のheader（SHA256 `ff7a58cca00aa42a90386d12b0437b5d8caf35eb0ad082a8a1a0b4b74001a2a1`）へ `configs/firmus-framework.json` の既存5 overlayを適用した状態を対照とする。source mirrorは変更していない。

CUDA 13.0.88によるCPU compileは成功した。384 threads、168 registersでWGMMA直列化warningはないが、**8 bytes spill stores / 8 bytes spill loadsが残り、spill 0の資源目標は未達**。GPU correctness、CUDA graph replay、速度、公式result parser、servingは未検証。126 µs達成やmainに対する性能優位性を主張しない。ユーザーの追加指示に従いpush、Actions、scheduler、GPUジョブ、新OCI buildは実行していない。

## 変更

- 2 math WGは同じN128 tileの半分を担当する代わりに、それぞれ独立したM64/N128 taskを担当する。FC1では1 WGがgate/upを合わせた128 channels、出力SF64 group全体を所有する。
- SwapABのM64 fragmentを2本扱い、各threadの最終FP32 accumulatorを32から64個に増やす。K loopを大きくunrollせず、各K反復内でWGMMAをdrainする。FC2のlow→high promotion順を保つ。
- 2 load warpはA/B分担からqueue専属のA+B+SF producerへ変わる。各queueは3 stage、full barrierはproducer 1、empty barrierはmath warp 4のarriveを待つ。別queueのempty/fullを待たない。
- task assignment ringを4 slots×2にし、各ringは専属128 math threadsだけが読み終わりを通知する。最初の264 taskを2×132 workerへ配り、その後は既存rank-wide counterを用いる。schedulerは132を加算するため、next-current-132のunsigned補正を使う。263→264では補正が`0xffffff7d`となり、modulo 2^32の加算で264へ戻る。
- outputをWGごとに20,480 bytes確保する。FC1の全64-channel amax、FP8 quantization、TMA storeとready公開を各WG内で完結する。4 threads/tokenの既存quantizerは128 threadsで32 tokenずつ2回処理する。
- FC2の2 fragmentをWGの独立BF16 bufferへ保存し、128-column vector scatterを行う。epilogueの同期をWG内128 threadsにする。最終grid/NVLink barrierとworkspace cleanupは維持する。
- host側の`DG_SM90_DECODE_INDEPENDENT_WG=1`がQwen EP4/128 experts/Top8/4096/1536のSwapAB Decodeだけを6 stageへ切り替える。未設定・0は既存8 stage。Prefillや他geometryは変更しない。headerは該当geometryの6-stage instantiationだけ独立WGになる。

## 資源計算

|領域|bytes|
|---|---:|
|expert countsとring用padding|1,024|
|dispatch send buffers|8,192|
|2 WGの独立output（FP32 16,384 + FP8 4,096ずつ）|40,960|
|A（8,192）+ B（16,384）+ SFA（512）×6 stages|150,528|
|dispatch/combine固定barriers|144|
|full/empty barriers×6 stages|96|
|合計|**200,944**|

現行は230,672 bytes。ringはexpert-count padding内の768..927 bytesに収まり、別途shared memoryを追加しない。FC2 outputはWGごと16,384 bytesなので20,480-byte出力領域を共用できる。register budgetは384-thread launchの168のまま。`setmaxnreg`だけを増やす回避はしていない。

## CPU検証

`python3 scripts/check-sm90-independent-wg.py` はvalid_m=1..64のFC1/SF64・FC2出力を各要素ちょうど1回所有すること、quantization quad、独立stage/ring範囲、shared memory総量、非同期task割当を検査する。task263→264のunsigned wrap、empty expert、複数M tile、expert境界、FC1→FC2境界も含む。PASS。host/header両patchの適用確認、Python構文、shell構文、`git diff --check`も実施した。

`containers/check-sm90-independent-wg.py` はGPUを初期化せず、image headerのcopyに既存5overlayとcandidateを適用してnvccとcuobjdumpを使う。実JITと同じ`--register-usage-level=10`を指定する。比較結果:

|版|spill stores/loads|WGMMA C7510/C7514|cubin bytes|SASS SHA256|
|---|---|---|---:|---|
|現行5overlay Decode 8stage|0/0|なし|125,784|`e9666dd91770cffdfdea1500418a487c44fe0cfab59e8f05874df8957b6e0301`|
|candidate header、opt-out Decode 8stage|0/0|なし|125,784|上と完全一致|
|現行5overlay Prefill2WG|0/0|なし|127,704|`f1c97eadd9d91af25082848bc2ce5e7e24d40d99947cb469096ffe88acaa09cd`|
|candidate header、Prefill2WG|0/0|なし|127,704|上と完全一致|
|独立WG Decode 6stage|**8/8**|なし|180,056|`ecf2f83971c644b884f3ae31e4483a2486ffa9180d43e892f547ce274c5581d2`|

候補のstack frameは16 bytes。spill 8/8のほかに既存combineの動的indexを持つ小配列のlocal-memory利用があるため、spill値だけを「全local-memory量」と呼ばない。

最初のcompileではscheduler callbackがinlineされずC7510とspillを生じた。scheduler/math/run_swap_ab lambdaをinline指定してC7510を除いた。FC2の一時accumulatorを1本にする、queue IDのhardware再読出し、raw thread IDからの計算、invalid tokenのscale predicate除去、phase型specializeも試したがspillを解消せず、余分な変更は撤回した。残8/8は成功扱いにしない。

## Buildと後続の検証

host patchを`patches/deep-gemm-host/series`へ追加したため、opt-in使用には新しいhost extensionを含む新OCI buildが必要。host `_C.so`自体の再buildは今回実行していない。candidate JIT patchの正本は `patches/deep-gemm-bench/independent-wg/sm90-decode-independent-wg.patch`。同directoryの`env`がopt-inを設定する。configの対照overlayは既存5本のままで、既存bench provisionerが候補copyにだけ6本目を適用する。Dockerfileとbuild contextへCPU compile helperを接続したが、installed base headerは変更せずoverlayをcopy上で検査する。

build helperはcompile failure、WGMMA直列化、opt-outのDecode/Prefill SASS変化では失敗する。spill量はJSONとwarningへ残し、`spill_zero_target_met=false`、`gpu_validated=false`を記録する。**sourceがcompile可能であることとspill 0の性能目標を分け、新image構築そのものはspillだけでは失敗させない。** この仕組みは性能採用の承認を意味しない。

次に実行が許可された場合は、同一新imageのcontrolは既存5overlay・env未設定、candidateはそれに6本目のheader patch・env1を加えて既存kernel-benchで比較できる。candidate headerのenv0経路も対照SASSと同一であることは上記CPU compileで確認済み。まずofficial routing由来、balanced/local、valid_m端数、idle/mixed rank、changed-input graph replayの数値gate、次に複数runの交互順timingを行う。3stageへの削減、B先読み専属warpの廃止、cubin増大、残spillが性能を悪化させる可能性がある。公式serving/result parserまで確認する以前にmainへ採用しない。

## 直接ソース再点検（同日、CUDA再compileなし）

canonical 5-overlay header（SHA256 `77ee7caea6e5736803b6034f83575f9a846d30c54ce4970ced7c65450a9be497`）から再構成して、ring・stage・epilogue・host生成引数を直接確認した。現行hostが生成する独立WG Decode条件で新たな確定バグは見つかっていない。これは実機の正当性確認ではない。

一方、kernel templateを手動instantiateする場合の未サポート組合せを拒否していなかったため、独立WGに限り `kSplitPhaseHotPath=false`、`kL2ArrivalCounter=false`、`kReuseAccumAsFinal=false` を要求する `DG_STATIC_ASSERT` を追加した。

- split-phase=true: loaderは独立2queueだがmathは `kSpecializePhase` 分岐を優先し、従来のstatic task streamへ入る。例えばSM0/WG1のmath task 0に対してloader queue 1はtask 1を供給し、ringもmath側から消費されない。
- arrival-counter=true: 専属loaderは24bitのFC1完了maskを待つが、FC1側はcounter方式へ入る。プロトコルが一致しない。
- reuse-accum=true: mathはSwapAB専用mainloopより先に通常layoutのaccumulator再利用分岐へ入り、その後のSwapAB epilogueとの対応を保証していない。

現在のhostはBM64/BN128/256math threadsのDecodeについて、上記3値をすべてfalseにするので拒否されない。独立WGを選ばない既存geometryには新しい制約を掛けない。`kL2EpilogueRequiresFullSync` は独立WG分岐で専用128-thread barrierへ置き換えるため、新たな値制限は不要。

`check-sm90-independent-wg.py` はpatchに実在するassert式を抽出し、通常C++の `static_assert` としてg++/clang++の `-fsyntax-only` で検査する。独立WGの現行host flagsは成功、他の7組合せは同assertにより拒否、非独立の8組合せは成功。既存配置checker、Ruff、canonicalからの `git apply --check` も成功した。レビュー時にはschedulerのexpert/phase/wave遷移をそのまま写した別CPUモデルと独立したflatten列を照合し、wave=1/2/4/8/16/32・空expert・複数M tileを含む102条件で、2×132 workerの非同期claimとunsigned補正が一致した。

今回の追補ではCUDA/nvcc compile、host `_C.so` build、GPU、SSH、Actions、scheduler、pushは行っていない。上表のCUDA資源・SASS値はassert追加前の実測であり、追加後headerのCUDA再compile結果ではない。

assert追加後の候補header SHA256は `c40d4c141aea45bb17a864ae8d7088a97363f12d0188b2584e558203aacb11ad`。基準headerからの再適用を親担当でも照合した。
