# Decode側でのEAGLE3初期化と中間状態取得（2026-10-03調査、10-04記録）

**Decode側で中間状態を計算することは可能だが、現在のEAGLE3と同じdraft初期状態を作るには全promptのtarget中間状態が必要である。ユーザー案の「Dで最初の通常decodeを行い、生成開始からdraftを初期化する」方法は全prompt hiddenを不要にできる有力な設計候補。ただしexactな現draft初期状態の移設とは別で、draftの有効context範囲を分離するsource変更が必要。exact初期状態を保つ直接的な案は、Pが既に計算するtargetの3層hiddenを転送し、Dでdraft prefillを実行する方法。CLIだけでPのspeculationを外してDだけEAGLE3にする構成は、固定版のPD経路では成立しない。** 現在のwireはtarget FULL hiddenではなく、draft prefill済みKVと最後のdraft hiddenを送っている。この違いを前提に比較する。

固定sourceはSGLang `fc275674a97d78553b22597ee72c8fd48f5420f4`。targetはQwen3-235B-A22B-Instruct-2507-FP8（94層、hidden4096、KV4heads、head_dim128）。draftはnebius由来のLlamaForCausalLMEagle3、1層、hidden4096、intermediate12288、KV4heads、BF16重み。c48正本 `37121275791/1/source/source/draft-runtime-config.json`を参照した。draft runtime configにaux層指定はなく、固定sourceの既定3層を使用する。現在target/draft KVはいずれもFP8であり、BF16 draftという呼称は重みのprecision。mirror・build・runtimeを変更せず、GPU/Actionsも追加していない。

## 現在、PとDが行う処理

| 境界 | Pの処理と状態 | Dに必要な状態 |
|---|---|---|
| Target prefill | promptから94層target KVを作り、target logitsから最初のoutput tokenを選ぶ。EAGLE3では3層の全token hiddenもcapture | 全target KV、最初のoutput token、sampling/logprob等のmetadata |
| Draft prefill | target全token hiddenとshifted prompt token列で1層draftを実行、draft KVを埋める | 全prompt分のdraft KV |
| 初回draft seed | draft最後のhidden、topk確率/indexを作る | `[B,4096]` draft recurrent hidden、topk、targetのfirst token（bonus token） |
| PD handoff | target/draft KV GPU転送＋metadata DRAM転送 | 両KVとmetadataの完了・room一致を確認してPREBUILT状態へ移行 |
| D開始 | Pでは以後のtarget generationをしない | Pで作られたseedからdrafting、target verification、draft extendを継続 |

固定版 [`eagle_worker_v2.py:1089`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/speculative/eagle_worker_v2.py#L1089) はtarget `CaptureHiddenMode.FULL`→target forward→`_draft_extend_for_prefill`の順。FULLは94層すべてのactivationを保存する意味ではない。Qwen3の既定capture層は`[2,47,91]`で、各H4096を結合した `[T,12288]`。明示指定時はsourceがindexを+1調整するため、学習configの層indexと内部capture位置を無条件に同じと呼ばない。Qwen3はQwen2MoeModelを継承し、選択layerのpostprocessがauxを収集する。[`qwen3_moe.py:1078`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/models/qwen3_moe.py#L1078)、`qwen2_moe.py:925–998`、[`logits_processor.py:667`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/layers/logits_processor.py#L667) がこのshapeの根拠。

Draft prefillは各requestについて入力を1token左へずらし、末尾へtarget sampled token（途中chunkではchunk-aware tail）を置く。`EagleDraftExtendInput.hidden_states`にtarget FULLを渡す。draft `fc`は通常12288→4096で、その後embeddingと結合し1層attention/MLPを実行する。全位置に対するdraft KVを作り、LAST capture `[B,4096]`とdraft logitsから次のtopkを返す。LASTはtarget3層hiddenとは別のテンソル。[`eagle_worker_v2.py:705`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/speculative/eagle_worker_v2.py#L705) と同:793–816、[`llama_eagle3.py:143`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/models/llama_eagle3.py#L143)、同:211–233。

Target KVとdraft KVは同じrequestのtoken indicesを使用し、`prefill.py:213–221`/`decode.py:450–459`でbuffer pointer/item-lengthをappendする。Pの結果処理 `prefill.py:707–719`はdraftのtopkとLAST hiddenをrequestへ保存する。metadata bufferはhidden `[slots,4096]`BF16、topk probability `[slots,16]`FP32、index `[slots,16]`INT64等。`MetadataBuffers.set_buf:509–520`、`NixlKVManager.send_aux:1848–1887`がDRAM経由で送る。Qwen通常MHAにDSA seedは不要。Dは`decode.py:1763–1871`でroomと完了状態を確認してfirst tokenをappendし、`eagle_disaggregation.py:25–78`で受信draft hidden/topkからEagleDraftInputを構成する。`decode.py:2278`/`decode_schedule_batch_mixin.py:25,113–154`のPREBUILTはmetadataで「prefill済み」を表し、Dで全prompt target/draft prefillを再実行する経路ではない。

## P speculationなし、DだけEAGLE3：既存CLIでは完成しない

Pの`--speculative-algorithm`を省くこと自体と、Dが正しいEAGLE3状態で開始できることは別。固定版に今回必要な「seedなしならDでexact draft prefillを再構築」処理は見つからない。

1. 非spec Pはtarget aux captureを有効にせず、draft model/KVを持たない。送信KVはtargetのみ。通常KVはtargetのattention K/V投影とRoPE等の結果であり、残差hidden4096や3層concat12288を可逆に保持しない。Qwenは各層KV4heads×128で、TP全体でもK/V各512次元。これらからtarget auxを復元する既存処理はない。
2. Dにdraft poolがあるからといって、targetのみのPがdraft KVを送信したことにはならない。`common/conn.py:747`に「Dだけdraft KV」のpointer-layout分岐があるが、これはtarget shardを対応先へ転送するためであり、draft prefixを初期化するコードではない。
3. `scheduler.py:1192–1201`ではEAGLE P/Dはdraft recurrent hiddenの4096/BF16 metadataを確保する一方、非spec Pはpadding16/FP32を使う。`send_aux`はP item lengthでdestination offsetを作る。DSA schemaを合わせるコメントがあっても、このhidden size/dtypeの差とdraft seed欠損は解消しない。受信側はspec有効ならtopk/hiddenを使用するので、ゼロmetadataを有効なseedとして流してはいけない。

したがって、この組合せをCLIのみで実験して成功扱いにするのは不可。明示handoff mode、wire schema、D初期化完了gateを作るsource変更が必要。

## 案A：Pでtarget hiddenを取得、Dでexact draft prefill（第一の設計候補）

Pはtarget forward、3層FULL capture、first token samplingを保持する。P上のdraft attention/MLP・draftKV確保・draft prefillを外し、prompt各位置のauxをDへ送る。DはtargetKVとaux受信後、既存 `_draft_extend_for_prefill`相当をshifted prompt/positions/requestcacheindicesで実行する。得られたdraftKVとLAST hidden/topkを既存PREBUILT seedへ接続する。first tokenはPでsampleしたものをそのまま使用し、Dがもう一度sampleしてtoken sequenceをずらさない。

最小の変更境界は、(a) Pのtarget-only workerにもdraft configに基づくcapture設定を与える、(b) token/chunk対応のGPU hidden bufferと送信payloadを追加する、(c) target-only KV転送とauxschemaを明示する、(d) Dのprebuilt待ちにdraft初期化stageを追加する、(e) draft初期化のcache positions・chunk tail・sampling・releaseを既存仕様と揃える、である。数行のCLI追加ではなく、PD lifecycleの改修。既存8KiB LAST metadata slotをfull hiddenへ拡張するだけではprompt/chunk容量・GPU転送・room lifetimeを解決しない。prefix cacheがPでhitした場合もcached prefix全体のdraft条件付きhiddenが必要で、KV cacheだけからauxは再生できない。まずcache-busted公式入力に範囲を限定しても、将来cache hitの扱いを明示する必要がある。

| ISL4000、FP8 KV、BF16 hiddenの論理payload（TP全体の非重複量） | サイズ |
|---|---:|
| 現target94層KV |367.1875MiB|
| 現draft1層KV |3.90625MiB|
| 現draft LAST hidden/req |8KiB＋topk等metadata|
| P→D target3層FULL hidden `[4000,12288]` |93.75MiB|
| T8192時の3層FULL buffer `[8192,12288]` |192MiB|

draftKV転送をFULL hidden転送へ置き換えると約89.84MiB/request増える。logical bytesであり、TP rank replication、NIXL descriptor、NIC実帯域や同時転送の所要時間は含まない。現在のlayoutでhidden4096はattention headのように当然分割されるわけではなく、重複転送を避けるsource/destination rank ownershipを設計する必要がある。

P側で既存draftのfcだけを適用し、正確なprojected `[T,4096]`を送る派生案なら4000tokensで31.25MiB、8192tokensで64MiBまで減らせる。学習済みfc/任意fc_norm、dtype・丸めを同じにし、Dがfcを再適用しない専用境界を作る。Pに小さいprojection weightと処理を残すが、draft attention/MLP/KVはDへ移せる。raw auxより転送は小さいものの、現draftKVより約27.34MiB/request大きい。lossy FP8 hidden圧縮を無条件の代替としない。

品質面ではtarget weights/first-token sampling/verificationを維持し、同じaux・shift・positions・draftKVで初期化できればexact draft状態の再現が目標になる。ただしbackend/TP/collectiveの数値順序差は残るので、seed/logits/KVの許容誤差、greedy outputs、acceptance length、公式parserで検証する。TTFTにはP計算削減と追加転送+Ddraft初期化が競合する。Pは送信bufferの完了まで保持するが、その後にDのdraft計算完了を待たず解放できるようgateを分けられる。Dは既に通常draft KVを持つため恒久KV量は概ね同じ、追加hidden receive/staging scratchとprefill graph/attention workspace、prefillによるdecode干渉が増える。実測stage時間なしに利得は予測しない。

## 案B：Dでtargetの中間状態を再計算する

target通常KVのみから3層hiddenを逆算する方法はない。Dがprompt embeddingsからtargetを前向きに再実行すれば取得できるが、必要な最深captureは91付近であり、全prompt exact auxを作るにはtargetのほぼ全層を再計算する。最後のlm-headや後続少数層を省ける可能性はあっても、KVがあればMoE/MLPを省略できるわけではない。全promptをqueryとして扱う場合は自己KVを二重appendしない別forward/allocator境界も必要。

全target prefillをDで繰り返す案は実装の既存forward再利用には向くが、P/D重複計算、D大batchによるITL悪化、TTFT、activation scratchが増え、PD分離の狙いを弱める。Pを完全に外してDでtarget+draft prefillする構成は別のcollocated serving topologyで、今回の「P負荷だけを減らす」比較ではない。部分target再計算を軽いhidden抽出と呼ばず、必要な最深layerと対象token数を示して判断する。

## 案C：Dで最初の1tokenを通常decodeし、短いdraft contextから開始する

Pのfirst tokenとtargetKVを受信し、Dがそのfirst tokenをtargetで通常decodeすれば、その位置の3層auxと次tokenを取得できる。ただし得られるのは生成開始位置のhiddenであり、全prompt分のdraftprefixKVはできない。この1token forwardだけで現EAGLE3のexact seedを再現することはできない。

別の設計として、draftのみ生成開始以降の短いcontextでcold startし、target verificationは全promptKVを維持することは考えられる。draftのKV valid range、absolute positions、attention mask/seq_len、targetと別のcache allocator/index mapping、初回通常decode→draft seedへ接続、chunk/retraction/graphの扱いを変更する。現実装のtarget/draft同一indicesとfull prefix seq_lenを使ったまま未初期化draftKVを読むことは不可。最後の1token hiddenだけを全prompt seedの代用にする実験も不可。

Targetによる検証と採用samplingを維持する限り、draft context短縮は主に提案品質/受理率の問題として評価できる。しかし今回既定はrejection samplingなしのtop1 chainであり、出力分布の厳密不変をコード読取だけで保証しない。受理率の低下と最初の通常target forwardが利益を消す可能性があるが、程度は未測定。品質とacceptanceを別々に測る。TTFTをprior first tokenで見せても第二tokenまでの待ち/ITLへ費用が移るため、first-token指標だけで優位としない。prompt全hidden転送/target再prefillを不要にできる設計余地はあるが、exact初期化移設とは別の大きい改修で、ユーザー案として優先してCPU設計を具体化する価値がある。

## PのPP2を解放できるか

`server_args.py:8436–8439`はPP>1にspeculative_algorithmなしとdisable_overlap_scheduleを要求する。Pをtarget-onlyにすればこのspec assertは避けられるが、案Aで必要な3層capture/転送がそのままPP対応になるわけではない。

Qwen3のcapture設定はlast PP rankのみで有効化され、Qwen2MoeModelの非last rankは`PPProxyTensors`にhidden/residualしか渡さない (`qwen3_moe.py:1079–1081`、`qwen2_moe.py:959–974`)。PP2でcapture2/47/91がstageを跨ぐなら、early-stage auxを後段へ運ぶか、各stageからDへ順序付きで送るsource変更が必要。最後のrankの通常最終hiddenだけではEAGLE3の3層入力を満たさない。P4TP×PP2は8GPU/workerを使い現在2×TP4 P workerを1workerへ変えるので、worker数・pipeline bubble・overlap無効・KV PP transferまで別のtopology比較になる。PP2はCLI制約解除だけで性能候補になったとは扱わない。

当面はユーザー案Cの短context管理をCPUで具体化し、exact案Aのraw/projected hiddenとの実装・転送比較を行う。D短contextやPP2は独立の改修候補として残す。今回のDP2/NIXL失敗は異種attentionTP transport問題の未確定事象であり、この初期化移設がそれを直すと主張しない。

## ユーザー案のtoken番号と最小変更の詳細

「Decode側で1トークン目だけEAGLE無し」の1tokenは番号を明確にする。現PDではPのprompt prefillがoutput第1token `y1`をsampleし、Dはhandoff時にoutput_idsへappendする。Dが最初の通常target forwardに入力するのは`y1`（absolute position=L）であり、そのlogitsからDが生成するのはoutput第2token `y2`。このforwardで得るauxは`y1`位置の3層hidden。そのaux＋shiftされたdraft入力`y2`で1rowのdraft初期化を行えば、draftは次の提案`y3`以降へ進める。bonus tokenは`y2`に更新する。Pでまだsampleしていないという仮定で「Dが第1tokenを作る」と説明すると、この既存handoffと1tokenずれる。

Pをtarget-onlyにし、Dは一度だけEAGLE workerを迂回して`target_worker.forward_batch_generation(..., capture_hidden_mode=FULL)`を通常DECODE shapeで呼ぶ。1token/reqならFULL出力は`[B,12288]`だけで全prompt tensorは不要。targetKVは[0,L)を保持し、`y1`を既存out_cache_locのposition Lへ書く。target logitsのsampling、grammar、logprob、output emissionを通常decodeと同じに保つ。このstageが完了するまでfake zero draft seedをbuildしない。P/D metadataは共通schemaへ合わせ、seedless handoffを明示する。D-first-normalというrequest状態を`decode_schedule_batch_mixin.process_prebuilt`/decode admission→worker dispatchへ接続する。既存CLIにこのtransitionはない。

その後1rowのdraft initializerを作り、aux_y1/入力y2/absolute position LでdraftKVを生成、LASThidden/topk→EagleDraftInput(bonus=y2)へ接続する。`_draft_extend_for_prefill`のhidden/shift/seed assemblyは再利用可能だが、batchを単にextend_len1へ変えて既存prefixLのまま呼ぶと未初期化draftKVを読む。`eagle_worker_common.prepare_for_draft_extend:154–165`はprefix_lensをtarget batch.seq_lensから作り、`prepare_for_draft:226–254`はtarget req_to_tokenとseq_lensからdraft-write cache位置を割り当てるため、どちらもcoldstart APIではない。model_runner init/weight loadは既存Dのdraft workerを使えるが、draft context初期化のAPIは追加する。

必要な状態はrequestごとの`draft_context_start=L`と`draft_valid_length`、target sequence長とは独立のdraft attention read-index list。初回はdraftのprefix0で1rowを書くがRoPE positionsは0へリセットせずLを保持する。以降draft attentionは[0,L)のslotを読み出さず、有効な生成開始以降だけを読む。target verificationは全promptのseq_len/positionsとreq_to_tokenを保持する。別のdraft req-to-token viewを用意するか、backendがstart-offset付きindexを生成する方式を選ぶ。単にtarget batch.seq_lensを1へ縮めるとtarget検証・cache allocation・sampling seedまで壊すので避ける。page1/top1の現条件に範囲を限定できるが、accepted prefixの分岐は省略できない。

`_draft_extend_for_decode:839–851`のaccept_lensにはbonusを含み、num_correct_drafts=accept_lens-1。target verification後に「acceptされた生成tokenのみ」のdraft valid length/KVを延長し、rejectされた枝は既存cleanupに対応してrollbackする。filter/merge、request終了、retraction、overlap relayにもdraftstart/validlengthを追随させる。絶対positionsとtarget seq_lensの関係、bonusのKV-ready境界は既存spec_v2のまま保つ。最初の初期化はeagerでshape/indicesを検証し、graph対応はその後の別検証とする。

これにより全prompt draftKV/全prompt aux転送を外せるため、PのdraftprefillとdraftKV memoryを減らす目的に沿う。Dは通常target1step＋draft1rowを開始時に負担し、その後受理率が十分なら利益が残る可能性がある。TTFTはPのy1返却時刻が変わり得る一方、y1→y2→draft readyに起動費用が集まり、ITL/第2token latencyを確認する必要がある。draftの提案精度とtargetによる最終検証の精度を混同せず、品質・accepted tokens/step・初期数stepとsteadyの性能を測る。性能利得は現データでは予測しない。未初期化KVを読まない短context設計が完成すれば、target側の精度を落とす変更は必要ない。

## 実装の優先順位と既存profileの限界

最小影響の順序は、(1) seedless handoffの明示状態とP/D共通metadata、(2) D最初の通常target1stepでFULL aux_y1を得るtransition、(3) eagerの1row draft initializerと独立draft valid context、(4) filter/merge/accepted-prefix/retractionの更新、(5) overlap/graphへの接続、である。まず現在TP4→TP4/page1/top1に限定し、異種TP・PP2と同時に実装しない。target verification/samplingの変更は不要で、draftだけcontext範囲を分離する。

既存Nsight `36387126341`は旧auto prefill条件のdecode側が主で、現在CUTLASS Pのtarget FULL capture、draftprefix extend、KV転送待ちの局所時間を分離できない。現在c44/c48 batchログはP待ちとbatch shapeの証拠であり、draftprefixやhidden収集のGPU時間ではない。sourceにtarget/draft stage境界はあるが、今回そのstageの現P実測profileを追加取得していない。94層target対1層draftという層数だけで最大利得を小さいと断定しない。draftのdense幅・lm-head・capture/copy、graph、P解放時点、D側干渉を段階別に測定して判断する。
