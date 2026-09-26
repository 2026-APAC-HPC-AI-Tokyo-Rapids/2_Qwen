# Issue #16: 実測で校正した構成探索

追跡: [Issue #16](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/16)。公式結果が出るまでは構成の優位性を結論にしない。

## 比較条件と正本

公式ターゲットは `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8@e156cb4efae43fbee1a1ab073f946a1377e6b969`。実配備SGLang sourceは `fc275674a97d78553b22597ee72c8fd48f5420f4`、コンテナSHA256は `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`。全16 GPU、PD、BF16 KV、ISL4000/OSL200、greedy、公式warmup60秒/測定120秒/grace30秒を使用する。

- 設定と変更理由: 各実験commit、`configs/aspire-serving.json`、`forward-plan.json`。
- 実行経過: Actions/PBSログ。
- raw: `ssh archive-host` の `/srv/qwen-artifacts/<Actions run>/1`。
- CPU探索・派生FPM・holdout: 同サーバーの `qwen-artifacts/research-issue-16/20260926`。
- 規約に合わせた性能判定には `scripts/parse-aiperf-result.py` のjoint SLA判定を使う。このスクリプトは主催者提供の公式parserではない。microbenchmarkや512-token generic parityで主催者の全件accuracy検査を代替しない。

## 校正の方法

`forward_probe.py` はstock one_batchを使い、warmupを除く3反復を逐次保存する。TP4はfull-prefill batch8までに制限した。batch16×4000のMoE一時tensorは3.90625 GiBになり、観測された空き3.69–3.84 GBを超えてOOMになるためである。これはdecode batch16が不可能という意味ではない。大きいdecode batchの初期KVは `chunk_probe.py` が8192-token以下の本物のmodel forwardで生成する。

AISimulateは `992dddc7ee2bad8fd523e7db99eea4ac64254766` に固定。external FPM schema6へ実測を変換し、独自の測定policyを明示的に受理するパッチを適用した。補間数式は変更せず、fallback/extrapolationを禁止する。decodeは31 forwardsの中央値をISL+16座標へ置く近似なので、per-iteration測定ではない。

最初のwhole-forward-onlyデータではprefix KV付きprefillを予測できなかった。追加した旧chunk gridでも `(batch=3,new=8190,cached=192)` とdecode graph境界の補間が失敗した。これを根拠に小さいprefix、batch3、batch rung隣接点を追加した。合成fixtureの成功は入力契約の確認に限り、性能値として利用しない。

## 回収済み結果

Actions `36226304028` / PBS `212982` は55条件・165 raw rowsを満たし、PBS Exit_status=0、walltime17分54秒。ActionsはSSH監視期限で失敗したが、回収後のstrict summary検証は通った。詳細とholdoutは `forward-36226304028` に保管。

入力4000を除いて学習した26点のholdout誤差は中央値0.672%、最大6.291%。2点は範囲外として予測を拒否した。同一割当内のshape holdoutなので、独立した公式AIPerfの予測精度は別途評価する。

さらに `compare_predictions.py` で、別割当の `36224333888` decode worker（TP8/EP8）の完了済み23条件×prefill/decodeを検証した。46点全て予測でき、汎用op modelはAPE中央値6.333%/最大29.920%、実測FPMは中央値0.222%/最大3.116%。検証元run全体はTP4 OOMで失敗しているため、この比較は完了したTP8 workerだけを明示的に抽出した。測定点は同じshapeの独立割当であり、未測定shapeやserving全体への一般化とは区別する。汎用backend release slotと配備source commitの差も含む比較である。

同runのNIXL geometry2は source TP4/8 × destination TP4/8 × 128/1024/4000/4200 tokens の16点。94層×K/Vの188 GPU segmentsを実際に書き、destination内容を検証。TP4→TP8ではKV headの複製が必要であり、4000tokenの7.887msをTP8→TP8の3.995msと区別する。registration・serving queue・block packingは測定範囲外。

汎用データによるSGLang/vLLM/TRT-LLMの36候補CPU screenは `backend-screen` に保管。SGLang P2TP4/D2TP4 c24を優先候補としたが、配備versionと実測ではないためbackendの優劣は未確定。

### Dense gridと実機対照

`36227604382` / `6b80944` / PBS `212992` はActionsを含め正常終了。55 full cellsと561 chunk cells（187条件×3 workers）を回収した。新dense gridで学習し、独立割当 `36226304028` の110点を検証すると、汎用op modelのAPE中央値7.479%/最大32.871%に対して実測FPMは中央値0.234%/最大3.572%。同じshapeの独立割当間の比較であり、未知shapeの精度を示すものではない。

`measured-search` の24候補中23候補は補間範囲内で完走。P2TP4/D1TP8 c32はdecode batch32の長いcontextが実測範囲外で拒否され、性能0の候補として扱わない。上位はP2TP4/D2TP4 c32（1960.2 predicted good tokens/s）、同c24（1530.7）。c16（1147.4）を対照として同c32を実機へ投入した。320件の合成requestによる探索なので、固定120秒のAIPerf結果と同一のworkloadではない。

最初のtarget-only `377b690` / Actions `36227731991` / PBS `212994` は、VPN更新でActionsのSSH接続が失敗した後もPBSが継続し、Exit_status=0、walltime12分33秒で完了した。rawを手動回収して規約準拠parserを再実行し、一致を確認した。canonical archiveは同runの `1/recovered-after-vpn`。

|P2TP4/D1TP8 c16|good tokens/s|平均TTFT ms|平均ITL ms|
|---|---:|---:|---:|
|実機、target-only|903.434|388.931|15.656|
|汎用op model|913.632|241.903|16.239|
|実測FPM|1022.552|233.913|14.402|

実機554 requests、0 errors、joint SLA fraction 1.0、OSL mismatch 0、ローカルparser valid。baseline一回の結果なので再現性は未確定。実測FPMのthroughput予測は実機より13.185%高く、汎用op modelの1.129%より誤差が大きかった。forward単体の精度改善をserving全体の精度改善と同一視しない。CPU overhead、queue/overlap、有限requestと定時間測定の差が未モデル化で、TTFTも過小予測している。

## ローカルモデル配置

計算ノードのTMPDIRは `/raid` 上のext4で、観測時に約27 TB空き。`stage-qwen-node-model.sh` でtargetをノードにつき1回コピーし、同一ノードのworker間で共有する。コピー時刻/所要時間、source/destination filesystem、byte量、indexとshardサイズを記録。copy+loadの総時間と各段階を比較し、serving throughputへの効果とは分けて評価する。コピーはそのallocationの一時領域に限り、終了時に削除する。

`36227604382` ではcopy203/214秒、local TP4 load31.76–32.68秒、TP8 load9.49–10.39秒。共有filesystemの `36226304028` はTP4 load789.29秒、TP8 load59.17–64.44秒。TP4のcopy+loadは短縮したが、TP8はcopyを含めると遅い。別割当の観測であり、filesystem負荷を固定した反復比較ではない。`0ba5297` 以降は子コピーに不要な76 GiBのHF `.cache` を除外する。共有source自体は削除しない。

## 投機デコード

node-local、c16、P2×TP4/EP1、D1×TP8/EP8の対照条件:

|条件|commit|Actions|
|---|---|---|
|target-only|377b690|36227731991|
|NGRAM 4 tokens / BFS1|e833b6e|36227653702|
|EAGLE3 3 steps / topk1 / 4 tokens|a912929|36227719367|

EAGLE3候補は [Nebius Instruct版](https://huggingface.co/nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507)、revision `ff0f0c0999f6c86b21dfd2f5803eebc4ac712224`、CC-BY-4.0。model cardはInfinity-Instructの一般データ由来と記載。公開受理率は今回の公式条件の予測値として扱わない。[RedHat版](https://huggingface.co/RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3)も存在するが、まずnative形式に近いNebiusを選択。[NVIDIA Thinking版](https://huggingface.co/nvidia/Qwen3-235B-A22B-Thinking-2507-Eagle3)よりターゲットと対応するInstruct版を優先する。

`prepare_eagle.py` はBF16 weights、integer token maps、hidden/vocabulary shapesを確認し、元config/model card/weight SHA256/runtime configを保存。SGLangのarchitecture selectorとBF16宣言のみ変更、重み無変更。PDで必要なaux hidden statesを転送するためprefill/decode両方にEAGLE3を指定する。公式targetは変更しない。

Nebius draftは572,538,880 BF16 parameters、weights 1,145,943,008 bytes（integer token mapsを含む）。NGRAM run `36227653702` の最初の致命的エラーはPD warmupの `ngram_worker.py:242`、`spec_info=None` による `accept_tokens` 属性参照であり、後続SIGKILLをOOMと解釈しない。`ed4c823` はPD初期NGRAM状態を構築する17行のsource patchを所有側build contextに追加し、新規imageで再実行する。既にoutput_idsにあるbonus tokenを重複させないゼロ初期accept lengthsをCPU torch回帰検証済み。target verificationは変更していない。GPUのE2E検証は別途必要。

## 実行経路の失敗

`cbd5b7e`/`fd1bbff` は既存profiler staging文字列契約でCPU validation失敗、GPU投入なし。`6b80944`で修正、145 testsと静的検査を通過。empty commit `26372b4` はpaths filterにより未起動なので対照runに数えない。

旧chunk run `36226700537` と旧baseline `36226778215` は監視SSH timeout後のworkflow cleanupで中断（PBS Exit_status143）。旧NGRAM `36226824669` は待機中に同じ失敗となり公式測定なし。chunk runのdecode workerだけはfull23条件＋chunk53条件が回収済みであり、全run完了とは扱わない。

`51b620c` はobserved remote shellの約78–110秒を根拠に1回の監視期限を60→180秒へ変更した。過去commitの実行には遡及しない。strict terminal/artifact判定と接続失敗時の有限retryは維持する。
