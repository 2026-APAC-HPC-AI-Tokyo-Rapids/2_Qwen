# Issue #16: 実測で校正した構成探索

追跡: [Issue #16](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/16)。公式結果が出るまでは構成の優位性を結論にしない。

## 比較条件と正本

入力共有とcacheの調査は [PREFIX.md](PREFIX.md)。実測prefix再利用は約0.077%。AIPerf 0.12.0の通常synthetic経路では、指定されたcache-bust markerがwireへ渡らない挙動をCPUで確認した。公式clientを変更せずに記録している。

公式ターゲットは `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8@e156cb4efae43fbee1a1ab073f946a1377e6b969`。実配備SGLang sourceは `fc275674a97d78553b22597ee72c8fd48f5420f4`、コンテナSHA256は `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`。全16 GPU、PD、BF16 KV、ISL4000/OSL200、greedy、公式warmup60秒/測定120秒/grace30秒を使用する。

- 設定と変更理由: 各実験commit、`configs/aspire-serving.json`、`forward-plan.json`。
- 実行経過: Actions/PBSログ。
- raw: `ssh archive-host` の `/srv/qwen-artifacts/<Actions run>/1`。
- CPU探索・派生FPM・holdout: 同サーバーの `qwen-artifacts/research-issue-16/20260926`。
- 規約に合わせた性能判定には `scripts/parse-aiperf-result.py` のjoint SLA判定を使う。このスクリプトは主催者提供の公式parserではない。microbenchmarkや512-token generic parityで主催者の全件accuracy検査を代替しない。
- **決定的推論モードは規約の必須条件ではない。** ユーザー指摘を受け[Q4](https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/1.%20Qwen3%20AI%20Task%20Rules.md#q4-short-answers)を再確認した。浮動小数点演算順序による違いは許容され、品質判定はlogitsやtoken列の完全一致ではなく、公式FP8基準のbenchmark scoreの99%以上。generic probeの不一致だけで候補を失格にしない。投機ではtargetによる検証を維持し、計算の省略・近似を入れない。通常モードで構成探索を続け、決定的推論の2本は診断結果としてのみ扱う。

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

実機554 requests、0 errors、joint SLA fraction 1.0、OSL mismatch 0、ローカルparser valid。後続の同条件反復は897.824で、初回との差は0.62%。実測FPMのthroughput予測は実機より13.185%高く、汎用op modelの1.129%より誤差が大きかった。forward単体の精度改善をserving全体の精度改善と同一視しない。CPU overhead、queue/overlap、有限requestと定時間測定の差が未モデル化で、TTFTも過小予測している。

### 追加の実機観測

|条件（全16 GPU、旧image）|Actions|good tokens/s|SLA fraction|平均TTFT / ITL ms|
|---|---|---:|---:|---:|
|P2TP4/D1TP8 c16 target-only 反復|36229736223|897.824|1.0|別途raw参照|
|P2TP4/D2TP4 c16 target-only|36228974704|972.042|1.0|381.866 / 14.445|
|P2TP4/D2TP4 c32 target-only|36228995654|1592.140|1.0|636.676 / 16.775|
|P2TP4/D2TP4 c16 EAGLE3|36229580995|1489.537|1.0|545.586 / 7.986|

target-only c16/c32のActions・PBS正常終了とrawを確認。c32のローカルparserを再実行して一致した。c16のPBS walltime11分34秒、c32は11分45秒。P4→D8 baseline反復はActions成功、初回との差0.62%。P4→D4 c32反復 `36230301903` は1595.780（差0.23%）、c40 `36230331081` は1839.777、SLA0.990418。両方Actions正常終了、home checksumとrawのparser再実行を確認した。c40/c48の追加CPU探索は実測範囲内で2230.4/2390.3 predicted good tokens/s、SLA fraction1.0/0.975。rawは `qwen16-measured-search-high-concurrency` に保管。servingを過大予測する傾向があるため実機SLAで判定する。

`collect.py` はspeculation algorithm/steps/draft length、draft weight/runtime-config hashes、deterministic inference modeを比較キーへ含める。初版のように同じTP/EPのtarget-onlyと投機設定を同一条件へまとめない。archive33件の中間inventory（`inventory-a0c9726.json`）ではASPIRE15件中、通常測定として来歴の揃う候補は回収済みbaseline1件のみだった。このsnapshotは上記追加runのarchive完了前であり、最新集計ではない。

有限request数の感度もCPUで確認した。P4/D4の実測FPM設定を固定し320→1280 requestsだけ変えると、c16/c32/c40は1166.2/2066.0/2413.7 predicted good tokens/sとなり、320件の1147.4/1960.2/2230.4より高い。c32のsimulation durationは123.9秒となるが、実機1592–1596との差は縮まらない。少ないrequest数だけでは過大予測を説明できない。60秒warmup/120秒profile/30秒drainや入力分布を再現した比較ではなく、CPU sensitivityとして `qwen16-request-count-sensitivity` のconfig/log/per-request出力を保存する。

### 通常モードの上限側（2026-09-26、回収中）

|条件、旧image・P2TP4/D2TP4|Actions|good tokens/s|joint SLA fraction|確認状態|
|---|---|---:|---:|---|
|target-only c40 反復|36232258544|1839.054|0.991282|PBS Exit0、回収中|
|target-only c48|36231906794|1945.962|0.954078|PBS Exit0、raw parser再実行一致、Actions監視失敗|
|EAGLE3 c32|36232547785|2218.384|1.0|PBS Exit0、回収中|
|EAGLE3 c48|36232941281|2305.296|0.929366|PBS Exit0、回収中|
|EAGLE3 c48 反復|36233305728|2294.650|0.927963|PBS Exit0、回収中|

c48 EAGLE3の反復差は約0.5%。ただしc32からの伸びは約4%で、TTFTによりjoint SLAの余裕が小さい。中間c40を `66f3ca6` / `research/aspire-eagle-c40`（Actions `36234985587`、同条件反復 `36235158063`）で追加した。concurrencyだけを変更し、image・draft・配置・通常推論・公式入力は固定する。

target-only c48のActions失敗は計算失敗ではない。監視checkerの単発180秒timeoutが続き、連続接続失敗300秒の上限でexit124になった。PBSは後にExit0、runnerのraw回収とhome保存も完了した。archive内accountingはR状態の古いsnapshotなので、別途取得したPBS履歴をterminal根拠とし、原archiveは改変しない。

### ブラックボックス探索の適用判断

ユーザー提案のOptunaも候補とする。現在は同一image・P4/D4・通常推論でconcurrencyだけを32→40→48と増やしてSLA境界を確認しているため、この小さな一次元探索に追加依存は導入しない。c32反復は1595.780 good tokens/s（初回1592.140、差0.23%）。c40は1839.777、joint SLA fraction 1137/1148=0.990418、平均TTFT/ITL 745.613/17.566 ms。両PBSはExit0、Actions・home archive・raw parser確認も完了。c48は `d133c40` / `36231906794` で投入した。

draft length・concurrency・chunk size等の相互作用を探索する段階では、[Optuna ask/tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html)で候補提案とActions実行を分離できる。目的値は実機のgood tokens/s、制約はjoint SLA fraction >=0.9とし、image・backend・決定的推論モード・workloadが異なる観測は同じstudyへ混ぜない。候補ごとにcommitを作成し、terminal状態とraw/parser確認後にtellする。通信失敗・未実行・OOMを性能0として学習させずFAILとし、予測値も実測trialへ混ぜない。出力正当性は独立した採用条件とする。seedとOptuna versionを固定し、最良候補は別割当で反復する。[TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)の初期ランダム試行数もGPU予算に含める。現時点でOptunaによる実機探索を実行したとは扱わない。

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

Nebius draftは572,538,880 BF16 parameters、weights 1,145,943,008 bytes（integer token mapsを含む）。NGRAM run `36227653702` の最初の致命的エラーはPD warmupの `ngram_worker.py:242`、`spec_info=None` による `accept_tokens` 属性参照であり、後続SIGKILLをOOMと解釈しない。`ed4c823` はPD初期NGRAM状態を構築する17行のsource patchを所有側build contextに追加し、新規imageで再実行する。既にoutput_idsにあるbonus tokenを重複させないゼロ初期accept lengthsをCPU torch回帰検証済み。target verificationは変更していない。後述の新image runでGPU E2E検証も完了した。

旧mixed-TP EAGLE3 `36227719367` / PBS212995はwarmup中のNIXL remote disconnect→NOT_FOUND→転送timeoutで失敗（Exit1、16分58秒）。home archiveをchecksum確認済み、性能値なし。equal-TP EAGLE3 `32c8539` / `36229580995` / PBS213006はExit0、11分46秒、904 requests/0 errors。decode workerごとの平均受理長は2.226/2.219（bonus tokenを含み、warmup等も含む累積値）。

ただしpost-benchmarkの8 generic promptsでtarget-only c16対照との出力差が5問にあった。同じTP/EP・imageのtarget-only c16とc32間でも3問に差異があり、投機固有の誤りとは未確定。SGLangのgreedy verifyがtarget logitsのargmaxを照合することはsourceで確認したが、bitwise parityを達成したとは報告しない。既存の `--enable-deterministic-inference` による対比較（EAGLE3 `8c5d670` / `36231207022`、target-only `7bdc927` / `36231225267`）で切り分ける。主催者のfull accuracy検査は依然として別の検証である。

上記診断は両方のgeneric probeで8/8問・512 tokens全て一致した。決定的推論のEAGLE3は769.837 good tokens/s、SLA0.983437、483 requests、0 errors、PBS213014 Exit0。一方target-onlyは314 requests/0 errorsでもSLA0.0のためparserが不合格とし、PBS213015 Exit1。通常モードの性能候補へこの遅い設定を強制しない。通常モードEAGLE3 c32は `f4a83e0` / `36232547785` で投入済み。

修正版NGRAM imageはAWS build26分20秒で成功、SHA256 `994d664d6771af8f722c95d7c45c0c0cd362323c92f4b927da20e2e88aace562`、12,225,540,096 bytes。torch2.11+cu130/NIXL1.4.1/AIPerf0.12.0のCPU smokeを確認。Firmus配布SSH失敗はcontinue-on-errorでASPIRE配布・実行を妨げず、再buildなしで継続した。新image target-only対照は `499d417` / `36230584981`。

修正版NGRAM `36228809354` / PBS213009はExit0（11分43秒）、998.081 good tokens/s、612 requests、joint SLA fraction1.0、平均受理長1.589。新image target-only `36230584981` / PBS213012もExit0（11分37秒）、903.940 good tokens/s、555 requests、SLA1.0。両者0 errors、同一image/TP/EP/c16の一回ずつの比較でNGRAMは10.4%高い。両Actions正常終了、home archiveのchecksumとraw parser再実行を確認済み。反復結果ではない。

NGRAMのgeneric probeは新image対照と7/8問一致し、prompt5（stack/queue説明）の最初のtokenから差があった。旧image対照反復とは8/8問・512 tokens全て一致する。旧image対照2回も全問一致、新旧imageの通常推論間では同じprompt5が異なる。この観測だけで数値誤差や投機の不具合と断定しない。比較JSONはhomeの `research-issue-16/20260926` に保存した。

## 実行経路の失敗

`cbd5b7e`/`fd1bbff` は既存profiler staging文字列契約でCPU validation失敗、GPU投入なし。`6b80944`で修正、145 testsと静的検査を通過。empty commit `26372b4` はpaths filterにより未起動なので対照runに数えない。

旧chunk run `36226700537` と旧baseline `36226778215` は監視SSH timeout後のworkflow cleanupで中断（PBS Exit_status143）。旧NGRAM `36226824669` は待機中に同じ失敗となり公式測定なし。chunk runのdecode workerだけはfull23条件＋chunk53条件が回収済みであり、全run完了とは扱わない。

`51b620c` はobserved remote shellの約78–110秒を根拠に1回の監視期限を60→180秒へ変更した。過去commitの実行には遡及しない。strict terminal/artifact判定と接続失敗時の有限retryは維持する。
