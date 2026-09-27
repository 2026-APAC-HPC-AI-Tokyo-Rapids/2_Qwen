# 追加最適化キャンペーン

追跡: [Issue #17](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/17)。2026-09-27開始。実行・結果・採否はIssueコメントを参照する。開始しただけの実験を完了とは扱わない。

基準: `66f3ca6`、公式FP8 target、16 H100、P2TP4/D2TP4 EP1、BF16 KV、Nebius EAGLE3 3 steps/4 tokens、c40。FP8 KV比較は [FP8-KV.md](FP8-KV.md) に完了記録がある。新規backend比較はまずtarget-onlyで、投機による効果と分ける。

## 測定範囲

条件ごとのcommit/Actionsから、投機2/3と5/6、native TTFT profile、TP4/EP4、TP8/EP8、vLLM0.24.0、TensorRT-LLM1.3.0rc20、target-only profileを投入した。通常設定比較は同一ノード対018/027、profileは別の004/009。concurrency36/44は基準の投機3/4を固定して別条件として実施し、各点を反復する。投機長変更のrunとはまとめない。

## serving予測の測定窓

前回の320件のfinite request予測に対し、同じAISimulate wheel（upstream `992dddc7...` + 測定policyのみの既存patch、SHA256 `6d4d3cd5a071ec1bc7686c6c99199b8a7e097c3bfcd54a70d9c4a08af6ea8be5`）を用いた。

まず4000件を予測し、仮想180秒までに到着する件数を求め、その件数だけで再実行する。これによりdrain中の追加到着を止める。60〜180秒に到着したrequestを評価し、最大30秒のgraceを設け、最初の測定requestから最後の完了まででgoodputを割る。`serving_window.py` は測定途中終了・drain中の追加到着を拒否する。合成固定長・closed-loop暖機の近似であり、AIPerf完全再現とは呼ばない。

|target-only P2TP4/D2TP4|測定窓付き予測 good tokens/s|予測TTFT ms|予測ITL ms|過去実測 good tokens/s|
|---|---:|---:|---:|---:|
|c16|1161.021|190.361|12.753|972.042|
|c32|2056.977|199.686|14.274|1592.140 / 1595.780|
|c40|2427.230|197.601|15.192|1839.777 / 1839.054|

測定窓を近づけても過大予測は解消せず、c32/c40では約29%/32%残る。元のforward補間の精度とserving全体の精度は区別する。次にnative profileで段階別の差を調べる。サービス時間の補正を試す場合、校正runと独立評価runを分け、GPU forwardの実測FPMを上書きしない。

`summarize_serving_profile.py` はprofile専用runのPrometheus前後差分を集計する。差分はwarmupを含むため公式120秒だけの値ではない。native host timestampにはCPU・待ち・通信が含まれ、GPU kernel時間や非重複の加算可能な区間とは限らない。

## 経験的goodput補正の独立run評価

c16 `36228974704` だけから実測/予測比0.8372308628を決定し、c32 `36228995654`/`36230301903` とc40 `36230331081`/`36232258544` の4runへ適用した。評価runの値は係数の算出に使わない。これらは既存研究で観測済みのrunで、盲検評価ではない。

c32の補正後予測1722.165、誤差7.92〜8.17%。c40は2032.152、誤差10.46〜10.50%。未補正の約29〜32%から減った。一点で学習した経験的なgoodput補正としては改善したが、TTFT/ITL・queue・SLA予測を直したわけではない。native engineや実測forward FPMを変更せず、別の比較器 `calibrate_serving_goodput.py` として残す。別backend、投機、TP/EP配置へ適用する根拠はない。

## 最初のGPU結果とTTFT調査

5 steps/6 tokensのc40は `36300950254` / PBS213341、Exit0、2283.863 good tokens/s、SLA97.9777%、0 errors。初回値は既存3/4より低い。rawのarchive確認はIssueで追跡する。

c48 profile `36300973745` / PBS213343は1506requests・0errorsだがSLA85.9230%でparserの90%ゲートに不合格、Exit1。通常性能値としては採用しない。native metricsはprefill queue平均1075/469ms、prefill forward751/650ms、prefill transfer完了待ち309/268ms、decode開始queue約0.12ms。warmup・healthを含み得る。decode受信待ち約1.7秒にはprefill待ちが入り、ネットワーク所要時間とは呼ばない。profileのソース上の計測境界も確認した。

別ノード対004/009で通常c40対照、chunk4096、prefill power_of_two policyを投入し、各要因を分けて比較する。profileのスコアを通常対照に使わない。

## 実機profileを踏まえた追試と制御経路の修正

target-only c40の別profileも取得した。prefill queue、forward、転送完了待ち、decode admissionをnative histogramで分離できたが、EAGLE c48とは投機・concurrencyが異なる。各段階の差を投機だけの効果と断定しない。集計rawはIssue記載の保存先へ退避した。

prefill chunk4096とpower_of_twoを通常runで比較し、後者はTTFT悪化によりSLA不適格だった。router0.3.2のpower-of-two実装は周期的に取得したtoken loadを参照するため、2workerで更新間に負荷が偏る可能性がある。これは原因仮説であり、prefillだけround_robinにした追試を追加した。decode routingは変更しない。

TP8/EP8とc36反復では、SSH wrapperが有効なPBS pending応答を出しながらtar転送エラー2で終了し、既存cleanupがジョブを削除する失敗を観測した。`main` の `6820b27` はこの完全なpending応答がある場合だけ継続する。実際のterminal失敗と不完全応答は引き続き失敗とする。動作テスト6件と制御系全151件、Actions検証が通った。修正を適用した追試の完走はIssueで追跡する。

集計キーにはbackendとprefill routing policyを含め、別backendのnative worker起動情報を検証できないrunは通常比較に含めない。vLLMのNIXL依存は固定ソースの要求1.2.0へ修正して新規buildした。古い0.10.1指定のrunは互換性調査として区別し、サポートされた構成の代表値にはしない。

## 係数固定後の新規run評価

係数とc40予測は `ffb43b8`（2026-09-27 07:04:48 UTC）で固定済みである。その後に投入・完走した通常target-only run `36305644422`（010/021）を追加評価し、係数を再学習せず適用した。既知のserving構成に対する新しいrunの検証であり、新しいbackendや投機構成の外挿検証ではない。過去4runの分割評価とは別に報告し、物理ノード対も明記する。入力と出力はhome archiveの `research-issue-17/20260927/goodput-prospective{,-input}.json`、検証結果と追加runはIssue #17へ記録する。
