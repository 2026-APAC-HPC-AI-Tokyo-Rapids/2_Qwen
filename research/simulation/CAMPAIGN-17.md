# 追加最適化キャンペーン

追跡: [Issue #17](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/issues/17)。2026-09-27開始。実行・結果・採否はIssueコメントを参照する。開始しただけの実験を完了とは扱わない。

基準: `66f3ca6`、公式FP8 target、16 H100、P2TP4/D2TP4 EP1、BF16 KV、Nebius EAGLE3 3 steps/4 tokens、c40。FP8 KV比較は [FP8-KV.md](FP8-KV.md) に完了記録がある。新規backend比較はまずtarget-onlyで、投機による効果と分ける。

## 測定範囲

条件ごとのcommit/Actionsから、投機2/3と5/6、native TTFT profile、TP4/EP4、TP8/EP8、vLLM0.24.0、TensorRT-LLM1.3.0rc20、target-only profileを投入した。通常設定比較は同一ノード対018/027、profileは別の004/009。concurrency36/44と反復は投機比較結果に続けて行う。

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
