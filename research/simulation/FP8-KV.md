# EAGLE3 c40: BF16からFP8 KVへの変更

2026-09-27のユーザー依頼による実機比較。FP8実装・2回の計算完走・raw回収・再解析・cleanupを完了した。**KV割当は半減したが、この固定条件でgoodputの優位性は確認できなかった。** FP8版は `research/aspire-eagle-c40-fp8kv` の `095134dbea5f351cbfa09ba11db11b85ce9622f1` で再実行できる。性能上の優位性が未確認のため、従来の反復済みBF16採用候補を上書きしない。

## 条件と判定

- 親commitはEAGLE3 c40の `66f3ca6`。Qwen3-235B-A22B-Instruct-2507-FP8、ASPIRE 2ノード16 H100、P2×TP4/D2×TP4・EP1、concurrency40。
- Nebius EAGLE3 Instruct draft、3 steps/topk1/4 tokens、node-local model、max-total-tokens131072、既存image `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5` を固定。新規buildは不要。
- `kv_cache_dtype=auto`（実効BF16）から `fp8_e4m3` へ変更し、prefill/decode両workerへ同じ指定を渡した。全workerのserver-infoでdtype、FA3、TP4/EP1、token上限を確認。target/draftのsource、weight identity、worker配置も対照と一致した。
- 最終比較の物理ノードは全て `a2ap-dgx018` / `a2ap-dgx027`。同時実行数、KV token上限、attention backendは変更していない。通常benchにprofileは混ぜない。
- 公式AIPerfコマンドを固定し、規約に合わせたローカルparser `scripts/parse-aiperf-result.py` でrawを再解析した。このparserは主催者提供ではない。
- [規約追補Q1/Q4](https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/1.%20Qwen3%20AI%20Task%20Rules.md#q1-basic-vanilla)はFP8 KV保存・転送を許可し、INT8/INT4/FP4を禁止する。主催者の全件accuracy検査は未実施。

## 結果

|同一ノード対・EAGLE3 c40|Actions|good output tokens/s|joint SLA|平均TTFT ms|平均ITL ms|PBS|
|---|---|---:|---:|---:|---:|---|
|BF16対照|36298251187|2349.382|98.0299%|1477.490|9.143|213329、Exit0、12分1秒|
|FP8初回|36298738175|2325.146|98.4171%|1435.543|9.579|213331、Exit0、11分52秒|
|FP8反復|36299355333|2332.215|98.8260%|1451.689|9.537|213334、Exit0、11分54秒|

全て0 errors、OSL mismatch0、規約に合わせたローカルparser valid。FP8平均2328.681は今回BF16対照比-0.881%。FP8の2回の差は約0.30%。過去のBF16反復2346.714/2315.183も踏まえると、明確な速度向上はなく、小差から統計的に有意なgoodput劣化とも断定しない。平均ITLはFP8約9.558msで今回BF16比約4.5%増えているが、原因の特定は別のprofileが必要。

TTFT p95はBF16 2410.392ms、FP8 2329.910/2568.575ms。ITL p95はBF16 10.807ms、FP8 11.378/11.290ms。

target KVの実dtypeは `torch.float8_e4m3fn`、131072 tokens。GPUあたりK/V各1.47 GBで、BF16の各2.94 GBから半減した（容量値はSGLangログ表記）。draft KVもFP8。容量節約は実測済みだが、同時実行数やKV上限を固定した今回の測定ではgoodput改善につながらなかった。

## 精度と実装上の範囲

SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4` は `fp8_e4m3` に対応する。FA3のFP8 KV経路はqueryもKV dtypeへcastするため、容量・通信量だけでなく演算経路も変わる。変換コスト、カーネル効率、draft受理率を未計測のまま性能差の原因とは断定しない。

公式target revision `e156cb4efae43fbee1a1ab073f946a1377e6b969` のsafetensors indexにはk_scale/v_scale/kv_scale tensorがない。quantization/kv_cache.pyは未収録時にscale=1.0を設定する。今回は外部scaleファイルを与えず、標準動作を測定した。

補助的な8問・各64 tokensのgreedy probeでは、今回BF16対照は過去BF16と8/8問一致。FP8は今回BF16対照と3/8問、反復は4/8問一致し、FP8同士は5/8問一致した。通常推論・FP8変更に伴うtoken列差を、そのまま規約違反やaccuracy低下とは判定しない。逆にAIPerf完走だけでも全件accuracy合格とは言わない。採用には主催者FP8基準scoreの99%以上を満たす必要がある。

## 失敗と復旧の経緯

- `d694aef` / Actions36298132846 / PBS213328: 最初のFP8投入。dgx016のdecode両workerがEnrootの `/dev/infiniband/rdma_cm` mountで失敗。SGLang起動前、Exit1、性能値なし。raw回収・home checksum・cleanup済み。この失敗を性能0として比較しない。
- `e5d05cd` / Actions36298644714: FP8を対照と同じdgx018/027へ固定したが、既存testのselect文字列assertionが一致せずCPU validation失敗。GPU投入なし。
- `095134d`: assertionを今回の配置に合わせ、全145 tests・Ruff・actionlintを確認して再投入。ノード固定は観測された環境失敗を避ける実験条件であり、mainの一般配置方針にはしない。
- 06:18 UTCにASPIRE ControlMasterがBroken pipeとなり、再接続はPermission denied (password)、Actions側はHost key verification failedとなった。ユーザーがVPNを切断・再接続していたことを後に確認した。FP8初回は計算成功後の回収中断、反復は計算中の監視中断であり、両Actionsはfailureのまま。BF16対照のActions全体はsuccess。
- SSH復旧後、反復もPBS F/Exit0を確認。初回の既存部分archiveを改変せず、両FP8の完全rawを別のrecovered-after-sshへ保存した。大きなストリームが途中で切れたため、共有run directoryに作成したtarを8 MiB単位で回収し、元tarのSHA256一致を確認した。
- 各FP8 archiveの49ファイル全てについてhome側のSHA256検証を通過。raw export・固定command・provenanceからparserを再実行し、両score.jsonと完全一致した。追加したPBS terminal履歴は古いaccounting snapshotと区別して保管。
- 両FP8の確認済みrun directoryと回収用tarを削除し、qstatで当該ジョブの残留がないことを確認。他のジョブの停止やホスト鍵検証の無効化は行っていない。

## 正本

`ssh archive-host` の以下を参照する。

- BF16: `/srv/qwen-artifacts/36298251187/1`
- FP8初回: `/srv/qwen-artifacts/36298738175/1/recovered-after-ssh`
- FP8反復: `/srv/qwen-artifacts/36299355333/1/recovered-after-ssh`
- 比較の派生結果: `/srv/qwen-artifacts/research-fp8-kv/20260927/comparison.json`

設定と変更理由は各commit、実行経過はActions/PBSログ、rawは上記archiveを正本とする。保存先の元の部分archiveやActionsのfailureをsuccessへ読み替えない。
