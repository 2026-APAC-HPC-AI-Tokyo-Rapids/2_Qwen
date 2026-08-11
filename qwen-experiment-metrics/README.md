# qwen-experiment-metrics v2

各producerは取得完了時に診断用JSONL fragmentをatomic renameで確定し、同じ
`run`, `key`, `value`行をBigQueryへ直ちに送信します。500行chunk、`insertId`
なし、retryなし、重複許容です。送信失敗はwarningとなり、benchmarkやPJMの
結果を変更しません。

GPU sampler、GPU profiler/reducer、SGLang request/token/stage/configuration、
IMB-MPI1、NCCL testsの収集は維持します。NCCL/IMBはgeneration後の独立stageで、
generation性能値には含めません。fragmentは診断artifactであり再送queueでは
ありません。中央merge、`metrics.jsonl`、physical projection、consumer view、
conflict tableはありません。

workflowはrun内にmode 0600のcredential fileを作り、PJMへはpathだけを渡します。
credential transport変数はenvironment snapshotとartifactから除外し、cleanupで
fileを削除します。高コスト処理前にrunner/compute双方からsmoke行を送ります。
