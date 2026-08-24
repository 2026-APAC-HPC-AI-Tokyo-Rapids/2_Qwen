# qwen-experiment-metrics v2

各producerは取得完了時に診断用JSONL fragmentをatomic renameで確定し、同じ
`run`, `key`, `value`行をBigQueryへ直ちに送信します。500行chunk、`insertId`
なし、retryなし、重複許容です。送信失敗はwarningとなり、benchmarkやPJMの
結果を変更しません。

現在のQwen実行経路で使うrun/stage lifecycle、provenance、hardware/topology、
workload、GPU sampler・profiler、SGLang request/token/configurationだけを扱います。
fragmentは診断artifactであり再送queueではありません。将来用のnetwork測定、
submission/cache管理、中央merge、physical projection、consumer viewは実装しません。

計算ノードは共有run directoryへfragmentを書きます。BigQuery資格情報はscheduler
jobへ渡さず、terminal状態を確認したlogin runnerだけがfragmentを送信します。
