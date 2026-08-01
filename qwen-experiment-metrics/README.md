# qwen-experiment-metrics

Qwen/SGLang実験のrun-level値を、BigQueryの`run`, `key`, `value`の3列へ
逐次送信する軽量collectorである。送信先は
`apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_metrics`。

認証JSONは`EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_JSON`、PJMへ搬送するときは
`EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_BASE64`から読み、ファイルへ保存しない。
`EXPERIMENT_METRICS_SINK=stdout`を指定するとBigQueryへ送らずJSON Linesを出力する。

SGLangのprivate状態や重複した実効設定は保存しない。具体的には
`sglang.runtime.*`と
`model.config.quantization_config.modules_to_not_convert.*`を送信対象外とする。
