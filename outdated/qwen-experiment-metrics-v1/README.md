# qwen-experiment-metrics

Qwen/SGLang実験のrun-level値を、BigQueryの`run`, `key`, `value`の3列へ
逐次送信する軽量collectorである。送信先は
`apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_metrics`。

認証JSONは`EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_JSON`、PJMへ搬送するときは
`EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_BASE64`から読み、ファイルへ保存しない。
`EXPERIMENT_METRICS_SINK=stdout`を指定するとBigQueryへ送らずJSON Linesを出力する。

収集単位は検索・比較可能なscalarであり、各rowの`value`は必ず文字列scalarとする。
ログ本文、artifact本文、raw GPU sample、artifact hashをBigQueryへ保存しない。
dictはleafごとに展開し、順序を持つ設定配列だけは実値を失わないcompact JSON文字列を
1つのscalar valueとして保存する。

SGLangのprivate状態や重複した実効設定は保存しない。具体的には
`sglang.runtime.*`と`model.config.*`を送信対象外とし、モデルは`model.name`だけを
保存する。
