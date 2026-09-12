# ベンチマーク性能・生成結果

## SGLang の構造化結果

当時の `offline_throughput` が内部生成する結果辞書（result dict）を正本とし、人間向けのコンソール出力テキストを正規表現等でパースすることは行わない。

| BigQuery キー | SGLang 結果フィールド |
|---|---|
| `result.benchmark.backend` | `backend` |
| `result.benchmark.successful_requests` | `successful_requests` |
| `perf.benchmark.generation.wall_time_sec` | `total_latency` |
| `result.token.input.total` | `total_input_tokens` |
| `result.token.output.total` | `total_output_tokens` |
| `perf.throughput.request_per_sec` | `request_throughput` |
| `perf.throughput.input_token_per_sec` | `input_throughput` |
| `perf.throughput.output_token_per_sec` | `output_throughput` |
| `perf.throughput.total_token_per_sec` | `total_throughput` |
| `perf.throughput.last_generation_token_per_sec` | `last_gen_throughput` |

`perf.benchmark.generation.wall_time_sec` は、`backend.generate()` の実際の推論実行区間のみを対象とする所要時間である。PJM 内でのベンチマークステージ全体の外側所要時間は `perf.stage.benchmark.wall_time_sec`、PJM ジョブ全体の経過時間は `scheduler.elapsed_sec` として記録し、それぞれの計測範囲を明確に区別する。

## 成否

```text
run.stage.benchmark.exit_code
run.stage.benchmark.signal
run.stage.benchmark.result_present
```

`results.jsonl` の欠落や JSON のパース失敗が発生した場合、それらを成功値やダミー値へ置き換えてはならない。想定された結果キーは空文字列として送信し、ログへ警告を出力する。ベンチマークプロセスの終了コードは、子プロセスからそのまま上位へ伝播させる。

なお、リクエストごとの個別出力テキストや詳細なトークン数は BigQuery へは保存しない。入出力トークン数は、`result.token.input.total` および `result.token.output.total` の実行全体集約値のみを記録する。

## 成果物（Artifact）

従来の運用で生成される以下の成果物ファイルはそのまま維持する。

```text
results/environment.log
results/throughput.log
results/results.jsonl
logs/<PJM_JOBID>.log
```

これらのファイルは BigQuery 送信のために副次的に作られるものではなく、開発者や利用者が実行結果を事後診断するための一次成果物である。各成果物についてはファイルパスおよびサイズバイト数のみを BigQuery へ記録し、ファイル本文は GitHub Actions の成果物（artifact）として保存・確認する。将来的に巨大なプロファイラトレースが追加された場合でも、その内容をスカラーメトリクスへ展開することはせず、成果物マニフェストへの登録のみにとどめる。
