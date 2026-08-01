# benchmark性能・生成結果

## SGLangの構造化結果

現行`offline_throughput`が生成するresult dictを正本とし、人間向けconsole outputは
parseしない。

| BigQuery key | SGLang result field |
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

`perf.benchmark.generation.wall_time_sec`は`backend.generate()`だけの時間である。
PJM stage全体は`perf.stage.benchmark.wall_time_sec`、PJM job全体は
`scheduler.elapsed_sec`とし、名前と範囲を分ける。

## 成否

```text
run.stage.benchmark.exit_code
run.stage.benchmark.signal
run.stage.benchmark.result_present
```

`results.jsonl`がないことやJSON parse失敗を成功値へ置き換えない。予定したresult keyを
空文字で送り、warningを出す。benchmark終了codeは元のprocessからそのまま伝播する。

requestごとの出力hashやtoken数は保存しない。入出力は
`result.token.input.total`と`result.token.output.total`のrun全体値だけを使う。

## artifact

従来の次の成果物は維持する。

```text
results/environment.log
results/throughput.log
results/results.jsonl
logs/<PJM_JOBID>.log
```

BigQuery transportのためにこれらを生成するのではない。利用者がrunを診断できる
一次成果物である。各artifactは少なくともpath、size、SHA-256を送る。
巨大なprofiler traceが将来追加された場合、内容をscalarへ展開せずmanifestにだけ載せる。
