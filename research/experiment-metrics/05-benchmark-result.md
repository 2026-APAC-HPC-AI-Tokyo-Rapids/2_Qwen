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

## 出力の変化検出

throughputだけでは、速くても異なる出力を生成したrunを区別できない。
prompt本文・生成本文をBigQueryへ保存せず、SGLang instrumentationで次を生成する。

```text
result.output.request_count
result.output
result.output.manifest_sha256
```

可能なら生成token ID列をcanonical byte encodingしてhashする。backendがtoken IDを
返さない場合はUTF-8生成textをhashし、keyを`text_sha256`として意味を偽らない。
`result.output`はrequest順を維持したhashとtoken countのlistをcompact JSON配列として
1行に保存する。manifestはrequest indexと個別hashを固定順で連結してSHA-256を取る。

hash差をaccuracy failureとは判定しない。temperature 0と固定seedでも、kernelや
floating-point reductionの差で出力が変わる可能性がある。BigQuery側で同条件runを
比較するための観測値として保存する。

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
