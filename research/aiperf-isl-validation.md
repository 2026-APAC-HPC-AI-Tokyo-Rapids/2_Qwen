# AIPerf 0.12.0 の入力長の公式判定

公式コマンドは `--synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --cache-bust first_turn_prefix` を指定し、`--apply-chat-template` を指定しない。このコマンド自体は変更しない。

Firmus job 16918（Actions 34686971404）の探索・profile実行では、入力長設定4000に対してexportの `input_sequence_length.avg` は3986、出力長は200だった。このrunは公式スコアの証拠には使わず、clientの入力長計測の確認に使う。

AIPerf 0.12.0、commit `be53bf2953d30e46c500e6a80fc1f8b6f84bc718` の実装を調べると、次の処理が対応する。

- [`estimate_marker_token_cost`](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/timing/strategies/cache_bust.py) は固定した8個のmarkerをtokenizeし、平均を整数に丸める。
- [`_init_isl_compensation`](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/dataset/composer/base.py) はfirst-turn prefixの費用をsynthetic ISLから引く予算にする。chat-template補正は指定時のみ有効なので、公式コマンドでは0となる。
- [`_generate_text_payloads`](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/dataset/composer/synthetic.py) はその予算を入力長設定から差し引いてpromptを生成する。

固定tokenizer `Qwen/Qwen3-235B-A22B-Instruct-2507` のrevision `ac9c66cc9b46af7306746a9250f23d47083d689e` で同じmarkerをtokenizeした結果は `16, 13, 13, 12, 14, 16, 14, 15`。AIPerf 0.12.0 では、合成 ISL の設定値 `4000` に対して `first_turn_prefix` が四捨五入された `14` トークンの減算推定値を適用する。しかし、プレフィックス内のマーカーが実際にトークン化されると `12`〜`16` トークンの幅が生じるため、クライアント側で観測されるリクエストごとの有効な ISL は `3984`〜`3988` の範囲に変動する。

単一の固定値（`4000` または `3986`）のみを受容する前提は実態に即していない。過去の観測結果との後方互換性として `4000` および `3986` の完全一致は引き続き許容されるものの、正式な判定ではリクエストごとの `min` および `max` が `3984`〜`3988` の許容範囲内に収まっていることが求められる。分布の根拠データが存在しない近傍の非整数平均値は無効として拒否する。

実際のスコア証拠である Firmus Slurm ジョブ `17003` のエクスポートデータでは、リクエスト総数 `555` 件に対し、`min` が `3986`、`max` が `3987`、`sum` が `2212231`、`average` が `3986.001801801802` となる実績値が確認された。パーサーはこれら `min`、`max`、`count`、`sum` を保持・検証し、`count == request_count`、整数 `sum`、および `average == sum / count` の一致を確認する。公式判定時に保存されたargv全体、AIPerf version、tokenizer revision、mode、concurrency、gatewayも固定値と照合する。`good_request_fraction >= 0.90` の条件は維持する。

Firmus Slurm `17005` ではリクエスト数 `558`、エラー数 `0`、good fraction `1.0`、goodput `4.536230620411887` req/s、暫定スコア `907.2461240823774` output tokens/s を記録した。復旧された集計エクスポートにおける観測 OSL は `min` `200`、`max` `201`、`count` `558`、`sum` `111601`、`average` `200.00179211469535`、`osl_mismatch_count` `0` であった。完全一致の平均値を前提とするバリデータでは一度却下されたが、ソースコードの精査により、この集計分布は固定された `AIPerf` の許容差を満たしていることが確認された。現在、バリデータは生の集計分布と不一致カウントを個別に検証する仕様となっている。

repositoryとstandalone sbatchのparserは同一内容を使い、回帰テストで一致を確認する。`4000`/`3986`の受理、実測分布の受理、範囲外または整合しない分布の拒否、argv改変・追加、来歴不一致・欠落の拒否を双方で検証する。
