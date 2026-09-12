# AIPerf 0.12.0 の入力長の公式判定

公式コマンドは `--synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 --cache-bust first_turn_prefix` を指定し、`--apply-chat-template` を指定しない。このコマンド自体は変更しない。

Firmus job 16918（Actions 34686971404）の探索・profile実行では、入力長設定4000に対してexportの `input_sequence_length.avg` は3986、出力長は200だった。このrunは公式スコアの証拠には使わず、clientの入力長計測の確認に使う。

AIPerf 0.12.0、commit `be53bf2953d30e46c500e6a80fc1f8b6f84bc718` の実装を調べると、次の処理が対応する。

- [`estimate_marker_token_cost`](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/timing/strategies/cache_bust.py) は固定した8個のmarkerをtokenizeし、平均を整数に丸める。
- [`_init_isl_compensation`](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/dataset/composer/base.py) はfirst-turn prefixの費用をsynthetic ISLから引く予算にする。chat-template補正は指定時のみ有効なので、公式コマンドでは0となる。
- [`_generate_text_payloads`](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/dataset/composer/synthetic.py) はその予算を入力長設定から差し引いてpromptを生成する。

固定tokenizer `Qwen/Qwen3-235B-A22B-Instruct-2507` のrevision `ac9c66cc9b46af7306746a9250f23d47083d689e` で同じmarkerをtokenizeした結果は `16, 13, 13, 12, 14, 16, 14, 15`。平均14.125を丸めた14を差し引くと `4000 - 14 = 3986` になり、job 16918のclient観測値と一致する。これはclientの観測値についての説明であり、serverが実際に処理した入力token数を4000と証明するものではない。

parserは任意の入力長差を許容しない。公式判定時に保存されたargv全体を検証し、追加引数、探索用の時間設定、別seed、別cache-bust設定などを拒否する。実行環境から記録するAIPerf versionとtokenizer cacheのrevisionも固定値と照合し、mode、concurrency、gatewayの一致を検証する。その条件を満たす場合に限り、従来の4000と上記の3986を厳密に受理する。出力長200および `good_request_fraction >= 0.90` の条件は維持する。

raw export、観測平均、`goodput × 200` の計算は変更しない。`score.json` の `official_validation` には判定の成否、入力長設定4000、出力長設定200、観測値との差、client/tokenizerの版を保存する。証拠不足や不一致ではparserは非zero終了し、失敗理由も同フィールドに残す。探索実行は従来どおりメトリクスを出力し、公式判定を付与しない。

repositoryとstandalone sbatchのparserは同一内容を使い、回帰テストで一致を確認する。4000/3986の受理、近接値の拒否、argv改変・追加、来歴不一致・欠落の拒否を双方で検証する。
