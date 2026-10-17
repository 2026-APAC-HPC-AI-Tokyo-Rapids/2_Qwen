# AIPerf入力とprefix cache（Issue #16）

2026-09-26。実配備AIPerf 0.12.0、source `be53bf2953d30e46c500e6a80fc1f8b6f84bc718`、tokenizer `Qwen/Qwen3-235B-A22B-Instruct-2507@ac9c66cc9b46af7306746a9250f23d47083d689e`。

## 入力の共有度

[生成器](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/dataset/generator/prompt.py)はShakespeare token corpusのランダムな開始位置から切り出す。同じ本文が途中に現れることと、先頭から同じtoken列を持つことは異なる。causal attentionのKVにはそれ以前の文脈も含まれるため、途中の共通部分を通常のprefix cacheで単純に再利用できない。

回収した `36230331081` の `inputs.json` は12,800件、異なる本文12,724件、重複の余分な件数76（0.594%）、最大多重度2。実際に使われたwarmupを含む1,692 requestsの本文は1,689種類だった。全データを同時に無制限に保持できるという楽観的な比較でも、最も似た他の入力との共通prefixはQwen chat headerを含め中央値4 tokens、p90は5 tokens。64 tokens以上の152件は上限64で打ち切った集計であり、実キャッシュヒット率ではない。chat header自体は3 tokens。

## 実機の再利用量

`analyze_prefix.py` はprofiling requestsの開始〜終了時刻から両端1秒を除いた区間を取り、SGTのprefill workerログにあるnew/cached tokenカウンタを集計する。SGLangのchunk継続は `PrefillAdder` がprefix分を0として集計する。時間が秒単位のbatchカウンタなので、リクエストごとの厳密なcache使用量や速度差そのものではない。

|通常推論・P4/D4|Actions|新規token合計|cached token合計|cached / (new + cached)|
|---|---|---:|---:|---:|
|c32|36228995654|3,833,609|2,957|0.0771%|
|c40|36230331081|4,524,442|3,506|0.0774%|

この再利用率からはprefix cacheによる大きなprefill計算削減は期待しにくい。cache bookkeepingや実行カーネルの差もあるため、0.077%をそのまま実測speedupとは呼ばない。cacheを無効化したGPU A/B試験は行っていない。

リクエスト内のautoregressive KV cacheは別に重要。94 layers × K/V × 4 KV heads × head_dim128 × BF16の2 bytes = 192,512 bytes/token。4000 tokensは770,048,000 bytes（約0.77 GB）の論理KVで、TP4では4 GPUへ分割される。cache容量とNIXL転送は引き続き最適化対象であり、「KV cache全体が不要」という結論ではない。

## `first_turn_prefix`の実装上の注意

当初「指定があるためwireに識別子が付く」と説明したが、実装と送信内容を確認して訂正した。

- [marker生成器](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/timing/strategies/cache_bust.py)は `[rid:...]` を作り、workerには注入処理がある。
- この版の通常のsynthetic/concurrency経路では、[ConversationSource.next](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/timing/conversation_source.py)がmarker未設定のsessionを返し、[request_rate strategy](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/timing/strategies/request_rate.py)もmarkerを付けずにturnを発行する。agentic replayのmarker設定とは異なる。
- `uv`で同じPyPI版0.12.0を起動し、ローカルCPU模擬HTTP endpointで4件の送信JSONを捕捉。synthetic4000/200、dataset12800、seed2026、`--cache-bust first_turn_prefix`でもmarkerは0/4件だった。関連する4 source filesのSHA256は上記commitのsourceと一致した。request-count4の短い診断であり、GPU性能・品質の測定ではない。
- GPU runのwire capture自体は取得していない。実機の短いprefix hitと、この同版・同経路の診断結果が整合している。入力長が約3986なのもmarker用の事前減算と整合するが、この数値だけを原因の証明には使わない。
- 公式指定のAIPerf commandと公開clientは変更していない。この挙動を利用する専用cache処理も追加しない。将来のclient版で変われば再測定する。

## 再実行と正本

```bash
uv run research/simulation/analyze_prefix.py RUN_ARCHIVE TOKENIZER_JSON OUTPUT_JSON
```

rawは元Actionsのhome archive。派生JSON、CPU wire capture（requests、command、source hash、client log、再現scriptのみ。模擬スコアは含めない）は `ssh archive-host` の `/srv/qwen-artifacts/research-issue-16/20260926/prefix-investigation` に保管する。
