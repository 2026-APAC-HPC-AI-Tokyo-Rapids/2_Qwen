# SGLang実効設定の収集

## なぜcommand lineだけでは不足するか

`offline_throughput`は`BenchArgs`と`ServerArgs`をdataclassとして生成する。
特に`ServerArgs.__post_init__()`は、device、GPU memory、model architecture、
attention backend、page size、parallelism、cacheなどの既定値を解決・変更する。

Qwen 3.5向けにもmodel architectureとGPU capabilityに応じたoverrideがある。
したがって、`--attention-backend`を指定しなかったという事実だけでは、実際に何が
選ばれたかを記録できない。

## 保存する3層

### 実argv

`run.command`にSingularityを含む完全なargvのshell表現を保存する。これはユーザーが何を
指定したかの正本である。

### benchmark引数

`dataclasses.asdict(bench_args)`を次へ展開する。

```text
sglang.benchmark_arg.backend
sglang.benchmark_arg.dataset_name
sglang.benchmark_arg.num_prompts
sglang.benchmark_arg.sharegpt_output_len
sglang.benchmark_arg.seed
...
```

CLIで省略したdefaultも含める。

### 実効server引数

Engine初期化前に解決済みの`ServerArgs`を次へ展開する。

```text
sglang.server_arg.<field>
sglang.server_arg.<field>.<nested-key>
```

`backend.get_server_info()`は`ServerArgs`の複製とSGLangのprivateな
`internal_states`を大量に含むため保存しない。`sglang.runtime.*`は生成しない。

## dict/listの展開規則

generic flattenerは次の規則にする。

- dataclassはfield名のdictへ変換する。
- dictはkeyを1階層として再帰展開する。
- list/tupleは順序を持つ設定値に限り展開せず、そのfieldのcompact JSON配列として
  1つの文字列scalarに保存する。artifactやログ本文をこの方式で格納しない。
- `None`は空文字にする。
- boolは`true`/`false`、有限数はcanonical decimalにする。
- Enumはそのscalar valueを使う。
- dict全体をJSON文字列にせず、keyごとに展開する。
- 空dictは空文字、空list/tupleは`[]`として保存する。
- 通常property名に使えないdict keyはUTF-8 byte列をhex化し、
  `encoded.<hex>`として可逆に表す。

例:

```python
{
    "cuda_graph_config": {
        "decode": {"backend": "full", "max_bs": 64, "bs": [1, 2, 4]},
        "prefill": {"bs": [4, 8, 12]},
    }
}
```

は次になる。

```text
sglang.server_arg.cuda_graph_config.decode.backend = full
sglang.server_arg.cuda_graph_config.decode.max_bs = 64
sglang.server_arg.cuda_graph_config.decode.bs = [1,2,4]
sglang.server_arg.cuda_graph_config.prefill.bs = [4,8,12]
```

## workload

requestごとのtoken数やその分布は保存しない。benchmark完了時にSGLang resultから得る
`result.token.input.total`と`result.token.output.total`だけを保存する。

## model

`model.config.*`は保存しない。モデル名からmodel repositoryの設定をたどれるため、
hostの`MODEL_PATH`末尾を次の1キーだけに保存する。

```text
model.name
```

## process間transport

host側`stage_runner`がUnix domain socketをlistenしてからSingularityを起動する。
SGLang patchは次のeventをlength-prefix付きJSONとして送る。

```text
sglang.configuration
sglang.benchmark_result
```

collectorはevent受信ごとにflatten・sendし、成功応答後にmemoryから破棄する。
socket切断やevent欠損でもbenchmarkは継続する。transport認証情報をcontainerへ渡す
必要はなく、BigQuery送信はhost側collectorだけが行う。

実装時のpatchはupstream側ロジックを書き換えず、情報を読み取るhookだけに限定する。
SGLang commit更新時にpatch適用とevent schema testを必須にする。

## security上の扱い

既決の「環境変数を全保存する」方針と同様、ServerArgsにもfield単位のfilterを設けない。
将来`api_key`や`admin_api_key`を設定すれば保存対象になる。この方針を変更する場合は、
暗黙のfilterを追加せず、設計文書とkey契約を先に改訂する。
