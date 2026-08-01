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
sglang.server_arg.<field>.<nested-key-or-index>
```

`backend.get_server_info()`は`ServerArgs`の複製とSGLangのprivateな
`internal_states`を大量に含むため保存しない。`sglang.runtime.*`は生成しない。

## dict/listのscalar展開規則

generic flattenerは次の規則にする。

- dataclassはfield名のdictへ変換する。
- dictはkeyを1階層として再帰展開する。
- list/tupleは`.0`, `.1`の0始まりindexへ展開する。
- `None`は空文字にする。
- boolは`true`/`false`、有限数はcanonical decimalにする。
- Enumはそのscalar valueを使う。
- 1つのcontainer全体をJSON文字列にして保存しない。
- 空dict/listは、そのfield自体を空文字で保存して「存在した空container」を表す。
- 通常property名に使えないdict keyはUTF-8 byte列をhex化し、
  `encoded.<hex>`として可逆に表す。

例:

```python
{
    "cuda_graph_config": {
        "decode": {"backend": "full", "max_bs": 64},
        "batch_sizes": [1, 2, 4],
    }
}
```

は次になる。

```text
sglang.server_arg.cuda_graph_config.decode.backend = full
sglang.server_arg.cuda_graph_config.decode.max_bs = 64
sglang.server_arg.cuda_graph_config.batch_sizes.0 = 1
sglang.server_arg.cuda_graph_config.batch_sizes.1 = 2
sglang.server_arg.cuda_graph_config.batch_sizes.2 = 4
```

## actual request dict

`offline_throughput`はrequestごとに`sampling_params` dictを作る。prompt本文や
requestごとのobjectそのものは保存せず、実際に構築された値を次へ縮約する。

```text
sglang.request_arg.temperature
sglang.request_arg.ignore_eos
sglang.request_arg.extra.<field>
sglang.workload.request_count
sglang.workload.input_tokens.total
sglang.workload.input_tokens.min
sglang.workload.input_tokens.max
sglang.workload.input_tokens.mean
sglang.workload.input_tokens.p50
sglang.workload.input_tokens.p95
sglang.workload.requested_output_tokens.total
sglang.workload.requested_output_tokens.min
sglang.workload.requested_output_tokens.max
sglang.workload.requested_output_tokens.mean
sglang.workload.requested_output_tokens.p50
sglang.workload.requested_output_tokens.p95
```

requestごとに異なる値を「代表値」へ潰さず、分布の集約であることが分かるkeyにする。

## model config

実際にSGLangが読む`/model/config.json`も同じflattenerで次へ展開する。

```text
model.config.model_type
model.config.architectures.0
model.config.hidden_size
model.config.num_hidden_layers
model.config.quantization_config.<field>
...
```

`model.config.quantization_config.modules_to_not_convert`はモデル内部のmodule名を
数百行へ展開する一方、実験比較に不要なので保存しない。

file全体を1つのJSON valueとして保存しない。tokenizerやgeneration configは別の正本
なので、必要になった時点で`model.tokenizer_config.*`,
`model.generation_config.*`として分離する。model configの値とServerArgsの値が
一致するかはcollectorでは照合しない。

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
