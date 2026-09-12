# SGLang 実効設定の収集

## コマンドライン引数のみでは不十分な理由

SGLang の `offline_throughput` ベンチマークは、内部で `BenchArgs` および `ServerArgs` を Python の dataclass として生成・初期化する。特に `ServerArgs.__post_init__()` は、実行対象デバイス、GPU メモリ容量、モデルアーキテクチャ、アテンションバックエンド、ページサイズ、並列化戦略、キャッシュ設定などのデフォルト値を動的に解決・上書きする。

Qwen 3.5 を実行する場合にも、モデルアーキテクチャおよび GPU のハードウェア能力に応じた自動オーバーライドが働く。したがって、ユーザーが `--attention-backend` をコマンドラインで明示しなかったという事実のみでは、実行時に実際にどのバックエンドが選定されたのかを記録・特定することができない。

## 記録する3層の構造

### 1. 実際の起動引数列（実 argv）

`run.command` に、Singularity 起動コマンドを含む完全な argv のシェル文字列表現を保存する。これは、ユーザーまたはワークフローが明示的に何を指示したかを示す正本となる。

### 2. ベンチマーク引数（BenchArgs）

`dataclasses.asdict(bench_args)` を以下のキー階層へ展開する。

```text
sglang.benchmark_arg.backend
sglang.benchmark_arg.dataset_name
sglang.benchmark_arg.num_prompts
sglang.benchmark_arg.sharegpt_output_len
sglang.benchmark_arg.seed
...
```

CLI で指定を省略したデフォルト値も含め、確定した全引数を網羅する。

### 3. 実効サーバー引数（ServerArgs）

Engine 初期化直前の段階で確定・解決された `ServerArgs` を以下のキー階層へ展開する。

```text
sglang.server_arg.<field>
sglang.server_arg.<field>.<nested-key>
```

なお、`backend.get_server_info()` は `ServerArgs` の重複コピーや SGLang 内部のプライベートな状態変数（`internal_states`）を膨大に含むため保存しない。また、`sglang.runtime.*` という動的プレフィックスは生成しない。

## 辞書（dict）および配列（list）の展開規則

汎用フラットナー（generic flattener）は、以下のルールに厳格に従って構造化データをフラット化する。

- dataclass はフィールド名をキーとする辞書（dict）へ変換する。
- 辞書（dict）はキーをドット区切りの1階層として再帰的に展開する。
- リストやタプル（list/tuple）は、順序性を持つ設定値に限定して個別行へ展開せず、当該フィールドのコンパクトな JSON 配列文字列として単一のスカラー値に格納する。成果物ファイルやログの本文をこの方式で格納してはならない。
- `None` は空文字列とする。
- 真偽値（bool）は `true` / `false`、有限小数は正規化された10進表記（canonical decimal）とする。
- 列挙型（Enum）はそのスカラー値そのものを用いる。
- 辞書全体を一括して JSON 文字列へシリアライズすることは行わず、キーごとに個別の行へ展開する。
- 空の辞書は空文字列、空のリスト／タプルは `[]` として記録する。
- 通常のプロパティ名として使用できない特殊な辞書キーは、UTF-8 バイト列を 16 進ダンプし、`encoded.<hex>` として可逆的に表現する。

例:

```python
{
    "cuda_graph_config": {
        "decode": {"backend": "full", "max_bs": 64, "bs": [1, 2, 4]},
        "prefill": {"bs": [4, 8, 12]},
    }
}
```

上記のデータ構造は、以下のように展開される。

```text
sglang.server_arg.cuda_graph_config.decode.backend = full
sglang.server_arg.cuda_graph_config.decode.max_bs = 64
sglang.server_arg.cuda_graph_config.decode.bs = [1,2,4]
sglang.server_arg.cuda_graph_config.prefill.bs = [4,8,12]
```

## ワークロード情報

リクエストごとの個別トークン数やその詳細な分布情報は BigQuery には保存しない。ベンチマーク完了時に SGLang の結果集計から得られる `result.token.input.total` および `result.token.output.total` の総数のみを記録する。

## モデル識別情報

`model.config.*` などのモデル詳細設定は保存しない。モデル名が特定できれば Hugging Face リポジトリ等の構成情報を追跡できるため、ホスト側の `MODEL_PATH` の末尾ディレクトリ名を以下の単一キーにのみ記録する。

```text
model.name
```

## プロセス間通信（トランスポート）

ホスト側の `stage_runner` があらかじめ Unix ドメインソケットをリッスンした状態で Singularity コンテナを起動する。SGLang 側へ適用した計測パッチは、以下のイベントを長さプレフィックス付き（length-prefixed）JSON としてソケット経由でホスト側へ送信する。

```text
sglang.configuration
sglang.benchmark_result
```

コレクターはイベントを受信するごとにインメモリでフラット化および BigQuery 送信を行い、送信成功のレスポンスを受領した直後にメモリから安全に破棄する。ソケット通信の切断やイベントの欠損が発生した場合でも、ベンチマーク本体の推論処理はそのまま継続する。BigQuery の認証情報をコンテナ内部へ持ち込む必要はなく、クラウドへのメトリクス送信はホスト側のコレクターのみが一元的に担当する。

実装時のパッチは upstream 側の推論ロジックには一切手を加えず、設定情報および結果イベントを読み取るフック処理のみに限定する。SGLang のコミットを更新する際は、パッチの適用可能性およびイベントスキーマの単体テストを必須の検証ステップとする。

## セキュリティに関する設計方針

すでに合意された「環境変数を全件保存する」方針と同様に、`ServerArgs` に関してもフィールド単位の秘匿フィルタリングは原則として設けない。将来的に `api_key` や `admin_api_key` などの機微な引数を指定した場合は、そのまま保存対象となる。この取り扱い方針を変更する場合は、暗黙の場当たり的なフィルターを追加するのではなく、設計ドキュメントおよびキー契約を改訂した上で明示的に実施する。
