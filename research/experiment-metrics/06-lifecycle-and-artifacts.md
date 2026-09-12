# メトリクス取得・送信ライフサイクル

## プロビジョニング（Provisioning）

GitHub Actions ランナーは、ジョブ実行単位のランタイムバンドルへ以下の資材を追加・配置する。

```text
qwen-experiment-metrics/
qwen-experiment-metrics/.venv/
```


ランナーは GitHub リポジトリのシークレット `EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_JSON` を Base64 エンコードし、以下のパラメータ群とともに PJM スクリプトへ環境変数として引き渡す。

```text
GITHUB_RUN_ID
GITHUB_RUN_ATTEMPT
GITHUB_REPOSITORY
GITHUB_SHA
EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_BASE64
```

## PJM ジョブの起動処理

PJM スクリプトは、受け取った Base64 形式の認証情報をシェル変数へ退避させ、直ちに環境変数テーブルから元の変数を `unset` して不可視化する。メトリクス CLI は、コマンド呼び出しごとにコマンド限定スコープの環境変数から認証情報を受け取り、プロセス開始直後にメモリ上へキャプチャした上で速やかに `unset` を実行する。

起動時におけるメトリクスの送信順序は以下のように固定する。

```text
run-start（実行開始）バッチ送信
  → host platform / env（ホスト環境変数）バッチ送信
  → source / artifact identity（ソース・成果物識別）バッチ送信
  → static NVIDIA GPU（GPU 静的構成）バッチ送信
  → benchmark stage（ベンチマークステージ）開始
```

## ベンチマークステージ

ホスト側のステージランナー（`stage_runner`）は、単一のスーパーバイザープロセスとして以下のライフサイクルを一元管理する。

```text
Unix ドメインソケットの listen
  → NVIDIA GPU サンプラープロセスの起動
  → Singularity コンテナ内ベンチマークの起動
  → SGLang configuration イベントの受信と即時送信
  → SGLang result イベントの受信と即時送信
  → ベンチマークプロセスの終了待機（wait）
  → サンプラーの停止および GPU 集約メトリクスの送信
  → ステージウォールタイムおよび終了コードの送信
```

イベントのソケット受信処理と GPU のサンプリング処理は、別スレッドまたは非同期タスク（async task）を用いて並行動作させる。送信モジュールは同一バッチ内の順序性を保ち、複数スレッドから BigQuery クライアントを同時に呼び出さないよう直列化制御を行う。

Unix ドメインソケットは実行固有のランタイムディレクトリ内に作成し、ステージ終了時に確実に `unlink` して削除する。ソケット自体にメトリクスデータをディスク永続化することはしない。仮に SGLang プロセスがソケットへ接続しなかった場合であっても、ベンチマーク処理自体は中断せず最後まで実行を継続する。

## PJM ジョブの終了処理

PJM スクリプトの `EXIT` トラップは、可能な限り以下のクリーンアップ手順を確実に実行する。

```text
run-end（実行完了）バッチ送信
  → 従来の JIT キャッシュ正常時マージ
  → ノードローカル一時ディレクトリの削除
```

メトリクスの送信失敗を理由として、ベンチマーク本体の成功ステータスを失敗へ上書きすることはしない。逆に、JIT キャッシュのマージ失敗など本来のパイプライン障害をメトリクスコレクターが握りつぶして隠蔽してはならない。

## ランナー側の終了時処理

PJM ジョブが終了した後、共有ランタイムディレクトリを削除する直前に、GitHub Actions ランナー側の CLI から `pjstat --history` を実行してアカウンティング情報を取得し、スケジューラバッチを BigQuery へ送信する。この際の認証用クレデンシャル JSON は、ランナー環境から直接メモリへキャプチャする。

ジョブの成功・失敗・キャンセルのいずれのケースにおいても、既存のログ／結果成果物のアップロードとスケジューラ統計の回収を必ず試行する。メトリクス収集基盤の都合によって、実行中の Actions ワークフローや計算キュー内の PJM ジョブを不必要に停止させてはならない。

## バッチと重複

同時に確定した値は、以下のバッチ単位にまとめて BigQuery へ送信する。

```text
1. run start（実行開始）
2. host platform + env（ホストプラットフォーム・環境変数）
3. source + artifact identity（ソース・成果物識別）
4. static GPU（GPU 静的構成）
5. SGLang configuration（SGLang 実効設定）
6. benchmark result（ベンチマーク結果）
7. GPU runtime aggregate（GPU 実行時集約統計）
8. stage result（ステージ実行結果）
9. run end（実行完了）
10. scheduler accounting（スケジューラアカウンティング）
```

