# 実行識別・ソース・入力成果物

## 実行名（run 名）の命名規則

GitHub Actions 経由でジョブを投入する標準経路においては、以下の命名規則に固定する。

```text
qwen-<GITHUB_RUN_ID>-<GITHUB_RUN_ATTEMPT>-<PJM_JOBID>
```

手動でジョブを投入する場合は、先行の OpenFOAM プロジェクトと同様に、PJM ジョブ ID と UTC 開始時刻から生成する。

```text
qwen-manual-<PJM_JOBID>-<YYYYMMDDTHHMMSSffffffZ>
```

当時のワークフロー定義では `GITHUB_RUN_ATTEMPT` および `GITHUB_REPOSITORY` が PJM スクリプト側へ渡されていなかったため、本メトリクス基盤の実装時に環境変数渡しを追加する。

## 実行ライフサイクル（Run Lifecycle）

| キー | 論理型 | 取得元 |
|---|---|---|
| `run.started_at` | timestamp | コレクターの UTC ウォールクロック |
| `run.finished_at` | timestamp | コレクターの UTC ウォールクロック |
| `run.elapsed_sec` | float | PJM スクリプトのモノトニッククロック |
| `run.outcome` | enum | スーパーバイザー判定 |
| `run.exit_code` | int | ベンチマークパイプラインの終了コード |
| `run.signal` | int | シグナル終了時の受信シグナル番号 |
| `run.command` | string | 実際の argv を `shlex.join()` で結合したシェル文字列表現 |

`run.outcome` には、`succeeded`、`failed`、`cancelled` の3値のみを使用する。PJM スクリプトが構築した引数配列をコレクターへ直接渡し、空白やクォートを含む複雑なコマンドライン引数も忠実に復元できるよう、`shlex.join()` を用いて単一のシェル実行文字列表現へ変換する。

## ソフトウェアソースとバージョン情報

リポジトリ側のコミット SHA は環境変数 `env.GITHUB_SHA` を正本とし、別名キーへの無用な複製は行わない。
コンテナ（SIF）内部の SGLang はリポジトリ側のコミットとは独立して管理されるため、以下の項目を個別に記録する。

| キー | 取得元 |
|---|---|
| `software.sglang.commit` | SIF 内の `/opt/sglang/sglang-commit.txt` |
| `software.sglang.version` | SGLang の `__version__` |
| `software.python.version` | ベンチマークプロセスの Python バージョン |
| `software.torch.version` | ベンチマークプロセスの PyTorch バージョン |
| `software.cuda.runtime_version` | PyTorch ランタイムが認識する CUDA バージョン |
| `software.cuda.driver_version` | NVIDIA ドライバクエリが返すバージョン |
| `software.singularity.version` | ホスト上の `singularity --version` 出力 |

インストール済みパッケージを無差別に全件展開すると、1実行あたりの行数肥大化や不要なバージョン差分ノイズを招くため、推論性能に直接影響を与える主要パッケージに限定して個別キーとして記録する。コンテナイメージ（SIF）は、ビルド元となった GitHub Actions ワークフローの実行番号、SGLang コミット、および主要パッケージのバージョンによって一意に識別する。

## 入力データと成果物（Artifact）

| キー | 取得元 |
|---|---|
| `artifact.sif.source_workflow_run_number` | `sif-source-run.txt` |
| `artifact.sif.size_bytes` | ファイルの `stat`（サイズバイト数） |
| `artifact.dataset.size_bytes` | ファイルの `stat`（サイズバイト数） |

成果物のファイルハッシュを実験条件の代理識別子として用いることはしない。ランタイムスクリプトは `env.GITHUB_SHA` および `run.command`、SIF は生成元ワークフロー実行番号、モデルおよびデータセットは SGLang の実効引数とワークフロー環境変数によってそれぞれ識別する。ファイルサイズ（`size_bytes`）は、成果物の存在性および転送完了状態を診断するための補助情報として記録する。
