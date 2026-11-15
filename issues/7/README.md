# Issue 7: CI/CDワークフロー統合、ビルド再利用高速化、メトリクス送信境界分離、スクリプト規約刷新

Author: yayoimizuha
Created: 2026-08-18T05:36:30Z
State: closed

## Description

## 背景と目的
`2_Qwen` のシステムアーキテクチャを改善し、リポジトリ運用の一貫性、ジョブ実行制御の安全性、ビルド成果物再利用の高速化・安定性、セキュリティ、およびコード品質の向上を図ります。

1. **ワークフロー構成の統一とエントリポイント集約**:
   - 単一エントリポイントワークフロー（`qwen.yml`）へ push および手動実行（`workflow_dispatch`）を集約
   - `validate.yml` による軽量なローカル静的検査（pytest、Ruff、bash 構文検査、YAML 構文検査）の先行実行
   - push 時のコミットメッセージタグ（`[cluster:genkai]`、`[nodes:...]`、`[profile]`、`[require-build]`、`[skip ci]`）による実行制御
   - クラスタタグが明示されていない push では計算ジョブを起動しない安全設計
   - 手動実行時における一貫した入力インターフェースの提供
2. **SIF ビルド成果物の再利用判定の簡素化・高速化（First-parent History Walk）**:
   - コミットの第一親（`first-parent ancestry`）を遡り、最新の利用可能な SIF を探索するロジックへの刷新
   - `[require-build]` タグがない限り、最新ビルド成果物を無条件で再利用
   - SIF 内部ファイルの検査（`singularity exec`）、インベントリ比較（`inventory-sif-runtime.py`）、`gh run view` によるステップ結論判定などの過剰・推測的な検証を全廃
   - `sif-sidecar.json` の `sha256` および `size_bytes` に基づくシンプルな整合性検証
3. **実験メトリクス収集と BigQuery 送信の境界分離・安全性向上**:
   - 計算ノード（PJM ジョブ内）からの BigQuery 直接送信、およびサービスアカウント JSON キーの計算ノード受け渡しを廃止
   - 計算ノードの責務を共有ランタイムディレクトリへの断片（fragment）出力のみに限定
   - PJM ジョブ完了後、ログインノード上のランナー（Login Runner）側で断片を回収し、BigQuery および GitHub アーティファクトへ一括送信
4. **スクリプトコーディング規約の刷新と不要資産の削除**:
   - スクリプト内のすべてのコマンド呼び出しを行継続（`\`）なしの1物理行に統一
   - 中継用環境変数（中継用 prefix/suffix 変数）の全廃
   - Python ツールの実行を `uv run` / `uvx` へ統一
   - `outdated/` 配下の不要なレガシー資産・ログ・旧 v1 モジュール等のクリーンアップ
5. **AGENTS.md およびドキュメントの共通化・日本語化**:
   - 最新の運用ルールを `2_Qwen/AGENTS.md` に反映
   - ワークフロー定義やドキュメントにおける人間向け文言の日本語化

---

## 具体的な実装方針

### 1. GitHub Actions ワークフロー構成の統一
- **`.github/workflows/qwen.yml` の新設（エントリポイント）**:
  - `push` および `workflow_dispatch` を受け付ける。
  - push 時の動作:
    - `paths-ignore`: ドキュメントやリサーチ資料など、計算に影響しないパスの変更を除外。
    - `[cluster:genkai]` を含む場合のみ Genkai ジョブを実行（クラスタタグ未指定時は validate のみ実行して終了）。
    - `[nodes:1]`、`[nodes:2]`、`[nodes:4]` によりノード数を判定（デフォルト: 2ノード）。
    - `[profile]` でプロファイリング実行へ切り替え。
    - `[require-build]` で SIF 再ビルドを強制。
    - `[skip ci]` で Actions 実行自体をスキップ。
  - 手動実行時（`workflow_dispatch`）の入力項目:
    - `run_genkai`（boolean, デフォルト: true）
    - `node_count`（choice: 1, 2, 4）
    - `profile`（boolean, デフォルト: false）
    - `require_build`（boolean, デフォルト: false）
- **`.github/workflows/validate.yml` の新設**:
  - `workflow_call` として定義。
  - ローカル環境で実行可能な検証（`pytest qwen-experiment-metrics/tests`、`ruff check`、`bash -n scripts/*.sh scripts/*.pjm`、YAML パース等）を先行実行。
- **`.github/workflows/genkai.yml` のリファクタリング**:
  - `workflow_call` を受け付ける再利用可能ワークフロー（reusable workflow）に変更。
  - 入力引数（inputs）として `profile`、`node_count`、`require_build` を受領。
  - concurrency group にクラスタ名を含め（例: `genkai-qwen-bench`）、不要な直列化や他クラスタへの干渉を防止。

### 2. SIF ビルド成果物の再利用判定の簡素化・高速化
- **First-parent History Walk**:
  - コミットの第一親（`first-parent ancestry`）を `git rev-parse "${commit}^"` で遡り、最新の利用可能な SIF アーカイブおよびメタデータを探索。
  - `sif-sidecar.json` の `sha256` / `size_bytes` と実ファイルの比較のみにより、高速かつ確実に判定。
- **不要処理の削除**:
  - `inventory-sif-runtime.py` による実行比較、SIF 内部の固定コミット SHA やテキストファイルの直接比較、`gh run view` のステップ結論判定など、推測的・過剰な検証を全廃。

### 3. メトリクス収集と BigQuery 送信の境界分離
- **計算ノード（`scripts/run-qwen35.pjm` / `run-qwen35-node.sh`）**:
  - `EXPERIMENT_METRICS_GOOGLE_CREDENTIALS_FILE` の受け渡しを廃止。
  - `smoke-send` や直接の BigQuery insert 処理を計算ジョブ内から削除。
  - メトリクス断片（JSON fragment、イベント、ログ、プロファイル生データ等）を `${RUNTIME_DIR}/domain/fragments` や `${RUNTIME_DIR}/results` へアトミックに書き出す処理のみに限定。
- **ログインノード上のランナー（Login Runner）**:
  - ジョブ完了後、Login Runner 上で `qwen-experiment-metrics` を用いて断片を回収・検証。
  - Login Runner から BigQuery へ raw insert を送信し、GitHub アーティファクト（`qwen35-sglang-v2-*.tar.zst`）をパッケージングしてアップロード。
  - 認証情報（パーミッション 0600）は Login Runner のみで管理し、アーティファクトやログから確実に除外。

### 4. スクリプトのコーディング規約刷新と不要資産削除
- **行継続の全廃**:
  - `scripts/` 配下のスクリプト、PJM スクリプト、ワークフロー定義内のすべてのコマンド呼び出しを行継続（`\`）なしの1物理行に記述。
- **環境変数の整理**:
  - 不要な中継用環境変数を排除し、直接引数渡しまたは標準的な変数名へ統一。
- **Python 実行基盤の統一**:
  - Python ツールの実行には `uv run` または `uvx` を使用。
- **不要ファイルの削除**:
  - `outdated/qwen-experiment-metrics-v1/`、使用されていない一時診断スクリプト、不要なログファイルを削除。

### 5. AGENTS.md およびドキュメントの更新・日本語化
- `2_Qwen/AGENTS.md` の運用ルールを最新化。
  - GitHub Actions タグの指定規約（`[require-build]`、`[cluster:genkai]`、`[nodes:...]`、`[profile]`、`[skip ci]`）
  - BigQuery 資格情報を計算ジョブへ渡さない原則の明記
  - コマンドの行継続禁止、環境変数中継の禁止
- ワークフロー定義（ジョブ名・ステップ名・入力パラメータ説明）および人間向けドキュメントの日本語化。

---

## 完了条件 (Acceptance Criteria)
- [ ] `.github/workflows/qwen.yml` が新設され、push / 手動実行のエントリポイントとして統合されていること
- [ ] `.github/workflows/validate.yml` が新設され、単体テスト・静的検査が先行実行されること
- [ ] push 時の `[cluster:genkai]`、`[nodes:...]`、`[profile]`、`[require-build]`、`[skip ci]` タグ制御が正常に機能すること
- [ ] クラスタタグが指定されていない push では、静的検証のみが実行され計算ジョブが起動しないこと
- [ ] SIF ビルド再利用判定が first-parent history walk と sidecar SHA256 / サイズ比較に簡素化・高速化されていること
- [ ] 計算ノード（PJM）への BigQuery 資格情報受け渡しと直接送信が廃止され、Login Runner からの断片回収・送信に一本化されていること
- [ ] スクリプト内のコマンド呼び出しから行継続（`\`）が全廃され、1物理行で記述されていること
- [ ] 不要な中継用環境変数が排除されていること
- [ ] `outdated/` 配下のレガシー資産・不要ファイルが削除されていること
- [ ] `2_Qwen/AGENTS.md` およびドキュメント・ワークフロー文言が最新の運用ルール・自然な日本語へ更新されていること
- [ ] `2_Qwen` 上でローカル検証（pytest、Ruff、bash 構文検査等）がすべて成功すること

---


## Conversation

### Comment 5368227367

yayoimizuha · 2026-08-21T09:42:17Z

制御処理の責務分離と整理をコミット `89393cd` に反映しました。

- 玄界および ASPIRE のワークフローで重複していた第一親成果物選択、BigQuery 断片（fragment）送信、公式成果物の梱包処理を共通スクリプトへ分離
- プロビジョニング時にスクリプトをランタイムディレクトリへスナップショット保存し、投入済みジョブをリポジトリ更新の影響から隔離
- SGLang ソース OID の取得処理を、計算ノード／ランクごとの Enroot 2回起動から、プロビジョニング時の1回抽出へ集約
- セレクタ、資格情報ファイル（権限 0600／ジョブ終了時削除）、ベンチマーク／プロファイル成果物パッケージの境界をフィクスチャ化

Actions run 32468843371 の validate ステップに成功し、bash 構文検査、actionlint、Ruff、pytest（全85件）もすべてパスしました。コンテナ構成、Buildah レイヤーキャッシュ、SGLang の `uv.lock` 非固定方針は変更していません。agy Gemini 3.7 Flash High による差分レビューも LGTM を確認済みです。


### Comment 5650613485

yayoimizuha · 2026-09-13T03:11:55Z

旧Qwen3.5 SIF/dummy構成から公式のQwen3-235B FP8 OCI/Enroot構成への移行が完了したため、本アンブレラIssueを現行アーキテクチャへの代替（superseded）・完了としてクローズします。

現行リポジトリでは以下が整備・反映済みです：
- 統合エントリーポイント（`qwen.yml`）および `validate.yml` の導入
- commit-tag ルーティングの実装
- 再利用可能なクラスタワークフローの整備
- ログインノードでの成果物ハンドリング
- 計算ノード上におけるBigQuery認証情報の排除
- シェルコマンドの行継続の排除
- 不要となった旧Qwen3.5コードの削除

検証状況：
- pytest（138件）および全静的チェックをパス
- クロスクラスタ環境チェックの成功
- 最終Firmus Actions（34732532321）/ Slurm（17021）ジョブの正常完了（0:0）

現行構成へ完全に移行・検証されたため、本Issueをクローズいたします。


