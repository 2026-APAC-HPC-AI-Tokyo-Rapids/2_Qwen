公式Slack（2026apachpcai-vtf5729.slack.com）、`/home/tomokazu/.local/bin/discrawl`、公式リポジトリ（`./task`）から得られる情報をもとに、このプロジェクトを更新してください。
TODOリストは進捗に応じて書き換え、チェック、追加、分割してください。サブタスクとサブサブタスクも自律的に作成してください。サブエージェントの利用も許可します。
このタスク全体についてcommit、push、CI、およびログインノードの操作を許可します。モデルは`MODEL_PATH.md`に記載されたものを使用してください。
`/home/tomokazu/actions-runner`のself-hosted runnerは、Actions仕様と各スパコンのログインノードへのアクセスを調べる際に参照できます。
意思決定に必要な場合は会話履歴を参照し、過去のCodexへの指示も判断材料にしてください。

## 全体TODO

- [x] 公式Task Rules、Application Notes、Slack、公式リポジトリ、Codex履歴、現行コード・Git履歴・Actionsログを再照合し、現状との差異を整理する。
- [x] メトリクス収集、ベンチマーク取得、プロファイリングを公式タスク定義に合わせて実装する。
- [ ] 最終構成までコードを整理し、YAGNI原則に従う正確で外科的、無駄がなくテスト可能で読みやすい状態にする。
  - [x] 旧Qwen3.5 dummy経路と不要な汎用処理を削除する。
  - [x] 通常benchとprofileを分離する。
  - [x] 公式result parserとstandalone parserを同じ判定へ揃える。
  - [x] SGLang source revisionとAIPerf versionを固定し、build手順を明記する。
  - [ ] 採用構成の確定後、探索専用設定と一時的な診断要素を棚卸しする。
- [ ] 実ジョブによる比較と最適化を、自分で進められる範囲で完遂する。

## 実行TODO

- [x] AIPerf v0.12.0とSGLang sourceを固定した再構築可能なコンテナを作る。
  - [x] SGLang commit `fc275674a97d78553b22597ee72c8fd48f5420f4`をbuild scriptで固定する。
  - [x] SquashFS、OCI、source、toolchainのprovenanceを保存する。
  - [x] repo全体をbuild inputとする具体的な再構築コマンドを`submission/README.md`へ記載する。
- [x] Firmusの2ノード16 GPUで、4×Prefill TP2と2×Decode TP4/EP4をNIXLで起動し、単一gatewayへ接続する。
- [ ] 公式AIPerf commandを変更不能な形で実行し、raw exportと全service logを成果物へ回収する。
  - [x] 公式argvを固定し、実行時のcommandとprovenanceを保存する。
  - [x] 探索/profile runでraw exportと全service logの回収経路を検証する。
  - [ ] 公式60秒warmup、120秒benchmark、30秒graceの通常runを完走する。
- [x] `profile_export_aiperf.json`から公式goodput、good request fraction、scoreを検証するparserとテストを追加する。
  - [x] `good_request_fraction >= 0.90`、OSL 200、score=`goodput × 200`を検証する。
  - [x] 固定argvと来歴が一致する場合だけ、設定ISL 4000とAIPerfのprefix補正後client観測3986を厳密に判定する。
  - [x] 近似値、改変argv、来歴欠落・不一致を拒否し、失敗理由を`score.json`へ残す。
- [x] 通常benchとprofileを分離し、GPU・CPU・通信の観測結果を性能runと混同せず保存する。
  - [x] Firmus profile job 16928のGPU、CPU、NIC、InfiniBandデータを`submission/analysis/profiling_data/16928`へ保存する。
- [x] 旧Qwen3.5 dummy経路を整理し、提出用の単一Slurm scriptとbuild手順を整える。
- [x] 安価な静的検査、fixture、単体テストを完走させる。
  - [x] 113件のpytest、Ruff、bash構文、actionlint、YAML、diff検査を通す。
- [x] 玄界とASPIRE 2A+で2ノードのscheduler、Enroot、GPU、CUDA、NIXL、SGLang環境を検証する。
  - [x] 玄界 PJM 6769520、2ノード、終了コード0を確認する。
  - [x] ASPIRE 2A+ PBS 208754.pbs111、2ノード、`Exit_status=0`を確認する。
- [ ] Firmusへ最終baseline jobを投入し、terminal状態、公式結果、成果物、残留jobなしを確認する。
- [ ] concurrencyまたはPD構成を一因ずつ比較し、有効な改善を固定して公式120秒runを完走させる。
  - [x] HCA選択、NIXL待機時間、concurrency 8/16、single railを一因ずつ比較する。
  - [x] all-active HCA条件の再現run 16935を評価する。
  - [ ] NIXL prewarm比較runを完了する。
    - [x] 1回の2-token throwaway requestを公式AIPerfの前に追加する。
    - [x] Actions 34701093440 / Slurm 16938を投入する。
    - [ ] Slurm 16938のterminal状態、prewarm時間、AIPerf結果、成果物を確認する。
  - [ ] prewarmが有効ならall-active HCAとprewarmを`main`へ統合する。
  - [ ] concurrency 16の公式固定runを投入し、60/120/30秒の全区間を完走する。
  - [ ] 公式raw export、score、good request fraction、ISL/OSL、error count、全service logを検証する。
- [ ] README、research、submissionへ最終構成、再現方法、結果、制約を記録する。
- [ ] 完了時に一時worktree、実験branch、不要な診断処理、重複・残留scheduler jobを棚卸しする。
  - [ ] ユーザーや他作業のjobを除外し、この作業が作成した残留jobだけを確認する。
