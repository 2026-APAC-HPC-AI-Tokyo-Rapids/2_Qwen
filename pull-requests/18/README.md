# Pull request 18: FirmusのvLLM/SGLang比較とAWS vLLMイメージビルドを追加

Author: komoriyuta
Created: 2026-10-07T12:55:43Z
Closed: 2026-10-10T16:24:41Z
Merged: —

This public copy preserves the sanitized conversation and its original authors and timestamps. Complete original API records are retained in the private publication backup. The unmerged original was archived through GitHub for restricted viewing.

## Description

## 概要

- 既存のFirmus reference-07比較をチームのActions、Slurm監視、成果物保管経路に接続する。
- 公式vLLM 0.30.0イメージをdigestで固定し、AWSの一時EC2とBuildahからDocker-load可能なアーカイブとFirmus用SquashFSを生成する手動workflowを追加する。
- EC2起動、SSM実行、ECRキャッシュ、S3、終了処理、ログ回収を既存SGLangビルドと共用する。
- CPU検査、来歴、sizeとSHA-256検証を追加し、新しいSquashFSは独立したvLLM archiveへ配布する。既存baseline画像や比較設定のpinは変更しない。

## 差分の範囲

ユーザー指定の既存比較ブランチからmainへのPRのため、今回追加したvLLMビルドだけでなく、同ブランチのFirmus比較処理とその検証・文書も含む。マージ後、通常のFirmus実行はこのreference比較経路を使う。

## 検証

- Linuxで218 tests、3 subtests成功。
- actionlint（ShellCheckとPyflakes有効）、Ruff、シェル構文検査成功。
- GitHub CI成功: https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37620748180
- 既存baselineのActions controlは比較ブランチの文書に記録済み。

## 未実行・注意点

- 新しいAWS vLLMイメージの実ビルドと、そのイメージによるGPU推論は未実行。
- 新workflowの手動実行にはdefault branchへの取り込みが必要。
- ビルド後も比較workflowの画像pinは自動更新しない。新画像の評価時は別の条件コミットでpath・size・SHA-256・sidecarを更新する。
- reference-07比較は公式時間条件のsubmissionではなく、profilingも未実装。比較workflowの明示的なrebuild/profile拒否を維持する。

## Source snapshots

- Base: recorded `6c4fa9242204dda96b1c7a1ce8c1e3be9a8b07ca`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/a8f4313033931452dec5462d51090b6f8a3408a2)
- Head: recorded `e6bf265d718507b90846b22a6ae9d91a27d7f05f`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/ee9c8c6c130899006e57572c0d64157c51c6e9d2)

## Conversation


## Reviews

### Review 5442638453

copilot-pull-request-reviewer[bot] · 2026-10-07T13:00:08Z · COMMENTED

<!-- ccr-overview-v2 -->

## Copilot review overview

### 🔵 Needs a closer look

The new AWS image build and resulting GPU inference path remain unexecuted despite successful static tests and prior baseline comparisons.

**Review effort:** Balanced  
**Findings:** None

<details>
<summary><strong>What changed in this PR</strong></summary>

FirmusでのSGLang/vLLM比較を既存の実験基盤へ統合し、AWSでvLLMコンテナを構築・配布する経路を追加します。

**Changes:**
- Firmus reference-07比較と結果検証を追加。
- AWS BuildahによるvLLM SquashFS/Docker archive生成を追加。
- 共通ビルド・配布処理とテスト、来歴記録を拡張。

| File | Description |
| ---- | ----------- |
| `scripts/​provision-firmus-framework.sh` | 比較環境と固定成果物を配備 |
| `scripts/​parse-firmus-reference-result.py` | reference-07結果を検証・採点 |
| `scripts/​materialize-qwen-container.sh` | vLLM成果物の配布に対応 |
| `scripts/​build-vllm-container.sh` | vLLM OCI、SquashFS、Docker archiveを生成 |
| `research/​vllm-container-build.md` | AWSビルド手順を記録 |
| `README.md` | vLLMビルド導線を追加 |
| `qwen-experiment-metrics/​tests/​test_vllm_container.py` | 配布境界をテスト |
| `qwen-experiment-metrics/​tests/​test_repository_contract.py` | workflow契約を更新 |
| `qwen-experiment-metrics/​tests/​test_orchestration.py` | Firmus固有経路を検証 |
| `qwen-experiment-metrics/​tests/​test_firmus_reference.py` | 比較結果と来歴をテスト |
| `qwen-experiment-metrics/​tests/​fixtures/​firmus-reference.json` | 実測結果fixtureを追加 |
| `platforms/​firmus/​framework-comparison/​vllm.sbatch` | vLLM比較ジョブを追加 |
| `platforms/​firmus/​framework-comparison/​vllm-proxy.py` | vLLM PD proxyを追加 |
| `platforms/​firmus/​framework-comparison/​sglang.sbatch` | SGLang比較ジョブを追加 |
| `platforms/​firmus/​framework-comparison/​README.md` | 比較条件と運用を記録 |
| `containers/​vllm.Dockerfile` | digest固定のvLLMイメージを定義 |
| `containers/​check-vllm-image.py` | CPU上でイメージを検証 |
| `configs/​firmus-framework.json` | 比較条件と画像pinを定義 |
| `aws/​run-build-instance.sh` | framework別ビルドに対応 |
| `.github/​workflows/​vllm-image.yml` | 手動vLLMビルドworkflowを追加 |
| `.github/​workflows/​validate.yml` | container検査をRuffへ追加 |
| `.github/​workflows/​qwen.yml` | Firmus比較経路を切り替え |
| `.github/​workflows/​firmus.yml` | reference比較を実行・回収 |
| `.github/​workflows/​aws-image-build.yml` | AWSビルド処理を共通化 |
| `.github/​workflows/​aws-container.yml` | 共通ビルドworkflowを利用 |
</details>

---

💡 <a href="/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/new/main?filename=.github/skills/code-review/SKILL.md" class="Link--inTextBlock" target="_blank" rel="noopener noreferrer">Add a `code-review` agent skill</a> or configure MCP servers for context-aware, tailored reviews. <a href="https://docs.github.com/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review?tool=webui#mcp-servers-and-agent-skills" class="Link--inTextBlock" target="_blank" rel="noopener noreferrer">Learn more in the docs.</a>

