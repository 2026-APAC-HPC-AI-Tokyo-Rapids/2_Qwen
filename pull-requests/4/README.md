# Pull request 4: Genkai上でのSIFビルドおよびPJM待機処理の簡素化

Author: yayoimizuha
Created: 2026-07-22T14:26:39Z
Closed: 2026-07-23T02:36:06Z
Merged: —

The original pull request is retained in a private archive for maintainers. This public copy preserves the sanitized conversation and its original authors and timestamps.

## Description

## 変更内容

- Genkai ログインノードランナーの `/var/tmp` 配下で SGLang SIF をビルド
- 完成した SIF を `/fast/PROJECT/sglang-sifs` へアトミックに配置・公開
- Astral 公式インストーラーを用いて `uv` をインストール
- 独自の PJM ポーリングおよびログ監視処理を直接的な `pjwait` 呼び出しへ置き換え
- 不要となった `scripts/pjm-job.sh` ヘルパースクリプトを削除

## 変更理由

自宅サーバー（home-lab）のビルダーが不要となり、Genkai 標準の `pjwait` によって独自のポーリングを行わずにスケジューラの必要な待機動作が実現できるため。

## 検証

- ログインノードの `/var/tmp` 配下において、SSH 経由での Singularity フルビルドおよび `%test` の実行が完了
- fakeroot による所有権や拡張属性（xattr）の操作が失敗するため、`/fast` をビルドルートとして使用できないことを確認
- シェル構文、ワークフロー YAML、および `git diff --check` の検証を完了
- 本コミットには `[require-build]` が含まれているため、push されたワークフローにおいて新しいビルド経路が実行される


## Source snapshots

- Base: recorded `902a794171d3817d8f3b134b77d11dfb0b9ee287`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/cee732fb4b473ccbe33e1a316bf505484ed81553)
- Head: recorded `88bc33bc940e36c2e9e046ce806e06c4bff74f0e`; original retained privately

## Conversation

