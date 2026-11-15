# Pull request 5: PJMジョブ待機中におけるログのストリーミング出力

Author: yayoimizuha
Created: 2026-07-23T02:36:25Z
Closed: 2026-07-23T02:37:37Z
Merged: —

The original pull request is retained in a private archive for maintainers. This public copy preserves the sanitized conversation and its original authors and timestamps.

## Description

## 変更内容

- `pjwait` がジョブの完了を待機している間、Qwen の PJM スケジューラログをストリーミング出力
- 正常終了時、異常終了時、および中断時のいずれにおいても、バックグラウンドの `tail` プロセスを停止
- ジョブ終了後における既存の `run-success` マーカー検証処理を維持

## 変更理由

既存の成功判定やクリーンアップ動作を維持しつつ、実行時の進捗を GitHub Actions ログ上で可視化するため。

## 検証

- `git diff --check`
- ワークフロー用スクリプトに対する Bash 構文検査

本プルリクエストはクローズされた PR #4 を置き換えるものであり、後続のログストリーミングに関する変更のみを含みます。


## Source snapshots

- Base: recorded `8986ea7200ac622e5810bdf9f00d94926bc9fdf2`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/452e225cf4bed780d2c91d0d7c67297aa1e8b244)
- Head: recorded `373def9f645a456f226183681eb6b19be76b2860`; [sanitized snapshot](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/commit/8a4925b9093dd94ce8e501fb40b8cc8f80c89c65)

## Conversation

