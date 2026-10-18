# 2026-10-18 Claude Code 停止作業の引き継ぎ

対象は Claude Code session `5abe09c0-a310-5471-9b41-b47ce68d0de8` とその W0〜W4。session limit で停止した後、既存方針に沿う基盤の仕上げ・投入済み run の確認を Codex が引き継いだ。最適化の採用、精度基準の変更、未検証の束ねの serving 投入は行っていない。

## 完了した範囲

W0 の multi-candidate kernel bench 本体は main `1a9bb76`。検証 [39085655145](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39085655145) の全 10 step と成果物回収が成功した。詳細は [kernel timeline の起動方法](2026-10-18-megamoe-kernel-timeline.md) に集約した。

停止時の未コミット差分を確認して、以下を仕上げた。

- 17 分の Slurm 枠を候補数 4 の固定値から合計 12 step に変更。検証 run の実測 71〜77 秒/step が根拠。image timeline、SGLang の両 tree の check/bench、各候補の timeline を数える。失敗や長い JIT の所要時間までは保証しない。
- NVML の瞬間 clock による時間の正規化を削除。before / after の観測値は残す。同一 kernel でも瞬間値が大きく違うため、正規化値を性能上の根拠にしない。
- timeline patch が正常に適用された場合の空のエラーログを削除。
- 実際の provisioning を使う fixture と summary の検査を更新。SGLang と image timeline による step 数超過も staging 前に拒否することを検査。

検証専用 branch は `archive/experiment/firmus-kernel-bench-multi-20261018`（`14c829c`）に退避して削除する。W2/W3 の結果を保持する実験 branch、W1/W4 の未統合 branch は引き続き必要なので保持する。

## 投入済み kernel 実験の確定状態

raw 出力と来歴の正本は `/srv/qwen-artifacts/<run>/1/`。ここには判断に必要な要点のみ記す。下記はいずれも 1 回の kernel bench であり、serving の公式 score や性能の再現性を証明しない。

| 担当 | Actions / commit | 結果 | 続行に必要なこと |
| --- | --- | --- | --- |
| W2 Prefill | [39086777829](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39086777829) / `bb80858` | `drop`・`tail-drop`・`tail-dyn`・`tail` は全 gate 成功、各 12/12 case の hash が image と一致。Slurm 24442 は COMPLETED、5分54秒。 | 同条件で再測定し候補を選ぶ。統合後の通常 build と公式 serving 検証は未実施。 |
| W3 Decode | [39086770417](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39086770417) / `f1b393b` | `decode-combine` と `komoriyuta` は gate と hash 12/12 成功。`decode-all`・`decode-sched` は gate 失敗。Slurm 24441 自体は COMPLETED、16分48秒、Actions は summary 判定で failure。 | 失敗した候補の同期・スケジューリングを診断。成功候補の採用は再測定後。 |

W2 の prefill-balanced-4096 は image 1.6657 ms に対し tail-drop 1.6076、tail-dyn 1.5823、tail 1.6570 ms。prefill-skew-4096 ではそれぞれ −6.6%、−3.4%、−4.9%。ケース間で順位が変わり、電力・clock の影響もあるため一つを採用する根拠には不足する。

W3 の decode-balanced-128 は image 0.2051 ms、decode-combine 0.2002（−2.4%）、komoriyuta 0.1994（−2.8%）。これも単回で、両候補の差から勝者は決めない。

W3 で確認できた最初の致命的なエラーは両失敗候補とも rank 3 の `torch.cuda.synchronize()`（`scripts/check-firmus-megamoe.py:89`）で報告された `CUDA error: unspecified launch failure`。該当ログは `logs/logs/decode-all-gate.log` と `decode-sched-gate.log`。非同期エラーなので、この行だけから不正な kernel 内命令は断定できない。両候補には flat barriers と dynamic schedule が含まれる。`decode-all` には no-prepull-barrier も含まれるが、それだけを原因とする根拠はない。gate 失敗後も他候補を計測し、最終 summary が failure を返す経路は実働した。

## 停止位置と重要な判断を残す範囲

- W1 runtime は branch `perf/firmus-max-runtime-20261018`、`77b3ef3` まで push 済み。[Tier 1 のノート](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/blob/28ac1cbdb33af47592dd5251dcdff64ec2902d16/research/runtime-performance/2026-10-18-firmus-runtime-tier1.md) に解析、上限 136 / c192、polling 2、optimistic 2、aligned chunks、router TCP_NODELAY の patch と配線案がある。Tier 2 の allocator 同期除去・DP 同負荷 tie-break・idle rank の geometry 修正は未完成。allocator と実行中の GPU step の所有権・順序を証明する前に有効化しない。
- W1 の追加サブエージェント `a39770e3a2f6aa4ce` は admission の全同期点を列挙する読み取り専用解析を依頼された直後に停止。完成レポートはない。既存 M1 の scripts/data は元 session の scratchpad `t1/m1/` に残っている。
- W4 SGLang small kernels は branch `perf/firmus-max-sglang-kernels-20261018`、`af55451`。3 patch と検査・bench script は commit 済みだが GPU 未検証、build への配線も未実施。worktree の `patches/sglang-bench/` は未追跡のまま保持している。保存されていた patched tree (`scratchpad/applytest`) に対する `check-firmus-latency-source.py` は全項目 PASS（CUDA arithmetic/transport は mock）。
- W4 bench は現状で投入不可。Writer が `kernel`・`rows`・`sha256` を書く一方、基盤は `case`・`output_sha256` を要求する。環境情報・variant 探索等の非 timing 行と候補のみの case も同じ JSONL に入り、summary と一致しない。`bench_rope` は候補の関数有無を表す `exact` 引数を数値比較結果で上書きし、image 側で候補専用関数を呼ぶ可能性がある。Ruff の確認も未完（今回の確認で当該 2 script に 27 件）。したがって GPU job は起動しなかった。
- W4 の新 QK-norm 実装は「未融合参照と bit 一致」を狙う一方、既存 image の実装は同じ参照と bit 一致しないケースが既存ログにある。image 比 hash 一致という kernel bench の採用条件と、どの演算を正本にするかを統合前に整理する必要がある。判定を緩めたり比較元を黙って変えたりしていない。
- geometry/redundant expert の机上評価は別 branch `perf/firmus-max-geometry-20261018`、`96031ab` に完了済み。候補を採用しない理由が記録されており、新たな GPU 実験は不要という元の結論を保持した。

## 検査と残留物

引き継ぎ時に Firmus を `ssh firmus` で読み取り専用確認。対象 3 run の runtime directory は既に削除済みで、同ユーザーの Slurm queue は空だった。ほかの担当の branch や実行中 Actions は変更・停止していない。main の未追跡の古い監査ノートと `task/`、ほかの worktree は引き継ぎ開始時から存在するため保持した。

今回の main の変更はローカルの全 pytest、Ruff（`qwen-experiment-metrics scripts containers`）、`bash -n`、コマンド行継続禁止検査、`git diff --check` を通過。既存 run の summary を再生成し、基盤検証と Prefill は成功、Decode は上記 2 案を失敗と判定することも確認した。
