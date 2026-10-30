# Basic: public SGLang 0.5.23 on Firmus

## Scope and evidence

ユーザー指定に従い、既存のsource/kernel変更版を **Bonus**、公開SGLangをオプション・環境変数・配置だけで調整する版を **Basic** と分ける。Firmusを優先する。主催者の10月29日のSlack回答も、公開版が提供するbackend/kernel/設定と独自実装を区別している。

判定条件は[公式要項](https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/1.%20Qwen3%20AI%20Task%20Rules.md)。公開FP8 target、16 GPU、plain PD、gateway経由、AIPerf 0.12.0の固定60/120/30秒、ISL4000/OSL200、TTFT≤3000 ms・平均ITL≤20 msを同時に満たすrequestが90%以上であることを維持する。scoreはgoodput×200。独自の短時間probeを提出スコアと扱わない。

検討の入力はGit/Actionsとraw artifacts、`research/`、この作業ディレクトリに対応するCodex・Claude Codeのローカル履歴。会話内の推測を測定値に格上げせず、以下の研究記録・runで来歴を確かめた。

- [Prefill supply rebalance](2026-10-09-prefill-supply-rebalance.md)、[campaign summary](2026-10-13-campaign-summary.md): 4429.9のP6TP2/D1TP4は旧source/patchを含む。今回のstock Basicの性能値として流用しない。
- [MegaMoE integration audit](2026-10-17-branch-integration-audit.md): 6344.96も当時の独自MegaMoE等を含む。旧版で「vanilla」と呼んだ設定も、最新公開版を十分に調整した対照にはならない。
- [draft比較](2026-10-28-fp8-draft-placement-and-fa3-attribution.md) §8: Nebius受理長2.251、Meituan1.860、RedHat1.945/2.015。異なるdraftの再探索より公開Nebiusを固定する。旧独自実装の受理長を今回の測定値とは扱わない。
- [PP4 round 4](2026-10-30-pp4-round4-9832.md)、[round 6](2026-10-30-pp4-round6-9828.md): Bonusの9832.74/9828.95。既存の提出パッケージ9120.51（再実行9132.03）とは別の構成。Basic対照の有効な反復測定が揃うまで、これらとの速度比を確定しない。

## Public image and launch boundary

キャンペーン開始時の最新公開release [v0.5.23](https://github.com/sgl-project/sglang/releases/tag/v0.5.23)、tag commit `297e54fd105e974cdf401e5b4d20253a22c16f90` を選択。公式linux/amd64 image `lmsysorg/sglang:v0.5.23` をmanifest digest `sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5` で固定する。

この版はQwenのSM90 block-FP8 MegaMoEを既に公開実装として備える（[Qwen3 model](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/models/qwen3_moe.py)）。旧版で独自patchだったという理由だけで最新Basicから除外しない。逆に旧Bonusのprivate DeepGEMM環境変数やkernel選択表を、そのまま公開版へ持ち込まない。

`containers/basic.Dockerfile` は公開serving package・DeepGEMM・kernelを変更しない。AIPerfは別venv、gatewayは公式image同梱のbinary。workerはpublic argv/envを記録して `exec` するだけで、monkeypatch、PYTHONPATH overlay、独自wheelは使用しない。NebiusのBF16重みを保持し、draft configのarchitecture selectorとdtype宣言だけをnative loader向けに整える。変更前後configとweight SHAを保存する。

ビルド [41046995676](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41046995676) は成功。SquashFSは16,417,116,160 bytes、SHA-256 `d6ea5d95d6d76ab87ea55aeb70a052f04dbba8f9bbd515c3ece394690b2a1058`。source/build/toolchainのsidecar、OCI検証、AWS終了まで確認済み。runtime比較はこの同一imageを使う。

BonusのPP4はそのまま候補にしない。公開Qwen3MoEの `set_eagle3_layers_to_capture` はnon-last PP stageでreturnし、必要なaux hiddenの受け渡しを旧Bonusと同等と確認できていない。一般的なPP対応の存在だけでは、このQwen/EAGLE3構成の動作根拠にならない。

## Initial layout and controlled comparisons

全条件共通: P3/D1、各worker 4 GPU（attention TP1/DP4、MoE EP4）、FA3、FP8 KV/page64、native MegaMoE + DeepGEMM、memory fraction0.90、Nebius EAGLE3 steps3/draft4/BF16。P0/P1を第1node、P2/D0を第2nodeに置く。Prefill graph無効、Decode graph有効。KV poolはP32768・D393216/rank、最大runningはP256・D128（32/rank）。Decode graph bucketsは1,2,3,4,6,8,12,16,20,24,28,32で固定し、client concurrencyと一緒に変更しない。

| Commit / Actions | client N | local Prefill chunk / request limit | 状態 |
| --- | ---: | --- | --- |
| `2f22722` / 41051125031 | 160 | 4096 / unlimited | 配備失敗。Firmusのunsquashfs4.4が`-cat`未対応。GPU未使用 |
| `27d865d` / [41053120260](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41053120260) | 160 | 4096 / unlimited | 測定完走、SLA不適格、全raw回収・保管済み |
| `4b10c27` / [41057394591](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41057394591) | 128 | 4096 / unlimited | attempt1は配備中にSSH接続reset。attempt2は配備完了、Slurm27641で資源待ち |
| `e8d963a` / [41057547575](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41057547575) | 128 | 4096 / 1 | attempt1/2とも配備中断、GPU未使用。attempt2の再試行は部分directoryと衝突。下記の同条件再投入へ |
| `00b0552` / [41058071649](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41058071649) | 128 | 8192 / 2 | 配備完了、Slurm27640で資源待ち |

| `5c55438` / 41060099701 | 128 | 4096 / 1 | 切断時cleanup修正を含む同条件再投入。fixture import順のRuffエラーでCPU段階失敗、GPU未使用 |
| `9d4d990` / [41060170465](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/41060170465) | 128 | 4096 / 1 | import整形後の同条件。配備2m23sで完了、Slurm27643で資源待ち |

1→2 request/batchの比較では、token枠も2 requestが収まる値にする。この2設定は物理バッチサイズという同じ主要因を表す。その他の条件を固定する。`--prefill-max-requests` は[公開schedulerの上限](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/schedule_policy.py)を使用し、独自のrequest-alignment実装は移植しない。

### Preparation and validation fixes

接続中断はモデル準備段階で起こり、benchmarkの失敗とは分ける。cap1 attempt2に残ったuv/HFプロセス3個は、対象runのdraft pathをargvと親子関係で確認してから終了し、その部分directoryだけを削除した。他の実行には触れていない。prepareのdraft downloadを独立process groupで起動・background waitし、HUP/TERM/INT時にowned groupと未完了directoryを片付ける。切断のCPU fixtureで子process停止とdirectory削除を確認した。

全体CIでは旧migration testのlauncher全体SHAが、意図したembedded parser更新に追従していなかった。`131c7e9`でserving shellの固定SHAとparser同期検証を分離。更新前後のshell本体はbyte一致を確認済み。Basicの限定CIにも提出物とEAGLE collectorの回帰検証を含めた。

`536a337`のreadback checkerは4 worker×4 DP stateそれぞれの実token容量・実request cap・graph/batch geometryを検査する。回収済みc160の全16 stateをread-only再検証して通過済み。collectorが旧launcher名を付けていたpost-benchmark metadataも`15fdd74`で修正した。これらはserving条件を変えない。

## c160 observation

Slurm27625、g1/g4、11:00、READYまで403秒。全workerのversion/topology/request capとrequested/effective KV capacityが一致。接続sanity応答は正常。AIPerfのrequest errorは0。計測中のprofilerは使用していない（AIPerfのphase名profilingは通常の測定phase）。

| metric | observed |
| --- | ---: |
| measured output token/s | 6031.94 |
| good output token/s | 4670.78 |
| good request fraction | 77.4339%（不適格） |
| TTFT avg / p90 / p99 (ms) | 2386.90 / 3816.06 / 5193.56 |
| ITL avg / p99 / max (ms) | 14.06 / 16.77 / 19.95 |
| requests | 3780 |

全ITLが20ms以内なので、SLA不合格はTTFT側。profiling phaseのrequest開始時刻で30秒ずつ分けたpass率は67.28%、79.09%、86.67%、77.07%で、冒頭だけの問題ではない。Prefillの20:33–20:34ログでもnew-seq2の分割バッチが多数あり、小さいrequest断片まで次のforwardへ持ち越している。ログは標本であり、瞬間input-throughputの平均から総Prefill容量を推定しない。

このrunではローカル検証器が旧 `/opt/sglang/venv/bin/aiperf` のみを許す不整合も判明した。`4b10c27`でBasicの `/opt/aiperf/bin/aiperf` だけを追加許可し、全workload引数・provenance・exportの完全一致は維持。回収済みrawを修正版でread-only再検証し、command/provenance/exportの整合は確認できたが、77.4%のSLA不合格は変わらない。元score/artifactsは書き換えていない。

raw正本: `/srv/qwen-artifacts/41053120260/1/`（checksum付きhome-server archive）、`logs/logs/`・`source/source/`・`results/results/`。run directoryはterminal・回収後にworkflowが削除済み。高コストのBasic imageは保持。

## Selection rule and remaining work

現時点で適格なBasic最良値は未確定。c128とrequest上限・batch geometryを比較し、選択候補と対照を反復する。node組が違う場合は同条件の再測定を行い、単一runの差を設定の効果と断定しない。Pが引き続き供給限界ならnative scheduling/routingやbatch geometryを優先し、DのITLが限界になった時点でEAGLEのsteps/draft数、graph bucket、P/D比率を検討する。既に劣ったdraftやprivate kernel envの再探索は後回し。

公式accuracy全件検査は主催者が行う。gatewayのsanity応答やSLA検証は、その代替にはならない。
