# Prefill並列構成の再評価（2026-10-03）

現行のP2×TP4・EP1を最適と仮定せず、TP/EP/PP/DPを再評価する。合計16 H100とD2×TP4・EP1を固定し、モデルはQwen3-235B-A22B-Instruct-2507-FP8、BF16 EAGLE3 draft、FP8 KV、FA3を維持する。独立ワーカー2個という複製と、SGLang内部のDP attentionは区別する。

## コードとCPUスクリーニング

固定SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4` を調査した。異なるHEADのmirrorは変更しない。

| Prefill候補 | 判断と根拠 |
|---|---|
| P2×TP4・EP1、DP1 | 比較対照。現在の独立2ワーカー |
| P2×TP4・EP1、DP attention2 | 実機候補。各4GPU内のAttention TP2/DP2、MoE TP4。現行CUTLASSを維持できる |
| P2×TP4・EP1、DP attention4 | 条件付き。Attention TP1/DP4、MoE TP4。DP2の成立・方向性を先に確認 |
| P2×TP4・EP4 | FP8 CUTLASSはEP1のみのassertがあり、そのままでは起動不能。EP対応バックエンドへの変更を含む別実験が必要 |
| P1×TP8・EP1 | MoE中間幅1536/8=192がFP8 block128に非整列。現行CUTLASSでは起動不能 |
| P1×TP8・EP8 | バックエンド変更が必要。既存FPMによるCPU比較はワーカー数半減を補えず、追加実機の優先度は低い |
| 独立P4×TP2 | 235B FP8モデルの重みだけで2×80GBを超えるため対象外 |
| PP2 | SGLangがspeculative decodingと非互換。現在のEAGLE3を維持する比較では対象外 |

EP制約は `server_args.py:6447–6455`、PP制約は `:8436–8439`。TP8 EP1の整列制約は `layers/quantization/fp8.py:1135–1147`、限定的なpaddingの実装は `layers/moe/fused_moe_triton/layer.py:292–300`。これらは現在の実装の制約であり、独自実装を含む理論上の不可能性ではない。

既存AISimulate 0.13.0・413行のwhole-forward測定データを使い、fallback禁止で候補を照合した。測定identityはTP4/EP1とTP8/EP8のみで、DP2/DP4/EP4/PP2の正確なcellは存在しない。未対応を性能不良と解釈せず、別identityからの流用もしない。DynoSimへ切り替えても不足タイミングは補えない。

c44・4000要求・chunk8192・D固定の合成リプレイでは、対照が2588.405 raw tok/s・平均TTFT351.564ms、P1TP8EP8が1462.088 raw tok/s・3223.674ms。後者のjoint-good率は0.5%で不適格。このデータは旧auto MoE/BF16 KV/target-onlyであり、現行CUTLASS/EAGLE3の速度予測や公式スコアではない。

過去実機EP4 run `36301094490` は2202.553 good tok/s、TP8EP8 run `36303631570` は起動・生成後にjoint-good率1.754%でSLA不適格となった。どちらもP/D両方・旧条件を変更しているため、Prefill単独のEP効果とは呼ばない。

## DP2を1条件に絞る理由

Attentionの分割度を下げることで通信と小さいGEMMの効率が改善する可能性がある一方、MoE前後のgather/scatter、rank間待ち、異種TPのKV転送が増える。シミュレータに対応データがないため改善率は予測できず、コード・メモリの成立性を確認したうえで、まず1条件の実機E2Eで方向性を測る。

Pのみ `--enable-dp-attention --dp-size 2` を追加する。SGLangの自動調整で各DP rankのchunkは8192→4096、Prefill graph最大も4096、schedule conservativenessは既定値の0.3倍になる。この複合的なネイティブDP動作を評価するもので、Attention通信だけを分離測定した実験ではない。client concurrency48、Decode設定、既存統合イメージを固定する。

Qwen3通常MHAは `models/qwen3_moe.py:463–479` でAttention TPに応じたhead分割を行う。NIXLにはP Attention TP2→D TP4のhead単位転送があり、EAGLE3のdraftにもDP attentionの専用TP contextがある。実際の転送成功・数値出力はE2Eで確認する。固定pool131072の場合、target KVの追加容量は概算2.9375GiB/rankで、さらにdraft/Attention重みの複製が増える。既存ログの空き容量は成立性の目安にとどまり、実行時のピーク保証ではない。

## 実行記録

CPU側argv fixtureを含む52テスト、Ruff、bash構文、固定ソースのDP調整処理を検証し、2系統のSolレビュー後に `bea3efb` を投入した。[Actions 37126228529](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37126228529)。branchは `research/aspire-prefill-dp2-20261003`、親は同一イメージc48の `ecd8952`。新規ビルドは要求せず、再利用予定SHAは `20c3a38359ccc61967343ec23d600f783df8d487c6339d12b9ca53c14b55d3d3`。初回は同SHAを再利用したが、サーバー起動で失敗し性能未評価となった。単一runで優位性を確定せず、結果が有望なら同条件対照との反復を行う。runtimeのmain反映は優位性確認後とする。


初回PBS `215925.pbs111` は14:04:20–14:08:31 UTCに実行され、Exit_status1で終了した。最初の時刻付き致命ログは22:07:48 SGTのPrefill0 detokenizerによる `tcp://127.0.0.1:30335` のbind失敗（Address already in use）。DP attentionは通常のIPCからAPI+233〜239のTCPへ切り替わるため、API30100/30101ではP0 detokenizer30335とP1 tokenizer30335が衝突する。公式bench、結果parser、8 greedy probe、KV転送、GPUメモリ評価には到達していない。後続のrun-success欠損によるtarエラーは採点前起動失敗の原因ではない。

Prefill1には時刻のない `sympy.integrals.laplace` importエラーも記録された。実イメージ内に同ファイルが存在し、抽出したSymPy/mpmathをPython3.12.12で直接importおよび4並列spawn importする短時間CPU診断は成功した。ただしフルコンテナ・Torch・DP子processの状態やエラー前後関係を再現しておらず、原因確定や二次障害とは扱わない。

最小修正 `a953ec8` はP1 APIのみ30101→30110とし、既存処理によりNCCLも31101→31110になる。bootstrap9001とD設定・c48・イメージ・kernelを維持した。actual worker/router argv fixtureと同nodeのAPI/NCCL/bootstrap/DP派生範囲/handshake+13の衝突検査を含む53 CPUテスト、Ruff、bash構文が成功した。原因修正後の有限E2E再試験1回を [Actions37129209750](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37129209750) として投入し、failure terminalまで確認した。未解決の失敗に備える汎用retryや依存変更は追加していない。

初回raw正本は `tomokazu-ubuntu-server:/hdd_raid/samba/net_share/qwen-artifacts/37126228529/1/`、全24ファイルchecksum確認済み。PBS terminalとruntime削除を確認した。派生監査と再現scriptは `research-issue-17/20261003-existing-data-analysis/prefill-dp2-audit/` に保存し、rawを重複退避しない。TP4EP4+Tritonには別候補としてコード経路があるが、backend変更と未測定の中間bufferコストを伴うため今回は追加実機を保留する。DP2が中立・負ならDP4へ自動拡張しない。


## 再試験の結果と暫定判断

再試験 `37129209750` はPBS `215943.pbs111`、14:23:33–14:39:33 UTC、Exit_status1で終了した。再利用イメージSHA `20c3a383…55d3d3` とcommit `a953ec8` を来歴から確認した。API衝突とSymPy importエラーは再発せず、全workerは起動した。Pの実ログでDP2、chunk4096、breakable graph最大4096、schedule conservativeness0.3を確認し、Dはrandom seedを除く起動設定がc48対照と一致した。Pにはnative DPの自動変更としてFlashInfer allreduce fusion backendがauto→Noneになる差分もあり、Attention通信だけを変えた比較ではない。

メモリ確保は成立した。Pのtarget KVは各GPUでFP8 K/Vそれぞれ約2.94GB、draft KVもFP8で各約0.03GB。BF16 draftは重みdtypeを指し、KV dtypeとは区別する。target重み約57.14GB、draft重み約1.16GB、target/draft graph構築後の空きメモリは8GPUで8.26–8.50GBだった。これらはログの瞬間値であり、実行中ピークの測定ではない。

最初の異常はwarmup中22:31:52 SGTのP1 DP0/TP0における `NIXL_ERR_REMOTE_DISCONNECT`（room3468568658199182252）。P0のDP1/TP2にも翌秒同エラーが出た。後続のroom削除後queued chunkに対するassertと、Dの300秒 `KVPoll.WaitingForInput` timeoutを初発原因とは扱わない。warmup runner表示はcompleted48/errors0だったが、raw `profile_export.jsonl` の48件は**すべてInvalidInferenceResultError（実際の応答本文なし）**である。profilingは48件送信・完了0、150秒後に全cancelとなり、AIPerfは `No profile results to export` で終了した。公式score/parser、8 greedy probes、post-benchmark server-infoには未到達で、スコア0やthroughput0という性能値を作らない。

固定sourceの独立監査ではDP bootstrap/cache key衝突を発見せず、head4・P Attention TP2→D TP4・page1・FP8のdescriptor算術は整合した。target/draftのKV dtypeも同じため、mixed-dtypeの仮説は今回の根拠にならない。remote UUIDはDの初期agentと一致し、先行するDのfatal/OOMログは見つからなかった。しかしNIXL session登録・接続・progressのどの状態変化がdisconnectを生んだかは、今回のinfoログでは確定できない。missing-room assertのガードやdeadline延長だけでは初発を直す根拠がない。将来この構成を再開する場合は、そのsession状態の根拠を先に得る必要がある。

したがってDP2は**PD転送が成立しなかったため現構成への採用を見送る**。性能不良という判定ではない。今回の範囲では追加NIXL開発・GPU投入を拡張せず、DP4・別backendのEP4も保留する。P2/D2各TP4・EP1・DP1・PP1を維持するが、これはTP/EP/PP/DP全体の最適性の証明ではなく、今回の探索の暫定結論である。

再試験raw正本は `tomokazu-ubuntu-server:/hdd_raid/samba/net_share/qwen-artifacts/37129209750/1/`、全32ファイルchecksum確認済み。初回と合わせてPBS terminal、runtime削除、短時間診断ディレクトリ削除を確認し、実験branchは再現用に保持する。派生監査正本に `retry-audit.py`、raw record集計、起動条件比較、独立NIXL診断を保存した。
