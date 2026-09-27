# EAGLE3 c40: BF16からFP8 KVへの変更

2026-09-27のユーザー依頼による実機比較。実験の親は反復済みEAGLE3 c40の `66f3ca6`。実行ブランチは `research/aspire-eagle-c40-fp8kv`。

**状態: 設定変更とFP8実機初回の計算は完了。SSH認証切断によりraw回収・反復確認が未完了で、性能上の採用判断は保留。**

## 条件と判定

- Qwen3-235B-A22B-Instruct-2507-FP8、ASPIRE 2ノード16 H100、P2×TP4/D2×TP4・EP1、concurrency40。
- Nebius EAGLE3 Instruct draft、3 steps/topk1/4 tokens、node-local model、max-total-tokens131072、既存image `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5` を固定。
- `kv_cache_dtype=auto`（実効BF16）から `fp8_e4m3` へ変更。prefill/decode両workerへ同じ指定を渡す。追加のメモリ容量調整やattention backend指定は行わない。
- 公式AIPerfコマンドを固定し、規約に合わせたローカルparser `scripts/parse-aiperf-result.py` でrawを再解析する。このparserは主催者提供ではない。通常benchにprofileを混ぜない。
- [規約追補Q1/Q4](https://github.com/hpcac/2026-APAC-HPC-AI/blob/main/1.%20Qwen3%20AI%20Task%20Rules.md#q1-basic-vanilla)はFP8 KV保存・転送を許可し、INT8/INT4/FP4を禁止する。主催者の全件accuracy検査は未実施。

## 配備sourceの確認

SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4` のserver_argsは `fp8_e4m3` に対応する。FA3はFP8 KV経路を持ち、queryもKV dtypeへcastするため、単なる容量半減だけではなく演算経路も変わる。変換コスト、カーネル効率、draft受理率によって性能は下がる可能性もある。

公式target revision `e156cb4efae43fbee1a1ab073f946a1377e6b969` のsafetensors indexにはk_scale/v_scale/kv_scale tensorがない。quantization/kv_cache.pyは未収録時にscale=1.0を設定する。今回外部のscaleファイルは与えず、標準動作を測る。性能測定やgeneric greedy probeの一致だけで全件accuracy合格とは扱わない。

## 実行経過

|Actions|commit|内容|
|---|---|---|
|36298132846|d694aef|FP8初回。PBS213328、dgx004/016。dgx016のdecode両workerがEnrootのrdma_cm mountで失敗。SGLang起動前、PBS Exit1、性能値なし。raw回収・home archive checksum確認済み。|
|36298251187|66f3ca6|設定を変更しないBF16対照。PBS213329、dgx018/027、Exit0、walltime12分1秒。Actions全体success。raw parser一致、home checksum確認済み。|
|36298644714|e5d05cd|FP8を対照と同じdgx018/027へ固定。既存testのselect文字列assertion不一致でCPU validation失敗、GPU投入なし。|
|36298738175|095134d|FP8実機初回。PBS213331、dgx018/027、Exit0、walltime11分52秒。計算成功後のraw転送が途中で切れ、Actions failure。|
|36299355333|095134d|同一設定のFP8反復。PBS213334、dgx018/027。06:17:17 UTCのR状態を最後にSSH接続不可、Actions failure、PBS terminal未確認。|

rawの正本は `ssh archive-host` の `/srv/qwen-artifacts/<Actions run>/1`。初回の環境失敗をFP8の性能0として比較しない。ノード固定はこの観測された失敗を避け、対照と物理ノードを揃える実験上の条件であり、mainの一般的な配置方針にはしない。

## BF16対照

今回の同一ノード対照は2349.382 good output tokens/s、joint SLA98.0299%、1472 requests、0 errors、OSL mismatch0。TTFT平均1477.490ms/p95 2410.392ms、ITL平均9.143ms/p95 10.807ms。過去2回（2346.714/2315.183）も補助的に参照するが、FP8との直接比較は今回のノード対照を優先する。

## FP8初回の観測（回収未完了）

計算ノード上で生成されたscore.jsonをSSHで読み取り、2325.146 good output tokens/s、joint SLA98.4171%、1453 requests、0 errors、OSL mismatch0、ローカルparser validを確認した。BF16同一ノード対照比は約-1.03%だが、反復未確認であり、回収後のraw再解析も未実施。速度向上や有意な性能劣化を断定しない。回収できたAIPerfログの平均TTFTは1435.54ms、平均ITLは9.58ms。

ログはtarget KVの実dtype=torch.float8_e4m3fn、131072 tokens、GPUあたりK/V各1.47 GBを記録する。BF16対照の各2.94 GBから半減した。draft KVもFP8。容量節約は実測済みだが、同時実行数やKV token上限を固定した今回の初回ではgoodput改善を確認できていない。

## SSH切断と回収再開

2026-09-27 06:18 UTC、ASPIRE ControlMasterがBroken pipeになり、再接続はPermission denied (password)、Actions側はHost key verification failedとなった。初回の回収はtar Unexpected EOFで中断し、home archiveは部分的なlogsとserver-infoだけを含む。archiveジョブのsuccessはrawの完全回収を意味しない。反復はartifactなし。cleanupもSSH失敗で実行できておらず、run directoryが残っている可能性がある。

ユーザーへ既存接続の再認証を依頼済み。接続コマンドは `ssh -T win-5im4vj0sfos -- ssh -F /dev/null -T -S /run/aspire2p-control/master -o BatchMode=yes -l ACCOUNT aspire2p.nscc.sg --`。ホスト鍵確認を無効化したり、PBSジョブを停止したりしていない。

復旧後に確認・回収する対象:

- 初回: PBS `213331.pbs111`、`/scratch/users/industry/ai-hpc/ACCOUNT/sglang-runs/pd-36298738175-1.KcldieaWjj`。terminal Exit0は取得済み。既存の部分archiveは改変せず、`36298738175/1/recovered-after-ssh` へ完全なrawを回収しchecksum確認する。
- 反復: PBS `213334.pbs111`、`/scratch/users/industry/ai-hpc/ACCOUNT/sglang-runs/pd-36299355333-1.vDE4srE7sk`。まずterminal状態を確認する。実行中なら継続を待つ。完了後に同runのhome archiveへ回収する。
- 両runのraw exportを固定コマンド・来歴とともにparserで再解析し、score.jsonと一致すること、実際のdtype/image/配置を確認する。補助greedy probeも比較するが主催者accuracyの代替にしない。
- 必要なrawの保存・checksum確認後に、この2件の終了済みrun directoryだけを削除し、PBS残留を確認する。

CPU validation失敗runはGPUを使っていない。初回のdgx016環境失敗runは回収・cleanup済み。BF16対照はActions全体success・PBS Exit0・完全raw再解析・checksum確認済み。追加のジョブは認証復旧まで投入しない。
