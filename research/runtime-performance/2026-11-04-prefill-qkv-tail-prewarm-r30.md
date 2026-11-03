# Round 30: QKV tail の serving process 内 prewarm

branch は `perf/pp4-r30-qkv-tail-prewarm-20261104`、親は round 29 記録 `eb6b44f`。image `a001f901` の Python runtime overlay に、各 Prefill process の起動時準備だけを追加する。**GPU 実行・新規 build・SSH・Actions 操作・Slurm 投入は今回行わない。性能・公式 score・実 GPU 上の準備時間は未測定。**

## 根拠と今回解く不確実性

AGENTS.md、round 29 patch・gate・config・sbatch・note、関連 commit 本文、kernel-theory-tracker/model と参照 Prefill profile JSON、round 24 PP4 summary、保存済み `codex-scratch/r29-codex.log` と `codex-scratch/astra30/astra30-final.md` を確認した。過去 Actions は退避済み run `41689469266` の source/manifest、worker log、Slurm log/accounting、results を読み取り専用で参照した。未提供の会話を取得したとは扱わない。

round 29 は全12 GPUで QKV-only gate を採用したが、score は9777.16へ悪化した。後続の astra30 分析の4節では、round 28/29 の定常 Prefill 供給は52.414/52.427 req/s、client 初 token の回帰切片は0.7565/2.2465秒、計測先頭バーストの TTFT 超過は111/193件だった。全12 serving process の初回 split dispatch は計測開始後で、公式 warmup にはなかった。warmup の主な ISL3989と計測の3986は異なり、M7988の split は計測時に初めて使われている。gate は別 process・別 `DG_JIT_CACHE_DIR` なので、gate 成功だけでは serving process の JIT/load 完了を証明できない。

約1.49秒の差を JIT 時間と断定はしない。今回変える主要因は、serving process 自身で body/tail を ready 前に初回実行することだけである。初回準備不足を除くと通常 eager serving が main を上回るかが、既存 artifact だけでは解けない不確実性である。round 29 の不採用は維持し、この新しい仮説を一回の候補として扱う。round 29 からの損失回復を main 比の改善と数えない。

## 起動順序と実装

image の SGLang source commit は `fc275674a97d78553b22597ee72c8fd48f5420f4`、build definition は `14c3e861d52185f4a2753620213caf629c5d0400`。image manifest の build patch を適用し、config の SGLang overlay 全33本を記載順に適用した source を読んだ。最後の `prefill-qkv-tail-r29.patch` を `prefill-qkv-tail-r30.patch` に置き換える。DeepGEMM host/header、FA3 AOT、コンテナ内容は変えない。

起動は `Scheduler.init_model_worker()` → `init_tp_model_worker()` → `TpModelWorker` → `ModelRunner.__init__()` → `initialize()` と進む。DeepGEMM config/device は `initialize()` より前に設定される。`initialize()` は `load_model()`、MoE top-k 準備、layer index 解決、post-load weight transforms、LoRA/deterministic 設定、KV dtype 設定の順で進む。その末尾に、target worker、`disaggregation_mode=prefill`、`SGLANG_PREFILL_QKV_TAIL_SPLIT=1` の条件で `prewarm_qkv_tail(self.model)` を呼ぶ。

この関数が返ってから scheduler が KV pool を割り当て、attention backend、CUDA graph を初期化する。scheduler の初期化通知・HTTP ready、sanity request、公式 AIPerf command はさらに後である。PP4 Prefill の既存起動引数は `--disable-cuda-graph --disable-overlap-schedule` のまま。準備は eager な通常呼出しであり、graph capture 中や forward/warmup callback の中で JIT を始めない。Decode および draft worker は hook に入らず、helper 自体も Prefill/flag を再確認する。worker の既存 flag unset と Prefill だけの env 読込みも維持する。

各 process の `model.modules()` から最初の対応 `Qwen3MoeAttention.qkv_proj` を選ぶ。既存の `linear_supported` で TP1 の QKV、N9216/K4096、FP8 E4M3 contiguous weight、FP32 block scale、SM90 と backend を検査し、実 weight と `weight_scale_inv` をそのまま読み取る。全ローカル層は同じ geometry/backend なので、PP rank ごとに一層分だけ実行して返る。準備済みフラグは process 内で一つだけである。対応層なしや実行エラーは起動エラーとなり、完了を偽装する fallback/retry は追加しない。

専用 buffer は A=(7988,4096) の E4M3 zero、SA=(7988,32) の FP32 one、out=(7988,9216) の BF16。SA は `(32,7988)` を transpose し、stride=(1,7988) とする。7988 は4行 alignment 済みで、serving の group quant と fused input norm の column-major/TMA-aligned scale allocation と同じ。body7936とtail52は親 SA のstrideを維持する。A/outもrow-majorの親bufferをsliceし、tailのoffsetはそれぞれ7936×4096要素、7936×9216要素、SAは7936要素となる。

実 serving と prewarm は `_qkv_tail_gemm` の**同一の二つの** `deep_gemm.fp8_gemm_nt` 呼出しを共有する。body→tailを同一 current stream で一回ずつ、`compiled_dims="nk"`、同じweight/scaleの引数で実行する。入力値は JIT geometry を変えず、公式 dataset は使わない。`torch.inference_mode()` 内で専用bufferだけに出力し、device synchronize 後に完了とする。model forward、KV、sampling、RNG、scheduler 統計への呼出しはなく、層の状態を変更しない。scratch tensorは関数から保持せず、processのJIT/load状態だけを残す。

prewarm は dispatch logger を呼ばない。完了・同期後に一度だけ `Prefill r30 qkv-tail prewarm done ms=... pid=...` を warning 出力し、wall time は allocation/JIT/load/GPU完了を含む。実 serving では従来どおり二つの呼出し後に `Prefill r30 qkv-tail using split 7936+52 rows projection=QKV M=7988 N=9216 K=4096 pid=...` を一度出す。prewarm によって serving dispatch の once flag を消費しない。QKV以外・M/K/N違い・flag0では従来の一回GEMMを維持する。

## Gate、成果物、image

gate は意図的に r29 の名前・規則を流用する。`scripts/firmus_prefill_r29.py`、`scripts/select-firmus-prefill-r29.py`、schema `firmus-prefill-qkv-tail-r29/v1`、`results/prefill-r29/qkv-tail/prefill-N/rank-R.json`、candidate/all `selection.json` を維持する。12 UUID・allocation/image/source/pin一致、byte一致、入力変更と復帰、二bufferで1200回、forward/reverse各10sample、各GPUの両順で max(2 µs,0.5%) 改善という判定は変えない。greedy後、worker90秒/全体100秒、`SECONDS + 100 + 660 <= 1020` の予算も維持する。

gate は `_qkv_tail_gemm` を使う dense helper を従来と同じ手順で測定するが、model startup hook は通らない。gate の準備と serving の準備は別である。`score.json` 確認後のログ回収だけ r30 へ置換し、`logs/prefill-N.log` の prewarm と split の全rank行を prefix/pid/time ごと `results/prefill-r30-dispatch-{0,1,2}.txt` に保存する。どちらも無ければ明示的な no-line を保存する。一方しか無い場合はその行だけを残すため、投入担当は各processで両方揃うことを確認する。gateの採用をserving dispatchの代用にしない。

image はSHA256 `a001f90142b571380894e9532e467c57b831397c9e2e9d613d9e04004c87def9`、12,298,305,536 byte、build run `41573354672` のまま。config の image/container全値と DeepGEMM overlay全13本は親から不変。`stage-qwen-overlays.sh` の copy→config順patch→bind mount による Python runtime overlay なので **build なし**で利用する。

有効仕事量は従来の603,073,806,336 FLOP/層、論理一回通過I/Oは218,734,080 byte/層。tail分離が削るpadding MMAは14,461,960,192 FLOP/層。今回の prewarm はこの仕事を減らさず、起動前へ一回の実行を追加するだけである。NVLink payloadは不変、実HBM byte、今回のGPU時間・scoreは未測定。旧gateからの約0.13 ms/24層、score約+11は条件付き予算であり、今回の達成値ではない。別runのclock/profileを合成しない。

## 成功条件と投入担当への引継ぎ

採用時には全3 worker×4 PP rankで ready 前のprewarm完了と、実serving split dispatchの両方を確認する。公式command、warmup/drain、SLA、input、PP配置、clock、既存gateは固定する。次を全て満たすことを性能上の成功条件とする。

- 計測先頭バーストの TTFT 超過が従来の **111〜113件以下**へ戻る。
- client 初 token 到着列の回帰切片が **約0.76秒**へ戻る。同じバースト・同じ回帰定義で比較する。
- 公式result parserの品質を満たし、通常servingの **scoreがmainの10020〜10025を上回る**。

起動遅延だけ戻ってmain同等なら改善として採用しない。単一runだけで性能優位と結論せず、改善が見えた場合に同条件の追試で確認する。今回の実装作業はGPU/E2E実行の確認ではない。過去round29の通常jobは877秒で、この値は今回のprewarm時間を含まず、今回の準備時間・予算内完走は未確定である。

Claude が投入する commit の先頭 tag は **`[cluster:firmus] [nodes:2] [prefill-pp4]`**。`[skip ci]`、`[require-build]`、`[profile]` は付けない。`.github/workflows/qwen.yml` の path filter は research/Markdownだけのpushを起動しないため、**`platforms/firmus/framework-comparison/sglang.sbatch` に投入記録の一行コメントを追加**する。`configs/firmus-prefill-pp4.json` もfilter対象外だが、値を変えずに履歴を付けるならsbatchを使う。今回の作業commitは先頭 `[skip ci]` で、このbranchだけへpushする。

投入担当はActions/Slurm terminal状態、gate全12 UUID、全12 processのprewarm/dispatch、公式parser/score、client rawとserver log、来歴・artifact回収を確認する。準備時間はwarning log、詳細経過はActions/Slurm、rawはrun directory/artifact、比較対象のmetricは既存送信経路へ残し、計算ノードへ資格情報は渡さない。

## ローカル検証とfixture

`prefill-r30-image-source.json.gz` は、読み取り専用のSGLang mirrorの固定commitと、退避image manifestに記録されたbuild patchのblobから再構成した**runtime overlay適用前**の24ファイルである。mirrorへcheckout/writeはしない。build patch blobのSHA256と、manifestに記録されたbuild後target全体のSHA256一致を作成時に確認した。manifestのSHA256は `26f58d624a2b089304d07ff6f69dc1704fc22727caa749d94a9758a8c4fdf4db`。fixtureにはimage/source/manifest来歴と各ファイルSHA256を含める。実imageを今回ローカルで展開・GPU実行したという意味ではない。

テストはfixtureを展開し、config順の全overlayを `git apply --check` と `git apply` で検証し、全PythonをAST parseする。test内でgit履歴・mirror・ネットワークは参照しない。実overlayから抽出した関数でbody/tailのshape/dtype/stride/offset、同一weight/compiled_dims、二回準備要求でも一回実行、prewarm後にも実serving once logが出ること、flag0/Decode除外、launch/sync失敗時の完了ログ抑止、実startup関数でpost-load後かつpool/graph前であること、ログ回収全rankを検証する。GPU byte/速度・通常serving scoreの代用にはしない。

指定の `export PATH=$HOME/.local/bin:$PATH` 後に以下を実行し、終了コードとlogを `codex-scratch/r30` に保存した。全体pytestは既存のGPU/source条件付きskipを含み、失敗なしで100%まで完了した。

| 検証 | 終了コード |
|---|---:|
| `uv run --project qwen-experiment-metrics --with 'numpy>=1.26,<3' --with pytest python -m pytest -q -p no:cacheprovider qwen-experiment-metrics/tests` | **0** |
| `uv run --project qwen-experiment-metrics ruff check qwen-experiment-metrics scripts containers` | **0** |
| r30 prewarm と r29 gate の関連50テスト | 0 |
| sbatch の `bash -n`、最終 staged `git diff --check` | 0 |

途中のfocused testでは回収testのshell抽出が `prewarm done` の単語中で切れる失敗があり、改行を含む `done` 境界へ修正した。runtime側の回収コマンドの不具合ではない。patchの空context行も余分な空白を除き、最終のclean適用・差分検査を通した。

source再構成の展開物、build patchの一時コピー、実装・診断scriptは削除済み。既存Unix socketテストには指定scratchの長いパスを使えないため、pytest自動生成の標準一時領域を使用し、今回生成したディレクトリだけ終了後に削除した。検証log/終了コードと説明のみscratchへ残す。今回作成・投入したscheduler jobはなく、他のjobやmain・他branch・SGLang mirror・ユーザー設定には触れていない。
