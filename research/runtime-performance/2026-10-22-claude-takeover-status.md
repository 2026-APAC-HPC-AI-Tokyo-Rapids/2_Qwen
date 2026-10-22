# 2026-10-22 Claude Code 停止後の Decode bundle 確認

Claude Code session `5abe09c0-a310-5471-9b41-b47ce68d0de8` の最新会話は、B1/B4/B5/B6 の再実行を待ち、2026-10-22 08:20 UTC の通知後に利用上限で停止している。依頼は重要な意思決定を伴わない範囲の続行。追加指示はジョブ投入を最低限にすること。今回の新規 Actions、build、scheduler job の投入は **0 件**。

## 確定した実行状態

- branch `perf/firmus-decode-combo-20261022`、commit `b327b87acfd026654ae98b9e9c82a03bd4c014c0`。
- [Actions 39727354451](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39727354451) は failure。Slurm 25417 は FAILED、ExitCode `1:0`、実行時間 1分41秒（08:13:42–08:15:23、scheduler 表示時刻）。待機を再開する必要はない。
- image は build 39711569005、SHA256 `98d1c1138308fcd204f13d9b44d4cc9853462b00d8310fd31481df179426fe5f`。今回の失敗だけで build を無効にしない。
- raw 出力の正本は `/srv/qwen-artifacts/39727354451/1/`。`source/source/provenance.json` と保存された gate script / overlay を確認した。
- モデル起動前の numeric gate で停止したため、公式 parser に渡す benchmark 出力も性能 score もない。B1/B4/B5/B6 の採否、main の 8356.18 に対する優位性、B5 を残すかどうかは未判定。

## 最初の致命的なエラー

`logs/logs/sm90-megamoe-gate.log:1647–1693` の traceback:

1. `check-firmus-megamoe.py:35` が decode kernel 検査を `runpy` で実行。
2. `check-firmus-decode-kernels.py:387` の `check_decode_router_padding()` から、比較参照 `reference()`（372 行目）の `moe_fused_gate(...)` を実行。
3. Triton の launcher 初期化で `compile_module_from_src()` → `_load_module_from_path()` が失敗。

```text
ImportError: /home/ACCOUNT/.triton/cache/X2KRTMQYOFQKLJBW46E3N7VELF3JRDD5CBMXFJEB4JH4MLSDL47A/__triton_launcher.cpython-312-x86_64-linux-gnu.so: cannot open shared object file: No such file or directory
```

これは数値比較 assertion の失敗ではない。ただし、router 検査全体の成功や最適化の正しさも証明しない。その後の `server-info-prefill-0.json` 不在は、モデル起動に到達しなかったことに伴う後続エラー。

前回 [39714885018](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39714885018) は、この手前の RoPE FP8 Q の bit 比較で停止していた。`b327b87` はその判定と重複 gate 起動を修正した commit。今回のログでは FP8 KV と従来の fused Top-K padding が4プロセスとも PASS した後、新 router 検査の一部が成功し、その途中で launcher 読み込みに失敗している。

## 「以前は止まらなかった」との関係

保存済みログと script 差分を照合した。

| run | 新しい `check_decode_router_padding()` | 今回の launcher ImportError |
| --- | --- | --- |
| 39447708936（main の2WG検証） | script に存在しない | なし |
| 39550627795（2WG + replica4） | script に存在しない | なし |
| 39699383079（cap132） | script に存在しない | なし |
| 39714885018（Decode bundle 初回） | 存在するが前段で停止 | なし |
| 39727354451（再実行） | 一部を実行、途中で停止 | あり |

新 router 検査は `993077c` で追加された。従来の `check_topk_padding()` と別の検査で、Triton router の padding 融合と非融合参照を比較する。以前の成功runはこの新経路の成功実績にはならない。一方、上の確認範囲を超えて「過去に同種のキャッシュ障害が一度もない」とまでは断定しない。

## 初回調査時点：原因は未確定、推測だけで変更しない

4プロセスは numeric-gate 用 HOME を共有する。`mounts numeric-gate` は run 配下の `home/numeric-gate` を HOME に bind mount し、保存元の `/scratch` は login node の `findmnt` / `stat -f` では BeeGFS。serving worker は別途 `/tmp/sglang-$name/triton` を使う。gate と serving の cache 配置は異なる。

image `39711569005.sqsh` 内の実際の `triton/runtime/build.py` と `cache.py` を、login node で必要な2ファイルだけ展開して確認した。`FileCacheManager.put()` は UUID を付けた一時ディレクトリへ書き、`os.replace()` で公開する。`compile_module_from_src()` は既存cacheの読み込みに失敗すれば一度ビルドし直し、最後に公開先からロードする。今回の traceback は最後のロード側に相当する。

したがって、4 rank が同時に動いたという事実だけでは、単純な不完全書き込み競合と断定できない。共有 filesystem 上の可視性なども含め、生成後のファイルを参照できなかった理由は未確定。rank ごとの cache 分離案をローカルで準備・確認したが、原因を確定しないまま本番経路へ追加することはせず、差分を取り下げた。retry、lock、cache 管理処理も追加していない。

## 続行時に残ること

- 同じ image の launcher 生成・ロードと gate の filesystem 配置を、GPU不要の範囲でさらに切り分ける余地がある。再現していない原因を確定扱いしない。
- 再投入する場合は、この障害だけでなく、まだ到達していない router の残り、FA3 候補・metadata共有、MegaMoE の数値ゲート、モデル起動、通常benchmark、profile、成果物回収までを確認する必要がある。
- 精度判定の再変更、検査の削除、B5 等の候補削除、mainへの最適化採用は行っていない。既存の実験branchとimageを保持する。

## 今回の確認・残留物

関連するローカル検査 `test_firmus_reference.py` / `test_firmus_kernel_bench.py` は48件成功（65.39秒）。一時的な cache 分離案の shell 構文と4プロセスの引数・cache分離もCPU上では成功したが、GPU障害の再現・解消確認には数えない。コード差分は残していない。

`ssh firmus` で Slurm の terminal 状態を再確認し、元 runtime directory `/scratch/home/ACCOUNT/qwen/runs/pd-39727354451-1.O49y6DdhCl` が削除済みであること、同ユーザーの queue が空であることを確認した。今回作った login node の一時展開ディレクトリも削除済み。既存の未追跡ファイル、他のworktree、branch、ジョブには変更を加えていない。

## 追調査：BeeGFS共有cacheの並行置換直後のロード失敗をCPUで再現

ユーザーから原因を特定するよう指示を受け、`ssh firmus` のlogin node上で短時間のCPU診断を行った。GPU、scheduler、Actionsは使っていない。上記の「原因未確定」は初回調査時点の記録であり、以下が追調査の結論。

実際のimageから抽出した `FileCacheManager` クラスを変更せず実行した（Triton本体のGPU依存を避けるため、ASTでクラスだけ抽出し、cache directoryを指定するknobsを注入）。GCCで作った `int qwen_cache_probe(void) { return 42; }` の共有ライブラリを `put()` で公開し、直後に `ctypes.CDLL()`（dlopen）でロードして戻り値を検査する。4プロセスを各反復の `put()` 直前でbarrier同期し、同じキーを使う。500反復×4プロセス＝2,000ロードを条件ごとに実施。

| 保存先・共有方法 | ロード回数 | ENOENT失敗 |
| --- | ---: | ---: |
| `/tmp`、4プロセスで同じキーを共有 | 2,000 | 0 |
| `/scratch/home/ACCOUNT/qwen/diagnostics`（BeeGFS）、同じキーを共有 | 2,000 | **9** |
| 同じBeeGFS、rankごとにキーを分離 | 2,000 | 0 |

BeeGFS共有の9件はいずれも、今回のgateと同じ `cannot open shared object file: No such file or directory`。失敗直後の `os.path.exists()` はtrueで、直後の再ロードは全件成功した。同期を入れない先行診断でもBeeGFS共有で同じ失敗を1件確認したが、別の3回×100反復では再現せず、タイミング依存である。

**結論：この環境ではBeeGFS上の共有cacheを並行置換した直後のdlopenが一時的に失敗する。** Tritonのatomic公開だけを読んで「分離なしで動く」と断言した説明は不十分だった。GPU計算、routerの数値、rank別CUDA contextを使わずに同種の障害を再現し、cacheの配置・共有だけで発生を切り分けられた。BeeGFS内部のどの処理がENOENTを返すかまでは調査対象にしていない。

最小修正として、`sglang.sbatch` のnumeric gateのコンテナ内で `TRITON_CACHE_DIR=/tmp/qwen-numeric-gate-${SLURM_JOB_ID}` を指定する。4 rankは引き続き同じcacheを共有し、BeeGFSへの配置を避ける。serving workerは既に`/tmp`を使っている。rank別wrapper、retry、lock、image再ビルド、精度判定変更は不要。

診断で作成した `/tmp` とBeeGFS上のディレクトリはすべて削除した。CPU再現経路での切り分けと修正のshell検証までを完了とし、新しいGPU jobは投入しない。修正を適用した公式servingのE2E完走は未確認であり、次の必要な実験にこの修正を含める。

## ユーザー指示による再投入：router通過、後続の候補検査でworkspace不足

ユーザーの「じゃあそれで投げて」を受け、`b327b87` を親とする branch `experiment/firmus-decode-combo-local-cache-20261022` に、numeric gateのcache配置変更だけを適用した。commit `fb1c7d2ae0022315fa49e4f826a56936b069c4b8`。[Actions 39754745523](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39754745523) は既存imageを再利用。事前のRuff、shell構文、全ローカルpytestは成功。条件・overlay・判定基準は前回と同じ。

attempt 1はNsight package取得時の `curl: (6) Could not resolve host: developer.download.nvidia.com` で配備に失敗し、Slurm未投入。DNS解決が復旧したことをlogin nodeで確認し、同じrunの失敗したjobから再試行した。Nsight取得処理は新規追加ではなく、2026-10-07の `276911a` から存在する。

attempt 2は配備に成功し、Slurm **25447** を1件投入。09:32:12 UTC投入、09:41:51開始、09:44:14 FAILED、実行2分23秒。**4 rankすべてで `PASS: Decode Triton router exact weights/IDs` を確認し、前回のlauncher ImportErrorはなし。** `/tmp`へのcache移動は実機の当該検査まで確認できた。

その後、`check_verify_attention_candidates()` → helper `alternative_cases()` の `BatchPrefillWithPagedKVCacheWrapper.plan()` で次のエラーとなった。

```text
Buffer overflow when allocating memory for batch_prefill_tmp_s with size 1048576 and alignment 16, but only 0 bytes available in AlignedAllocator. Increase the workspace buffer size.
```

根拠は回収した `logs/logs/sm90-megamoe-gate.log:3172`（他rankも同じ）。helperはworkspaceを **128 MiB固定** で確保している。実際のimageのFlashInfer `attention/scheduler.cuh` はsplit-KV時に一時出力 `batch_prefill_tmp_v`、次にsoftmax統計 `batch_prefill_tmp_s` をfloat workspaceへ割り当てる。今回は前者の割当後に残量0となり、後者の1 MiBを確保できなかった。GPU全体のメモリ不足や数値比較の失敗ではなく、検査helperが渡すworkspace容量の不足である。

image内のPython wrapperには、同じplan条件から必要なfloat/int workspace bytesを返す `workspace_size(...)` と、その容量を渡す `reset_workspace_buffer(...)` が存在する。次に修正するなら、固定容量の推測上の増量より、このAPIで候補ごとの必要量を確保してからplanするのが直接的な対応。検査の削除や精度判定変更は不要。今回はこの追加修正やGPU再投入は行っていない。

raw正本は `/srv/qwen-artifacts/39754745523/2/`。48ファイルのSHA256を照合し、commitとimage SHAが予定どおりであることを確認した。Actionsもfailureで終了し、runtime削除stepは成功。通常benchmark・profileには到達せずscoreなし、最適化の採否は引き続き未判定。今回のGPU投入は1件、buildは0件。継続調査用に実験branchを保持し、今回作成したローカルworktreeは削除した。

## 未採用候補のgateを外した再実行：E2E成功、7582.76

ユーザーから「使ってない奴まで試すのは要らない」と指示があり、性能探索は別kernel benchに保持し、serving前の必須gateから未採用候補の検査だけを外す方針を確認したうえで、再投入の明示指示を受けた。

同じ実験branchの commit `9f93944bebd4a175b2776789d04190e9691ba066` は、`check_verify_attention_candidates()` のFlashInfer prefill/decode・Triton attention・FA3 knob/要求分割の探索検査と、norm-quantの8種類の演算順序診断を削除した。本番B1/B4/B5/B6の検査、通常実装との数値参照比較、許容誤差、境界条件、CUDA Graph再実行、B4の94層metadata共有検査は保持。別benchとhelperは変更していない（helperはB4検査の入力生成にも使用するが、未採用候補関数は呼び出さない）。FlashInfer workspaceの増量やimage変更は行っていない。

事前Ruffと関連pytestは44件成功・Torch未導入による31件skip。Actions側の全検査も成功した。[Actions 39759800897](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/39759800897) は **success**。既存image `39711569005` を再利用し、build・別kernel benchなし、GPU jobは1件のみ。Slurm **25453** は10:08:06 UTC投入、10:09:35開始、**COMPLETED / 0:0 / 15分18秒**。全数値gate、モデル起動、通常benchmark、別profile、設定検証、回収と退避まで完走した。

| 通常benchmark | 値 |
| --- | ---: |
| good output tok/s | **7582.756** |
| 要求数 / 要求エラー | 4793 / 0 |
| good request fraction | 98.3518% |
| TTFT 平均 / p99 | 1700.357 / 3663.497 ms |
| ITL 平均 / p99 | 16.056 / 18.656 ms |
| SLA eligible / official workload validation | true / true |

保存したexport・command・client provenanceを用い、リポジトリの `parse_result()` と `validate_official_result()` を再実行し、保存済みscoreの一致と公式workload条件の成功を確認した。これはtask accuracyの公式受理を主張するものではない。スコアは先行する通常計測のもので、後続のinstrumented profile/clientの値は採用スコアに使っていない。

mainの記録8356.18、2WG + replica4・c192の参照run 39550627795の8366.32より、今回のスコアは低かった。1回のbundle実行であり、個々の変更の採否や再現性は確定しない。Decode profileのkernel-summaryには次の内訳がある（µs/call、いずれも同じworkerの集計値。kernel合計をそのままstep時間としない）。

| Decode kernel | 参照39550627795 | 今回39759800897 |
| --- | ---: | ---: |
| dense FP8 QKV | 17.255 | **44.330** |
| dense FP8 O | 14.284 | **30.538** |
| MegaMoE | 199.172 | 198.381 |
| FA3 verifyの主要kernel | 50.789 | 48.998 |

B5 small-Mのdense GEMM時間増加が目立つ。単純に94層分のcall増分を足すと約4.07 ms/step相当だが、これはprofile内訳の目安であり、bundle全体の通常scoreへの独立した因果効果の測定ではない。最適化のmain採用、B5の削除、追加実験は行っていない。

raw正本は `/srv/qwen-artifacts/39759800897/1/`。CIFS経由の読み出しが遅かったため、保存サーバーへ `ssh tomokazu@100.85.119.7` で接続し、同じ共有の実体 `/srv/qwen-artifacts/39759800897/1/` で144ファイルすべてのSHA256を照合した。`run-success`、予定のcommit/image SHA、4本の `.nsys-rep`、`validation.json` のvalid=true・4worker各4GPU（計16GPU）を確認した。

`ssh firmus` でSlurmのCOMPLETEDとruntime `/scratch/home/ACCOUNT/qwen/runs/pd-39759800897-1.hiTWjH3DEr` の削除を再確認。今回のローカルworktreeと一時再解析ファイルも削除した。継続評価用に実験branchは保持し、他のbranch・ジョブ・既存未追跡ファイルには変更を加えていない。
