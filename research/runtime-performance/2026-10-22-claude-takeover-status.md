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

## 原因は未確定、推測だけで変更しない

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
