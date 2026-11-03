# Round 29: QKV 専用 M-tail split、round 25 image 再利用

作業 branch は `perf/pp4-r29-qkv-tail-20261104`、親は round 28 記録の `ecacaa3`。今回の変更は QKV 専用 runtime overlay と、その独立 gate・dispatch 回収である。O は従来の一回 GEMM に戻す。C2 と C3 は config、overlay、gate、test から外し、C2 の FA3 AOT row・build patch も削除した。**新規 build、SSH、Actions 操作、Slurm 投入、GPU 実行はしていない。round 29 の採否、通常 serving dispatch、公式 score は未測定。**

## 根拠と比較条件

AGENTS.md、round 28 note の「Firmus 実行結果」、実装と commit 本文、tracker/model、model が参照する旧 Prefill profile JSON、round 24 の PP4 profile summary、保存済み roofline 分析・会話由来の `codex-scratch/r28/astra29-final.md` を確認した。提供されていない会話履歴を取得したとは扱わない。Actions の過去実行は、読み取り専用の退避済み run `41656547563` の raw JSON、Slurm accounting、worker log、dispatch txt を参照した。外部 API での取得・操作は行っていない。

raw 正本は `/srv/qwen-artifacts/41656547563/1/qwen-firmus-pd-41656547563-1/results/results/prefill-r28/`。12 UUID の各 forward/reverse を別々に中央値集計すると、QKV は **24/24 セルで max(2 µs, 0.5%) の閾値を満たし、3.868〜8.437 µs 短縮、−0.754〜−1.593%**。O は全24セルで悪化し、22.043〜28.211 µs 増、+5.044〜+6.343% だった。round 28 note の GPU ごとの要約と、ここでの両順別範囲は集計単位が異なる。QKV の改善を O と共通の gate で取り消す構成を解消する。

ただし前回の実測 image は `0e0deb09` で、今回再利用する `a001f901` の結果ではない。前回の約5.5 µs/層、約0.13 ms/24層は既存 gate の換算であり、今回の serving 改善や score +11 を保証しない。今回解消すべき不確実性は、round 25 image の QKV 単独経路が同じ byte・耐久・両順速度基準を満たすか、採用後に実 serving で dispatch するか、公式 parser の品質・score を維持するかである。

## QKV の実装と仕事量

対象は FP8 E4M3 入力/weight、FP32 K128 scale/promote、BF16 出力の **M7988/N9216/K4096 のみ**。`patches/sglang/prefill-qkv-tail-r29.patch` が `prefill_fp8.linear` から `dense_gemm` を呼び、projection が QKV かつ `SGLANG_PREFILL_QKV_TAIL_SPLIT=1` の場合にのみ分割を許可する。helper でも A=(7988,4096)、weight=(9216,4096) を検査するため、O=(7988,4096,8192) や別 M/K/N は flag によらず従来の一回呼出しになる。

body7936行→tail52行の順で同じ stream、同じ output の互いに重ならない view へ既存 DeepGEMM を二回呼ぶ。7936 は BM256・group128・TMA の16-byte alignment に整合し、scale の column stride を維持する。weight、K 累積順、scale 適用、DeepGEMM host/header/mainloop は変更しない。round 28 の CPU/CUDA 証拠では body は BM256/BN128/BK128・stage3・cluster2、tail は BM64/BN80/BK128・stage11・cluster1・REG248/spill0。これは CUDA13.4 の既存証拠であり、今回 image の GPU 検証の代わりにはしない。

有効 FLOP は **603,073,806,336/層**。A/scale、B/scale、出力の論理一回通過 I/O は **218,734,080 byte/層**で、追加 launch の B 再読出しや cache を含む実 HBM byte ではない。tail の N9280 padding を含めて削る MMA は **14,461,960,192 FLOP/層、24層で0.347087 TFLOP**。round 24 の1.410 GHz/1524.70 TFLOP/sを仮定した仕事量換算は約0.228 msだが、別runの実測時間と合成した改善値ではない。通信処理・NVLink payload は不変、実 HBM/NVLink byte と round 29 の net gain は未測定。model JSON の未確定値をゼロで埋めたり、既存 profile を今回の通常 serving 実測に読み替えたりしていない。

O の tail-first は同じ stream の二回 launch の総仕事量を減らさず、別 stream は body と tail の資源・weight 読出しの競合を変える。今回の raw は直列 tail の悪化を示すだけで、これらの正味改善を裏付けない。独立 O gate は追加しない。round 9 の mainloop 分岐による悪化、round 28 の O 悪化は棄却理由として維持する。C2 は全shape遅く、C3 は byte 不一致かつ SMEM 削減なしのため、修正・再試験を今回へ広げない。

## Gate・budget・dispatch

config の `prefill.qkv_tail_split_gate=true` と、serving の default-off flag `SGLANG_PREFILL_QKV_TAIL_SPLIT=0` を分離する。`scripts/firmus_prefill_r29.py` と `scripts/select-firmus-prefill-r29.py` が QKV 一shapeだけを測定し、schema は `firmus-prefill-qkv-tail-r29/v1`、candidate は `qkv-tail`。完了した12 rank の証拠を `results/prefill-r29/qkv-tail/prefill-N/rank-R.json`、候補の判定を同ディレクトリの `selection.json`、全体判定を `results/prefill-r29/selection.json` へ保存する。

先行 pair gate と同じ allocation の **12 UUID の完全一致・重複なし**、H200、run/image/source hash と現行 C/P/P3 pin の一致を要求する。QKV の byte 一致、入力v0→v1→v0復帰、変更入力での graph replay、二つの出力 buffer を使う1200回の通常 launch と復帰を確認する。24回 graph の一呼出し換算 µs を一 sample とし、forward/reverse 各10 sample、各 GPU の両順中央値で **max(2 µs, control の0.5%) 以上の改善**を要求する。pooled 値による救済や前回 raw の流用はしない。seed、入力生成、screen の精度・耐久手順は round 28 から維持する。

greedy gate 後・model 起動前、3 worker×4 GPU を並行起動する。worker90秒、全体100秒、worker終了時刻も開始+90秒以内とする。`SECONDS + 100 + 660 <= 1020` の場合だけ始め、660秒の serving 予約を維持する。時間不足・不完全証拠では flag0 のまま。前回 job は904秒、gemm-tail 自体は例示rank約14.5〜14.7秒だったが、cold JIT と image が異なる今回の wall time は未測定で、短縮を予算へ先取りしない。gate の cache は既存方式の node-local `/tmp/qwen-prefill-r29-${SLURM_JOB_ID}-${worker}/{triton,deep-gemm,tvm}`、serving cache は既存設定を維持する。

worker は継承 flag を unset し、Prefill だけ採否後の env を読む。**実際に QKV の body/tail の二回 GEMM を呼び出した後**、process ごとに一度 `Prefill r29 qkv-tail using split 7936+52 rows projection=QKV M=7988 N=9216 K=4096 pid=...` を warning に出す。flagを立てただけ、O、別shapeではこのログは出ない。gate log と serving log は別であり、`score.json` 確認後に serving の `logs/prefill-N.log` から全rank行を保持して `results/prefill-r29-dispatch-{0,1,2}.txt` へ回収する。該当行がなければ no-dispatch を明記する。採用判定だけを実 serving 使用の証拠にしない。

## Image と E2E、投入担当への引継ぎ

config の `images` 全体と `container_*` 全項目、13本の DeepGEMM overlay はローカル main `3b1d99e` の round 25 構成と一致することを確認した。SGLang image は `/scratch/home/ACCOUNT/qwen/containers/41573354672.sqsh`、SHA256 `a001f90142b571380894e9532e467c57b831397c9e2e9d613d9e04004c87def9`、12,298,305,536 byte、source `14c3e861d52185f4a2753620213caf629c5d0400`、sidecar `41573354672.container-sidecar.json`。FA3 build.py と候補表は main と byte 一致へ戻した。a001f901 にない C2 の AOT row を参照しない。

`stage-qwen-overlays.sh` は既存 image の SGLang package を copy し、config 順で patch して bind mount する。今回の追加は Python runtime overlay のみで、host/header/AOT/container 内容を追加しないため **build 不要、image a001f901 を再利用**する。Actions の既存成果物選択→provision/overlay/必須 PP4 CPU gate→Slurm 内の既存 pair/prepack/P3/greedy→r29 gate→通常 serving→公式 parser/score→dispatch と raw 成果物回収、という経路を維持する。計算ノードへの資格情報や結果送信経路は変更しない。

今回の作業 commit は先頭 `[skip ci]`、この branch だけへ push する。**Claude が後から通常 Firmus serving を投入する commit の先頭 tag は `[cluster:firmus] [nodes:2] [prefill-pp4]`。`[skip ci]`、`[require-build]`、`[profile]` は付けない。** `.github/workflows/qwen.yml` は `research/**` と `**/*.md` を paths-ignore にしているため、research/Markdownだけ・空commitではpush起動されない。投入を追跡する一行 comment 等を **`configs/firmus-prefill-pp4.json` または `platforms/firmus/framework-comparison/sglang.sbatch`** に含める。JSONにはコメントを入れられないので、値を変えない追跡 comment なら後者を使う。config の gate=true と image pin は維持する。

投入担当は terminal 状態、12 UUID の採否、採用時の serving dispatch の全12 process 行、公式 parser の品質・score、raw/ログ・来歴の回収を確認する。優位性は同条件の複数 run で評価し、main merge はその確認後に判断する。今回のローカル作業だけで E2E 完走・採用・性能改善とはしない。

## ローカル検証と片付け

最終実装に対し `export PATH=$HOME/.local/bin:$PATH` 後、次の指定コマンドを実行した。CI の shallow checkout に対応し、新テストは git の commit 履歴を参照しない。image/container の期待値は固定値とし、実 config 順の overlay 適用、QKV/O/別shape の呼出し、serving の once log、rank行を落とさない dispatch 回収、12 UUID gate の各種不正証拠、期限・予算境界・timeout を検査した。

| 検証 | 終了コード |
|---|---:|
| `uv run --project qwen-experiment-metrics --with 'numpy>=1.26,<3' --with pytest python -m pytest -q -p no:cacheprovider qwen-experiment-metrics/tests` | **0** |
| `uv run --project qwen-experiment-metrics ruff check qwen-experiment-metrics scripts containers` | **0** |
| 最終 overlay の37テスト | 0 |
| 関連 integration regression | 0 |
| sbatch/provision の `bash -n`、staged `git diff --check` | 0 |

全体 pytest は既存の source/GPU 条件付き skip を含む。GPU の実測による byte/耐久/速度、serving dispatch、公式 score を今回の CPU 検証で確認したとは扱わない。

指定 scratch `scratchpad/codex-scratch/r29` に terminal log・exit code・検証説明を保存した。展開 source、途中・最終の自分の pytest temp、source の bytecode cache は削除済み。今回作成・投入した scheduler job はなく、他の job や main・他 branch・SGLang mirror・ユーザー設定は変更していない。

途中で全体テストの TMPDIR も指定 scratch 配下へ移したが、既存 UNIX socket テストの socket 名が Linux のパス長上限を超えることを短い診断で確認した（診断先だけでも161 byte、`AF_UNIX path too long`）。自分の途中実行のみ停止して終了143とログを保存し、最終の指定コマンドは TMPDIR 上書きなしで実行して終了0を確認した。標準一時領域に生成した自分の pytest ディレクトリも終了後に削除し、他作業の一時ファイルには触れていない。これは runtime/gate のエラーではなく、ローカル検証時の一時パス指定の訂正である。

## Firmus 実行結果（run 41689469266、commit d33b595、image a001f901、build なし）

Slurm job 28586 は COMPLETED、elapsed 14:37。Decode greedy は採用。QKV gate は 12 GPU すべてで byte 一致・耐久合格、両順中央値で 514〜532 µs から **−1.08〜−1.45%** となり採用された。serving でも 3 worker の全 PP rank が `Prefill r29 qkv-tail using split 7936+52 rows` を出しており、実際に dispatch された。

しかし公式 parser の score は **9777.16**（good request 0.969）で、round 28 の 10020.36（0.982）から −243 と大きく下がった。

| 指標 | round 28（41656547563） | round 29（41689469266） |
|---|---:|---:|
| TTFT p50 | 1290.9 ms | 1271.8 ms |
| TTFT p99 | 3878 ms | 5440 ms |
| TTFT max | 5122 ms | 6726 ms |
| ITL avg | 15.63 ms | 15.70 ms |
| 30 s 区間ごとの TTFT>3 s 件数 | 0 s:232、60 s:111 | 0 s:232、60 s:193 |

60 s 付近の SLA 超過の山は、過去の run（41280565072・41305946419・41317543745・41656547563）で 111〜113 件と安定しており、193 件は通常の揺らぎを大きく超える。TTFT p50 は改善しているので、定常の QKV は gate どおり速いが、負荷が溜まる区間で tail 分離が待ちを増やしたと見られる。原因（追加 launch と PP 段間の同期、負荷時の clock など）は未特定。単一 run だが、悪化幅が見込みの利得（約 +11）を桁違いに上回るので、再試行せずに不採用とする。

結論：QKV/O の M tail 分離は、gate の単体速度が改善しても serving では採用しない。kernel 単体の 1% 前後の差は、serving の SLA 超過の変化で簡単に打ち消される。
