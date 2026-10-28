# 2026-10-28 FP8 draft の確認、局所配置の棄却、FA3 固定費の帰属、sm_90a の命令制約

[10-26 の 10k 計画](2026-10-26-prefill-10k-plan.md)の候補のうち、komoriyuta の 10-27 の結果（[2026-10-27-prefill-memory.md](2026-10-27-prefill-memory.md)）で残ったものを実装・検証した。統合ブランチは `perf/firmus-bundle-20261028`（first parent は `perf/moe-fc2-recovery-20261028`。main の first-parent 履歴は image 40197369083 の build commit cbca2b3 に届かず、container 選択が古い image を選ぶため）。

## 1. serving [40621030371](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40621030371)（de7a970、image 40197369083、build なし）

**9171.48 tok/s**（公式判定 有効、5829 要求、good 0.97770）。Slurm 26860 は profile pass の最後で 17 分の上限に当たり FAILED 1:0（17:09）だが、score は通常 pass で書かれている。正本は `/srv/qwen-artifacts/40621030371/1/`。

- **FP8 draft を採用**：9161.67（40480817670）と 9171.48。同じ構成の BF16 draft 4 回（9118.01 / 9120.51 / 9129.28 / 9132.03）をどちらも上回り、差は約 +0.4%。受理長は 4 rank で 2.207〜2.211（BF16 2.211）で変わらない。greedy では検証が target の argmax を取るので出力は変わらない。
- kernel 選択は job 内で再実行し、Prefill n192、Decode p128split2-s3-fp8 / QKV 128,144,5 で前回と同じ。前回選択の再利用（`reuse_kernel_selection`）は、routing capture が overlay と config を足すため再利用記録の検査で正しく拒否されるので外した。
- g1 GPU1 は 10-27 の運営による GPU リセット後なので、負荷時 1530 MHz（p90 1575）で他と同等。リセット前（40225743220、40480817670）は無負荷・負荷とも 1500 MHz 固定だったが、Prefill GPU は電力上限でもともと 1485〜1530 MHz なので影響は全体で 0.3% 以下と見積もる。

## 2. 局所配置（P1'）は棄却

profile pass で komoriyuta の capture が 3 worker × 4 rank × 8 forward の生 routing（96 NPZ）を記録した。`scripts/optimize-prefill-expert-placement.py`（e39f6a4）は forward ID 順に前半を最適化、後半を検証に使う。

| 予測短縮（ms/反復） | 前半 12 forward（最適化） | 後半 12 forward（検証） |
|---|---:|---:|
| α=0（計算と padding のみ） | 1.99 | 0.05 |
| α=0.5 | 2.32 | 0.10 |
| α=1 | 2.49 | 0.15 |

前半の改善は後半に再現せず、生成器は配置を出力しなかった。Prefill は DP attention で各 rank に要求がランダムに割り振られ、入力も合成 prompt なので、source rank ごとの expert の偏りがない。前半の利得は、特定の forward の M128 padding の偶然に合わせたものである。**rank 間の局所性を使う配置は、この負荷では効かない。** 現行の hot-4 複製を維持する。capture の仕組みは残し、既定では無効にした（e42ccfe）。

## 3. FA3 Prefill の M-tile 固定費（F2）：3% 以上の単独の手はない

10-25 の NCU（40172396762、N224 control、全 5,760 PC、40,484 sample）を、制御フローで区間に分けて帰属した（Codex、GPU なし）。

| 区間 | 命令 % | sample % | 主な stall |
|---|---:|---:|---|
| (a) prologue・tile 準備 | 3.2 | 5.2 | barrier・wait・long_sb |
| (b) 初回 KV・初回の対角 mask | 10.0 | 7.9 | wait・barrier・mio |
| (c) 追加の masked 反復 | 0 | 0.0 | — |
| (d) 定常ループ | 73.7 | 72.5 | wait 6090・barrier 5519・long_sb 4512 |
| (e) 最終 PV・drain | 2.1 | 3.9 | long_sb |
| (f) 正規化・epilogue・store | 10.5 | 6.1 | wait・short_sb |
| (g) scheduler・tile 間同期 | 0.5 | 4.4 | barrier 1495 |

- M128 は Q64・GQA16 の 8 token なので、N192 / N224 では追加の対角 KV block がなく、(c) は 0 だった。
- (b) のうち mask の比較・選択そのものは 1.1%。FP8 group-128 出力の SHFL は 0.23%、量子化と store 全体でも 3.3%。
- (g) の barrier は N224 の動的 scheduler のもので、現行 n192 の analytic scheduler へは直接当てはまらない。
- (d) 以外の合計 27.5% は、M1 回帰の固定費 20.3% より約 7 point 大きい。(b) と (e) には通常の KV の仕事も含まれ、PC 区間を回帰係数へ直接対応させられないため。

単一の変更で FA3 を 3% 以上削る根拠はなく、候補は作らなかった。詳細は Codex の作業ディレクトリに残した帰属表と PC 区間（M2 base `0x100057dce00` からの offset）による。

## 4. sm_90a の命令の制約（ローカル CUDA 13.4 で確認）

| PTX | sm_90a | sm_100a |
|---|---|---|
| `ex2.approx.f16x2` / `ex2.approx.ftz.bf16x2` / `tanh.approx.f16x2` | MUFU 2 本 + PRMT | 同じ |
| 3 入力 `max.f32` | ptxas が拒否（sm_100 以上） | FMNMX3 |
| `fma.rn.f32x2` | ptxas が拒否（sm_100 以上） | FFMA2 |
| `cvt.rn.satfinite.e4m3x2.{f16x2,f32}` | どちらも F2FP 1 本 | 同じ |

2 要素を 1 命令で計算する MUFU は Hopper にも Blackwell にもない。インライン PTX は ptxas が SASS に変換するので、存在しない命令は出せない。10-26 計画 §4.5 の F1 はこの点で成り立たない。FA3 の定常ループの命令（MUFU 114、FMNMX 116、F2FP 56、FFMA 116、FADD 112、FMUL 66 / warp-tile）は sm_90a で出せる最少にほぼ一致する。10-26 計画 §4.3 の「ALU に PRMT ほか 122」は誤りで、定常ループの PRMT は 1 本である（`permute_Cregs_fp8` はレジスタの付け替えで済んでいる）。

## 5. Decode の kernel 内 merge（D2）：原因未特定

10-25 の 6 候補が全 rank で同じ相対 L2 0.7315 で失敗した件（40201493911）。CPU で両経路の添字を模擬すると 24 ケースすべて一致し（量子化前の相対 L2 0、FP8 code・scale の差 0）、5 つの制御フロー仮説（動的 split 数、batch 番号の空間、split 数の符号化、FP8 出力の上書き、到着カウンタ）もソース上ですべて否定された。原因は GPU 上でしか決まらないので、1 forward で separate combine と merge の途中値を段階ごとに記録する玄界 1 GPU の診断 probe を用意する。

## 6. 次の方向

理論分析（kernel と系全体）の結論では、今の Decode 容量（SLA 内 約 49 req/s）が score を 9,500〜9,600 で止める。10,000 には Prefill −22 ms/反復と Decode step −3.6% の両方、またはこの釣り合いを動かす系全体の変更が要る。系全体の手（投機デコードの受理率、Decode の ITL 分布の平準化、開始時の一斉投入、資源配分）の検討を進めている。
