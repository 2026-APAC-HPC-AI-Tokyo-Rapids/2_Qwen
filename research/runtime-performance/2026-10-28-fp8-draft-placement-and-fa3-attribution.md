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

## 7. 系全体の施策：立ち上がり整列と concurrency 215（serving [40630160781](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40630160781)）

系全体の分析（離散事象模型で実測の score・good 件数・ITL を ±0.3% で再現）で、次の 2 点が分かった。

- AIPerf の計測時間は「最初の送信から最後の応答終了まで」で、120 s ではない（40480817670 で 124.32 s）。定常の完了率 48.04 req/s が窓全体で続けば約 9,607 になる。差の 4.6% は開始時の TTFT 不合格（2.1%）と立ち上がり・末尾の drain（2.5%）である。
- c240 では定常で約 43 要求が Prefill の待ち行列に並ぶだけで、throughput に寄与しない。模型は c210〜220 で +0.3〜0.5%、全 rank の立ち上がりを揃えると +0.2〜0.7% とした。

実装（1d9ffa4 ほか）：

- `prefill-start-align.patch`：Prefill worker の全 TP rank が完全に空いた状態から要求が来たときだけ、全 DP rank に ready な要求が揃うまで最初の forward を最大 100 ms 待つ（開始時の 240 要求の送信幅は実測 97 ms）。rank の合意は CPU の Gloo all_reduce（32 B）で、CUDA の同期は増やさない。定常では待ち行列があるので発動しない。
- concurrency 215。提出パッケージは c240 のまま。
- Prefill の処理順は変えない。40313004834 の最初の 10 秒の 519 要求は、どの rank でも到着順に forward を始めていた（rank 内 FCFS）。

| run | 構成 | score | good | TTFT 不合格（すべて開始 10 s 以内） | ITL 不合格 | 計測時間 | 定常 TTFT p50 / p99 | ITL 平均 / p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 40480817670 | c240、FP8 draft | 9161.67 | 0.9780 | 122 | 6 | 124.32 s | 1.81 / 4.41 s | 15.87 / 18.70 ms |
| 40621030371 | 同上（routing 記録付き） | 9171.48 | 0.9777 | 121 | 9 | 124.27 s | 1.79 / 4.37 s | 15.89 / 18.87 ms |
| 40630160781 | c215 ＋ 立ち上がり整列 | **9235.28** | **0.9830** | **95** | 4 | 123.70 s | 1.26 / 3.73 s | 15.89 / 18.79 ms |

+0.7%（模型の見込み +0.5〜1.2% の範囲）。TTFT 不合格が 121 → 95、定常 TTFT p50 が 1.79 → 1.26 s（模型の c215 の予測 1.27 s）に下がった。ITL は変わらず、Decode 側への影響はない。単回の比較だが、改善は TTFT 不合格と待ち行列の短縮という機構どおりに現れている。

## 8. EAGLE3 draft の受理長（Firmus 診断 [40628856183](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40628856183)）

1 node、TP4 server 2 本を並列（PD なし、FP8 target・KV・draft、EAGLE3 3/1/4、CUDA graph なし）、公式の合成入力 128 要求（並列 32）。

| draft | revision | 受理長（bonus 込み） | 比 |
|---|---|---:|---:|
| nebius（現行） | ff0f0c09 | 2.251 | 1.00 |
| lmsys SpecForge-Meituan | 7f8040f1 | 1.860 | 0.83 |

Meituan 版は受理長が低く、draft の計算量も約 2 倍なので採らない。

RedHat の speculators 版（87cd7aa9）は `norm_before_residual: true`（最初の decoder layer で正規化後の target hidden を residual に使う）と独自の embed_tokens が固定版 SGLang と異なる。probe 専用の `eagle3-norm-before-residual.patch`（branch `perf/firmus-eagle-drafts-20261028`）で forward を合わせた（CPU で RedHat の layer 順序と一致、flag なしでは元と bit 一致）。aux 層は config が null で、speculators の学習データ生成（層 2・47・91 の出力）と vLLM 0.11.2 の推論（同じ層の入力）で 1 層ずれるため、両方を同じ条件で並べた（[40636912520](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40636912520)）。

| draft | SGLang aux ids | 受理長（bonus 込み） | nebius 比 |
|---|---|---:|---:|
| redhat-vllm | [1,46,90] | 1.945 | 0.86 |
| redhat-train | [2,47,91] | 2.015 | 0.90 |

学習側の流儀の方が高いが、どちらも nebius（2.251）を下回る。モデルカードの要約（CNN/DM、k=3）2.15 と近く、実装が大きく外れている兆候はない。**公開 draft 3 種はいずれも現行の nebius に及ばない**ので、draft は変えない。受理長を上げる残りの手段は自前の fine-tune だけで、締切までには現実的でない。

## 9. kernel 束（Prefill 4 段 pipeline、Decode 前処理・router gate）：すべて不採用

branch `perf/firmus-kernel-bundle-20261028`、image [40648657907](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40648657907)。どの候補も既定は無効で、kernel bench か job 内の選択に通ったものだけを有効にする作り。

**Prefill MegaMoE の 4 段 pipeline（`DG_SM90_PREFILL_STAGE4`）と komoriyuta の FC2 アドレス案**：kernel bench [40658033568](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40658033568)（MegaMoE 1 回の中央値、出力は 5 ケースとも control と一致）。

| ケース | control | 4 段 | FC2 アドレス |
|---|---:|---:|---:|
| balanced | 1.4757 ms | +2.6% | −0.1% |
| local | 1.2977 ms | −1.8% | −0.8% |
| real43 | 1.4296 ms | −1.8% | −0.8% |
| real62 | 1.4186 ms | −1.2% | −0.5% |
| real7 | 1.4332 ms | −0.7% | 0.0% |

4 段は実 routing 3 ケースのうち 2 つが採用下限 20 µs/層に届かず（10〜26 µs/層）、balanced では遅くなる。bench は 1.5〜2.0 GHz で、電力上限で約 1.5 GHz の serving ではさらに縮む見込み。どちらも採らない。

**Decode の MegaMoE 前処理の warp 集計（`DG_SM90_DECODE_PREPARE`）と router gate の 1 kernel 化（`SGLANG_DECODE_ROUTER=cluster`）**：serving job 内の選択（`select-firmus-decode-prologue.py`、94 層の router＋pre-dispatch＋MegaMoE を CUDA graph で交互に 15 回ずつ 2 block、4 rank の最大）。数値 gate は router（fp32 参照 相対 L2 ≤ 0.0017）・MegaMoE serving gate とも通った。

| rows | control（µs/層） | 候補 | 差 |
|---:|---:|---:|---:|
| 144 | 207.0 / 209.4 | 241.5 / 244.8 | +34〜35 |
| 152 | 209.8 / 210.6 | 245.5 / 247.2 | +36〜37 |
| 160 | 214.9 / 213.8 | 253.1 / 251.3 | +38 |
| 168 | 217.2 / 217.1 | 255.0 / 255.1 | +38 |
| 176 | 218.9 / 218.2 | 258.0 / 257.4 | +39 |

候補は 17% 遅く、選択は正しく棄却した（[40689226339](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/40689226339)）。router と前処理を束で測ったので、どちらが遅いかはこの計測では分けられない。選択スクリプトは最初の 2 回の run で動かなかった：同じ torchrun rendezvous で NCCL を再初期化して古い bootstrap id を読んだ（40660502072、gate を別 torchrun に分離）、bench の `routing()` の戻り値（ids, weights）をそのまま ids に渡した（40680955755、修正済み）。

**新 image での control の繰り返し**（3 回とも serving は現行 kernel、c215・立ち上がり整列・FP8 draft）：9204.30（40660502072）、9211.35（40680955755 attempt 2）、9229.18（40689226339）。旧 image 40197369083 の 9235.28 と差はなく、受理長も 2.208 で同じ。採用する変更がないので、この branch と image は main へ統合せず、main は image 40197369083 のままとする。
