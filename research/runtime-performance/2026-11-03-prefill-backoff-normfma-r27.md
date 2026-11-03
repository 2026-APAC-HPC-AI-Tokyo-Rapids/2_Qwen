# Round 27: Prefill global polling backoff / norm_quant square FMA

実装は `perf/pp4-r27-backoff-normfma-20261103`、base main `3f4dc0d`、実装 commit `d652575`。image pin は round25 の `a001f901` のまま。SSH・Actions 操作・GPU/Slurm 投入はしていない。以下の採用判定は将来の同一 serving job 内で行う。今回の CPU 証拠を GPU 性能・精度・通常 score の結果と扱わない。

## 調査した正本と比較の境界

指定された `codex-scratch/astra28/astra28-final.md` を全文確認した。併せて tracker/model JSON、round24 profile note と `pp4-profile-r24-prefill-summary-20261102.json`、round26 note・CPU/CUDA/snapshot/budget 証拠、remote r26 branch の gate/host/header/sbatch、main の round25/26 commit 本文を確認した。過去 Actions の実行経過は既に退避された r25/r26 の Slurm log・selection/rank JSON を読み、Actions API は操作していない。今回参照できる会話の正本はユーザー依頼と指定 astra の保存回答であり、未提供の会話履歴を取得したとは扱わない。

baseline header は base series と PP4 の既存 overlay を全て適用した `fbfaa6be207b26dd7127aae406204b44668b4341ff5ce9e2928510c932e57852`。r26 archive の control と一致する。swap tail は一切取り込まない。source は `sgl-deep-gemm 0.1.5` の host revision `1c8138a59b3df5dac50b76b730834ed57982fdfb`、現 image の CUDA は 13.0.88、今回 CPU compile は 13.4.92。異なる run の clock と kernel 時間を合成しない。

## A: global-memory polling の列挙と変更

`patches/deep-gemm-overlay/sm90-prefill-poll-backoff.patch` は PP4 の DeepGEMM overlay 最後に適用する。base series/header.sha256 は変更しない。対象は以下。行番号は変更前の上記 header と共通 header。

| 場所 | 待つ値と load | 今回の扱い |
|---|---|---|
| `impls/sm90_fp8_mega_moe.cuh:1172` A loader / `process_a_sfa_block` | L1 `get_l1_arrival_count_ptr()`、`ld_acq` | 対象 |
| 同 `:1178` A loader FC2 | L2 `get_l2_arrival_mask_ptr()` を uint32 counter として `ld_acq` | 対象。直後の `fence.proxy.async.global` を維持 |
| 同 `:1190` mask 版 | L2 uint64 mask、`ld_acq_gpu` | 対象。counter 版との選択条件を維持 |
| 同 `:1133/1136` `process_task_load` | L2 mask / L1 count の `ld_acq_gpu` / `ld_acq` | 同じ待ちを扱うが、この independent-WG Decode 分岐は Prefill macro では実行されない |
| `comm/barrier.cuh:35/41` `grid_sync` | completion bit の `ld_acq` do-while、通常/verbose 分岐 | 対象。old/new bit、scope、arrive、sync_scope を維持 |
| `scheduler/mega_moe.cuh:193` `fetch_expert_recv_count` | `get_expert_recv_count_sum_ptr()` の `ld_volatile`、上位32 bit の finalization count | 対象。lane ごとの count/下位32 bit と syncwarp を維持 |
| `comm/barrier.cuh:83` `nvlink_barrier` | system-scope signal の `ld_acq_sys` | 対象。timeout/phase/sign/counter を維持。採用対象の lean EP1 ではこの cross-rank 経路を省略済み |
| `impls/sm90_fp8_mega_moe.cuh:3482` Prefill self-signal | system-scope signal の `ld_acq_sys` | 対象。ただし今回の lean EP1 はこの非 lean 分岐を通らない |

scheduler の仕事割当・expert 遷移などのローカル loop は待ちの global load ではない。TMA/mbarrier `try_wait`、named barrier、ring の barrier、release/atomic、acquire の scope、phase、proxy fence は変更しない。

最初の check を即時実行し、最初の3回の失敗では sleep しない。4回目以降の失敗だけ、固定上限64 ns の `__nanosleep(64)` を入れる。counter は3で飽和し overflow しない。32〜256 ns の範囲の低めを選び、短い到着待ちに追加する遅延を制限しつつ反復 load/branch を減らす単一候補とした。1410 MHz で64 ns は約90 cycleという目安にすぎず、実際の suspend 時間・省電力 W・滞在時間は未測定。CTA の register/shared memory 解放、occupancy 改善は主張しない。

`patches/deep-gemm-host/sm90-prefill-poll-backoff.patch` は host build series 末尾。既存 `prefill_split_accum`（EP1、four-stage lean、M128/N256、2WG）でだけ `DG_SM90_PREFILL_POLL_BACKOFF=64` を読み、JIT TU の include 前に macro を追加する。Decode では env があっても macro は出ない。flag 無しは retry state/本体が preprocessor で消える。off の generated source も従来と同一。

## B: square FMA の範囲

`patches/sglang/prefill-norm-square-fma.patch` を既存 `prefill-fp8-integration.patch` の直後に加え、同 patch が作る `add_rmsnorm_group_quant.py` の `norm_quant` を変更する。既存 patch 自体は過去の選択記録の SHA 照合対象なので変更せず、記録の来歴も書き換えない。probe `kernels/probes/norm-quant.py` は同じ候補に更新した。

`SGLANG_PREFILL_NORM_SQUARE_FMA=1` のときだけ constexpr `SQUARE_FMA=True`。各 thread の16要素を静的 split/reshape で取り出し、最初の平方は FMUL、残り15要素を明示 `tl.fma(value,value,sum)` で累積し、256 lane の reduction へ渡す。一般 `tl.gather` は初回の CPU compile で共有メモリ往復を増やしたため、最終版では使わない。FMA 以外の fusion は有効にせず `enable_fp_fusion=False` を維持する。BF16 residual store、未丸め FP32 h、eps/rsqrt、weight multiply、BF16 y への丸め、E4M3/scale/column-major layout は現行と同じ。Decode の `_post_norm_moe_input` は変更しない。flag は Prefill の env にだけ配備する。

## CPU の証拠

### A

`prefill-backoff-r27-cuda-validation-20261103.json` は CUDA 13.4.92 / sm_90a / O3、r26 と同じ24 instantiation の before/off を再 compile した結果。nvdisasm `-hex` の instruction/control word を byte 列に戻して比較し、**24/24 が同一**。Decode・EP4・以前の Prefill geometry も含む。候補は現 serving specialization のみ compile し、spill store/load **0/0**、ptxas launch-bound register **168**、role の `setmaxnreg` を含む最大参照 register **R212**（control R206）、SASS encoding **157696 → 166400 byte**、C75xx 警告なし。dynamic shared は従来の228560 byte。baseline の一部 geometry に既存 spill があることを候補の spill と混同しない。

`prefill-backoff-r27-host-validation-20261103.json` は実 host selector/Args/JIT text を g++ で検証した112条件。flag の読み取り対象になった2条件だけ macro が加わり、それ以外は generated source/descriptor/config が byte 同一。GPU、driver descriptor encoding、実 host extension の build/link/load を実行した証拠ではない。CUTLASS/CUTE include は r26 archive と既存 uv cache の同一入力を使用。実 image CUDA13.0.88 での fresh build と GPU load は投入側で必要。

### B

`prefill-norm-fma-r27-triton-validation-20261103.json` はローカル Triton **3.6.0** を GPU 無しで `ASTSource` / `GPUTarget(cuda,90,32)`、8 warps、fusion false、CUDA13.4 ptxas で compile。flag off の SASS は旧版と byte 同一。before/candidate は REG **64/64**、LOCAL **0/0**、LDL/STL **0/0**、shared **1024/1024 B**。FFMA **92→107**、FMUL **81→66**、FADD **58→43**で算術15命令/threadを削減。encoding は alignment を含め **15872→15616 B**。spill の判定は cuobjdump LOCAL と SASS local access の不存在であり、GPU driver による load は未実行。実 image と Triton/compiler が同じという保証に昇格させない。

`uv run --no-project --python 3.12 --with numpy python scripts/check-prefill-norm-fma-cpu.py --output research/runtime-performance/prefill-norm-fma-r27-cpu-validation-20261103.json` を実行。実 shape **7988×4096**、極値、相殺、ゼロ、全 finite FP8 code の中点と隣接値を含む。FP32 product を FP64 で保持して最後に一度だけ FP32 へ丸める FMA 模型と、独立 FP64 正規化/FP8 code 参照を比較した。NumPy の sqrt、16 strided 値/lane と binary cross-lane reduction を使う模型で、GPU rsqrt/compiled execution の証拠ではない。

| case | control/candidate 最大正規化誤差 | 最大 dequant 誤差 | FP8 code 不一致 control/candidate |
|---|---:|---:|---:|
| 実7988×4096 | 9.754769e-7 / 同値 | 0.186267636 / 同値 | **11664 / 11665** |
| 極値/相殺 | 5.743011e-7 / 同値 | 0.188754482 / 同値 | 200 / 200 |
| 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| FP8境界 | 7.412488e-7 / 同値 | 0.178741323 / 同値 | 0 / 0 |

**この CPU 模型では、厳密な code 非退行（候補 ≤ 現行）は満たさない。** 最大誤差は同じで、増えたのは3270万個のうち1個である。FP32 の演算順が変わると、FP8 の丸めの境目にある code がどちらの向きにも少数だけ入れ替わる。精度方針は FP の順序の違いを許しているので、Claude の投入前の見直しで、GPU gate の code 数の許容を `control + max(8, control // 1000)` に変えた（最大誤差は厳密な ≤ のまま）。この許容を越える増加は不採用にする。

## Gate、独立採否、予算

`scripts/select-firmus-prefill-r27.py` を3 worker ×4 GPUで同時実行。greedy gate の後、model startup の前。各 rank は短い norm gate を先に行い、同じ resident process で MegaMoE gate に進む。候補の例外は失敗 rank JSON に残し他方を試す。採用条件は候補ごとに12個の一意なH200 UUID、allocation/run id、image/source hash、deadline、healthy の全件確認。global worker exit が失敗でも、他候補の12個の完了済み証拠が揃えばその候補は独立して判定する。片方の採否を前提にしない。

- A は r26 の44 routing case と最終 BF16 output、FC1 FP8・FP32 scale、FC2 BF16 partial を byte 比較。pool の非決定順序は metadata から source token/top-k 順へ戻す。通常の serving API で1200回、2 input version・2 workspace、変化入力の最後の出力と元入力復帰後の全4 component を検証する。historical/balanced/hot は control→candidate と candidate→control の両順各12 sample。**全GPUの historical 中央値で max(5 µs, 0.5%) 以上短縮**し、**hot/balanced は両順とも遅くならない**ことを要求する。historical の margin は r26 のまま、他2 routing の非退行は許容遅延ゼロ。timing に snapshot/copy は含めない。
- B は7988/7989/3994/2049行に加え zero/extreme/FP8-boundary を検証。独立 FP64 正規化から BF16・FP8 code を求め、出力の FP8×scale 対 FP64 正規化の要素最大絶対誤差と FP8 code 不一致数が、case ごとに control 以下であること。residual は control/FP64 sum の BF16 と byte 一致。2入力/2出力 buffer の1200回と復帰で candidate の再現性を確認する。速度は全12GPU×4実 shape×両順各12 sample の中央値で **max(0.5 µs, 0.5%)**以上短縮。60 µs 級 norm に5 µsは条件付き利得6 µsの大半を要求するため、event 分解能より十分大きい0.5 µsを下限とする。CUDA graph の1 sampleは reset copyを含む8 call平均で、両版で同じresetを行う。copy を含むため通常servingのkernel単体時間とは区別する。
- 不一致、失敗、欠測、時間不足はその候補の flag を0のまま保つ。初期 env は両方0、採用時だけ候補別に書き換える。更新は temporary env から atomic replaceし、集約の kill 後も sbatch で temporary を削除する。元の gate/serving を停止する処理は追加していない。

r26 raw 12 rank の再読込では run_id `28515-1793697318` → 最終 `1793697370.320553`、wall **52.321 s**、rank **17.285〜17.888 s**。この実測から worker **60 s**（kill grace2 s）、集約まで **70 s**を予約する。B の cold JIT を含めた r27 wall は未測定であり、上限超過なら採用せず serving へ進む。

開始条件は厳密に `SECONDS + 100 + 660 <= 1020`、すなわち job 経過260秒以下（worker 90秒・集約100秒。2 gate と Triton の cold compile を 1 container で行うので、r26 の 52秒に余裕を持たせた）。greedy の元の90+660秒条件は変更しない。r25 の最後の greedy は222〜236秒なので、その後に開始できる。236秒なら gate は最大336秒、serving reserve の終端996秒、残り24秒。r26 の125秒を greedy の前に置いた構成は再利用しない。順序・時刻式・260/261秒境界・独立失敗/欠測/timeout をCPU testで固定。詳細は `prefill-r27-budget-20261103.json`。

## Dispatch、build、投入側で変える箇所

A host は実際に flag を選んだ process ごとに1回、`DeepGEMM sm90 Prefill polling backoff enabled: ns=64` を出す。B は初回の eligible wrapper 呼出で `Prefill norm_quant square FMA enabled: explicit 16-term accumulation` を出し、lru_cacheで以後の重複を抑える。sbatch は通常 serving の `prefill-{0,1,2}.log` だけから抽出し、`results/prefill-r27-dispatch-{worker}.txt` に保存する。gate log の使用表示を serving 使用の証拠にしない。**今回は未投入なので実 dispatch 記録はない。**

B は `scripts/stage-qwen-overlays.sh` の provisioning overlay（実 image の package をコピーして順に git apply、Enroot mount）であり、その変更だけなら既存 image を使える。A の header も同じ overlay、host は `containers/Dockerfile` → `containers/rebuild-deepgemm-host.py` → host series で image build 中に再構築されるので、今回まとめて **新規 build 必須**。

Claude の投入 commit では先頭の skip-ci を外し、require-build と Firmus cluster/node tag を付けて新 image を build する。config の2つの gate key は既に true、image pin は今回変更していない。provisioning が fresh image の path/SHA/size/source を runtime config に差し替える既存経路を利用する。candidate env を手動で1/64へ pinしない。GPU gate terminal状態、候補ごとのselection.json、serving dispatch、公式 parser の score/eligible/error、ログ・成果物を確認してから採否を記録する。同条件 repeat の通常 serving がない段階では性能優位や main merge を主張しない。

## 理論上の仕事量と条件付き見込み

A の代表 shape は T7988/H4096/I1536/E128/top8、BF16 input/output、FP8 weight/FC1、FP32 scale・carried accumulation、BF16 FC2 partial。FC1+FC2 有効計算 **2.412295225344 TFLOP/層**、FP8 weight **2.416 GB**、dispatch pool **261.75 MB**、FC2 partial **523.50 MB**、最終 BF16 **65.44 MB**の論理量は不変。EP1 の expert 間 NVLink はないが PP/KV転送は別。global poll の反復数・実HBM sector/byte・省電力Wは未確定で、ゼロで埋めない。減らすのは失敗した polling の load/branch 発行頻度のみ。mbarrier の待ち時間を削減可能量に数えない。

B は N=32718848要素、16平方+15加算と1平方+15FMAは共に31 FLOP/16要素で、算術FLOPは同じ。residual add・norm/weight・quant multiply・scale/divを含む単純演算数は約 **0.196832308 GFLOP/呼出**（rsqrt/比較/変換・div内部命令を同一のthroughputとは扱わない）。2本のBF16読出し、BF16 residual書出し、FP8出力、FP32 scale、8 KiB weight の最小論理量は約 **230.063 MB/呼出**。実HBM byteは未測定、今回その量を減らさない。減らす仕事は約15命令/thread、約30.7M lane命令/呼出。

r24 の norm 呼出は stage ごと21〜24回、約1.36〜1.50 ms/周期。既存の約 **0.15 ms/周期・score +13**は条件付き見積もりを維持する。power軸で追加上積みを足さない。A に正の時間/W見積もりはない。swap-tail、A gather、direct FC2、shared reciprocal、その他過去の棄却は覆していない。B の数値非退行は今回のCPU模型でも未達であり、小さい利得を採用可能性と混同しない。

## 検証と片付け

要求された repo root からの全体検証を terminal まで確認した。

- `uv run --project qwen-experiment-metrics --with 'numpy>=1.26,<3' --with pytest python -m pytest -q -p no:cacheprovider qwen-experiment-metrics/tests`：**終了コード0、1975 passed / 90 skipped、計2065件**。CPython3.13.12 / NumPy2.5.4 / pytest9.1.1。skipをGPU成功と扱わない。
- `uv run --project qwen-experiment-metrics ruff check qwen-experiment-metrics scripts containers`：**終了コード0**。
- `bash -n`（sbatch / provision）、`git diff --check`、全r27証拠JSONのsource SHA照合：終了コード0。
- 初回の全体pytestは終了コード1、旧env/overlay順序・配備記述に関する期待値9件を修正した。配備の動的loopは既存の配備検査でも確認できる明示install/cpへ揃えた。途中の旧版再実行は修正後に中断し終了コード2。その後、上記の全体を新たに実行して0を確認した。初回/中断をpass件数に混ぜない。

対象testは候補ごとの独立採否、12GPU fixture、byte/component欠落、FP64誤差、両順全shape、greedy順/予算、config、off SASS、once log、実prepare/provisionを含む。最終のraw log/exitは指定scratchの `pytest-complete.log` / `pytest-complete.exit`、`ruff-final.log` / `ruff-final.exit`、集約は `test-results.json`。

GPU測定・scoreは未実施。新しいPJM/Slurm/Actionsジョブは作っていない。他 branch/main/mirror や既存ジョブは変更・停止していない。CPU用 source/cubin/cache は指定 `codex-scratch/r27` 内で作成し、raw source/cubin/SASS/ptxas log/再現scriptを `round27-cpu-proof-20261103.tar.gz` に保存した。SHA256 は `d789af6ef07705552b6c28b6faa053d78ed0e0b5bb9c10c37a910a321a54604a`、tar生成・gzip検査とも終了コード0。作業用の展開 source/cubin/cache、診断/編集補助script、今回生成したworktreeのvenv/bytecode/cacheを削除し、archive・検証log・終了コードを残した。`kernel-theory-model.json` は変更していない。

## GPU結果（run 41622902468、Slurm 28534、commit e5aa4f0）

- **image：** fresh build `d3cd43073ae4d61d2b6eb000bc475e94e282b0c524268e4c282bb55449e0610d`（未pin）。
- **予算：** Decode greedy gateが先に終わり（rank 0の終了はepoch 1793709028、r27 gateの開始は1793709061）、採用された。servingでも4 rankが融合kernelを使った。r27 gateはその後に始まり、jobは910秒、READYは614秒で収まった。
- **A（polling backoff 64 ns）：** 12 GPUのすべてで、全case・中間のbyte一致と1200回の耐久は合格した。しかし全GPUで遅くなった：

| case | 遅くなった幅（12 GPU） |
|---|---|
| historical | +1.10〜+2.26% |
| hot | +1.01〜+2.17% |
| balanced | +0.55〜+2.54% |

  待ちの多くは短く、64 nsの睡眠が到着の検知を遅らせる。そのコストが、電力の節約による clock の上昇を上回った。不採用。
- **B（norm二乗和のFMA）：** 11 GPUで精度は非退行だった。最大誤差は同じかそれ以下で、7988行のFP8 code数は現行に対して−1〜+9（許容の範囲内）。しかし全shapeで遅くなった：

| 行数 | 遅くなった幅 |
|---|---|
| 7988 | +0.64〜+0.76% |
| 7989 | +1.2〜+1.5% |
| 3994 | +1.35〜+1.62% |
| 2049 | +3.3〜+4.3% |

  命令は減ったが、Tritonの`tl.fma`の累積でloopが直列の依存鎖になり、命令の並列度が落ちたとみる（SASSでの確認はしていない）。不採用。
- **g1のtimeout：** prefill-0 rank 3のnorm gateで、home（BeeGFS）上のTriton cacheの`__triton_launcher...so`が `ImportError`（ENOENT）になった。そのworkerは90秒で打ち切られ、`workers_ok=false` になった。速度の結果が明確なので採否には影響しないが、gateでTritonを使うなら、`TRITON_CACHE_DIR` をnode-localにする必要がある（numeric gateで以前に起きた問題と同じ）。
- **serving：** 現行の経路（greedy融合あり）で、公式score **10021.06**（eligible、good 98.20%）。round 25の10021.60と同じ構成で、ほぼ同じ値になった。
- **結論：** 電力の観点から挙がった2候補は、どちらも実時間を悪化させた。電力上限下でも、待ちの電力の節約や命令数の削減は clock の上昇として現れず、latency と依存鎖の悪化が勝った。branchはremoteに残し、mainには文書だけを記録する。
