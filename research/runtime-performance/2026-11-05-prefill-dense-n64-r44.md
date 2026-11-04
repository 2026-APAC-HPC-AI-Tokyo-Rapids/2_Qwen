# Round 44：Prefill dense N64 二重 temporary と promotion 露出上限

**submission-ready の CPU 実装。GPU 未実行、候補未採用、次回は require-build が必要。** 親 `e2f8689`、branch `perf/pp4-r40-megamoe-mainloop-20261104`。commit、push、job 投入・停止、image build、実装準備時の SSH は行っていない（後述の M 再調査では read-only SSH を使用）。image pin と `submission/` は保持した。

r43 の NO-GO は promotion の露出が小さいと測定した結論ではない。r43 の126行目に残る正の余地を CPU だけで確定できないため、親の明示判断により、**一つの通常 serving job 内で候補自体と診断上限を測る**段階へ進めた。今回の採用閾値はユーザー指定の QKV＋O **10 µs/層以上**。r43 の25 µs目標や、MegaMoE r41 の77.16 µs上限を新しい dense の実測値に流用しない。

## 入力と不変条件

AGENTS.md、r43 note/proof、astra43 report の dense 節、r41 note/scripts/host・device patch、r37 の snapshot・gate・prewarm と r38/r39 の失敗修正、保存会話・commit 本文を確認した。tracker、theory model、入力 profile JSON、r24 集計、r43 に保存された Actions/Slurm の来歴も確認した。後から見つかった C7514 の原因は `2026-10-18-megamoe-decode-mainloop-pipeline.md` と `2026-10-30-prefill-megamoe-split-accumulator.md` の既観測と一致する。

対照は DeepGEMM `1c8138a59b3df5dac50b76b730834ed57982fdfb`、SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4`、現 image `2fb619c4a293a01b98ecd178d92414ab5cf493e2584acd945a1a8910ba3b4467` の serving 設定。これは次回 build 後の image hash ではない。r43 保存 header `915bd4d3ee23e0ce6da735cd6b273700caa0d94a15a6917fb6eed93a4ac72350` を基準にした。

- QKV `M7988/N9216/K4096`、O `M7988/N4096/K8192`（通常 profiling の主計時）、single は M3994。warmup は M4001/4002、pair は M8002〜8004。数値 gate は M7988/8003/3994、Normal、compiled_dims=nk、E4M3 A/B、FP32 K128 activation scale と128×128 weight scale、FP32 promotion、BF16 output。
- BM256/BN128/BK128、128 TMA threads＋256 math threads、3 stage、cluster2 の A multicast、132 persistent CTA、元の scheduler 順序、全 K 積、first-K peel を維持する。
- full の期待50176 B/count1、empty の2 CTA×8 math warp=16 arrival、peer/self の条件、TMA、epilogue は維持する。新しい named barrier は追加しない。

## 候補と実際の発行鎖

`sm90-prefill-dense-n64-r44.patch` を host series と PP4 device overlay に追加した。host JIT generator が新 flag `DG_SM90_PREFILL_DENSE_N64` を毎 launch/capture 時に読む。既定0。二つの N/K、元 geometry、Normal、nk、K-major B scale、Identity epilogue だけに候補 define を付ける。M は runtime dimension なので候補 eligibility へ含めない。対象外は元 source。**flag off の生成 source は元と全 byte 一致**する実 C++ generator テストを持つ。既存 first-K flag は1のまま。

各 WG の M64 wave に対する N128 を N64×2 に分ける。temporary は各32 registers、最終和は128 registers/thread。配列容量の合計192は元と同じだが、補助値の live range は増える。N64/N128 の CUTLASS CLayout と row scale lane を全8192対応で照合した。

A=(M0,Nlo)、B=(M0,Nhi)、C=(M1,Nlo)、D=(M1,Nhi) として、8 K128 の chunk 内は次の鎖になる。

```text
full(k); issue A(T0); issue B(T1)
wait1; promote A; issue C(T0)
wait1; promote B; issue D(T1)
wait1; promote C
advance stage; full(k+1); load next scales; issue A_next(T0)
wait1; release old stage; promote D; issue B_next(T1)
...
chunk last: wait0; release old stage; promote D; next chunk
```

旧 stage を保存し、D の完了後だけ旧 empty へ arrive する。次 K の scale で旧 M1 の scale を上書きしない。pending な他 temporary に compiler operand fence を掛けない。出力要素ごとの K128 順、各 K128 内の K32×4、FP32 scale 積、`__fmaf_rn(scale, product, positive_zero)` の初回加算を維持する。

最初の cross-backedge 案は C7515/C7517 と80/52 B spill、最終 K を分離した案は spill0 だが C7514 が全 QGMMA を wait0 で直列化した。**pending な temporary を loop backedge へ持ち越さず、8 K128 ごとに drain** することで解消した。この drain は無料ではなく、候補全体の gate 時間に含まれる。棄却案の source/log は proof 内だけに保存し、runtime へ残さない。

fresh CUDA13.4.92、sm_90a、O3/C++20、extended lambda/relaxed constexpr の結果。168は launch metadata、math role の budget は232。現 image の CUDA13.0.88 での GPU 実行の証明ではない。

| 項目 | QKV production / N64 / 診断 | O production / N64 / 診断 |
|---|---:|---:|
| launch registers | 168 / 168 / 168 | 168 / 168 / 168 |
| 最大参照 GPR | R202 / R227 / R151 | R202 / R225 / R151 |
| stack・spill store・spill load | 全て0 | 全て0 |
| LDL/STL、C75xx | 全てなし | 全てなし |
| 全 kernel 静的 QGMMA | 128 / 256 / 64 | 128 / 256 / 64 |
| wait0 / wait1 | 32/0、2/62、1/8 | 同左 |
| promotion FFMA | 2048 / 2048 / 0 | 同左 |
| encoded SASS byte | 62208 / 71040 / 19840 | 61440 / 70400 / 19200 |

production/off の encoded SASS は r43 の QKV `9630609e…`、O `11f685d4…` と一致した。N64 は QKV `0a13568e…`、O `63e98594…`。対照の静的2 chunk に対し、候補の各 chunk は QGMMA128、wait1 31、wait0 1、FFMA1024。QKV の動的 chunk は4、Oは8。実 PC と前後の発行鎖を proof の `*-candidate-chains.json` に保存した。静的 PC 幅を latency へ換算しない。

同じ M で比較した production/candidate の仕事量は不変。通常 profiling の代表 pair M7988 で有効 FLOP は QKV603073806336、O536065605632、合計1139139411968/層。論理一回通過 I/O は218734080/166482944 B（activation/weight、BF16 output、FP32 scales の論理payload）。この M7988 は後述の phase 別調査で再確認した。warmup M8002 の模型とは区別する。N64 の operand SMEM 読出し模型は96→128 KiB/CTA-K128、QKV +2415919104 B、O +2147483648 B。QGMMA 発行数は倍だが FLOP は不変。実 HBM/L2/NVLink byte、GPU 時間、score、正味節約は未測定。既棄却の tail skip、retiling、epilogue 融合、stagger、stage4、FC1 候補へは広げない。

## 診断上限の隔離

新 TVM-FFI `fp8_gemm_nt_promotion_diagnostic` だけが内部 `DiagnosticNoPromotion=true` を渡す。通常 GEMM の SM90 1D2D dispatch は literal false。診断選択の env は無く、SGLang/Python serving wrapper から別 entry を参照しない。import 前に診断 macro の環境変数混入を値0でも拒否する。builder の ABI import 検査も新 entry を確認する。

診断だけが `sm90_fp8_gemm_promotion_diagnostic_impl` を instantiate する。FP32 tensor-core accum128 を K 全体へ持ち越し、per-K scale load・scale multiply・promotion FFMA を使わない。両 M64 wave を1 group とし、wait1 後に前 stage を解放する。**定常 K loop に wait0 は無い**。tile 最後の出力と stage 解放には唯一の wait0 drain が必要。FFMA は全 kernel で0。

出力は数値的に誤っており、タイミング測定専用。production minus diagnostic は scale/promotion、group/wait、register pressure の解消を含む**promotion 経路全体の楽観上限**であり、FFMA 単体の露出や N64 の実現可能な節約ではない。同じ resident GPU・shape・入力・image・clock 窓で採る。r41 の MegaMoE 上限を転記しない。

## bounded gate と数値

旧 r41 の scripts、host/device diagnostic patches、専用 test/fixture、staging を削除した。研究 note、過去の validation/proof/result は保持する。同じ Decode greedy 後・Prefill launch 前の窓で、3 worker を同時起動し、各 torchrun 4 process、計12 GPU。head 2 worker、tail 1 worker。BigQuery 資格情報を追加せず、共有 results directory に raw JSON/clock と、node-local JIT cache の保存copyを出す。

checkpoint は最初の dense layer の Q/K/V/O weight と scale だけを safetensors から読む。r37 の `dense_snapshot` を再利用する形で移植し、FP8 weight と**FP32 scale parameter の empty+copy**、dim0 の Q/K/V 順を再現する。BF16 checkpoint scale をそのまま渡して失敗した r37 の問題を戻さない。tensor 名・dtype・shape・byte hash と snapshot 全体の hash を記録する。

activation は既存 `prefill_r37.Dense` の決定 seed、BF16 N(0,0.25)、実 production の K128 quantizer。主計時は7988行を独立3994×2として記録する。single M3994 と partial-tile M8003 は全出力数値検査だけを行う。warmup専用の4001/8002は8003とgeneric code pathを共有し、今回のGPU数値gateから削除した。GEMM は row-local で、attention を追加実行する probe ではない。JIT、3 warmup、4 call/graph の capture/replay は計時外。

全出力の候補/production の **bit一致を必須**とする。bitが異なる場合は下記参照の誤差範囲内でも不採用。独立参照は DeepGEMM を使わず、TF32 無効・FP32 K128 matmul と scale promotion から作る。FP8 operand と scale 粒度は同じなので、FP8 量子化誤差を tolerance に加えない。FP32 K128 内の和順序の差と最終 BF16 丸めだけを許す。

```text
n = 2 * (128 + 2 + K/128), u = 2^-24
allowance = n*u/(1-n*u) * sum(abs(scaled_products))
            + abs(fp32_reference)/256 + 2^-126
```

両 FP32 評価の誤差を含めるため n に2を掛ける。candidate/production と candidate/reference の max absolute/relative error、reference 検査、bit 検査を記録する。relative の分母下限は2^-24。非有限値は失敗とし、error metric は null にして診断結果の保存を妨げない。production 自身も独立参照へ同じ allowance で照合する。非 bit 一致の場合の production/candidate 差は、両 BF16 結果を含める2×allowance以内を要求し、comparison_mode と判定を保存する。bit 一致した場合の後続耐久検査は、候補自身の初回 byte との一致により production との強い条件も維持する。

各 shape で専用1-call graph を500回連続 replay し、毎回全出力 byte を候補自身の初回出力と比較する。GPU checker の結果を累積し、500回後に同期する。計時は別の4-call graph、forward production→candidate、reverse candidate→productionを4 roundずつ交互に実行する。続いて別のproduction→diagnostic/逆順を同数計時する。診断上限にもその窓のproduction sampleを使い、候補計時と別窓を合成しない。QKV/O の対応 sample の節約を先に加算し、各 GPU の8差の中央値、さらに12 GPU中央値を計算する。SM clock はcandidate計時前からdiagnostic計時終了まで、GPU UUID 指定で100 msごとに採る。このCSVは間のdiagnostic JIT待ちも含み、平均clockを各kernelの実効clockや他runへの換算に使わない。

全12 GPUの来歴・numerics・500回耐久・clock・round・source identity が揃い、QKV＋O 節約中央値が10 µs以上のときだけ adopted=true。境界10は採用、9.999は不採用。欠測、期限超過、不健全は採用しない。全workerの終了0、全12rankの正常fast-exit markerとcleanup完了、期限内の完全JSONを要求する。timeoutや非0終了を公開済みJSONで救済する規則は撤去した。exit codeも保存する。

`results/prefill-dense-r44/summary.json` に全 rank の raw numbers、全 numerics、耐久、両順 sample、clock、JIT identity と診断上限を含める。数値検査で不採用でも、全 timing が揃えば12 GPU診断上限を保持する。summary complete/adopted=true と bounded supervisor の成功の両方を確認した sbatch だけが、`source/prefill-kernels.env` の候補 flag を0→1へ変更する。timeout/skipped/失敗は production のまま。

採用時は全 Prefill process が model load/transform 後、readiness 前に実 QKV/O weight・M7988 と M3994 の両 shape を prewarm・同期し、done を記録する。実 forward の候補 dispatch 後に、eligible N/K の最初の実 M を process 一度だけ log する。M7988 等の行数条件は付けない。aiperf 後、r37 A/B/C とともに `results/prefill-dense-r44/dispatch-0/1/2.txt` へ回収する。

## 予算と検証の限界

開始条件は厳密に `SECONDS + 120 + 630 <= 1020`。最後の開始は270秒。worker110秒、TERM後2秒kill grace、全 launch/wait/aggregate は120秒 SIGKILL process-group timeout。r41相当246秒開始なら996秒、257秒なら1007秒、270秒なら1020秒。630秒のserving reserveは維持する。r41のserving steps .12〜.16は544秒で、このreserveには86秒の余裕があった。未来のserving時間を保証する値ではない。

優先順は M7988/8003/3994 の両projection全出力bit＋独立FP32参照、M7988各500callの耐久、M7988 production対candidateの正逆計時、最後にno-promotion診断の独立正逆計時。single-M計時、warmup専用M4001/8002の数値参照を削除し、各順6→4round・4call/graphとした。参照対象のM合計は31988→19985（37.5%削減）、計時kernel数はpair/single合計576→candidate128＋diagnostic128=256（55.6%削減）。候補の要素ごとの和順序は維持するため、FP32誤差内なら非bit一致を採用する経路も撤去する。

DeepGEMM pinned sourceの `csrc/jit/compiler.hpp` は `DG_JIT_CACHE_DIR`、runtime/disk cache hit、UUID temporary directoryから完成directoryのatomic renameを既に実装する。共有cacheだけではcold miss競合による重複compileを防がない。そこで**worker containerごと**にfresh `/tmp/prefill-dense-r44.XXXXXXXX` を作り、その4rankで共有する。rank0がQKV、rank1がOをcompileし、CPU Gloo barrier後に他rankが既存disk cacheを読む。まずproduction/candidate4kernel、優先作業後にdiagnostic2kernel。owner以外のcheckpoint読込と一部重なる。compiler、lock、retry、cache publish機構は追加しない。HEADの別container間ではcacheを共有しないのでnode全体一回とは主張しないが、従来の24dense compile/workerを6へ、HEAD48を12へ削減する。`/scratch`はBeeGFS共有、`/tmp`はcontainer側local pathでmountsの共有scratch/homeとは別。cache filesystem型をworker logへ記録し、GPUジョブでの実filesystem確認は未実施。全rankのsource/cubin identityを照合、rank0が共有resultsへ6kernelをcopy後、wrapperのEXIT trapでlocal cacheを削除する。

既存CUDA13.4の6 TU実時間合計42.409秒を2shape ownerで分担するとJIT critical pathはphaseごとのowner最大値の和22.189秒（HEAD条件込みで約21〜25秒の条件付き想定）。import/torchrun/実weight読込を約8〜12秒、削減後の3M参照・耐久・capture・256call計時・identity/copy/cleanupを約20〜23秒と置き、**torchrun開始から約50〜60秒を目標**とする。後二項とimage CUDA13.0/HEAD同時4compilerの実時間は未測定。r41のHEAD container/torchrun起動約35秒を別に加えるとlaunchから約85〜95秒、worker110秒まで15〜25秒の想定余裕。旧70〜90秒見積は起動費を含まず、105〜125秒になり得るため撤回した。rank JSONとflush付きworker logはtorchrun開始基準の各phase秒数を保存し、次回の見積検証に使う。これは保証やGPU性能結果ではない。

### r41 teardown 調査

read-only `ssh firmus` の sacct と既存archiveを照合した。Slurm28739は17:28:51 UTC開始、probeのrun_idは1793813577=17:32:57、HEAD .9/.10は95/94秒で**COMPLETED、ExitCode 0:0**、TAIL .11は56秒。HEAD worker0のrank0開始はlaunch+35.678秒、clock CSV最終sampleは17:33:46.675=launch+49.675秒。HEAD .9終了17:34:32まで約45秒が残る。しかしtimeoutのsignal/killやfatal errorはなく、95秒limitでkillされた証拠はない。sacctの終了0を失敗へ読み替えない。

r41 sourceはCUDA同期・graph reset・barrier・DLPack全view解放・symm buffer destroy・EP1破棄・barrier・WORLD破棄後に最終JSONをwriteする。ログのmulticast export警告は全workerのsetup時に出たgraceful skipで、teardown failureの根拠ではない。cleanupの各時刻、main復帰/torchrun終了/srun完了時刻は未記録であり、残り45秒をsymm memory/NCCL destructor、Python/CUDA静的destructor、srun/container step completionのいずれかへ特定できない。archiveのmtimeは回収時刻なので代用しない。

r44 denseにはsymm bufferもEP1/NCCL groupも作らず、共有cache調整用Glooだけを使う。CUDA同期→全graph reset→barrier→cache退避→barrier→`destroy_process_group()`を正常pathとし、その完了後だけcomplete JSONをatomic writeしてstdout/stderrをflushし、**`os._exit(0)`を通常の終了方法**とする。interpreter/CUDA静的destructorの待ちを回避し、失敗後のfallbackとしては使わない。cleanup例外、期限超過、非0worker、12GPU中一つでも欠測なら不採用。CPU subprocessテストで結果/flush/exit0とatexit非実行、cleanup順と失敗時非公開を検証する。Slurm自身のstep終了遅延が残るかは次の許可されたE2Eでのみ確認可能。

CPU emulation は M256/N128 の1 CTA、2 WG×2 M64、K4096/K8192 の全 K、各32768出力で production と候補の FP32 bit が一致した。E4M3で表せる試料とFP32 scaleを用いる。これは実 checkpoint の全 runtime M における GPU output、FTZ/NaN、WGMMA hardware の数値証明ではない。GPUでのbit判定・独立FP32参照・耐久・clock・採否・通常serving公式parserは次回の一つの job に残る。通常 score の優位性も一回だけでは結論しない。

CPU source fixture を使う tests は git history や外部 mirror を必要としない。generator off byte identity・on guard・診断隔離・実scale変換signatureのC++検査、全12rank採否と閾値/欠測/期限、正逆対応集計、並列 supervisor、staging/回収、FP32 scale copy、実参照関数の NumPy double、prewarm/dispatch once を検証する。最終の指定 pytest/Ruff、archive hash、後片付けの結果は隣接 `prefill-dense-n64-r44-cpu-validation-20261105.json` に記録する。

変更箇所は次のとおり。r41 の実行用ファイルは削除、historical note/proof は保持する。

| 経路 | 主なファイル |
|---|---|
| host/device/serving | `patches/deep-gemm-host/series`、両 `sm90-prefill-dense-n64-r44.patch`、`patches/sglang/prefill-dense-n64-r44.patch` |
| build/配備 | `containers/rebuild-deepgemm-host.py`、`configs/firmus-prefill-pp4.json`、`scripts/provision-firmus-framework.sh`、`scripts/prepare-firmus-pp4-settings.py` |
| gate/launch/回収 | `scripts/probe-firmus-prefill-dense.py`、`scripts/launch-firmus-prefill-dense-probe.sh`、`scripts/run-firmus-prefill-dense-probe.sh`、`platforms/firmus/framework-comparison/sglang.sbatch` |
| CPU回帰 | `test_prefill_dense_r44.py` と `prefill-dense-r44-source.json.gz`、既存 FA3 pin/optional-gate・overlay-list・staging の期待値 |
| 正本 | 本 note、`kernel-theory-tracker.md`、隣接 CPU-validation JSON と CPU-proof archive |

最終 source 監査で `advance_pipeline(uint32_t&)` が K index も更新することを確認し、N64 の次 B scale を `kb+1` で二重 advance していた箇所と、診断の loop increment と pipeline advance を重ねていた箇所を修正した。N64 は advance 後の `kb` をそのまま scale index とし、診断は for-update の `advance_pipeline(kb)` だけで進む。これは全 K 積を保持するための修正で、修正前を性能上限に使わない。実 header の lambda/loop 本文をそのまま C++ CPU stub へ挿入する回帰検査を追加し、32/64 K128 の全 full wait・phase・release と、N64 512/1024、診断256/512の論理MMA発行、128 fragment出力のbit一致を確認した。旧本文は候補の数値 assert と診断の全K数 assert の両方で棄却する。修正版の全6 TUを別ディレクトリで再構成し、上表のspill0・命令数と新encodingを再検証した。これは論理 pipeline 検査であり、実 GPU の非同期進行・数値の代替ではない。

最終検証：指定の PATH 設定と SIGHUP=SIG_DFL の後、指定の完全 pytest は100%まで完走して終了0、指定 Ruff も終了0。初回実装の dense CPU 52 case（Mレビュー時は99 case）、実 loop の32/64 K128、最終 patch/header全byte一致、shell構文、tracked/new file の whitespace を確認した。初回の全体実行では新規default flagを既存pin期待値へ追加する修正が必要で、実装監査に伴うローカル検証の再起動も行った。最終版で完全再実行したログと、それ以前の失敗・中断ログは proof に区別して保存する。GPU/Actions/scheduler/image build は実行していない。archiveの全member hash検査と、この作業用の展開/compileディレクトリ削除は CPU-validation JSON に記録する。


## 親レビュー後の M 修正と tail 監査

旧実装の M7988 eligibility と dispatch 条件は不具合だった。`compiled_dims="nk"` では SHAPE_M=0、実 M は launch argument と TMA descriptor に入る。M7988 の gate/prewarm だけで採用しても、最初の warmup M4001 や他の M では production source に戻る。host generator の M equality と診断 API の不要な M7988 制約を削除した。N/K・geometry・Normal・nk・K-major scale・no epilogue の条件、通常 API の diagnostic=false は保持する。generator の off 全byte一致と on define を M1/192/255/256/257/3994/3995/4001/4002/7988/7989/7990/8001/8002/8003/8004/8064/8192 などで検査し、Mを変えても生成sourceとcache keyを変えないことを確認する。

**M7988 は通常 profiling の代表 pair として正当化できるため主計時に残した。M4001 は warmup の観測で、通常 profiling の M ではない。** 保存 run `41875058470/1/logs/logs/prefill-0.log` の422/426/430/434行に最初の r37 A dense dispatch `rows=4001`、438〜445行に single 4032 / pair 8064 の scheduler 表示がある。これは01:40:13以降の AIPerf warmup である。他の Prefill worker も同じ最初の実 M を記録する。export の phase 区分では warmup3125件、profiling6342件。通常 profiling の client ISL は3986が6313件、3987が29件である。

最初はこの warmup log を通常 profiling の M と解釈したが、実 template/phase の再計数で修正した。pinned image `41853888382.sqsh` を read-only SSH で読み、image 内 AIPerf の `workers/worker.py`、`common/constants.py`、`endpoints/openai_chat.py`、`records/inference_result_parser.py` を確認した。warmup は system message `warmup` を追加し、profiling は追加しない。ISL は送信payloadの本文をtemplate前に再tokenizeする値であり、cache-bustを別の固定token数として重複加算しない。`serving_chat.py` はHF templateをrenderしてから同じtokenizerでencodeする。モデルのtokenizer.json/configは `ssh firmus` の `/scratch/models/Qwen3-235B-A22B-Instruct-2507-FP8/` からread-onlyで取得し、保存tokenizer.sha256と照合した。login.hpcai2025.smcでmodel/imageの両パスが共有 `/scratch` BeeGFS上にあることも確認した。保存12800入力の素の本文は3986が12747件、3987が53件。通常のuser-only templateは+8 tokenで **single3994/3995、pair7988/7989/7990**。warmup system prefixはさらに+7 tokenなので **single4001/4002、pair8002/8003/8004** となる。実初回4001はこのwarmup templateで再現できる。例えば先頭入力は通常本文3986→template3994、warmupの本文結合ISL3989→template4001。warmup ISLの分布3987〜3990は本文連結時の空白/BPE境界にも影響され、単純に通常ISLへ固定差を足すための根拠にしない。正確なpairの組合せ順は保存ログから復元しておらず、7989/7990や8003/8004は各single長の和として可能な値である。

SGLang `fc275674` の `schedule_policy.py:750` は `_update_prefill_budget` のローカル変数を `ceil_paged_tokens` へ丸め、783行で log_input_tokens へ加える。`metrics_reporter.py:562` が表示する。対して `schedule_batch.py:2192` は実 fill_ids を連結し、2193行で実行token数を求める。PP4 Prefill は TP/DP/EP=1、CUDA graph disabled。page budget の4032/8064は dense tensor へ padding しない。`prefill_fp8.linear` は a.shape[0] のまま出力を確保し `fp8_gemm_nt(..., compiled_dims="nk")` へ渡す。attention backend の `attention(..., padded_rows=q.shape[0])` も元QKVの行数を保持する。この padded_rows はschedulerのpage countではない。従って QKV と O は同じ実 M を使う。

N64 は M64 wave 内の N128 product/accumulator を半分ずつ扱うだけで、runtime M、M tile数、A/scale TMA producer、D descriptor、STSM/TMA epilogue を変えない。元 producer と epilogue の全byte一致、A/D/scale descriptor が実 m を受け取ることを source fixture で検査した。範囲外 A は既存TMA bounds、範囲外出力は既存D descriptor/storeが扱う。N64は元N128 register layoutへ戻してから同じepilogueに入るので、tailを全tile前提へ変えない。CPU emulation は M1/255/256/257/3994/3995/4001/4002/7988/7989/7990/8001/8002/8003/8004 の最終CTAをモデル化し、valid rowのbit一致と出力guardを検査する。これはTMA hardwareのGPU証明ではない。

親Mレビュー時のv2 gateは5M参照とsingle timingを要求していた。今回のHEAD予算レビューでv3へ更新し、全12GPUのM7988/8003/3994のbit/FP32参照、500call、pair計時と診断を要求する。prewarmは7988/3994、eligibilityはruntime M非依存のまま。M4001を通常profilingと誤解した途中案は撤回済み。CUDA/source/headerのcompile proofは変更しておらず、今回のレビューではCUDA compile、image build、GPU実行、job投入を行っていない。今回の最終指定pytest/RuffとCPU回帰、read-only accounting/原ログ調査の証拠をvalidation/proofへ追記する。

HEAD予算レビューの最終検証：指定のPATHをexport後、指定の完全pytestは100%まで到達して終了0（2072 passed、90 skipped）、指定Ruffも終了0。dense CPU回帰107件、workerのnormal fast-exit/cleanup例外/全12rank完備、compile owner・barrier、実sbatch blockの270秒adopt/249秒reject/257秒timeout/271秒skip、shell構文、whitespaceを確認した。GPU実行・CUDA/image build・job投入・commit/pushは無し。
