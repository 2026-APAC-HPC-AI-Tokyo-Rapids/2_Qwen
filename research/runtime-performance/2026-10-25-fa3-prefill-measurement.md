# Prefill FA3 判別計測の probe 準備

対象は pinned sgl-attn f89bc230 / CUTLASS 57e3cfb、SM90a、FP8 入力と group-128 FP8 出力。serving 本体は変更しない。GPU ジョブはこの作業では投入していない。

`kernels/probes/prepare-fa3-prefill.sh` は M1、`kernels/probes/prepare-fa3-prefill-profile.sh` は M2。既存 Genkai kernel probe workflow の gpu_count=1、candidate_patch 空欄を使う。両者とも provenance の calibration=true により 1 trial / 165 秒。M0 は login node で前者に `--compile-only` を渡す。f=M192/N128 と V_colmajor=1 は初回 M0 の確定結果を記録に残し、通常 prepare と compile-only の両方のビルド対象から外した。

M1 は shape4096、shape4x1024、既存 n192(BF16) / fp8(N192 fused)、nobarrier、regs232、lazy を測る。N192 fused は自身の BF16→FP8 control と bitwise 比較し、N224 との比較は既存 FP32 数値 gate を使う。他の fused 候補は N224 stock+quant の FP8 bytes / scales と bitwise 一致を要求する。lazy-counter は別の共有ライブラリで、計時グラフには含めない。Q/K は既存の合成入力で、skip 率は実活性化の代表値ではない。

PackGQA M128 の tile は 8 token × 16 Q head、KV head は4個。iteration は各 tile が訪れる N224 KV block 数の和。shape4096 は I=19760/T=2048、4×1024 は I=5760/T=2048。T_us=I*X+T*Y を解き、Y の寄与は 2048*Y / shape4096_us。scheduler prepare を含む per-call の二項近似であり、shape 依存の固定費や occupancy の差は分離できない。GPU 間の throughput 比較に使わず、次候補の判別材料にする。

`results/fa3-prefill-m1.json` に全 sample、median、gate、X/Y、skip、ptxas、SM clock を保存する。cycles は sample 前・中・後の nvidia-smi SM clock MHz の中央値 × µs の近似換算で、profiler の cycle counter ではない。M2 は report 内の SM cycles/sec を併記できる。

M2 は control shape4096 の warmup 3 launch 後の1 launchに、SourceCounters、InstructionStats、shared-memory tables、XU/ALU/FMA/tensor counters と opcode/modifier/category metrics を適用する。SASS に DEPBAR、BAR、MUFU、LDL の PC stall samples を対応付ける。テンプレート内の FlashAttnFwdSm90 名を照合するため `--kernel-name-base demangled` を指定し、raw CSV には `--print-metric-instances details` で opcode の instance 内訳を残す。

玄界の NCU は **2025.2.1.0 (build35987062)**。GPUを使わないGH100 metric queryで全16指定metricの実在を確認し、3 sectionの展開とCLI引数のparseも確認した。初期案の `sass__inst_executed_per_opcode_pipeline` はこの版に存在しないため削除した。F2FP判別には、`sass__inst_executed_per_opcode` と `sass__inst_executed_per_opcode_with_modifier_all`、`smsp__sass_thread_inst_executed_op_conversion_pred_on.sum`、XU/ALU/FMA各pipeの `smsp__thread_inst_executed_pipe_*_pred_on.sum` を使う。SourceCountersの `inst_executed` / `thread_inst_executed_true` と合わせ、F2FPの各modifier、MUFU、その他の変換命令の実行数をhardware pipe数と照合する。warp単位とthread単位、predicate有効/無効を混同しない。名前の実在確認だけでF2FPの実行pipeは確定せず、実測後の照合結果が曖昧なら未判定と記録する。shared throughputは `l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed` を明示指定。このshared指標はLSUのwavefront利用率であり、TMA/WGMMAを含む全shared trafficのbyte数と同一視しない。profile下のdurationはM1の性能値に混ぜない。

未実行のため Actions/PJM job ID、GPU cycle 数、gate 成否、skip 率の実測値は未確定。親の実行後、raw artifact を参照してこの記録に job ID と clock/cycle 要約を追記する。


M0 login-only の結果：N224は168 registers、spill stores80/load140 B、SMEM191488 B。local-memory 命令は producer24 区間の LDL35/STL20、consumer240区間は0/0。regs232/producer40はspill8/8 B。f=M192×N128はSMEM141312 Bだがspill272/292 BとC7512があり、既定kill条件に該当した。

初回の停止理由はV_colmajor=1のvnew/Qv/PagedKVManager stride型不一致だった。ユーザーがこの結果を確定し継続を指示したため、sourceを修正せず対象から除外してM0/M1/M2のlogin-only準備を再検証した。GPU実測とdispatchは親が実施する。


判断確定：V_colmajor=1（c1）の単純 instantiate は pinned source の型エラーとして修正せず、V_colmajor 経路を使わない c2（smem_v byte image の前段生成）を妨げない。


最終のlogin-only検証（2026-10-25、ssh genkai.hpc.kyushu-u.ac.jp）：compile-onlyは522秒、M1通常prepareは515秒、M2 profile prepareは147秒で終了コード0。M0/M1のptxas.jsonはcontrol/nobarrier/regs232/lazy/lazy-counterの5キー、M2はcontrolのみで、すべてcalibration=trueのprovenanceまで生成した。M2の最後の変更はNCU runtime filterだけなのでcontrolビルドを再利用し、prepareと同じwrapper配置手順を再実行して最終版との一致を確認した。最終CLIの16metric/3section、demangled指定、opcode instance details指定はinstalled CLIで受理を確認した。

dry-runはbase commit 8b7d984の未commit作業ツリーを使い、テストしたsourceのSHA256を回収してローカル最終版と照合した。親のdispatch時のprovenanceは親が作るcommitのGITHUB_SHAになる。GPUゲート、X/Y、skip率、実際のF2FP routing、NCU収集はまだ未実行。既存workflowはrun directoryを削除するためJSONとCSV全内容をstdoutにも保存し、binary ncu-repの永続回収は親の担当とする。

規則7のruff・行継続・bash-n・全pytestを再実行して成功。追加のprobe ruff、CPU fixture、compiled symbolとdemangled regexの一致も確認した。回収物は担当workのfinal-prepares.tar.gz（各build/resource/SASS、ptxas/provenance、NCU問い合わせ証拠）。専用領域 /fast/PROJECT/sglang-runs/kernel-probe-fa3m-5abe09c0 はchmod -R u+w後に削除し、不存在を確認済み。GPUジョブ、Actions起動、git commit/push/branch操作は行っていない。
