# 現行CUTLASS Prefillの有限プロファイル

更新日: 2026-10-04。実行中の計測であり、性能改善の採否はまだ判断しない。

## 条件と来歴

成功c48 `ecd89520e2ec900741b42cb80f336280629e5620` / Actions37121275791を基準に、image `20c3a38359ccc61967343ec23d600f783df8d487c6339d12b9ca53c14b55d3d3`、SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4`を再利用する。Qwen3-235B-A22B-Instruct-2507-FP8とnebius BF16 EAGLE3 draft、両方FP8 KV、FA3、P CUTLASS、2P/2D各TP4 EP1 DP1 PP1、16H100、並行度48、dgx026/030を固定する。失敗DP2を計測対象にしない。

実験branch `research/aspire-prefill-profile-20261004`、commit `1ec209daeba5708f79c94bbeaf5f9a27e8878656`、[Actions37135273516](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37135273516)。本体sourceとimageを変更せず、外部Nsight Systems2025.3.2をPだけread-only mountする。D commandと全serving引数は対照と一致する。

## 取得窓と成立条件

AIPerf warmup完了後、本計測開始markerから5秒待ち、P2へ並列startを送る。両応答後5秒の有限窓を取り、並列stopする。Torch CPU record_functionとCUDA_PROFILER APIを併用し、NsightはCUDA graph node kernel、NVTX、OS runtimeを取得する。既定spec_stage_spanはNVTX無効なので、CPU spanとkernel correlationでtarget/draftを照合する。

stop HTTP timeoutは180秒。旧profileで5秒timeoutが失敗し約60秒後にexportが完了した実測に基づく。固定sourceはCPU export後にcudaProfilerStopするため、flush tailを解析対象の定常5秒と分離する。部分start/中断ではbest-effort parallel stopし元errorを保持するが、通常stop失敗の再試行はしない。

安価検証107tests、Ruff、bash構文、diff whitespaceを通過し、独立Sol reviewでblockerなし。2つのnsys reportと8つのTorch CPU traceがない場合run-successを認めない。terminal監査では実kernel数、8rank、時間窓、graph/node/correlation、公式parser、実引数、image、checksumと残留PBSを確認する。profile scoreはobserverの影響を受けるため採用評価値として扱わない。

## 判別範囲と残る限界

実P target/draft、MoE quant/route/shuffle/GEMM/SiLU/combine、AllReduceとGPU idle、CPU launchの時間をまず測定する。CUDA graph replayはhost範囲と単純に一致しないため、stage推定にはconfidenceを付ける。expert histogram、NIXL RDMA room-level状態、厳密なP release因果は既存toolだけでは保証しない。全120秒captureや新規instrumentation基盤は追加しない。

raw正本はrunのサーバーartifact、派生解析とpreflightは `research-issue-17/20261004-current-prefill-profile/` に保管する予定。現時点の結果はqueued、terminal監査待ち。
