# 現行CUTLASS Prefillの有限プロファイル

更新日: 2026-10-04。計測E2Eは成功。全8rankの実kernel解析を完了。次候補の順位を更新したが、非profileの改善採否はまだ判断しない。

## 条件と来歴

成功c48 `ecd89520e2ec900741b42cb80f336280629e5620` / Actions37121275791を基準に、image `20c3a38359ccc61967343ec23d600f783df8d487c6339d12b9ca53c14b55d3d3`、SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4`を再利用する。Qwen3-235B-A22B-Instruct-2507-FP8とnebius BF16 EAGLE3 draft、両方FP8 KV、FA3、P CUTLASS、2P/2D各TP4 EP1 DP1 PP1、16H100、並行度48、dgx026/030を固定する。失敗DP2を計測対象にしない。

実験branch `research/aspire-prefill-profile-20261004`、commit `1ec209daeba5708f79c94bbeaf5f9a27e8878656`、[Actions37135273516](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37135273516)。本体sourceとimageを変更せず、外部Nsight Systems2025.3.2をPだけread-only mountする。D commandと全serving引数は対照と一致する。

## 取得窓と成立条件

AIPerf warmup完了後、本計測開始markerから5秒待ち、P2へ並列startを送る。両応答後5秒の有限窓を取り、並列stopする。Torch CPU record_functionとCUDA_PROFILER APIを併用し、NsightはCUDA graph node kernel、NVTX、OS runtimeを取得する。既定spec_stage_spanはNVTX無効なので、CPU spanとkernel correlationでtarget/draftを照合する。

stop HTTP timeoutは180秒。旧profileで5秒timeoutが失敗し約60秒後にexportが完了した実測に基づく。固定sourceはCPU export後にcudaProfilerStopするため、flush tailを解析対象の定常5秒と分離する。部分start/中断ではbest-effort parallel stopし元errorを保持するが、通常stop失敗の再試行はしない。

安価検証107tests、Ruff、bash構文、diff whitespaceを通過し、独立Sol reviewでblockerなし。2つのnsys reportと8つのTorch CPU traceがない場合run-successを認めない。terminal監査では実kernel数、8rank、時間窓、graph/node/correlation、公式parser、実引数、image、checksumと残留PBSを確認する。profile scoreはobserverの影響を受けるため採用評価値として扱わない。

## 判別範囲と残る限界

実P target/draft、MoE quant/route/shuffle/GEMM/SiLU/combine、AllReduceとGPU idle、CPU launchの時間をまず測定する。CUDA graph replayはhost範囲と単純に一致しないため、stage推定にはconfidenceを付ける。expert histogram、NIXL RDMA room-level状態、厳密なP release因果は既存toolだけでは保証しない。全120秒captureや新規instrumentation基盤は追加しない。

raw正本はrunのサーバーartifact、派生解析とpreflightは `research-issue-17/20261004-current-prefill-profile/` に最小派生証跡を保管した。Actions・PBS・公式結果確認・artifact回収・home archive・cleanupは成功。PBS216010、walltime11分55秒、Exit0、dgx026/030。raw79ファイルの全hashをlocal/remoteで照合した。

## Terminal監査

2nsys reportと8Torch CPU traceが存在する。capture開始要求16:13:49.749Z、両ack後active16:13:55.383Z、stop要求16:14:00.388Z、両stop完了16:14:51.370Z。ack後の観測窓は約5.004秒、export/flushは約50.98秒で、180秒timeout内に完了した。raw全体334MiB。

同image/固定source、4workerのserver-infoは対照c48との差がrandom_seed/internal_statesだけで、schedule_conservativeness1.0、chunk8192、Pgraph8192、Dgraph13/15等の設定は一致。target KVは各rank K/V各1.47GB、draft各0.02GB、max tokens131072、P起動後available GPU memory12.62GB。benchmark後の8greedy probeは各64completion/logprobを確認し、学習用途ではない。残留PBS一覧は空、runtime削除を確認した。

公式parser再実行は保存scoreと一致。889profiling requests、error0/cancel0、121件TTFT>3秒、ITL超過0、jointgood768/889=86.3892%、公開90%閾値を満たさずscore0。CPU observer/trace収集がある本runの値であり、対照の性能劣化や候補棄却の根拠にはしない。目的はkernel/boundary計測で、実CUDA kernel解析とactive窓の切り出しは以下のとおり照合完了した。

## 実CUDA kernel解析

run 37135273516 / commit1ec209d、image20c3a383。正本 `/hdd_raid/samba/net_share/qwen-artifacts/37135273516/1`。P2TP4EP1DP1/D2TP4EP1、c48、EAGLE3、BF16 draft weights、FP8 KV。instrumented runであり通常scoreとして採用しない。

### 範囲と検証

capture_active 2026-10-03T16:13:55.383751Zからstop_requested16:14:00.388142Zの5.004391秒を使った。stop完了まで約51秒のCPU export尾を定常分母に加えない。SQLite utcEpochNsとTorch baseTimeNanosecondsから両traceをUTCに揃え、CPU user_annotationのstep[EXTEND] / draft_extend内のCUDA runtime launchをglobalPid/correlationIdでdevice kernelへ結合した。GPU時刻の単純包含によるstage分類ではない。CUDA Graph nodeも同じlaunch correlationを保持し、全8rankでtarget/draft分離可能だった。窓境界を跨ぐkernelは除外した。

P0各rank target52253/draft816/未分類1128kernel、P1 target52371/draft868/未分類1157。未分類GPU時間和は各rank2.90–2.97ms。全8rankにgraph node/attention/MoE/collectiveが存在し、kernel unionは4.613–4.632秒（窓の約92.2–92.6%）。kernel時間和とunionの差は小さいが、表は時間和であり改善可能なcritical-path時間そのものではない。rankを合算しない。

KinetoのCUPTI multiple-subscriber warningはCPU profiler開始時にある。Nsight側はCUDA initialization成功、非空全rank CUDA kernel/runtime/graph tableがあり、DIAGNOSTIC_EVENTにdrop/lost/error/subscriber報告は無い。driver13.2に対しNsight CUPTI13.1を使うwarningは両reportにある。これだけで絶対完全性は証明できないため、この一回の順位判断の制約として残す。capture toolsは固定外部Nsight、CPU exportはhome nsys2026.3.2で行った。

### 実測された費用

以下は各Pworker rank0のactive窓内GPU kernel時間和。単位ms。stage分離後のtarget値。up/downは同じCUTLASS template名なので、source順序 `shuffle(float)→get_group_gemm_starts→GEMM(up)→fused SiLU/group quant→get_group_gemm_starts→GEMM(down)` で照合した。対象GEMMの未分類は0。shape gridも同じ(66,2,1)なのでgridだけでは分けられない。

| 処理 | P0 TP0 | P1 TP0 | 所見 |
|---|---:|---:|---|
| target全体 |4562.824|4560.193|主要費用|
| draft全体 |68.896|73.936|targetに比べ小さい。移設利益の上限とはしない|
| MoE grouped up GEMM |714.658|714.412|ES/up実装の優先根拠|
| MoE grouped down GEMM |577.083|578.770|ES/downも大きい|
| target BF16 ring LL allreduce |1043.302|1031.623|実通信と待ちが混在|
| FA3 target attention |330.320|331.891|無視できない|
| down combine apply_shuffle_mul_sum |323.798|324.880|中規模改修の有力費用|
| input FP8 shuffle |265.934|266.675|copy/layout改善候補|
| input scales shuffle |65.672|65.929|上記と合計331.606/332.604|
| QKV/output DeepGEMM計 |352.186|354.758|MoE groupedとは別|
| fused add RMSNorm計 |279.837|277.053|主stage一覧への注意|
| compute_arg_sorts |114.717|115.555|route準備の主要費用|
| compute_problem_sizes |39.870|40.079|route準備|
| MoE input quant |45.471|45.659|隣接compute_arg_sortsで分類|
| 既存fused SiLU+group quant |65.944|66.196|既に統合済、再候補にしない|

MoEのinput shuffle+scale shuffle+down combineだけで655.4/657.5msあり、layout/copy/combineを一括で軽視できない。一方個別routing kernelをそのまま全部消せるものではない。

CPU draft_extendはP0TP0で17回・計2757.49ms（約162ms/回）だが、correlated GPU draftは窓内68.90ms・16回相当である。target graphを非同期launchした後の待ちがCPU draft spanに入るためCPU162msをdraft計算コストと解釈してはいけない。窓境界によるhost17回/device16回差もある。既存target FULL hidden captureの費用がtarget内で独立分離されたわけではないので、draft GPU比率約1.5%だけから移設最大gainを断言しない。

### Allreduceの到着待ちをどこまで分けられるか

各workerの全BF16 ring LL呼出しは4rankともP0 3202回、P1 3212回で、同名kernelの時間順序を揃えた（draft呼出しも含む）。開始skew中央値27.87/23.48us、p90 36.40/32.12us、max3428/1569us。各rank開始から最後rank開始までの時間和はP0 17.97/57.61/92.16/48.98ms、P1 41.45/17.25/72.30/64.41ms。4rank共通kernel在留区間の時間和は974.56/972.94ms。

従って見えている開始skewだけでは約1秒のcollective費用全体を説明しない。ただし共通在留区間にはNCCL内部同期、転送、protocol処理、GPU資源競合も含まれ、純link時間には分離できない。開始順序alignmentはcollective sequenceと整合するが、payload/algorithm別の厳密通信帯域測定ではない。NVLSで1秒全てが消えるというgain予測は禁止。NVLSは低コストの対照probeとして有望、NCCL費用が大きいという根拠は今回得られた。

### 優先順位

1. **設定だけの次検証: NVLS対照**。大きいAR費用が測定されたので優先を上げられる。通常非instrumented同条件runで成立と改善を確認する。flagだけでNCCL実選択が変わるか、CUMEM等の既存設定との依存も確認が必要であり、今回は追加実機を行わない。Hopper ESは実装確認後の次点小改修: grouped up714ms/down579msが根拠で、upを先に検証しdownも分けて比較。native CUTLASSの代替性能は未測定。
2. **中規模: down combine/copy/layoutとMoE compact**。input shuffle+scales+combineが約656ms、route sort約115ms。DeepGEMM compactとfusionはこれらをどう置換し、grouped GEMMを悪化させないかを比較すべき。EP4 Tritonはgroup幅変化に加えbackend変更なので費用順位から勝ちとは判定できず、上記より不確実性が高い。
3. **低コスト・条件付き: P request cap2/chunk**。queueや同時request制御のprobe価値は維持するが、今回GPU busy約92%と以前8192充填93%から大gainの確証はない。
4. **大改修: D cold-start draft/PP+EAGLE/TP8 padding/NIXL DP2**。cold-startは品質検証を維持する設計として可能だが、P draftGPUが小さいため性能向上の初手として順位低下。NIXL DP2は互換性修正の前提が未解決、今回profileから改善量は判定できない。PP/TPは新topology/backend/runtime制約を解く必要があり、このprofileだけでは順位確定不可。

追加GPU測定は本解析では投入していない。KV transferはNsightにrequest-room対応の直接spanが無く、sampler帯域やmemcpyからE2E transfer待ちを捏造しない。今回の最小profile目的（現P target/draft/MoE/ARの費用判断）は達した。

解析証跡: stage_window.py / substages.py / stage-window-p0.json / stage-window-p1.json / substages-p0.json / substages-p1.json。SQLiteはローカル派生、raw正本は変更していない。
