# Prefillの既存オプションと改修候補（2026-10-03）

現状の目星は、**既存オプションではP側のNVLSと要求数上限、コードでは既存H100 expert-specialization GEMMへの接続**である。大きな改修なら、DeepGEMMのGEMM本体より先にEP1 prefillのデータ配置を直す。これらは候補の優先順位であり、新たな高速化の実測結果ではない。GPU・Actions・schedulerジョブは追加していない。

## 対象と根拠

実行イメージのSGLangは `fc275674a97d78553b22597ee72c8fd48f5420f4`。調査時のmirror HEADは `583417f1` へ進んでいたため、全参照を `git show` / archiveで固定コミットへ合わせた。mirrorは変更していない。mainの採用構成と、未採用のSiLU/draft head統合候補を区別する。今回のバッチ観測は主に統合c44 `37119291598` と同一イメージのc48 `37121275791`。従来のNsight trace `36387126341` は旧ノード・auto prefillのdecode側であり、現在のCUTLASS prefillの各処理時間とは扱わない。

実際のP設定はTP4/EP1、2 worker、CUTLASS MoE、FA3、FP8 KV、chunk8192、max-prefill-tokens16384、max-running-requests48、prefill-max-requests未指定、FCFS、radix cache有効、通常のoverlap scheduler有効、prefill CUDA Graphはbreakable・最大8192。FlashInfer all-reduce fusionはauto。NVLS、QKnorm/RoPE融合、TBO/SBOは無効。EAGLE3はP側のdraft prefill経路を含めて構成されており、Pの変更がDへの供給間隔に影響する。

c44→c48でgood tok/sは2667.013→2673.508（＋0.24%）、平均TTFTは1297.590→1581.031 ms、P待ち数はworker別平均0.432/0.529→1.762/1.985、実効D並列数は26.097→26.082。増やしたのは**client concurrency/workers-max**であり、P workerのmax-running-requestsは両方48のまま。これはP側の待ちが制約である根拠だが、PのどのGPUカーネルが支配的かを証明するものではない。

## 既存オプションの優先順位

| 優先度 | P側だけに適用する候補 | 狙い・判断 |
|---|---|---|
| A：通信側の候補 | `--enable-nccl-nvls` | T3992/8192×H4096×BF16は約31.2/64MiB。FlashInfer AR/norm融合の2048トークン上限とcustom ARの8MiB上限を超え、通常の大バッチではNCCL側を調整する根拠がある。外部envで上書きされなければ現状コードはNVLS_ENABLE=0を設定し、このフラグは、外部envが0でも明示的に1へ変える。実際のNVLS選択・対応fabric・通信時間の短縮は未確認。CUMEMやsymmetric allocationまで自動で有効になるという意味ではない。|
| A/B：待ち時間の候補 | `--prefill-max-requests 2`、chunk8192を維持 | サーバーの非分割要求は主に3992トークン。8192には2本＋3本目の208トークンが入り得るため、その断片生成を抑える。7984/8192の充填率低下と引き換えであり、goodput改善は未保証。継続要求も枠を使うので全断片の消滅を保証しない。|
| B：条件付き | `--chunked-prefill-size 12288` | 8192→16384の間の候補。3要求をまとめてGEMM効率を上げる狙いだが、長い1回の処理と大きいgraph/活性化領域がTTFT・メモリを悪化させ得る。3992×3=11976なので、単独では4本目の312トークンを分割する。`prefill-max-requests 3`との組み合わせも別の仮説。既存CPUモデルで大chunkは不利であり、最初の実機候補には置かない。|
| B/C：上記の補助 | `--cuda-graph-bs-prefill`に3992/7984等を追加 | 単位は要求数でなくトークン数。要求数制限でこの形状が増えた場合にだけ組み合わせを検討。必要な既存バケットと最大値は残す。現状の大半は8192ちょうどなので単独の優先度は低い。|
| B/C：小融合 | `--enable-fused-qk-norm-rope` | CUDA BF16 QKV・対応head dimension・非MRotary等を満たすQ/K normとRoPEの既存融合。各層でpositionsをint32へ変換するため、フラグだけでは追加キャストが利得を相殺し得る。FP8 KV cast/storeまで融合するフラグではない。現状は大きいMoE/通信の候補より後。|

NVLSの根拠は固定版 [`engine.py:1502`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/entrypoints/engine.py#L1502)、[`communicator.py:162`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/layers/communicator.py#L162)、`distributed/device_communicators/custom_all_reduce.py:40,260`。実トポロジー・実env・選択アルゴリズムを確認して初めて有効な比較になる。

要求数上限と分割は [`schedule_policy.py:1068`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/managers/schedule_policy.py#L1068) および1195–1242。継続chunkは新規待ち要求より先に追加される。ログは集計値なので「全バッチが2本＋208トークン」とまでは断定しない。

### 現状では優先しない、または効かない設定

- `--max-prefill-tokens`だけを16384より増やしても、chunk8192という実際の予算は増えない。`--max-running-requests`と`--schedule-conservativeness`も、現在観測していないKV不足・退避を解決するために上げ下げする理由は弱い。
- `--disable-cuda-graph-padding`は固定版P runnerのトークン切り上げを制御しない。D側のgraph構成まで変えるのでP最適化目的で使わない。
- `--enable-prefill-delayer`のqueue条件はrunning_batch>0を要求するが、今回のPログはrunning_req=0。バッチをまとめる名前だけで有効と判断しない。
- `--enable-two-batch-overlap`は現状EP1/no-A2A/DP-attention無効と非互換。Qwen側のdispatch/combine実装にもEP条件がある。`--enable-single-batch-overlap`のdown/combine経路は現状CUTLASSを対象にしていない。
- `--enable-fused-moe-sum-all-reduce`はCUTLASS側のdown/combine融合にはならない。Triton側での数値・動作も別途扱う必要がある。
- `--flashinfer-allreduce-fusion-backend auto`は既に有効。明示するだけではT>2048の制限を越えない。
- `flashinfer_cutlass`は現在のnative `cutlass`と別物で、当該checkpointの通常FP8形式は固定版の選択制約に合わない。CuteDSLのNVFP4用経路を混同しない。
- CacheやFCFSの変更は後回し。Pのcached tokensは新規約8000トークンに対して小さく、要求長はほぼ一定。現在のround-robin routingでも両P workerに待ちが生じている。ルーターpolicyとSRTの`--load-balance-method`は別の設定である。

## DeepGEMMへの単純切替を第一候補にしない理由

`--moe-runner-backend deep_gemm`は既存のFP8経路として指定可能で、decodeで遅かったことはprefillでの敗北を証明しない。しかしEP1のstandard dispatcherはprefillでも [`deep_gemm.py:668`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py#L668) から `use_masked_gemm=True` を返す。

[`ep_moe_kernels.py:1533`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/kernels/ops/moe/ep_moe_kernels.py#L1533) のM容量は `(T//256+1)*256`。T8192/E128/H4096では次の差がある。

| ランク・層当たりの割当容量 | CUTLASS | EP1 masked DeepGEMM |
|---|---:|---:|
| FP8 gate-up入力 |256MiB|4.125GiB|
| BF16 down出力 |512MiB|8.25GiB|

これは**確保容量**であり、実行FLOPs・HBM転送量・同時ピークメモリではない。それでも、現状のprefillで無条件に切り替える前に確認すべき大きな差である。`SGLANG_OPT_DG_MASKED_M_CAP`は既定falseで、有効化しても各層の`.max().item()`によるD2H同期を伴い、CUDA Graph capture中はcapを適用しない。現状breakable graphの単純な救済スイッチとは扱えない。

## コード変更ならここを狙う

### 1. 既存H100 expert-specialization GEMMへの接続：最初の小改修

[`layers/quantization/fp8.py:2444`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/layers/quantization/fp8.py#L2444) は `enable_es=(use_mxfp8,use_mxfp8)` を渡すため、今回のH100・通常FP8では両方false。一方、[`cutlass_moe.py:220`](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src/blob/fc275674a97d78553b22597ee72c8fd48f5420f4/python/sglang/srt/layers/moe/cutlass_moe.py#L220) と286にSM90向け `es_fp8_blockwise_scaled_grouped_mm` のup/down分岐が既にある。

変更箇所はこの呼び出し条件。H100・block128・対象shapeに限定して既存ESを使い、モデル・FP8境界・top8・FP32 scale・SiLU融合・combineは保持する。up N768/K4096、down N4096/K384は同梱テストの128アラインメント条件を満たす。新しいGEMMを書く前に検討する価値がある。

ただしES実装は補助workspace、2つの追加stream、eventsを使う。現状workspace90000 bytesを受け取り複製し、同梱テストは1GiBを渡しているが、これは必要最小量の証明ではない。同梱benchmark/testは現在のbreakable graph適合性を証明しない。実形状のcan_implement、workspaceサイズ、capture/replay、有限値・誤差を短い診断で確認してから全体比較する。既定offであった経緯は取得できた履歴境界より前であり、無効な理由を性能以外も含めて推測で埋めない。

### 2. EP1 prefillをDeepGEMMの連続配置経路へ渡す：次の中規模改修

変更箇所は `moe_runner/deep_gemm.py` の `pre_permute_standard_to_deep_gemm` / `post_permute_deep_gemm_to_standard` と `kernels/ops/moe/ep_moe_kernels.py`。GEMM本体ではなく、GPU上でexpert件数→prefix sum→scatterを作り、連続したexpert行へ並べて既存contiguous grouped GEMMへ渡す。

必要なのは `m_indices`、expert境界、`all_tokens`に相当する静的容量、復元map。DeepEP normalの既存adapterはCPU件数を扱うため、そのままEP1へ移植しない。graphで可変件数をhostへ読み返さず、kernelが要求するアラインメント、余裕行の無効化、全top8復元を設計する。現在のCUTLASSが行うrouter-weightのdtype変換とFP32合成を維持する。これはmasked容量問題を避けられる設計候補であり、実装済みでも速度保証でもない。

### 3. CUTLASSのルート準備と入力コピー：境界を保った改修

`kernels/aot/csrc/moe/prepare_moe_input.cu` のcount/assignmentはexpertごとにroute配列を走査する。E128・R65536では論理2ER回の参照だが、元のID配列256KiBはcacheableであり、HBM律速とは断定しない。

最初は `a_map/c_map/expert_offsets/problem_sizes` の契約を保ってhistogram→scan→scatterへ置換する。さらに `cutlass_moe.py:209–211` のFP8入力とscaleの2つのshuffleを統合する。量子化は元のT行で1回行い、top8分の量子化を重複実行しない。入力複製自体を消すgather対応GEMMは、その先の大きい変更とする。

### 4. down出力＋合成、large-prefill norm/quant：大きいが後順位

`cutlass_moe.py:214,285–339` の `c2[R,H]` と `apply_shuffle_mul_sum` は、T8192で512MiBのBF16中間出力を作る。ここを省くなら8expertを合成する出力タイルの所有権とFP32蓄積が必要。BF16 atomic追加で代用しない。

`communicator.py:162–178,580–620` のlarge-T制限を外すだけでも不十分。H100向けRMSNorm＋group128量子化を作るなら、router/residual用BF16出力も保持し、QKV/MoEのconsumerが再量子化しないよう接続する。既存のquant付きAR/norm helperはROCm/AITER用であり、NVIDIAで有効にするCLIはない。大容量AR融合はworkspaceと通信進行を確認する別案件とする。

SiLU/group128融合は統合候補で既に測定したため、再び未実装の利得として数えない。最終FP8という理由だけでFP32 reductionやnormを省かない。

## 既存ログから絞れた範囲

バッチログを既定graph bucketへ切り上げるCPU集計では、3992/7984追加による「切り上げ行数」の削減proxyは統合c44で0.456%、c48で0.092%。8192ちょうどのバッチはそれぞれ614/837（73.36%）、769/827（92.99%）。実行カーネルの行数・時間・FLOPsの測定ではないが、現在のprefill graph paddingだけを主対象にする根拠は弱い。

現在のCUTLASS prefillについて、route prepare、shuffle、up GEMM、SiLU/quant、down GEMM、combine、大容量ARの各時間は未分離。旧decodeのカーネル合算時間をここへ流用しない。次に計測するなら、この境界と実expert件数を同じ入力で確認し、採点中のinstrumentationは避ける。必要な短時間診断を後続の同じ割当の採点前にまとめ、追加ジョブの総当たりはしない。

実装時はQwen側 `submission/run/submit.sbatch` のASPIREかつprefill分岐へ必要な既存引数を追加し、decodeを固定する。新しい汎用設定・中継envを増やさない。ソース改修はQwen所有パッチとして私有build cloneへ適用し、管理mirrorを更新しない。現行buildは`sglang-kernel==0.4.5`のwheelを入れており、`aot/csrc`を編集してeditable SGLangを入れるだけではnative kernelは変わらない。native wheelの明示的なビルドか、範囲を限定したTriton/JIT実装が必要。Pythonから既存ESを呼ぶ改修でも新規イメージと来歴を残す。

当面は**NVLS＋断片抑制のruntime案**と、**CUTLASS＋既存ESのsource案**に絞る。単純DeepGEMM切替は容量・graph問題を先に解決し、大規模down/combine融合は局所時間を確認してから着手する。依存バージョン差を管理し、比較は公開SLA内のgood tok/sで評価する。ここでは実装・投入を行っていない。

## 証跡とレビュー

GPT-6.1-Solの独立したオプション監査・カーネル監査を統合し、最終案を両者で再確認した。TASKの日本語追記はagy草稿を実測範囲に合わせて修正した。正本は `archive-host:/srv/qwen-artifacts/research-issue-17/20261003-existing-data-analysis/prefill-option-code-audit/`。21ファイルのリモートハッシュ照合に成功し、manifest SHA256は `1315277cb0412f9974da43c636a2b422fbb7ef6266c0a35ac835f42403567315`。固定ソース12ファイル、実設定、再集計script/JSON、容量計算、各監査ノートを保持し、既存rawログは複製していない。
