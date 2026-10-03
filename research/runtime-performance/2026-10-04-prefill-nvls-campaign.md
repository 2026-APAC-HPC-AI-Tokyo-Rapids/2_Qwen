# Prefill-only NVLS対照（2026年10月4日）

条件commit `2433b2bdb85e1fa7b74d913da0c61eaa0e82e9f2`、[Actions37145086734](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37145086734)。ES-off対照37140273788のterminal監査後に投入。image4618103aを再利用し新buildなし、c48/P2TP4EP1DP1PP1/D2TP4EP1/EAGLE3/FP8KV/CUTLASSを固定、profileなし。Pだけ `--enable-nccl-nvls`、D変更なし。既に完了したES probeを外し再実行しない。

固定SGLang fc275674のentrypoints/engine.py1501–1510は既定NVLS0/CUMEM0、CLI flagでNVLS1を設定する。新imageログはNCCL2.28.9。公式[pinned NVLS実装](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/nvls.cc)のsupport判定と専用buffer allocationはcuMulticast/cuMemを直接使用し、一般NCCL_CUMEM_ENABLE allocator設定とは別。MNNVLのCUMEM要件と混同せずCUMEM1を追加しない。ALGO/PROTO強制、symm_mem、graph設定の変更もない。PのみINFO INIT,NVLSで初期化証跡を残し、per-collective TRACEは行わない。CUDA graph登録経路はnativeに存在するが実機成功はterminal確認事項。

NVLS support初期化は各collectiveでの実NVLS選択証明とは区別する。通常非profile公式スコアを同image対照と比較し、有望なら同条件反復を確認後に採否判断する。profileでARが約23%だったことは候補の根拠であり、改善率の保証ではない。NVLSをMoE GEMM TFLOP/s改善と呼ばない。

変更2files、実P/D argv fixture・187tests・Ruff・bash・diffと独立review通過。Luna監視、firstfatal/terminalの公式parser・image一致・raw request・8greedy・全checksum・cleanup監査まで継続する。現時点の性能向上は未確認。

## terminal監査と採否

37145086734はActions/ASPIRE/回収/正本/cleanup成功。PBS216075 Exit0/11m59、同image4618103a・build親b557fcdを確認。全48manifest対象hash、公式parser再計算と保存score一致。Pだけenable_nccl_nvls=True、D=False、実設定の差はPflagとランダムseed・観測内部統計のみ。P2それぞれNVLS rank allocation/nvlsRanks4を確認した。custom_all_reduce_v2のmulticast=Trueは別のworkspace capabilityであり、NVLS成立証拠に使わない。8件greedy各64tokensは対照とtoken・logprobまで一致した（有限診断であり一般品質証明ではない）。runtime削除・残留PBSなし。

根拠を比較し、NVLSは**改善未確認として未採用、追加paid反復なし**と判断した。ESの実装・実機gateは成立したが不採用、shuffle融合はCPU準備のみで単独build/GPU保留。現baseline ESoff/NVLSoffを維持する。FP8 peakからの差は帯域・padding・schedule・通信・launch等を含み、そのまま回収可能なGEMM改善余地ではない。実装branchは再現用保持しmainへmergeしない。

### P-only NVLS独立性能比較（初回pair）

baselineoff 37140273788 → NVLS 37145086734。raw読取のみ、公式scoreとAIPerf exportを照合した。image/source/config/品質/terminal正当性はaudit担当の別監査に従う。今回は通常非profile run。SGLang custom allreduceのmulticastTrueとNCCL NVLSを混同せず、NCCL NVLS initialization成功と実collectiveごとのalgorithm選択も分ける。新kernel profileは無い。

| metric | off | NVLS | 評価 |
|---|---:|---:|---|
|公式score tokens/s|2679.351|2686.426|+0.2640%、単pair|
|総output tokens/s|2708.634|2715.637|+0.2586%、公式scoreとは別|
|requests|1665|1672|+7|
|joint good fraction|98.9189%|98.9234%|両run不適格request18件|
|TTFT avg ms|1561.865|1557.215|-0.298%|
|TTFT p50 / p95 / p99 ms|1539.920 / 1840.116 / 3102.198|1529.416 / 1800.256 / 3112.904|p95改善方向、p99は小悪化|
|ITL avg ms|9.766900|9.719715|-0.483%|
|ITL p50 / p95 / p99 ms|9.710666 / 11.548058 / 12.248866|9.710408 / 11.160650 / 11.718369|p50ほぼ同じ、tail改善方向|
|request latency avg ms|3505.478|3491.421|-0.401%|
|effective P concurrency avg|21.1526|21.1443|ほぼ同じ|
|effective D concurrency avg|26.3241|26.2647|ほぼ同じ|

**僅かな改善方向はあるが、NVLSで性能向上を確認したとは言えない。** 単一off/on runでvarianceや有意性が分からない。TTFT p99と不適格件数は改善しておらず、明確なP待ち削減も見えていない。request数の+7によるthroughput微差は閉ループ到着/schedulingやrun境界変動でも生じ得る。

### benchmark窓のlog sample

AIPerf start/endと同じnaive時刻体系のSGLang TP0 logで選別した（logは秒精度なので境界の1batch程度は曖昧）。以下はunweighted sample meanであり、時間重み付きoccupancy/孤立service timeではない。

| metric | off | NVLS |
|---|---:|---:|
|P0/P1 samples|415 / 411|408 / 409|
|P0/P1 queue mean|1.540 / 1.946|2.017 / 1.711|
|P pooled queue mean|1.742|1.864|
|P pooled8192batch fraction|93.099%|98.164%|
|D0/D1 running request mean|15.080 / 15.208|14.603 / 15.836|
|D0/D1 accept len mean|2.23208 / 2.22144|2.22651 / 2.23025|
|D pooled accept len|2.22676|2.22835|
|D pooled transfer req|7.840|8.089|
|D queue|両worker0|両worker0|

NVLSでP batchがより充填されるというschedule差があり、sourceのARだけの費用改善とは切り分けられない。P0queueは増えP1queueは減り、pooledも減っていない。D acceptlenはほぼ同じで、D0/D1負荷分配は少し変わる。P-only knobに明確なdecode側品質悪化やqueue形成が伴う様子は無いが、大きいP改善を示す根拠も無い。

### 反復価値と推奨

**現時点では追加GPU反復を保留し、NVLSを既定採用しないことを推奨する。** 成立・greedy品質は別監査で確認できたが、score差+0.264%、平均latency差も0.5%未満で、joint不適格requestは両18件。PqueueとTTFT p99に改善根拠が無く、初期profileの大きなAR費用をtask gainへ転換できたとは言えない。単pairなのでnoise量自体も未推定であり、「有意差なし」という統計的断定とは異なる。

新buildは不要だが、観測PBS所要約12分・16GPU割当から追加off/on pairは約24分、約6.4GPU時間の資源を使う。小さな方向を確認するだけの単独追加pairの費用対効果は弱い。後続の関連E2Eで比較機会が生じるなら、同image・同node pair/clock policy・c48/seed/全CLI・ESoff固定、finite probe無し、NCCL初期化loggingも同じにし、on→offの逆順を使う再比較は可能。そこで微差が維持されても少数runの限界を明示する。大量反復や追加Nsight captureは今回勧めない。

根拠script `compare.py` と `comparison.json`。raw正本は非変更・再DLなし。profile時間からscore gainへ外挿していない。
