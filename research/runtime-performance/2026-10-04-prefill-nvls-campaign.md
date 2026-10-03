# Prefill-only NVLS対照（2026年10月4日）

条件commit `2433b2bdb85e1fa7b74d913da0c61eaa0e82e9f2`、[Actions37145086734](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37145086734)。ES-off対照37140273788のterminal監査後に投入。image4618103aを再利用し新buildなし、c48/P2TP4EP1DP1PP1/D2TP4EP1/EAGLE3/FP8KV/CUTLASSを固定、profileなし。Pだけ `--enable-nccl-nvls`、D変更なし。既に完了したES probeを外し再実行しない。

固定SGLang fc275674のentrypoints/engine.py1501–1510は既定NVLS0/CUMEM0、CLI flagでNVLS1を設定する。新imageログはNCCL2.28.9。公式[pinned NVLS実装](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/nvls.cc)のsupport判定と専用buffer allocationはcuMulticast/cuMemを直接使用し、一般NCCL_CUMEM_ENABLE allocator設定とは別。MNNVLのCUMEM要件と混同せずCUMEM1を追加しない。ALGO/PROTO強制、symm_mem、graph設定の変更もない。PのみINFO INIT,NVLSで初期化証跡を残し、per-collective TRACEは行わない。CUDA graph登録経路はnativeに存在するが実機成功はterminal確認事項。

NVLS support初期化は各collectiveでの実NVLS選択証明とは区別する。通常非profile公式スコアを同image対照と比較し、有望なら同条件反復を確認後に採否判断する。profileでARが約23%だったことは候補の根拠であり、改善率の保証ではない。NVLSをMoE GEMM TFLOP/s改善と呼ばない。

変更2files、実P/D argv fixture・187tests・Ruff・bash・diffと独立review通過。Luna監視、firstfatal/terminalの公式parser・image一致・raw request・8greedy・全checksum・cleanup監査まで継続する。現時点の性能向上は未確認。
