# Prefill NCCL symmetric memoryの検証

ユーザーが[Stas Bekman投稿](https://x.com/StasBekman/status/2106978810247917610)の検証を指示した。公開embed APIで原投稿本文を取得し、そのリンク先[著者のprimary README](https://github.com/stas00/ml-engineering/blob/master/network/README.md#symmetric-memory)を確認。small/medium payloadのregistered NCCL symmetric bufferで速度向上を見込む説明だが、示されたH200/B200・TP8 micro測定をH100TP4/Qwen E2Eへ転用しない。NCCL2.27+/PyTorch2.9+を要し、実kernel使用はINFO/TUNINGの`AllReduce [Symmetric]`ログで確認する説明である。

同image182fbb6dのread-only CPU診断でtorch2.11.0+cu130/NCCL2.28.9を確認。最初のlogin Enroot診断はGPUdriverhook（NVIDIA_VISIBLE_DEVICES=none）がlibnvidia-ml無しで失敗し、voidによるGPU非要求診断で成功した。scheduler GPU障害やnewbuild理由ではない。sourcefc275674のfp8.py2413–2443はCUTLASS最終出力をuse_symmetric_memoryで割り当ててoutput引数へ渡し、routingpatchもoutputを維持する。DPattention無効ではallocation symmetric条件が成立する。DeepSeekV4向け[PR29007](https://github.com/sgl-project/sglang/pull/29007)のDeepGemm/Triton修正をこのCUTLASS試験へ持ち込まない。

既P-only NVLS37145086734（+0.264%単pair未採用）とは、NCCL allocator/register/graph同期経路の追加条件が異なる。現CustomAR共有buffer23.01MB/multicastTrueとも別。symm有効時通常allreduceはpyncclをcustomARより先に選ぶが、torch.compile枝は迂回して適合customAR優先を残す（parallel_state.py668–705/878–884）。4GiB/GPU target preallocationと登録費用を含め、flagのみで実適用や改善を断定しない。

## 通常性能条件

parent e484c760（routingon公式control）から`f229f661de9f4a1ebbfda59def7f43fa97497450` / [Actions37604857808](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115/37604857808) / PBS217308を投入した。runtime `pd-37604857808-1.GsyhWW9BWK`。P-only `--enable-symm-mem`を追加し、baseline実pairP030/D032へselectorを固定した。D/c48/P2TP4EP1/D2TP4/Pshuffle1/Proute1/FA3/FP8KV/EAGLE3BF16/profilefalse維持、customARdisable/NVLS/algoprotooverride無し、newcast無し。新sourcebuild不要。

51CPU tests（argv/orchestration/monitor/contracts）とbash/actionlint/diff通過。ASPIRE実sidecar/runtimeprovenanceでimage182fbb6d/size12231192576/buildsource9c5a5a3/sglangfc275674/commitf229f661/profilefalse、EC2/mirror/S3buildskippedを照合。PBSQ/aiq2/mem不足、030032各112CPU8GPU1880GB/excl/30minを確認。旧routingfixed216741/216810は取消せずLuna監視維持。

通常benchにはdebug/profilerを追加しない。terminal時parser/latency/精度診断/rawhash/actualargv/UUID/cleanupを監査し、baseline37576508318の2779.551953との一対比較は暫定とする。既定ログで実symmetric kernel証拠が得られなければ、別追跡profile条件のINFO/TUNINGで確認し、そのscoreを通常性能値として使わない。公式結果・実適用・主催者完全accuracy・採用は現時点未確認。
