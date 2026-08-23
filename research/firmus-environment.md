# Firmus実行環境

2026-08-23時点の`ssh firmus`はRocky Linux 9.8のPodman login containerで、Slurm 26.05.2を介して`g1`と`g4`の2計算ノードを利用する。各ノードは128 CPU、約2 TB memory、NVIDIA H200 141 GBを8基持つ。専有`hpcai` partitionの上限は17分、共有`shared` partitionの上限は30分であり、4ノード要求は不可能である。

`/home`と`/scratch`は同じBeeGFS上の同一user directoryを指す。login containerの`/tmp`は約63 GBだが`/dev/shm`は64 MBしかないため、container build領域を`/dev/shm`へ置いてはならない。高コスト成果物とrun directoryは`/scratch/home/ACCOUNT`、rootless build storageは容量を確認したnode-local filesystemへ置く必要がある。

`nvhpc-hpcx-cuda13/26.5` moduleにはOpen MPI 5.0.10rc2、UCX 1.21.0、NCCL 2.29.3、CUDA 13.2が含まれる。Slurm、Git、GitHub CLI、Jq、Curl、Wget、SquashFS toolsはlogin containerにある。

Qwen E2Eでlogin runnerに不足しているものはBuildah、Skopeo、Umoci、uv/uvxである。`/usr/bin/enroot`はlogin nodeとcompute nodeの双方にあるが、設定された`~/.local/lib/enroot`のruntime libraryと`~/.local/etc/enroot`のconfiguration/hooksがなく、実行できない。login containerにはGPU deviceと`nvidia-smi`がない一方、Slurm compute nodeには`/usr/bin/nvidia-container-cli`があり、GPUも正常に認識する。

`[cluster:firmus]`は、依存導入前にはlogin inventoryと短いSlurm GPU smokeを実行してartifactを回収し、その後に不足依存を明示して失敗する。Qwen本体のbuild・provision・benchmarkへは、不足依存とEnroot NVIDIA hookが揃い、公式E2E経路を検証してから接続する。
