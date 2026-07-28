# 現在の SGLang SIF ビルド構成

ここでいう「現在」は `main` の [Genkai workflow](../../.github/workflows/genkai.yml)、[build script](../../scripts/build-sglang-sif.sh)、[SIF recipe](../../containers/sglang.def) を指す。

## ビルド入力

SIF は次の三つを入力として作る。

- `containers/sglang.def`
- recipe が取得する upstream SGLang `main`
- `patches/sglang/*.patch`

base image は `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` である。Python 3.12 と `uv` を使い、SGLang は editable install する。GPU JIT を実行時に利用できるよう、CUDA development environment と build tool も残す。

SGLang の Rust extension は debug symbol を保持する。解決済み Python dependency は `/opt/sglang/requirements-frozen.txt` に記録する。FlashInfer は Python package、cubin、JIT cache artifact の version を揃え、Hopper 向け kernel artifact はビルド中に取得する。

将来の比較に備え、次の通信 backend も導入する。

- DeepEP
- MSCCL++
- Mooncake
- NIXL

これにより、通常の tensor parallelism に加え、expert parallelism や prefill/decode 分離に必要な backend を、実行引数や SGLang を変更した段階で試せる。

## デバッグ用構成の境界

SIF には libc debug symbols と、NCCL、OpenMPI、UCX、PMIx、libfabric、RDMA の user-space library、header、provider を含める。

次のものは実行 host の責務であり、SIF には封入しない。

- gdb、Nsight Systems、Nsight Compute、strace、perf
- hardware 診断 CLI
- GPU と RDMA の kernel driver

## login node でのビルド

新規ビルドは `genkai` label の self-hosted runner から login node 上で行う。作業領域は run ごとの `/dev/shm/qwen-sglang-build/run-<run-id>-<attempt>`、共通の Singularity cache は `/dev/shm/qwen-sglang-build/cache` である。

Singularity CE 4.1.3 で recipe から sandbox を作り、root filesystem を SquashFS に変換して SIF を組み立てる。SIF の test は GPU に依存せず、主要 Python package、通信 backend、MPI、UCX、`uv` を検証する。

完成後は共有 filesystem 上の一時ファイルへ copy し、`${SIF_ARCHIVE_DIR}/<run-number>.sif` へ rename する。既定の保存先は `/fast/PROJECT/sglang-sifs` である。build directory は成功・失敗にかかわらず、その run 固有の directory だけを cleanup する。

## provenance

SIF 内の `build-provenance.txt` には次を記録する。

- workflow run ID、attempt、run number
- upstream SGLang commit
- recipe SHA-256
- patch manifest の SHA-256
- CUDA base image
- Singularity version
- UTC build time

Actions artifact には build log、provenance、SIF metadata、runtime test log を残す。

## 新規ビルドと再利用

次の場合は新しい SIF を作る。

- push の先頭コミットメッセージが `[require-build]` で始まる
- 手動実行で `require_build=true` を指定する
- 再利用可能な SIF が見つからない

それ以外では、現在のコミットから第一親履歴を遡る。候補 run について、SIF が実在し、`Build and publish SGLang SIF` または `Hardlink the selected SIF` step が成功していることを確認する。最も近い候補を現在の run number 名へ hardlink する。

再利用は source、recipe、patch、依存関係の互換性を自動判定しない。これらを変えた場合は `[require-build]` を指定する。benchmark 条件や runtime script だけを変える場合は通常の push とし、既存 SIF を再利用できる。
