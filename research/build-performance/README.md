# Genkai における SGLang SIF ビルドの調査記録

このディレクトリは、Qwen 3.5 benchmark に使う SGLang 環境を再現可能な SIF として構築し、ビルド時間を必要な場合だけ負担するための設計を記録する。

現在の workflow は、recipe、upstream SGLang `main`、`patches/sglang/*.patch` から完成済み SIF を作る。通常の push では第一親履歴上の最も近い SIF を再利用し、明示した場合だけ新しくビルドする。

現行構成は [current-build.md](current-build.md)、その構成に至った実験と失敗は [trial-record.md](trial-record.md) を参照する。

## この構成で確立していること

- 公開済み SGLang image を親にせず、CUDA 13.0.1 と Ubuntu 24.04 を基準に source から SGLang を導入する。
- SIF は Genkai の login node で Singularity CE 4.1.3 を使ってビルドし、PJM と GPU 計算ノードはビルドに使わない。
- build tree、Singularity cache、一時領域は node-local な `/dev/shm/qwen-sglang-build` に置く。
- 完成 SIF は `/fast/PROJECT/sglang-sifs/<run-number>.sif` に保存し、同一 filesystem 内で atomic に公開する。
- SIF には source、recipe、patch、build tool、workflow run を追跡する provenance を格納する。
- SGLang、recipe、依存関係を変更する push は、コミットメッセージの先頭に `[require-build]` を付ける。

`research/**` を含む Markdown だけの push では、[workflow](../../.github/workflows/genkai.yml) は起動しない。
