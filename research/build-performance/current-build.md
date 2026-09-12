# SGLang SIF ビルド現行構成

この文書は、OCI / Enroot移行前の正本を履歴として残したものである。現行構成については [OCI / Enroot移行記録](../container-runtime-migration.md) を参照のこと。

ここでいう「現在」は、当時の`main`における [Genkai workflow](../../.github/workflows/genkai.yml)、[build script](../../scripts/build-sglang-sif.sh)、および [SIF recipe](../../containers/sglang.def) を指している。

## ビルド入力

SIFは次の3点を入力として生成していた。

- `containers/sglang.def`
- 共有bareミラーから取得するOrganization管理の`5_SGLang_Src/main`
- `patches/sglang/*.patch`

ベースイメージには`nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04`を採用した。Python 3.12および`uv`を使用し、SGLangはeditable installとした。GPU JITを実行時に利用できるよう、CUDA開発環境およびビルドツールもコンテナ内に保持した。

SGLangのRust拡張はデバッグシンボルを保持させた。解決済みのPython依存パッケージは`/opt/sglang/requirements-frozen.txt`に記録した。FlashInferはPythonパッケージ、cubin、およびJITキャッシュ成果物のバージョンを整合させ、Hopperアーキテクチャ向けカーネル成果物はビルド中に事前取得した。

将来的な性能比較に備え、以下の通信バックエンドも導入した。

- DeepEP
- MSCCL++
- Mooncake
- NIXL

これにより、通常のテンソル並列（TP）に加え、エキスパート並列（EP）やPrefill–Decode分離（PD分離）に必要なバックエンドを、実行時引数やSGLangソースの変更に応じて即座に検証できる構成とした。

## デバッグ用構成の境界

SIF内にはlibcのデバッグシンボル、ならびにNCCL、Open MPI、UCX、PMIx、libfabric、RDMAのユーザースペースライブラリ、ヘッダー、プロバイダを含めた。

一方、以下の要素は実行ホスト側の責務とし、SIF内には含めない設計とした。

- gdb、Nsight Systems、Nsight Compute、strace、perf
- ハードウェア診断CLI
- GPUおよびRDMAのカーネルドライバ

## ログインノードでのビルド

新規ビルドは、`genkai`ラベルのself-hosted runnerから玄界のログインノード上で実行した。共有ミラー`/fast/PROJECT/git-mirror/5_SGLang_Src.git`はOrganization管理下の`5_SGLang_Src`がpush契機で自動更新し、本リポジトリからは読み取り専用で参照した。作業領域にはランごとの`/dev/shm/qwen-sglang-build/run-<run-id>-<attempt>`、共通のSingularityキャッシュには`/dev/shm/qwen-sglang-build/cache`を用いた。

ビルドスクリプトは共有ミラーからラン固有のソースツリーを生成し、`main`ブランチの先端をdetached checkoutした。レシピはそのツリーを`%files`ディレクティブにより`/opt/sglang/source`へと取り込んだ。DeepEPはこの経路を用いず、公式リポジトリの固定コミットアーカイブを直接取得した。

Singularity CE 4.1.3を用いてレシピからsandboxを構築し、root filesystemをSquashFSへ変換してSIFを生成した。SIFのテストはGPUに依存せず、主要Pythonパッケージ、通信バックエンド、MPI、UCX、および`uv`の健全性を検証した。

ビルド完了後は共有ファイルシステム上の一時ファイルへコピーし、`${SIF_ARCHIVE_DIR}/<run-number>.sif`へとアトミックにリネームして公開した。既定の保存先は`/home/ACCOUNT/share/sglang-sifs`とした。ビルドディレクトリは、成否にかかわらず当該ラン固有のディレクトリのみを確実にクリーンアップした。

## 来歴情報（provenance）

SIF内部の`build-provenance.txt`には以下の情報を記録した。

- ワークフローのrun ID、attempt、run number
- SGLangのコミットOID
- レシピファイルのSHA-256
- パッチマニフェストのSHA-256
- CUDAベースイメージ名
- Singularityのバージョン
- ビルド実行時刻（UTC）

ActionsのArtifactには、ビルドログ、来歴情報、SIFメタデータ、およびランタイムテストログを保存した。

## 新規ビルドと再利用

以下の条件に該当する場合に新規SIFを構築した。

- push時の最新コミットメッセージが`[require-build]`で始まる場合
- 手動実行時に`require_build=true`を指定した場合
- 再利用可能な既存SIFが存在しない場合

それ以外の場合は、現在のコミットから第一親（first-parent）履歴を遡り、候補ランについてSIFが実在し、かつ`Build and publish SGLang SIF`または`Hardlink the selected SIF`ステップが成功していることを確認した上で、直近の有効なSIFを今回のrun number名へとハードリンクして再利用した。

再利用機構はソースコード、レシピ、パッチ、依存関係の互換性を自動判定しない。これらを変更した場合は明示的に`[require-build]`を指定する運用とした。一方、ベンチマーク引数や実行時スクリプトのみを変更する場合は通常のpushとし、既存SIFをそのまま再利用可能とした。
