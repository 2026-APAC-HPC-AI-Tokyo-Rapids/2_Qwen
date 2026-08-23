# Genkai における SGLang SIF ビルドの調査記録（旧構成）

この記録はOCI / Enroot移行前の履歴である。現行構成は [OCI / Enroot移行記録](../container-runtime-migration.md) を参照する。

このディレクトリは、Qwen 3.5 benchmark に使う SGLang 環境を再現可能な SIF として構築し、ビルド時間を必要な場合だけ負担するための設計を記録する。

現在の workflow は、recipe、Org 管理の `5_SGLang_Src/main`、`patches/sglang/*.patch` から完成済み SIF を作る。通常の push では第一親履歴上の最も近い SIF を再利用し、明示した場合だけ新しくビルドする。

現行構成は [current-build.md](current-build.md)、その構成に至った実験と失敗は [trial-record.md](trial-record.md) を参照する。

## この構成で確立していること

- 公開済み SGLang image を親にせず、CUDA 13.0.1 と Ubuntu 24.04 を基準に source から SGLang を導入する。
- SIF は Genkai の login node で Singularity CE 4.1.3 を使ってビルドし、PJM と GPU 計算ノードはビルドに使わない。
- build tree、Singularity cache、一時領域は node-local な `/dev/shm/qwen-sglang-build` に置く。
- 完成 SIF は `/home/ACCOUNT/share/sglang-sifs/<run-number>.sif` に保存し、同一 filesystem 内で atomic に公開する。
- SIF には source、recipe、patch、build tool、workflow run を追跡する provenance を格納する。
- SGLang、recipe、依存関係を変更する push は、コミットメッセージの先頭に `[require-build]` を付ける。

## SGLang source mirror

Genkai では `5_SGLang_Src` の bare mirror を Git object cache として共有する。

```text
/fast/PROJECT/git-mirror/5_SGLang_Src.git
```

Org 管理の `5_SGLang_Src` は push 後に自身の GitHub Actions からこの mirror を
更新する。このリポジトリの workflow は mirror を読み取り専用として扱い、新規
SIF ビルド時だけ run 固有の `/dev/shm` 作業ディレクトリへ clone して `main` の
先端を detached checkout する。recipe はそのローカル tree を `%files` で
取り込み、SGLang の公式リポジトリを直接 clone しない。

DeepEP は mirror の対象外である。現行どおり公式
`deepseek-ai/DeepEP` の固定 commit
`9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee` の archive を recipe 内で直接取得する。
MSCCL++、PyPI/uv、FlashInfer、Hugging Face のモデルとデータセットもこの Git
mirror の対象外である。

### 初回構築

初回だけ Genkai へ SSH し、次を実行する。この処理は workflow YAML に含めない。

```bash
install -d -m 2775 /fast/PROJECT/git-mirror

git clone --mirror \
  https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src.git \
  /fast/PROJECT/git-mirror/5_SGLang_Src.git

chgrp -R PROJECT \
  /fast/PROJECT/git-mirror/5_SGLang_Src.git
chmod -R g+rwX /fast/PROJECT/git-mirror/5_SGLang_Src.git
git -C /fast/PROJECT/git-mirror/5_SGLang_Src.git \
  config core.sharedRepository group
```

`git clone --mirror` は初回だけであり、通常運用では `5_SGLang_Src` の push を
契機にソースリポジトリ側の workflow が mirror を更新する。この cache は取得量と
準備時間を減らすためのものである。

### 障害時の再作成

bare repository が壊れた場合は、ソースリポジトリの mirror 更新workflowが実行中で
ないことを確認してから、壊れたrepositoryを別名へ退避する。その後、上記の初回構築
手順で新しいmirrorを作り、次を確認してからworkflowを再実行する。

```bash
git -C /fast/PROJECT/git-mirror/5_SGLang_Src.git fsck --full
git -C /fast/PROJECT/git-mirror/5_SGLang_Src.git \
  show-ref --verify refs/heads/main
test "$(git -C /fast/PROJECT/git-mirror/5_SGLang_Src.git \
  config --get core.sharedRepository)" = group
```

`research/**` を含む Markdown だけの push では、[workflow](../../.github/workflows/genkai.yml) は起動しない。
