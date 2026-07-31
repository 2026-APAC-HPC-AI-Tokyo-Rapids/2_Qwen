# Genkai における SGLang SIF ビルドの調査記録

このディレクトリは、Qwen 3.5 benchmark に使う SGLang 環境を再現可能な SIF として構築し、ビルド時間を必要な場合だけ負担するための設計を記録する。

現在の workflow は、recipe、Org 管理の `5_SGLang_Src/main`、`patches/sglang/*.patch` から完成済み SIF を作る。通常の push では第一親履歴上の最も近い SIF を再利用し、明示した場合だけ新しくビルドする。

現行構成は [current-build.md](current-build.md)、その構成に至った実験と失敗は [trial-record.md](trial-record.md) を参照する。

## この構成で確立していること

- 公開済み SGLang image を親にせず、CUDA 13.0.1 と Ubuntu 24.04 を基準に source から SGLang を導入する。
- SIF は Genkai の login node で Singularity CE 4.1.3 を使ってビルドし、PJM と GPU 計算ノードはビルドに使わない。
- build tree、Singularity cache、一時領域は node-local な `/dev/shm/qwen-sglang-build` に置く。
- 完成 SIF は `/fast/pj26000080/sglang-sifs/<run-number>.sif` に保存し、同一 filesystem 内で atomic に公開する。
- SIF には source、recipe、patch、build tool、workflow run を追跡する provenance を格納する。
- SGLang、recipe、依存関係を変更する push は、コミットメッセージの先頭に `[require-build]` を付ける。

## SGLang source mirror

Genkai では `5_SGLang_Src` の bare mirror を Git object cache として共有する。

```text
/fast/pj26000080/git-mirror/5_SGLang_Src.git
```

workflow は毎回、リポジトリ単位の `flock` を取得して Org 側の
`2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src` から `fetch --prune` し、
`main` が存在することを検証する。新規 SIF ビルドでは同じ lock の共有 lock を
取得し、更新済み mirror から run 固有の `/dev/shm` 作業ディレクトリへ clone
して `main` の先端を detached checkout する。recipe はそのローカル tree を
`%files` で取り込み、SGLang の公式リポジトリを直接 clone しない。

DeepEP は mirror の対象外である。現行どおり公式
`deepseek-ai/DeepEP` の固定 commit
`9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee` の archive を recipe 内で直接取得する。
MSCCL++、PyPI/uv、FlashInfer、Hugging Face のモデルとデータセットもこの Git
mirror の対象外である。

### 初回構築

初回だけ Genkai へ SSH し、次を実行する。この処理は workflow YAML に含めない。

```bash
install -d -m 2775 /fast/pj26000080/git-mirror

git clone --mirror \
  https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src.git \
  /fast/pj26000080/git-mirror/5_SGLang_Src.git

touch /fast/pj26000080/git-mirror/5_SGLang_Src.lock
chgrp -R pj26000080 \
  /fast/pj26000080/git-mirror/5_SGLang_Src.git \
  /fast/pj26000080/git-mirror/5_SGLang_Src.lock
chmod -R g+rwX /fast/pj26000080/git-mirror/5_SGLang_Src.git
chmod 0664 /fast/pj26000080/git-mirror/5_SGLang_Src.lock
git -C /fast/pj26000080/git-mirror/5_SGLang_Src.git \
  config core.sharedRepository group
```

`git clone --mirror` は初回だけであり、通常運用では
`scripts/update-sglang-source-mirror.sh` が mirror を更新する。fetch が失敗した
場合、古い ref へ黙ってフォールバックせず workflow を失敗させる。この cache は
取得量と準備時間を減らすためのもので、GitHub 非接続時の継続動作は保証しない。

### 障害時の再作成

bare repository が壊れた場合は、実行中 workflow がないことを確認してから、壊れた
repository と lock file を別名へ退避する。その後、上記の初回構築手順で新しい
mirror と lock file を作り、次を確認してから workflow を再実行する。

```bash
git -C /fast/pj26000080/git-mirror/5_SGLang_Src.git fsck --full
git -C /fast/pj26000080/git-mirror/5_SGLang_Src.git \
  show-ref --verify refs/heads/main
test "$(git -C /fast/pj26000080/git-mirror/5_SGLang_Src.git \
  config --get core.sharedRepository)" = group
```

`research/**` を含む Markdown だけの push では、[workflow](../../.github/workflows/genkai.yml) は起動しない。
