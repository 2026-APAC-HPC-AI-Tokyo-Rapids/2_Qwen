# Genkai における SGLang SIF ビルドの調査記録（旧構成）

この記録は、OCI / Enroot移行以前の旧構成に関する履歴である。現行構成については [OCI / Enroot移行記録](../container-runtime-migration.md) を参照のこと。

本ディレクトリは、Qwen 3.5ベンチマークに用いるSGLang実行環境を再現可能なSIF（Singularity Image Format）として構築し、必要な場合にのみビルドコストを負担するための設計内容を記録している。

当時のワークフローでは、Singularityレシピ、Organization管理の`5_SGLang_Src/main`、および`patches/sglang/*.patch`から完成済みSIFを生成していた。通常のpushでは第一親（first-parent）履歴上で直近のSIFを再利用し、明示的に指定した場合のみ新規ビルドを実行する運用とした。

当時の構成詳細は [current-build.md](current-build.md)、当該構成に至った実験と試行錯誤の記録は [trial-record.md](trial-record.md) を参照のこと。

## この構成で確立した事項

- 公開済みSGLangイメージをベースとせず、CUDA 13.0.1およびUbuntu 24.04を基準としてソースコードからSGLangをビルド・導入する。
- SIFのビルドは玄界のログインノード上でSingularity CE 4.1.3を用いて実行し、PJMスケジューラおよびGPU計算ノードはビルド処理に使用しない。
- ビルドツリー、Singularityキャッシュ、および一時領域は、ノードローカルなtmpfsである`/dev/shm/qwen-sglang-build`上に配置する。
- 完成したSIFは`/home/pj26000080/share/sglang-sifs/<run-number>.sif`に保存し、同一ファイルシステム内でのアトミックな移動（rename）により公開する。
- SIF内には、ソースコード、レシピ、パッチ、ビルドツール、およびGitHub Actionsワークフローの実行情報を追跡する来歴（provenance）を格納する。
- SGLang本体、レシピ、または依存関係を変更するpushでは、コミットメッセージの先頭に`[require-build]`を付与する。

## SGLangソースミラー

玄界上では、`5_SGLang_Src`のbare mirrorをGitオブジェクトキャッシュとして共有している。

```text
/fast/pj26000080/git-mirror/5_SGLang_Src.git
```

Organization管理下の`5_SGLang_Src`リポジトリは、push発生時に自身のGitHub Actionsワークフローからこのミラーを自動更新する。本リポジトリのワークフローはこのミラーを読み取り専用として扱い、新規SIFビルド時のみラン固有の`/dev/shm`作業ディレクトリへとクローンして`main`ブランチの先端をdetached checkoutする。レシピ側はそのローカルツリーを`%files`ディレクティブで取り込み、ビルド中にSGLangの公式リポジトリを直接クローンすることはしない。

DeepEPはこのミラー管理の対象外である。従来どおり公式リポジトリ（`deepseek-ai/DeepEP`）の固定コミット`9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`のアーカイブをレシピ内で直接取得する。MSCCL++、PyPI/uv、FlashInfer、ならびにHugging Face上のモデルおよびデータセットも本Gitミラーの対象外である。

### 初回構築手順

初回のみ玄界へSSH接続し、以下のコマンドを実行する。なお、本処理はワークフローYAMLには含めない。

```bash
install -d -m 2775 /fast/pj26000080/git-mirror

git clone --mirror \
  https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/5_SGLang_Src.git \
  /fast/pj26000080/git-mirror/5_SGLang_Src.git

chgrp -R pj26000080 \
  /fast/pj26000080/git-mirror/5_SGLang_Src.git
chmod -R g+rwX /fast/pj26000080/git-mirror/5_SGLang_Src.git
git -C /fast/pj26000080/git-mirror/5_SGLang_Src.git \
  config core.sharedRepository group
```

`git clone --mirror`による初期化は初回限定の作業であり、通常運用では`5_SGLang_Src`へのpushを契機としてソースリポジトリ側のワークフローがミラーを更新する。本キャッシュの主目的は、ネットワーク取得量とビルド準備時間の削減である。

### 障害時の再作成手順

bareリポジトリが破損した場合は、ソースリポジトリ側のミラー更新ワークフローが実行中でないことを確認した上で、破損したリポジトリを別名へと退避する。その後、上記の初回構築手順に沿って新しいミラーを再作成し、以下の整合性を確認してからワークフローを再実行する。

```bash
git -C /fast/pj26000080/git-mirror/5_SGLang_Src.git fsck --full
git -C /fast/pj26000080/git-mirror/5_SGLang_Src.git \
  show-ref --verify refs/heads/main
test "$(git -C /fast/pj26000080/git-mirror/5_SGLang_Src.git \
  config --get core.sharedRepository)" = group
```

なお、`research/**`配下のMarkdownファイルのみを変更したpushでは、[ワークフロー](../../.github/workflows/genkai.yml) は起動しない。
