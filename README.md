# Qwen 3.5 with SGLang on 玄界 / ASPIRE 2A+ / Firmus

玄界、ASPIRE 2A+、Firmusで同じQwen 3.5 workloadを実行するリポジトリです。実行入口は [Qwen workflow](.github/workflows/qwen.yml) に統合されています。

コンテナは [Dockerfile](containers/Dockerfile) からrootless Buildahの`chroot` isolationで作成します。Buildahが出力したOCI layoutをSkopeoで正規化し、Umociで展開検証したroot filesystemをSquashFS化して、GPU計算ノード上のEnrootで実行します。Singularity/SIFは現行経路では使用しません。

モデル重みはまだ配置していないため、現在の検証は`Qwen/Qwen3.5-397B-A17B-FP8`の小さな設定・tokenizerファイルだけを配備し、SGLangを`--load-format dummy`およびrandom datasetで実行します。この結果は実モデルの性能値として扱いません。

pushで計算を起動する場合は`[cluster:genkai]`、`[cluster:aspire2ap]`、または`[cluster:firmus]`をコミットメッセージへ付けます。複数指定すれば独立したrunnerから並行実行します。`[nodes:1]`、`[nodes:2]`、`[nodes:4]`でノード数を選び、省略時は2ノードです。Firmusは2ノードしかないため1または2を指定します。新しいコンテナが必要な変更では、コミットメッセージの先頭を`[require-build]`にします。クラスタタグのないpushは検証だけを実行し、`[skip ci]`を先頭に付けたpushではGitHub Actions自体を起動しません。

設計と実機上の制約は [OCI / Enroot移行記録](research/container-runtime-migration.md)、従来構成に至った履歴は`research/build-performance/`と`research/runtime-performance/`を参照してください。
