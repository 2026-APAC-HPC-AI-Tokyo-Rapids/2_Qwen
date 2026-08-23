# Qwen 3.5 with SGLang on 玄界 / ASPIRE 2A+ / Firmus

玄界、ASPIRE 2A+、Firmusで同じQwen 3.5 workloadを実行するリポジトリです。実行入口は [Qwen workflow](.github/workflows/qwen.yml) に統合されています。

コンテナはAWS Singaporeリージョンの一時的な`m7i.2xlarge`から、root権限、Buildahの`chroot` isolation、OverlayFSを使って構築します。Buildahが出力したOCI layoutをSkopeoで正規化し、Umociで検証したroot filesystemをSquashFS化します。完成物の再利用正本は玄界へ保存し、private S3 bucketは全クラスタへの1日限りの配布に使います。各クラスタは実行ごとにOIDCで短期AWS資格情報を取得し、1時間の署名付きURLからSquashFSを必ず新規ダウンロードした後、GPU計算ノード上のEnrootへ`.sqsh`を直接渡してFUSE mountします。クラスタのログインノードへAWS CLIは導入しません。Singularity/SIFとrootless VFSは現行経路では使用しません。

モデル重みはまだ配置していないため、現在のschedulerジョブは`python -m sglang.check_env`だけを実行します。GPUやモデルを使うbenchmark経路へ戻すまでは、環境検査の結果を性能値として扱いません。

pushで計算を起動する場合は`[cluster:genkai]`、`[cluster:aspire2ap]`、または`[cluster:firmus]`をコミットメッセージへ付けます。複数指定すれば独立したrunnerから並行実行します。`[nodes:1]`、`[nodes:2]`、`[nodes:4]`でノード数を選び、省略時は2ノードです。Firmusは2ノードしかないため1または2を指定します。新しいコンテナが必要な変更では、コミットメッセージの先頭を`[require-build]`にします。クラスタタグのないpushは検証だけを実行し、`[skip ci]`を先頭に付けたpushではGitHub Actions自体を起動しません。

AWS資源、費用、寿命、資格情報境界は [AWS SquashFSビルド](research/aws-container-build.md)、実機上の制約は [OCI / Enroot移行記録](research/container-runtime-migration.md)、従来構成に至った履歴は`research/build-performance/`と`research/runtime-performance/`を参照してください。
