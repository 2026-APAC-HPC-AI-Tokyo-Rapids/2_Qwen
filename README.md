# Qwen 3.5 with SGLang on 玄界 / ASPIRE 2A+ / Firmus

玄界、ASPIRE 2A+、Firmusで同じQwen 3.5 workloadを実行するリポジトリです。実行入口は [Qwen workflow](.github/workflows/qwen.yml) に統合されています。

コンテナはAWS Singaporeリージョンの一時的な`m7i.2xlarge`から、root権限、Buildahの`chroot` isolation、OverlayFSを使って構築します。Buildahが出力したOCI layoutをSkopeoで正規化し、Umociで検証したroot filesystemをSquashFS化します。玄界、ASPIRE 2A+、Firmusはそれぞれ自分のローカルarchiveだけを独立に検索し、全クラスタで再利用元が見つかった場合はbuildしません。1クラスタでも見つからない場合だけAWSで1回buildします。private S3 bucketはその新規buildを全クラスタへ1日限りで配布するときだけ使い、完了マーカーや永続保管は行いません。build時は3クラスタがBoto3で同じSquashFSを1回ずつ取得してローカルarchiveへ保存します。既存buildを使うrunではS3を使わず、各クラスタが自分のarchive内で再利用元sqshを今回run IDへハードリンクします。クラスタ間でcacheをコピーしません。GPU計算ノードではarchiveからrun directoryへ配備した`.sqsh`をEnrootへ直接渡してFUSE mountします。クラスタのログインノードへAWS CLIは導入しません。Singularity/SIFとrootless VFSは現行経路では使用しません。

モデル重みはまだ配置していないため、現在のschedulerジョブは`python -m sglang.check_env`だけを実行します。GPUやモデルを使うbenchmark経路へ戻すまでは、環境検査の結果を性能値として扱いません。

クラスタ処理を担当するjobは`[self-hosted, home-lab]`で起動し、`genkai-shell --github-shell`、`aspire2p-shell --github-shell`、`firmus-shell --github-shell`をcustom shellとして直接使います。checkoutとstep環境だけをrun固有一時領域へ転送し、SquashFS archive、Enroot設定、dataset、model、run directoryは各ログインノード上の既存パスを維持します。

各クラスタの`check_env`ログは一度GitHub Artifactへ回収し、選択した全クラスタの終了後にhome-lab runnerから`/srv/qwen-artifacts/<run-id>/<attempt>`へチェックサム付きで移管します。home RAIDへの配置に成功した後だけ、転送済みのGitHub Artifactを削除します。

pushで計算を起動する場合は`[cluster:genkai]`、`[cluster:aspire2ap]`、または`[cluster:firmus]`をコミットメッセージへ付けます。複数指定すれば独立したrunnerから並行実行します。`[nodes:1]`、`[nodes:2]`、`[nodes:4]`でノード数を選び、省略時は2ノードです。Firmusは2ノードしかないため1または2を指定します。新しいコンテナが必要な変更では、コミットメッセージの先頭を`[require-build]`にします。クラスタタグのないpushは検証だけを実行し、`[skip ci]`を先頭に付けたpushではGitHub Actions自体を起動しません。

AWS資源、費用、寿命、資格情報境界は [AWS SquashFSビルド](research/aws-container-build.md)、実機上の制約は [OCI / Enroot移行記録](research/container-runtime-migration.md)、従来構成に至った履歴は`research/build-performance/`と`research/runtime-performance/`を参照してください。
