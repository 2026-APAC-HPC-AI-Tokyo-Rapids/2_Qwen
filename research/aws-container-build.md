# AWS SquashFSビルド

## 構成

GitHub Actionsのhosted runnerは制御と小さなsource archiveの作成だけを行う。新規buildが必要な場合、Singapore (`ap-southeast-1`) のdefault public subnetへ`m7i.2xlarge`を1台起動し、SSM経由でbuildを実行する。instanceは8 vCPU、32 GiB RAM、300 GiB gp3 root volume（6,000 IOPS、500 MiB/s）を持ち、inbound portとSSH keyは持たない。

AWS VMではrootを使用し、Buildah `chroot` isolationとOverlayFSでbuildする。VFS、fakeroot、user namespace向けownership緩和は使わない。OCI layoutはSkopeo copyとUmoci statで検証し、Buildahでmountした同じroot filesystemを8 worker、8 GiB cacheの`mksquashfs`で圧縮する。32 GiB RAMでは展開rootfs約26.6 GBに加えてlayer、OCI layout、圧縮入出力の全てをtmpfsへ同時保持できないため、全RAM buildではなく高速gp3を使う。

instanceはbuild成功時にActionsがterminateし、失敗時にもtrapでterminateする。さらにinstance内に6時間後のshutdownを登録し、Actions runner自体が消失した場合の上限を設ける。root EBS volumeは`DeleteOnTermination=true`であり、VMとEBSの常時課金はない。S3のinputとクラスタ配布用SquashFSは1日、build logは30日で期限切れになる。再利用対象のSquashFSとsidecarの正本は玄界の`/home/pj26000080/share/sglang-containers`へrun ID名で保存する。

新規buildではprivate ECRの`apac-qwen-build-cache`をBuildahの`--cache-from`と`--cache-to`へ指定し、変更されていない中間layerをVM間で再利用する。repositoryは最新64 imageだけを残すlifecycle policyを持つ。EBSを常設せず、実際に保存された圧縮layer分だけを課金対象にするため、300 GiB gp3を保持する方式より安い。cacheがない初回buildはcold buildとなり、以後の`[require-build]`でtoolchainやPython dependency layerが一致するときだけ短縮される。完成SquashFSの再利用判定とは独立している。

## 費用の目安

2026-08-24時点のSingapore Linux on-demand価格は`m7i.2xlarge`が0.504 USD/hourである。gp3は300 GiB、追加3,000 IOPS、追加375 MiB/sをAWSの30日月で時間按分すると0.090 USD/hour、public IPv4は0.005 USD/hourなので、VM稼働中は合計0.599 USD/hourとなる。実buildで得た12,718,911,488 byte imageをAWS CLI既定の8 MiB multipart partでuploadするS3 Tier 1 requestは約0.0076 USDである。したがってbuildだけの概算は、1、2、4時間なら約0.607、1.206、2.404 USDとなる。S3 artifact保管料はユーザー指定により含めない。

EC2と同一SingaporeリージョンのS3間転送は無料である。一方、S3から外部へこのimageを1回downloadすると12.719 GBとなり、account全体の月100 GB無料枠を使い切った後のSingapore先頭tier 0.120 USD/GBでは1.526 USDかかる。新規buildでは玄界の再利用archiveへの保存が1回加わるため、3クラスタを実行するrunの外部downloadは合計4回、6.105 USDである。再利用runではarchive保存を行わず3クラスタ分の3回、4.579 USDとなる。無料枠が残っていればこれらの転送料は0 USDである。料金根拠はAWSの [EBS pricing](https://aws.amazon.com/ebs/pricing/)、[S3 pricing](https://aws.amazon.com/s3/pricing/)、公開Pricing APIのAWSDataTransfer Singapore offerである。

実測cold buildはinstance稼働1,556秒、build job 26分53秒、AWS内build小計0.2665 USDだった。ECR cacheがある同一sourceのwarm buildは28 step中2から23までcache hitとなり、instance稼働869秒、build job 15分26秒、build小計0.1522 USDだった。したがって3クラスタへ新規buildを配布する1 runの総AWS費用は、月100 GB無料枠が残っていればcold 0.267 USDまたはwarm 0.152 USD、無料枠消化後はcold 6.372 USDまたはwarm 6.257 USDが目安になる。S3 artifact保管料は含めない。

ECR cacheのSingapore保管単価は0.10 USD/GB-monthである。初回成功build後は27個のcache imageについて`describe-images`の累積sizeが233.4 GBに見えるが、共有layer digestを重複排除した圧縮blobは13.103 GBであり、実ストレージ相当は約1.31 USD/月である。これはbuild 1回ごとの費用ではなく、保持中だけ発生する唯一のbuild cache常時費用である。64 imageの保持上限で変更後のlayer増加を制限する。料金根拠はAWSの [ECR pricing](https://aws.amazon.com/ecr/pricing/) と公開Pricing APIのAmazonECR Singapore offerである。

workflowはinstanceの実際のlaunch時刻から終了後までを秒で測り、`scripts/estimate-aws-build-cost.sh`でEC2、EBS、public IPv4、multipart requestを分離した`aws-build-cost.json`をbuild log artifactとJob Summaryへ残す。最終報告ではこの実測秒数と実際のSquashFS sizeを使い、archive保存と選択したクラスタへの外部転送を加えた値も再計算する。Cost Explorerへの反映には時間差があるため、この値は請求明細ではなく公開単価による見積りである。

`m7i-flex.2xlarge`は0.4788 USD/hourだがEBS baselineが1,250 Mbpsで、`m7i.2xlarge`の2,500 Mbpsより低い。差額は0.0252 USD/hourに留まり、container buildは大量の小ファイルI/Oを伴うため`m7i.2xlarge`を選ぶ。

## 資格情報と配布

CloudFormation template `aws/qwen-build-infrastructure.yml`はprivate S3 bucket、egress-only security group、EC2 instance profile、GitHub OIDC provider、Actions roleを作る。Actions roleは対象repositoryだけをtrustし、build用instanceの起動・終了、SSM command、対象bucketの操作だけを許可する。privateなSGLang sourceは、そのrepository自身のworkflowが管理する玄界上のread-only bare mirrorからQwen workflowが入力archiveへ複製する。消費側repositoryからsource repositoryへ書き込まず、広いPATや無効化されているdeploy keyに依存しない。

玄界、ASPIRE 2A+、Firmusの各self-hosted runnerも同じOIDC roleを一時的にassumeする。ログインノードへAWS CLIは導入せず、一時資格情報を使うBoto3 APIで64 MiB part、最大8並列のuploadとdownloadを行う。sidecarとSquashFSはrun固有directoryへ毎回downloadする。AWS資格情報はscheduler jobへ渡さず、計算ノードへ渡すのはdownload済み`.sqsh`、run固有Enroot設定、実行scriptだけである。

`[require-build]`が最新commit messageの先頭にある場合は必ず新規buildする。それ以外では第一親履歴を遡り、過去の成功runと対応する玄界ローカルのSquashFSについてsidecar、file size、SHA-256が一致する最近傍の成果物を再利用する。選んだローカル成果物は現在run IDへハードリンクしてから一時S3 objectとしてuploadするため、各クラスタ側のdownloadは省略しない。クラスタタグがないpushは静的検証だけを行い、AWS資源を起動しない。
