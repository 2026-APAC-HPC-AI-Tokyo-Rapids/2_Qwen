# AWS SquashFSビルド

## 構成

GitHub Actionsのhosted runnerは制御と小さなsource archiveの作成だけを行う。新規buildが必要な場合、Singapore (`ap-southeast-1`) のdefault public subnetへ`m7i.2xlarge`を1台起動し、SSM経由でbuildを実行する。instanceは8 vCPU、32 GiB RAM、300 GiB gp3 root volume（6,000 IOPS、500 MiB/s）を持ち、inbound portとSSH keyは持たない。

AWS VMではrootを使用し、Buildah `chroot` isolationとOverlayFSでcold buildする。VFS、fakeroot、user namespace向けownership緩和は使わない。OCI layoutはSkopeo copyとUmoci statで検証し、Buildahでmountした同じroot filesystemを8 worker、8 GiB cacheの`mksquashfs`で圧縮する。32 GiB RAMでは展開rootfs約26.6 GBに加えてlayer、OCI layout、圧縮入出力の全てをtmpfsへ同時保持できないため、全RAM buildではなく高速gp3を使う。

instanceはbuild成功時にActionsがterminateし、失敗時にもtrapでterminateする。さらにinstance内に6時間後のshutdownを登録し、Actions runner自体が消失した場合の上限を設ける。root EBS volumeは`DeleteOnTermination=true`であり、VMとEBSの常時課金はない。S3のinputは1日、build logは30日で期限切れになる。再利用対象のSquashFSとsidecarだけは期限を設けずに残す。

## 費用の目安

2026-08-24時点のSingapore Linux on-demand価格は`m7i.2xlarge`が0.504 USD/hourである。gp3は300 GiB、追加3,000 IOPS、追加375 MiB/sをAWSの30日月で時間按分すると0.090 USD/hour、public IPv4は0.005 USD/hourなので、VM稼働中は合計0.599 USD/hourとなる。既知の12,718,784,512 byte imageをAWS CLI既定の8 MiB multipart partでuploadするS3 Tier 1 requestは約0.0076 USDである。したがってbuildだけの概算は、1、2、4時間なら約0.607、1.206、2.404 USDとなる。S3 artifact保管料はユーザー指定により含めない。

EC2と同一SingaporeリージョンのS3間転送は無料である。一方、S3から外部の3クラスタへ既知imageを各1回downloadすると38.156 GBとなり、account全体の月100 GB無料枠を使い切った後のSingapore先頭tier 0.120 USD/GBでは4.579 USDかかる。よって新規1時間buildと3クラスタへの配布を一体で数えると5.185 USD、無料枠が残っていれば0.607 USDが目安になる。再利用runはEC2/EBS費用がなく、外部転送だけが発生する。料金根拠はAWSの [EBS pricing](https://aws.amazon.com/ebs/pricing/)、[S3 pricing](https://aws.amazon.com/s3/pricing/)、公開Pricing APIのAWSDataTransfer Singapore offerである。

workflowはinstanceの実際のlaunch時刻から終了後までを秒で測り、`scripts/estimate-aws-build-cost.sh`でEC2、EBS、public IPv4、multipart requestを分離した`aws-build-cost.json`をbuild log artifactとJob Summaryへ残す。最終報告ではこの実測秒数と実際のSquashFS sizeを使い、3クラスタの外部転送を加えた値も再計算する。Cost Explorerへの反映には時間差があるため、この値は請求明細ではなく公開単価による見積りである。

`m7i-flex.2xlarge`は0.4788 USD/hourだがEBS baselineが1,250 Mbpsで、`m7i.2xlarge`の2,500 Mbpsより低い。差額は0.0252 USD/hourに留まり、container buildは大量の小ファイルI/Oを伴うため`m7i.2xlarge`を選ぶ。

## 資格情報と配布

CloudFormation template `aws/qwen-build-infrastructure.yml`はprivate S3 bucket、egress-only security group、EC2 instance profile、GitHub OIDC provider、Actions roleを作る。Actions roleは対象repositoryだけをtrustし、build用instanceの起動・終了、SSM command、対象bucketの操作だけを許可する。privateなSGLang sourceは、そのrepository自身のworkflowが管理する玄界上のread-only bare mirrorからQwen workflowが入力archiveへ複製する。消費側repositoryからsource repositoryへ書き込まず、広いPATや無効化されているdeploy keyに依存しない。

玄界、ASPIRE 2A+、Firmusの各self-hosted runnerも同じOIDC roleを一時的にassumeする。sidecarとSquashFSごとに有効期間1時間のpresigned GET URLを実行直前に生成し、`curl`でrun固有directoryへ毎回downloadする。URLとAWS資格情報はscheduler jobへ渡さず、計算ノードへ渡すのはdownload済み`.sqsh`、run固有Enroot設定、実行scriptだけである。

`[require-build]`が最新commit messageの先頭にある場合は必ず新規buildする。それ以外では第一親履歴を遡り、sidecarのsource commit、S3 object size、SHA-256 metadataが一致する最近傍の成果物を再利用する。クラスタタグがないpushは静的検証だけを行い、AWS資源を起動しない。
