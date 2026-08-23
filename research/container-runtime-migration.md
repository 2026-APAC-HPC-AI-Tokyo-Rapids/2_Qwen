# OCI / Enroot移行記録

## 現行経路

ビルドはAWS Singaporeリージョンでrunごとに起動する`m7i.2xlarge`で行う。VMでは管理者権限を使用できるため、rootless互換処理を使わず、rootのBuildah、OverlayFS、`chroot` isolationでDockerfileを処理する。Buildah storageからOCI image layoutへpushし、Skopeoで別layoutへcopyしてtransport境界を、Umociでmanifestとlayer履歴を検証する。検査したBuildah working containerをrootでmountし、同じroot filesystemを`-all-root -no-xattrs`でSquashFS化する。厳密な元UID/GID、xattr、ACLの維持は要件に含めない。

Dockerfileはsystem toolchain、NCCL tests、Python project metadata由来の依存、GPU通信依存、SGLang sourceの順にレイヤを分ける。SGLangの`pyproject.toml`とmetadata filesだけをsource本体より先にcopyし、`uv.lock`をbuild入力へ含めずに`uv sync --no-install-project`を実行する。依存関係はcold build時に解決し、SGLang sourceや計測patchだけを変えた場合は解決済みの高コストな依存レイヤを再利用する。埋め込みmanifestは最後にcopyし、来歴だけの変更でビルド処理を無効化しない。

VMのroot volumeは300 GiBのgp3（6,000 IOPS、500 MiB/s）とし、Buildah graphroot、runroot、context、OCI layout、SquashFS作業出力を同じlocal ext4上へ置く。instance、root volumeはbuild終了時に必ずterminateし、失敗時も6時間の自動shutdownを上限とする。VM/EBSの常時課金とクラスタ共有filesystem上の305–345 GiBのVFS cacheを廃止し、変更されていない中間layerだけを保持数制限付きprivate ECRへ保存する。

Enroot 4.2.0はDocker archiveやOCI archiveを実行形式として直接使えない。このため、Buildah working containerのroot filesystemを`mksquashfs`でEnroot imageへ変換する。OCI内容はSkopeoと`umoci stat --json`で検証し、既にBuildah chroot buildとcontainer内検査を通った同一treeを再利用してroot filesystemの二重展開を避ける。クラスタ側では`.sqsh`を展開せず、そのpathを`enroot start`へ直接渡す。

AWS VMは32 GiB RAM全体を一時buildに使えるため、`mksquashfs`は8 worker、8 GiB cache、gzip level 2で実行する。全RAM化は、26.6 GBのrootfsに加えてBuildah layer、OCI layout、圧縮入出力を同時に保持すると32 GiBを超えるため採用しない。高速gp3上でOverlayFSのcopy-on-writeを使う方が容量と速度の境界が明確である。

完成物の再利用正本は玄界の`/home/ACCOUNT/share/sglang-containers/<workflow-run-id>.sqsh`とsidecarである。sidecarはSquashFSのsize、SHA-256、Qwen/SGLang source commitを持つ。`[require-build]`がない実行はGitの第一親を順番に遡り、過去の成功runに対応するローカルfileのsizeとSHA-256を照合して最初の有効な成果物を再利用する。選択後は現在run IDへのハードリンクを作り、private S3の`containers/<Qwen-commit>/<workflow-run-id>-<attempt>/`へ1日限りの配布物としてuploadするため、再利用時にもクラスタ側のdownloadは省略しない。各クラスタrunnerはOIDC資格情報とBoto3 APIを使い、64 MiB part、最大8並列でrun directoryへ直接downloadする。AWS CLIを配置せず、bucketもpublicにせず、AWS資格情報をscheduler jobへ渡さない。

## 制御処理の責務

AWS workflowはbuild/reuseとS3公開を、クラスタworkflowは毎回のdownloadとscheduler差だけを担当する。モデル未配置の現在は3クラスタともGPU nodeで`sglang.check_env`だけを実行し、従来のdummy model、dataset、BigQuery、profile処理は起動しない。

- `scripts/build-sglang-container.sh`はroot Buildah、Skopeo、Umoci、SquashFS変換を、`aws/run-build-instance.sh`はS3入出力とEC2側log回収を担当する。
- `scripts/select-qwen-container.sh`はGit第一親をたどる玄界ローカル再利用候補の選択とsidecar/file検証だけを担当する。
- `scripts/provision-qwen-check-env.sh`はlogin runnerでBoto3 multipart download、SHA-256検証、run固有Enroot libraryのpoll上限変更を行う。
- `scripts/run-qwen-check-env.sh`は計算node上のnode-local Enroot作業pathを作り、共有filesystem上のdownload済みSquashFSを展開せず直接起動する。

run directoryへ置くscriptはprovision時のcommitからsnapshotする。このため、workflowを待機している間にrepository側を更新しても投入済みjobの実装は変化せず、各runのコード来歴をcommitへ固定できる。

## AWS配布と実imageのE2E確認

Actions run `32661356215`はAWS上でcold buildを実行し、instance稼働1,556秒で12,718,911,488 byteのSquashFSを作成した。このimageをS3から取得した玄界PJM job `6571584`とFirmus Slurm job `12847`は、共有filesystem上の`.sqsh`をEnrootで直接起動し、どちらも`CUDA available: True`まで到達した。当時のActions失敗は`check_env`がGPU IDを`GPU 0,1,...`とまとめて出力することを旧assertionが受理しなかった後段判定であり、Enroot起動の失敗ではない。

Actions run `32663724386`はwarm buildを実行し、28 step中2から23までECR cache hit、instance稼働869秒となった。生成物は12,718,911,488 byte、SHA-256 `4149856aa250add011ef671eec4ebf3efd45c1bd91e204718e229a3be80c5cfc`で、玄界の再利用archiveへ保存した。PJM job `6571616`はH100を検出してcheck_envとworkflow cleanupまで成功した。Actions run `32665044917`はこのローカル成果物を再利用し、EC2 buildを省略しつつS3へ再公開し、玄界へ改めて1分45秒でdownloadした。PJM job `6571620`もH100、CUDA、直接SquashFS起動、cleanupを含めworkflow全体が成功した。これにより、`[require-build]`なしのローカル再利用と、実行時の無条件な再downloadが両立することを確認した。

Actions run `32665563155`のFirmus Slurm job `12867`は24秒で`COMPLETED (0:0)`となり、8基のNVIDIA H200、CUDA available、NVIDIA topology、`run-success`を確認した。このrunもローカル成果物を再利用しつつ、FirmusがS3 APIでSquashFSを毎回downloadした後の直接Enroot起動である。

Actions run `32670652493`はASPIREの予約変数回避後の最終E2Eで、再利用imageのS3再公開、ASPIREへの無条件download、全DGX PBS job `204070.pbs111`、artifact回収、run directory cleanupまでworkflow全体が成功した。jobは`a2ap-dgx005`で112 CPU、8 GPUを割り当て、36秒、`Exit_status=0`で終了し、8基のNVIDIA H100 80GB HBM3、`CUDA available: True`、NVIDIA topology、`run-success`を確認した。これにより玄界、ASPIRE 2A+、Firmusの3クラスタすべてで、同系統の12.72 GB SquashFSを展開せずEnrootから直接起動する実経路を確認した。

## クラスタ差

玄界はdownload先と実行領域を`/fast/PROJECT/sglang-runs`へ置き、`b-batch`の1 GPU nodeで確認する。ASPIRE 2A+は`/scratch/users/industry/ai-hpc/ACCOUNT/sglang-runs`へ置き、project `59010014`から各node 112 CPU、8 GPU、1500 GBを`place=scatter:excl`で要求する。1 GPU、8 CPU、32 GBの非exclusive配置を試したPBS job `204056.pbs111`と`204057.pbs111`は、部分使用中の同じnode `a2ap-dgx035`へ割当後、script本体を一度も実行せず`resources_used.walltime=0`、`Exit_status=-2`、`Stageout_status=1`で終了したため採用しない。小さなexclusive要求もsite側で14 CPU、235 GBのsmall poolへ正規化され、空きnode待ちとなった。従来成功済みで実benchmarkの割当てとも一致するDGX全体要求を使う。Firmusは`/scratch/home/ACCOUNT/qwen/runs`へ置き、`hpcai` partitionで1 GPU、8 CPU、32 GB、17分を要求する。これはmodelを取得しない環境確認用の要求であり、実benchmarkでは各siteのnode全体を割り当てるruntime adapterへ接続し直してE2E検証する。

ASPIREでは`qsub`へscript file pathを引数で渡すと、全DGXを割り当てたjob `204058.pbs111`でもユーザーscript開始前に`Exit_status=-2`となった。切り分けで、stdinから最小scriptを投入したjob `204059.pbs111`と`204060.pbs111`は、同じGPU nodeでshell、user ID、`~/.local/.setenv`、Enroot pathの読込みまでexit 0となった。その一方、同じcheck script fileをpath引数で投入したjob `204061.pbs111`は、PBS MoMの`/var/spool/pbs/mom_priv/jobs/204061.pbs111.SC`が実行時に存在せずexit 127となった。このsite側script staging障害を回避するため、同じGit管理済みscript内容を`qsub` のstdinへリダイレクトする。

stdin化後のjob `204062.pbs111`も起動前に`-2`となったため、さらに`qsub -v`の変数を1つずつ分離した。`RUNTIME_DIR`だけを渡し、その配下の実行ファイルを起動したjob `204066.pbs111`はexit 0、同じhostnameだけのscriptに`CONTAINER_IMAGE`を追加したjob `204069.pbs111`は再びscript開始前に`-2`となった。したがってASPIREのサイトhookが`CONTAINER_IMAGE`を予約変数として解釈することが直接原因である。別名の中継変数は作らず、計算nodeのscriptが`RUNTIME_DIR/sglang.sqsh`を直接導出し、schedulerへは`RUNTIME_DIR`だけを渡す。

## 従来のクラスタ内ビルドとGPU dummy実行の記録

以下はAWS集中buildへ移行する前の障害と設計判断を履歴として残す。

ASPIRE 2A+はビルド領域をログインノードの`/dev/shm`、成果物・実行領域・モデルメタデータを`/scratch/users/industry/ai-hpc/ACCOUNT`へ置く。PBS project `59010014`を指定し、default routingから`aiq2`へ1ノード当たり112 CPU、8 GPU、1500 GB memoryを排他的に要求する。`aiq2`の直接指定はqueue ACLで拒否され、projectなしではbudgetを確定できない。システムEnroot 3.4.1の既定pathは一般ユーザーが書けないため、ユーザー領域へEnroot 4.2.0を導入し、cache、data、runtime pathをユーザーが書ける場所へ設定する。

FirmusはBuildah VFS cache、公開成果物、dataset、model metadata、run directoryをBeeGFS上の`/scratch/home/ACCOUNT/qwen`へ用途別に配置する。login containerの`/dev/shm`は63 MiB、`/var/tmp`の空きは約184 GiBであり、既存クラスタで305–345 GiBを使うVFS cacheはlocal overlayに収まらない。BeeGFS graphroot上のBuildah pull、working container作成、`chroot`実行、削除を固定BusyBoxで実証したため、永続graphrootだけを`qwen/buildah`へ置き、runroot、build context、OCI layout、SquashFS作業出力は`/var/tmp`へ分離する。完成したSquashFSとsidecarだけを共有成果物として公開する。計算ノードでは`/tmp`が約894 GiBのnode-local XFS、`/dev/shm`が約1 TiBなので、Enroot展開rootfsとJIT cacheを`/tmp`、socketと短命metricsを`/dev/shm`へ置く。`hpcai` partitionの2ノード、各8基のH200を利用し、17分のsite上限をSlurm adapterに明示する。

ASPIRE 2A+のhome Lustreでは、既存の`.local/.setenv`や`.gitconfig`に対して`Cannot send after transport endpoint shutdown`、`Input/output error`、`Network is down`が断続的に発生した。同時刻でもhome内の新規ファイル、scratch、`/dev/shm`への作成は成功し、home使用率は9%、scratch使用率は16%だった。新しいinodeへの置換で読取りが復旧したため、容量不足や権限不足、filesystem全域の停止ではなく、home Lustre clientまたは既存inode/dentryに限定された障害と判断する。再発時はlogin nodeとLustre serverのログを管理者に確認してもらう。

2026-08-23のActions run `32614319756`ではhome Lustre障害の再発によりlogin runnerが`.profile`を読めず、再利用経路のcontainer選択前に無条件実行していた`buildah --version`がexit 127となった。第一親成果物の再利用選択が必要とするのはGit、sidecarのsize・SHA-256検証だけであり、Buildah、Skopeo、Umociは新規build script、EnrootとNVIDIA hookは計算nodeの実行scriptがそれぞれ実際に利用する時点で検証する。高コスト成果物の再利用を無関係なtoolchain/home障害で止めないため、両クラスタのselect jobから全container toolの事前probeを削除した。

3クラスタともEnrootではhostの`/dev`全体をrecursive bindせず、`ENROOT_RESTRICT_DEV=yes`で最小device filesystemを作る。NVIDIAとMellanoxのhookは保持し、GPU計算時だけ必要なdeviceとdriverを注入する。玄界ではPJM cgroup pathとEnroot 4.2.0の`10-cgroups.sh`が整合しなかったため、このuser-local hookだけを無効化し、資源制御をPJMに委ねる。FirmusのSlurm cgroup pathでは標準hookを含むEnroot起動が成功しているため無効化しない。

read-onlyなSquashFSから起動するため、計算nodeごとに一時的なruntime homeを作り、containerの`HOME=/runtime-home`と`XDG_CACHE_HOME=/runtime-home/.cache`へbindする。JIT libraryが標準cache pathへ書く場合もこの単一の書込み境界へ収容し、libraryごとにhost home配下の例外mountを増やさない。実行可能な共有libraryを生成するruntime homeとFlashInfer、DeepGEMM、TorchInductor、Tritonのworkspaceは、schedulerが`TMPDIR`、`TEMPDIR`、`TEMP`の順で示すnode-local disk、未指定時は`/tmp`へ置く。またSquashFS直接起動時のmount helperがSGLang終了時の子プロセスkillに巻き込まれてstale file handle (Errno 116 / SIGBUS) となることを防ぐため、node-local一時領域配下に`ENROOT_DATA_PATH`/`ENROOT_RUNTIME_PATH`/`ENROOT_CACHE_PATH`/`ENROOT_TEMP_PATH`を設定し、`enroot create`で展開済みnamed containerを作成してから`enroot start`する。socketと短命なtelemetryだけは`/dev/shm`へ残し、用途の異なる一時領域を分ける。job終了時は両方を削除する。

玄界PJM job `6552532`では2ノード8 GPUのdummy初期化が完了した後、`tvm-ffi`がhost由来のread-onlyな`$HOME/.cache`へJIT出力を書こうとして失敗した。runtime homeを一旦`/dev/shm`へ移したjob `6554500`では書込み自体は成功したが、全rankが生成済み`.so`を`failed to map segment from shared object`でloadできなかった。過去のSingularity経路でも玄界計算nodeの`/dev/shm`は実効`noexec`と確認済みであるため、JITをnode-local diskへ戻した。

ASPIRE PBS job `203640.pbs111`は`resources_used.walltime=0`、`Exit_status=-2`、`Stageout_status=1`でscript本体の前に終了し、`PBS_O_WORKDIR`はephemeral runnerのcheckoutを指していた。ASPIREのPBS 2025.2.2は`qsub -d`を実装していないため、`qsub`前にrun固有scratchへ`cd`して投入時working directoryをruntime lifecycleへ固定する。玄界も同じ理由で`pjsub`前にruntime directoryへ移動する。また、provision用`uvx`のmanaged Pythonはrunner一時領域を使う一方、後続jobでも使う計測CLIのmanaged Pythonはruntime directory内へ配置し、venvのinterpreterをrun lifecycleと同じ寿命にする。PBS logが生成されない起動前失敗でもaccountingを先に保存・表示し、成果物へ回収する。

玄界の直近成功runでhost driver 595.71.05を確認した。一方、固定CUDA 13.0.1 base imageの`NVIDIA_REQUIRE_CUDA`は既知の575系driverまでしか列挙していない。新しいdriverはCUDA 13を満たすため、GPU実行時は`NVIDIA_DISABLE_REQUIRE=1`をEnrootへ明示して古いimage内制約だけを無効化する。device、driver library、GPU topologyは引き続きNVIDIA hookとcontainer内`nvidia-smi`で検証する。container内GPU topologyのsource schemaも`singularity_argv`からruntime非依存の`container_argv`へ移行した。GPU topology probeはimmutableなimage archiveではなく計算ノード上で展開済みのcontainer target（named rootfs）を起動して検査し、SquashFSの不変性照合は`container-sidecar`およびprovenance層へ一元化する。

ASPIRE 2A+にはsystemの`nvidia-container-cli` 1.17.4がある。玄界には配置されていなかったため、NVIDIA公式RPM repositoryの署名を検証し、`libnvidia-container1`と`libnvidia-container-tools` 1.20.0を`~/.local`へ展開した。これによりroot権限を使わずEnrootのNVIDIA hookがhost driverとdeviceをcontainerへ注入できる。

Enroot 4.2.0の標準NVIDIA hookはhostの`ldconfig`を指定する。`libnvidia-container` 1.20.0はこの指定時にPID namespace内へ`/proc`を再mountするが、玄界のPJM jobでは`operation not permitted`になる。OCI imageは`/sbin/ldconfig`を含むことをbuild時に確認し、provision時にEnroot system設定をrun directoryへ複製してNVIDIA hookをcontainer側`/sbin/ldconfig`へ固定する。これによりhost `ldconfig`用の`/proc`再mountを避ける。PJM job `6548374`のwritable起動に加え、実経路と同じread-only起動のjob `6548385`でもcontainer内から4基のNVIDIA H100とdriver 595.71.05を確認し、exit code 0となった。ASPIRE 2A+でも同じ設定を使ったPBS job `203572.pbs111`がH100とdriver 595.71.05を確認してexit status 0となった。run directory側の設定は各実行と一緒に削除されるため、共有されたuser installationをActions中に変更しない。

## 小規模imageによる実機確認

固定digestのBusyBox `sha256:3c6ae8008e2c2eedd141725c30b20d9c36b026eb796688f88205845ef17aa213`を使い、DockerfileのBuildah chroot build、作業containerの`buildah run`、OCI layoutへのpush、Skopeo copy、Umoci rootless unpack、SquashFS作成を玄界とASPIRE 2A+のlogin nodeで確認した。さらにASPIRE 2A+でBuildah VFS working containerのmount、直接SquashFS作成、OCI push、`umoci stat --json`を一時BusyBox imageで確認した。玄界でもmountしたVFS treeからの直接SquashFS作成まで確認した。厳密なxattrと元ownershipを要求しないため、Lustreが対応しないxattrの警告は許容する。

ASPIRE 2A+ではPBS job `203570.pbs111`を1 GPU compute node `a2ap-dgx037`へ投入し、小規模`.sqsh`のpathをそのまま`enroot start`へ渡して`ENROOT_SMOKE_OK`とexit status 0を確認した。玄界ではPJM job `6548346`をcompute node `b0004`へ投入し、同じく小規模`.sqsh`の直接起動で`ENROOT_SMOKE_OK`、PJM exit code 0を確認した。FirmusのSlurm job `12613`は1 GPU compute node `g4`でOpen MPI、node-local XFS上のEnroot `create/start`、NVIDIA hookを通してH200とdriver 610.57.04を確認したが、これは展開済みnamed rootfsの起動であり`.sqsh`直接起動の証拠ではない。2026-08-24のSlurm job `12739`で小規模`.sqsh`の直接起動を試すと、同じ`g4`に`/dev/fuse`と`fusermount3`はあるものの`squashfuse` executableがなく、Enroot 4.2.0は`Command not found: squashfuse`でmount前に終了した。その後、SquashFUSE 0.6.2とlibfuse 3.18.2を`~/.local`へ導入し、user-local `fusermount3`は計算nodeのsystem helperを指すsymlinkとした。Slurm job `12745`は`shared` partitionのcompute node `g1`で、root所有かつsetuidされたsystem `/usr/bin/fusermount3` 3.10.2、world-readable/writableな`/dev/fuse`、user-local SquashFUSEとlibfuseを確認した。共有Enroot installationを変更せず、job内copyだけ既知raceのmount poll上限を100回から5000回へ変更して小規模`.sqsh`を直接`enroot start`した結果、`FIRMUS_DIRECT_SQSH_OK`、direct start 570,413,934 ns、Slurm exit code 0となった。これによりFirmusでもSquashFS直接起動の基本経路は成立した。検証用image、script、logは確認後に削除し、schedulerに検証jobを残していない。

Firmusで直接起動を本番採用する前に、次を確認する。

- production runtimeにもEnrootのmount poll上限増加を適用する。job `12745`の変更はnode-localな一時copyだけであり、共有Enroot installationと現行workflowは未変更である。
- 実際の約12.7 GB imageをBeeGFS上から直接起動し、cold/warmそれぞれの開始時間と、必要blockだけを読む実行中のI/Oを測定する。
- GPU allocationでNVIDIA hook、H200、driver libraryをcontainer内から確認する。job `12745`はCPU 1個、1 GiB memoryだけを要求したfilesystem検証である。
- 実際のSGLangを起動から終了まで走らせ、mount helperがteardownに巻き込まれず、stale file handle、SIGBUS、unmount残留が発生しないことを確認する。
- 2 node実行では各nodeが同じ`.sqsh`を直接mountできること、runtime homeとJIT cacheが従来どおりnode-local XFSに置かれ、read-only imageへ書込みが発生しないことを確認する。

2026-08-23に玄界のCPU compute node `a0041`で、実成果物の12,718,784,512 byteのSquashFS（展開後26,567,442,083 byte、185,800 entry）を使って起動方式を比較した。共有filesystem上のimageをEnrootで直接起動すると0.538秒、`/dev/shm`へ6.861秒で複製したimageを直接起動すると0.151秒で、いずれも`/bin/sh`を実行してexit code 0となった。kernel SquashFS mountはrootless user namespace内でもloop device設定に失敗したため利用できない。`unsquashfs`による`/dev/shm`への全展開は11.000秒だったが、玄界の`/dev/shm`は`noexec`であり、展開rootfsからの`/bin/sh`は`Permission denied`となる。RAM上のimageからnode-local `/tmp` XFSへの展開は5分を超えた時点でも84%で、実用候補にならないためjob `6570638`を打ち切った。このXFS全展開方式は以後検討しない。

一方、以前のCPU node `a0011`上の小規模imageではEnroot直接起動が`fusermount: mount failed: Operation not permitted`で失敗しており、直接FUSE起動の可否にはnodeまたはjob環境による差が残る。また今回の直接起動はNVIDIA hookを使わない短時間のfilesystem確認であり、既知のmount helper寿命とstale file handle問題を否定しない。実運用をnode-local展開から直接起動へ変更する前に、実際のGPU jobでNVIDIA hook、SGLangの全実行時間、終了処理までをE2E確認する。

上記の`a0011`失敗時は標準のEnroot system設定、`a0041`と`a0025`の成功時は空の`ENROOT_SYSCONF_PATH`を使っており、当初の比較は厳密な同一条件ではなかった。2026-08-24に当時と同じ1,945,600 byteの小規模image、標準system設定、`ENROOT_MOUNT_HOME=no`、node-local `/tmp`上の4種類のEnroot path、同一PJM resource groupと同一`enroot start`引数を使う1 node jobを4本並行実行した。job `6570976`の`a0128`、`6570977`の`a0134`、`6570978`の`a0142`は成功した一方、job `6570979`の`a0143`だけが`Failed to mount`で失敗した。したがって起動オプション差だけではなく、node固有またはその時点のnode状態による差が実在する。成功nodeではkernel 4.18.0-477.27.1、Enroot 4.2.0、SquashFUSE 0.6.2、`/dev/fuse`のmajor/minorとmode、user namespace作成、PJM device cgroupの`c 10:229 rwm`、capability、NoNewPrivs、seccompが一致した。失敗nodeの同じ詳細項目は取得できていないため、差の正体がdevice cgroup、host側FUSE状態、その他の隠れた設定のどれかは未確定である。なおEnroot外で素の`squashfuse`を実行すると成功nodeでもEPERMとなる。Enrootはuser/mount namespaceを作ってからSquashFUSEを起動するため、この素の実行結果をEnroot可否の判定には使わない。

追加診断により、この差は固定的なnode設定差ではなくEnroot 4.2.0のmount成立待ち上限と起動時間のばらつきによるraceと特定した。Enrootの`runtime::_mount_rootfs_shim`はSquashFUSE起動後、`mountpoint`を最大100回だけpollし、成立していなければ生存中のhelperを含むprocess groupを終了する。12 node同時試験では標準設定の初回だけが7 nodeで失敗したが、同じjob内の直後の再実行は成功した。失敗時と成功時のnamespace内部でUID/GID map、capability、seccomp、`/dev/fuse`、setuid fusermount、mount flags、device cgroupの`c 10:229 rwm`は一致した。標準の100回上限を維持し、image、SquashFUSE、fusermount、共有libraryをページキャッシュから外したinstrumented job `6571098`では、node `a0061`でhelperが生存したまま101回、340,201,706 nsに達して`outcome=timeout`となった。直前のjob `6571083`でもnode `a0059`は95回、326,555,659 nsを要しており、上限間際だった。

暫定回避策として、Enrootの`runtime.sh`にある`runtime::_mount_rootfs_shim`のpoll上限を100回から5000回へ増やせば、このraceを回避できる。上限だけを5000回へ変更した一時的なEnroot library copyを使うjob `6571104`から`6571111`は、同じcold化条件で8 nodeすべて成功した。共有Enroot installationは変更していない。恒久対応ではiteration数への依存を避け、十分なwall-clock timeoutまでhelperの生存とmount成立を待つ実装を運用側へ依頼する。直接SquashFS起動を採用する場合は、まずGPU job用のuser-local Enrootへpoll上限増加を適用してE2E確認する。

同日の追加PJM job `6570665`と`6570674`では、元imageと同じ内容を別inodeへ複製し、CPU node `a0025`でLinux `mincore(2)`により計算nodeのページ常駐率が0%であることを確認してから直接起動した。この「計算nodeのページキャッシュがcold」の初回起動は2.447秒と1.924秒、直後の2回目は0.133秒と0.132秒だった。初回起動後に常駐したのは全12,718,784,512 byte中444 page、1,818,624 byte、0.0143%だけであり、3回目も0.134秒と0.136秒だった。image全体の順次readは2.794秒と2.762秒を要し、常駐率を100%にしたが、その後の起動は0.135秒と0.136秒で、初回起動だけで必要blockが温まった場合より速くならなかった。したがって起動短縮のために12.7 GB全体を事前readする必要はなく、同一nodeで一度直接起動するだけで十分である。なお新inodeは実験直前に共有Lustre上で作成したため、Lustre clientの当該inodeはcoldと実測できる一方、Lustre serverおよびstorage device側cacheのcoldまでは利用者権限で保証できない。

## dummy実行と計測境界

モデル重みは取得しない。ログインrunnerはHugging Faceから固定revisionのmodel JSON、tokenizer、text metadataだけを取得し、ShareGPTは固定revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`の単一JSONをクラスタごとの共有dataset cacheへ取得する。計算ジョブは`--load-format dummy`、固定seedの`random` datasetで分散SGLangを起動する。Enrootの`-m`はcolon区切りfstabであるため、`none:x-create=file,bind,ro,nosuid,nodev`を明示して単一JSONのtargetを作成し、read-only mountしてwarmupと本計測の両方で使う。datasetのrepository、revision、filename、host path、sizeをmanifestへ記録・検証する。dummy runはcontainer、GPU injection、MPI/NCCL、SGLang分散起動、計測、成果物回収を確認するための診断であり、公式モデル性能ではない。

玄界PJM job `6555945`ではnode-local disk上のJITと2ノード8 GPUのCUDA graph captureが完了し、従来の`failed to map segment`は解消した。その後、SGLang `7ec0eb3d7a01f872d0bd2f5095f792c0c1792320`の`random` datasetがShareGPTを暗黙に取得しようとし、計算nodeの`HF_HUB_OFFLINE=1`境界で終了した。

最初の再検証Actions run `32536408333`はPJM投入前のprovisionで、計測ライブラリのclosed-schema検証だけが旧`random`値を要求していたため`dummy dataset manifest differs`となった。これを同期したPJM job `6556998`では、dataset registryに存在する`random-ids`を`offline_throughput.py`のCLI choicesが受理せずexit 2となった。また同CLIは選択datasetと無関係にShareGPT由来のrandom warmupを先に構築するため、`generated-shared-prefix`へ変えるだけでも暗黙downloadは残る。ShareGPTをlogin nodeで事前取得して公式`random`経路へ明示することで、計算nodeのoffline境界を保ちながら入力1024、出力128、8 promptの診断条件を維持する。これはruntime入力とprovenanceだけの変更であり、container再ビルドを必要としない。過去runのmanifestとartifactは書き換えない。

玄界Actions run `32550368591`、PJM job `6558594`では2ノード8 GPUのdummy generationが8件すべて成功し、benchmark stageもexit code 0だった。しかし、`primary-result/v1`が正確なdecimalとして文字列`"2.342779523"`を保存したのに対し、後段の成功判定がjqのnumber型だけを選ぶ`numbers`を使っていたためexit code 4となり、`run-success`を作成できなかった。成功判定は計測契約と同じく文字列・numberの両方を受理して正値へ正規化する`tonumber`を使う。node 1のraw wait status 9とGloo peer resetはrank 0終了後の分散SGLang teardownで観測されたが、benchmark stageの判定結果ではなく、今回のActions failureの直接原因ではない。

計算ジョブへBigQuery資格情報を渡さない。各producerは共有runtime directoryの`domain/fragments`へ完了済みJSONLをatomicに書く。scheduler terminal状態と`run-success`を確認した後だけ、ログインrunnerがmode 0600の一時資格情報を`/dev/shm`へ作り、全fragmentを一度送信する。資格情報はruntime directoryやActions artifactへ入れない。
