# OCI / Enroot移行記録

## 現行経路

ビルドは各クラスタのログインrunnerで行う。`buildah bud --isolation chroot --layers`でDockerfileを処理し、Buildah storageからOCI image layoutへpushする。Skopeoは別のOCI layoutへcopyしてtransport境界を検証し、Umociはそのlayoutのmanifestとlayer履歴を検証する。実行用SquashFSは検査に使ったBuildah VFS working containerのroot filesystemを`buildah unshare --mount`のuser namespace内で読み、埋め込みmanifestを照合してから`-all-root -no-xattrs`で所有者をrootへ正規化して作る。厳密な元UID/GID、xattr、ACLの維持は要件に含めない。

Dockerfileはsystem toolchain、NCCL tests、Python project metadata由来の依存、GPU通信依存、SGLang sourceの順にレイヤを分ける。SGLangの`pyproject.toml`とmetadata filesだけをsource本体より先にcopyし、`uv.lock`をbuild入力へ含めずに`uv sync --no-install-project`を実行する。依存関係はcold build時に解決し、SGLang sourceや計測patchだけを変えた場合は解決済みの高コストな依存レイヤを再利用する。埋め込みmanifestは最後にcopyし、来歴だけの変更でビルド処理を無効化しない。

rootless BuildahのgraphrootはLustre上へ置かず、runner hostごとのlocal XFSである`/var/tmp/apac-qwen-buildah-<uid>`へ永続化する。runroot、build context、OCI layoutはrunごとの`/dev/shm`に置き、workflow終了時に削除する。2026-08-21に固定BusyBoxを用いたcold/warm buildを両クラスタで行い、`--layers`を使う3つの中間stepすべてが`Using cache`となることを確認した。runner hostが変わるかlocal storageが消去された場合だけcold buildとなる。

Enroot 4.2.0はDocker archiveやOCI archiveを直接importできない。このため、Buildah VFS working containerのroot filesystemを`mksquashfs`でEnroot imageへ変換する。ASPIRE 2A+では巨大OCIを`umoci unpack --rootless`すると、squashの有無にかかわらずlayer適用中にexit 137となった。OCI内容はSkopeoと`umoci stat --json`で検証し、既にBuildah chroot buildとcontainer内検査を通った同一imageのVFS treeを再利用することで、root filesystemの二重展開とそのmemory peakを避ける。ログインノードではSquashFS mountや`/dev`のrecursive bindが制限される場合があるため、Enrootの最終確認はGPU計算ノードで行う。

ASPIRE 2A+の`mksquashfs`はlogin host全体のmemoryから既定cacheを128865 MiBと計算し、runnerの実効制約を超えてexit 137となった。2 GiB、4 workerへ制限しても起動直後にkillされたため、圧縮cacheを256 MiB、workerを1に下げ、duplicate探索とfragment packingを無効化する。出力先もRAM-backedな`/dev/shm`ではなくnode-local XFSの`/var/tmp`とし、完成後だけ共有scratchの一時pathへcopyして原子的に公開する。これは元ownershipやxattrと同様に厳密なbit-for-bit最小化よりlogin nodeでの安定した変換を優先する境界であり、完成したSquashFS自体のsizeとSHA-256はsidecarで引き続き検証する。

完成物は`<workflow-run-id>.sqsh`と`<workflow-run-id>.container-sidecar.json`である。sidecarはSquashFSのsizeとSHA-256を持つ。`[require-build]`がない実行はGitの第一親を順番に遡り、対応するworkflow runの成果物についてsizeとSHA-256だけを照合し、最初の有効なものを再利用する。コンテナ内部検査や過去Actions stepの成否は再利用条件にしない。

## 制御処理の責務

クラスタworkflowはscheduler、filesystem配置、GPU数などのサイト差だけを持つ薄いadapterとする。共通処理は実行段階ごとに分離し、玄界とASPIRE 2A+で同じ実装を使う。

- `containers/sglang/Dockerfile`はimage layerとcontainer内部の依存関係、`scripts/build-sglang-image.sh`はBuildah、Skopeo、Umoci、SquashFS変換と成果物公開を担当する。
- `scripts/select-qwen-container.sh`はGit第一親をたどる再利用候補の選択とsidecar検証だけを担当する。
- `scripts/provision-qwen-runtime.sh`はログインノードでruntime directoryを一度構成し、固定source OIDなどcontainer内の不変metadataを一度だけ抽出する。計算ノードやrankごとに同じmetadataを再取得しない。
- `scripts/run-qwen35-node.sh`は既にprovisionされた設定とmetadataを読み、計算ノード上のEnroot起動とbenchmarkだけを担当する。
- `scripts/send-qwen-fragments.sh`はscheduler終了後の資格情報境界とBigQuery送信、`scripts/package-qwen-runtime.sh`は回収対象とprofile選択を担当する。どちらもログインrunnerでのみ実行する。

run directoryへ置くscriptはprovision時のcommitからsnapshotする。このため、workflowを待機している間にrepository側を更新しても投入済みjobの実装は変化せず、各runのコード来歴をcommitへ固定できる。

## クラスタ差

玄界はビルド領域をログインノードの`/dev/shm`、成果物を`/home/pj26000080/share/sglang-containers`、実行領域とモデルメタデータを`/fast`へ置く。PJMの`jobenv=singularity`はサイトがcontainer向けuser namespaceを許可するためのscheduler属性として残すが、Singularity executableやSIFは利用しない。1ノード当たり4 GPUを使う。

ASPIRE 2A+はビルド領域をログインノードの`/dev/shm`、成果物・実行領域・モデルメタデータを`/scratch/users/industry/ai-hpc/apacsc14`へ置く。PBS project `59010014`を指定し、default routingから`aiq2`へ1ノード当たり112 CPU、8 GPU、1500 GB memoryを排他的に要求する。`aiq2`の直接指定はqueue ACLで拒否され、projectなしではbudgetを確定できない。システムEnroot 3.4.1の既定pathは一般ユーザーが書けないため、ユーザー領域へEnroot 4.2.0を導入し、cache、data、runtime pathをユーザーが書ける場所へ設定する。

ASPIRE 2A+のhome Lustreでは、既存の`.local/.setenv`や`.gitconfig`に対して`Cannot send after transport endpoint shutdown`、`Input/output error`、`Network is down`が断続的に発生した。同時刻でもhome内の新規ファイル、scratch、`/dev/shm`への作成は成功し、home使用率は9%、scratch使用率は16%だった。新しいinodeへの置換で読取りが復旧したため、容量不足や権限不足、filesystem全域の停止ではなく、home Lustre clientまたは既存inode/dentryに限定された障害と判断する。再発時はlogin nodeとLustre serverのログを管理者に確認してもらう。

両クラスタともEnrootではhostの`/dev`全体をrecursive bindせず、`ENROOT_RESTRICT_DEV=yes`で最小device filesystemを作る。NVIDIAとMellanoxのhookは保持し、GPU計算時だけ必要なdeviceとdriverを注入する。玄界ではPJM cgroup pathとEnroot 4.2.0の`10-cgroups.sh`が整合しなかったため、このuser-local hookだけを無効化し、資源制御をPJMに委ねる。

read-onlyなSquashFSから起動するため、計算nodeごとに一時的なruntime homeを作り、containerの`HOME=/runtime-home`と`XDG_CACHE_HOME=/runtime-home/.cache`へbindする。JIT libraryが標準cache pathへ書く場合もこの単一の書込み境界へ収容し、libraryごとにhost home配下の例外mountを増やさない。実行可能な共有libraryを生成するruntime homeとFlashInfer、DeepGEMM、TorchInductor、Tritonのworkspaceは、schedulerが`TMPDIR`、`TEMPDIR`、`TEMP`の順で示すnode-local disk、未指定時は`/tmp`へ置く。socketと短命なtelemetryだけは`/dev/shm`へ残し、用途の異なる一時領域を分ける。job終了時は両方を削除する。

玄界PJM job `6552532`では2ノード8 GPUのdummy初期化が完了した後、`tvm-ffi`がhost由来のread-onlyな`$HOME/.cache`へJIT出力を書こうとして失敗した。runtime homeを一旦`/dev/shm`へ移したjob `6554500`では書込み自体は成功したが、全rankが生成済み`.so`を`failed to map segment from shared object`でloadできなかった。過去のSingularity経路でも玄界計算nodeの`/dev/shm`は実効`noexec`と確認済みであるため、JITをnode-local diskへ戻した。

ASPIRE PBS job `203640.pbs111`は`resources_used.walltime=0`、`Exit_status=-2`、`Stageout_status=1`でscript本体の前に終了し、`PBS_O_WORKDIR`はephemeral runnerのcheckoutを指していた。ASPIREのPBS 2025.2.2は`qsub -d`を実装していないため、`qsub`前にrun固有scratchへ`cd`して投入時working directoryをruntime lifecycleへ固定する。玄界も同じ理由で`pjsub`前にruntime directoryへ移動する。また、provision用`uvx`のmanaged Pythonはrunner一時領域を使う一方、後続jobでも使う計測CLIのmanaged Pythonはruntime directory内へ配置し、venvのinterpreterをrun lifecycleと同じ寿命にする。PBS logが生成されない起動前失敗でもaccountingを先に保存・表示し、成果物へ回収する。

玄界の直近成功runでhost driver 595.71.05を確認した。一方、固定CUDA 13.0.1 base imageの`NVIDIA_REQUIRE_CUDA`は既知の575系driverまでしか列挙していない。新しいdriverはCUDA 13を満たすため、GPU実行時は`NVIDIA_DISABLE_REQUIRE=1`をEnrootへ明示して古いimage内制約だけを無効化する。device、driver library、GPU topologyは引き続きNVIDIA hookとcontainer内`nvidia-smi`で検証する。container内GPU topologyのsource schemaも`singularity_argv`からruntime非依存の`container_argv`へ移行した。

ASPIRE 2A+にはsystemの`nvidia-container-cli` 1.17.4がある。玄界には配置されていなかったため、NVIDIA公式RPM repositoryの署名を検証し、`libnvidia-container1`と`libnvidia-container-tools` 1.20.0を`~/.local`へ展開した。これによりroot権限を使わずEnrootのNVIDIA hookがhost driverとdeviceをcontainerへ注入できる。

Enroot 4.2.0の標準NVIDIA hookはhostの`ldconfig`を指定する。`libnvidia-container` 1.20.0はこの指定時にPID namespace内へ`/proc`を再mountするが、玄界のPJM jobでは`operation not permitted`になる。OCI imageは`/sbin/ldconfig`を含むことをbuild時に確認し、provision時にEnroot system設定をrun directoryへ複製してNVIDIA hookをcontainer側`/sbin/ldconfig`へ固定する。これによりhost `ldconfig`用の`/proc`再mountを避ける。PJM job `6548374`のwritable起動に加え、実経路と同じread-only起動のjob `6548385`でもcontainer内から4基のNVIDIA H100とdriver 595.71.05を確認し、exit code 0となった。ASPIRE 2A+でも同じ設定を使ったPBS job `203572.pbs111`がH100とdriver 595.71.05を確認してexit status 0となった。run directory側の設定は各実行と一緒に削除されるため、共有されたuser installationをActions中に変更しない。

## 小規模imageによる実機確認

固定digestのBusyBox `sha256:3c6ae8008e2c2eedd141725c30b20d9c36b026eb796688f88205845ef17aa213`を使い、DockerfileのBuildah chroot build、作業containerの`buildah run`、OCI layoutへのpush、Skopeo copy、Umoci rootless unpack、SquashFS作成を玄界とASPIRE 2A+のlogin nodeで確認した。さらにASPIRE 2A+でBuildah VFS working containerのmount、直接SquashFS作成、OCI push、`umoci stat --json`を一時BusyBox imageで確認した。玄界でもmountしたVFS treeからの直接SquashFS作成まで確認した。厳密なxattrと元ownershipを要求しないため、Lustreが対応しないxattrの警告は許容する。

ASPIRE 2A+ではPBS job `203570.pbs111`を1 GPU compute node `a2ap-dgx037`へ投入し、Enroot 4.2.0で`ENROOT_SMOKE_OK`とexit status 0を確認した。玄界ではPJM job `6548346`をcompute node `b0004`へ投入し、同じく`ENROOT_SMOKE_OK`、PJM exit code 0を確認した。検証用image、script、logは確認後に削除し、schedulerに検証jobを残していない。

## dummy実行と計測境界

モデル重みは取得しない。ログインrunnerはHugging Faceから固定revisionのmodel JSON、tokenizer、text metadataだけを取得し、ShareGPTは固定revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`の単一JSONをクラスタごとの共有dataset cacheへ取得する。計算ジョブは`--load-format dummy`、固定seedの`random` datasetで分散SGLangを起動する。Enrootの`-m`はcolon区切りfstabであるため、`none:x-create=file,bind,ro,nosuid,nodev`を明示して単一JSONのtargetを作成し、read-only mountしてwarmupと本計測の両方で使う。datasetのrepository、revision、filename、host path、size、SHA-256をmanifestへ記録・検証する。dummy runはcontainer、GPU injection、MPI/NCCL、SGLang分散起動、計測、成果物回収を確認するための診断であり、公式モデル性能ではない。

玄界PJM job `6555945`ではnode-local disk上のJITと2ノード8 GPUのCUDA graph captureが完了し、従来の`failed to map segment`は解消した。その後、SGLang `7ec0eb3d7a01f872d0bd2f5095f792c0c1792320`の`random` datasetがShareGPTを暗黙に取得しようとし、計算nodeの`HF_HUB_OFFLINE=1`境界で終了した。

最初の再検証Actions run `32536408333`はPJM投入前のprovisionで、計測ライブラリのclosed-schema検証だけが旧`random`値を要求していたため`dummy dataset manifest differs`となった。これを同期したPJM job `6556998`では、dataset registryに存在する`random-ids`を`offline_throughput.py`のCLI choicesが受理せずexit 2となった。また同CLIは選択datasetと無関係にShareGPT由来のrandom warmupを先に構築するため、`generated-shared-prefix`へ変えるだけでも暗黙downloadは残る。ShareGPTをlogin nodeで事前取得して公式`random`経路へ明示することで、計算nodeのoffline境界を保ちながら入力1024、出力128、8 promptの診断条件を維持する。これはruntime入力とprovenanceだけの変更であり、container再ビルドを必要としない。過去runのmanifestとartifactは書き換えない。

計算ジョブへBigQuery資格情報を渡さない。各producerは共有runtime directoryの`domain/fragments`へ完了済みJSONLをatomicに書く。scheduler terminal状態と`run-success`を確認した後だけ、ログインrunnerがmode 0600の一時資格情報を`/dev/shm`へ作り、全fragmentを一度送信する。資格情報はruntime directoryやActions artifactへ入れない。
