# Firmus実行環境

2026-08-23時点の`ssh firmus`はRocky Linux 9.8のPodman login containerで、Slurm 26.05.2を介して`g1`と`g4`の2計算ノードを利用する。各ノードは128 CPU、約2 TB memory、NVIDIA H200 141 GBを8基持つ。専有`hpcai` partitionの上限は17分であり、4ノード要求は不可能なので、Firmus workflowは1または2ノードだけを受理する。

`/home/apacsc14`と`/scratch/home/apacsc14`は同じBeeGFS user directoryを指す。共有するQwen資産は`/scratch/home/apacsc14/qwen`を正本とし、Buildah VFS cacheを`buildah`、完成したSquashFSを`containers`、固定datasetを`datasets`、任意のsource mirrorを`git-mirror`、weightを含まないモデルmetadataを`models`、run固有のcontrol・log・fragment・resultを`runs`へ分ける。Actions runner本体とcheckoutは既存の`/scratch/home/apacsc14/actions-runner`に残し、高価な成果物の寿命と混ぜない。

login containerの`/dev/shm`は63 MiBしかなく、`/var/tmp`の空きも約184 GiBである。既存クラスタの単一SGLang imageに対するVFS cacheは305–345 GiBなので、永続graphrootは容量6.9 TiBのBeeGFS上の`qwen/buildah`へ置く。BusyBox固定digestを用いたBeeGFS graphroot上のpull、working container作成、`chroot`実行、削除は成功した。runroot、build context、OCI検証layout、SquashFS作業出力はlogin container localの`/var/tmp`へ分離し、完成したSquashFSとsidecarだけをBeeGFSへ原子的に公開する。rootless Buildahは単一UID/GIDのuser namespace、`chroot` isolation、VFSを使い、SkopeoとUmociでOCI layoutを検証する。

計算ノードの`/tmp`はnode-local XFSで約894 GiB、`/dev/shm`は約1 TiBである。展開済みEnroot rootfsとJIT cacheは各ノードの`/tmp`、socketと短命なmetricsだけは`/dev/shm`へ置き、job終了時に削除する。共有BeeGFS上の`.sqsh`を直接mountして実行せず、各ノードで`enroot create`してからnamed containerを起動する。

FirmusのユーザーランドにはBuildah 1.43.2、Skopeo 1.24.0、Umoci 0.6.0、Enroot 4.2.0、uv/uvx 0.12.5がある。login containerにはSquashFUSEと`nvidia-container-cli`がないが、provision時のsource OIDは`unsquashfs -cat`で読み、Enrootは計算ノードだけで使うためSquashFUSEは要求しない。`nvidia-container-cli`は計算ノードにあり、runへ複製したEnroot NVIDIA hookから利用する。

Slurm job `12613`では、`nvhpc-hpcx-cuda13/26.5`のOpen MPI 5.0.10rc2、node-local一時領域、Enroot `create/start`、NVIDIA hookを同じ境界で検証し、H200とdriver 610.57.04をcontainer内から認識して10秒、exit code 0で完了した。これを根拠に`[cluster:firmus]`を共通のbuild、provision、benchmark、fragment送信、公式artifact回収経路へ接続する。

初回Actions run `32620606416` attempt 2ではbase imageと約350 packageのtoolchain導入後、同じlayer末尾のRust導入だけが一時的DNS失敗`Could not resolve host: sh.rustup.rs`で終了した。直後に同hostの名前解決とHTTPS 200を確認したため、損失の大きい実測箇所に限りcurlを3回、2秒間隔、全転送error対象で再試行する。
