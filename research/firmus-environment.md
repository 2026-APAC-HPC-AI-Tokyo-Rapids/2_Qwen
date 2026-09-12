# Firmus実行環境

2026-08-23時点の`ssh firmus`はRocky Linux 9.8のPodmanログインコンテナであり、Slurm 26.05.2を介して`g1`および`g4`の2基の計算ノードを利用する。各ノードは128基のCPU、約2 TBのメモリ、およびNVIDIA H200 141 GBを8基備えている。専有パーティション`hpcai`のジョブ実行時間上限は17分であり、4ノードの要求は不可能なため、Firmusワークフローでは1または2ノードの実行要求のみを受け付ける。

`/home/ACCOUNT`と`/scratch/home/ACCOUNT`は同一のBeeGFSユーザーディレクトリを指している。共有するQwen関連資産は`/scratch/home/ACCOUNT/qwen`を正本とし、BuildahのVFSキャッシュを`buildah`、完成したSquashFSイメージを`containers`、固定データセットを`datasets`、任意のソースミラーを`git-mirror`、重みを含まないモデルメタデータを`models`、そしてラン固有の制御ファイル・ログ・fragment・実行結果を`runs`へと分離して配置する。GitHub Actionsのrunner本体およびリポジトリcheckoutは既存の`/scratch/home/ACCOUNT/actions-runner`に残し、高コストな生成成果物のライフサイクルと混在させない。

ログインコンテナの`/dev/shm`は63 MiBしかなく、`/var/tmp`の空き容量も約184 GiBにとどまる。他クラスタでの単一SGLangイメージに対するVFSキャッシュの実績は305〜345 GiBに達するため、永続graphrootは容量6.9 TiBのBeeGFS上（`qwen/buildah`）に配置する。固定digestのBusyBoxを用いたBeeGFS graphroot上でのイメージ取得（pull）、作業コンテナの作成、`chroot`実行、および削除が正常に動作することを確認済みである。runroot、ビルドコンテキスト、OCI検証用レイアウト、およびSquashFS作業出力はログインコンテナのローカル領域`/var/tmp`へ分離し、完成したSquashFSとsidecarのみをBeeGFS上へアトミックに公開する。rootless Buildahは単一UID/GIDのユーザー名前空間（user namespace）、`chroot` isolation、およびVFSを使用し、SkopeoとUmociによってOCI layoutを検証する。

計算ノードの`/tmp`はノードローカルなXFSで約894 GiB、`/dev/shm`は約1 TiBの容量を持つ。展開済みのEnroot rootfsおよびJITキャッシュは各ノードの`/tmp`に、通信用ソケットと短命なメトリクスのみを`/dev/shm`に配置し、ジョブ終了時に確実に削除する。共有BeeGFS上の`.sqsh`ファイルを直接マウントして実行する方式は採らず、各ノード上で`enroot create`を実行して名前付きコンテナを作成してから起動する。

Firmusのユーザーランド環境には、Buildah 1.43.2、Skopeo 1.24.0、Umoci 0.6.0、Enroot 4.2.0、およびuv/uvx 0.12.5が配備されている。ログインコンテナ内にはSquashFUSEおよび`nvidia-container-cli`が存在しないが、プロビジョニング時のソースコミットOIDは`unsquashfs -cat`により読み取り可能であり、またEnrootは計算ノード上でのみ利用するためSquashFUSEは不要である。`nvidia-container-cli`は計算ノード側に配置されており、ラン環境へ複製したEnrootのNVIDIAフックから呼び出して利用する。

Slurmジョブ`12613`において、`nvhpc-hpcx-cuda13/26.5`のOpen MPI 5.0.10rc2、ノードローカル一時領域、Enrootの`create/start`、およびNVIDIAフックを同一境界上で検証した。その結果、コンテナ内からH200 GPUおよびドライバ 610.57.04を正常に認識し、所要時間10秒、終了コード0で完了した。この検証結果を根拠として、`[cluster:firmus]`を共通のビルド、プロビジョニング、ベンチマーク、fragment送信、および公式成果物回収の各経路へ接続する。

初回のActions run `32620606416`（attempt 2）では、ベースイメージの展開および約350パッケージのツールチェーン導入が完了した後、同一レイヤー末尾のRust導入処理において、一時的なDNS解決失敗（`Could not resolve host: sh.rustup.rs`）により異常終了した。直後に同ホストの名前解決とHTTPSステータス200を確認したため、再実行の損失が大きい実測処理に限り、全転送エラーを対象として2秒間隔・最大3回のcurlリトライ処理を設ける。
