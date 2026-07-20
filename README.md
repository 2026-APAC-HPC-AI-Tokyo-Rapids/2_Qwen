# Qwen 3.5 with SGLang on Genkai

このリポジトリは、SGLangの変更をpushするたびに同じQwen 3.5 benchmarkを実行し、
変更の影響を比較するためのものです。workflowは成果物の寿命に合わせて`build`、
`provision`、`run`に分かれます。

`build`の成果物は、recipe、upstream SGLang `main`、`patches/sglang/*.patch`から
生成した完成済みSIFです。recipeはupstream `docker/Dockerfile`のCUDA 13.0.1、
Ubuntu 24.04、Python 3.12、GPU JIT向けruntime構成を基準にし、公開済みSGLang
imageを親にしません。SIFは`/fast/PROJECT/sglang-sifs/<run number>.sif`
として保存されます。commit messageに`[require-build]`を含めたpushと、手動実行で
`require_build`を指定した場合だけ新しくbuildします。それ以外では第一親履歴を遡り、
最も近い実build済みSIFをhardlinkします。

新規buildは二段階です。ログインノードでは並列数を抑えてrecipeをsandboxまで
provisionし、非圧縮rootfs tarを引き渡します。課金対象のA計算ノードではtarを
公式SquashFS Tools 4.7.5の`sqfstar`へ直接入力してSquashFSへ圧縮し、ログインノードで
recipe、metadata、SquashFSをSIFへ格納します。共有ログイン
ノードの1プロセスCPU時間・メモリ制限を越える処理はせず、圧縮のCPU負荷だけを
batch jobへ隔離します。圧縮はGenkai上の実測で速度とサイズのバランスが最も良かった
gzip level 1を60 coreで実行します。

`provision`は今回の実行だけに必要な束を共有filesystem上へ作ります。検証済みSIFの
hardlink、実行スクリプト、logs、resultsが一つのdirectoryに揃うため、`run`はrepository
checkoutやbuild directoryを参照しません。固定されたmodelとdatasetのpathはworkflow全体の
`MODEL_PATH`、`DATASET_PATH`で定義し、補助ファイルには書き出しません。GPU jobの完了後、
logsとresultsをActions artifactへ回収して、この一時runtime directoryを削除します。
provisionではmodelをloadせず、host側の`jq`でmodel configを検証し、`uvx hf cache verify`で
ローカルcheckpointをHugging Face Hubの最新revisionと照合します。SIF内では
`python -m sglang.check_env`だけを実行してSGLang、PyTorch、依存関係を記録します。GPU
architectureやdeviceを必要とするTriton、DeepGEMMなどのJITはcompute job側だけで実行します。

SGLangを変更する場合は`patches/sglang/`にpatchを追加し、commit messageに
`[require-build]`を付けてpushします。recipeや依存関係を変えた場合も同様です。
benchmark条件だけを変える場合は`scripts/run-qwen35.sh`を編集して通常のpushを行えば、
親のSIFを再利用して新しい条件だけを速く検証できます。

workflowはActions、PJM、SIFと一時runtimeの寿命、artifact回収を担当します。
`scripts/provision-qwen35.sh`は引数を取らず、`MODEL_PATH`からQwen 3.5 configと最新Hub
revisionに対するcheckpoint完全性を検証します。Hub側のmodelが更新された場合は既存checkpointが
検証に失敗します。`scripts/run-qwen35.sh`は計算ノード用の固定PJM要求を持ち、workflowから
`pjsub -x`で渡される`RUNTIME_DIR`、`MODEL_PATH`、`DATASET_PATH`を使ってCUDA module、
Singularity、一時領域、bind mount、環境記録、benchmark実行を担当します。
`scripts/pjm-job.sh`はbuildとrunで共通のPJM log待機、監視、cancel、終了待ち、cleanupを担当します。
GPU jobの投入からnode割当までと、割当後の実行監視は別stepなので、長いqueue待ちと
計算ノード実行時間をActions画面から判別できます。

SGLang JIT cacheの永続先は共有filesystemですが、benchmark中は計算ノードの`/dev/shm`へ
stagingしたcacheをbindします。staging時は共有`flock`、成功時の`rsync`書き戻しは排他
`flock`で保護します。失敗またはcancelされたjobのlocal cacheは書き戻さず、終了時に削除します。

benchmarkコマンドは`scripts/exec-profiler.sh`を経由します。現時点のwrapperは受け取った
引数をそのまま実行するだけです。将来nsysなどを導入するときは、このwrapperへprofilerの
起動設定を追加すればbenchmark本体やworkflowを変更せずに測定できます。

SIFにはlibc debug symbolsと、NCCL、OpenMPI、UCX、PMIx、libfabric、RDMAのuser-space
library、header、providerを含めます。SGLangのRust extensionもdebug symbolsを保持し、解決済み
Python dependencyは`/opt/sglang/requirements-frozen.txt`へ記録します。gdb、nsys/ncu、strace、
perf、hardware診断CLI、GPU・RDMAのkernel driverは計算ノード側の責務であり、SIFには封入しません。
将来の高速化用にDeepEP、MSCCL++、Mooncake、NIXLとHopper向けSGLang kernel artifactsも
build時に導入します。通常のTP、expert parallelism、prefill/decode分離で必要なbackendを、
実行引数やSGLang本体を変更した時点ですぐ比較できる構成です。

結果はruntimeの`results/`直下に集約します。`throughput.log`には実行コマンド、benchmarkの
標準出力・標準エラー、最後に`exit_code=N`を記録します。`environment.log`はジョブ、module、
GPU、containerなどの環境情報、`results.jsonl`はbenchmarkの構造化結果です。benchmarkの
終了コードは実行スクリプトからPJMへ伝播し、成功markerが作られないためActionsも失敗します。

入力はSGLangが標準で参照するShareGPTを使います。最初に玄界のログインノードで
`scripts/fetch-sharegpt.sh`を一度実行すると、dataset repositoryのrevisionとSHA-256を
検証して`/fast/PROJECT/datasets/ShareGPT_Vicuna_unfiltered/<revision>/`へ原子的に
配置します。datasetはSIFへ封入せず、`provision`と`run`で同じJSONだけを
`/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro`へ明示的にbindします。

コンテナは`--containall`で起動し、hostからはmodel、ShareGPT、node-local JIT cache、今回の
出力directoryだけを明示的にbind mountします。推論引数は環境変数にせず、
`scripts/run-qwen35.sh`のcommandとして版管理します。
