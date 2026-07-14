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
node-local filesystemへ展開してSIFへ圧縮する処理だけを行います。共有ログイン
ノードの1プロセスCPU時間・メモリ制限を越える処理はせず、圧縮のCPU負荷だけを
batch jobへ隔離します。

`provision`は今回の実行だけに必要な束を共有filesystem上へ作ります。検証済みSIFの
hardlink、実行スクリプト、model path、dataset path、logs、resultsが一つのdirectoryに揃うため、
`run`はrepository checkoutやbuild directoryを参照しません。GPU jobの完了後、logsと
resultsをActions artifactへ回収して、この一時runtime directoryを削除します。
Python bytecodeはbuild時に生成し、model config、tokenizer、benchmark moduleなどGPUを
必要としないmodel依存の初期化はprovision時に済ませます。GPU architectureやdeviceを
必要とするTriton、DeepGEMMなどのJITだけがcompute job側に残ります。

SGLangを変更する場合は`patches/sglang/`にpatchを追加し、commit messageに
`[require-build]`を付けてpushします。recipeや依存関係を変えた場合も同様です。
benchmark条件だけを変える場合は`scripts/run-qwen35.sh`を編集して通常のpushを行えば、
親のSIFを再利用して新しい条件だけを速く検証できます。

入力はSGLangが標準で参照するShareGPTを使います。最初に玄界のログインノードで
`scripts/fetch-sharegpt.sh`を一度実行すると、dataset repositoryのrevisionとSHA-256を
検証して`/fast/PROJECT/datasets/ShareGPT_Vicuna_unfiltered/<revision>/`へ原子的に
配置します。datasetはSIFへ封入せず、`provision`と`run`で同じJSONだけを
`/dataset/ShareGPT_V3_unfiltered_cleaned_split.json:ro`へ明示的にbindします。

コンテナは`--containall`で起動し、hostからはmodel、ShareGPT、JIT cache、実行スクリプト、
今回の出力directoryだけを明示的にbind mountします。推論引数は環境変数にせず、
`scripts/run-qwen35.sh`のcommandとして版管理します。
