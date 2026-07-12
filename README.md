# Qwen 3.5 with SGLang on Genkai

このリポジトリは、SGLangの変更をpushするたびに同じQwen 3.5 benchmarkを実行し、
変更の影響を比較するためのものです。workflowは成果物の寿命に合わせて`build`、
`provision`、`run`に分かれます。

`build`の成果物は、recipe、upstream SGLang `main`、`patches/sglang/*.patch`から
生成した完成済みSIFです。SIFは`/fast/PROJECT/sglang-sifs/<run number>.sif`
として保存されます。commit messageに`[require-build]`を含めたpushと、手動実行で
`require_build`を指定した場合だけ新しくbuildします。それ以外では第一親履歴を遡り、
最も近い実build済みSIFをhardlinkします。

`provision`は今回の実行だけに必要な束を共有filesystem上へ作ります。検証済みSIFの
hardlink、実行スクリプト、model path、logs、resultsが一つのdirectoryに揃うため、
`run`はrepository checkoutやbuild directoryを参照しません。GPU jobの完了後、logsと
resultsをActions artifactへ回収して、この一時runtime directoryを削除します。

SGLangを変更する場合は`patches/sglang/`にpatchを追加し、commit messageに
`[require-build]`を付けてpushします。recipeや依存関係を変えた場合も同様です。
benchmark条件だけを変える場合は`scripts/run-qwen35.sh`を編集して通常のpushを行えば、
親のSIFを再利用して新しい条件だけを速く検証できます。

コンテナは`--containall`で起動し、hostからはmodel、JIT cache、実行スクリプト、今回の
出力directoryだけを明示的にbind mountします。推論引数は環境変数にせず、
`scripts/run-qwen35.sh`のcommandとして版管理します。
