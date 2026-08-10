# qwen-experiment-metrics v2

Qwen/SGLang実験のメトリクスを、計算ノードではローカルfragment/sourceへだけ記録する
v2完全実装です。通常依存にはcredential・HTTP clientを含めず、外部送達はrunnerが
`publish` optional dependencyを明示して使う時だけ有効になります。

公開する責務は次に限定します。

- machine-readable key/producer registry
- RFC 8785 JCSに基づく`Fact` encoding
- CSPRNG run IDを持つclosed run manifest
- buffered fragmentを正常終了時だけatomic renameする`Recorder`
- exact argv/cwd、structured termination、monotonic timingを記録する`StageRunner`
- ownership、required producer/fact、dedupe、conflictを検査するmerge
- Qwen configuration/primary-result domain recordのpure parser
- length-prefixed Unix domain socketからlocal fragmentへ記録するadapter
- IMB-MPI1/nccl-testsのstrict parserとnetwork projection
- profile-only node GPU raw samplerとmeasured-generation reducer
- fixed run artifact setのpackager
- artifact manifestのrun/role/size/hashを検証してから送るrunner-only publisher

workflowはsubmit前にmanifestを作成し、compute終了後にrunner上で
merge → package → GitHub Artifact upload → BigQuery publishを順に実行します。
compute processはcredential、BigQuery、HTTP、artifact uploadを使用しません。
