# Qwen 3.5 with SGLang on Genkai

Genkai 上で同じ Qwen 3.5 benchmark を実行し、SGLang の変更による影響を比較するためのリポジトリです。

調査内容と現行構成は以下に分けて記録しています。

- [SGLang SIF のビルド](research/build-performance/README.md)
- [Qwen 3.5 benchmark の実行](research/runtime-performance/README.md)

実装は [Genkai workflow](.github/workflows/genkai.yml)、[SIF recipe](containers/sglang.def)、[実行スクリプト](scripts/run-qwen35.pjm)を参照してください。
