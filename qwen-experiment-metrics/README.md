# qwen-experiment-metrics v2

各producerは計測や収集の完了時に、診断用JSONL fragmentをアトミックなリネームによって確定します。BigQueryへの送信時は、`run`, `key`, `value`形式の行を500行単位のチャンク、`insertId`なし、リトライなし、重複許容のポリシーで送信します。送信失敗時は警告（warning）を出力するのみとし、ベンチマークやスケジューラジョブの終了判定には影響を与えません。

現在のQwen実行経路で利用されるrun/stage lifecycle、来歴（provenance）、ハードウェア／トポロジ、ワークロード、GPUサンプラー／プロファイラー、およびSGLangのrequest/token/configurationのみを取り扱います。fragmentは診断用の成果物（artifact）であり、再送用キューではありません。将来想定されるネットワーク計測、提出（submission）／キャッシュ管理、中央マージ、物理プロジェクション、およびコンシューマービューは実装対象外とします。

計算ノードは共有runディレクトリへfragmentを書き出します。BigQueryの認証情報はスケジューラジョブ（計算ノード）には渡さず、ジョブの終了状態（terminal状態）を確認したログインノード側のrunnerのみがfragmentを送信します。
