# Firmus PD実験記録

この文書は、Firmusの2ノード16基のH200で実行したQwen3 PD探索runを比較する。公式採点runは、公式AIPerfコマンドを固定した別runとして明示する。`[profile]`付きrunとTIMEOUT runの数値は公式スコアとして扱わない。

## 固定条件

- Model: `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`、revision `e156cb4efae43fbee1a1ab073f946a1377e6b969`
- Container: 12,106,153,984 bytes、SHA-256 `6b4d81eb6400fa33400e5725a2842f63565b700d579f9be812a7c229a80da3b5`
- Topology: node 0 = 4 Prefill workers (TP2)、node 1 = 2 Decode workers (TP4/EP4)、NIXL
- Gateway: `sglang-router`の単一`/v1/chat/completions` endpoint
- 探索AIPerf: ISL/OSL 4000/200、cache-bust `first_turn_prefix`、10秒warmup、15秒benchmark、15秒grace
- Firmus `hpcai` partition limit: 17分

## 比較結果

| Actions / Slurm | 主要因 | 結果 | 観測 |
|---|---|---|---|
| 34686971404 / 16918 | active HCA 10本、c16、profile | `COMPLETED 0:0`、15:55 | warmup 16/16を約505秒で完了。10 requests、errors 0、good fraction 1.0、goodput 0.3332992672 req/s、score 66.65985345 output token/s。profile runのため公式値には不採用。 |
| 34688206988 / 16919 | 上記に制御IP固定を追加 | `TIMEOUT`、17:21 | warmup 12/16。制御IP固定だけでは初回転送遅延を解消しなかった。 |
| 34692295495 / 16925 | NIXL待機上限を300秒から600秒に変更 | `TIMEOUT`、17:29 | 最初の返却は約540秒、600秒で10/16、630秒で11/16。待機延長は改善にならなかった。 |
| 34693201033 / 16926 | activeかつInfiniBandのHCA 8本、c16 | `TIMEOUT`、17:29 | router model登録待機は成功。300秒で9/16、450秒で14/16、600秒でも14/16。 |
| 34693215337 / 16927 | cold bootstrap待機延長 | `TIMEOUT` | warmup 8/16。採用せず。 |
| 34693236779 / 16928 | InfiniBand HCA 8本、c16、profile | `TIMEOUT`、17:11 | warmup 14/16、scoreなし。回収した計測値は`submission/analysis/profiling_data/16928`に保存。 |
| 34694189173 / 16931 | `mlx5_0:1` 1本だけ、c16 | `TIMEOUT`、17:29 | 480秒で15/16、630秒でも15/16。単一HCAは採用せず。 |
| 34695012071 / 16933 | InfiniBand HCA 8本、c8 | `TIMEOUT`、17:20 | 360秒まで0/8、630秒で5/8。concurrency低下は採用せず。 |
| 34697041944 / 16935 | active HCA 10本、c16、router修正込み | `TIMEOUT`、17:11 | 300秒まで0/16、420秒で10/16、630秒でも10/16。16918の完走は再現せず。 |

## Profile run 16928

回収区間のGPU平均使用率はnode 0で5.681%、node 1で5.025%、CPU idleはそれぞれ85.900%、88.913%だった。InfiniBandの累積counter差分は各HCAで数十KB以下にとどまった。低い計算利用率と小さい通信counter差分は、ログに記録された`KVPoll.WaitingForInput`と整合する。ただし、これはTIMEOUTした単一profile runであり、HCA間の性能差や因果関係を単独で確定する根拠にはしない。

## 次の比較

同梱SGLang sourceのNIXL文書は、UCX backendの最初のrequestに一度だけ接続cold-startが生じるため、サービス起動後にthrowaway requestを1回送ることを推奨している。commit `f430437`では公式AIPerfコマンドを変更せず、その直前にgateway経由の2-token requestを同期実行する。このrequestの入力、応答、所要時間は通常のAIPerf成果物から分離する。Actions run 34701093440 / Slurm job 16938で、初回接続を事前に完了させることでAIPerf warmupの同時16 requestsに残る転送待ちが減るかを検証する。
