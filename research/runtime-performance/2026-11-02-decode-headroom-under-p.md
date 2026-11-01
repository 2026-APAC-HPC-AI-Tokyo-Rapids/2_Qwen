# 2026-11-02 P 採用後の Decode の余地：定常は既に Prefill 容量に達している（分析のみ、GPU job なし）

gpt-6-astra による分析。P の 3 run（41264009819、41280565072、41287043170 attempt 2）の `profile_export.jsonl` と Decode log を読み直した。

## 定常は Prefill 容量で頭打ち

| P run | 公式 score | 定常完了率（計測開始後 20〜110 秒） |
|---|---:|---:|
| 41264009819 | 10009.54 | 52.422 req/s |
| 41280565072 | 10008.67 | 52.433 req/s |
| 41287043170 | 10020.58 | 52.422 req/s |

PP0 周期（113.2 / 113.8 / 116.6 ms）からの Prefill 容量は 2/0.1132 + 2/0.1138 + 2/0.1166 = **52.395 req/s** で、実測と一致する。Little の法則でも、定常の E2E 約 4.419 s と c232 から 52.45 req/s となり、実測を説明できる。

- `2026-11-02-pp4-concurrency-under-p.md` の serving 模型が示した「Decode が 1% 速くなると約 +0.6%」は、模型の局所的な傾きで、いまの地点の予測としては強すぎる。step_scale 0.98 を基準にすると 1% で +0.43%、2〜3% でほぼ頭打ち。現在の log に合わせた別の較正（step 37.88 ms@45.1/rank）では 1% で +0.16〜0.18%。計画には「Decode の 1% 短縮で score +0〜20」を使う。
- Decode の scheduler step は約 37.88 ms/step（`decode_log_interval=40` の log 間隔から、4 rank とも）。受理長は約 2.224。client の ITL × 受理長（約 34.7 ms）を step とみなす換算は使えない。
- 200 × λ_P = 10,479 tok/s は定常で全要求が合格した場合の処理容量で、開始直後の TTFT 不合格（約 111 件）と終了時の drain を含む公式 score の予測ではない。

## 現在の Decode step（37.88 ms）の見積もり

旧 profile（10/29、n44 で 37.78 ms）の実測と、その後に採用した gate の実測から：MegaMoE 約 21 ms、FA3 verify 約 6.8〜7.3 ms、dense QKV 約 1.7 ms、O 約 1.7 ms、input norm＋quant 約 0.27 ms、post-attn norm/router 約 1.1 ms、EAGLE draft 約 1.3〜1.5 ms、LM head＋logits 約 0.45 ms、sampling/verify 約 0.18 ms。現在の kernel 別 profile はなく、条件付きの値。

## 候補（いずれも score の見込みは run 間のばらつき以下）

| 候補 | 条件付きの step 短縮 | score の見込み |
|---|---:|---|
| greedy verify 後処理（BF16→FP32 logits、penalty、argmax）の融合、bit 一致 | 0.10〜0.18 ms | +0〜10 |
| B44 だけ FA3 verify を h2/split3 に | 0.55〜0.80 ms × 該当 step の比率（不明） | 比率不明 |
| FA3 verify h2/split4 | 0〜0.3 ms | +0〜15 |
| draft LM head の BF16 small-M GEMM | 0.08〜0.18 ms | +0〜10 |
| MegaMoE dispatch の独立 2-token buffer | 0〜0.3 ms | +0〜15 |

- Decode の上限 48 → 52 は、SwapAB の 192 行 guard と FA3 の B≤48 guard を外れ、容量にも余裕があるので扱わない。EAGLE の steps 3/draft 4 → 2/3 は受理長が約 10% 下がり、steps 4/draft 5 は verify が複数の最適化 guard を外れる。どちらも採らない。
- 採用済みのもの（wide norm、MoE 入力融合、QKV M192、penalty broadcast）と、過去に負けたもの（dense SwapAB、FA3 2 CTA、h4 stream split、FA3 kernel 内 merge）は再投入しない。

## 結論

- いまの score の上限は Prefill の容量で決まる。Decode を速くしても、Prefill の待ち行列が伸びるだけになる。
- Prefill 側は FA3（round 14〜21）、MegaMoE（round 19・20、consumer の見直し）、dense GEMM（10/31）、concurrency（11/02）まで調べ、検証で裏付けられた上積みは残っていない。10,020.58（P の構成は 6 run で 10,002.75〜10,020.58）からさらに上げるには、Prefill の kernel を新しく設計し直すか、構成（PP×TP、worker の数、GPU の割り当て）を変える必要がある。
