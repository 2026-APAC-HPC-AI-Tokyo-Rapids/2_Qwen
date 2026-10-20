# 2026-10-20 DeepInfer 調査：Firmus（H200、Qwen3-235B、SGLang PD）に持ち込める技術

対象は [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) 774ce24（2026-10-20）と、それが同梱する DeepGEMM 1fc9c66、DeepEP 93eb6eb、FlashMLA 2e5429f、DeepSelect bfa4507、TileKernels c1cf914。DeepInfer は SM100 と Ascend、DeepSeek-V4.1-Flash 向けで、H200（SM90）にそのまま使える kernel はない。持ち帰れるのは設計と運用の考え方。

## 結論

| 順 | 施策 | 見込み | 費用 | 根拠 |
|---:|---|---:|---|---|
| 1 | MegaMoE の役割分担を SM100 型に寄せる（math WG は STSM まで、TMA store と FC2 の NVLink scatter は別 warp が smem から行う） | MegaMoE −60〜−120 µs/層 | header 大改修（2WG host image 前提） | DeepGEMM `sm100_fp8_fp4_mega_moe.cuh` の MMA 1 warp と epilogue 専用 WG。2WG 化で縮んだ K loop（FC1 −7%、FC2 −12%）を epilogue が相殺している問題への直接の答え |
| 2 | Prefill の DP 振り分けで、負荷を送る側でも数える | score +1.5〜3% | SGLang overlay（約 20 行） | `doc/inference_deployment_qa.md` L107-115。SGLang の `DPBudget` は 20 ms ごとに scheduler の snapshot で上書きされ、Prefill は forward（約 280 ms）終了まで publish しないので、その間の dispatch が数えられない（コードからの推定、ログで未確認） |
| 3 | hot expert を全 rank に複製して通信を減らす（均衡ではなく局所性のため） | score +3〜6%（試算） | 大（MegaMoE の expert 数前提、guard、配置生成） | `eplb/policy.py`。層ごとの上位 8 expert が割り当ての約 52% で、全 rank 複製で remote 比率 0.75 → 0.36、約 +10.7 GB/GPU |
| 4 | upstream 型の L1/L2 interleave scheduler（warmup wave、L2 優先、K block 単位の XOR mask） | MegaMoE −100〜−200 µs/層 | header 大（1 の後） | `scheduler/mega_moe.cuh`。7a179ad（+8%）は scatter を math warp が行っていたため FC1 を止めた |
| 5 | rank あたりの Prefill token 増（8192/rank） | +2〜4% | config（run 39379601121 で実行待ち） | `examples/sweep`、`prefill_pareto.png`（4096→8192 で +19%、ただし step 68 ms の軽いモデル） |
| 6 | QK-norm/RoPE prologue の書き直し（grid の平坦化、x の先行 load、PDL） | 134 → 70〜100 µs/層 | SGLang JIT overlay | `tilelang_ops/cuda/fused_transform.py`。当方の prologue は 3 列目の block が 256 thread 中 32 thread しか動かない |
| 7 | combine 前の全体 NVLink barrier を peer ごとの readiness tag に置き換える | 数 µs〜MegaMoE の約 1% | header | `sm100_fp8_fp4_mega_moe.cuh:1405-1441` |
| 8 | FA3 の epilogue で O を FP8（FP32 per-128）化、norm＋quant が MegaMoE buffer へ直接書く、PDL の確認 | 各 0.3〜0.8% | image 再 build を含む | FlashMLA の O epilogue、`mlp.py:274-301`、`runtime_config.py:64-66` |

## 使えない、または効かないもの

- DeepGEMM 1fc9c66 に SM90 MegaMoE はない（host が `arch_major == 10` 以外を止める）。SM90 の FP8 dense GEMM は v0.1.5 と実質同じ。
- 量子化は per-32 UE8M0（2 の冪 scale）と FP4 で、当方の FP32 per-128 より粗い（規則違反）。kFastMath の SwiGLU も近似。
- SWA bounded replay / YOCO は近似 prefill（規則違反、かつ V4.1 専用）。
- ZOS（非同期 scheduling）は Decode 用で、Prefill の step は同期実行。SGLang の overlap と本質的に同じ。
- DSpark / 可変長検証は、Decode が律速でない今は効かない（DeepInfer 自身の表でも 32 seq/rank で +1.6%）。Decode が詰まったときの候補として残す。
- balanced CP、動的 EPLB（静的配置で偏り 1.034 と DeepInfer の 1.074 より良い）、prefix 再利用（client が cache-bust）、DeepEP の FP8/LogFMT combine、BF16 の remote 加算、multimem。
- doc にある compute-quota による chunk 長の決定と各 Load Balancer は OSS に入っていない（`flops_estimator.hpp` は MFU の記録用）。

## 追記：DP 振り分けの負荷会計（候補 2）のログ検証

main の 8258.34 run（39225913979）の Prefill ログ（batch 行の `#new-token / input throughput` から各行の時刻を 20 ms 以内で復元）で検証した。
- 仮説の仕組みは実在する。Prefill は要求を受け取ってから snapshot を公開するまで約 150 ms かかり、その間に送られた要求は controller の推定 +1 が上書きされて忘れられる。
- 計測区間の定常状態では、他 rank に待ちがあるのに idle だった rank-forward は 0 だった。Prefill が飽和していて、各 rank に常に 2〜4 件の待ちがあるため。
- 以前の 1.6〜3.1% は、rotate-ties 導入前に「忘却」と「同負荷なら小さい index」が重なって出ていたもの。CPU 模擬では、旧方式で 2.6〜5.9%、現行の回転方式で 0.1〜0.2% になった。
- 残る損失は計測開始時の一斉投入直後の 2 forward（全体の 0.21%）だけで、これは忘却では説明できない（原因未特定）。

送る側で数える patch（`dp-inflight-request-accounting`、4 file、+19 行、CPU テスト付き）は作ったが、現構成での効果はノイズ以下と見積もり、serving には出さない。一斉投入直後の偏りを追うなら、ms 単位の振り分けログと snapshot 公開ログを入れた診断 run が先。

## 追記：DeepInfer 由来の候補の追加検討（2026-10-20、Codex に分担）

| 候補 | 結論 | 根拠 |
|---|---|---|
| post-attention の add＋RMSNorm が FP8 を MegaMoE buffer へ直接書く（pre-dispatch の廃止） | 見送り | main run 39225913979 の profile で pre-dispatch 16.1 µs/層。ただし norm に量子化を足すと、量子化なしの norm 31.6 µs → 既存の入力側融合 norm＋quant 47.0 µs と約 15.4 µs 増えるので、差し引きは約 0.7 µs/層。DeepGEMM host の変更は不要で、SGLang overlay だけで実装できることは確認した |
| FA3 Prefill の epilogue で O を FP8（FP32 per-128）にする | 条件付きで実現可能。保留 | BF16 staging は残し、global store の直前で FP8 化する。Prefill 専用の FA3 を companion `.so` として image に追加する必要がある（image 再 build）。見込みは 1 層 20〜30 µs（Prefill 約 0.8%）。難所は FA3 の register 増加と、graph 経路を 2 出力にすること |
| kernel 間の gap・PDL・GC・NUMA | 対応不要 | Prefill の 94 層で gap 1.29〜1.61 ms（forward の 0.48〜0.60%）。gap がすべて消えても上限は 0.84〜0.98%。PDL は既定で有効。GC イベントの記録はなく、NUMA の自動 bind は既定で有効（成否はログ不足で未確認） |
| combine 前の readiness tag | 作らない | SM90 で `%gridid` を世代値に使う場合、CUDA graph replay での一意性を裏付けられなかった |
| flat barrier を Prefill geometry に限定する | そのままでは不可 | Prefill instance の中でも、token 0 の rank は Decode geometry を選ぶ（gate の混在 case）。片方の geometry だけ protocol を変えると、待ち合いが成立せず hang する。全 geometry 版の header を Prefill の worker にだけ配る方式で作り直している |
