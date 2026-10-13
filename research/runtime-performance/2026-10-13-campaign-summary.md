# 2026-10-09〜10-13 改善キャンペーンのまとめ

目標は、他チームの報告値であるASPIRE 2A+（H100）6000、Firmus（H200）10000 good output tok/sに近づくこと。結論として、どちらも届かなかった。最良値は次のとおり。各試行の詳細は[2026-10-09-prefill-supply-rebalance.md](2026-10-09-prefill-supply-rebalance.md)（§1〜§17）、DP attention計画は[2026-10-10-dp-attention-deepep-plan.md](2026-10-10-dp-attention-deepep-plan.md)を参照。

| クラスタ | 最良 | 構成 | commit / branch | 開始時点 |
|---|---:|---|---|---:|
| ASPIRE（H100×16） | **3313.0** | P3×TP4 / D1×TP4、D pool 393216、c72、EAGLE3 steps3、FP8 KV、page 1 | ed8c534（`research/aspire-eagle-decode-arfusion`の設定） | 約2860（P2/D2 c48） |
| Firmus（H200×16） | **4429.9** | P6×TP2 / D1×TP4、D pool 524288、c104、page 64、EAGLE3 steps3、FP8 KV | 3519fc0（`research/firmus-p6tp2-d1tp4`、headは52d9a3c） | 3450.9（P4TP2/D2TP4 c64、チームメイト） |

どちらも単回の値である。ASPIREの3313は、ITLが速めに出るnode対（dgx004/dgx026）での値で、遅い組ではc72の余裕がほとんどない（§12）。採用するには、同一node対での反復が要る。

## 効いたこと（考え方）

1. **goodputはPrefill供給でほぼ決まる。** 4000/200のworkloadでは、出力1 tokenあたり20 tokenのprefillが要る。cache-bustでprefix cacheは効かない。good上限は概ね「Prefill GPU数 × GPUあたり持続prefill / 20」になる（§1）。
2. **Decodeの余裕をPrefillへ回す。** 旧P2/D2は、Decodeのrunningが約17・ITL約10 msと余り、Prefillが律速だった。ASPIREはP3/D1（+16%）、FirmusはP6TP2/D1TP4（+28%）でPrefill GPUを増やした。
3. **単一Decodeのpoolを大きくする。** 旧P3/D1（1863.6）は、D poolの上限で詰まっていた。pool 393216／524288でrunning 60〜100を受け入れられるようになった（§2〜§3）。
4. **H200ではPrefillをTP2にする。** 重みがTP2で載るので、GPUあたりの効率が高い（約8.8k tok/s持続。TP4は約8.1k）。
5. **page size 64（Firmusのみ）。** 転送記述子が減り、TTFTと、Decodeと同居するprefillの遅れが縮んだ（+1.4%、§14）。ASPIREでは、Decode律速のため効果がなかった（§11）。

## 効かなかったこと

| 試行 | 結果 | 理由・所見 | 節 |
|---|---|---|---|
| ASPIRE P3/D1 c88 | 不適格（67.5%） | 同居prefill-2に待ちが集中し、TTFTが悪化 | §5 |
| Decode MoEをCUTLASSに | 不適格 | ITLが約1.7 ms悪化。Tritonが重み読出しの帯域上限で、より速い | §5 |
| Decode verifyをSM90 XQAへ | 出力が壊れた | 1 GPU probeではkernel自体は正しく（page 64のFA3より0〜9%速いだけ）、serving統合側の不具合。修正しても利益は小さい | §8、§11 |
| AR融合のtwoshot化 | ITL +5% | H100 TP4・約2 MBでは、oneshotのほうが速い | §9 |
| page 64（ASPIRE） | −5.8% | Decode律速では、Prefill側の利得が出ない | §11 |
| EAGLE3 steps4/draft5 | 中立 | 受理長+5%を、verifyとdraftの費用が打ち消す | §12 |
| Firmus Decode TP4＋TP2混成 | 4262（−2.4%） | 同居prefillが遅く、prefillの数も減る | §13 |
| Firmus c112 | 不適格（86.3%） | 単一Decodeの限界（ITL p90 20.1 ms） | §16 |
| FA3 split-KV / GQA packing掃引 | 約1 ms/step | L2を考慮すると、もともと帯域効率35〜52%。4倍の余地はない | §15 |
| DP8 attention＋DeepEP EP8 Prefill | GPUあたり6.9k | TP2＋CUTLASSの約8.8〜10.75kより遅い。attention TP1にしてもDP2版（6.6k）と同程度 | §16 |
| TensorRT-LLM 1.3.0rc29（NVIDIA recipe配置） | Prefill約8〜9k/GPU、投機なしITL 26 ms | Prefill効率はSGLangと同水準 | §17 |
| TensorRT-LLM＋EAGLE3 | ITL 26 ms（受理長2.2） | requestあたりのstep費用は約0.36 msで、SGLangの約0.35 msと同じ。見込みは約3.8k | §17.1 |

## 分かったこと

- **律速の所在。** 両クラスタの最良構成は、どちらも単一TP4 DecodeのITLで頭打ちになる。ITLはrunningにほぼ比例する。H100では、step費用がrequestあたり約0.35 ms、固定費が約17 msである。固定費の大半は、MoE重み（rankあたり約57 GB）を毎step読む費用で、帯域上限に達している（§7）。
- **Decode kernelの内訳。** H100のrunning約63では、MoE 18.5 ms、FA3 7.5 ms、AR融合 5.6 ms、group量子化 3.0 ms／stepだった（§7）。H200（チームメイトのnsys、run 38045244544）では、MoE 38%、AR融合 17%、FA3 13%、量子化 9%。
- **Prefillの効率は、エンジンでも分割方式でも変わらない。** SGLang TP2、TRT-LLM TP2はどちらも約8〜9k tok/s/GPU（FP8 denseピークの約27%）。DeepEP系はそれより遅かった。H200で10000に届くには、14 GPUをprefillに回してもGPUあたり約14k（約1.6倍）が要る。
- **他チームとの差は未解明。** エンジン（TRT-LLM）、分割（DP attention／EP）、kernel設定のいずれでも説明できなかった。他チームの値が同じ公式条件・同じ指標によるものかは確認できていない。

## 運用上の教訓

- Firmusはfair shareで、利用が続くとアカウントの優先度が下がり（377〜1132）、1回の待ちが数時間になる。監視は軽いshell loopにし、2時間の上限で張り直した。
- Actionsのrunnerが、Slurm待ちの間に落ちることがある。その場合もSlurm jobは残し、runディレクトリから直接回収した（c112）。
- AIPerfを`--workers-max 160`で動かすと、4 CPUではサービス登録が10秒のtimeoutに間に合わず、計測できない。32 CPUにした（bc64a59）。
- TRT-LLM rc29をFirmusで公式clientに通すには、次の5点が必要だった。`--server_role`なしの静的URL構成、`/v1/models`の確認を外す、NCCLの3変数（NVLS_ENABLE、RUNTIME_CONNECT、GRAPH_REGISTER）を事前にexport、`max_completion_tokens`を受け付けるpatch、ignore_eos時に意味のない`min_tokens`を外すpatch（§17、§17.1）。
- NASのアーカイブ（`/srv/qwen-artifacts`）はrunner専用の権限なので、手動回収分は自宅サーバーの`~/qwen-artifacts/research-runtime/`にSHA256SUMS付きで置いた。

## 残っている手段（いずれも未着手、image再buildが必要）

| 候補 | 対象 | 見込み |
|---|---|---|
| AR＋RMSNorm融合の出力側でgroup128 FP8量子化を済ませる | Decode（両クラスタの律速） | 1 stepで約2.5〜3.5 ms（6〜8%） |
| reduce-scatter＋FP8 all-gather（Prefillのみ） | Prefill律速の配置 | +8〜12% |
| TP group内のCUTLASS EP（`codex/native-cutlass-ep`、未実行の下書き） | Prefill MoE | +15〜25%の見積もり。ただし固定容量のゼロ初期化と全域活性化が利得を削る |

すべて積み上げても、Firmusで5000前後の見込みであり、目標との差は埋まらない。

## branchとraw

- 残すbranch：`research/aspire-eagle-decode-arfusion`（本稿、研究ノート、ASPIREの設定）、`research/firmus-p6tp2-d1tp4`（Firmusの最良設定。チームの派生branchの元）、`research/firmus-trtllm-p6tp2-d1tp4`（TRT-LLMの経路と互換patch。再利用用で、採用はしない）。
- 削除したbranch：`research/firmus-p8dpep-d2tp4`（DP8＋DeepEP、不採用）。条件はcommit 629b199/bc64a59の本文と§16に、sbatchの実物はNAS `/srv/qwen-artifacts/38068166465/1/source/source/submit.sbatch`に残る。
- TRT-LLM image：Firmus `/scratch/home/ACCOUNT/qwen/framework-comparison/images/trtllm.sqsh`（sha256 42937a72…、来歴JSONとbuild手順を同じ場所に置いた）。
- raw：NAS `/srv/qwen-artifacts/<run>/<attempt>/`（Actionsで回収したもの）、自宅サーバー`~/qwen-artifacts/research-runtime/2026101*`（手動で回収したもの、SHA256SUMS付き）。
