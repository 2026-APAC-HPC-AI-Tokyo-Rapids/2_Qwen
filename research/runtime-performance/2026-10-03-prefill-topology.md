# Prefill並列構成の再評価（2026-10-03）

現行のP2×TP4・EP1を最適と仮定せず、TP/EP/PP/DPを再評価する。合計16 H100とD2×TP4・EP1を固定し、モデルはQwen3-235B-A22B-Instruct-2507-FP8、BF16 EAGLE3 draft、FP8 KV、FA3を維持する。独立ワーカー2個という複製と、SGLang内部のDP attentionは区別する。

## コードとCPUスクリーニング

固定SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4` を調査した。異なるHEADのmirrorは変更しない。

| Prefill候補 | 判断と根拠 |
|---|---|
| P2×TP4・EP1、DP1 | 比較対照。現在の独立2ワーカー |
| P2×TP4・EP1、DP attention2 | 実機候補。各4GPU内のAttention TP2/DP2、MoE TP4。現行CUTLASSを維持できる |
| P2×TP4・EP1、DP attention4 | 条件付き。Attention TP1/DP4、MoE TP4。DP2の成立・方向性を先に確認 |
| P2×TP4・EP4 | FP8 CUTLASSはEP1のみのassertがあり、そのままでは起動不能。EP対応バックエンドへの変更を含む別実験が必要 |
| P1×TP8・EP1 | MoE中間幅1536/8=192がFP8 block128に非整列。現行CUTLASSでは起動不能 |
| P1×TP8・EP8 | バックエンド変更が必要。既存FPMによるCPU比較はワーカー数半減を補えず、追加実機の優先度は低い |
| 独立P4×TP2 | 235B FP8モデルの重みだけで2×80GBを超えるため対象外 |
| PP2 | SGLangがspeculative decodingと非互換。現在のEAGLE3を維持する比較では対象外 |

EP制約は `server_args.py:6447–6455`、PP制約は `:8436–8439`。TP8 EP1の整列制約は `layers/quantization/fp8.py:1135–1147`、限定的なpaddingの実装は `layers/moe/fused_moe_triton/layer.py:292–300`。これらは現在の実装の制約であり、独自実装を含む理論上の不可能性ではない。

既存AISimulate 0.13.0・413行のwhole-forward測定データを使い、fallback禁止で候補を照合した。測定identityはTP4/EP1とTP8/EP8のみで、DP2/DP4/EP4/PP2の正確なcellは存在しない。未対応を性能不良と解釈せず、別identityからの流用もしない。DynoSimへ切り替えても不足タイミングは補えない。

c44・4000要求・chunk8192・D固定の合成リプレイでは、対照が2588.405 raw tok/s・平均TTFT351.564ms、P1TP8EP8が1462.088 raw tok/s・3223.674ms。後者のjoint-good率は0.5%で不適格。このデータは旧auto MoE/BF16 KV/target-onlyであり、現行CUTLASS/EAGLE3の速度予測や公式スコアではない。

過去実機EP4 run `36301094490` は2202.553 good tok/s、TP8EP8 run `36303631570` は起動・生成後にjoint-good率1.754%でSLA不適格となった。どちらもP/D両方・旧条件を変更しているため、Prefill単独のEP効果とは呼ばない。

## DP2を1条件に絞る理由

Attentionの分割度を下げることで通信と小さいGEMMの効率が改善する可能性がある一方、MoE前後のgather/scatter、rank間待ち、異種TPのKV転送が増える。シミュレータに対応データがないため改善率は予測できず、コード・メモリの成立性を確認したうえで、まず1条件の実機E2Eで方向性を測る。

Pのみ `--enable-dp-attention --dp-size 2` を追加する。SGLangの自動調整で各DP rankのchunkは8192→4096、Prefill graph最大も4096、schedule conservativenessは既定値の0.3倍になる。この複合的なネイティブDP動作を評価するもので、Attention通信だけを分離測定した実験ではない。client concurrency48、Decode設定、既存統合イメージを固定する。

Qwen3通常MHAは `models/qwen3_moe.py:463–479` でAttention TPに応じたhead分割を行う。NIXLにはP Attention TP2→D TP4のhead単位転送があり、EAGLE3のdraftにもDP attentionの専用TP contextがある。実際の転送成功・数値出力はE2Eで確認する。固定pool131072の場合、target KVの追加容量は概算2.9375GiB/rankで、さらにdraft/Attention重みの複製が増える。既存ログの空き容量は成立性の目安にとどまり、実行時のピーク保証ではない。

## 実行記録

CPU側argv fixtureを含む52テスト、Ruff、bash構文、固定ソースのDP調整処理を検証し、2系統のSolレビュー後に `bea3efb` を投入した。[Actions 37126228529](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/actions/runs/37126228529)。branchは `research/aspire-prefill-dp2-20261003`、親は同一イメージc48の `ecd8952`。新規ビルドは要求せず、再利用予定SHAは `20c3a38359ccc61967343ec23d600f783df8d487c6339d12b9ca53c14b55d3d3`。実行中で結果未確認。単一runで優位性を確定せず、結果が有望なら同条件対照との反復を行う。runtimeのmain反映は優位性確認後とする。
