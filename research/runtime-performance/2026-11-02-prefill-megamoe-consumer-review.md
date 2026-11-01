# 2026-11-02 Prefill split MegaMoE の consumer 側の見直し：2 案とも CPU 段階で棄却（GPU job なし）

gpt-6-astra による分析のみ。CUDA 13.4.92、sm_90a の代替証拠（image は CUDA 13.0.88）。serving header sha256 `fbfaa6be…` と、control の encoded SASS `d582cf20…`（157,696 byte）を再現した。数え上げは `megamoe-prefill-consumer-block-accounting-20261102.json`、判定は `megamoe-prefill-consumer-cpu-verdict-20261102.json`。

## 現行の 1 K block（M128/N256/K128、2 math WG、split accumulator、各 WG が m64n128 の 2 half）

| 項目／active math warp | FC1（chunk 内 K=4） | FC2 |
|---|---:|---:|
| QGMMA.64x128x32 | 8 | 8 |
| WARPGROUP.DEPBAR / ARRIVE | 2 / 2 | 2 / 2 |
| accumulator の FMUL | 128 | 128 |
| scale・ratio の FMUL | 8 | 8 |
| promotion の FFMA | 0 | 0 |
| MUFU.RCP | 4 | 4 |
| 命令の合計（待ち経路の反復を除く） | 188 | 188 |

- 各 half に残る依存は「前の half の完了 → 64 個の独立な FMUL → 次の QGMMA」で、64 段の直列ではない。QGMMA は 64 個全部を operand に使うので、途中から発行できない。
- full CTA の 1 K block は 8,388,608 FLOP、理想の Tensor サービス量は 1024 clk。round 4 の split 1272〜1278 clk との差 248〜254 clk は idle の上側の枠で、除去できる量ではない。
- 現行はすでに 1 要素 1 FMUL で、FFMA の promotion にしても 128 → 128 命令で register の読み出しが増える。異なる K block の scale はまとめられない（Σ s_k P_k ≠ s_last Σ P_k）。WGMMA の scale-d・imm-scale は任意の FP32 倍率を吸収できない。
- 84〜90 µs/層を clock の上昇だけで得るには、実効 clock が約 3.7〜4.0% 上がる必要がある。今回の候補にその証拠はない。

## 2 案の判定

| 実体 | spill store/load | C75xx | 最大参照 R | 静的 SASS 命令数 |
|---|---:|---|---:|---:|
| control | 0/0 B | なし | 206 | 9856 |
| 案 1（前回の合成 scale 4 個を持ち越す） | 0/0 B | なし | 206 | 9856 |
| 案 2（4 本の WGMMA を 1 つの inline PTX にまとめ descriptor を内部更新） | 60/80 B | なし | 221 | 10208 |

- **案 1：棄却。** compiler がすでに chunk 内で合成 scale を再利用していた。代表定常 block の命令は 188 のままで、FMUL 4 本の減少は MOV 4 本の増加と入れ替わり、依存も短くならない。round 13-M の先読み移動とは別の案（算術の重複除去）。
- **案 2：棄却。** 代表 block が 188 → 204 命令（R2UR 12 → 21、UMOV 12 → 18〜19）、mainloop の中に STL/LDL が出た。過去の issue rotation（発行順の変更）や LDSM（A の読み方の変更）とは別の案。
- 両案とも既定 off の serving 経路の SASS は control と byte 一致。GPU での数値・速度は未検証。

## 結論

- consumer 側は、定常の 128 FMUL を安全に減らすか、追加費用なしにその完了待ちを隠す新しい CPU 証拠が出るまで打ち止めとする。
- 再開の条件：両 half と chunk 境界を含む正の短縮、spill 増なし、C75xx なし、control 不変を CPU で示すこと。その後の GPU gate は P の完全 gate を先に終え、完全 MegaMoE 時間、3 routing、12 GPU、正順・逆順、bit 一致または参照以下の誤差、耐久で判定する。
- round 19〜20 と合わせると、Prefill MegaMoE で残る差（routing の偏りによる約 130〜140 µs/層）の大半は 64 行単位の padding で、FC1 の割り当て順の変更（round 20）も consumer の再設計（本 note）も効かなかった。
