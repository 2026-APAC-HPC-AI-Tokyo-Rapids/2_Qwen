# 2026-10-19 Prefill attention prologue と連続 FP8 KV

no-prefix の Prefill EXTEND 用に、16-byte load の QK-norm/RoPE/FP8 kernel と、同じ kernel が出力する連続 FP8 K/V を読む FA3 varlen 経路を追加した。**CPU コンパイルと source 検査のみ完了。GPU の精度・速度・公式判定は未実行**。push、GPU job 投入、既存 job の操作はしていない。パッチは未配線で、統合担当の build と serving で判断する。

## 1. 根拠と範囲

起点は `16bdbe1`。SGLang `fc275674a97d78553b22597ee72c8fd48f5420f4` の私有 worktree に、現在の `scripts/build-sglang-container.sh` の **23 本**を配列順で適用した。その後に `patches/sglang/prefill-attention-contiguous-kv.patch` を当てる。共有 source は変更していない。DeepGEMM は変更しない。

[ボトルネック地図](2026-10-19-firmus-prefill-bottleneck-map.md)、[timeline](2026-10-18-megamoe-kernel-timeline.md)、[Prefill tail](2026-10-18-megamoe-prefill-tail.md)、[mainloop](2026-10-18-megamoe-prefill-mainloop.md)、[FA3 再検討](2026-10-18-firmus-fa3-reconsideration.md)、[小 kernel](2026-10-18-sglang-small-kernels.md)、関連 commit 本文を参照した。指定の geometry-and-redundancy note と `task/1. Qwen3 AI Task Rules.md` はこの worktree に存在しない。精度条件は今回のユーザー指示に従う。ローカル Codex 会話履歴には今回の kernel を説明する追加記録は見つからなかった。

run `39190318626` の `/srv/qwen-artifacts/39190318626/1/results/results/probe/{fa3,gemm,rope}.jsonl` を直接読み、logs/logs の probe log と Slurm accounting（24705、COMPLETED、2:47）も確認した。full clock 1.98 GHz で、FA3 3986 token の paged は 399.28 µs、varlen は 353.52 µs、non-paged kvcache は 350.72 µs。prologue は 4027 行で 122.12 µs、Q/K のみは 102.34 µs、copy floor は 38.08 µs。copy と型変換は同じ仕事ではなく、cast floor も 96.30 µs あるので、copy の 1.3 倍という目標は達成済みの下限ではない。

serving `39114949128` の `profile/profile/kernel-summary.json` も直接確認した。地図にある prologue 152 µs/層、FA3 約492 µs/層を serving の比較基準とする。単体 bench と serving は clock 条件が異なるため、その絶対値を直接比較して性能結論にはしない。

## 2. Kernel と精度

新入口 `apply_qknorm_rope_fp8_prefill` / `QKNormRopeFP8Kernel<PDL>::run_prefill`。1 CTA は 1 token の最大32 head、256 thread、1 head は8 lane。64 Q + 4 KV head は token あたり3 CTA になる。各 lane は NeoX の前半・後半をそれぞれ16 byte 読み、head を順番に回す loop をなくした。cos/sin の FP32 128要素と Q/K norm weight の BF16 256要素を CTA ごとに shared に置く。最後の CTA は4 KV headだけが有効で、無効laneもshuffleまでは参加する。

32 lane/head の旧配置を各物理lane内の4本の「仮想lane」に写像した。二乗和の各部分は旧laneと同じ順序（x0, x1, y0, y1）で FP32 加算。旧 XOR16/8/4 は8 lane内の XOR4/2/1を4本に実行し、旧 XOR2/1 はレジスタ間の `(partial0 + partial2) + (partial1 + partial3)` にした。数学的な reduction tree は維持し、命令配置だけを変更した。compiler の FMA 縮約差は規則上許容されるが、新 gate は旧融合 kernel との有限値 byte 一致も要求する。

RMSNorm は FP32、weight 乗算後は BF16 丸め、NeoX RoPE は FP32、回転後は BF16 丸め、その後だけ既存と同じ E4M3 FP8 変換をする。Q は FP8 のみ出力する既存契約を維持。K/V は paged cache と連続 `(tokens, kv_heads, 128)` の両方へ同じ FP8 値を書く。入力 BF16 QKV は変更しない。slot の範囲 assert と slot 0 の paged 書き込み抑制を維持し、連続出力には padding 行も一意に書く。新たに BF16/FP32 を FP8 化する計算区間や、FP8 未満の形式はない。追加連続 buffer は4027行で合計4.12 MB。

## 3. Routing / graph / DP

Qwen の既存 FP8 Q / 融合 KV 条件の内側で、no-prefix EXTEND だけ新 kernel を使い、K/V を FP8 の連続 tensor として attention に渡す。paged KV の書き込み済み flag により FA3 の二重 cache store は起きない。TARGET_VERIFY は従来入口を呼び続ける。

共通 host predicate `prefill_contiguous.py` は plain EXTEND、非空、全 request の prefix 0、encoder なし、LSE merge なしを要求する。backend はさらに FA3、MHA、FP8 Q/K/V、非 CP、非 local/SWA、causal、scale なし、cache 書き込み済みを確認する。連続 FA3 は既存 `self.flash_attn_varlen_func` を使い、`cu_seqlens_q` を Q/K 共通にし、real max length、通常の softmax scale/softcap、`num_splits` を渡す。pack_gqa、scheduler metadata、Decode の knob は変更しない。

request-aligned chunks の no-prefix Prefill はこの経路になる。prefix が1件でもあるバッチ、chunked continuation、Decode/IDLE/TARGET_VERIFY/draft/mixed は既存経路のまま。full prefill CUDA graph は1 captureを prefix有無で replay するため対象外にし、capture時の host predicateを固定して prefix KV を落とす事故を避けた。eager と breakable CUDA graph は forwardごとの host分岐を使う。既存 `tc_compiler == inductor` 除外は維持する。

DP attention では rank-local `extend_seq_lens_cpu` の合計を使い、他rankの `global_num_tokens_cpu` を長さに使わない。Q/K/V と出力をreal行数でsliceするので、DP/BCG paddingは FA3 のrequestに混入しない。padding時も FA3 が全体出力bufferの先頭へ直接書き、末尾だけゼロにする。大きな出力全体の再コピーは追加しない。`_attn_output` があればそのbufferを使う。

## 4. 検査と CPU コンパイル

- `check-firmus-prefill-attention-source.py <sglang-tree>`：実 predicate、実 Qwen producer、実 backend の分岐 AST を CPU tensor stub で実行。EXTEND no-prefix、prefix/継続、全 Decode mode、DP padding、full graph、CP、local/SWA、scale/dtype、LSE の契約を確認。PASS。
- 既存 `check-firmus-latency-source.py` / `check-firmus-megamoe-source.py`：PASS。latency mock に新 helper と graph backend 設定だけを補った。既存の期待結果を変更していない。
- `check-firmus-decode-kernels.py`：従来の gate と閾値を維持。新 kernel の0/1/3/31/128/129/1023/1024/3986/4027/4096行、packed/contiguous、int32/int64、slot 0、乱数slot、BF16 V全65536 bit pattern、入力不変、128/4027行で入力/position/slotを変えるCUDA graph replayを追加。非融合 norm→RoPE→cast の既存許容差と、旧融合経路との厳密 byte 比較を両方要求する。FA3 は複数requestとpadding形状で paged/varlenを比較する。**追加GPU gateは未実行**。
- `bench-firmus-sglang-kernels.py`：`prefill-prologue-dual-kv-*`、`prefill-extend-attention-*`、`prefill-prologue-and-extend-*` を追加。3986/4027行と複数requestを、回転する入力とCUDA graphで別々・合計の両方計時する。image は従来 prologue/paged、候補は dual KV/varlen。hashだけで精度判定しない。

`ssh firmus` の login CPU で、`unsquashfs` により `/scratch/home/ACCOUNT/qwen/containers/39110496019.sqsh` から `/scratch/home/ACCOUNT/claude-attn/image/` へ CUDA 13.0 と tvm_ffi include を抽出した。nvcc は **13.0.88**。`-std=c++20 -O3 --expt-relaxed-constexpr --ptxas-options=-v,-warn-spills -DSGL_CUDA_ARCH=900 -gencode arch=compute_90a,code=sm_90a -Xcompiler=-fPIC -c`、fast_math なし。既存 build check の10入口と新PDL有/無の2入口を object compile。新 kernel の PDL有/無 × int32/int64 は全て **46 registers、stack 0、spill store/load 0/0、shared 1024 byte、barrier 1**。cuobjdump で入力 `LDG.E.128`、weight `LDS.128`、8 lane subgroup shuffle を確認した。既存 token-warp store の instantiation も出力され、同じ条件では64 registers、spill 4/4 byte（本変更の新kernelではない）。FFI load/GPU実行はしていない。抽出物・object・診断用ファイルは作業終了時に削除した。

## 5. 統合担当の正確な配線

1. `scripts/build-sglang-container.sh` の現行 `gigatoken-encode.patch` までの全chainの**最後**に `patches/sglang/prefill-attention-contiguous-kv.patch` を追加する。manifest対象は `python/sglang/kernels/jit/csrc/elementwise/rope.cuh`、`python/sglang/kernels/ops/attention/rope.py`、`python/sglang/srt/models/qwen3_moe.py`、`python/sglang/srt/layers/attention/flashattention_backend.py`、新規 `python/sglang/srt/layers/attention/prefill_contiguous.py` の5本。
2. build/source validation で `python3 scripts/check-firmus-prefill-attention-source.py <source_dir>` を実行する。imageの必須patch一覧にも新patchを追加する。既存 build/image test の変更は本branchには入れていない。
3. `containers/check-rope-kernel-build.py` の specializations に `QKNormRopeFP8Kernel<true>::run_prefill` と `<false>::run_prefill` を追加する。JIT の新入口は rope.py 自身に登録済みなので、別共有libraryや追加dependencyは不要。
4. serving 前の既存 `check-firmus-decode-kernels.py` 呼び出しで追加gateも実行される。既存 MegaMoE gate の閾値・呼び出しも維持する。benchを走らせる場合は更新済み script を使う。新 runtime flag、dispatch input、matrix は不要。

## 6. 次の serving で見るものとリスク

まずgate全通過、既存Decode/verify kernelとpaged FA3の維持、Prefillで新prologueと非paged FA3が選ばれること、paged KV転送が正常なことを確認する。次に段階別 profile でprologue時間、FA3時間、pad output時間、staging/transfer時間を分けて見る。統合後の公式 workload/result parser とartifact回収まで完走することが必要。

FA3 の測定済み比率（353.5/399.3）なら約11.5%短縮の余地がある。prologue が目標50 µs/full clockまで下がり、servingを約1.25倍と仮置きできる場合、94層でprologue約8.4 ms + FA3約5.3 ms、合計約13〜14 ms/Prefill step（282 msの約5%）が目安。ただし**新prologueの実測はなく、帯域律速化も未証明**。cast命令量、shared access、最後のCTAの空きlane、launch数、allocator、電力clockの影響で目標未達もあり得る。単体50 µsとservingの改善は別々に判定する。

FA3のFP8 P tiling変更による既測定rel-L2約3e-3は許容される演算順差だが、公式精度判定の代用にはならない。full CUDA graphの連続経路は今回対象外。BCG/DPの実capture/replayはCPU stubでは証明できないためGPU gateとservingで確認する。1 runだけで性能優位を結論にせず、同条件の対照・再測定で比較する。
