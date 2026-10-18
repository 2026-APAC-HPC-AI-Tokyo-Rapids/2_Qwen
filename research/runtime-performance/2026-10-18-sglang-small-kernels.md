# 2026-10-18 SGLang の attention 前後の小 kernel（Prefill / Decode）

対象は [Firmus 計画](2026-10-18-firmus-maximum-performance-plan.md) §2.3 の 2-2（融合 QK-norm＋RoPE＋FP8 Q の帯域化）、2-3（RMSNorm＋量子化の融合）、2-6（FA3 decode の runtime knob）、2-7（Decode 小 kernel の融合）。branch は `perf/firmus-max-sglang-kernels-20261018`。GPU での計測はまだ行っていない（ユーザー指示により、全担当の候補を一つの kernel-bench job にまとめて投入する。単独で push した run 39094348304 は Slurm 開始前に取り消された）。以下の効果は profile と帯域からの見積もりで、確定値ではない。

## 1. 精度の扱い

ユーザーが task/1 の Q1/Q3/Q4 に照らして確認した基準に従う。浮動小数の加算順序・FMA 縮約・reduction 順序による差は許す。未融合の FP32/BF16 参照に対して丸め程度の許容差に収めることを求める。精度を下げること（いまの image より低い精度での保存・転送・計算、FP8 未満）と、計算の省略・近似は行わない。本稿の変更はどれも、image が既に FP8 で持っている Q、KV cache、QKV GEMM 入力以外を FP8 にしない。

各 check の許容差は次のとおり。

| 対象 | 比較先 | 許容差 |
|---|---|---|
| QK-norm＋RoPE＋FP8 Q（既存 check） | fused_inplace_qknorm → apply_rope_inplace → `.to(fp8)` | 既存の粗い screen（相対 L2 ≤ 0.05、最大誤差/最大値 ≤ 0.25）。本変更で image と byte 一致を保つので緩めていない |
| FP8 KV 書き込みの融合（komoriyuta） | 同じ tree の FP8 Q kernel ＋ torch の FP8 cast ＋ store_cache | NaN の payload を除き byte 一致 |
| masked Top-K（komoriyuta） | AOT topk_softmax ＋ padding mask | byte 一致 |
| 加算 RMSNorm ＋ group-128 量子化の融合 | flashinfer fused_add_rmsnorm ＋ per_token_group_quant（列優先・TMA 整列） | residual は byte 一致。scale は 1 BF16 ulp（相対 2^-7）以内。逆量子化値は E4M3 の 1 段（相対 2^-3、subnormal 域は 2^-9×scale）＋scale 差以内。異なる FP8 code は全体の 0.1% 以下 |

komoriyuta の gate run 39077653567 にある `rope_k_normTrue_31 bitwise_equal=false`（相対 L2 4.8e-7）は、KV 融合ではなく image の既存 kernel の性質。image の融合 kernel は lane l が要素 2l, 2l+1, 64+2l, 65+2l を持って二乗和を取るのに対し、参照の fused_qknorm_warp は lane l が 4l..4l+3 を持つ。lane 内の部分和の組が違うので FP32 の丸めが変わる。上の基準では許容される差で、既存の許容差の範囲に入っている。参照と byte 一致する版（lane 配置を fused_qknorm_warp に揃え、NeoX の相手要素を xor-16 shuffle で交換し、回転を fused_rope_kernel の SM90 の縮約形 `fma(x,cos,-(y·sin))`, `fma(y,cos,x·sin)` で明示する版）も作り、SASS で確かめた（old-af55451 の系列）。image と hash が一致するほうが kernel-bench の比較が単純なので、採用版は image と同じ演算にした。

## 2. patch の系列

`qknorm-rope-fp8.patch` の後に次の順で当てる。fc275674 ＋既存 15 本の上で `git apply --check` が順に通り、得られた tree が作業 tree と一致することを確認した。

| 順 | patch | 作者 | 対象ファイル |
|---|---|---|---|
| 16 | `qknorm-rope-fp8-kv.patch` | komoriyuta（`perf/firmus-evidence-20261018` 6102745 をそのまま） | rope.cuh、ops/attention/rope.py、models/qwen3_moe.py |
| 17 | `qwen-topk-padding.patch` | komoriyuta（同、そのまま） | jit/csrc/trtllm_lora_temp/topk_softmax_pack.cuh、ops/moe/trtllm_lora_temp/topk_softmax_pack.py、layers/moe/topk.py |
| 18 | `qknorm-rope-fp8-token-warps.patch` | 本作業 | rope.cuh、models/qwen3_moe.py |
| 19 | `add-rmsnorm-group-quant.patch` | 本作業 | 新規 jit/csrc/elementwise/add_rmsnorm_group_quant.cuh、新規 ops/layernorm/add_rmsnorm_group_quant.py、models/qwen3_moe.py |

komoriyuta の 2 本の不具合と修正は本人の commit にある。0 行入力の gate 失敗は参照側が `torch.empty_like(0 行の packed view)` で FP8 Q を確保し、stride (9216,128,1) が残って FFI の密な出力という条件に合わなかったもの（6ca388d）。Decode の graph 初期化失敗は fc275674 の ForwardBatch に `token_to_kv_pool` が無いことによるもので、`get_token_to_kv_pool()` に替えた（6102745）。run 39087648892 で FP8 KV・Top-K・EP4 の数値 gate と CUDA graph gate は通過済み。

### 2.1 token-warp の QK-norm＋RoPE＋FP8 Q（18）

image の kernel は 1 head を 1 warp に割り当てる persistent grid で、head ごとに position 読み → cos/sin 読み → head 読み → shuffle reduction を直列に行う。warp あたり同時に飛んでいるのは 1 head の 256 B だけで、Prefill 4027 token × 68 head の 110 MB 前後を 141 µs（約 0.8 TB/s）で処理していた。

新しい kernel は warp が 1 token の head {g, g+G, g+2G, …} を受け持つ。position、cos/sin、q/k 両方の norm weight を token あたり 1 回だけ読み、6 head 分の読み込みを reduction の前にまとめて発行する。G は 1024 token 以上（Prefill）で 4（1 CTA = 1 token、warp あたり 17 head）、それ未満（Decode）で ceil(68/6) = 12（warp あたり 1 chunk）。head ごとの演算、lane 配置、KV 書き込みの意味（slot 範囲の device assert、CUDA graph の padding 用 slot 0 は書かない）は 16 と同じで、出力は image と byte 一致する見込み。FFMA の連鎖、xor 16..1 の reduction、rsqrt、回転の SASS を login node の nvcc（image 内の cu13.4）で旧 kernel と照合した。入口は `FusedRopeKernel::run_norm_fp8` / `run_norm_fp8_store` のまま残したので、rope.py と build 時の specialization 一覧は 16 から変わらない。fused_rope_kernel から kQKNorm/kFP8KVStore を外した。登録数は 56〜64 register、spill なし（KV 書き込み版は assert 呼び出しの 8 B stack のみ）。

融合 KV 書き込みは Prefill の EXTEND にも広げた。FA3 は Prefill の KV も同じ `set_kv_buffer(out_cache_loc)` で書き、Prefill では FP8 cast 2 回＋store_kvcache が 1 層 21.6 µs ある。Prefill の server に speculative 引数が無い場合も通るよう、top-1 条件は `speculative_eagle_topk in (None, 1)` にした。

### 2.2 加算 RMSNorm ＋ QKV 量子化（19）

input_layernorm（加算つき）の BF16 出力は qkv_proj にしか使われない。そこで RMSNorm と per-token-group-128 の FP8 量子化を一つの kernel にし、BF16 の正規化出力（Prefill で 33 MB）の書き出しと再読込をなくす。FP8 の (q, scale) は image でも QKV GEMM の入力として作っているもので、精度は変わらない。kernel は flashinfer の CuTe DSL FusedAddRMSNormKernel（H=4096 で 128 thread、1 CTA 2 行、64 thread/行、列 512b+8t の 16 B vector、xor 1..16 の warp reduction と 2 warp の部分和、rsqrt.approx.ftz）と QuantTrait<bf16, fp8, 128> に沿う。DSL の compiler が決める 3 点（thread 内の加算順、二乗の FMA 縮約、mean＋eps の FMA 縮約）は 8 通りの template にしてあり、check が各 variant の byte 一致を情報として出す。既定の variant は許容差を満たせばよい。DeepGEMM の block-FP8 linear は (q, scale) の tuple を既に受け付けるので、decoder layer はその tuple を attention に渡す。経路に入るのは residual がある層で行がある場合に限り、EAGLE3 の aux hidden 取得層、all-reduce 融合、attention TP > 1、batch-invariant mode、UE8M0 scale では従来の経路のまま。

## 3. 見積もり（未計測）

根拠は A1 profile（run 39070273796、prefill-0 / decode-0 の kernel-summary）と、H200 の実効帯域を 3〜3.5 TB/s と置いた転送量。

| 項目 | 今の 1 層 | 見積もり 1 層 | Prefill ms/step（94 層） | Decode ms/step | 根拠 |
|---|---:|---:|---:|---:|---|
| QK-norm＋RoPE＋FP8 Q（Prefill） | 141 µs | 45〜55 µs | −8〜9 | — | 読み Q 66 MB ＋ K 4 MB、書き Q 33 MB ＋ K 4 MB |
| 同（Decode、126 行） | 5.3 µs | 3〜4 µs | — | −0.1〜0.2 | token あたりの依存読みが 1 回になる |
| FP8 KV 書き込みの融合 | Prefill 9.3×2＋3.0、Decode 2.5×2＋1.7 µs | kernel 内で +2〜3 µs / +0.5 µs | −1.7 | −0.55 | cast 2 回と store_kvcache の launch がなくなり、V 4 MB の読みと K/V 2 MB の書きが残る |
| 加算 RMSNorm＋QKV 量子化 | Prefill 31.6＋20.2 µs、Decode 3.2＋2.0 µs | 35〜40 µs / 3〜3.5 µs | −1.1〜1.6 | −0.15 | BF16 正規化出力 33 MB の書きと読みがなくなる。post_attention_layernorm は対象外 |
| masked Top-K | Decode 1.3 µs | 0 | — | −0.12 | mask launch がなくなる |
| 合計 | | | −11〜12（約 4%） | −0.9〜1.0（約 2.5%） | |

Prefill 1 要求の GPU 時間に比例して供給上限が上がるので、Prefill の約 4% がそのまま供給上限の上積みの目安。Decode は ITL の余裕に回る。

### 3.1 FA3 decode（2-6）

sgl_kernel の `flash_attn_with_kvcache` が受ける runtime knob は `num_splits`（0 は heuristic）、`pack_gqa`（None は heuristic）、`scheduler_metadata`。TARGET_VERIFY は scheduler metadata を事前計算しておらず、層ごとに prepare_varlen_num_blocks（3.5 µs/層、0.33 ms/step）が走る。DP attention では事前計算を backend が意図的に無効にしている（rank 間で cache_seqlens/num_splits が変わると split-KV combine で範囲外読みが起きた、という記録がある）ので、本作業では有効にしていない。bench script は batch 32/36 × seqlen 3900〜4200 × FP8 × page 64 で、knob ごとの時間と既定呼び出しとの最大絶対差を出す。num_splits の変更は加算順序の差で、上の基準では許されるので、計測で速ければ backend の `self.num_splits` を TARGET_VERIFY だけ変える小さな runtime patch が候補になる。sgl-kernel の再ビルドは対象外。

## 4. 検査

- CPU のみで実施：19 本の patch 系列の適用、py_compile、Ruff（`qwen-experiment-metrics scripts containers`。指摘は main 既存の bench-firmus-megamoe.py の E402 のみ）、`check-firmus-latency-source.py`（FP8 KV の選択に EXTEND と speculative 引数なしを追加、Top-K の条件、融合 norm＋量子化の選択）と `check-firmus-megamoe-source.py` を patch 済み tree に対して実行して全 PASS。login node の nvcc で rope.cuh（build check の全 specialization）、add_rmsnorm_group_quant.cuh（8 variant）、topk_softmax_pack.cuh（--use_fast_math）が compile できることを確認。validate job（run 39094348304 の validate と配備）も通過し、配備段で 4 本の patch が image の tree に当たることも確かめた。
- bench script の出力は summary script の契約（case ごとに 1 行、両 tree で同じ case 集合、`case`・`median_us`・`p10_us`・`p90_us`・`output_sha256`）に合わせ、合成した JSONL で `summarize-megamoe-kernel-bench.py` が表を作ることを確認した。summary は hash 不一致を FAILED にするが、上の精度基準では `input-norm-quant-*` の hash 差は情報扱い。
- GPU での check と bench は統合 kernel-bench job で行う。`patches/sglang-bench/10〜40` に 16〜19 を置いてある。

## 5. 配線（統合担当向け）

- `scripts/build-sglang-container.sh`：patch 変数に `qknorm-rope-fp8-kv.patch`、`qwen-topk-padding.patch`、`qknorm-rope-fp8-token-warps.patch`、`add-rmsnorm-group-quant.patch` をこの順で `qknorm-rope-fp8.patch` の後に追加（適用 loop と manifest loop の両方）。targets は、kv と token-warps が `rope.cuh rope.py qwen3_moe.py`（token-warps は rope.py を変えないが同じ組でよい）、topk が `trtllm_lora_temp/topk_softmax_pack.cuh`、`ops/moe/trtllm_lora_temp/topk_softmax_pack.py`、`layers/moe/topk.py`、norm-quant が `jit/csrc/elementwise/add_rmsnorm_group_quant.cuh`、`ops/layernorm/add_rmsnorm_group_quant.py`、`srt/models/qwen3_moe.py`。
- `containers/check-rope-kernel-build.py`：komoriyuta 版（`run_norm_fp8_store` の 2 specialization と topk の `--use_fast_math` compile を追加したもの）をそのまま使える。加えて `elementwise/add_rmsnorm_group_quant.cuh` の `AddRMSNormGroupQuantKernel<true,0>::run` と `<false,0>::run` を同じ方法で compile する項目を足す。
- `containers/Dockerfile` と build script：komoriyuta 版の `check-firmus-latency-source.py --installed` を image 内で実行する 2 行（Dockerfile の COPY/RUN と build script の cp）。本 branch の `scripts/check-firmus-latency-source.py` は komoriyuta 版に EXTEND と norm-quant の検査を足したもの。
- `scripts/check-firmus-image.sh`：applied_patches の必須一覧に 4 本を追加。
- serving の gate は `check-firmus-decode-kernels.py` が新 kernel を含めて実行する（追加の配線は不要）。

## 6. 残るリスク

- すべて未計測。token-warp kernel の 45〜55 µs は帯域からの見積もりで、chunk 6 / G=4 の選び方は計測後に調整の余地がある。kernel のコード量が増え（6 head 展開）、I-cache の影響は未確認。
- 融合 norm＋量子化は 89 register で 1 SM 5 CTA。flashinfer 版の cp.async 経由の読みより遅い可能性があり、Prefill で負ける場合は採らない。
- Prefill EXTEND への KV 融合の拡大は CPU の選択検査のみ。Prefill の FA3 が EXTEND でも cache から読むこと（BF16 K を使わないこと）はコードで確認したが、serving の 2 run での確認が要る。
- 融合 norm＋量子化は model 側で LayerCommunicator.prepare_attn を迂回する。DP attention・attention TP 1 の現行構成以外（TP>1、scatter、all-reduce 融合）は条件で外しているが、構成を変えるときは見直しが要る。
- FA3 の num_splits は本 branch では変えていない。速くても DP rank 間で split 数が揃うかを確かめてから入れる。
