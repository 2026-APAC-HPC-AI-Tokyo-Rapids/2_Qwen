# Round 41：MegaMoE scale 経路の露出上限 probe

**CPU 準備完了、GPU 未実行・未投入。完全な submission には require-build が必要。** B/C pin と r37 gate 廃止は前回の作業どおり保持した。親 session は、FMUL だけでなく RCP と scale load も消える診断を、scale 経路全体の露出の **上限** として受理した。前回の手順3停止はこの判断で解除された。診断出力を serving に使用する経路は無い。

## 目的と判定規則

r40 は、旧 r24 完全 kernel の 2648.357750 µs/層へ rescale 露出 0、50、100 µs/層のいずれを配賦しても同じ合計へ fit でき、mainloop 再設計を判断できなかった。今回解くのは、現 first-K 採用済み H200 MegaMoE の scale load、reciprocal、rescale multiply とその依存をまとめて取り除いた場合の露出上限である。FMUL 単独の実測、exact redesign の実現可能な speedup、通常 serving score とは区別する。

同一 GPU の production minus diagnostic の対応差を両順各12 roundで測り、GPU ごとの24対応差の中央値、その12 GPU の中央値 Δus を求める。

- **12 GPU 中央値が 50 µs/層未満なら mainloop 再設計方向を閉じる。** scale の仕事を保持する再設計はこの上限を超えて短縮できない、という今回の判断規則による。
- **50 µs/層以上なら条件付きでのみ開く。** 測定差は依然上限であり、loads/RCP の寄与を分離する必要が残る。達成可能な exact speedup としない。
- 欠損、clock 不足、worker 非0終了、timeout は incomplete、開始予算不足は skipped。未知値を0で埋めず、これらに採否を付けない。

H100 模型、別 run の clock や計装 timeline を移植しない。測定値と通常 serving の公式 parser 結果は別であり、今回の CPU 準備から性能の結論は出さない。

## 維持した B/C 固定と gate 廃止

親 `4052d13`、branch `perf/pp4-r40-megamoe-mainloop-20261104`。AGENTS.md、前回 prompt/final、既存 diff、r37–r40 note、tracker/model、保存 profile 集計、commit 本文、前回 Actions 実績を確認した。今回の継続では SSH は不要だった。

`SGLANG_PREFILL_QKNORM_FIXED=1` と `SGLANG_PREFILL_ONEPASS_DIV448=1` は A と同じ `PP4_SELECTED` 経路、両方の根拠 run は `41822975123`。source/results manifest と Prefill env に反映し、Decode env へ渡さない。根拠は run41822975123/image5018d322 の全12 GPU gate 合格、全12 process の prewarm/実 dispatch、公式10058.37。単一 score を新しい優位性の証拠としない。

r37 の sbatch selection、selector/wrapper、staging、config/manifest key、選択専用 tests は廃止済み。production kernel、ready 前の A/B/C prewarm と aiperf 後 `prefill-r37-dispatch-*.txt` 回収を保持する。他の既存 gate は変更しない。pin/廃止だけなら build 不要だが、以下の host 変更を含む今回全体は build が必要。

## 診断の到達性と production JIT の検証

既存 host interface は固定 template 引数列と FC1 first-K define だけを生成し、独立した診断引数を公開していない。header-only overlay だけでは選択不能なので、`patches/deep-gemm-host/sm90-rescale-exposure-probe-r41.patch` を host series に追加した。内部 `DiagnosticNoRescale` template flag は既定 false。通常 TVM-FFI `fp8_mega_moe` は **literal false** に固定し、別 entry `fp8_mega_moe_rescale_exposure_diagnostic` だけが true を instantiate する。env による診断選択は無い。builder の CPU ABI import でも別 entry の存在を確認する。

runtime header overlay は flag 未定義なら0とし、split FC1/FC2 の8本の rescale 乗算文だけを constexpr guard で囲む。診断 JIT source だけが flag=1と別 kernel symbol `sm90_fp8_mega_moe_rescale_exposure_diagnostic_impl` を define する。true は EP1/T7988/top8/FC1 first-K の host assertion も通る必要がある。通常 API の source 生成文字列は旧実装と同じで、CPU の実 generator compile/実行で検証する。SGLang、serving env、production Python wrapper は別 entry を参照しない。

probe は GPU import 前と import 後に、serving env に `DG_R41_DIAGNOSTIC_NO_RESCALE` が存在しないことを assert する。値0でも拒否する。通常 Prefill と同じ `prefill-kernels.env` を読み、worker と同じ LOCAL_EP1/2WG を設定し、4stage/lean/split/first-K、tail-pair無効、compiler override無しを assert する。

production は **image の `deep_gemm.fp8_mega_moe` そのもの**を呼び、同 wrapper が通常 `_C.fp8_mega_moe` を参照し診断 entry を参照しないことも assert する。prewarm 後、rank 専用 fresh JIT cache に実際に生成された両 source を読み、共通 header hash、FC1 first-K define、r40 の正確な28引数列、診断 define/symbol prefix 以外の全文一致を assert する。同じ host Compiler object で両方を build するため compiler flags も同じであり、probe は別 compiler を設定しない。各 variant の JIT cache key、cuobjdump で得た kernel name、generated source SHA256 を rank 結果へ記録する。

親判断に従い、現 image SASS との別の equality proof は要求しない。CUDA13.4 compile を image CUDA13.0 の実測 SASS と呼ばない。今回 require-build 後も production variant はこの通常 JIT path を通る。

## ワークロードと測定

EP1、T7988=2×3994、E128/top8/H4096/I1536、capacity8448。FP8 E4M3 operand、元 group128 scale、FP32積和、FC2 BF16 partial。first-K、4stage/lean/split、M128/N256/K128、384 threads、132 CTAs は production と同じ。固定 seed の synthetic FC1 `(128,3072,4096)`、FC2 `(128,4096,1536)` FP8 weight と非定数正 FP32 block scale を使い、実 model weight は読まない。weight 本体計2.25 GiB/GPU、作成時 BF16 temporary と full-pool buffer は別途必要。

balanced 代用でなく、既存 `kernels/probes/megamoe-local-ep1-routing.npz` の `logical_ids[:,0,0,:3994,:].reshape(7988,8)` を使う。SHA256 `1907184191e49a6dcc1431cdadafe9bfef2eb90bf4ad3f58407f6ed32a136b52` を assert。これは旧 historical routing であり round39 の実 routing を再現する主張ではない。有効63904行、実行68288行、定常292358 CTA-K128相当は r40 と同じ。

Decode greedy gate 後・Prefill server 起動前、3 worker × 1 torchrun × 4 GPUを並列実行する。各 rank は単独 EP1 buffer を持つ。production/diagnostic の JIT、各3 warmup、別 CUDA graph capture と replay は計時前。入力 pre-dispatch は計時外で一度実行し、両 graph は同じ入力 buffer を再利用する。1 graph は8 layer call、event で測った replay 時間を8で割る。12 round、それぞれ forward prod→diag と reverse diag→prod を交互に保存し、計384 timed call/GPU。診断出力は破壊された counterfactual なので、数値一致や serving adoption gate としない。

計時中は GPU UUID 指定の `nvidia-smi --loop-ms=100` で timestamp/UUID/SM MHz を rank CSV に採取する。rank JSON にも有効 sample を含め、同 probe の平均 MHz と Δus から `b = Δus × 132 × fMHz / 292358` clock/定常K128 CTA相当を算出する。1410 MHzなら50 µsは31.830838 clock。同窓 clock sample の平均は一定の effective clock の証明とはしない。 UUID は GPU-prefix を正規化して query と CSV 照合を一致させる。計時後は r33/r34 で成功した順序に従い graph参照、全DLPack view、EP1、WORLD の順に解放し、終了失敗も incomplete とする。

`results/megamoe-rescale-probe/worker-N/rank-R.json` に paired raw timing、両順と統合中央値、clock samples、UUID、image/run/fixture provenance、実 template と env、両 JIT identity を保存する。raw clock CSV、JIT cache の source/cubin、worker logs も残す。host aggregate が12 unique GPUと全worker終了0等を確認し `summary.json` に GPU別中央値と12GPU中央値・上限の判定を書く。既存 Firmus workflow は logs/profile/results/source 全体を terminal 後回収するため probe も artifact に含まれる。serving settings へ書き戻す処理は無い。

## CPU compile / SASS

最終 runtime overlay を実適用し、CUDA13.4.92/sm90a/O3/C++20 で両 variant を fresh compile。production 9.614秒、diagnostic 8.794秒、直列合計18.408秒。両方168 launch registers、stack/spill store/load は0。各 encoding は前回保存 proof と全byte一致。production `f157a6dddb926f5944f751e46448afb69091eb9cfbcccd17e21fe852205e8c41`、diagnostic `3017e8b4ab55415359b77e2636f034fb2a19b5c15758a834daaa0a081ffcff7b`。

| 静的命令 | production | diagnostic |
|---|---:|---:|
| FMUL | 4074 | 522 |
| MUFU.RCP | 178 | 70 |
| LDG.E.CONSTANT / LDS | 64 / 61 | 12 / 9 |
| QGMMA / UTMALDG.2D | 224 / 51 | 224 / 51 |
| WARPGROUP.ARRIVE / DEPBAR.LE | 56 / 56 | 56 / 56 |

FMUL3552、RCP108、scale load104本が消える。integer/MOV/register割当も再配置される。WGMMA/TMA/barrier/epilogue source は保持し、静的主要本数も同じ。**この差を FMUL 単独の露出と解釈せず、scale 経路全体の上限とすることが受理条件。** 初回の不合格 admission 記録も raw proof に残し、現在の受理判断と区別する。

## 予算

旧 gate はGPUで21–22秒、今回 CPU compile 直列18.408秒。これは4rank同時cold JITやimage CUDA13.0の所要時間を保証しない。384 call × 旧r24約2.65msなら timing 約1.02秒。torchrun/import、synthetic weight/buffer生成、cold JIT、capture、clock取得を含め worker **35–65秒という条件付き見積もり**、95秒内保証は未測定。

start condition は厳密に `SECONDS + 105 + 630 <= 1020`、開始上限285秒。worker は95秒 TERM/2秒kill grace、全体は **105秒 SIGKILL group timeout**。worker timeout は foreground にして外側の同一 process group に残し、3 workerを並列起動して全終了コードを集約する。開始不足なら skipped、失敗/timeoutなら incomplete を残し、通常 serving を続ける。630秒 reserve と serving 設定は保持する。

## 仕事量、検証と残る制約

通常側の有効仕事はFC1 1.608196816896、FC2 0.804098408448、計2.412295225344 TFLOP/層。主要論理I/OはFC1 47.571849216、FC2 34.325483520 GB/24層。実HBM/L2/NVLink byte、GPU時間、clock、公式scoreは未測定。診断はscale経路の仕事を意図的に削るが、serving精度変更ではない。過去のFFMA spill、shared-RCP不一致、split-LDSM回帰、tail/direct-FC2/A-gather/sched-r20等の棄却理由はr40/trackerどおりで、再試行しない。

CPU tests は git履歴を使わず、既存 B/C pin/prewarm/dispatch、host patch適用、旧/新production generator一致、別APIのliteral false/true、flag拒否、cache source/geometry assertion、12GPU schema/閾値、欠損/clock不足/timeout等の不完全集約、sbatch予算/staging/成果物回収を検証する。指定 pytest/Ruff の最終終了コードと logs は隣接 CPU-validation JSON、および指定 `codex-scratch/r41b/` を正本とする。

image rebuild、GPU/JIT実行、scheduler起動、SSH、commit/pushは今回行っていない。CPU compile/SASS はimage内 host extensionの実buildやGPU動作確認の代替ではなく、実測は将来の明示 submission に残る。`submission/` は変更していない。
