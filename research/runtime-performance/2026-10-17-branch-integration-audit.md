# 2026-10-17 ブランチ統合時の監査

`perf/aspire-megamoe-port`（Firmus 6344.96、ASPIRE 3895.05）をmainへ統合した際、mainにもport側にも入っていないコミットを全ブランチから洗い出し、性能以外の修正も含めて仕分けた記録。対象は317コミット。patch-idの照合と、port側の木に同じ変更があるかの確認で判定した。

## 統合した基点

| クラスタ | 構成 | commit / run | good output tok/s |
|---|---|---|---:|
| Firmus（H200×16） | P3×DP4/EP4、D1×DP4/EP4、SGLang MegaMoE、c192、EAGLE3 3/4、FP8 KV | `e6e295b` / Actions 38664373784 | 6344.96（単回） |
| ASPIRE 2A+（H100×16） | MegaMoE P3/D1、c112 | `6d8d718` / Actions 38796892855 | 3895.05（単回） |

統合はport側を第1親にしたmerge commitで行った。`scripts/select-qwen-container.sh` は第1親を辿って再利用イメージを探すため、mainの旧ASPIREイメージではなくActions 38538953568のイメージ（SHA-256 `8df47c11…`）が選ばれる。

統合しなかったもの:
- komoriyutaが作業中のSM90 kernel系（`perf/firmus-*-20261017`）。`a6acf5f`（FC2 pipeline、Prefill rescale/phase、Decode tile、QK RMSNorm＋RoPE＋FP8 Q）はActions 38945855851で6521.98（単回、SLA適合94.86%）。個別検証後に別途マージする。
- vLLM系、TensorRT-LLM系、旧条件実験。結果は各研究ノートとActions履歴にある。

## 統合に合わせて取り込んだ修正

- profile回収のtimeoutを900秒に延長、profile clientが失敗してもcaptureとreport書き出しを待つ、検証失敗時も派生SQLiteを削除（`a54355a`、`96f2a3d` のSGLang該当部分）。
- 役目を終えた一時処理の削除。Slurm 21736 / 22351 専用の回収job、どこからも呼べなくなった1GPU CUTLASS grid probe。
- mainにだけあった修正の保持。`36a7268`（good outputの実測値と旧local scoreの分離）、worker別の起動コマンドログ、監視の通信断対策（port側が上位互換）。

## 未取り込みの候補（別ブランチで実装・検証する）

| 候補 | 元commit | 内容と注意点 |
|---|---|---|
| Nsight欠損警告の記録 | `a54355a` | `DIAGNOSTIC_EVENT` の「Not all CUDA events」とdriver非対応警告の検出。不合格にすると現行のSGLang profileも全滅するため、警告として記録する形が妥当。#24 §4 |
| Nsight exportをGPU割当外へ移す | `a54355a` | 割当内でexportがtimeoutした実績あり |
| 監視のSSH断パターン追加 | `8e80e60` | `Connection reset by`、`client_loop: send disconnect` |
| capture開始数をworker数と照合 | `7fd6803` | 固定値をやめる小さな堅牢化 |
| ベンチ前後の `/metrics` 保存 | `1ddc5a9`、`5e3b628` | prefill queue・forward・転送待ちの段階分解。#24 §1の検証に使える |
| ASPIREの区間Nsight採取 | `f84ea21`、`1ec209d`、`b8a7674` | 現行ASPIREの `[profile]` はkernelを取らない。nsysのreport書き出し中はSLA不適格になるため、profile時はeligibility判定を外す必要がある |
| 1GPU attention probe | `0e65dee`、`7b36982` | `research/aspire-eagle-decode-arfusion` にある |
| GPUサンプラーにmemory 2列 | `6cfd8f5` | 無害な小変更 |

## コミット本文にしかなかった知見

- **vLLMのE8M0再量子化（Addendum Q1違反の可能性）。** vLLM 0.30.0は既定で `VLLM_USE_DEEP_GEMM_E8M0=1` で、Hopperも除外しない。DeepGEMMを選ぶdense FP8 linearでは、checkpointのFP32 block scaleで逆量子化してpower-of-two scaleで再丸めする（`requant_weight_ue8m0_inplace`）。公開checkpointの再量子化にあたる。SGLang側は `check-firmus-latency-source.py` が `ue8m0 == False` を検査しているので影響はない（`1ddc5a9`）。結果はE1（E8M0 off）4702.50、E2（prefill MoEをDeepGEMMへ）5274.86、c108 5265.31。
- **router 0.3.2のload更新周期。** power_of_twoが使うload更新は `load_check_interval_secs=5` ではなく `--worker-startup-check-interval`（既定30秒）で回る。1秒にするとSLA適合が20.8%から98.35%（2226.3）に回復した（`cc5beca`）。port側のrouterパッチやmanual policyを評価するときの比較キーにする。
- **ASPIREでのTensorRT-LLM rc20。** DeepGEMMのNVRTCがヘッダ探索にpipを使う、NVRTC smoke checkに `-default-device` が要る、gatewayに `NVIDIA_DRIVER_CAPABILITIES=compute,utility` とGPU0が要る、MPIは `localhost:TP --oversubscribe`。最後は `max_completion_tokens` の `extra_forbidden` でHTTP 400になり、scoreは出ていない（`research/aspire-trtllm-localmpi-c40`）。Firmus rc29での対処は [2026-10-09-prefill-supply-rebalance.md](2026-10-09-prefill-supply-rebalance.md) §17。
- **NGRAM＋PD。** 初期 `spec_info` が無いSGLangのバグを直すパッチがある（`ed4c823`）。NGRAMは998.1でtarget-only 903.9より上だが、EAGLE3には大きく劣る。
- **AIPerfのcache-bust marker。** [../simulation/PREFIX.md](../simulation/PREFIX.md) はAIPerf 0.12.0でmarkerがwireに出なかった（CPU捕捉で0/4件）と記録している。一方 [2026-10-09-prefill-supply-rebalance.md](2026-10-09-prefill-supply-rebalance.md) はmarkerをprompt先頭に置くと説明している。どちらも「prefix再利用はない（実測約0.077%）」という結論は同じだが、仕組みの説明が食い違っており、どちらが正しいかは未確認。
- **Firmus P6TP2/D1TP4でDecode MoEをDeepGEMMにした試行**（`81d693e`、Actions 38029268165、Slurm 22295）はFAILED 1:0で終わり、原因は調べていない。

## 残すブランチ

mainの文書が参照しているか、そこにしかないコードがあるため、削除しない。
- `research/cpu-screen-20261003-sol`（`5207e1b`。[2026-10-03-bounded-optimization.md](2026-10-03-bounded-optimization.md) が再現用の正本として参照）
- `research/aspire-eagle-decode-arfusion`（ASPIRE旧最良 `ed8c534`、attention probe）
- `research/aspire-dynosim-aisimulate`（`c7f89db`。AISimulateの校正・予測ツール一式。文書だけ [../simulation/](../simulation/) へ取り込んだ）
- `research/firmus-trtllm-p6tp2-d1tp4`（TensorRT-LLM rc29の互換patch。参照用）

## ブランチの棚卸し（同日）

- mainに完全に含まれていた26本は、そのまま削除した。
- 結果を記録済みの実験ブランチ72本は、先端に注釈付きtag `archive/<元のブランチ名>` を打ってから削除した。復元は `git push origin 'archive/<名前>^{commit}:refs/heads/<名前>'`。
- 残したブランチは、komoriyutaが作業中の `perf/firmus-*-20261017` 7本、上の「残すブランチ」4本、queuedのまま止まっているrunを持つ2本（`research/aspire-eagle-control-pair2` の36301755811、`research/aspire-moe-campaign-20261003` の37091361072）、Claude PR Assistantの雛形 `add-claude-github-actions-1792083267104`。

tagのpushで注意すること:
- tagのpushでは、tag先のコミットにある `qwen.yml` が使われる。mainで `branches: ["**"]` を指定（`b078913`）しても、古いコミットに打つtagには効かない。tag push ではpath filterも評価されない。
- 古いブランチの先端にはクラスタタグ付きのコミットが多い。1本だけpushしたところrun 38996491119が起動し、Firmusの配備中（sbatch前）にキャンセルした。
- GitHubは、1回のpushで4本以上のtagを作るとpushイベントを作らない。4本で確認したあと、残り67本を1回のpushで送り、runが起動しないことを確かめた。
