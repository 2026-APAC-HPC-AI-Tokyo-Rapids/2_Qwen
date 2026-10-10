# DP attention＋DeepEP への構成転換：検証手順書（2026-10-10）

担当: 安価なモデル／人手で実行可能なように、判断の根拠・変更箇所・合否基準を列挙する。本稿は計画であり、結果は本稿末尾の「結果記録」節に追記する。

## 0. なぜこれか（根拠は 2026-10-09-prefill-supply-rebalance.md §1/§4/§7）

- 6000 good tok/s は入力 120k tok/s、10000 は 200k tok/s を要する。現行の持続Prefill速度は H100 TP4 7.3k/GPU、H200 TP2 10.75k/GPU。PD分離でDecodeに4 GPU残す限り、**現行効率では両クラスタとも算数上到達不能**（ASPIRE: 12 GPU×7.3k=88k→4400、Firmus: 12×10.75k=129k→6450）。
- Prefill profile（§4）では kernel時間の 23% が TP all-reduce、19% が MoE補助（全65536 route の shuffle/combine/sort）、MoE GEMM は 28%。FP8 dense MFU 約16%。**TP4/TP2 という分割単位そのものが律速**。
- Decode（§7）は TP4 で全expert重み 57 GB/rank を毎stepストリームし MoE だけで 18.5 ms/step の固定費。8 GPU で1コピーにすれば固定費は半分。
- DP attention（attention は rank ごとに独立、all-reduce 消滅）＋ EP（expert を rank 間で分割、DeepEP all-to-all）は SGLang の Qwen3-235B/DeepSeek PD recipe の標準形。ルール（task/1. Qwen3 AI Task Rules.md）も「attention 単位の細かい分割は許容」「cross-node TP が遅いのは自己責任」と、この方向を前提にしている。
- 過去の試行（research/optimization-history/PARALLELISM.md）: 「P DP attention2」は NIXL REMOTE_DISCONNECT で**性能未測定**、TP8EP8 は旧条件で SLA不適格のみ。DeepEP を正しく使った run は一度もない。
- 検証済みで問題ない前提: AIPerf の ITL は token基準 `(latency−TTFT)/199`（spec decode の効果は計上される）、cache-bust で prefix cache は効かない、score = goodput(req/s)×200、accept len は約2.25（ランダム語列のため低い）。

## 1. 固定 image に必要物は揃っている（確認済み）

image `/scratch/home/ACCOUNT/qwen/containers/37886841020.sqsh`（sha256 4b919a74…、Firmus 現行）に以下が入っている。再 build は不要。

- `site-packages/deep_ep`（1.2.1）、`deep_ep_cpp…so`、`deep_gemm`、`nvidia/nvshmem/lib/libnvshmem_host.so.3`
- `sglang_router-0.3.2`
- SGLang 側（`unsquashfs -cat` で読んだ server_args.py / disaggregation / data_parallel_controller）:
  - `--moe-a2a-backend deepep` を付けると `ep_size` は自動で `tp_size` になる（`_a2a_ep_size`）。**`--ep-size 1` を同時に渡さない**。
  - `--deepep-mode normal` は CUDA graph を無効化する（Prefill には無関係。Decode に使うなら `low_latency`）。
  - DP attention は `chunked_prefill_size` を `dp_size` で割る（**65536 を渡すと rank あたり 8192**）。`schedule_conservativeness` も ×0.3 される。
  - PD の Prefill worker は `disaggregation_mode == prefill` で dispatch が `follow_bootstrap_room`（`bootstrap_room % dp_size` の rank へ）になり、Decode 側は `_resolve_prefill_dp_rank` で同じ rank を解決する。router 側の追加設定は不要。
  - EAGLE3 と DP attention の併用コードパスは存在する（`speculative/` 内で `enable_dp_attention and is_eagle3()` 分岐）。起動時 assert が出たらその文言をそのまま記録する。

## 2. 実験A（最優先）: Firmus P1×world8（TP8 / DP8 attention / EP8 DeepEP）＋ D2×TP4

目的: **GPU あたり持続 Prefill tok/s** を DP attention＋DeepEP で測る。Decode は既知構成（TP4、page64、pool 524288）を 2 本にして P を飽和させる。

### 2.1 ブランチと親

- 親: `origin/research/firmus-p6tp2-d1tp4` の **3519fc0**（P6TP2/D1TP4 page64 c104 = 4429.9、暫定最良）。d039410（c112）は結果待ちなので親にしない。
- 新ブランチ名: `research/firmus-p8dpep-d2tp4`。チームメイトの `codex/firmus-*` は触らない。
- コミットメッセージ先頭: `[cluster:firmus] [nodes:2]`。本文に本稿の根拠を要約。

### 2.2 `configs/firmus-framework.json`

- `concurrency`: **160**（D2×TP4 で running 約2×80、TTFT 余裕を見た初期値。P が飽和しない場合は 192 へ）。
- `decode_max_total_tokens`: 524288 のまま。他は変更しない。

### 2.3 `platforms/firmus/framework-comparison/sglang.sbatch`

1. ヘッダ: `#SBATCH -J sglang-firmus-p8dpep-d2`、コメントとログの `topology=P8DPEP-D2TP4`。
2. `worker-endpoints.tsv` を 3 行にする: `prefill-0 HEAD 30100 8`、`decode-0 TAIL 30200 4`、`decode-1 TAIL 30201 4`。
3. `worker.sh`:
   - `SGLANG_PREFILL_FUSED_FP8_SHUFFLE` / `SGLANG_PREFILL_TILED_ROUTING` の export を**削除**（TP＋CUTLASS 経路用の自作パッチ。DeepEP 経路では未検証なので外す）。
   - `exec` 行から `--ep-size 1` を外し、共通部分は `--tp-size "$tp" --mem-fraction-static 0.90 --max-total-tokens 131072 --chunked-prefill-size 8192 --page-size 64 --kv-cache-dtype fp8_e4m3 --attention-backend fa3 --nccl-port "$((port + 1000))" --disaggregation-mode "$role" --disaggregation-transfer-backend nixl "$@" --host 0.0.0.0 --port "$port"` のままにする（後ろの `"$@"` で上書きできる。SGLang は同一フラグの最後の値を使う）。
4. srun 3 本（既存 7 本を置換）。1 コマンド 1 物理行。
   - prefill-0: `srun --exact -N1 -n1 -w "${HEAD}" --gres=gpu:8 --cpus-per-task=64 … bash "${ROOT}/source/worker.sh" prefill-0 30100 "${MODEL}" 8 "${HEAD_IP}" "${speculation[@]}" --dp-size 8 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode normal --chunked-prefill-size 65536 --disaggregation-bootstrap-port 9000 --max-running-requests "${N}" >"${ART}/prefill-0.log" 2>&1 &`
     - `--enable-symm-mem` と `--moe-runner-backend cutlass` は付けない（DeepEP 経路は FP8 blockwise → DeepGEMM contiguous が auto 選択される想定。server-info の `moe_runner_backend` を記録する）。
   - decode-0: 現行の decode-0 行そのまま（port 30200、`--gres=gpu:4 --cpus-per-task=32`）。
   - decode-1: decode-0 行の複製で `decode-1 30201`（nccl-port は 31201 になる）。
   - `--cpus-per-task=64` が node の CPU 数を超える場合は `scontrol show node g1 | grep CPUTot` で確認し 8 GPU 分に収める。
5. `wait_http` は 3 本分（30100、30200、30201）。
6. router: `--prefill "http://${HEAD_IP}:30100" 9000 --decode "http://${TAIL_IP}:30200" --decode "http://${TAIL_IP}:30201"`。`--prefill-policy round_robin` は P が 1 本なので外してよい。
7. server-info の jq 検査を分岐で書き換える:
   - prefill: `.tp_size == 8 and .dp_size == 8 and .enable_dp_attention == true and .moe_a2a_backend == "deepep" and .ep_size == 8 and .chunked_prefill_size == 8192 and .max_total_tokens == 131072 and .max_running_requests == $config[0].concurrency and .enable_symm_mem == false`
   - decode: 現行どおり（`.tp_size == 4 and .ep_size == 1 and … .cuda_graph_bs_decode == $config[0].decode_cuda_graph_bs`）。
   - 共通: `.kv_cache_dtype == "fp8_e4m3" and .context_length == 8192 and .attention_backend == "fa3" and .page_size == 64 and .speculative_algorithm == "EAGLE3" …` は維持。

### 2.4 テスト `qwen-experiment-metrics/tests/test_firmus_reference.py`

`test_sglang_launch_matches_p6tp2_d1tp4_large_pool_conditions` を `…_p8dpep_d2tp4_…` に改名し、以下に合わせる。

- worker 数 3、GPU 合計 `8 + 4 + 4 = 16`（`--gres=gpu:8` の分岐を追加）。
- prefill: `--tp-size 8`、`--dp-size 8`、`--enable-dp-attention`、`--moe-a2a-backend deepep`、`--deepep-mode normal`、**最後の** `--chunked-prefill-size` が `65536`、`--ep-size` が含まれない、`--moe-runner-backend` と `--enable-symm-mem` が含まれない、cpus 64。
- decode: `--tp-size 4`、`--ep-size 1`、`--max-total-tokens` の最後の値が 524288、port 30200/30201、cpus 32。
- `shuffle`/`routing` の probe は `0/0` を期待（env export を削除したため）。
- 実行: `cd qwen-experiment-metrics && uv run pytest tests/test_firmus_reference.py -q`（ローカル、GPU 不要）。`ruff check` と `bash -n platforms/firmus/framework-comparison/sglang.sbatch` も通す。

### 2.5 投入と監視

- push → Actions `Firmus` job → Slurm（partition hpcai、g1/g4、17 分 wall）。Actions runner が Slurm 待ちで落ちた前例（c112 の 2 回）があるので、Actions が失敗しても **Slurm job は残し**、`sacct -j <id> -X -n -o State` を 120 秒間隔の shell ループで待つ。
- 終了後、`/scratch/home/ACCOUNT/qwen/runs/pd-<run>-<attempt>.<id>/` から `results/`、`logs/` を回収し `/srv/qwen-artifacts/<run>/<attempt>/` へ退避、回収後に run ディレクトリを削除。

### 2.6 合否と読み方

1. **起動**: prefill-0.log に `DeepEP MoE is enabled` と `chunked prefill size is adjusted from 65536 to 8192` が出て、`/health` が 600 秒以内に返ること。assert/例外はその 1 行目を本稿「結果記録」に転記（最初の致命的エラーのみ）。
2. **Prefill 持続速度**: DP attention では rank ごとに `[… DP<k> TP0] Prefill batch, … #new-token: …` が出る。全 rank の `#new-token` 合計を、`#queue-req ≥ 3` の飽和区間の秒数で割る（`/home/USER/qwen-artifacts/research-runtime/20261009-p3d1-decode-profile/` 隣接の `ana/prefill_rate.py` と同じ方法）。**GPU あたり ≥ 12k tok/s** なら成功（TP2 の 10.75k を上回る）。≥ 15k なら 10000 への道が開く（8 GPU×15k=120k → 6000、P12 相当で 180k）。
3. **E2E**: `results/score.json` の good tok/s、good 率、TTFT p90、ITL p90。D2×TP4 なので decode 側上限は約 2×4400 相当。P が飽和していなければ（queue が常に 0）、concurrency を 192 に上げて 1 回だけ再投入。
4. **greedy parity**: DeepEP/DeepGEMM は数値が変わる。主催者 accuracy check の対象になるので、既存の parity 手順（8 prompt×64 token）を同 image で 1 回取る。

## 3. 実験B（A が成功したら）: Decode 側の固定費半減

- **B1（簡単・先に）**: tail を D1×TP8（EP1、DeepEP なし）にする。`--gres=gpu:8 --cpus-per-task=64 … decode-0 30200 "${MODEL}" 8 …`、pool は `--max-total-tokens 1048576`、`decode_max_total_tokens` も同値。MoE 固定費 12→6 ms（H200）、running 1 件あたりの傾きも低下が見込める。ITL ≈ running ÷ gen throughput を running 別に集計し、傾き（現行 H200 0.12 ms/req）と切片を比較。
- **B2**: D1×world8 を DP8 attention＋`--moe-a2a-backend deepep --deepep-mode low_latency`。low_latency は NVSHMEM IBGDA を要する可能性がある（`NVSHMEM_IB_ENABLE_IBGDA=1`、peermem/gdrcopy）。起動失敗時は B1 で止める。
- 比較は 1 変数ずつ（A の P 構成を固定し、D だけ変える）。

## 4. ASPIRE（H100）への展開

- H100 80 GB では TP2 に重みが載らないが、**world8/EP8 なら MoE 重みは node 内 1 コピー（約 27 GiB/rank）**なので載る。A が Firmus で成立してから、`submission/run/submit.sbatch` に同じフラグで P1×world8（node0）＋ D 側（node1: D2×TP4 または D1×TP8）を 1 回だけ投入する。
- 1 run 約 420 SU。投入前後に `myprojects -p 59010014` で残高（直近 11505、In Doubt 1024、保護分 6144 は未使用）を確認し報告に含める。

## 5. 未回収の結果（先に片付ける）

- Firmus Slurm **22359**（Actions run 38037162346 attempt 2、d039410 = P6TP2/D1TP4 page64 **c112**）は PENDING（Priority）。終了後に `/scratch/home/ACCOUNT/qwen/runs/pd-38037162346-2.NnEE2NOSGB/` から回収・解析し、4429.9（c104）と比較して §14 の後に記録。c112 が劣れば `research/firmus-p6tp2-d1tp4` の head を 3519fc0 の構成に戻す（revert d039410）。
- 回収後、run ディレクトリを削除する。

## 結果記録

（実行者が追記）
