# round 19: MegaMoE trace 読み出しの境界バグ

## 結論と証拠の限界

**parser の境界バグを source と CPU 再現で確定した。round 18 の実障害もこれが最有力だが、旧 run の allocation アドレスと raw がないため実機上の配置まで確定とは呼ばない。** kernel/host の計算や trace ABI を変更せず runner で直す。round 18 の image 41296967433 を build なしで再利用できる。今回 GPU、SSH、Actions 操作、scheduler 投入は行わない。

関連する AGENTS、round 18 note、909a2ea の本文と実装、393c776/7030826 の履歴、保存済み round 18 Codex final/exec log、今回渡された Actions 由来の worker log と rank fragment を確認した。source は host API → kernel → build → runner → 集計 → sbatch の順で調べた。DeepGEMM v0.1.5 のローカル読み取り専用 checkout は 1c8138a59b3df5dac50b76b730834ed57982fdfb。mirror を更新していない。

## host・kernel・parser の取り決め

- `sm90-prefill-diagnostic.patch`: TVM-FFI `fp8_mega_moe_diag` だけが `mega::fp8_mega_moe<true>` → `sm90_fp8_mega_moe<true>` を呼ぶ。通常は template の default false。診断では split、EP1、T7988 を assert し、通常 buffer サイズに private tail 用の 4096 + 132×12×16416×8 byte が付くことを assert する。
- host の `get_symm_buffer_size_for_sm90_mega_moe` は nullptr を起点に workspace、input、L1、L2、combine の layout を足し、combine の末尾を byte サイズとして返す。kernel は `sym_buffer.get_base_ptr()` を起点に同じ layout を足す。SF は BLOCK_M 128 の必要最小値ではなく、host が全 candidate の最大として確保した 1,474,560 token を template に渡す。kernel もその template 値を使う。SF capacity 差による別位置への書込みではない。
- `split-timeline.patch` は combine 末尾で全 thread に `r18::Trace` を構築する。`trace.cuh` はその絶対アドレスを 4096 に切り上げ、CTA/warp ごとに `(blockIdx.x×12 + threadIdx.x/32)×16416` word を加える。header は各 warp の lane 0 が書く。
- `build-prefill-megamoe-diag.py` は baked header に PP4 overlay を適用した隔離 tree で診断 header を生成し、通常 header を復元する。package に追加するのは診断 header と trace header のみ。Dockerfile は host extension の rebuild と通常 header build の後に診断生成を行う。`build-sglang-container.sh` は host series、trace、diagnostic patch、config の overlay 一式を同じ build context に配置する。
- runner は API のサイズに EXTRA_BYTES を追加し、control/diagnostic で同じ buffer と pointer list を渡す。host の末尾と kernel の末尾は一致する。旧 parser の読み出し開始位置だけが、kernel の page 揃えと一致しない。

GPU 不要の C++ 検査では、元の host と kernel の layout 構築部分を読み取り専用 source から抽出し、実際の `layout::SM90Workspace` / `layout::Buffer` で計算した。CUDA 13.4 の nvcc で host 実行用 binary を作成・実行し、両末尾の一致を exit 0 で確認した。これは CUDA kernel の build/実行や image build ではない。

| 項目 | byte または token |
|---|---:|
| workspace | 5,572,640 byte |
| pool capacity | 92,160 token |
| padded SF capacity | 1,474,560 token |
| host_size = kernel combine end の相対値 | 1,445,431,328 byte |
| host_size mod 4096 | 2,080 byte |
| 旧 read offset = floor((size−4096)/4096)×4096 | 1,445,425,152 byte |
| page 揃え base の trace offset = ceil(size/4096)×4096 | 1,445,433,344 byte |
| 旧 read offset から header まで | 8,192 byte = 1,024 word |

旧 `raw[:1024]` は word 0..1023 のみを調べる。page 揃え base では magic がちょうど word 1024 にあり、通常どおり書かれてもゼロ件となる。これは search 幅をわずかに広げる修正ではなく、`align_up(buffer.data_ptr() + size, 4096) − buffer.data_ptr()` を正確な読み出し offset とする修正にした。base 自体が非 page 揃えのケースも、256/2048/3840 byte の余りを含む fixture で検証する。

## 原因候補の判定

| 候補 | 判定と根拠 |
|---|---|
| search 窓の境界ずれ | source 上の欠陥は確定。旧配置と同サイズの CPU fixture は検索ゼロ件、新 offset で parse 成功。実 run の base は未記録なので、実障害の原因としては最有力。 |
| host と kernel の buffer / SF / combine layout 差 | 否定。同じ workspace/layout と capacity template。C++ 検査の両末尾も一致。 |
| 診断 API が通常 variant を呼ぶ | source 上は否定。true specialization が include と symbol を `_diag` に置換する。7988/EP1/split の assert も通らなければ旧 parse 地点へ到達しない。実 run の loaded symbol 自体は旧証拠にないため観測上の完全証明は不可。 |
| 通常 variant と JIT cache の衝突 | source 上は否定。build 名は同じでも compiler の cache signature は生成 CUDA text 全体を含み、置換後の include/symbol が違う。build 記録の template も capacity 8448、132 CTA、384 thread、split の想定と整合する。なお include hash の生成後に診断へ置換する構造なので、将来診断 header 自体を編集する際の hash 更新は別途注意が要る。今回 header は変更しない。 |
| CTA/WG の記録条件が成立しない | 否定。constructor は role 分岐より前に実行され、全 12 warp の lane0 が header を書く。math leader 条件は詳細イベントだけを絞る。 |
| 同期不足 | 否定。最後の診断呼出し後に device synchronize、続いて D2H copy。各 timing pair でも end event を同期済み。 |
| dtype / endianness | source 上は否定。kernel と Python の magic はともに 0x18314d4f45444941。int8 → int64 view → uint64 view は bit 再解釈であり数値変換ではない。little endian の先頭 byte は 414944454f4d3118。新 fragment に実 byte を残す。 |
| tail 不足 | 否定。208,027,648 byte の追加確保に対し trace 本体は 208,023,552 byte、page padding は最大 4095 byte。新 runner は実 offset + trace bytes が allocation 内に収まることも検査する。 |

## round 18 の JSON と時間経過

prefill-2 の 4 fragment は同じ build 記録、capacity 8448、T7988、fixture SHA、passed=false、cases が空、elapsed 12.93〜13.19 秒。`cases` の代入は trace parse より後だったので、空だから kernel や timing が走らなかったとは言えない。旧 source と traceback の到達位置から、最初の balanced について warmup、6 pair の時間測定、byte 一致と finite の assert、最後の診断 call と同期までは通っていたと分かる。実測時間そのものや raw は回復できない。

image build は CUDA 13.0.88。通常前後の SASS encoding SHA はともに `81d9ee08780a8ac639bbecc179bcadfd1df2837986f9450d103b8a0b500973dc`、155,648 byte。診断は 350,080 byte、spill 0/0、C75xx なしで、以前のローカル CUDA 13.4 の spill 28/44 とは異なる。build JSON の gpu_validated=false は build-only の記録であり、実 run の API 呼出し有無を表すフラグではない。

run_id の起点は 19:28:29。prefill-2 の torchrun 起動表示は 19:28:42、parse 失敗と rank 内の elastic SIGTERM は19:28:58〜59。prefill-0/1 の torchrun 起動表示は19:28:58で、全 srun の Killed は19:29:14、起点から45秒。**prefill-2 が落ちた瞬間に他 worker を止めた証拠はなく、共通45秒期限と整合する。** shell は以前から全 PID を `wait ... || true` で待つ。他 worker に kill を伝える分岐はない。一方 torchrun は同一 worker 内の rank 失敗で peer を SIGTERM することが log に明記されているため、runner の例外を fragment に記録して正常 return するようにした。

## 修正と次 run に残る記録

1. 実 allocation pointer と host API の size から kernel と同じ絶対 page 揃えを計算し、その位置から trace 本体を読む。header を探す推測窓は廃止する。
2. 各 case で `trace_buffer.head_hex` に trace 先頭4096 byteを byte 順そのまま保存する。dtype、little endian magic、先頭1024 word内の magic 位置、buffer pointer/byte数、combine end、trace pointer/offset/byte数、旧 offset を残す。dump は parse 前に fragment へ保存する。
3. host 呼出しの実 tensor pointer/shape/stride/dtype/byte数、buffer pointer list、rank、T、capacity、experts/topk、recipe、activation、fast_math、weights/SF と activation view、stream、固定 grid/block、build template、関係する既存 Prefill 環境を記録する。trace pointer は host ABI に独立引数として存在せず、実 pointer と size から kernel と同じ式で算出した値であることを区別する。
4. parse 失敗時にはエラーを保存した後、既存 JIT cache の通常版・診断版 `kernel.cu` と template 一致フラグ、CUBIN path、build template が一致する CUBIN の `cuobjdump -symbols` の結果または取得エラーを記録する。cache は読むだけ。kernel_name は explicit API source に基づく名前と明示し、生成 source / compiled symbol の証拠と区別する。function pointer の直接観測や launch hook を追加したとは主張しない。symbol 取得は失敗時のみ1呼出し1秒上限、retry はない。
5. balanced / historical-routing / edges の時間比較と pair ごとの byte/finite を trace より先に保存する。parse 失敗は case 内に記録して次の routing へ進む。byte 不一致でも false を残し、診断成功へ格上げしない。rank 例外では passed=false と traceback を保存し、他 EP1 rank を巻き添え停止させない。import 前にも最小 fragment を作る。
6. 集計は失敗 rank/case の timing、byte/finite、samples とエラーを残し、他の正常 case/rank も解析する。欠損や parse 失敗があれば complete=false、adoption=null、当該 case の fine attribution の採用可否は false。
7. shell の独立 wait を維持し、その意図をコメントに明記した。1 worker が即失敗して他2 workerが遅れて完了する shell fixture を追加した。起動込み45秒、2秒 kill猶予、3秒集約、既存 prepack 180秒 envelope 共有の制約は変更しない。起動待ちで枠を消費すれば未完了となる点は残る。P の gate、env、selection JSON、serving の計算経路を変更しない。

## build と Claude の serving commit

**新規 image build は不要。** product の変更は runtime runner/集計と sbatch のコメントだけで、host/kernel/trace header/build script/Dockerfile は変更しない。provision は checkout の runner/集計を runtime scripts/source へコピーするので、image を焼き直す必要がない。

Claude は serving commit で `configs/firmus-prefill-pp4.json` の `images.sglang` を run 41296967433 の実 artifact に揃え、既存 provisioning が必要とする対応する container provenance も一貫させる。必要な digest はこの作業の証拠にないため推測しない。`prefill.megamoe_diag=true` は維持し、P と MegaMoE の serving pin、FA3 pair/prepack gate は変えない。今回 branch の image pin は変更していない。runtime-only の再利用試験なので require-build は付けない。

## 検証と片付け

- 指定構成の全体 suite: **1440 passed、78 skipped、25 subtests passed、1518件収集、exit 0**。
- 最終 source の関連 suite: **99 passed、1 skipped**。page 境界の16組、旧 word 1024 の再現、magic/warp/overflow/truncated の各失敗、dump/host 情報の先行保存、全3 case の時間と byte 比較、rank 正常 return、worker 独立 wait、JIT source/symbol、wiring、selection boolean の全8組を検証。
- GPU 不要の native layout 検査: host_size=kernel_end=1445431328、exit 0。変更 Python の Ruff、sbatch の bash 構文、git diff の whitespace 検査も成功。
- 全体 suite は round 18 と同じ短い directory-fd alias を必要な3つの AF_UNIX fixture にだけ適用する scratch plugin で実行した。product/test source の動作を短縮していない。suite 中に行った補足記録と JIT 証拠収集の最終変更は、関連 suite でも再確認した。
- round 18 の元 JSON を新集計で読み取り確認し、complete=false のまま、利用可能な4 rank の fragment が失敗の情報を持ったまま残ることを確認。旧 raw/時間の欠損を補ったとは扱わない。
- 最終検証 log、counts、native layout JSON を指定 scratch に保持し、pytest fixture tree、native 中間 source/binary、scratch plugin と cache、commit message の一時ファイルを片付ける。round 18 の証拠は保持する。H200、実 image ABI、3 routing×12 GPU の trace/byte/time、45秒内完了、通常 serving の公式結果は今回の作業では未実行。CPU fixture を実機 E2E 成功とは扱わない。
