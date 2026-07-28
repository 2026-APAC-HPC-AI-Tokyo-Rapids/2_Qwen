# SGLang SIF ビルドの試行記録

この文書は、2026-07-12 から 2026-07-25 までの Codex 会話、GitHub Actions、Genkai 上の手動診断から、現在の設計判断に関係する結果を整理したものである。途中の構成で得た時間は現在の recipe と直接比較できない。run number、導入 package、runner、共有環境の状態が異なるため、性能値ではなく採否理由を残すことを目的とする。

## 1. `build → provision → run` に戻した理由

最初は公開済み SGLang image を使う `provision → run` の二段構成も検討した。しかし、このリポジトリの目的は SGLang source を変更して同じ Qwen benchmark で比較することである。Python、Rust extension、CUDA kernel、依存関係を変えた実験を扱うには、実行環境そのものを成果物として固定する build が必要になる。

そこで OpenFOAM 側と同じく、次の方針にした。

- source、recipe、patch、依存関係を変えたときだけ明示的に build する。
- runtime script や benchmark 条件だけを変えたときは既存成果物を再利用する。
- build の成否と実験の成否を分け、downstream が失敗しても完成済み SIF は再利用できるようにする。
- SGLang source の変更は repository 内の `patches/sglang` で管理する。

Singularity CE の cache は OCI blob の取得を省けるが、Docker BuildKit のように `%post` の各段階を再利用しない。同一 recipe の `--fakeroot` build を繰り返しても `%post` は再実行された。このため、途中 cache の複雑化ではなく、完成 SIF を第一親履歴から再利用する方式を採用した。

## 2. build directory の比較

### `/var/tmp`

Genkai login node の `/var/tmp` は fakeroot、xattr、package install を含む完全 build に成功した。2026-07-22 の手動試験では約34分45秒、SIF は約9.9 GiBだった。

一方、実体は node-local disk で、4 KiB direct random write の同時測定は約362 IOPS、平均 latency 88.3 msだった。Actions の別 run では `apt-get install` だけで約56分を使い、disk utilization 約93%、約118 write IOPSとなった。停止ではなく、dpkg が作る多数の小ファイルで random write が律速していた。

したがって `/var/tmp` は正しく動く fallback ではあるが、通常の build tree には遅すぎる。

### `/fast/pj26000080`

同じ診断で逐次 I/O は `/var/tmp` より大幅に速く、4 KiB random write も約6,819 IOPSだった。しかし Lustre 上では rootless container 用 xattr を保存できず、`apt` や `dpkg` が必要とする ownership 変更も失敗した。OCI 展開時の xattr warning 自体は継続可能だったが、その後の `chown: Operation not permitted` により package 設定が失敗した。

このため、共有 `/fast` は source archive や完成 SIF の保存先には適するが、fakeroot で package manager を動かす build tree には採用しない。

### `/dev/shm`

login node の `/dev/shm` は約504 GiBの tmpfs で、rootless container 用 xattr は持たないものの、fakeroot 内の `chown` と小型 SIF の build・実行に成功した。大量の小ファイルを扱う build tree では node-local disk より明確に適している。

比較を受け、現在は run 固有の build tree、Singularity cache、一時領域を `/dev/shm/qwen-sglang-build` に置く。完成物だけを `/fast` の一時名へ copy し、rename で公開する。

login node は接続ごとに変わり得るため、node-local directory を別 SSH 接続や別 jobへ引き継がない。Actions の一つの build job内で完結させる。

## 3. SIF packaging の試行

過去の一気通貫 `singularity build` では、依存導入より SquashFS 作成が支配的になった構成があった。追加 backend と debug symbol を含む試行では、sandbox build 約578秒に対し、SIF packaging が約23分、SIF は約9.7 GiBだった。

同じ rootfs を使って gzip、LZO、XZ と worker 数を比較した結果、次の傾向を確認した。

- gzip は worker 数を増やすと大きく短縮した。
- Genkai 標準の LZO は `lzo1x_999` で、全 worker 数において gzip より遅く、生成物も大きかった。
- XZ は約10%小さくなったが gzip より遅かった。
- 60 worker 付近から伸びが鈍り、120 worker が常に最速とは限らなかった。
- zstd は当時の Genkai kernel で SquashFS mount support がなく、通常の SIF 実行には採用できなかった。

gzip level 1 と新しい `sqfstar` を使う案も実環境で検証し、tar から SquashFS を直接作ることで展開と大量 inode の cleanup を省けることを確認した。しかし、その後は PJM 計算ノードを packaging に使わず、login node 内で完結する現在の方式へ整理した。

現在残っている重要な判断は、遅い directory-to-SIF 再圧縮を避けることである。[build script](../../scripts/build-sglang-sif.sh) は fakeroot namespace 内で rootfs を tar stream にし、`mksquashfs -tar -numeric-owner` へ渡す。これにより UID/GIDを維持したまま SquashFS を一度だけ作り、definition、metadata、Primary System partition を `singularity sif add` で直接格納する。

現在の gzip level 2、32 processors、64 GiB memory limit は過去の全条件と同一ではない。変更する場合は、現在の buildを基準に SIF size、pack時間、実行可否を再測定する。

## 4. build 時間の内訳から分かったこと

home-lab runner を使っていた Actions run #78 は、現在とは異なる構成だが、時間の所在を調べる材料になった。

| 区間 | 時間 |
|---|---:|
| base image bootstrap | 1分31秒 |
| `apt-get install` | 5分02秒 |
| SGLang と Python dependency | 2分43秒 |
| FlashInfer cubin / JIT cache | 1分23秒 |
| DeepEP | 1分33秒 |
| MSCCL++ | 3分04秒 |
| kernel lock / download | 22秒 |
| SIF 作成 | 9分28秒 |
| inspect と Genkai への公開 | 2分11秒 |

この run の build と公開は合計28分19秒、SIF は9.88 GiBだった。現在は home-lab と SSH/SCP を使わないため数値を比較してはいけないが、network、package install、CUDA extension build、SquashFS を別々に観測する必要があることを示している。

実際、FlashInfer package index の応答だけで約9分44秒かかった run もあり、次の run では16秒に戻った。小さな recipe 変更でも、配信元と共有環境の変動により full build時間は大きく変わる。単一 run の wall timeだけで並列度や storage の効果を判断しない。

## 5. 失敗から残した実装

### host `/tmp` の quota

初期の build は、Singularity `%post` が host の quota 付き `/tmp` を使い、apt の署名検証用一時ファイルや `get-pip.py` を作れず失敗した。recipe 内の `TMPDIR` を sandbox 内へ固定し、workflow の build scratch は `/dev/shm` へ分離した。

### fakeroot sandbox の cleanup

fakeroot build 後の sandbox には subuid 所有の directory が残る。通常ユーザーの `chmod -R` と `rm -rf` では削除できず、build 自体が成功しても cleanup step が失敗した。

現在は独立した `busybox:1.36.1-musl` を fakeroot で起動し、run 固有の build directory を bindして削除する。生成途中の SIF に依存しないため、build のどの段階で失敗しても同じ cleanup を使える。

### FlashInfer の version と test

Actions #91 と #94 は、次の不整合により計算ノードで失敗した。

```text
flashinfer-python=0.6.15.post1
flashinfer-cubin=0.6.14
flashinfer-jit-cache=0.6.14+cu130
```

upstream SGLang `main` が更新された一方、recipe の companion artifact が固定値のままだったことが原因である。現在はインストールされた `flashinfer-python` の version を読み、cubin と JIT cache を同じ releaseへ揃える。

その後、login node の `%test` で `flashinfer` や `sgl_kernel` を importすると、書き込み可能 cache や `libcuda.so.1` を要求して失敗した。これは package 不整合ではなく、GPUのない login nodeで runtime pathを検査したことが誤りだった。

現在の `%test` は、通常の SGLang import、FlashInfer三 package の metadata上の version一致、backend packageの存在、MPI・UCXなど、GPU非依存の範囲に限定する。GPU依存の import、JIT、benchmark は計算ノードの責務とする。
