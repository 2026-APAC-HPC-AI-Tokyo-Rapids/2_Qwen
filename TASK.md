# Qwen最終提出物のブラッシュアップ

作成日: 2026-11-05 JST

## 目的と範囲

Qwen担当として、既存提出物を、審査員が再実行する構成・改善内容・根拠を確認しやすい形へ整える。最優先は最終提出要件の充足である。文章の美しさはGeminiで照合すると良いかも?
対象はBox提出フォルダの `qwen/` と、ルートの `00 Read me first.md` のQwen関連項目。OpenFOAMの記載を保持する。発表資料はQwenの主張とページ番号を照合する資料として扱い、必要な修正が見つかった場合に対象箇所を特定する。
任意の深さのサブチェックリストの作成を許可し、メモを残しながら作業することで引継ぎを容易にしてください。
少数回であればFirmusにのみジョブを投げて良い。Codex,ClaudeCode,OpenCode Go,Antigravityの任意のモデルの使用を許可します。クオータの残量を見つつ適切に使ってください。

## 参照先と最終提出要件

| 項目 | 参照先・内容 |
| --- | --- |
| 最終提出案内 | Pengzhi Zhu氏の2026-11-05のSlack投稿 |
| 更新期限 | 2026-11-06 12:00 Singapore / 13:00 JST。投稿には、間に合わない場合は無理をせず最終更新後にメールへ返信する旨もある |
| Box接続 | `ssh tomokazu-windows-desktop` |
| Box提出先 | `C:\Users\tomokazu\Box\Tokyo_Rapids_with_RIKEN_Submission` |
| リポジトリ側の提出物 | `submission/`。Boxではその内容が `qwen/` に配置されているため、対応関係を差分で確認する |
| 発表資料 | Box提出先の `presentation/Tokyo Rapids with RIKEN Presentation.pptx` |
| komoriyuta氏の会話履歴 | `/srv/apac-conversation-history/2026-11-05/` の `README.md`、`manifest.json`、`transcripts/` |

Slack指定の `00 Read me first.md` には、発表資料、Qwen Basic、発表したQwen実験ごとの説明、Highlightsを記載する。該当しないブロックは省略する。Highlightsは発表の再録ではなく、注目してほしい変更・スライドページ・実行証跡に絞る。

## 予備確認済みの事項

- [x] WindowsへSSH接続し、Box提出先と `qwen/`、`presentation/`、`00 Read me first.md` の存在を確認した。
- [x] 現在の `00 Read me first.md` を読み、OpenFOAMの項目はあるがQwenの項目がないことを確認した。
- [x] 会話履歴103件の本文ファイルが揃っていることを確認した。Codex42件、Codex補助54件、ChatGPT7件。
- [x] リポジトリの提出文書、関連research、直近コミット履歴、既存テストの構成を予備確認した。

以下は既存提出文書の記載値であり、最終稿ではraw出力・設定・実行ログと照合する。

| 結果 | good output tok/s | Actions / Slurm | 最終稿での扱い |
| --- | ---: | --- | --- |
| Basic c160 測定1 | 6321.12 | 41078642203 / 27682 | 再実行候補の具体的なrun |
| Basic c160 測定2 | 6339.31 | 41093781934 / 27709 | 再実行候補の具体的なrun |
| Basic 2回平均 | 6330.21 | 上記2件 | 平均値。単一runのJob IDと混同しない |
| Bonus round 34 発表値 | 10054.49 | 41767886175 / 28660 | 最高観測値。単発の結果として明示する |
| Bonus提出パッケージの単独実行 | 10038.11 | 41781374312 / 28673 | logprob relay overlayを含む実行。発表値と構成・スクリプト改訂時点を区別する |
| 主催者Common baseline | 1832 | Slurm 25013 | チームの初回full-length runやBasicとは区別する |

## 作業計画

### 1. 提出物と証拠を照合する

- [x] Boxの `qwen/` とリポジトリの `submission/` を比較し、内容・更新時点・欠落・Box固有の変更を確認する。
  - 共通330ファイルはSHA-256一致。Boxのみの27ファイルは正規のprofile成果物であり保持する。ローカルのみのPython bytecodeは提出対象外。
- [x] 発表済みのQwenスライドと実験を列挙し、スライドページ、設定、commit、image、Actions、Slurm、rawログを対応付ける。
- [x] 関連コード、research、コミット本文、過去のActionsログ、会話履歴から現在の構成になった理由を確認する。
- [x] 会話中の説明は調査の手掛かりとし、数値と条件は実際の設定・ログで確定する。古い構成の説明を現行結果へ流用しない。
- [x] Basicの再実行対象を具体的な1 runとして選び、反復平均は別記する。
- [x] チームのFirmus初回full-length runを特定する。初回完走、当時の判定失敗、後日の再解析が異なる場合は、その経緯と根拠を確認する。

### 2. `00 Read me first.md` のQwen部分を完成させる

- [x] Presented slidesにQwenのページ範囲を追記する。
- [x] BasicのPlatform、Framework、draft名、good output tok/s、good_request_fraction、Concurrency N、初回full-length性能、ボトルネック、Job ID、Script、Logを埋める。
- [x] 発表した実験ごとに、変更点、同じ計算である理由と証拠、Basic比、Job ID、Script、Logを記載する。
- [x] Qwen Highlightsを、変更の要点1文、スライドページ、jobまたはlogへ絞って追加する。
- [x] パスはBox提出フォルダから辿れる相対パスとし、既存のOpenFOAM本文を保持する。

### 3. Qwenの入口文書を整理する

- [x] `SUBMISSION.md` を全体案内、`basic/README.md` をBasic再現手順、`README.md` をBonus説明として、役割と相互リンクを整理する。
- [x] 冒頭で結果、構成、再実行する入口、証拠の場所を確認できるようにする。
- [x] 発表値10054.49と単独実行値10038.11、Basicの各runと平均、主催者baselineを一貫して区別する。
- [x] 重複説明、古い構成への参照、内部作業向けの記述、根拠を超える断定を修正する。
- [x] 既存の英語提出文書に合わせ、審査員向けの簡潔な英語に整える。

### 4. 改善の説明と正しさの根拠を補強する

- [x] 発表した各改善を、変更点、対象ボトルネック、測定結果、証拠の順で説明する。
- [x] 構成全体のBasic比と、条件を固定した個別変更の効果を区別する。複数要因が変わる比較から個別パッチの寄与を断定しない。
- [x] パッチ台帳のbit一致、浮動小数点演算順序、丸め位置、数値影響なしの分類と、その根拠を実装・検証記録に照合する。
- [x] 未検証の範囲を明記し、greedy probe、kernel一致検証、SLA適格性、主催者のfull-set accuracy checkを区別する。
- [x] profileは取得した構成と観測範囲を明記し、通常serving結果や別runのtimelineと混同しない。

### 5. 既存成果物で再現性を検証する

- [x] 提出物内のリンク、相対パス、必要ファイル、チェックサムを検証する。
- [x] 保存済みAIPerf出力を再解析し、score、good_request_fraction、error数などの記載値と照合する。ローカル検証器と公式判定の役割を区別する。
- [x] 変更したスクリプトの構文検査と、関連する既存のGPU非依存テストを実行する。
- [x] 提出フォルダ単体でbuild、prepare、scheduler job、公式benchmark、結果回収へ辿れるかを、既存E2E証拠と併せて確認する。
- [x] 既存証拠では解けず完了判断を変える問題があれば、不確実性と必要な検証を具体化する。文書整理のためだけにGPUジョブや新規buildを起動しない。

### 6. Boxへ反映して最終確認する

- [x] リポジトリが所有する変更は `main` 上で記録し、変更理由をコミット本文へ残す。文書のみなら `[skip ci]` を用いる。
- [x] 検証済みの変更をBoxへ反映する。反映直前に再度差分を確認し、他担当の更新を上書きしない。
- [x] rawログ・測定結果を保持し、説明文と整合させるためにraw出力を書き換えない。
- [x] Box提出フォルダを起点に読み直し、必須項目、数値、スライド番号、参照先、未記入項目、同期後の内容を確認する。
- [x] この作業の不要な一時ファイル・診断処理・ジョブが残っていないことを確認する。他の作業には触れない。
- [x] 最終変更点、実施した検証、未解決事項を報告する。案内メールへの返信は必要な提出後手続きとして報告し、明示的な送信指示なしには送信しない。

## 文章の再点検（2026-11-05）

- [x] 一般的な編集点検項目（結論までの前置き、重複注記、抽象語、長い列挙、不自然な英語、根拠を超える断定）で7文書を再読する。
- [x] 独立したCodexレビュアーの指摘を反映し、改稿後も意味・数値・再現情報・検証範囲を再確認する。
- [x] Gemini 4 Argon (High) にも読み取り専用レビューを依頼し、長文、曖昧な語句、注意書きの重複を修正する。数値の正しさはモデルの評価でなく既存rawとチェックに基づく。
- [x] 内部の編集履歴を短縮し、必須の検証限界はValidation scopeへ集約する。構成とファイル案内は比較しやすい表へ整理する。
- [x] 18実験の必須フィールド、提出値10054（正確値10054.49）、確認値10038.11、Basic値・平均、コマンド、143行の台帳本文、rawを保持する。
- [x] 既存テスト8件、チェックサム426件、相対リンク289件、rootの参照86件を確認する。
- [x] mainへ記録し、Boxに反映後、クラウド側のハッシュとOpenFOAM本文の保持を確認する。コミット `fbc5dcc` をpush済み。更新7文書のBoxクラウドSHA-1一致を確認した。

## 提出コードの整理状況の再確認（2026-11-05）

- [x] Basic/Bonusのprepare・worker・scheduler・AIPerf・scorer・draft/GPU配置helperとbuild entrypointを確認した。
- [x] `b5b4b97`・`0b07356`・`4eb4282` の変更理由を確認した。研究用collector、選択gate、旧構成分岐、入力再検査、失敗trap、AIPerf wrapperは11月4日までに削除済み。
- [x] 正常起動に必要なreadiness待機、Prefill/Decode分岐、未測定GPUの配置維持、AIPerfの省略可能error metricを保持する。組み込みのregistration設定は観測済みの起動問題への対策であり、測定時の条件を保つ。
- [x] 使われなくなったBasicの空profile directory作成だけを削除。既存fixtureは保存し、意図した省略を比較テストへ追記した。
- [x] `bash -n`、既存の実行経路比較4件と研究gate/profile/tuningを含めない検査2件が通過した。worker argv/env・HTTP呼び出し・bench・出力ファイルは同じ。
- [x] 測定imageのbuild/sourceはそのまま保持。Buildahの片付け、OCI検証、一時archiveからの確定処理を一律削除しない。build/GPU jobは投入していない。
- [x] `51b9427`をmainへpushし、Boxのprepare.shのクラウドSHA-1一致を確認した。

## Claude Codeによるコード整理の独立レビュー（2026-11-05）

- [x] ユーザー指示に従い、Claude CodeのOpus 5.5をタイムアウトなしで再実行し、exit 0 / completedまで待った。セッションID `454d0bba-9c81-4d88-89c1-c9367ee6e767`。レビューはコードを渡した読み取り専用の静的評価。
- 最終報告はClaude Codeの回答に基づく。先行したClaude Codeの300秒打ち切り、Antigravity Opusの240秒打ち切り、Antigravity Sonnetの回答とは区別する。ユーザーの追加指示以降Antigravityは使用していない。
- Claude Codeの結論: 大規模リファクタリングは不要。提出経路の研究用retry/fallbackはほぼ整理済み。ただし以下の小修正を推奨。
  - run/submit.sbatchのbench後grep・probe・server-infoで、score.json取得後でもjobがFAILEDとなりrun-successが作られない可能性がある。公式計測の成功と診断完了を混同しない。
  - Bonusのモデルパスは/scratchまたはRUN_DIR内でなければcontainerから見えない。usageで明示するかread-only mountを追加する余地がある。
  - 凍結済みbuild/sourceにrootでのBUILD_DIR削除、Buildah rm --all、private clone fallbackが残る。sourceは保存し、wrapperで空directory・mirror前提・専用storage範囲を確認する余地がある。
  - 任意: 実image hashとoverlay記録のconfig由来hash、draft download pinとhelper直書きpinの整合を明示する。
- 見送り可能: server-infoの前後別保存や軽微な整理。残す処理: readiness待機、未測定GPUの配置維持、AIPerfの省略error metric、性能設定、build入力・OCI検証。
- ここではレビュー結果を記録し、提出コード・Box・測定値への追加変更は行っていない。私の前回答の「ビルド側も含めそのままでよい」は粗く、上記の削除範囲は別途検討すべきだった。

## Claude Code Opus 5.5との改善イテレーション（2026-11-05）

- [x] ユーザー指示によりAntigravityは使用せず、Claude Codeの`claude-opus-5-5`と4回の読み取り専用レビューを往復した。セッション `8a790fe2-3794-454a-9774-d757f920b813`。各回はタイムアウトなしで正常終了まで待った。
- [x] 第1回: 計測完了直後のmarkerと診断失敗の明示警告、modelの明示read-only mount、BUILD_DIR新規作成と専用storage、image/metadata照合、draft provenanceの入力一致、server-infoの開始時1回取得を実装した。
- [x] 第2回: Basic image/sidecar/configのSHA-256・size・commit一致、probe全体の60秒timeoutと5秒kill猶予、残時間90秒判定、共有cacheと過去実行時間の説明を修正した。
- [x] 第3回: Slurm終了時刻不明時の正確なskip理由を追加。harnessは同梱Basic configを直接使い、metadataのsize/commit不一致も検証した。
- [x] 第4回: 実際のBasic同梱config/provenanceにprepare.shと同じjq式を測定hash/sizeで評価してtrueを確認。Bonusの実metadataも一致。Claude Codeの最終判定は「前提確認は閉じられます。未解決の指摘はありません」。
- [x] 既存+追加CPUテスト56件すべてpass、skipなし。ruff、全shell構文、diff検査、426チェックサム、289相対リンクが正常。
- [x] workerコード・launch argv/env、AIPerf command、公開pins、143台帳行、raw、frozen build/sourceは保持。変更はhost側準備・scheduler orchestration・記録の整合・文書に限定した。
- [x] GPU再実行・新規image buildは行っていない。検証範囲を文書に保持し、静的レビューとCPU fixtureをGPU実測と扱わない。
- [x] コミット `b4c450d` をmainへpushし、Boxの更新10ファイルのクラウドSHA-1一致を確認した。OpenFOAMとroot Readme、PPTXは今回変更していない。
- [x] この作業のreview/transfer一時ファイルを削除した。Claude Codeと各テストは正常終了し、test内の擬似serverはprocess groupで終了済み。build/GPU/scheduler jobは起動していない。

## 作業メモ（2026-11-05）

- ユーザー追加指示: 提出値は10054 good output tok/s（正確値10054.49）。10038.11で置き換えず、単独実行の確認値として区別する。
- 意思決定の背景は `ssh tomokazu@tomokazu-ubuntu-server` 先の `/home/tomokazu/.local/bin/discrawl` でDiscordを確認する。それでも確定しない点はユーザーへ質問する。

- Boxの現行版を読み取り取得し、Qwen・ルートReadme・PPTXを照合中。OpenFOAM本文をUTF-8で保持した。
- Slack最終提出案内を再読し、公式READMEとQwen Task Rulesがローカルtaskの版と一致することをGitHub APIで確認した。
- 発表資料は145枚。Qwenの本編は38–67、現行Q&Aは69–82。96以降はArchiveで旧構成を含む。
- スライド64の33 SGLang overlaysは発表runを示す。提出パッケージはlogprob relay追加後の34 overlays。スライド70/81は旧9.12kパッケージを参照しているため、提出文書に現行構成との区別を明記する。
- Actions 41781374312/41767886175/41093781934のsuccessと、Slurm COMPLETEDをログでも確認した。
- 初回full-lengthはActions 34725941535 / Slurm 17003の902.36。固定時間clientは完走したが旧length validatorでFAILED。Actions原文を保持し、最初のclean terminal成功909.68（34732532321 / 17021）のrawも追加した。
- Discord追跡で11月3日の運営回答の共有を発見: https://discord.com/channels/1471662672784261193/1471663521564721336/1567175772437872640 。Slack原文も再確認し、Basicは公開release、source変更はBonusという最終区分を採用した。10054.49をBonus提出値、10038.11を確認値として区別する。
- 発表18項目をEXPERIMENTS.mdへ対応付け、22件の過去runから必要なraw・設定・script・accountingだけを抽出した（合計約2.9MB）。原本manifestとのSHA-256一致を確認した。PBSの不足scriptは記録commitから取得し、その旨を明記した。
- 発表資料の誤読箇所を索引へ補足: slide42のvLLMはc32、slide58の9036は44/rank、slide63の最終値はprepack以外の変更も含む。PPTX自体は編集せず、ページ番号と当時の内容を保持する。
- パッチ台帳143項目の分類・有効範囲とsource/gateを照合し、gigatokenの有効状態と2WG epilogueのpurposeの古い説明を訂正。分類がfull-set accuracy認定ではないことを明記した。新しい数値的同等性の実測は行っていない。
- 既存CPU検証33件passed。文書・checksum変更後の関連4件passed。新規証拠を含むAIPerf再計算26件、checksum426件、相対MDリンク284件、ルートQwenパス104件、Boxのみのprofile27件の検証がpassed。現行entrypointのbash構文・git diff --checkもpassed。
- scripts・model・kernel・build入力は変更せず、新規GPU実行と新規buildは行わない。最終readability rewriteのGPU未実行とfull-set accuracy未実施を提出文書へ明記する。

## 完了記録

- 文章の再確認（ユーザー追加依頼）: commit `04a3aba` で英文の断片文・不自然な表現・注意書きの重複を整理し、本文の速度比を1.59倍へ丸めた。測定値・設定・raw出力は保持。checksum426件と相対リンク285件を検証し、変更した7ファイルのBoxクラウドSHA-1も全一致。

- mainのcommit `eee1dee` をpush済み。リモートmainと同一であり、追跡ファイルの未コミット差分なし。既存の未追跡ファイルには触れていない。
- Box反映直前に既存357 QwenファイルとルートReadmeが初期snapshotから変わっていないことを確認した。レビューした343 QwenファイルとルートReadmeだけを反映した。
- Windows側の最終694ファイルはSHA-256全一致、欠落・余分・差分なし。ルートReadmeを読み戻し、OpenFOAM本文を元のbyte列と比較して一致。発表PPTXも不変。既存profile27ファイルも保持した。
- Box MCPの同一提出フォルダ（424691279125）から125フォルダ分を取得し、クラウド上の694ファイルのSHA-1をローカル期待値と比較。欠落・余分・差分とも0。共有や権限は変更していない。
- 新規build・GPU job・Actions runは起動していない（commitは `[skip ci]`、そのcommitのActions一覧も空）。作業用snapshot・Discord取得物・診断helper・転送tarは削除して完了する。
- 残る制約: full-set accuracyは未実施、readability rewrite後のGPU再実行は未実施。提出文書に範囲を明記した。運営への更新完了メールは未送信であり、ユーザーの返信手続きが必要。

## 完了条件

- [x] `00 Read me first.md` がQwenの必須項目を満たし、既存のOpenFOAM記載が保持されている。
- [x] 再実行対象のBasicが具体的な設定・job・script・logで特定されている。
- [x] 発表したQwen実験について、変更内容、計算の同等性の根拠、Basic比、実行証跡が辿れる。
- [x] 発表資料、入口文書、設定、保存済み結果の間で、数値・条件・説明が整合している。
- [x] 必要な安価な検証が完了し、確認済み範囲と残る制約が明記されている。
- [x] Box上の最終内容を再確認し、Qwen提出物の改善結果を報告している。

---

## 過去のタスクと完了記録

以下は既存の記録を保持したもの。旧構成・当時の作業方針を含む。今回の作業範囲と未完了項目は上記の計画を参照する。

公式Slack（2026apachpcai-vtf5729.slack.com）、`/home/tomokazu/.local/bin/discrawl`、公式リポジトリ（`./task`）から得られる情報をもとに、このプロジェクトを更新してください。
TODOリストは進捗に応じて書き換え、チェック、追加、分割してください。サブタスクやサブサブタスクなど、任意の深さにタスクを切って自律的に作成してください。サブエージェントの利用も許可します。
このタスク全体についてcommit、push、CI、およびログインノードの操作を許可します。モデルは`MODEL_PATH.md`に記載されたものを使用してください。
`/home/tomokazu/actions-runner`のself-hosted runnerは、Actions仕様と各スパコンのログインノードへのアクセスを調べる際に参照できます。
意思決定に必要な場合は会話履歴を参照し、過去のCodexへの指示も判断材料にしてください。
日本語の文章を作成するときは、agyコマンド経由でGemini 3.8 Flash(High)を呼び出し、英語で入力して、技術文書として適切な日本語に訳してもらってください。
ジョブを長期間待つときは、lunaなどの軽量モデルを使ったり、「コマンド終了=CI終了」であるようなコマンドを作ってそれをタイムアウト無制限で呼び出したりして使うとよいでしょう。
また、大量の出力が予想されるコマンドは適宜リダイレクトしたりgrepしたりしてコンテキストを節約することを心がけてください。
全クラスタ(玄界,Firmus,ASPIRE 2A+)について、余裕があれば同時並行で作業することをお勧めします。なるべく共通のコードで動かせることが望ましいですが、無理に統一する必要はありません。


## 全体TODO

- [x] 公式Task Rules、Application Notes、Slack、公式リポジトリ、Codex履歴、現行コード・Git履歴・Actionsログを再照合し、現状との差異を整理する。
- [x] メトリクス収集、ベンチマーク取得、プロファイリングを公式タスク定義に合わせて実装する。
- [x] 最終構成までコードを整理し、YAGNI原則に従う正確で外科的、無駄がなくテスト可能で読みやすい状態にする。
  - [x] 旧Qwen3.5 dummy経路と不要な汎用処理を削除する。
  - [x] 通常benchとprofileを分離する。
  - [x] リポジトリのローカルバリデータとスタンドアロンスクリプト組み込みのバイト一致パーサーの判定を同一に維持する
  - [x] SGLang source revisionとAIPerf versionを固定し、build手順を明記する。
  - [x] 採用構成の確定後、探索専用設定と一時的な診断要素を棚卸しする。
- [x] 実ジョブを用いた1要因ずつの診断と信頼性向上を完了し、統制された性能比較は再監査タスク側で個別に追跡する

## 実行TODO

- [x] AIPerf v0.12.0とSGLang sourceを固定した再構築可能なコンテナを作る。
  - [x] SGLang commit `fc275674a97d78553b22597ee72c8fd48f5420f4`をbuild scriptで固定する。
  - [x] SquashFS、OCI、source、toolchainのprovenanceを保存する。
  - [x] repo全体をbuild inputとする具体的な再構築コマンドを`submission/README.md`へ記載する。
- [x] 最終Firmusトポロジは2ノード16GPU構成とし、NIXLを用いたTP2 prefillワーカー4台、TP4 EP4 decodeワーカー2台、ゲートウェイ1台とする
- [x] 公式AIPerf commandを変更不能な形で実行し、raw exportと全service logを成果物へ回収する。
  - [x] 公式argvを固定し、実行時のcommandとprovenanceを保存する。
  - [x] 探索/profile runでraw exportと全service logの回収経路を検証する。
  - [x] 公式60秒warmup、120秒benchmark、30秒graceの通常runを完走する。
- [x] `profile_export_aiperf.json`から公式goodput、good request fraction、scoreを検証するparserとテストを追加する。
  - [x] `good_request_fraction >= 0.90`、OSL 200、score=`goodput × 200`を検証する。
  - [x] 固定argvと来歴が一致する場合だけ、設定ISL 4000とAIPerfのprefix補正後client観測3986を厳密に判定する。
  - [x] 近似値、改変argv、来歴欠落・不一致を拒否し、失敗理由を`score.json`へ残す。
- [x] 通常benchとprofileを分離し、GPU・CPU・通信の観測結果を性能runと混同せず保存する。
  - [x] Firmus profile job 16928のGPU、CPU、NIC、InfiniBandデータを`submission/analysis/profiling_data/16928`へ保存する。
- [x] 旧Qwen3.5 dummy経路を整理し、提出用の単一Slurm scriptとbuild手順を整える。
- [x] 安価な静的検査、fixture、単体テストを完走させる。
  - [x] 138件のpytest、Ruff、bash構文、actionlint、YAML、diff検査を通す。
- [x] 玄界とASPIRE 2A+で2ノードのscheduler、Enroot、GPU、CUDA、NIXL、SGLang環境を検証する。
  - [x] 玄界 PJM 6769520、2ノード、終了コード0を確認する。
  - [x] ASPIRE 2A+ PBS 208754.pbs111、2ノード、`Exit_status=0`を確認する。
- [x] Firmusへ最終baseline jobを投入し、terminal状態、公式結果、成果物、残留jobなしを確認する。
- [x] concurrencyまたはPD構成を一因ずつ比較し、有効な改善を固定して公式120秒runを完走させる。
  - [x] HCA選択、NIXL待機時間、concurrency 8/16、single railを一因ずつ比較する。
  - [x] all-active HCA条件の再現run 16935を評価する。
  - [x] NIXL prewarm比較runを完了する。
    - [x] 1回の2-token throwaway requestを公式AIPerfの前に追加する。
    - [x] Actions 34701093440 / Slurm 16938を投入する。
    - [x] Slurm 16938がprewarmを開始後120秒で失敗し、AIPerf未到達・成果物回収済みであることを確認する。
    - [x] Actions 34709739292 / Slurm 16971: 要求ウォールタイム30分がFirmus hpcaiパーティションのMaxTime（17分）を超過し計算ノードが未割当だったことを確認、GPU実行時間0でキャンセルしアーティファクトを回収
    - [x] Slurmのウォールタイムを17分に戻し、外部prewarmクライアントのタイムアウトとSGLang内部のSGLANG_DISAGGREGATION_WAITING_TIMEOUTをともに600秒に設定
    - [x] コミット 34dd8fd にて Actions 34710922973 を投入
    - [x] Actions 34710922973 / Slurm 16975 を g1 および g4 で実行したが、単一 prewarm が GPU 使用率 0% のまま約 5 分 52 秒間進捗しなかったため frozen-job ルールに基づきキャンセル。AIPerf は未達、artifact 10304165265 を回収
    - [x] 既存の SquashFS に公式 NIXL LIBFABRIC プラグインおよび libfabric 共有ライブラリが含まれていることを確認
    - [x] コミット 600b96e にて NIXL 転送バックエンドのみを UCX から LIBFABRIC へ変更し、Actions 34713003837 を投入
    - [x] Actions 34713003837 / Slurm 16978 は AIPerf 到達前に 4分42秒で終了コード 1 により自然終了。全 NIXL エージェントが createBackend unsupported backend LIBFABRIC および NIXL_ERR_NOT_FOUND を報告、artifact qwen-firmus-pd-34713003837-1 を回収し残存ジョブなしを確認
      - [x] 安価な 1-GPU 診断 Slurm 16981: プラグイン一覧には LIBFABRIC と UCX が表示されるものの LIBFABRIC バックエンド生成は引き続き NIXL_ERR_NOT_FOUND で失敗、UCX は正常にインスタンス化され 8 基の IB デバイスを検出。リソース待機中に強制終了されたジョブがないことを確認
    - [x] NIXL ソースコード精査により NIXL device_list で全 HCA を重複指定しても同等であることが判明したため、明確な一因子検証として UCX_MAX_RMA_RAILS=1 を選定
    - [x] コミット 77d7ba0 / 最終 head 6c9d518、Actions 34715060358 / Slurm 16983: 2ノード g1/g4、経過時間 8:03、終了コード 28。PD 内部ウォームアップは約1秒で完了したものの、単一 prewarm が 0 バイト返却のまま 120 秒の curl タイムアウトに到達し、AIPerf およびスコアは未達。成果物（artifact ID 10304423647）を回収し残存ジョブなしを確認。本設定は不採用
      - [x] 誤ってトリガーされた Actions 34715024132 は Firmus Slurm ジョブ投入前にキャンセルされ、計算ノードを使用しなかったことを記録
    - [x] コミット 68d6c07 / Actions 34715741449、Slurm 16985: g1/g4、経過時間 7:35、FAILED（終了コード 28）。NIXL UCX および内部 PD ウォームアップは完了したものの、prewarm レスポンスが 0 バイトのまま 120 秒の curl タイムアウトが発火し、AIPerf およびスコアは未達。成果物（artifact ID 10304664445）を回収し残存ジョブなしを確認。num_threads=1 は不採用
    - [x] コミット f4c6d03 / Actions 34716724019、Slurm 16987: g1/g4、経過時間 7:41、FAILED（終了コード 28）。NIXL UCX 初期化および内部 PD ウォームアップは完了したものの、prewarm レスポンスが 0 バイトのまま 120 秒の curl タイムアウトに到達し、AIPerf およびスコアは未達。成果物（artifact ID 10305465805）を回収し残存ジョブなしを確認。num_threads=0 は不採用
      - [x] 診断ジョブ Slurm 16988・16990: 転送コード実行前のパス解決バグにより各1秒で終了したセットアップ失敗（残存ジョブなし、計算資源消費は極小）として記録。パス解決バグは 16991 で修正済み
      - [x] 診断ジョブ Slurm 16991: 固定ポート5555使用中による診断スクリプトのセットアップ失敗（転送前・23秒で終了）として記録。イニシエータ側はUCX生成および512 MiB登録まで到達、残存ジョブなしを確認
      - [x] Slurm 16992: 2ノード（g1/g4、各H200×1）にて23秒でCOMPLETED（exit 0）。NIXL UCXによる4096ディスクリプタ・512 MiB転送および完全検証を完了（転送 0.0123235秒 / 40.5727 GiB/s、メタデータ 0.29581秒、ターゲット検証 0.12340秒）、成功マーカーあり、残存ジョブなしを確認。Firmusにおけるノード間NIXL/UCX CUDAデータプレーンおよび大規模ディスクリプタリスト処理の正常稼働が実証されたため、16-GPU再試行を一時停止したまま調査をSGLang PDブートストラップ／制御統合へ移行
    - [x] ソースコードおよび成果物ログの解析によりポート・IP競合の可能性を棄却し最有力原因を特定: Prefill TP2からDecode TP4の構成では各Prefillランクに2つのDecodeメタデータ提供が必要だが、Run 16983/16985/16987はいずれもブートストラップ要求カウント1で停滞しバルク転送が開始されていなかった
    - [x] 次の一因子実験として、Node 1のDecode 2×TP4/EP4および全16 GPU構成を維持しつつ、Node 0をPrefill 4×TP2から2×TP4ワーカーへ変更する構成を策定
    - [x] Actions 34720852761: トポロジー来歴JSON（provenance JSON）が古かったため、バリデーション完了後・Firmus Slurm投入前に意図的にキャンセルし、Firmus GPU時間消費ゼロであることを確認
    - [x] 修正コミット c949519 および Actions 34720906809 の実験完了履歴
      - [x] 試行1・2はFirmus SSHの間欠的切断によりSlurm投入前に失敗しGPU時間を消費せず。試行3にてg1/g4上にSlurm 16994を投入（2ノード/16 GPU使用）、単一prewarmが0バイト返却のまま120秒のcurlタイムアウトに到達し、7分40秒・終了コード28で自然終了。NIXL UCXおよび内部PDウォームアップは完了したものの、AIPerfおよびスコアは未達。Node 0を2基のTP4 prefillワーカーへ統一してもストールは解消されず、異種TPのみを根本原因とする仮説は棄却
      - [x] リモート実行ディレクトリを保持した状態で、サービスログを /tmp/qwen-artifact-34720906809-attempt3 へ回収完了。squeue にて残存ジョブがないことを確認
      - [x] ログ解析により、選択された TP4 prefill/decode ペアが room 2519058693820158481 を共有し、選択された prefill は AbortReq までブートストラップ要求カウント 1 で停滞、非選択の prefill は 0 のまま推移したことを確認。ストールは実際の prefill / KV 転送前の段階で依然として発生していることを実証
    - [x] 次の一因子実験の策定: 同等な TP4/TP4 構成、router、HCA、decode 設定を維持しつつ、修正済み SGLang TP4/TP4 NIXL 機能テストと完全に一致させるため、prefill 側のみ SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS に num_threads=2 を設定
    - [x] コミット 24f774b / Actions 34722527667、Slurm 16997: 2ノード g1/g4、経過時間 7:25、終了コード 28 で自然終了。prewarm が 0 バイト返却のまま 120 秒でタイムアウト。NIXL/UCX、内部 PD ウォームアップ、ルーターへのモデル登録は成功したものの、AIPerf およびスコアは未達。成果物（artifact ID 10307031946）を /tmp/qwen-artifact-34722527667-16997-final へダウンロード完了、残存ジョブなしを確認。prefill 側のみ num_threads=2 とする設定はストールを解消しなかったため不採用
      - [x] 診断ジョブ Slurm 16999: コンテナ内に ip コマンドが存在しなかったことによる 26 秒のセットアップ失敗として記録
      - [x] 診断ジョブ Slurm 17000: g1/g4 上で 28 秒で完了。自動検出 IP と明示的 IP が同一（g1: 198.18.151.51、g4: 198.18.151.54）であることを確認し、両モードにおいて g4 から g1 へのランダムポート ZMQ 通信が正常に機能することを実証、残存ジョブなし。これにより、単に SGLANG_HOST_IP を削除することは有意な要因ではないと判定
    - [x] 次の一因子実験の策定: 同等な TP4/TP4 構成およびデフォルトの NIXL 設定を維持しつつ、SGLang デバッグログを有効化し、クレジット節約のため診断用 prewarm タイムアウトのみを 30 秒に短縮
    - [x] commit `4089da6` / Actions `34724960835` / Firmus Slurm `17002` の実行結果確認: ノード `g1` および `g4` で動作し、6分6秒で自然終了（exit code 28）。アーティファクト ID `10307547627` を `/tmp/qwen-artifact-34724960835-17002-final` に回収、残留ジョブなしを確認。診断 prewarm が 0 bytes 受信のまま 30 秒でタイムアウトしたため、AIPerf およびスコア算出には未到達。
      - [x] 全 4 基の decode rank が正常な TP4 prefill bootstrap エンドポイントを取得し、全 4 基の prefill rank が 64,054 バイトの multipart メタデータを受信完了。ただし、いずれの rank も KVArgs 登録成功をログ出力せず。
      - [x] ソースコードおよびログ解析による根本原因の特定: equal-TP での eager prepared-dlist 構築により、各 rank で 1,531,197 KV スロット × 188 K/V 領域 = 287,865,036 個の記述子が実体化。最終的な uint64 Nx3 配列は約 6.43 GiB、一時コピーは各 rank 最低約 12.86 GiB に達し、NIXL 転送前の CPU 側遅延の主因であることを解明。
    - [x] 公式 prepared 構成（4x prefill TP2 / 2x decode TP4）をベースとした次回単一因子実験の設計: 全アクティブ HCA および明示的 prewarm を維持しつつ、全ワーカーで `max-total-tokens=131072` を設定。全リクエストが単一ワーカーに集中した場合でも公式並行数 16 を十分収容可能とし、eager 記述子のカーディナリティを約 12 倍削減。
      - [x] commit 148c97c / Actions 34725941535: Firmus Slurm ジョブ 17003（g1/g4）実行完了。公式パーサーによる ISL 拒絶のみを理由に 11分20秒で FAILED（exit 1:0）終了。残留ジョブなし。プリウォームは 18秒で成功（443 バイト応答）。修正済み公式 AIPerf（60/120/30 フェーズ）はリクエスト数 555、エラー 0、good request fraction 1.0、goodput 4.511811222461859 req/s、暫定スコア 902.3622444923719 output token/s で完了。成果物 ID 10308033052 を `/tmp/qwen-artifact-34725941535-17003-final-v2` に回収済み。
        - [x] 生 ISL 統計値の解明: min 3986、max 3987、count 555、sum 2212231、average 3986.001801801802。3986 厳密一致パーサーが有効な分布を拒絶していたことを特定。
        - [x] パーサー修正と検証完了: パーサーおよび同期スタンドアロンパーサーにて、固定された `first_turn_prefix` の理論範囲（3984..3988）内の min/max/count/sum を保持・検証するよう改修。120 件のテストに加え Ruff/actionlint/shell/JSON/diff チェックをすべてパスし、回収した生ジョブ 17003 のリプレイでも公式に valid 判定となることを確認。
        - [x] 最終Firmus E2E実行（commit b856810、Actions 34726983099）
          - [x] g1/g4上のSlurm 17005は12分37秒でFAILED 1:0終了（バリデータが平均OSLの厳密な200一致を要求したことのみが原因）
          - [x] prewarmは74秒で成功
          - [x] 公式固定設定（60/120/30）AIPerf完了: 558リクエスト、エラー0、good fraction 1.0、goodput 4.536230620411887 req/s、暫定スコア 907.2461240823774 output tokens/s
          - [x] 回収集計OSL: min 200、max 201、count 558、sum 111601、avg 200.00179211469535、osl_mismatch_count 0
          - [x] アーティファクト10309090452を `/tmp/qwen-artifact-34726983099-17005-final` に回収、ランタイムクリーンアップ済み、残留Slurmジョブなし
        - [x] OSLバリデーションを固定AIPerf 0.12.0の許容値へ修正し、17005のリプレイで公式valid判定とスコア907.2461240823774を確認
        - [x] Actions 34728353522 / Firmus Slurm 17007: g1/g4、7分50秒でFAILED 28:0。全16ランクでCUDAグラフおよび内部PDウォームアップ完了（OOM/UCX/NIXLエラーなし）。prefill-3からdecode-0への外部事前ウォームアップがbootstrap-req 1のままクライアントの120秒タイムアウトに達しAIPerf未達。アーティファクト10308936949を `/tmp/qwen-artifact-34728353522-17007-final` に回収、ランタイムはクリーンアップ済みで残存ジョブなし。
          - [x] 成功した17005では事前ウォームアップ前のprefillワーカーが初回の実入力までに約125秒要していたため、120秒設定では健全なbounded-KVブートストラップを誤拒絶してしまう可能性があることを確認。
        - [x] bounded prewarmのタイムアウトのみを180秒に引き上げ、1リクエストおよび固定された公式AIPerfコマンドを維持して最終Firmus E2Eを実行し、結果・成果物・クリーンアップを検証する。
          - [x] Actions 34729264020 / Slurm 17011は、内部ウォームアップ完了後も外部prewarmが180秒超にわたり応答0バイト、ログ進行停止、GPU使用率0%であったためフリーズと判定し、自ジョブのみを6分44秒で停止した。SlurmはCANCELLED by 3014、artifact 10308133762を回収、runtime cleanup済み、残留ジョブなし。有限なクレジットを守るため追加の盲目的再試行は行わない。
  - [x] all-active HCA選択および上限付きKV容量をmainへ統合、テストを経て追加の外部プリウォームを不採用
  - [x] concurrency 16の公式固定runを投入し、60/120/30秒の全区間を完走する。
  - [x] 公式raw export、score、good request fraction、ISL/OSL、error count、全service logを検証する。
- [x] README、research、submissionへ最終構成、再現方法、結果、制約を記録する。
- [x] 完了時に一時worktree、実験branch、不要な診断処理、重複・残留scheduler jobを棚卸しする。
  - [x] ユーザーや他作業のjobを除外し、この作業が作成した残留jobだけを確認する。

- [x] completion-audit ギャップの解消: Slurm 17011 で最終の 180秒外部プリウォーム設定が E2E 完了しなかった問題に対処
  - [x] 失敗しやすい余分なプリウォームを削除し、固定の公式 AIPerf 60秒ウォームアップを直接開始
  - [x] ローカル検証を実施後、Firmus 2-node/16-GPU 公式 E2E ジョブを1回実行し、ログと GPU アクティビティからフリーズが確認された場合のみキャンセル
  - [x] 終了状態、修正後の公式パーサー結果、artifacts、クリーンアップ、残留ジョブなしを確認し、main へ統合して実験 branch/worktree を削除


## 最終完了記録

- [x] **先行実行の対処**: GitHub Actions 34731113261 / Slurm 17019 で検出された再帰的 sbatch スプールパス障害の特定、ジョブキャンセル、およびクリーンアップ完了
- [x] **スクリプト配置**: コミット `909dae3` により共有ストレージ上へスタンドアロンスクリプトをステージング完了
- [x] **再実行の完了**: GitHub Actions 34732532321 / Slurm 17021 が g1/g4 上で正常終了（COMPLETED 0:0、実行時間 00:11:16）
- [x] **AIPerf 測定結果**: 固定 AIPerf（60/120/30、並行度 16）にて 558 リクエスト完了、エラー 0、Good Request Fraction 1.0、Goodput `4.548422736312085` requests/s、設定OSL `200` を乗じたローカル正規化スコア `909.6845472624169` output tokens/s、生のAIPerf出力トークンスループット `909.6763959671906` tokens/sを記録
- [x] **検証および成果物保管**: ローカルパーサー検証 true、フルアーティファクトを `/hdd_raid/samba/net_share/qwen-artifacts/34732532321/1` へアーカイブ、精選エビデンスを `submission/results/logs/17021` 配下に追加完了
- [x] **テスト・静的解析全件パス**: 138 件の pytest ケース、Ruff、bash 構文、actionlint、YAML、CloudFormation、パーサー等価性（parity）、エビデンスチェックサム、リプレイ検証をすべて通過
- [x] **後処理・統合完了**: 残留 Firmus ジョブなし、ドキュメント、Issue、プレゼンテーション、ブランチ統合、および一時 worktree のクリーンアップ完了

## 再監査TODO

- [x] 現行の公式GitHubルールおよびSlackのクロスチェック
- [x] スケジューラジョブの非存在確認後、フリーズしていたGitHub Actions 4件（34692013266、34692446176、34692531776、34692727165）のキャンセル
- [x] 重複したモジュラーFirmusランタイムおよび未使用JSON設定の削除
- [x] ドキュメントおよび発表資料における過大主張（overclaims）の修正
- [x] タスク所有の一時ファイルおよび未使用ブランチのクリーンアップ
- [x] 発表資料向けQwen性能の比較対照評価
  - [x] 同一の最終2ノードFirmusコードの適用
  - [x] 固定60/120/30 AIPerfワークロード・並列度8（Actions 34734863213、Slurm 17023）と並列度16（Slurm 17021）の比較
  - [x] 生エビデンスの保全
  - [x] 並列度以外の因果関係を主張しないことの徹底
- [x] 発表資料および技術記録の更新
- [x] 最終発表パッケージのメタデータ検証（16:9比率、全17スライド、両リポジトリコピー間のバイト完全一致、全スライドのレンダリングおよび目視確認完了）


## Basic / Bonus separation — 2026-10-30

- [x] Classify the source-modified stack and existing submission package as Bonus.
- [x] Select public SGLang 0.5.23 and pin its upstream commit and official amd64 image digest for Basic.
- [x] Build and verify an isolated, unmodified Basic image through Actions (41046995676).
- [x] Complete the Firmus Basic Actions → Slurm → official AIPerf → artifact collection path (41058071649: eligible 4867.53, full checksum archive).
- [x] Compare built-in settings in separately tracked commits, repeat the selected candidate and control, and report the measured Basic configuration (c160 selected; candidate/control each repeated; see the Basic experiment record).
- [x] Document the Firmus/official-workload Basic and Bonus references, their differing source/configuration boundaries, and both reproduction entrypoints; keep the frozen Bonus package separate from newer PP4 main results.
