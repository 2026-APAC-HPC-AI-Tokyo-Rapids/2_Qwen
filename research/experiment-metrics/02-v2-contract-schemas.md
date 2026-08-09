# Qwen/SGLang v2 contract schema handoff

## 役割

この文書は[概要計画](./00-v2-plan.md)と[観測マップ](./01-v2-observation-map.md)を、
`qwen-v2.toml`とJSON Schemaへ転記するためのfield-level正本である。値のsource、field名、型、順序、
reducerを実装者が追加解釈しない。raw valueを別名で複製せず、source artifact、`run/key/value` fact、
query-time consumerの関係だけを定義する。

本projectのfailure modelは単純にする。取得できなかった値とrun variantで非適用の値はconsumerで`NULL`とし、
field別reason/status/欠損用placeholderを作らない。run-level statusは`collection_complete`だけである。

以下のclosed objectは、特記しないfieldがすべてrequired、field名末尾の`?`はoptionalかつ非該当時absent、
値型の`|null`だけがnullableである。optionalとnullableを同じ記号で表さない。
`additionalProperties=false`である。integerは64-bit範囲、`uint`はnonnegative、decimalはfinite、hashはlowercase
64-hex SHA-256、arrayは記載順を意味として保持する。

## canonical identity

| 項目 | v1定義 |
|---|---|
| repository ID | fixed string `apac2026-qwen`。runtime Git remote URLはidentityに使わない |
| regular run ID | `r-` + OS CSPRNG 16 byteのlowercase hex 32文字 |
| allocation ID | `a-` + OS CSPRNG 16 byteのlowercase hex 32文字 |
| qualification child ID | `q-` + OS CSPRNG 16 byteのlowercase hex 32文字。allocation/armをIDへencodeしない |
| JSON canonical bytes | RFC 8785 JCS、UTF-8。duplicate key、NaN、Infinity禁止 |
| committed TOML identity | checkoutしたfileの実byte列のSHA-256 |
| Git object identity | `{object_hash_algorithm:enum "sha1"|"sha256",oid}`。oidはalgorithmに応じexact lowercase 40-hexまたは64-hex |
| artifact member path | archive rootからのrelative POSIX path。absolute、`..`、空componentは禁止 |
| member reference | closed `{path:relative POSIX path,size_bytes:uint,sha256:sha256}` |
| environment snapshot | name byte順の`name\0value\0`反復。最後もNUL |

manifest自身のhashは本文へ入れず、参照側またはouter manifestに置く。

## source-track diff v2

qualification ONだけが`control/source-track-diff.json`へ次を一度記録する。

1. clean checkoutは`git status --porcelain=v2 -z`がemptyで確認する。
2. `git rev-parse --show-object-format`と`git rev-parse HEAD`を同じcheckoutで各一回実行し、algorithm/full OIDを得る。
3. ancestryは`git merge-base --is-ancestor <base> <head>`で確認する。
4. diff sourceは`git diff-tree --no-commit-id -r --raw -z --no-renames <base> <head>`である。
5. 各entryを`old_mode/new_mode/old_oid/new_oid/status/path_b64`へ変換し、Git pathの実byte順にsortする。
6. decoded path集合をscore bundleの`source_paths`とexact一致させる。protocolの`control_plane_files`はbase/headで
   同じblob OIDを持ちdiffへ現れないことを検査する。

recordは全field required、`additionalProperties=false`の
`{schema_version:const "source-track-diff/v2",repository_id:const "apac2026-qwen",
object_hash_algorithm:enum "sha1"|"sha256",base_oid:oid,head_oid:oid,
diff_algorithm:const "git-diff-tree-raw-z-no-renames+exact-source-path-set/v2",entries:entry[]}`である。
entryは`additionalProperties=false`の
`{old_mode:^[0-7]{6}$,new_mode:^[0-7]{6}$,old_oid:oid-or-zero,new_oid:oid-or-zero,
status:enum "A"|"D"|"M"|"T",path_b64:base64}`、decoded path byte順uniqueである。Aはold、Dはnewのmode/OIDを
algorithm長のall-zero、M/Tは両OID nonzeroにする。diff-derived hashと一致booleanはentriesから完全導出できるため
保存しない。bench/profile/qualification OFFはこのfile/schemaをexpected集合へ入れない。
branch名をidentityにしない。

`source-track-projection/v1`は全field required、`additionalProperties=false`の
`{schema_version:const "source-track-projection/v1",run_id,run_purpose:enum "bench"|"profile"|"qualification",qualification_arm:enum "off"|"on"|null,
repository_identity:const "apac2026-qwen",object_hash_algorithm:enum "sha1"|"sha256",
score_reference_commit:oid|null,profile_base_commit:oid|null,source_head_commit:oid,
source_track_manifest_sha256:sha256,source_member:relative POSIX path,source_sha256:sha256}`である。benchはscore reference、
profileはprofile base、qualificationはscore reference+armを持ち、非適用fieldだけnullにする。record hashはcanonical
source-track record、source SHAはそれを内包するrun manifest bytesのlineageである。

subject manifestにembedするrecord自身は全field required、`additionalProperties=false`の
`{schema_version:const "source-track/v1",object_hash_algorithm:enum "sha1"|"sha256",
base_role:enum "score_reference"|"profile_base",base_commit:algorithm-constrained full OID,
head_commit:algorithm-constrained full OID}`だけを持つ。bench/qualificationはscore reference、profileはprofile baseを選ぶ。
purpose、qualification arm、repository IDはsubject manifest top-levelが所有しrecordへ複写しない。
validatorはtop-level variantと同じrepository/object algorithmで、benchは`base_role=score_reference`かつbaseがheadのancestor、
profileは`base_role=profile_base`かつbaseがheadのancestor、qualification OFFは`base_role=score_reference`かつ
`head_commit=base_commit`、qualification ONは同roleかつbaseがheadのancestorを要求する。full OIDが存在しない、
別repository/algorithm、non-clean checkout、non-ancestorはjob投入前にrejectする。

## subject、contract、protocol

`control/run-manifest.json`、allocation parent、qualification child manifestは次を持つ。

```text
schema_version: enum "run-manifest/v1"|"allocation-manifest/v1", subject_kind, run_id?, allocation_id?,
parent_manifest_sha256?, run_purpose, qualification_arm?, pair_id?,
ordered_child_run_ids?, repository_id, source_track?, contract_id,
contract_sha256, protocol_id, protocol_sha256, dimension_policy_hash, lane_id,
observed_control_plane_sha256?,
instrumentation_decision_id?, instrumentation_decision_sha256?,
instrumentation_bundle_sha256?, planned_treatment?
```

`subject_kind`は`regular_run|qualification_allocation|qualification_child`、`run_purpose`は
`bench|profile|qualification`、armは`off|on`である。非該当fieldはnullでなくabsentにする。
`source_track=<source-track/v1>`はregular/qualification childでrequired、
allocation parentではabsentである。projectionはこのembedded recordをrun/child manifestから読む。
`observed_control_plane_sha256`はregular/allocation parentでrequired、qualification childではabsentで、childはallocation IDから親へjoinする。
`pair_id`はqualification allocation/childでrequired、paired bench protocolだけregularにも存在し、その他ではabsentである。
decision ID/hashはadopt済みscore benchだけrequiredでprofile/qualificationではabsentである。
`instrumentation_bundle_sha256`はregular/childでrequired、allocation parentではabsentである。
`planned_treatment`はregular/childでrequired、allocation parentではabsentのclosed objectである。
expected producer/file集合はmanifestへ複写せず、immutable contractのvariant applicability、protocol、bundle selectionから
finalizerが一回導出する。

```text
schema_version: const "planned-treatment/v1"
block_id?: nonempty string
block_order?: uint
replicate_index?: uint
planned_replicate_group_id?: nonempty string
planned_comparison_set_id?: nonempty string
dimensions: sorted unique [{name:nonempty string, logical_type:contract type ID, canonical_value:JCS value}, ...]
```

regularは`block_id/block_order/replicate_index`をrequired、qualification childは三fieldをabsentにする。childのblock IDは
allocation ID、replicate indexはprotocol pair array offset、arm orderはparent/set array offsetから導出する。
benchはgroup/set、qualification childはsetだけrequired、profileは両方absentにする。bench/profileのattempt/max/stoppingは
protocol dimensionだけが所有しmanifestへ複写しない。qualificationはprotocol pair arrayのoffsetがreplicate indexを所有する。
dimensionsはdimension policyで当該purposeにapplicableな
`must_match|treatment`をexactly once、name UTF-8 byte順に持ち、valueはlogical typeのcanonical JCS表現にする。
top-levelのrun purpose、arm、pair ID、contract/protocol/policy hashをtupleへ複写しない。bench replicate-groupは
`SHA-256(JCS({namespace,purpose,must_match,treatment}))`、bench/qualification comparison-setは
`SHA-256(JCS({namespace,purpose,must_match}))`とする。`namespace`はclosed
`{contract_id,contract_sha256,protocol_id,protocol_sha256,dimension_policy_hash}`、dimension arrayは
`planned_treatment.dimensions`と同じcanonical entryのname byte順である。armはqualificationのprotocol指定treatmentとして
treatment ID/paired contrastには使うが、arm/pair/block/order/replicateをcomparison-set hashへ入れない。
group/set IDはpre-run join anchorとして意図的に保存する。

manifestへ汎用`subject_id`は保存しない。projection/package時だけregular/childは`run_id`、allocationは
`allocation_id`から導出する。qualification package validatorは、(a) parent ordered child IDs = set/projection IDs、
job-entryが実在する時だけそのordered IDsも一致、(b) parent pair IDがprotocol `pairs`のexactly one entryを選び、
set/projectionのarray順armがその`arm_order`、各childのrun ID/armがarray entry/index、(c) child allocation/pair ID =
parent、child parent hash = archived parent-manifest member SHA-256、(d)両childのcomparison-set/must-match
dimensionsが一致し、差はprotocol指定treatmentとtop-level armだけ、(e) parentと両childの
contract ID/hash、protocol ID/hash、dimension policy hash、lane IDが一致する、(f) scheduler requestのmanifest SHA-256/
lane IDがarchived parent manifest SHA-256/lane IDと一致する、(g) OFF/ON childのbundle hashとarchived bundle bytesが
protocol `bundles.reference/score`のpath/hashへそれぞれ一致し、両bundleの`contract_id/contract_sha256`が
namespace contract ID/hashと一致する、
(h) OFF SIF sidecar `sif_build_definition` OID = OFF source-track head = protocol reference OID、
ON diffのalgorithm/base/head = ON source-track algorithm/base/head = protocol reference OID/ON SIF sidecar
`sif_build_definition` OID、を一回検査する。
ここでparent/set/protocol scheduleとscheduler requestの検査は常に行う。child manifest/bundle/source/SIFの検査は
`artifact_manifest`がnon-nullなchildだけ、両arm equalityは両childがmaterializeしたpairだけに行う。null childを持つsetも
parent evidenceとしてpublish可能だが、そのpairはeligibleにしない。

`control/protocol.toml`はGit管理された実byteをそのままcopyする。subject manifestの`protocol_sha256`はこのfileの
実byte SHA-256であり、protocol definition commitやlane fileだけのhashではない。全table/field required、
`additionalProperties=false`相当のclosed `protocol/v1`は次である。`qualification_target`だけpurpose依存である。

```text
schema_version: const "protocol/v1"
protocol_id: nonempty string
contract_id: nonempty string
contract_sha256: sha256
dimension_policy_hash: sha256
run_purpose: enum "bench"|"profile"|"qualification"
repository: {repository_id:const "apac2026-qwen",object_hash_algorithm:enum "sha1"|"sha256"}
control_plane_files:nonempty sorted unique [{path:repository-relative POSIX path,
  blob_oid:repository algorithm-constrained full OID}]
lane: {lane_id:nonempty string,source:{repository_path:repository-relative POSIX path,sha256:sha256},
       resolved:contract lane-schema object}
workload: {workload_id:nonempty string,expected_workload_identity:sha256}
sampling?: {planned_attempts:positive uint,max_attempts:positive uint,stopping_rule_id:nonempty string}
dimension_policy: {entries:sorted unique [{name:nonempty string,
  class:enum "must_match"|"treatment"|"nuisance"|"outcome",logical_type:contract type ID}]}
planned_value_policy: {
  must_match:sorted unique [{name:nonempty string,canonical_value:JCS value}],
  treatments:nonempty [{treatment_id:nonempty unique string,
                        values:sorted unique [{name:nonempty string,canonical_value:JCS value}]}]
}
qualification_target?: {
  reference_source_oid:repository algorithm-constrained full OID,
  bundles:{reference:{path:repository-relative POSIX path,sha256:sha256},
           score:{path:repository-relative POSIX path,sha256:sha256}},
  pairs:exactly 5 [{pair_id:unique nonempty string,
                    arm_order:exact ["off","on"]|["on","off"]}]
}
```

qualificationではtarget requiredかつsampling absent、bench/profileではtarget absentかつsampling requiredである。
pair replicate indexはarray offsetから導出し、初版arm orderは
offset 0からOFF→ON / ON→OFFを交互にする。top-level `control_plane_files[].path`はscore bundle `source_paths`と非重複で、
各blob OIDはreference/final treeとparent実行checkoutでexact一致する。protocol/bundle metadata pathはscore `source_paths`へ
含めてよいが、runner/PJM/control executable pathはcontrol-plane filesにだけ置く。別resolved lane fileは作らない。
`dimension_policy_hash=SHA-256(JCS(dimension_policy))`である。planned value policyのmust-match namesはpolicyの同classを、
各treatment valuesはpolicyのtreatment namesをexactly once持ち、logical typeはpolicyから取る。subject
`planned_treatment.dimensions`はmust-match valuesと選択したexactly one treatment tupleだけを合成する。
nuisance/outcomeはpre-run valueがないためplanned value policyへ入れない。qualification treatment IDは`off|on`に固定し、
protocol pair scheduleのarmから一意に選ぶ。
`expected_workload_identity=SHA-256(JCS({warmup:<ordered-workload warmup source SHA-256>,
measured:<ordered-workload measured source SHA-256>}))`である。regularは自身、qualificationはallocation parentが所有する
二manifestの実file SHA-256を同じfield順で合成し、planned identityへ照合する。二hashをFactへ別名で複写しない。
`observed_control_plane_sha256=SHA-256(JCS([{path,blob_oid},...]))`はregularの実行checkout、qualification parentの
実行checkoutからprotocol path順に一度観測する。protocol arrayの同hashとの一致だけをeligibilityへ使い、actual arrayを
別artifact/Factsへ複写しない。
初版laneは2 PJM nodes、2 MPI node processes、4 GPU/node、TP=8、rank 0 hostname advertise、SGLang port
25000、node-agent port 25001、rendezvous 60秒、ready 120秒である。別resolved-lane file、port allocator、
fallback discoveryを作らない。

`control/instrumentation-bundle.json`は全field required、`additionalProperties=false`の次のclosed objectである。

```text
schema_version: const "instrumentation-bundle/v1"
bundle_track: enum "reference" | "score" | "profile"
contract_id: nonempty string
contract_sha256: sha256
source_paths: sorted unique repository-relative POSIX paths
components: component[]
environment_capture_scopes: sorted unique subset of ["job_entry", "host_exec_node", "sglang_launch_node", "worker_global_rank"]
sink_bindings: [{sink_id:nonempty string, source_id:existing source registry ID,
 buffering_mode:enum "line_buffered"|"stage_buffered"}, ...]
```

`component`は全field requiredの
`{component_id:nonempty string,sink_ids:sorted unique string[],activation:activation}`で、
component ID UTF-8 byte順uniqueとする。`activation`は`kind`でtagした
次の五つだけで、各union memberは全field required、`additionalProperties=false`である。

```text
{kind:"in_process_hook", source_commit_role:const "sif_build_definition",
 source_path:repository-relative POSIX path, symbol:nonempty string, event_schema_id:nonempty string}
{kind:"executable", artifact_relative_path:relative POSIX path, content_sha256:sha256,
 argv_template:nonempty argv-argument-template array}
{kind:"sampler", artifact_relative_path:relative POSIX path, content_sha256:sha256,
 argv_template:nonempty argv-argument-template array, sample_interval_sec:positive finite decimal}
{kind:"control", control_id:nonempty contract-registry ID, timeout_sec:positive finite decimal}
{kind:"postrun_parser", parser_id:nonempty string, source_schema_id:nonempty string}
```

template segmentはclosed union `{literal:nonempty string}`、
`{token:enum "runtime_root"|"artifact_root"|"model_path"|"cache_root"|"component_executable"|"node_rank"|
"global_rank"|"invocation_id"}`、`{token:const "sink_path",sink_id:nonempty string}`とする。argv argument
templateはnonempty segment arrayでadjacent literalを一つに結合する。sink pathはbinding `source_id`のregistry
`path_template`だけから得る。`artifact_root`は
source-track headのclean checkoutから作るread-only instrumentation payload root、`artifact_relative_path`はそのroot基準である。
runnerはactivate直前にrealpathがroot外へ出ないことと実fileのsize/content SHA-256をbundle宣言へ一致させ、
`component_executable`をその検査済みabsolute realpathだけへ展開する。executable/samplerで別root、PATH検索、
hash不一致fallbackを許さない。
runnerは`reference -> qualification_off`、`score -> bench|qualification_on`、`profile -> profile`の固定写像でbundleを一つ選び、
その全componentを一度activateする。component別applicabilityはこの写像の完全な複写になるため保存しない。
`sink_bindings`はsink ID byte順uniqueかつsource ID uniqueで、全component `sink_ids`のunionとbinding ID集合は
exact一致する。各binding sourceのproducer activationはそのsink IDを持つcomponent IDの`bundle_component`とexact一致する。
sink token/componentのIDは対応bindingに存在する。runnerはactivate前に当該
variant/laneのcontractから展開した全producer instance `(sink_id,node_rank,global_rank,stage_name,invocation_id)`でsinkをrenderし、
全pathがrelative-safeかつglobal uniqueなことを検査する。同一single-instance bindingだけは一pathでよい。
executable/samplerのargv第0 argumentはexactly一つの`component_executable` segment、同tokenはtemplate内exactly onceとし、
shellを介さずargv arrayを直接spawnする。hookの`source_path`はbundle `source_paths`に存在し、SIF sidecarの
同path `score_source_tree` entryとexact一致することを検査する。v1のinstrumentation sourceはbuild-definition repositoryだけに置き、
upstream `sglang_source`を変更する場合は別contractでそのrepositoryのdiff/treeを追加する。最終runtime byte identityはSIF semantic hashが所有し、
bundleへcommit OIDを複写しない。qualification ONと採用後benchはscore bundleの同一
実byte/SHA-256を使う。実argv/cwdはStageRunner Fact、sink path templateはsource registry、実byteを含むouter file identityは
artifact manifestがそれぞれ一度所有する。
post-runの第二activation判定、fallback bundle、曖昧な軽量labelは作らない。

reference bundleは`source_paths=[]`、`components=[]`、`environment_capture_scopes=[]`、`sink_bindings=[]`に固定する。
score/profile bundleは`source_paths`と`components`をnonemptyにする。共通の外側timer/log/neutral resultはcontract graphが
所有し、reference bundleの疑似componentにしない。
contractはclosed `source_path_policy={score_allowed:sorted unique repository-relative POSIX paths,
profile_only:sorted unique repository-relative POSIX paths}`を一度所有し、両集合をdisjointにする。score `source_paths`は
`score_allowed`のsubset、profileは両集合のunionのsubsetである。score allowlistはinstrumentation実装とそのbundle/protocol
metadataだけを列挙し、model/dataset/workload/lane/PJM/common runner、profile-only pathを含めない。

## SIF sidecar v1

`source/sif-sidecar.json`はbuild終了後に作る、全field required、`additionalProperties=false`のclosed objectである。

```text
schema_version: const "sif-sidecar/v1"
sif_artifact: {artifact_relative_path:relative POSIX path,size_bytes:uint,sha256:sha256}
source_commits: exactly 2 [{repository_id:enum "apac2026-qwen"|"apac2026-sglang",
  commit_role:enum "sif_build_definition"|"sglang_source",object_hash_algorithm:enum "sha1"|"sha256",
  oid:algorithm-constrained full OID}]
score_source_tree: sorted unique [{path:repository-relative POSIX path,state:const "absent"} |
  {path:repository-relative POSIX path,state:const "present",mode:^[0-7]{6}$,blob_oid:algorithm-constrained full OID}]
recipe: {path:relative POSIX path,blob_oid:algorithm-constrained full OID}
applied_patches: sorted unique [{path:relative POSIX path,blob_oid:algorithm-constrained full OID}]
base_oci: {reference:nonempty digest-pinned string,inspect_raw:member,
  manifest_digest:string,config_digest:string,layer_digests:string[]}
lock_manifest: {path:relative POSIX path,size_bytes:uint,sha256:sha256}
build_command: nonempty string[]
toolchain_commands: exactly 4 [{role:enum "singularity_version"|"python_version"|"cuda_version"|"compiler_version",
  argv:nonempty string[],stdout:member,stderr:member,outcome:process-outcome}]
embedded_manifest: {member_path:const "/opt/apac/manifest.json",size_bytes:uint,sha256:sha256,
  manifest_identity:sha256}
runtime_files: exactly 4 [{role:enum "python_executable"|"sglang_package_entry"|"cuda_runtime"|"nccl_runtime",
  realpath:container absolute path,size_bytes:uint,sha256:sha256}]
```

source commits、toolchain commands、runtime filesは各記載role順にeach exactly onceである。roleからrepository IDへの
対応も記載順に固定し、source OIDは各build checkoutの`rev-parse`から取る。recipe/patch blobのalgorithmは
`sif_build_definition` commitから継承する。

OCI identityはbuild前の固定command `skopeo inspect --raw docker://<digest-pinned-reference>`のraw JSONと
そのSHA-256から得る。roleを持たない曖昧なbinary hashは追加しない。SIF内部manifestとouter sidecarはSIF SHA-256を
自己参照させず、outer sidecarだけが完成byteをhashする。reuse時は実際にlaunchするSIF byteとsidecar hash、
embedded manifest/runtime file hashを照合する。`manifest_identity`はembedded fileから抽出した
commit/OCI/patch/lock closed tupleのJCS SHA-256で、build/reuse時にouter canonical tupleのSHA-256と一度照合する。
tuple自身はsidecarへ再掲しない。GitHub workflow run/attemptはoperational relationだけが所有する。

PublisherがBigQueryへ一度転記する`qwen-sif-provenance-projection/v1`は、全field required、
`additionalProperties=false`の次の一rowである。

```text
schema_version: const "qwen-sif-provenance-projection/v1"
sif_artifact_sha256: sha256
source_commits: exactly 2 sidecar source commit objects in declared role order
score_source_tree: sidecar score source entries in path-byte order
recipe: {path:relative POSIX path, blob_oid:algorithm-constrained full OID}
applied_patches: sorted [{path:relative POSIX path, blob_oid:algorithm-constrained full OID}, ...]
base_oci: {reference:string, manifest_digest:string, config_digest:string, layer_digests:[string, ...]}
lock_manifest: {path:relative POSIX path, sha256:sha256}
build_command: nonempty string array
toolchain_observations: exactly 4 [{role:enum "singularity_version"|"python_version"|"cuda_version"|"compiler_version",
  outcome:process-outcome,stdout_member:relative POSIX path,stdout_b64:base64,
  stderr_member:relative POSIX path,stderr_b64:base64}, ...] in declared role order
embedded_manifest_identity: sha256
runtime_files: exactly 4 [{role:enum "python_executable"|"sglang_package_entry"|"cuda_runtime"|"nccl_runtime",
  realpath:string, size_bytes:uint, sha256:sha256}, ...] in declared role order
source_member: relative POSIX path
source_sha256: sha256
```

recipe/patch blobのalgorithmは`source_commits[role="sif_build_definition"]`から継承する。workflow run/attemptは
operational relationだけが所有し、このscientific projectionへ複写しない。`source_sha256`はsidecar identityであり、
run Fact `artifact.sif_manifest_sha256`がこの一rowへのjoin anchorになる。
`score_source_tree`のpath集合はbuild時に選んだscore bundle `source_paths`とexact一致し、reference bundleでは`[]`である。
entry OIDのalgorithmは`sif_build_definition` commitから継承する。qualification ONではdiff entryのnew stateとexact一致させ、
A/M/Tはpresentのnew mode/blob、Dはabsentとする。

## runtime launcher validation v1

regularはrun bundle、qualificationはallocation parent bundleが`source/runtime-launchers.json`を一度所有する。closed sourceは
`{schema_version:const "runtime-launcher-validation/v1",
executables:exactly 2 [{role:enum "mpirun"|"singularity",requested_path:string,resolved_realpath:absolute POSIX path,
size_bytes:uint,sha256:sha256}],versions:exactly 2 [{role:enum "mpirun"|"singularity",
argv:exact [same-role resolved realpath,"--version"],outcome:process-outcome,stdout:member,stderr:member}]}`で、両arrayは記載role順、
各role exactly onceである。outer MPI stageとnode wrapperはこのresolved absolute pathを使い、PATH検索しない。

publisher projectionは全field requiredの
`{schema_version:const "runtime-launcher-projection/v1",subject_kind:enum "regular_run"|"qualification_allocation",
subject_id:string,executables:[{role,requested_path,resolved_realpath,size_bytes,sha256},...],
versions:[{role,argv,outcome,stdout_member:relative POSIX path,stdout_b64:base64,
stderr_member:relative POSIX path,stderr_b64:base64},...],source_member:relative POSIX path,source_sha256:sha256}`である。
`runtime-launcher-semantic/v1`はpath/member locatorを除いたordered
`{schema_version:const "runtime-launcher-semantic/v1",launchers:[{role,sha256,outcome,stdout_b64,stderr_b64},...]}`で、
consumer identityは`SHA-256(JCS(object))`とする。採否に使うeligible 3–5 pairのparent projectionで一値に一致したhashだけを
decision `observed_runtime_launcher_semantic_sha256`へ置き、adopted benchもexact一致させる。

## model、dataset、cache

### model manifest v1

`source/model-manifest.json`の`source`はclosed tagged unionである。

- `{kind="hub_revision", repository:string, revision:full immutable hex commit}`
- `{kind="prebuilt_manifest", shards:[{path:relative path,size_bytes:uint,sha256},...]}`。
  shardsはpath byte順で、必要なsemantic IDは`SHA-256(JCS(shards))`としてviewで導出し、同recordへ複写しない

run recordはsourceに加え、次のsmall metadata allowlistの実在memberだけをpath byte順で持つ。

```text
config.json
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
generation_config.json
chat_template.json または tokenizer_config内chat template source record
```

top-levelは`schema_version="model-manifest/v1"/source/small_metadata`、各metadata memberは
`role=configuration|index|tokenizer|tokenizer_config|special_tokens|generation_config|chat_template`、
`path/size_bytes/sha256`を持つ。indexが参照するshard集合はindex fileをparseしてprebuilt manifestへ
照合するが、run manifestへ再掲しない。runごとに約400 GBのshard本文を再hashしない。

### dataset manifest v1

`source/dataset-manifest.json`は`schema_version="dataset-manifest/v1"/repository:string/
full_revision:string/relative_path:relative path/size_bytes:uint/sha256:sha256`を持つ。
download commandはこれらから完全導出できるため保存しない。moving `main`やshared filesystem pathをidentityにしない。

### cache manifest v1

`source/cache/node-<node-rank>-input-manifest.json`は`schema_version="cache-manifest/v1"/run_id/node_rank:uint/
members`を持ち、cache rootからのrelative path byte順でmemberを並べる。
regular fileは`{type="file",path,size_bytes:uint,sha256}`、symlinkは
`{type="symlink",path,target_b64:base64}`を持つ。
mtime、uid、gid、absolute staging pathはidentityに含めない。Fact identityはsubject bindingを除いた
`{schema_version:"cache-members/v1",members:<sorted members>}`のJCS SHA-256とし、同じcache内容ならrunを跨いで一致する。
source file byte hashはartifact lineageだけに使う。semantic identityだけを
`cache.node.<node_rank>.input_manifest_sha256`へ索引化し、file count/total size/cold-warm labelをrawへ複製しない。

## environment、topology、runtime config

全`.snapshot`はname byte順の`name\0value\0`反復である。parserはname/valueを変更せず、BigQueryではnameを
単射escape、valueをRFC 4648 Base64にする。scopeは`job_entry|host_exec_node|sglang_launch_node|
worker_global_rank`で、full snapshotを保持する。common/deltaの第二raw表現は作らない。

minimal CPU hardwareはregularではnode wrapperが各nodeで一度、qualificationではparentがarm 1前に一度だけ
`LC_ALL=C lscpu --json`を実行する。`source/hardware/node-<node-rank>.json`は`hardware-node/v1`で、
`schema_version:const "hardware-node/v1"/node_rank:uint/hostname:string`、`commands={lscpu:{argv:["lscpu","--json"],exit_code:int,
stderr_member:relative POSIX path}}`、`lscpu_json:object|null`、closed
`platform_settings={kernel_release:string|null,
online_cpus:uint[]|null,isolated_cpus:uint[]|null,nohz_full_cpus:uint[]|null,
memory:{total_bytes:uint|null,hugepage_size_bytes:uint|null,hugepages_total:uint|null},
numa_balancing_enabled:bool|null,boost:{source_path:string|null,raw:string|null,enabled:bool|null},
cpufreq_policies:[{policy_id:uint,related_cpus:uint[],driver:string|null,governor:string|null,
min_hz:uint|null,max_hz:uint|null},...]|null}`を持つ。同じsource recordへlscpuからderived CPU identityを複写しない。
kernelは`/proc/sys/kernel/osrelease`、CPU setは
`/sys/devices/system/cpu/{online,isolated,nohz_full}`、memory/hugepageは`/proc/meminfo`を使いkBを1024倍する。
NUMA balancingは`/proc/sys/kernel/numa_balancing`、boostはcpufreq `boost`を優先し、なければintel_pstate
`no_turbo`を反転する。cpufreqはnumeric policy ID順にrelated CPUs/driver/governor/min/maxを読み、kHzを1000倍する。
CPU-list fileは存在して空なら`[]`、interface不存在/read不能ならnull、cpufreq rootが
存在してpolicy 0件なら`[]`、root不存在/列挙不能ならnullにする。BigQueryはnode-rank順に投影し、consensusはviewで導出する。
cpufreq `related_cpus`はASCII whitespace区切りuintをstrictly increasing unique arrayへ変換し、`nohz_full`だけは
whitespace trim後のliteral `(null)`も`[]`へcanonicalizeする。

`minimal-hardware-projection/v1`は全field requiredの
`{schema_version:const "minimal-hardware-projection/v1",subject_kind:enum "regular_run"|"qualification_allocation",subject_id:string,node_rank:uint,
hostname:string,cpu_identity:{architecture,model_name,vendor_id,cpu_family,model,stepping:string|null,
physical_cores:uint|null,numa_nodes:uint|null},platform_settings:<hardware-node/v1のclosed settings>,
source_member:relative POSIX path,source_sha256:sha256}`である。CPU identityはlscpu label
`Architecture/Model name/Vendor ID/CPU family/Model/Stepping/Socket(s)/Core(s) per socket/NUMA node(s)`を
`field`のASCII末尾colonをexactly oneつ除去して照合し、physical coresはsocket数×core/socketとする。regularとallocationは
同じrelationを使い、allocation projectionやchild Factへhardware objectを複写しない。

`source/topology/node-<node-rank>.json`は次のclosed source recordを持つ。

- `schema_version="node-topology/v1"/run_id/node_rank:uint`、outer MPI
  `global_rank:uint/local_rank:uint/size:uint/hostname:string`
- exact `nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version,persistence_mode,
  compute_mode,power.limit,clocks.max.sm,clocks.max.memory,mig.mode.current,ecc.mode.current
  --format=csv,noheader,nounits`
  のhost/container別`{argv:string[],exit_code:int,stdout_member,stdout_size_bytes:uint,stdout_sha256:sha256,
  stderr_member,stderr_size_bytes:uint,stderr_sha256:sha256}`
- host `nvidia-smi topo -m`の同じhash-sealed command record
- node別actual `node_wrapper_argv`、`singularity_argv`、`sglang_argv`のstring array、host CPU affinity uint array、NUMA mems string

GPU arrayはraw stdoutから`topology/v1` reducerが一度parseし、node-rank/GPU index順の
`{node_rank,index,name,uuid,pci_bus_id,memory_total_bytes,driver_version,persistence_mode,compute_mode,power_limit_w,
max_sm_clock_mhz,max_memory_clock_mhz,mig_mode,ecc_mode,scope=host|container}`としてBigQueryへ
投影する。source JSON内へparsed copyを持たず、node argvや`hardware.gpu.nodes`の第二raw factも作らない。
`node-launch-projection/v1`は`schema_version:const "node-launch-projection/v1"/run_id/node_rank/global_rank/local_rank/size/hostname/node_wrapper_argv/
singularity_argv/sglang_argv/cpu_affinity/numa_mems/source_member/source_sha256`を持つ。
`topology-projection/v1`は上記GPU recordに`schema_version:const "topology-projection/v1"/run_id/source_member/source_sha256`を加える。

BenchArgs/ServerArgs event payloadは`config-snapshot/v1`で、`config_type=BenchArgs|ServerArgs`、
`definition={source_path:repository-relative POSIX path,class_name:nonempty string,
source_blob_oid:algorithm-constrained full OID}`、field名byte順の
`fields=[{name,declared_type,value},...]`を持つ。`value`は実dataclass値のJCS valueで、全fieldを保存する。このclosed
meta-shapeによりSGLang commitごとのfield増減を失わず、手書きallowlistも作らない。wide Runs列はcontractが指定した
scalar leafをquery時に投影するだけである。event typeはそれぞれ`bench_args_effective`、
`server_args_effective`へ固定する。
blob OIDのalgorithmはSIF provenanceの`source_commits[commit_role="sglang_source"]`から継承し、そのcommitの
tree下で`source_path`が指すblobとexact一致させる。SGLang commitはbuild projectionだけが所有し、eventへ複写しない。

worker-ready envelopeは次のclosed fieldを持つ。

```text
schema_version:const "qwen-domain-event/v1", run_id, producer_id, node_rank, global_rank, sequence,
event_type="worker_ready", payload={
  tp_rank, gpu_id, gpu_uuid, cpu_affinity, numa_mems,
  prefill_attention_backend, decode_attention_backend,
  torch_distributed_backend, transport_backend
}
```

expected global rankは0..7、node rankは0..1である。値をServerArgsから推測しない。

## event、lifecycle、phase

全domain eventの共通envelopeは
`schema_version:const "qwen-domain-event/v1"/run_id/producer_id/node_rank/global_rank?/sequence/event_type/payload`である。
producer内sequenceは0から連続増加する。clock IDはLinux boot IDとclock名を組み合わせた
`<boot_id>:CLOCK_MONOTONIC`で、別nodeのmonotonic値を減算しない。

`control/lifecycle/rendezvous.json`は`schema_version:const "lifecycle-rendezvous/v1"/run_id/protocol_version/rank0_hostname/address/
port=25001/created_at`、summaryは`schema_version="lifecycle-summary/v1"/run_id/nodes`を持つ。node entryは
`node_rank:uint/ready:{sequence:uint,clock_id:string,received_monotonic_ns:uint,ack_monotonic_ns:uint}/
shutdown:{sequence:uint,invocation_id:string,clock_id:string,received_monotonic_ns:uint,ack_monotonic_ns:uint}|null`である。
node-agent summaryがshutdown受信の唯一のsource、wrapper lifecycleが
child exitの唯一のsourceである。固定port衝突/timeoutはrun失敗とし、自動再試行しない。
`lifecycle-summary-projection/v1`はこのclosed summaryへ
`source_member:relative POSIX path/source_sha256:sha256`を加えた全field required objectである。

wrapperがFact `parallel.node.<node_rank>.process_lifecycle`へ出す`node-lifecycle/v1`もclosed objectとし、
次に固定する。

```text
schema_version: const "node-lifecycle/v1"
clock_id: nonempty string
spawn_attempt_monotonic_ns: uint
child_started_monotonic_ns: uint|null
child_finished_monotonic_ns: uint|null
termination: one of
  {kind:const "waited", raw_wait_status:uint}
  {kind:const "spawn_error"}
```

spawn成功時だけ両child timestampを持ち、attempt <= started <= finishedとする。`spawn_error`では両方null、
`waited`では両方non-nullかつraw statusがPOSIX `WIFEXITED xor WIFSIGNALED`を満たす。exit/signal分類と
code/signalはraw statusからconsumerが一度だけ導出する。wrapperはそれらやshutdown record、
early/coordinated分類を複写しない。
外側Factの`run`とkey内`node_rank`をvalueへ再掲しない。node childは`node_agent.py`内の小さなPython supervisorが
`os.waitpid(child_pid, 0)`で一度回収し、そのraw statusを保存する。Bash `wait`の`$?`はこのsourceに使わない。

`control/phase/summary.json`は`phase-summary/v1`でprofileだけに存在する。invocationごとに
`invocation_id/kind=warmup|measured`、rank 0の
`{clock_id,start_broadcast_ns,all_start_acks_received_ns,generation_start_ns,generation_end_ns,
end_broadcast_ns,all_end_acks_received_ns}`、node-rank順の
`{node_rank,clock_id,start_sequence,end_sequence,start_received_ns,start_ack_ns,end_received_ns,end_ack_ns}`を持つ。
rank 0の6 timestampは記載順にnondecreasingで同じclock IDを使う。start/end sequenceは同node control stream上で
strict increasing、GPU sampleとnode windowのclock_idは一致必須である。
GPU phase windowは同nodeの`[start_received_ns,end_received_ns)`で、ACK時刻はlead/trail診断だけに使う。
warmup/measuredのrank 0 generation境界は通常result writerが使う同じmonotonic-ns値であり、uniformな
`perf.phase.warmup_generation.wall_time_sec`または`perf.benchmark.generation.wall_time_sec`は
`(generation_end_ns-generation_start_ns)/1_000_000_000`のexact decimalに固定する。profileでは同じinvocation kindで
equalityを検査する。scalarは全variant共通result schema、timestamp pairはprofile window containmentに必要なためだけ併存する。
GPU sample intervalの初期値は1.0秒。
coverage thresholdを根拠なく固定せず、observed coverageとvalid durationを表示する。

`phase-summary-projection/v1`は全field requiredの
`{schema_version:const "phase-summary-projection/v1",run_id,invocation_id,kind,rank0:{clock_id,start_broadcast_ns,
all_start_acks_received_ns,generation_start_ns,generation_end_ns,end_broadcast_ns,all_end_acks_received_ns},
nodes:[{node_rank,clock_id,start_sequence,end_sequence,start_received_ns,start_ack_ns,end_received_ns,end_ack_ns},...],
source_member:relative POSIX path,source_sha256:sha256}`で、上記型/順序制約をそのまま持つ。

GPU sampleは次のclosed schemaである。

```text
schema_version="gpu-sample/v1", run_id:string, node_rank:uint, sequence:uint,
clock_id:string, monotonic_ns:uint, wall_time_utc:RFC3339 UTC string,
gpu_uuid:string, logical_index:uint,
utilization_percent:decimal|null, memory_used_bytes:uint|null, power_w:decimal|null,
temperature_c:decimal|null, sm_clock_mhz:decimal|null
```

`sequence`はnode poll-frame ordinalで、そのpollから得た全expected GPU rowが同じ値を持つ。frameはexpected UUIDを
exactly one rowずつ持つ時だけcompleteで、途中frameはartifactに残してもprojectionへ入れない。complete frameのsequence
gapは許すが補間用の架空sampleを作らない。utilizationは0..100、他のnon-null measurementはnonnegativeである。取得できないfieldはnullとし、poll
status/reason fieldは作らない。publisherはprofile raw tarをparseし、同じsample fieldに
`source_member:relative POSIX path/source_sha256:sha256`を加えたprojectionへ転記する。topologyとprofile hardwareも同様にexact source recordを
`qwen_topology_raw_v2`、`qwen_profile_hardware_raw_v2`へ転記し、hashだけで値を隠さない。
この追加後のclosed objectを`gpu-sample-projection/v1`とし、schema versionをconst同名に置換する。

profile-only CPU/NUMA rawはnode-rankごとに
`profile/raw/hardware/node-<node-rank>/{proc-meminfo.txt,
numa/node-<id>-cpulist.txt,numa/node-<id>-meminfo.txt,capture.json}`へ固定する。`capture.json`は各commandの
全field required、`additionalProperties=false`のclosed
`profile-hardware-capture/v1={schema_version:const "profile-hardware-capture/v1",run_id:run ID,node_rank:uint,hostname:nonempty string,
members:[{role:enum "proc_meminfo"|"numa_cpulist"|"numa_meminfo",numa_node_id:uint|null,
path:relative POSIX path,size_bytes:uint,sha256:sha256}]}`を持つ。
membersはrole/node/path順unique、proc_meminfoはnode ID nullでexactly one、NUMA二roleは検出nodeごとに同じIDを一件ずつ持つ。
読取元はrole/node IDから`/proc/meminfo`、`/sys/devices/system/node/node<id>/cpulist`、同`meminfo`へ固定導出し、
同じ`source_path`をrecordへ複写しない。読取境界はprofile node-agentの既存lifecycleが一度所有し、node別captureへ
argv/outcomeを複写しない。
publisher rowは全field required、`additionalProperties=false`の
`profile-hardware-projection/v1={schema_version:const "profile-hardware-projection/v1",run_id,node_rank:uint,hostname:nonempty string,record_kind:enum
"node_summary"|"numa_node",numa_node_id:uint|null,postrun_memory_available_bytes:uint|null,
cpulist:string|null,mem_total_bytes:uint|null,mem_free_bytes:uint|null,
source_members:[{role:enum "capture"|"proc_meminfo"|"numa_cpulist"|"numa_meminfo",
path:relative POSIX path,sha256:sha256},...]}`である。全rowにexactly one `capture`を持たせ、node summaryではNUMA field、
NUMA rowではpostrun fieldをnullにし、NUMA rowのsource memberはcapture/cpulist/meminfoを別entryで保持する。
captureは`gethostname()`を一度記録し、同runのminimal hardwareとnode-rank集合・hostname集合および各
`(node_rank,hostname)`をexact一致させる。`/proc/meminfo`はASCII `Key: uint kB`の`MemAvailable`、NUMA meminfoは
`^Node <id> (MemTotal|MemFree):[ \t]+([0-9]+)[ \t]+kB$`を各exactly once選び、別node ID prefixをrejectして
kBを1024倍してbytesへ変換する。duplicate key、非uint、overflow、unit不一致では
該当captureを投影しない。cpulistはASCII whitespaceをtrimし、kernel list grammarからstrictly increasing unique CPU集合を
最小range表記へcanonicalizeする。空fileは空string、read不能/parse不能値だけNULLにする。minimal hardwareにある
CPU identity/settingsを再収録しない。

GPU aggregateはraw factへ保存しない。`gpu-phase/v1` reducer/viewは`phase,node_rank,gpu_uuid` grainで次のclosed列を返す。

```text
phase_duration_sec:decimal,
valid_sample_count:uint,
valid_utilization_duration_sec:decimal, utilization_coverage:decimal,
utilization_mean:decimal|null, utilization_p95:decimal|null,
valid_power_duration_sec:decimal, power_coverage:decimal,
energy_j:decimal|null, power_mean:decimal|null,
valid_memory_sample_count:uint, memory_max_bytes:uint|null,
valid_temperature_sample_count:uint, temperature_max_c:decimal|null,
valid_sm_clock_sample_count:uint, sm_clock_min_mhz:decimal|null
```

`valid_sample_count`はschema-validなcomplete frameに属し、下記step intervalがphase windowと正durationで交差する
当該GPU row数で、全measurementがnullでも数える。metric別countとmemory/temperature/SM-clockのmax/minは、同じ
overlap条件を満たし当該measurementがnon-nullなrowだけを使う。したがってpre-start rowも有効区間がphaseへ入る時は
count/valueの双方へ入る。各sample値は次sample、
phase end、`sample_time + 2 * nominal_interval`の最小まで一定とする。duration系は
このstep intervalとphase windowのoverlapだけを足し、coverageはvalid duration / phase duration、energyは
valid power intervalをstep積分して欠損intervalへ外挿しない。utilization p95はoverlap durationで重み付けした
nearest-rank、`power_mean=energy_j/valid_power_duration_sec`とする。fleet projectionはexpected GPU gridへleft joinし、included/valid GPU count、coverage min、GPU統計の
min/median/max、energy sumをquery時に導出する。complete/status booleanは作らず、必要入力が足りないprojectionはNULLにする。

## workloadとresult

### ordered workload v1

`source/workload-{warmup,measured}.json`は
`schema_version:const "ordered-workload/v1"/dataset_manifest_sha256/kind/requests`を持つ。`kind`は`warmup|measured`、warmupだけ`mode` enum
`random|dataset`を追加する。run/allocation IDはsemantic workload bytesへ入れず、subject bundle manifestとpublisher
projectionだけがbindingを所有する。このため同じactual workloadはrunを跨いでも同じsource SHA-256になる。
requestsは実generate順で、各要素は次を持つ。

```text
source:{kind="dataset_row", array_index:uint} | {kind="generated", generator_id:string, seed:uint, generator_ordinal:uint},
prompt_payload:{kind="text", text:string, input_token_count:uint} | {kind="token_ids", token_ids:[int,...]},
requested_output_token_count:uint,
sampling_params:[{name:string, declared_type:string, value:JCS value}, ...]
```

`prompt_payload`は現行text pathを勝手にtoken ID pathへ変えず、実generateへ渡す値をtagどおり保存する。textの
`input_token_count`はrequest選択時にtokenizerが実測した値で、token_idsではarray lengthから導出するため保存しない。
sampling paramsは実generateへ渡す全fieldをname byte順に持つ。manifest identityはrequest array順とexact
prompt payloadを含む。generated本文は保存しない。publisherはmanifest top-levelを
`qwen_workload_manifests_raw_v2`、同じrequest recordを`qwen_workload_requests_raw_v2`へ転記する。source member/hashは
manifest relationだけが所有し、request relationはmanifest identityでjoinする。

`workload-manifest-projection/v1`は全field required、`additionalProperties=false`の
`{schema_version:const "workload-manifest-projection/v1",subject_kind:enum "regular_run"|"qualification_allocation",subject_id:string,
workload_kind:enum "warmup"|"measured",dataset_manifest_sha256:sha256,warmup_mode:enum "random"|"dataset"|null,
source_member:relative POSIX path,source_sha256:sha256}`を持つ。measuredのmodeはnullで、
`source_sha256`がsubject bindingを除いたcanonical workload fileのsemantic manifest identityを兼ねる。
`workload-request-projection/v1`は全field requiredの
`{schema_version:const "workload-request-projection/v1",subject_kind,subject_id,workload_kind,ordinal:uint,request:<上記request closed object>}`を持ち、親manifest projectionを
`subject_kind/subject_id/workload_kind`でjoinする。manifest hashとsource lineageをrequest rowへ複写しない。

`ordinal`はordered `requests` arrayの0-based offsetからpublisherがnormalization時に一度だけ付与し、source itemへは複写しない。

manifestだけがordered requestのraw正本である。selected count、planned input/output token total、warmup modeの
別raw keyを作らず、consumerがarrayから導出する。qualificationではparentがarm 1前にmanifestを一度生成し、OFF/ON
両方のneutral request-manifest harnessが同じbyteを入力として順番通り送る。childへmanifest byteを複写しない。

### primary result v1

`source/primary-result.json`はrank 0がgenerate return直後に一度write/renameし、次を持つ。

```text
schema_version: const "primary-result/v1", run_id:nonempty string, invocation_id:nonempty string,
generation_wall_sec:positive finite decimal, returned_count:uint, error_count:uint,
observed_input_tokens:uint|null, observed_output_tokens:uint|null
```

parse不能responseは`error_count`へ数え、取れないtoken totalを0で補わない。generate例外時はprimary resultを
捏造しない。`benchmark_valid`はerror=0、returned=workload requests length、input observed=workload由来planned total、generation wallが正有限、
`ignore_eos`固定時はoutput observed=requestedであることだけから導出する。

view formulasは次で、throughputをrawへ保存しない。

```text
request_throughput = returned_count / generation_wall_sec
input_token_throughput = observed_input_tokens / generation_wall_sec
output_token_throughput = observed_output_tokens / generation_wall_sec
total_token_throughput = (observed_input_tokens + observed_output_tokens) / generation_wall_sec
```

qualificationだけの`source/qualification-result.json`は、score referenceとfinalの両方に入る共通
`QualificationResultWriter`がresponse validation直後にwrite/renameする。closed schemaは次である。

```text
schema_version: const "qualification-result/v1", run_id:nonempty string, invocation_id:nonempty string,
generation_wall_sec:positive finite decimal, returned_count:uint, error_count:uint,
observed_input_tokens:uint|null, observed_output_tokens:uint|null
```

予定request count、backend、throughput、verdictは保存しない。eligibilityとthroughputは共通workloadと上記原始値から
導出する。ONでは`primary-result/v1`とのreturned/error/token/wall一致を検査してwriter自体をqualificationする。
二sourceを持つのはONだけで、contractの`duplication_justification`にこの独立writer検証を記す。

## control、merge、package

以下はすべて全記載field required、`additionalProperties=false`である。`member`はclosed
`{path:relative POSIX path,size_bytes:uint,sha256:sha256}`、`process-outcome`はclosed union
`{kind:const "exit",code:int}` | `{kind:const "signal",signal:positive int}` |
`{kind:const "spawn_error"}`である。rare failure用のretry counter、reason、代替sourceは追加しない。

```text
scheduler-request/v1 = {
  schema_version:const "scheduler-request/v1",subject_manifest_sha256:sha256,
  lane_id:nonempty string,nodes:const 2,mpi_processes:const 2,
  resource_group:nonempty string,transport_policy:nonempty string,
  pjm_script:{repository_path:repository-relative POSIX path,sha256:sha256},argv:nonempty string[]
}
submission-result/v1 = {
  schema_version:const "submission-result/v1",request_sha256:sha256,
  outcome:process-outcome,stdout:member,stderr:member,parsed_job_id:nonempty string|null
}
job-entry/v1 =
  {schema_version:const "job-entry/v1",subject_kind:const "regular_run",run_id:run ID,
   pjm_job_id:nonempty string,entry_utc:RFC3339 UTC,environment_snapshot:member} |
  {schema_version:const "job-entry/v1",subject_kind:const "qualification_allocation",allocation_id:allocation ID,
   ordered_child_run_ids:exactly 2 run IDs,pjm_job_id:nonempty string,entry_utc:RFC3339 UTC,
   environment_snapshot:member}
scheduler-terminal/v1 = {
  schema_version:const "scheduler-terminal/v1",argv:nonempty string[],outcome:process-outcome,
  stdout:member,stderr:member,observed_terminal_utc:RFC3339 UTC
}
stage-launch/v1 = {
  schema_version:const "stage-launch/v1",argv:nonempty string[],cwd:string,
  stderr_mode:enum "stdout"|"separate"
}
stage-termination/v1 = {schema_version:const "stage-termination/v1",outcome:process-outcome}
bundle-manifest/v1 =
  {schema_version:const "bundle-manifest/v1",subject_kind:enum "regular_run"|"qualification_child",
   run_id:run ID,contract_sha256:sha256,members:member-with-role[]} |
  {schema_version:const "bundle-manifest/v1",subject_kind:const "qualification_allocation",
   allocation_id:allocation ID,contract_sha256:sha256,members:member-with-role[]}
artifact-manifest/v1 = {
  schema_version:const "artifact-manifest/v1",logical_artifact_name:nonempty string,
  files:[{role:enum "run_bundle"|"metrics"|"profile_raw",
          size_bytes:uint,sha256:sha256}, ...]
}
```

`member-with-role`は`member`へcontract registryのclosed source-roleを一つ加えたobjectで、path byte順uniqueにsortする。
bundle manifestはself/outer sidecarを含めない。artifact filesは記載role順uniqueで、bench/qualification childは
`run_bundle,metrics`、profileは`run_bundle,metrics,profile_raw`をexactly once持つ。filenameはroleから
`run-artifacts.tar.zst|metrics.jsonl|profile-raw.tar.zst`へ固定導出し、manifestへ複写しない。
artifact manifest自身とupload IDは含めない。途中失敗でこの集合を作れないchildはset manifestで
`artifact_manifest=null`だけとし、partial manifest、placeholder、reasonを作らない。compute/collectorが失敗してもsupervisorが
finalizerまで到達できたchildは、完成済みlog/sourceと`collection_complete=false`を持つ通常artifact setを一度作る。
nullにするのはfinalizer前のcrashまたはpackage失敗でmanifestが存在しない場合だけで、両者を区別する原因fieldは持たない。
`request_sha256`は実scheduler request file bytes、terminal argv末尾job IDはsubmission/job-entryの一致済みIDである。

`scheduler-accounting/v1`は固定`pjstat --history -S --choose jid,st,sdt,elp,ec,sn,pc,nnumu,mmszu,uctmut,sctmut
--data --delimiter "\t" <source-specific-job-id>`の一rowをclosed
`{schema_version:const "scheduler-accounting/v1",jid:string,st:string,sdt:string,elp:{lexeme:string,seconds:finite decimal|null},
ec:{lexeme:string,value:int|null},sn:{lexeme:string,value:int|null},pc:{lexeme:string,value:int|null},
nnumu:{lexeme:string,value:int|null},mmszu:{lexeme:string,bytes:uint|null},
uctmut:{lexeme:string,seconds:finite decimal|null},sctmut:{lexeme:string,seconds:finite decimal|null}}`で持つ。
返却`jid`はargv末尾IDと一致必須である。`elp.seconds`は括弧内decimal seconds、CPU timeはmillisecondを
`decimal / 1000`、memoryは括弧内decimalをbytesへparseし、parse不能時はnumericだけnullにする。

field別missing/reason/issue count、collection validation fileは持たない。finalizerはimmutable contract/protocol/bundleから
導出したexpected setとobserved setをメモリ上で一回照合し、raw fact `collection_complete:bool`と
merge済み`metrics.jsonl`を作る。

`run-lifecycle/v1`の境界は、regularではPJM script entry直後からcompute source/log回収終了まで、
qualification childではそのchild orchestration開始直前からそのsource/log回収終了までとする。
二childは別々のUTC wall/monotonic boundaryを持ち、allocation-wideのPJM entry/accounting/terminalは
`qwen_qualification_allocations_raw_v2`だけが所有しchild Factへ複写しない。別のallocation lifecycle objectは作らない。

Git管理の`instrumentation-decision/v1`はclosed
`{schema_version:const "instrumentation-decision/v1",decision_id:nonempty string,qualification_namespace:
{contract_id:nonempty string,contract_sha256:sha256,protocol_id:nonempty string,protocol_sha256:sha256,
dimension_policy_hash:sha256},qualified_source_diff_sha256:sha256,
observed_sif_pair:{reference_sif_semantic_sha256:sha256,
final_sif_semantic_sha256:sha256},observed_runtime_launcher_semantic_sha256:sha256,
decision:enum "adopt"|"reject"|"inconclusive"}`を持つ。
このnamespaceのexact protocol relationが`lane_id`、reference source object、
workload identity、reference/score bundle path/SHA-256、control-plane file blob集合、protocol順のexactly 5 pair scheduleを一行で所有する。
decisionへこれらの
完全な部分集合を複写しない。

SIF semantic SHA-256はprovenanceから次のclosed objectを構成し、`SHA-256(JCS(object))`で得る。

```text
schema_version: const "sif-semantic/v1"
sglang_source: {repository_id:const "apac2026-sglang",object_hash_algorithm:enum "sha1"|"sha256",
 oid:algorithm-constrained full OID}
score_source_tree: score source entries above
recipe: {path:relative POSIX path,blob_oid:algorithm-constrained full OID}
applied_patches: sorted [{path:relative POSIX path,blob_oid:algorithm-constrained full OID}, ...]
base_oci: {reference:string,manifest_digest:string,config_digest:string,layer_digests:[string, ...]}
lock_manifest: {path:relative POSIX path,sha256:sha256}
build_command: nonempty string[]
toolchain_observations: exactly 4 role/outcome/stdout_b64/stderr_b64 objects in declared role order
runtime_files: exactly 4 role/realpath/size_bytes/sha256 objects in declared role order
```

`sif_build_definition` commit、完成SIF byte hash、embedded manifest identity、outer sidecar path/hashは除外する。
pairの二hashは各qualification childがFact `artifact.sif_manifest_sha256`でjoinしたSIF provenanceから再計算し、
採用後benchも同じsemantic値を`final_sif_semantic_sha256`へ一致させる。
採否に使うeligible pair（最低3 pair）の実在OFF childは同一reference SIF semantic hash、実在ON childは同一final SIF semantic hashと同一
`SHA-256(JCS(source-track-diff/v2.entries))`を持ち、そのdiff hashはdecision `qualified_source_diff_sha256`、その順序付き一pairが
decision `observed_sif_pair`と一致する。欠損/conflict/異なるSIFの混入があれば`adopt`にできない。
未起動/null childはpairを欠測にするだけで、他の3–4 eligible pairによるdecisionを妨げない。
各qualification childのcontract/protocol/policy composite keyはqualification namespaceとexact一致させる。採用後benchは
別のscore protocolでもよいが、manifestのdecision ID/hashでこの不変record全体をexact joinする。

Git管理relation `qwen_instrumentation_decisions_v2`はrecord全fieldと
`decision_sha256=SHA-256(JCS(record))`を一行に持ち、grainは`(decision_id,decision_sha256)`である。adopt済みscore benchだけ
run manifestのID/hashでexact joinし、profile/qualificationや未回答benchはNULLにする。
run artifactへdecision recordのimmutable copyは作らない。authoring時の一時JSONはGit-managed relationに取り込んだ後、
runの参照sourceにしない。

logical `qwen_instrumentation_decision_context_v2`はdecisionの`qualification_namespace` composite keyを
Git管理qualification protocol relationの`contract_id/hash,protocol_id/hash,dimension_policy_hash`へexact inner joinし、
decision fieldsとprotocolのlane/reference/workload/reference+score bundles/control-plane files/5-pair scheduleを一行で露出する。
これは保存copyでなくviewで、grainはdecision ID/hashである。adopted benchはmanifestのdecision ID/hashからこのviewへ一段joinし、
actual contract/policy、lane、reference、workload、score bundle、control-plane blobs、final SIF semantic、runtime launcher semanticを照合する。
qualified source diff hashはqualification evidenceとしてcontextに残すが、unrelated later commitを許すadopted benchのcurrent HEAD diffへ
再照合しない。

qualificationはOpenFOAM基準と同じ外側artifact setにする。

```text
qualification/<allocation-id>/qualification-set-manifest.json
qualification/<allocation-id>/allocation/allocation-artifacts.tar.zst
qualification/<allocation-id>/runs/<off-child-id>/{run-artifacts.tar.zst,metrics.jsonl,artifact-manifest.json} # off artifact_manifest non-null時だけ
qualification/<allocation-id>/runs/<on-child-id>/{run-artifacts.tar.zst,metrics.jsonl,artifact-manifest.json}   # on artifact_manifest non-null時だけ
```

`qualification-set-manifest/v1`は全field required、`additionalProperties=false`のclosed objectである。

```text
schema_version: const "qualification-set-manifest/v1"
allocation_id: allocation ID
allocation_archive: {path:relative POSIX path,size_bytes:uint,sha256:sha256}
ordered_children: exactly 2 [{run_id:run ID,arm:enum "off"|"on",artifact_manifest:member|null}]
```

array offsetが実行ordinalを所有し、同値integerをchild entryへ複写しない。childのrun bundle/metrics/profile
outer identityは実在する各`artifact-manifest.json`だけが所有し、set manifestへ再掲しない。未起動またはpackage前に
失敗したchildは`artifact_manifest=null`だけとし、reason/placeholder artifactを作らない。allocation archiveは
parent manifest、contract、protocol、submission result、job-entry、
scheduler request/accounting/terminal、job-entry snapshot、runtime launcher validation、node別minimal hardware、model/dataset manifest、共通measured/warmup workload manifest、
PJM log、tar root bundle manifestを一度だけ収録する。publisherはこれをallocation-grain
`qwen_qualification_allocations_raw_v2`へ次のclosed `qualification-allocation-projection/v1`として一row投入する。

```text
schema_version: const "qualification-allocation-projection/v1"
allocation_id, lane_id, pair_id: nonempty string
contract_id, protocol_id: nonempty string
contract_sha256, protocol_sha256, dimension_policy_hash: sha256
ordered_children: exactly 2 [{arm:enum "off"|"on", run_id:nonempty string}]
scheduler_request: scheduler-request/v1
submission_job_id: string|null
job_entry_job_id: string|null
scheduler_accounting: scheduler-accounting/v1|null
scheduler_terminal: scheduler-terminal/v1|null
observed_control_plane_sha256: sha256
job_entry_environment: [{name_b64:base64, value_b64:base64}, ...] | null
bundle_members: bundle-manifest/v1 members array
bundle_manifest_member: relative POSIX path
bundle_manifest_sha256: sha256
allocation_archive_member, set_manifest_member: relative POSIX path
allocation_archive_sha256, set_manifest_sha256: sha256
```

`bundle_members`はarchive内`bundle.manifest.json`のfull member arrayを順序も含めて一度転記し、選択subsetを作らない。
normal pathでは`allocation_manifest/contract/protocol/scheduler_request/runtime_launchers/model_manifest/dataset_manifest/
workload_warmup/workload_measured` roleをexactly 1持ち、optional sourceは実在時だけ一memberを持つ。
`bundle_manifest_member/hash`はarchive内manifest自身のlocatorで、selfを`bundle_members`へ入れない。
`allocation_archive_member/set_manifest_member`はouter setからarchiveへ辿るlineageで、tar内の`bundle_members`へ
重ねない。job-entry snapshotが実在してenvironmentが空なら`[]`、member欠損またはdecode失敗なら`null`にする。
`ordered_children`のarray offset 0/1がordinalを所有し、arm off/onを各一件持つ。parent manifest/set/projectionの
ordered run IDsを常に一致させ、job-entryが実在する時だけそのIDsも照合する。各childのrun ID/arm/
childのtop-level armはset/projection entryと一致させる。
childのparent hashは`bundle_members[role="allocation_manifest"].sha256`と一致させる。二job IDのequalityはviewだけが導出する。
minimal hardwareはsubject-keyed hardware projection、ordered warmup/measured workload payloadは
`qwen_workload_requests_raw_v2`が`subject_kind="qualification_allocation"/subject_id=allocation ID`で一度だけ所有し、
allocation rowへ複写しない。child queryはallocation IDでjoinし、親を擬似runにしない。

runtimeで`archive-staging/bundle.manifest.json`を作り、tar rootの`bundle.manifest.json`にする。GitHub Artifact内の
基本fileは`run-artifacts.tar.zst`、`metrics.jsonl`、`artifact-manifest.json`である。profileでは
`profile-raw.tar.zst`を追加する。

- `run-artifacts.tar.zst`: control/source/domain/log/result。telemetry rawを含めない
- `profile-raw.tar.zst`: `telemetry/**`とprofile-only hardware raw、D-004で適用と決まったprofiler rawだけ。
  run file三点と同じGitHub logical artifact set内の別outer fileである
- `artifact-manifest.json`: outer file identityだけ
- workflow operational row: GitHub run ID/attempt、artifact ID、home destination、upload/SCP/publish outcome

同じ実byteを二tarへ入れない。qualification allocation archive/set manifestもpackage ownershipに含む。
merge後のproducer fragmentは`metrics.jsonl`の完全な部分集合なのでartifactへ入れず削除する。merge前の
partial/tmpは失敗診断で手元に残り得るが、publisherのFact入力にしない。
bundle URIはraw metricへ戻さずworkflow operational rowからjoinする。home pathは
`/hdd_raid/samba/net_share/qwen-artifacts/<github_run_id>/attempt-<github_run_attempt>/`へ固定する。OpenFOAMと同様、
GitHub Artifact upload後にruntimeをcleanupし、SCP成功時だけGitHub Artifactを削除する。remote transaction、
reconciler、delivery ledgerは作らない。

## artifact projection registry v1

artifactをBigQueryでqueryableにする転記は`[[artifact_projection]]`で閉じる。各entryのcanonical fieldは
次だけである。

```text
id, base_table_id, relation_id, conflict_relation_id,
source_artifact_id, source_schema_id, row_schema_id, row_grain,
subject_scopes, publisher_id, transform_id,
conflict_rule, source_member_field?, source_sha256_field?,
source_instance_grain?, source_instance_sha256_field?, parent_projection_id?, parent_join_columns?,
duplication_justification
```

`subject_scopes`はtagged arrayで、run ownerは
`{owner_scope="run",applicable_run_variants=[...]}`、allocation ownerは
`{owner_scope="qualification_allocation"}`、build ownerは`{owner_scope="build"}`とし、後二者へrun variantを置かない。publisherは一source instanceを一つの
BigQuery load jobでphysical base tableへappendする。`relation_id`は全列完全一致を一件へdedupeする。
同じ`source_instance_grain`に異なるinstance hashがあればそのinstanceの全rowを、次に同じ`row_grain`へ異なるpayloadが
あればその全rowを`conflict_relation_id`へ残してconsumerから外す。再publishしたGPU sampleやworkload requestを
二重計数しない。direct lineageの二field、または`parent_projection_id`と明示的`parent_join_columns`の組の
どちらか一方だけを必須にする。workload requestはmanifest projectionをparentにし、join mappingは
`subject_kind=subject_kind,subject_id=subject_id,workload_kind=workload_kind`である。
v1 entryは次である。

| id / relation | source -> row schema | grain | applicability |
|---|---|---|---|
| `qwen.source_track` / `qwen_source_tracks_raw_v2` | subject manifest source-track record -> `source-track-projection/v1` | run | run: all variants |
| `qwen.sif_provenance` / `qwen_sif_provenance_raw_v2` | `sif-sidecar/v1`+referenced raw -> `qwen-sif-provenance-projection/v1` | source SHA-256 | build |
| `qwen.runtime_launcher` / `qwen_runtime_launchers_raw_v2` | `runtime-launcher-validation/v1` -> `runtime-launcher-projection/v1` | subject kind/ID | run: bench/profile; qualification allocation |
| `qwen.workload_manifest` / `qwen_workload_manifests_raw_v2` | `workload/v1` -> `workload-manifest-projection/v1` | subject kind/ID, kind | run: bench/profile; qualification allocation |
| `qwen.workload_request` / `qwen_workload_requests_raw_v2` | `workload/v1` -> `workload-request-projection/v1` | subject kind/ID, kind, ordinal | run: bench/profile; qualification allocation; parent manifest join above |
| `qwen.minimal_hardware` / `qwen_minimal_hardware_raw_v2` | `hardware-node/v1` -> `minimal-hardware-projection/v1` | subject kind/ID, node rank | run: bench/profile; qualification allocation |
| `qwen.node_launch` / `qwen_node_launch_raw_v2` | `node-topology/v1` -> `node-launch-projection/v1` | run, node | run: all variants |
| `qwen.topology` / `qwen_topology_raw_v2` | `node-topology/v1`+stdout -> `topology-projection/v1` | run, node, scope, GPU index | run: all variants |
| `qwen.gpu_sample` / `qwen_gpu_samples_raw_v2` | `gpu-sample/v1` -> `gpu-sample-projection/v1` | run, node, sequence, GPU UUID | profile |
| `qwen.profile_hardware` / `qwen_profile_hardware_raw_v2` | `profile-hardware-capture/v1` -> `profile-hardware-projection/v1` | run, node, kind, NUMA ID | profile |
| `qwen.lifecycle_summary` / `qwen_lifecycle_summaries_raw_v2` | `lifecycle-summary/v1` -> same closed row + source member/hash | run | run: all variants |
| `qwen.phase_summary` / `qwen_phase_summaries_raw_v2` | `phase-summary/v1` -> one invocation + source member/hash | run, invocation ID | profile |
| `qwen.qualification_allocation` / `qwen_qualification_allocations_raw_v2` | `qualification-set-manifest/v1` + hash-verified referenced allocation archive -> `qualification-allocation-projection/v1` | allocation ID | qualification allocation |

direct entryのsource-instance grainはsource track=`run + subject manifest member`、SIF provenance=`source_sha256`、runtime launcher=`subject kind/ID + launcher member`、workload manifest=`subject kind/ID/kind + manifest member`、minimal hardware=
`subject kind/ID/node + node member`、node launch/topology=`run/node + topology member`、GPU sample=
`run/node + JSONL member`、profile hardware=`run/node + capture member`、lifecycle/phase=`run + summary member`、
qualification allocation=`allocation ID`で、instance hashはsource trackのsubject manifest、SIF sidecar、各run member、
またはqualification set manifestのSHA-256に固定する。
topology member自体が参照する全stdout/stderr hashをsealするため、そのmember hashがparse入力集合もidentity化する。workload requestはconflict-free manifest
parentへjoinするためdirect source-instance fieldを持たない。
以下のoverrideを除くdirect entryは`source_member_field=source_member`、
`source_sha256_field=source_instance_sha256_field=source_sha256`に固定する。
profile hardware entryは`source_member_field=source_members[role="capture"].path`、
`source_sha256_field=source_instance_sha256_field=source_members[role="capture"].sha256`という一意selectorに固定する。
qualification allocation entryは`source_member_field=set_manifest_member`、
`source_sha256_field=source_instance_sha256_field=set_manifest_sha256`に固定する。set manifestがallocation archive hashをsealする。

## producer/source registry v1

contractの`[[producer]]`は共通field
`id:nonempty unique string/activation/instances`に、次のclosed ownership unionを加える。

```text
{owner_scope:const "run", applicable_run_variants:nonempty unique variant[]}
{owner_scope:const "qualification_allocation"}
```

allocation ownerにrun variantを置かず、activationは`{kind:"always"}`だけにする。run ownerのactivationは
`{kind:"always"}`または`{kind:"bundle_component",component_id:nonempty string}`で、後者のcomponent IDは各applicable variantが
選ぶbundleにexactly one存在する。`instances`は`{kind:"singleton"}`または
`{kind:"cartesian",axes:nonempty sorted unique subset of ["node_rank","global_rank","invocation_id","stage_name"]}`である。
axis expected集合はresolved lane、protocol schedule、contract stage setだけから得る。GPU UUIDはruntime topology record内の値であり、
expected source pathを決めるaxisにしない。

`path-template`はnonempty segment arrayで、segmentは`{literal:nonempty string}`または
`{token:enum "node_rank"|"global_rank"|"invocation_id"|"stage_name"}`のclosed unionである。adjacent literalを一つに結合し、
render後はrelative POSIX pathとする。templateのtokenはproducer axesのsubsetだけを使う。

`[[source]]`は全field requiredの
`{id:nonempty unique string,producer_id:existing producer ID,source_role:nonempty contract role,
storage:enum "fact_fragment"|"run_bundle"|"profile_raw"|"allocation_bundle",
path_template:path-template,schema_id:nonempty schema ID,required_when_applicable:bool}`である。同じstorage内で展開後pathをglobal uniqueにする。
producerごとの`storage="fact_fragment"` sourceは0-or-1件とし、そのsource `path_template`だけをwriter/finalizerの正本にする。
producerへ同じtemplateを複写しない。各`[[fact]].source_id`はexactly one source、`member-with-role.role`はexactly one registry roleへ解決する。
finalizerはvariant/bundleからproducerを選び、instanceを一度展開し、
source path集合とrequired fragment集合を得る。別のcode-side expected table、fallback discovery、manifestへのexpected集合copyを作らない。

## factとconsumer routing

各`[[fact]]`は
`id/key_pattern/source_id/schema_id/logical_type/unit/cardinality/capture_id/validator_id/reducer_id/duplication_justification?`を持つ。
sourceからproducer、owner scope、applicable variant、requirednessを一意にjoinし、factへ複写しない。
`(source.schema_id,fact.schema_id,validator_id,reducer_id)`はcontractのclosed reducer signature catalogにexactly one存在し、
同じvariantで有効なfact key pattern同士をoverlapさせない。

v1 routingは次に固定する。

| family | source | reducer |
|---|---|---|
| `run.identity.*`, `experiment.*`, protocol | subject/contract/protocol manifests | `identity_projection/v1` |
| `run.metrics_contract` | immutable contract copy and manifest binding | `manifest_identity/v1` |
| `run.started_at|finished_at|elapsed_sec|exit_code` | supervisor clock/exit record | `run_lifecycle/v1`。`run_outcome`はexit code 0/非0からview導出 |
| `run.stage.<name>.launch|termination`, `perf.stage.<name>.wall_time_sec` | StageRunner spawn/wait | `stage_record/v1` |
| `perf.phase.*.wall_time_sec` | rank 0 monotonic boundaries | `finite_seconds/v1`。profile warmupはphase pairとのexact equalityを持つuniform-result重複 |
| `perf.benchmark.generation.wall_time_sec` | purpose-authoritative result artifact | `finite_seconds/v1`。profile measuredはphase pairとのexact equalityを持つuniform-result重複 |
| `collection_complete` | finalizerのin-memory expected-vs-observed set | `completeness/v1` |
| `artifact.sif_manifest_sha256|artifact.model.manifest_sha256|artifact.dataset.manifest_sha256|instrumentation_bundle_sha256` | package/source manifests | `manifest_projection/v1` |
| `cache.node.<node_rank>.input_manifest_sha256` | cache semantic member manifest | `identity/v1`。consumer `cache_nodes/v1`はexpected node順のclosed arrayへreduce |
| `scheduler.request` | submission request | `scheduler_request/v1` |
| `scheduler.accounting` | fixed pjstat raw | `scheduler_accounting/v1` |
| `scheduler.submit_job_id|scheduler.job_entry_job_id` | submission result + job-entry | `identity/v1`、欠損時key absent |
| `env.*` | scope別snapshot | `environment_decode/v1` |
| `sglang.node.<node_rank>.bench_args_effective|server_args_effective` | node event JCS object | `node_config/v1` |
| `parallel.worker.<global_rank>.runtime` | global-rank event | `worker_ready/v1` |
| `result.*` | purpose-authoritative result artifact | `primary_result/v1`または`qualification_result/v1` |
| `parallel.node.<node_rank>.process_lifecycle` | wrapper lifecycle | `node_lifecycle/v1` |

consumer columnは`name/type/unit/source_refs/scope_cardinality/reducer_id/conflict_rule/transform_id/
applicable_run_variants/dimension_name/allowed_classes`で閉じる。`source_refs`はordered
`[{kind="fact"|"projection"|"dimension",id,json_path?,bindings?},...]`で、異種sourceの混合とreducer入力順を明示する。
`dimension`は同じGit releaseで作る`contract|protocol|instrumentation_decision_context` relationのclosed union、
projectionはconflict-free logical relationをqueryする。
archive memberを直接読む`artifact` refは許さず、lineageだけprojectionのsource member/hashから辿る。
projection refの`bindings`はnonempty
`[{applicable_run_variants,consumer_field,projection_field,projection_subject_kind}]`でvariant集合を重複なく覆い、
run-keyed projectionは`run_id -> run_id`、regular subject-keyed projectionは
`run_id -> subject_id, regular_run`、qualification childはsubject-keyed projectionへ
`block_id -> subject_id, qualification_allocation`、allocation projectionへ`block_id -> allocation_id`、build projectionは
Fact `artifact.sif_manifest_sha256 -> source_sha256, build`を明記する。
dimension refの`bindings`はordered nonempty `[{consumer_field,dimension_field},...]`である。contract relationは
exact `(contract_id,contract_hash) -> (contract_id,contract_sha256)`、protocol relationはexact
`(contract_id,contract_hash,protocol_id,protocol_hash,dimension_policy_hash) ->
(contract_id,contract_sha256,protocol_id,protocol_sha256,dimension_policy_hash)`を使う。branch、最新release、
IDだけによるfallback joinは許さない。
decision relationはexact `(instrumentation_decision_id,instrumentation_decision_sha256) ->
(decision_id,decision_sha256)`でjoinし、artifact内copyやlatest decisionをSQLから探索しない。
sourceが欠損または非適用なら`NULL`だけを返す。Sources tabは実在値のlineageだけを表示し、missing rowを生成しない。

termination consumerは次のexact contractで閉じ、`ANY_VALUE`や多数決を使わない。

| consumer | ordered source refs / scope cardinality | reducer and precedence |
|---|---|---|
| `benchmark_termination_kind|benchmark_exit_code|benchmark_signal` | Fact `run.stage.benchmark.termination`、runにつきexactly 1 | `stage-termination/v1` unionをそのまま列へ分解する。kindに非該当のcode/signalはNULL |
| `engine_termination_kind|engine_exit_code|engine_signal` | Fact `parallel.node.<node_rank>.process_lifecycle`の`node_rank=0`、runにつきexactly 1 | `spawn_error`、またはraw wait statusからPOSIX `exit|signal`と`WEXITSTATUS|WTERMSIG`を一度導出する。非該当のcode/signalはNULL |
| `worker_termination_status` | ordered [Fact `parallel.node.<node_rank>.process_lifecycle`の`node_rank=1`, projection `qwen.lifecycle_summary`の`nodes[node_rank=1].shutdown`]、各exactly 1 | lifecycleが`spawn_error`なら`spawn_error`。それ以外はraw statusからexit/signalを導出し、同じclock IDでchild finishがshutdown receive未満なら`early_exit|early_signal`、以上なら`coordinated_exit|coordinated_signal`。source欠損、競合、shutdown null、invocation不一致、clock不一致はNULL |

lifecycle summaryのshutdown invocation IDはpurpose-authoritative result artifactの`invocation_id`と一致させてから
worker reducerへ渡す。qualification ONの二result writerは先に同一invocation IDであることを検証する。上表はrankを
固定するためnode集合から暗黙にworst statusを選ばない。
