# Publication redactions

The public working tree uses anonymized infrastructure identifiers. Scores, request counts, latency metrics, model and source revisions, image hashes, kernel choices and numerical arguments retain their recorded values.

## Edited identifiers

- The two recorded image build logs replace the private ECR account number with `000000000000`. This is a placeholder, not a usable registry account.
- Nine historical PBS accounting files replace `Job_Owner` with `ACCOUNT@scheduler`.
- Research notes and derived analysis records use `archive-host`, `/srv/qwen-artifacts` and `/home/USER` for private archive hosts, archive roots and personal analysis directories. These locations are illustrative; configure local storage before running an analysis command.
- Links to private organiser chat and planning documents have been removed. The clarification date and its Basic/Bonus classification remain stated.

Unedited originals are retained in a private archive outside this repository. `SHA256SUMS` files authenticate the public copies, including the anonymized accounting records. These edited records must not be described as byte-identical to the original scheduler output. Runtime SGLang Python, its dependency lock, the raw boundary commit and kernel patches remain unchanged. The reconstruction bundle omits the internal mirror-maintenance workflow throughout its three team commits; the standalone builder pins the resulting public source ID. [The source map](build/source-map.json) relates it to the source ID of the measured image. Site account components in saved paths and configuration defaults are anonymized placeholders; adapt them to the reproduction account.

## Local configuration

The GitHub repository secret `QWEN_ARTIFACT_ARCHIVE_ROOT` supplies the persistent archive destination. The GPU-clock analysis helper accepts the same name as a local environment variable. Local source-mirror tests may use `SGLANG_SOURCE_MIRROR`; tests that require that optional mirror skip when it is not configured. Connection settings and personal operational instructions are local files excluded from Git tracking. The historical task checklist is retained with the same publication redactions.

The submitted presentation is distributed separately. The experiment index preserves its Qwen slide references.

The cleanup has also been applied throughout the Git branch and tag history. Original commit IDs retained in experiment provenance still identify the recorded builds and runs; they have not been relabelled as builds of the rewritten commits. Recovery archives and the old-to-new commit map are stored privately outside this repository. GitHub-managed PR references, comments and historical Actions output require separate handling.

## Issue references

Issue and document navigation links use sanitized commit IDs. Additional historical snapshots referenced by those links are retained under `issue-history/<original-commit-id>` tags after applying the same redactions and updating their public checksum manifests. Author identities, timestamps, merge structure and numerical results are preserved; original source and image IDs recorded as measurement provenance remain unchanged.

## Actions history

Repository Actions remains enabled for retained run metadata. Current workflow YAML has no automatic event triggers and every job is disabled by a literal false gate, while preserving its original condition. File-backed workflows are also disabled individually so old definitions cannot be re-run.

[The publication archive](https://github.com/2026-APAC-HPC-AI-Tokyo-Rapids/2_Qwen/tree/public-actions-history-20261115) contains sanitized downloadable logs, textual artifact evidence and pull-request conversations. The original run IDs, authors, timestamps, conclusions and measured values are retained. Complete raw originals are backed up privately before affected hosted copies are removed. Opaque profiler binaries and large reproducible model inputs remain in that private backup. Navigation links point to the retained public records.

Seven historical CPU-proof archives have also been sanitized internally, including archive ownership and embedded debug paths. Their public manifests describe the revised bytes; executable ELF sections and numerical JSON evidence are unchanged.
