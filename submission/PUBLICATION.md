# Publication redactions

The public working tree uses anonymized infrastructure identifiers. Scores, request counts, latency metrics, model and source revisions, image hashes, kernel choices and numerical arguments retain their recorded values.

## Edited identifiers

- The two recorded image build logs replace the private ECR account number with `000000000000`. This is a placeholder, not a usable registry account.
- Nine historical PBS accounting files replace `Job_Owner` with `ACCOUNT@scheduler`.
- Research notes and derived analysis records use `archive-host`, `/srv/qwen-artifacts` and `/home/USER` for private archive hosts, archive roots and personal analysis directories. These locations are illustrative; configure local storage before running an analysis command.
- Links to private organiser chat and planning documents have been removed. The clarification date and its Basic/Bonus classification remain stated.

Unedited originals are retained in a private archive outside this repository. `SHA256SUMS` files authenticate the public copies, including the anonymized accounting records. These edited records must not be described as byte-identical to the original scheduler output. The image source snapshots, reconstruction bundle, raw boundary commit and code patches remain unchanged. Historical competition storage paths remain where they describe the measured configuration or fixed build inputs; adapt them to the reproduction account.

## Local configuration

The GitHub repository secret `QWEN_ARTIFACT_ARCHIVE_ROOT` supplies the persistent archive destination. The GPU-clock analysis helper accepts the same name as a local environment variable. Local source-mirror tests may use `SGLANG_SOURCE_MIRROR`; tests that require that optional mirror skip when it is not configured. Connection settings and personal operational instructions are local files excluded from Git tracking. The historical task checklist is retained with the same publication redactions.

The submitted presentation is distributed separately. The experiment index preserves its Qwen slide references.

The cleanup has also been applied throughout the Git branch and tag history. Original commit IDs retained in experiment provenance still identify the recorded builds and runs; they have not been relabelled as builds of the rewritten commits. Recovery archives and the old-to-new commit map are stored privately outside this repository. GitHub-managed PR references, comments and historical Actions output require separate handling.

## Issue references

Issue and document navigation links use sanitized commit IDs. Additional historical snapshots referenced by those links are retained under `issue-history/<original-commit-id>` tags after applying the same redactions and updating their public checksum manifests. Author identities, timestamps, merge structure and numerical results are preserved; original source and image IDs recorded as measurement provenance remain unchanged.
