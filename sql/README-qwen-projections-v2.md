# Qwen v2 physical projections

`qwen_projection_tables_v2.sql` declares the 25 allowlisted physical tables.
`qwen_projection_replay_views_v2.sql` additively migrates replay metadata and
creates one `_logical` and one `_conflicts` view per table.
The nested and repeated contract records use BigQuery `JSON` columns so the
source schema is retained; they are not reduced to scalar summaries.

Render a dry-run DDL without credentials or network access:

```bash
python scripts/deploy-qwen-projections.py \
  --project apac-hpc-ai-tokyo-rapids-26 \
  --dataset qwen_experiment_metrics > /tmp/qwen-projections.sql
```

Applying requires an explicit `--apply` and a local authenticated `bq` CLI.
The physical publisher is intentionally separate from the existing raw-fact
publisher.  Its integration point is:

```python
from qwen_experiment_metrics.physical_publish import (
    PhysicalPublisherConfig,
    publish_projection_batch,
)

publish_projection_batch(
    projection_batch,  # already deduplicated; conflicts fail closed
    PhysicalPublisherConfig(project, dataset, location),
)
```

Rows are sent with `skipInvalidRows=false`, `ignoreUnknownValues=false`,
deterministic `insertId`s, and at most 500 rows per `insertAll` request.
Each row also stores publisher-owned `_row_grain_key` and `_payload_sha256`.
The logical view removes exact replay duplicates and excludes conflicting
grains; the conflicts view retains every differing payload. Partial
`insertErrors` fail the call. Replaying the same closed row produces the same
insert ID and payload hashes.
