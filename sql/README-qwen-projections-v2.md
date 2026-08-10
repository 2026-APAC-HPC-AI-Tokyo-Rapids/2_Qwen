# Qwen v2 physical projections

`qwen_projection_tables_v2.sql` declares the 25 allowlisted physical tables.
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
Partial `insertErrors` fail the call.  Replaying the same closed row produces
the same insert ID.
