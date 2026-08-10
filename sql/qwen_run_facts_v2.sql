-- BigQuery Standard SQL.
--
-- Exact duplicate rows are collapsed.  A (run,key) with multiple distinct
-- values is excluded from this canonical relation and retained losslessly by
-- qwen_run_fact_conflicts_v2.
CREATE OR REPLACE VIEW `apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_run_facts_v2` AS
WITH distinct_facts AS (
  SELECT DISTINCT
    CAST(`run` AS STRING) AS run_id,
    CAST(`key` AS STRING) AS fact_key,
    CAST(`value` AS STRING) AS fact_value
  FROM `apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_metrics`
  WHERE `run` IS NOT NULL
    AND `key` IS NOT NULL
    AND `value` IS NOT NULL
),
key_cardinality AS (
  SELECT
    run_id,
    fact_key,
    COUNT(*) AS distinct_value_count
  FROM distinct_facts
  GROUP BY run_id, fact_key
)
SELECT
  facts.run_id AS `run`,
  facts.fact_key AS `key`,
  facts.fact_value AS `value`
FROM distinct_facts AS facts
JOIN key_cardinality AS cardinality
  USING (run_id, fact_key)
WHERE cardinality.distinct_value_count = 1;
