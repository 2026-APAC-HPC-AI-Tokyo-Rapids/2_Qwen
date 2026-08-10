-- BigQuery Standard SQL.
--
-- One row per distinct conflicting value.  Consumers must not choose an
-- arbitrary value from this relation.
CREATE OR REPLACE VIEW `apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_run_fact_conflicts_v2` AS
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
conflicting_keys AS (
  SELECT
    run_id,
    fact_key
  FROM distinct_facts
  GROUP BY run_id, fact_key
  HAVING COUNT(*) > 1
)
SELECT
  facts.run_id AS `run`,
  facts.fact_key AS `key`,
  facts.fact_value AS `value`
FROM distinct_facts AS facts
JOIN conflicting_keys
  USING (run_id, fact_key);
