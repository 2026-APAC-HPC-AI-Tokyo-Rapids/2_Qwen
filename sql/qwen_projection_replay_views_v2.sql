-- BigQuery Standard SQL.
--
-- The ALTER statements are additive migration for tables created before the
-- replay metadata existed.  No table is dropped or rewritten.  The view
-- script uses stored SHA-256 metadata rather than relying on BigQuery JSON
-- equality or object-key ordering.

ALTER TABLE `__PROJECT__.__DATASET__.qwen_source_track_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_sif_provenance_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_runtime_launchers_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_model_manifests_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_dataset_manifests_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_cache_manifests_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_workload_manifests_raw_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_workload_requests_raw_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_minimal_hardware_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_node_launch_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_topology_raw_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_environment_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_lifecycle_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_node_lifecycle_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_primary_results_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_phase_summaries_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_gpu_samples_raw_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_gpu_phases_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_profile_hardware_raw_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_profiler_events_raw_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_config_snapshots_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_placement_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_artifacts_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_mpi_network_results_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;
ALTER TABLE `__PROJECT__.__DATASET__.qwen_nccl_network_results_v2`
  ADD COLUMN IF NOT EXISTS _row_grain_key STRING,
  ADD COLUMN IF NOT EXISTS _payload_sha256 STRING;

-- Each logical view keeps one exact payload per grain.  The conflict view
-- retains every payload for grains with more than one payload hash.
BEGIN
DECLARE relation STRING;
FOR record IN (
  SELECT relation FROM UNNEST([
    'qwen_source_track_v2', 'qwen_sif_provenance_v2',
    'qwen_runtime_launchers_v2', 'qwen_model_manifests_v2',
    'qwen_dataset_manifests_v2', 'qwen_cache_manifests_v2',
    'qwen_workload_manifests_raw_v2', 'qwen_workload_requests_raw_v2',
    'qwen_minimal_hardware_v2', 'qwen_node_launch_v2',
    'qwen_topology_raw_v2', 'qwen_environment_v2', 'qwen_lifecycle_v2',
    'qwen_node_lifecycle_v2', 'qwen_primary_results_v2',
    'qwen_phase_summaries_v2', 'qwen_gpu_samples_raw_v2',
    'qwen_gpu_phases_v2', 'qwen_profile_hardware_raw_v2',
    'qwen_profiler_events_raw_v2', 'qwen_config_snapshots_v2',
    'qwen_placement_v2', 'qwen_artifacts_v2',
    'qwen_mpi_network_results_v2', 'qwen_nccl_network_results_v2'
  ]) AS relation
)
DO
  EXECUTE IMMEDIATE FORMAT(
    '''CREATE OR REPLACE VIEW `__PROJECT__.__DATASET__.%s_conflicts` AS
       WITH conflicted AS (
         SELECT _row_grain_key
         FROM `__PROJECT__.__DATASET__.%s`
         WHERE _row_grain_key IS NOT NULL AND _payload_sha256 IS NOT NULL
         GROUP BY _row_grain_key
         HAVING COUNT(DISTINCT _payload_sha256) > 1
       ), ranked AS (
         SELECT source_table.*,
           ROW_NUMBER() OVER (
             PARTITION BY _row_grain_key, _payload_sha256
             ORDER BY source_sha256, source_member
           ) AS _replay_row_number
         FROM `__PROJECT__.__DATASET__.%s` AS source_table
         WHERE _row_grain_key IN (SELECT _row_grain_key FROM conflicted)
       )
       SELECT * EXCEPT (_replay_row_number)
       FROM ranked
       WHERE _replay_row_number = 1''', relation, relation, relation
  );
  EXECUTE IMMEDIATE FORMAT(
    '''CREATE OR REPLACE VIEW `__PROJECT__.__DATASET__.%s_logical` AS
       WITH conflicted AS (
         SELECT _row_grain_key
         FROM `__PROJECT__.__DATASET__.%s`
         WHERE _row_grain_key IS NOT NULL AND _payload_sha256 IS NOT NULL
         GROUP BY _row_grain_key
         HAVING COUNT(DISTINCT _payload_sha256) > 1
       ), ranked AS (
         SELECT source_table.*,
           ROW_NUMBER() OVER (
             PARTITION BY _row_grain_key, _payload_sha256
             ORDER BY source_sha256, source_member
           ) AS _replay_row_number
         FROM `__PROJECT__.__DATASET__.%s` AS source_table
         WHERE _row_grain_key IS NOT NULL AND _payload_sha256 IS NOT NULL
       )
       SELECT * EXCEPT (_replay_row_number)
       FROM ranked
       WHERE _replay_row_number = 1
         AND _row_grain_key NOT IN (SELECT _row_grain_key FROM conflicted)''',
    relation, relation, relation
  );
END FOR;
END;
