-- BigQuery Standard SQL.
--
-- Fixed-width consumer projection over conflict-free scalar facts.  JSON
-- values remain text because the raw Fact owns the complete source object.
-- Missing and conflicted keys are NULL; no placeholders are synthesized.
CREATE OR REPLACE VIEW `apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_run_consumer_v2` AS
WITH raw_runs AS (
  SELECT DISTINCT CAST(`run` AS STRING) AS run_id
  FROM `apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_metrics`
  WHERE `run` IS NOT NULL
),
conflict_summary AS (
  SELECT
    `run` AS run_id,
    COUNT(DISTINCT `key`) AS conflict_key_count
  FROM `apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_run_fact_conflicts_v2`
  GROUP BY run_id
),
pivoted AS (
  SELECT
    `run` AS run_id,
    MAX(IF(`key` = 'run.identity.run_id', `value`, NULL)) AS identity_run_id,
    MAX(IF(`key` = 'run.identity.run_purpose', `value`, NULL)) AS run_purpose,
    MAX(IF(`key` = 'run.identity.repository_id', `value`, NULL)) AS repository_id,
    MAX(IF(`key` = 'run.identity.instrumentation_bundle_sha256', `value`, NULL)) AS instrumentation_bundle_sha256,
    MAX(IF(`key` = 'source.track.head_commit', `value`, NULL)) AS source_track_head_commit,
    MAX(IF(`key` = 'source.track.score_reference_commit', `value`, NULL)) AS source_track_score_reference_commit,
    MAX(IF(`key` = 'source.track.profile_base_commit', `value`, NULL)) AS source_track_profile_base_commit,
    MAX(IF(`key` = 'source.track.manifest_sha256', `value`, NULL)) AS source_track_manifest_sha256,
    MAX(IF(`key` = 'source.track.source_sha256', `value`, NULL)) AS source_track_source_sha256,
    SAFE_CAST(MAX(IF(`key` = 'collection_complete', `value`, NULL)) AS BOOL) AS collection_complete,

    SAFE_CAST(MAX(IF(`key` = 'scheduler.request.nodes', `value`, NULL)) AS INT64) AS execution_nodes,
    SAFE_CAST(MAX(IF(`key` = 'scheduler.request.mpi_processes', `value`, NULL)) AS INT64) AS execution_mpi_processes,
    SAFE_CAST(MAX(IF(`key` = 'scheduler.request.gpus_per_node', `value`, NULL)) AS INT64) AS execution_gpus_per_node,
    SAFE_CAST(MAX(IF(`key` = 'scheduler.request.tensor_parallel_size', `value`, NULL)) AS INT64) AS execution_tensor_parallel_size,
    MAX(IF(`key` = 'scheduler.request.resource_group', `value`, NULL)) AS execution_resource_group,
    MAX(IF(`key` = 'scheduler.request.transport_policy', `value`, NULL)) AS execution_transport_policy,
    MAX(IF(`key` = 'scheduler.request.advertise_source', `value`, NULL)) AS execution_advertise_source,
    SAFE_CAST(MAX(IF(`key` = 'scheduler.request.sglang_port', `value`, NULL)) AS INT64) AS execution_sglang_port,
    SAFE_CAST(MAX(IF(`key` = 'scheduler.request.node_agent_port', `value`, NULL)) AS INT64) AS execution_node_agent_port,
    SAFE_CAST(MAX(IF(`key` = 'scheduler.request.rendezvous_timeout_sec', `value`, NULL)) AS INT64) AS execution_rendezvous_timeout_sec,
    SAFE_CAST(MAX(IF(`key` = 'scheduler.request.ready_timeout_sec', `value`, NULL)) AS INT64) AS execution_ready_timeout_sec,

    MAX(IF(`key` = 'artifact.sif_manifest_sha256', `value`, NULL)) AS artifact_sif_manifest_sha256,
    MAX(IF(`key` = 'artifact.model.manifest_sha256', `value`, NULL)) AS artifact_model_manifest_sha256,
    MAX(IF(`key` = 'artifact.dataset.manifest_sha256', `value`, NULL)) AS artifact_dataset_manifest_sha256,
    MAX(IF(`key` = 'source.sglang.commit', `value`, NULL)) AS source_sglang_commit,

    MAX(IF(`key` = 'run.stage.benchmark.launch', `value`, NULL)) AS stage_benchmark_launch_json,
    MAX(IF(`key` = 'run.stage.benchmark.termination', `value`, NULL)) AS stage_benchmark_termination_json,
    SAFE_CAST(MAX(IF(`key` = 'perf.stage.benchmark.wall_time_sec', `value`, NULL)) AS FLOAT64) AS stage_benchmark_wall_time_sec,
    MAX(IF(`key` = 'run.stage.network_imb_mpi1.launch', `value`, NULL)) AS stage_network_imb_mpi1_launch_json,
    MAX(IF(`key` = 'run.stage.network_imb_mpi1.termination', `value`, NULL)) AS stage_network_imb_mpi1_termination_json,
    SAFE_CAST(MAX(IF(`key` = 'perf.stage.network_imb_mpi1.wall_time_sec', `value`, NULL)) AS FLOAT64) AS stage_network_imb_mpi1_wall_time_sec,
    MAX(IF(`key` = 'run.stage.network_nccl_all_reduce.launch', `value`, NULL)) AS stage_network_nccl_all_reduce_launch_json,
    MAX(IF(`key` = 'run.stage.network_nccl_all_reduce.termination', `value`, NULL)) AS stage_network_nccl_all_reduce_termination_json,
    SAFE_CAST(MAX(IF(`key` = 'perf.stage.network_nccl_all_reduce.wall_time_sec', `value`, NULL)) AS FLOAT64) AS stage_network_nccl_all_reduce_wall_time_sec,
    MAX(IF(`key` = 'run.stage.network_nccl_all_gather.launch', `value`, NULL)) AS stage_network_nccl_all_gather_launch_json,
    MAX(IF(`key` = 'run.stage.network_nccl_all_gather.termination', `value`, NULL)) AS stage_network_nccl_all_gather_termination_json,
    SAFE_CAST(MAX(IF(`key` = 'perf.stage.network_nccl_all_gather.wall_time_sec', `value`, NULL)) AS FLOAT64) AS stage_network_nccl_all_gather_wall_time_sec,
    MAX(IF(`key` = 'run.stage.network_nccl_reduce_scatter.launch', `value`, NULL)) AS stage_network_nccl_reduce_scatter_launch_json,
    MAX(IF(`key` = 'run.stage.network_nccl_reduce_scatter.termination', `value`, NULL)) AS stage_network_nccl_reduce_scatter_termination_json,
    SAFE_CAST(MAX(IF(`key` = 'perf.stage.network_nccl_reduce_scatter.wall_time_sec', `value`, NULL)) AS FLOAT64) AS stage_network_nccl_reduce_scatter_wall_time_sec,

    SAFE_CAST(MAX(IF(`key` = 'perf.benchmark.generation.wall_time_sec', `value`, NULL)) AS FLOAT64) AS generation_wall_time_sec,
    SAFE_CAST(MAX(IF(`key` = 'result.request.returned_count', `value`, NULL)) AS INT64) AS result_returned_count,
    SAFE_CAST(MAX(IF(`key` = 'result.request.error_count', `value`, NULL)) AS INT64) AS result_error_count,
    SAFE_CAST(MAX(IF(`key` = 'result.token.input.total', `value`, NULL)) AS INT64) AS result_input_tokens,
    SAFE_CAST(MAX(IF(`key` = 'result.token.output.total', `value`, NULL)) AS INT64) AS result_output_tokens,
    MAX(IF(`key` = 'result.token.output.ordered_token_ids_sha256', `value`, NULL)) AS result_ordered_output_identity
  FROM `apac-hpc-ai-tokyo-rapids-26.qwen_experiment_metrics.qwen_run_facts_v2`
  GROUP BY run_id
)
SELECT
  runs.run_id,
  pivoted.identity_run_id,
  pivoted.run_purpose,
  pivoted.repository_id,
  pivoted.instrumentation_bundle_sha256,
  pivoted.source_track_head_commit,
  pivoted.source_track_score_reference_commit,
  pivoted.source_track_profile_base_commit,
  pivoted.source_track_manifest_sha256,
  pivoted.source_track_source_sha256,
  pivoted.collection_complete,
  COALESCE(conflicts.conflict_key_count, 0) > 0 AS has_conflicts,
  COALESCE(conflicts.conflict_key_count, 0) AS conflict_key_count,
  pivoted.execution_nodes,
  pivoted.execution_mpi_processes,
  pivoted.execution_gpus_per_node,
  pivoted.execution_tensor_parallel_size,
  pivoted.execution_resource_group,
  pivoted.execution_transport_policy,
  pivoted.execution_advertise_source,
  pivoted.execution_sglang_port,
  pivoted.execution_node_agent_port,
  pivoted.execution_rendezvous_timeout_sec,
  pivoted.execution_ready_timeout_sec,
  pivoted.artifact_sif_manifest_sha256,
  pivoted.artifact_model_manifest_sha256,
  pivoted.artifact_dataset_manifest_sha256,
  pivoted.source_sglang_commit,
  pivoted.stage_benchmark_launch_json,
  pivoted.stage_benchmark_termination_json,
  pivoted.stage_benchmark_wall_time_sec,
  JSON_VALUE(pivoted.stage_benchmark_termination_json, '$.kind') AS benchmark_termination_kind,
  SAFE_CAST(JSON_VALUE(pivoted.stage_benchmark_termination_json, '$.code') AS INT64) AS benchmark_exit_code,
  SAFE_CAST(JSON_VALUE(pivoted.stage_benchmark_termination_json, '$.signal') AS INT64) AS benchmark_signal,
  pivoted.stage_network_imb_mpi1_launch_json,
  pivoted.stage_network_imb_mpi1_termination_json,
  pivoted.stage_network_imb_mpi1_wall_time_sec,
  pivoted.stage_network_nccl_all_reduce_launch_json,
  pivoted.stage_network_nccl_all_reduce_termination_json,
  pivoted.stage_network_nccl_all_reduce_wall_time_sec,
  pivoted.stage_network_nccl_all_gather_launch_json,
  pivoted.stage_network_nccl_all_gather_termination_json,
  pivoted.stage_network_nccl_all_gather_wall_time_sec,
  pivoted.stage_network_nccl_reduce_scatter_launch_json,
  pivoted.stage_network_nccl_reduce_scatter_termination_json,
  pivoted.stage_network_nccl_reduce_scatter_wall_time_sec,
  pivoted.generation_wall_time_sec,
  pivoted.result_returned_count,
  pivoted.result_error_count,
  pivoted.result_input_tokens,
  pivoted.result_output_tokens,
  pivoted.result_ordered_output_identity,
  SAFE_DIVIDE(pivoted.result_output_tokens, pivoted.generation_wall_time_sec) AS result_output_tokens_per_sec
FROM raw_runs AS runs
LEFT JOIN pivoted
  USING (run_id)
LEFT JOIN conflict_summary AS conflicts
  USING (run_id);
