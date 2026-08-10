-- BigQuery Standard SQL.
--
-- This file is rendered by scripts/deploy-qwen-projections.py.  JSON columns
-- intentionally retain the closed nested contract objects and arrays; they
-- are not reduced to a small scalar subset.

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_source_track_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL,
  run_purpose STRING NOT NULL, repository_id STRING NOT NULL,
  source_track JSON NOT NULL, source_track_sha256 STRING NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_sif_provenance_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL,
  sif_artifact_sha256 STRING NOT NULL, source_commits JSON NOT NULL,
  score_source_tree JSON NOT NULL, recipe JSON NOT NULL,
  applied_patches JSON NOT NULL, base_oci JSON NOT NULL,
  lock_manifest JSON NOT NULL, build_command JSON NOT NULL,
  toolchain_observations JSON NOT NULL, embedded_manifest_identity STRING NOT NULL,
  runtime_files JSON NOT NULL, source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_runtime_launchers_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, role STRING NOT NULL,
  requested_path STRING NOT NULL, resolved_realpath STRING NOT NULL,
  size_bytes INT64 NOT NULL, sha256 STRING NOT NULL, argv JSON NOT NULL,
  outcome JSON NOT NULL, stdout_member STRING NOT NULL, stdout_b64 STRING NOT NULL,
  stderr_member STRING NOT NULL, stderr_b64 STRING NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_model_manifests_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, kind STRING NOT NULL,
  source_kind STRING NOT NULL, repository STRING, revision STRING,
  prebuilt_shards JSON, small_metadata JSON NOT NULL, manifest_sha256 STRING NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_dataset_manifests_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, kind STRING NOT NULL,
  repository STRING NOT NULL, full_revision STRING NOT NULL, relative_path STRING NOT NULL,
  size_bytes INT64 NOT NULL, content_sha256 STRING NOT NULL, manifest_sha256 STRING NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_cache_manifests_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, node_rank INT64 NOT NULL,
  semantic_manifest_sha256 STRING NOT NULL, members JSON NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_workload_manifests_raw_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, workload_kind STRING NOT NULL,
  dataset_manifest_sha256 STRING NOT NULL, warmup_mode STRING,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_workload_requests_raw_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, workload_kind STRING NOT NULL,
  ordinal INT64 NOT NULL, request JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_minimal_hardware_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, node_rank INT64 NOT NULL,
  hostname STRING NOT NULL, cpu_identity JSON NOT NULL, platform_settings JSON NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_node_launch_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, node_rank INT64 NOT NULL,
  global_rank INT64 NOT NULL, local_rank INT64 NOT NULL, size INT64 NOT NULL,
  hostname STRING NOT NULL, node_wrapper_argv JSON NOT NULL, singularity_argv JSON NOT NULL,
  sglang_argv JSON NOT NULL, cpu_affinity JSON NOT NULL, numa_mems STRING,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_topology_raw_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, node_rank INT64 NOT NULL,
  scope STRING NOT NULL, `index` INT64 NOT NULL, name STRING NOT NULL, uuid STRING NOT NULL,
  pci_bus_id STRING NOT NULL, memory_total_bytes INT64 NOT NULL, driver_version STRING NOT NULL,
  persistence_mode STRING NOT NULL, compute_mode STRING NOT NULL, power_limit_w BIGNUMERIC,
  max_sm_clock_mhz BIGNUMERIC, max_memory_clock_mhz BIGNUMERIC, mig_mode STRING NOT NULL,
  ecc_mode STRING NOT NULL, source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_environment_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, scope STRING NOT NULL,
  name STRING NOT NULL, value_base64 STRING NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_lifecycle_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, kind STRING NOT NULL,
  summary JSON NOT NULL, source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_node_lifecycle_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, node_rank INT64 NOT NULL,
  lifecycle JSON NOT NULL, source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_primary_results_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, invocation_id STRING NOT NULL,
  generation_wall_sec BIGNUMERIC NOT NULL, returned_count INT64 NOT NULL, error_count INT64 NOT NULL,
  observed_input_tokens INT64, observed_output_tokens INT64,
  ordered_output_token_ids_sha256 STRING, source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_phase_summaries_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, invocation_id STRING NOT NULL,
  kind STRING NOT NULL, rank0 JSON NOT NULL, nodes JSON NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_gpu_samples_raw_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, node_rank INT64 NOT NULL,
  sequence INT64 NOT NULL, clock_id STRING NOT NULL, monotonic_ns INT64 NOT NULL,
  wall_time_utc STRING NOT NULL, gpu_uuid STRING NOT NULL, logical_index INT64 NOT NULL,
  utilization_percent BIGNUMERIC, memory_used_bytes INT64, power_w BIGNUMERIC,
  temperature_c BIGNUMERIC, sm_clock_mhz BIGNUMERIC,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_gpu_phases_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, phase STRING NOT NULL,
  node_rank INT64 NOT NULL, gpu_uuid STRING NOT NULL, phase_duration_sec BIGNUMERIC NOT NULL,
  valid_sample_count INT64 NOT NULL, valid_utilization_duration_sec BIGNUMERIC NOT NULL,
  utilization_coverage BIGNUMERIC NOT NULL, utilization_mean BIGNUMERIC,
  utilization_p95 BIGNUMERIC, valid_power_duration_sec BIGNUMERIC NOT NULL,
  power_coverage BIGNUMERIC NOT NULL, energy_j BIGNUMERIC, power_mean BIGNUMERIC,
  valid_memory_sample_count INT64 NOT NULL, memory_max_bytes INT64,
  valid_temperature_sample_count INT64 NOT NULL, temperature_max_c BIGNUMERIC,
  valid_sm_clock_sample_count INT64 NOT NULL, sm_clock_min_mhz BIGNUMERIC,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_profile_hardware_raw_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, node_rank INT64 NOT NULL,
  hostname STRING NOT NULL, record_kind STRING NOT NULL, numa_node_id INT64,
  postrun_memory_available_bytes INT64, cpulist STRING, mem_total_bytes INT64,
  mem_free_bytes INT64, source_members JSON NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_profiler_events_raw_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, global_rank INT64 NOT NULL,
  activity STRING NOT NULL, event_category STRING NOT NULL, event_name STRING NOT NULL,
  call_count INT64 NOT NULL, inclusive_duration_sum_ns BIGNUMERIC NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_config_snapshots_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, config_type STRING NOT NULL,
  definition JSON NOT NULL, fields JSON NOT NULL, source_member STRING NOT NULL,
  source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_placement_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, node_rank INT64 NOT NULL,
  global_rank INT64 NOT NULL, server_args JSON NOT NULL, worker_ready JSON NOT NULL,
  topology JSON NOT NULL, source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_artifacts_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, role STRING NOT NULL,
  path STRING NOT NULL, size_bytes INT64 NOT NULL, sha256 STRING NOT NULL,
  run_purpose STRING NOT NULL, logical_artifact_name STRING NOT NULL,
  source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_mpi_network_results_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, benchmark STRING NOT NULL,
  process_count INT64 NOT NULL, message_bytes INT64 NOT NULL, iterations INT64 NOT NULL,
  time_min_us BIGNUMERIC, time_avg_us BIGNUMERIC, time_max_us BIGNUMERIC,
  bandwidth_mbytes_per_sec BIGNUMERIC, source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);

CREATE TABLE IF NOT EXISTS `__PROJECT__.__DATASET__.qwen_nccl_network_results_v2` (
  schema_version STRING NOT NULL, run_id STRING NOT NULL, collective STRING NOT NULL,
  message_bytes INT64 NOT NULL, element_count INT64 NOT NULL, data_type STRING NOT NULL,
  reduction_op STRING, buffer_mode STRING NOT NULL, mpi_process_count INT64 NOT NULL,
  gpu_count INT64 NOT NULL, time_us BIGNUMERIC NOT NULL,
  algorithm_bandwidth_gbps BIGNUMERIC NOT NULL, bus_bandwidth_gbps BIGNUMERIC NOT NULL,
  error_count INT64 NOT NULL, source_member STRING NOT NULL, source_sha256 STRING NOT NULL
);
