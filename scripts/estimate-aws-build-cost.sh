#!/usr/bin/env bash
set -euo pipefail

: "${BUILD_ELAPSED_SECONDS:?build elapsed seconds are required}"
: "${COST_OUTPUT:?cost output path is required}"

[[ "${BUILD_ELAPSED_SECONDS}" =~ ^[0-9]+$ ]]
container_size_bytes="${CONTAINER_SIZE_BYTES:-0}"
external_download_count="${EXTERNAL_DOWNLOAD_COUNT:-0}"
[[ "${container_size_bytes}" =~ ^[0-9]+$ && "${external_download_count}" =~ ^[0-9]+$ ]]

billable_seconds="${BUILD_ELAPSED_SECONDS}"
if (( billable_seconds < 60 )); then billable_seconds=60; fi

ec2_hourly=0.504
gp3_capacity_gb_month=0.096
gp3_extra_iops_month=0.006
gp3_extra_throughput_month=0.048
public_ipv4_hourly=0.005
s3_tier1_request=0.000005
internet_egress_gb=0.120

costs="$(awk -v seconds="${billable_seconds}" -v size="${container_size_bytes}" -v downloads="${external_download_count}" -v ec2="${ec2_hourly}" -v capacity="${gp3_capacity_gb_month}" -v iops="${gp3_extra_iops_month}" -v throughput="${gp3_extra_throughput_month}" -v ipv4="${public_ipv4_hourly}" -v request="${s3_tier1_request}" -v egress="${internet_egress_gb}" 'BEGIN {hours=seconds/3600; ec2_cost=hours*ec2; ebs_cost=seconds/(30*86400)*(300*capacity+3000*iops+375*throughput); ipv4_cost=hours*ipv4; part_count=(size==0 ? 0 : int((size+8388607)/8388608)); request_cost=(part_count+10)*request; external_gb=size/1000000000; external_cost=external_gb*downloads*egress; build_cost=ec2_cost+ebs_cost+ipv4_cost+request_cost; printf "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %d", hours,ec2_cost,ebs_cost,ipv4_cost,request_cost,build_cost,external_gb,external_cost,part_count}')"
read -r billable_hours ec2_cost ebs_cost ipv4_cost request_cost build_cost external_gb external_cost multipart_parts <<<"${costs}"
total_cost="$(awk -v build="${build_cost}" -v external="${external_cost}" 'BEGIN {printf "%.9f", build+external}')"

jq -n --arg calculated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg region ap-southeast-1 --arg instance_type m7i.2xlarge --argjson elapsed_seconds "${BUILD_ELAPSED_SECONDS}" --argjson billable_seconds "${billable_seconds}" --argjson billable_hours "${billable_hours}" --argjson container_size_bytes "${container_size_bytes}" --argjson external_download_count "${external_download_count}" --argjson multipart_parts "${multipart_parts}" --argjson ec2_usd "${ec2_cost}" --argjson ebs_usd "${ebs_cost}" --argjson public_ipv4_usd "${ipv4_cost}" --argjson s3_requests_usd "${request_cost}" --argjson build_usd "${build_cost}" --argjson external_transfer_gb "${external_gb}" --argjson external_transfer_usd "${external_cost}" --argjson total_usd "${total_cost}" '{schema_version:"aws-build-cost-estimate/v1",calculated_at:$calculated_at,region:$region,instance:{type:$instance_type,elapsed_seconds:$elapsed_seconds,billable_seconds:$billable_seconds,billable_hours:$billable_hours},artifact:{size_bytes:$container_size_bytes},distribution:{external_download_count:$external_download_count,data_transfer_gb_per_download:$external_transfer_gb},estimated_cost_usd:{ec2:$ec2_usd,ebs_gp3:$ebs_usd,public_ipv4:$public_ipv4_usd,s3_tier1_requests:$s3_requests_usd,build_subtotal:$build_usd,external_data_transfer:$external_transfer_usd,total:$total_usd},rates:{effective_date:"2026-08-24",ec2_per_hour:0.504,gp3_capacity_per_gb_month:0.096,gp3_extra_iops_month:0.006,gp3_extra_throughput_per_mibps_month:0.048,public_ipv4_per_hour:0.005,s3_tier1_per_request:0.000005,internet_egress_per_gb_first_10tb_beyond_free_tier:0.120},assumptions:["300 GiB gp3, 6000 IOPS, 500 MiB/s", "30-day EBS month and 60-second minimum", "S3 artifact storage is excluded", "internet egress can be zero while the account-wide 100 GB monthly free tier remains"]}' >"${COST_OUTPUT}"
jq . "${COST_OUTPUT}"
