#!/usr/bin/env bash
set -euo pipefail

exec > >(tee -a /var/log/qwen-build-bootstrap.log) 2>&1
trap 'status=$?; printf "%s\n" "${status}" >/var/lib/qwen-build-bootstrap-status' EXIT
systemd-run --unit=qwen-build-expiration --on-active=6h /sbin/shutdown -h now
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y awscli buildah git jq skopeo squashfs-tools tar umoci zstd
touch /var/lib/qwen-build-ready
