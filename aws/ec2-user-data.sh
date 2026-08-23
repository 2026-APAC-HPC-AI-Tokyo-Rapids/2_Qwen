#!/usr/bin/env bash
set -euo pipefail

exec > >(tee -a /var/log/qwen-build-bootstrap.log) 2>&1
trap 'status=$?; printf "%s\n" "${status}" >/var/lib/qwen-build-bootstrap-status' EXIT
systemd-run --unit=qwen-build-expiration --on-active=6h /sbin/shutdown -h now
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y buildah curl git jq skopeo squashfs-tools tar umoci unzip zstd
curl --fail --location --retry 3 --retry-all-errors --retry-delay 2 https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip --output /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
aws --version
touch /var/lib/qwen-build-ready
