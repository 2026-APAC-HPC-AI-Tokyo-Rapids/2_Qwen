#!/usr/bin/env bash
set -euo pipefail

exec > >(tee -a /var/log/qwen-build-bootstrap.log) 2>&1
trap 'status=$?; printf "%s\n" "${status}" >/var/lib/qwen-build-bootstrap-status' EXIT
systemd-run --unit=qwen-build-expiration --on-active=6h /sbin/shutdown -h now
export DEBIAN_FRONTEND=noninteractive
# Run 40739933764 hit a mirror sync race on the stable Packages.xz URL.
# Fetch the signed release's exact hash and retry one refresh before failing.
if ! apt-get -o Acquire::By-Hash=force -o APT::Update::Error-Mode=any update; then
  sleep 30
  apt-get -o Acquire::By-Hash=force -o APT::Update::Error-Mode=any update
fi
apt-get install -y buildah curl git jq skopeo squashfs-tools tar umoci unzip zstd
curl --fail --location --retry 3 --retry-all-errors --retry-delay 2 https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip --output /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
aws --version
touch /var/lib/qwen-build-ready
