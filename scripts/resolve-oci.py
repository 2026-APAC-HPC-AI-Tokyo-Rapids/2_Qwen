#!/usr/bin/env python3
"""Resolve a Docker Hub tag to the immutable linux/amd64 image manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

INDEX_TYPES = {
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
}
ACCEPT = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.v2+json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("--raw-output", type=Path, required=True)
    arguments = parser.parse_args()
    repository, tag = _parse_reference(arguments.reference)
    token_url = (
        "https://auth.docker.io/token?"
        + urllib.parse.urlencode(
            {
                "service": "registry.docker.io",
                "scope": f"repository:{repository}:pull",
            }
        )
    )
    token = json.loads(_get(token_url, {}).decode())["token"]
    headers = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}
    raw = _get(_manifest_url(repository, tag), headers)
    document = json.loads(raw)
    if document.get("mediaType") in INDEX_TYPES:
        choices = [
            item
            for item in document.get("manifests", [])
            if item.get("platform") == {"architecture": "amd64", "os": "linux"}
        ]
        if len(choices) != 1:
            raise RuntimeError("OCI index does not have one linux/amd64 manifest")
        raw = _get(_manifest_url(repository, choices[0]["digest"]), headers)
        document = json.loads(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    config = document.get("config", {}).get("digest")
    layers = [item.get("digest") for item in document.get("layers", [])]
    if not isinstance(config, str) or not layers or any(
        not isinstance(item, str) for item in layers
    ):
        raise RuntimeError("OCI image manifest is incomplete")
    image_name = arguments.reference.split("@", 1)[0].rsplit(":", 1)[0]
    arguments.raw_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.raw_output.write_bytes(raw)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    print(
        json.dumps(
            {
                "reference": f"{image_name}@{digest}",
                "inspect_raw": {
                    "path": "sif-build/oci-inspect.raw.json",
                    "size_bytes": len(raw),
                    "sha256": raw_sha256,
                },
                "manifest_digest": digest,
                "config_digest": config,
                "layer_digests": layers,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def _parse_reference(reference: str) -> tuple[str, str]:
    if "/" not in reference:
        raise ValueError("expected a Docker Hub repository reference")
    if "@" in reference:
        image, tag = reference.split("@", 1)
        if not image or not tag.startswith("sha256:"):
            raise ValueError("invalid digest-pinned reference")
    else:
        image, separator, tag = reference.rpartition(":")
        if not separator or not image or not tag:
            raise ValueError("expected repository:tag")
    if image.startswith("docker.io/"):
        image = image.removeprefix("docker.io/")
    elif "." in image.split("/", 1)[0]:
        raise ValueError("only Docker Hub references are supported")
    return image, tag


def _manifest_url(repository: str, reference: str) -> str:
    return f"https://registry-1.docker.io/v2/{repository}/manifests/{reference}"


def _get(url: str, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


if __name__ == "__main__":
    raise SystemExit(main())
