#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)

    upload = subparsers.add_parser("upload")
    upload.add_argument("source", type=Path)
    upload.add_argument("bucket")
    upload.add_argument("key")
    upload.add_argument("--sha256")

    download = subparsers.add_parser("download")
    download.add_argument("bucket")
    download.add_argument("key")
    download.add_argument("destination", type=Path)

    args = parser.parse_args()
    client = boto3.client("s3")
    transfer = TransferConfig(multipart_threshold=64 * 1024**2, multipart_chunksize=64 * 1024**2, max_concurrency=8, use_threads=True)

    if args.operation == "upload":
        extra_args = {"Metadata": {"sha256": args.sha256}} if args.sha256 else None
        client.upload_file(str(args.source), args.bucket, args.key, ExtraArgs=extra_args, Config=transfer)
    else:
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(args.bucket, args.key, str(args.destination), Config=transfer)


if __name__ == "__main__":
    main()
