"""Privately unpack the exact existing wheel and reuse Komori's precision patch."""

import argparse
import subprocess
import zipfile
from pathlib import Path

from contract import (
    HEADER,
    ORIGINAL_HEADER_SHA256,
    PATCH_SHA256,
    WHEEL_SHA256,
    verify,
    verify_package,
)


def install(wheel: Path, destination: Path) -> None:
    verify(wheel, WHEEL_SHA256)
    patch = Path(__file__).with_name("sm90-late-topk-fp32.patch")
    verify(patch, PATCH_SHA256)
    if destination.exists():
        raise FileExistsError(
            f"Private provider destination must be new: {destination}"
        )
    destination.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            if (
                not target.is_relative_to(destination.resolve())
                or (entry.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Unsafe wheel member: " + entry.filename)
        archive.extractall(destination)
    package = destination / "deep_gemm"
    verify(package / HEADER, ORIGINAL_HEADER_SHA256)
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            "apply",
            "--check",
            str(patch),
        ],
        cwd=destination,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "apply", str(patch)],
        cwd=destination,
        check=True,
    )
    verify_package(package)
    print("PASS: privately staged exact wheel and unchanged Komori precision patch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    install(args.wheel.resolve(), args.destination.resolve())
