"""Immutable existing SM90 provider contracts; no kernel implementation."""

import hashlib
from pathlib import Path

WHEEL_NAME = "sgl_deep_gemm-0.1.5-py3-none-manylinux2014_x86_64.whl"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/bb/36/93c38fd790f7c7ea8b19f0c4980603c64ba65cb4d712f638d2d225ceeef4/"
    + WHEEL_NAME
)
WHEEL_SHA256 = "e5e9f92cdf0515fe878bd81131bc25d2984e00c005824d2932af01a5e0d75261"
PATCH_SHA256 = "462d68230da3ab9a1e69ca85d9fa804b8edaf6a77bf0975726611a08fda6622b"
ORIGINAL_HEADER_SHA256 = (
    "a53d83cad76109588b3d5053be02877a0c2a47c38b27472269dc1319df62676d"
)
PATCHED_HEADER_SHA256 = (
    "5479807104bf1aeed4cc3ecc8ce57d2a835d68457f9cf4962ef3f15cc176e8d2"
)
NATIVE_LIBRARY_SHA256 = (
    "ab8dc0accf34adbd64037202c917250e1c084fe5417f086805036cef99ef2959"
)
VLLM_IMPORT_SHA256 = "5401bbf2556f072f210817888c3e13918773b6b91fdcb67870a235cf205fb73e"
HEADER = "include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
SOURCES = {
    "__init__.py": "b33e89deacdce241f01f5070d321918f5e5480e3e6d3af569678d4192db4f2a7",
    "cuda_helpers.py": "6afbea22f11214c1ff2431877ccff1f1b79b115c78d64f423df9681686700c5c",
    "mega/__init__.py": "a3fa9f4ab614df8af29a0fcd8e1571e9e05cb1e9f2c02bbb77f4c730264d0339",
    "utils/__init__.py": "bf31f6e9fc30482ef83a73d63fdb6ed4d46deec847b35c0f9d39816d9b923bb4",
    "utils/dist.py": "acf0fc25aa44ca5a54f2f757a39d462f43cf4ea1585d49f3feb59deecffefee3",
    "utils/layout.py": "36dc5ef744c613944c5f355ddd5cfd52ca3af9daae020aa15daad308f211d295",
    "utils/math.py": "fccc9c4380f75aec31adeadf04f79625fe8c5ea87c79e0841a2063d91a3773c5",
    "testing/__init__.py": "860e35af472f2e293883f8d06b928058f05fd5c32a4f108e2d774958ee198529",
    "testing/bench.py": "478d694e5fe95cca78a2c6ee53f7436fc906dc4851cf51ab88a5a9b617142332",
    "testing/numeric.py": "de92b73f28acb3c5f560fbb942161c88c7e849e6c52bb38fd20c67b8e4d747c7",
    "testing/utils.py": "20e5e4450cec023dabd8e92e32d9d6b7500991b9264d1068e942999aa52afec0",
}


def verify(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Unexpected source: {path}: {actual}")


def verify_package(package: Path) -> None:
    for name, expected in SOURCES.items():
        verify(package / name, expected)
    verify(package / "_C.so", NATIVE_LIBRARY_SHA256)
    verify(package / HEADER, PATCHED_HEADER_SHA256)
    if (package / "VERSION").read_text().strip() != "0.1.5":
        raise ValueError("Unsupported SM90 provider version")
