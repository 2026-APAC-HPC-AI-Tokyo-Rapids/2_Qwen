"""Check pinned native staging and runtime provenance without GPU/Torch imports."""

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
HELPER = ROOT / "containers/firmus-sm90-megamoe"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(HELPER))
    spec = importlib.util.spec_from_file_location("contract", HELPER / "contract.py")
    contract = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "contract", contract)
    spec.loader.exec_module(contract)
    spec = importlib.util.spec_from_file_location("native_stager", HELPER / "stage_native.py")
    stager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stager)
    return contract, stager


def archive(path, files, *, link=False):
    with tarfile.open(path, "w:gz") as output:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = "/outside"
                output.addfile(member)
            else:
                member.size = len(payload)
                output.addfile(member, io.BytesIO(payload))


def test_stage_real_archives_require_hash_before_any_write(tmp_path, modules):
    _, stage = modules
    bad = tmp_path / "bad"
    bad.write_bytes(b"not the pinned public source")
    destination = tmp_path / "stage"
    with pytest.raises(ValueError, match="Unexpected source"):
        stage.stage(bad, bad, destination)
    assert not destination.exists()


@pytest.mark.parametrize("name,link", [("root/csrc/../escape", False), ("root/csrc/link", True)])
def test_source_tree_rejects_traversal_and_links(tmp_path, modules, name, link):
    _, stage = modules
    source = tmp_path / "source.tar.gz"
    archive(source, {name: b"bad"}, link=link)
    with pytest.raises(ValueError):
        stage.copy_tree(source, "root/csrc/", tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_source_tree_copies_only_selected_regular_files(tmp_path, modules):
    _, stage = modules
    source = tmp_path / "source.tar.gz"
    archive(source, {"root/csrc/api.cpp": b"expected", "root/other.txt": b"excluded"})
    stage.copy_tree(source, "root/csrc/", tmp_path / "out")
    assert (tmp_path / "out/api.cpp").read_bytes() == b"expected"
    assert list((tmp_path / "out").iterdir()) == [tmp_path / "out/api.cpp"]


def runtime_package(tmp_path, contract, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    code = b"exact retained Python source"
    header = b"exact retained kernel math"
    monkeypatch.setattr(contract, "SOURCES", {"__init__.py": hashlib.sha256(code).hexdigest()})
    monkeypatch.setattr(contract, "HEADER", "kernel.cuh")
    monkeypatch.setattr(contract, "PATCHED_HEADER_SHA256", hashlib.sha256(header).hexdigest())
    (package / "__init__.py").write_bytes(code)
    (package / "kernel.cuh").write_bytes(header)
    (package / "VERSION").write_text("0.1.5")
    (package / "_C.so").write_bytes(b"rebuilt private library")
    report = {
        "schema": "apac-sm90-native-build/v1",
        "source_commit": contract.NATIVE_SOURCE_COMMIT,
        "source_archive_sha256": contract.NATIVE_SOURCE_SHA256,
        "fmt_commit": contract.FMT_COMMIT,
        "fmt_archive_sha256": contract.FMT_SOURCE_SHA256,
        "original_native_sha256": contract.NATIVE_LIBRARY_SHA256,
        "native_sha256": hashlib.sha256((package / "_C.so").read_bytes()).hexdigest(),
        "torch": "2.13.0+cu130",
        "tvm_ffi": "0.1.11",
        "cpu_bridge_probe": True,
    }
    (package / "native-build.json").write_text(json.dumps(report))
    return package, report


def test_rebuilt_runtime_hash_and_provenance_pass(tmp_path, modules, monkeypatch):
    contract, _ = modules
    package, report = runtime_package(tmp_path, contract, monkeypatch)
    assert contract.verify_runtime_package(package) == report


@pytest.mark.parametrize("change", ["binary", "python", "kernel", "torch", "probe", "prebuilt", "missing"])
def test_rebuilt_runtime_rejects_changed_payload_or_unverified_build(tmp_path, modules, monkeypatch, change):
    contract, _ = modules
    package, report = runtime_package(tmp_path, contract, monkeypatch)
    if change in ["binary", "python", "kernel"]:
        name = {"binary": "_C.so", "python": "__init__.py", "kernel": "kernel.cuh"}[change]
        (package / name).write_bytes(b"changed")
    elif change == "missing":
        (package / "native-build.json").unlink()
    else:
        if change == "torch":
            report["torch"] = "2.12.0"
        elif change == "probe":
            report["cpu_bridge_probe"] = False
        else:
            report["native_sha256"] = contract.NATIVE_LIBRARY_SHA256
        (package / "native-build.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, FileNotFoundError)):
        contract.verify_runtime_package(package)
