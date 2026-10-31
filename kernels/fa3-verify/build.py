"""AOT-build the bounded SM90a verify companion without initializing CUDA."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

PINS = {
    "fa3": ("sgl-project/sgl-attn", "f89bc2306632d1ec5f97b014dded4254f5b4a907", "418b5681584dc3efff496a1cab5ffd58d2728d89dcfe0ea16e6985d6ef35c68c"),
    "cutlass": ("NVIDIA/cutlass", "57e3cfb47a2d9e0d46eb6335c3dc411498efa198", "09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5"),
}
SM90_OPTIN_BYTES = 227 * 1024
# Two co-resident CTAs need both dynamic allocations plus the 1 KiB per-CTA reservation in 228 KiB.
SM90_SM_BYTES = 228 * 1024


def candidates(root):
    return json.loads((root / "candidates.json").read_text())


def generate_dispatch(root, work):
    """Generate C++ dispatch and unique main-kernel TUs from the sole table."""
    rows = candidates(root)
    kernels = {}
    entries = []
    for row in rows:
        key = (row["block_n"], row["stages"], row["splits"] > 1, row.get("merge", False), row.get("ctas", 1), row.get("heads", 1))
        symbol = f"verify_n{key[0]}_s{key[1]}_split{int(key[2])}_merge{int(key[3])}" + (f"_c{key[4]}" if key[4] != 1 else "") + (f"_h{key[5]}" if key[5] != 1 else "")
        kernels[key] = symbol
        entries.append(f'    {{"{row["name"]}", {row["block_n"]}, {row["stages"]}, {row["splits"]}, {str(row["output"] == "fp8").lower()}, {str(row.get("merge", False)).lower()}, {str(row.get("uneven", False)).lower()}, {key[4]}, {key[5]}, {symbol}}},')
    declarations = [f"void {symbol}(Flash_fwd_params&, cudaStream_t, int*);" for symbol in kernels.values()]
    (work / "candidates.h").write_text("\n".join([*declarations, "static const Candidate candidates[] = {", *entries, "};", ""]))
    tasks = []
    for (block_n, stages, split, merge, ctas, heads), symbol in kernels.items():
        source = work / (symbol + ".cu")
        source.write_text(f'#define VERIFY_NAME {symbol}\n#define VERIFY_BLOCK_N {block_n}\n#define VERIFY_STAGES {stages}\n#define VERIFY_SPLIT {str(split).lower()}\n#define VERIFY_MERGE {str(merge).lower()}\n#define VERIFY_CTAS {ctas}\n#define VERIFY_HEADS {heads}\n#include "p128split.cu"\n')
        tasks.append((source, symbol, True))
    return tasks


def generate_prefill_dispatch(root, work):
    """Independent Prefill candidate table; Decode dispatch is unchanged."""
    rows = json.loads((root / "prefill-candidates.json").read_text())
    tasks, declarations, entries = [], [], []
    for row in rows:
        symbol = "prefill_" + row["name"].replace("-", "_")
        if row.get("retired"):
            # Keep published IDs as uncallable slots; do not build retired TUs.
            entries.append(f'    {{"{row["name"]}", false, false, nullptr, 0}},')
            continue
        if row["name"] == "control":
            symbol = "prefill_fixed"
        else:
            source = work / (symbol + ".cu")
            source.write_text(f'#define PREFILL_NAME {symbol}\n#define PREFILL_N {row["block_n"]}\n#define PREFILL_VARIANT {row["variant"]}\n#include "prefill_variant.cu"\n')
            tasks.append((source, symbol, True))
        declarations.append(f"void {symbol}(Flash_fwd_params&, cudaStream_t, int*);")
        entries.append(f'    {{"{row["name"]}", {str(row["byte_image"]).lower()}, {str(row.get("varlen", False)).lower()}, {symbol}, {row.get("native_v_n", 0)}}},')
    (work / "prefill-candidates.h").write_text("\n".join([*declarations, "static const PrefillCandidate prefill_candidates[] = {", *entries, "};", ""]))
    return tasks


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def build(root):
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    started = time.monotonic()
    work = root / "build"
    work.mkdir()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    sources = {}
    for name, (repository, commit, digest) in PINS.items():
        archive = work / (name + ".tar.gz")
        subprocess.run(["curl", "-fLsS", f"https://github.com/{repository}/archive/{commit}.tar.gz", "-o", str(archive)], check=True)
        assert sha(archive) == digest, (name, "source archive SHA mismatch")
        target = work / name
        target.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
        sources[name] = next(target.iterdir())
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "fixed-specializations.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "fp8-output.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "fixed-prefill.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "prefill-variants.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "prefill-overlap.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "prefill-pv.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "decode-merge.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "prefill-direct-o.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "prefill-varlen.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "decode-two-cta.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "decode-wide-h2.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "prefill-pipe.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "decode-stream-h4.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "prefill-geometry.patch")], cwd=sources["fa3"], check=True)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(root / "prefill-vsum.patch")], cwd=sources["fa3"], check=True)
    cuda = Path("/usr/local/cuda")
    includes = [str(root), str(work), str(sources["fa3"] / "hopper"), str(sources["cutlass"] / "include"), str(sources["cutlass"] / "tools/util/include"), str(cuda / "include"), str(cuda / "include/cccl"), *include_paths()]
    common = ["-O3", "-std=c++17", "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-fno-gnu-unique", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    common += ["-Dflash=apac_fa3_private", "-DFlash_fwd_params=ApacFA3VerifyParams"]
    flags = ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DCUTLASS_DEBUG_TRACE_LEVEL=0", "-DNDEBUG", "-Xptxas=-v,--warn-on-spills", "-Xcompiler=-fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden,-fno-gnu-unique", *common[6:]]
    layout_cmd = [str(cuda / "bin/nvcc"), *flags, *["-I" + p for p in includes], str(root / "prefill_layout_host.cu"), "-o", str(work / "prefill-layout-host")]
    subprocess.run(layout_cmd, check=True)
    generated = subprocess.check_output([str(work / "prefill-layout-host")], text=True)
    assert generated == (root / "prefill_v_layout.cuh").read_text(), "CuTe V byte map changed"
    tasks = [(root / "wrapper.cpp", "wrapper", False)]
    tasks += generate_dispatch(root, work)
    tasks += generate_prefill_dispatch(root, work)
    tasks += [(root / (name + ".cu"), name, True) for name in ("combine", "prefill", "prefill_fixed", "prefill_image", "prefill_native_image", "decode_prepare")]
    tasks += [(sources["fa3"] / "hopper/flash_prepare_scheduler.cu", "prepare", True)]

    def compile_one(task):
        source, name, is_cuda = task
        obj = work / (name + ".o")
        cmd = [str(cuda / "bin/nvcc") if is_cuda else "g++", *(flags if is_cuda else common), *["-I" + p for p in includes], "-c", str(source), "-o", str(obj)]
        begin = time.monotonic()
        with (logs / (name + ".log")).open("w") as log:
            print(json.dumps({"tu": name, "command": cmd}), flush=True)
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = time.monotonic() - begin
        record = {"tu": name, "seconds": elapsed, "exit_code": result.returncode, "command": cmd}
        (logs / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise RuntimeError((logs / (name + ".log")).read_text())
        return record, obj

    # At most two heavy compilers on both the login node and image builder.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        compiled = list(pool.map(compile_one, tasks))
    binary = root / "libapac_fa3_verify.so"
    libs = library_paths(device_type="cuda")
    cmd = ["g++", "-shared", "-Wl,-Bsymbolic", *[str(obj) for _, obj in compiled], *["-L" + p for p in libs], *["-Wl,-rpath," + p for p in libs], "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda", "-lcudart", "-o", str(binary)]
    subprocess.run(cmd, check=True)
    torch.ops.load_library(str(binary))
    smem = torch.ops.apac_fa3_verify.resources()
    prefill_smem = torch.ops.apac_fa3_verify.prefill_resources()
    assert prefill_smem == 191488, prefill_smem
    prefill_rows = json.loads((root / "prefill-candidates.json").read_text())
    prefill_sizes = torch.ops.apac_fa3_verify.prefill_variant_resources()
    assert len(prefill_rows) == len(prefill_sizes)
    assert all(x == 0 if row.get("retired") else 0 < x <= SM90_OPTIN_BYTES for row, x in zip(prefill_rows, prefill_sizes))
    rows = candidates(root)
    assert len(rows) == len(smem)
    for row, size in zip(rows, smem):
        assert 0 < size <= SM90_OPTIN_BYTES, (row, size, SM90_OPTIN_BYTES)
        assert row.get("ctas", 1) * (size + 1024) <= SM90_SM_BYTES, (row, size, SM90_SM_BYTES)
    fixed_smem = torch.ops.apac_fa3_verify.prefill_fixed_resources()
    assert fixed_smem == prefill_smem and fixed_smem <= SM90_OPTIN_BYTES, fixed_smem
    manifest = {"schema": "apac-fa3-verify/v1", "pins": {name: {"repository": repo, "commit": commit, "archive_sha256": digest} for name, (repo, commit, digest) in PINS.items()}, "architecture": "sm_90a", "cuda": subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True), "ptxas": subprocess.check_output([str(cuda / "bin/ptxas"), "--version"], text=True), "torch": torch.__version__, "cuda_flags": flags, "max_compile_jobs": 2, "translation_units": [r for r, _ in compiled], "specializations": [{**row, "shared_bytes": size} for row, size in zip(rows, smem)], "source_sha256": {p.name: sha(p) for p in root.iterdir() if p.suffix in (".cu", ".cpp", ".h", ".cuh", ".py", ".patch") or p.name in ("candidates.json", "prefill-candidates.json")}, "binary_sha256": sha(binary), "binary_bytes": binary.stat().st_size, "elapsed_seconds": time.monotonic() - started, "gpu_executed": False}
    manifest["prefill"] = {"tile_n": 224, "shared_bytes": prefill_smem, "output": "group128-fp8-fp32-scales", "fixed_shared_bytes": fixed_smem, "fixed": "single-sequence-3584..4096-analytic-LPT-no-LSE-PDL"}
    manifest["prefill_candidates"] = [{**row, "shared_bytes": size} for row, size in zip(prefill_rows, prefill_sizes)]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([str(cuda / "bin/cuobjdump"), "--dump-resource-usage", str(binary)], stdout=(logs / "resources.txt").open("w"), check=True)
    subprocess.run(["nm", "-D", "--defined-only", str(binary)], stdout=(logs / "symbols.txt").open("w"), check=True)
    print(json.dumps(manifest), flush=True)
    shutil.rmtree(work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/apac/fa3-verify"))
    build(parser.parse_args().root.resolve())
