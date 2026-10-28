"""Login-node AOT build; use the serving pins, patch order, flags and dispatch."""
import concurrent.futures
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path


def build(workspace, runtime):
    source = runtime / "source"
    root = source / "companion"
    shutil.copytree(workspace / "kernels/fa3-verify", root)
    shutil.copyfile(workspace / "kernels/probes/fa3-fp8-output.patch", root / "fp8-output.patch")
    api = runpy.run_path(str(root / "build.py"))
    names = ("p128split3-s3-fp8", "p128-merge-s3-fp8")
    rows = [row for row in api["candidates"](root) if row["name"] in names]
    assert len(rows) == 2 and all((r["block_n"], r["stages"], r["splits"]) == (128, 3, 3) for r in rows)
    (root / "candidates.json").write_text(json.dumps(rows, indent=2) + "\n")
    work = source / "build"
    work.mkdir()
    sources = api["prepare_sources"](root, work)
    tasks = api["generate_dispatch"](root, work)
    # The separate combine source stays untouched. Its probe copy records the
    # actual pre-quantization values through the same helper as in-kernel merge.
    combine = (root / "combine.cu").read_text()
    for old, new in (
        ("const int* order, int batch) {", "const int* order, int batch, DecodeProbe probe) {"),
        ("scales + int64_t(head) * scale_stride + token);", "scales + int64_t(head) * scale_stride + token, probe, token * 64 + head, row);"),
        ("p.varlen_batch_idx_ptr, p.b);", "p.varlen_batch_idx_ptr, p.b, p.verify_merge.probe);"),
    ):
        assert combine.count(old) == 1, old
        combine = combine.replace(old, new)
    combine = combine[:combine.index("void combine(Flash_fwd_params& p, cudaStream_t stream) {")] + "void combine(Flash_fwd_params& p, cudaStream_t stream) { launch_combine<3, true>(p, stream); CHECK_CUDA_KERNEL_LAUNCH(); }\n}\n"
    (work / "combine.cu").write_text(combine)
    tasks += [(work / "combine.cu", "combine", True), (sources["fa3"] / "hopper/flash_prepare_scheduler.cu", "prepare", True), (workspace / "kernels/probes/fa3-decode-merge.cu", "probe", True)]
    includes = [root, work, sources["fa3"] / "hopper", sources["cutlass"] / "include", sources["cutlass"] / "tools/util/include"]
    flags = [*api["cuda_flags"](1), "-DFA3_DECODE_MERGE_PROBE=1", "-ccbin", os.environ["CXX"]]
    logs = runtime / "logs"

    def compile_one(task):
        src, name, _ = task
        obj = work / (name + ".o")
        cmd = ["nvcc", *flags, *["-I" + str(p) for p in includes], "-c", str(src), "-o", str(obj)]
        (logs / ("build-" + name + ".json")).write_text(json.dumps(cmd, indent=2))
        with (logs / ("build-" + name + ".log")).open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError((logs / ("build-" + name + ".log")).read_text())
        with (logs / ("sass-" + name + ".txt")).open("w") as log:
            subprocess.run(["cuobjdump", "-sass", str(obj)], stdout=log, check=True)
        return obj

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        objects = list(pool.map(compile_one, tasks))
    binary = source / "fa3-decode-merge.so"
    subprocess.run(["nvcc", "-ccbin", os.environ["CXX"], "--shared", "--cudart", "static", "-Xlinker=-Bsymbolic", *map(str, objects), "-o", str(binary)], check=True)
    provenance = {"commit": os.environ.get("GITHUB_SHA"), "kernel": "FA3 Decode merge diagnostic", "calibration": True, "scope": "one GPU, one B40 forward per candidate, no serving", "candidates": rows, "pins": api["PINS"], "cuda_flags": flags, "cuda": subprocess.check_output(["nvcc", "--version"], universal_newlines=True), "binary_sha256": api["sha"](binary), "source_sha256": {str(p.relative_to(workspace)): api["sha"](p) for p in [*sorted((workspace / "kernels/fa3-verify").glob("*")), *sorted((workspace / "kernels/probes").glob("*decode-merge*")), workspace / "kernels/probes/fa3-fp8-output.patch", workspace / "scripts/check-firmus-decode-kernels.py"] if p.is_file()}}
    (source / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    build(Path(os.environ["GITHUB_WORKSPACE"]), Path(os.environ["RUNTIME_DIR"]))
