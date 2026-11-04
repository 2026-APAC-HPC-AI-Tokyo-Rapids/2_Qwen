"""Record what a submission script does on a cluster, with every cluster command stubbed.

A record holds each srun step's argv (keyed by its mount name), each Bonus worker's final launch_server
argv and environment, the ordered HTTP and helper calls of the job, the external commands of a
preparation (image extraction, patch application, draft download), and a hash of every file left in
the run directory. Two versions of a script that produce the same record launch the same servers
with the same settings in the same order.

Regenerate a reference record: python submission_harness.py KIND REPO OUT.json
KIND is bonus-sbatch, bonus-prepare, basic-sbatch or basic-prepare; REPO a checkout of this repository.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON = sys.executable
KINDS = ("bonus-sbatch", "bonus-prepare", "basic-sbatch", "basic-prepare")
# Draft provenance the real prepare_eagle.py writes for the pinned public draft.
PROVENANCE = {"revision": "ff0f0c0999f6c86b21dfd2f5803eebc4ac712224", "weight_sha256": "bf226a966868fe413a22e62e471aa1130bd94f17fd2c40d322b1c8b4def1ae86", "weight_bytes": 1145943008, "precision": "BF16"}


def stub(directory, name, body):
    path = directory / name
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)


def sbatch_stubs(harness):
    bin_dir = harness / "bin"
    stub(bin_dir, "scontrol", 'printf "g1\\ng4\\n"')
    stub(bin_dir, "getent", 'case "$2" in g1) echo "10.0.0.1 STREAM g1";; g4) echo "10.0.0.4 STREAM g4";; esac')
    stub(bin_dir, "whoami", "echo tester")
    stub(bin_dir, "date", "echo 2026-01-01T00:00:00+00:00")
    stub(bin_dir, "nvidia-smi", f'if [[ "$*" == *index,uuid* ]]; then cat {harness}/gpus.csv; else echo "0, NVIDIA H200, 143771 MiB, 570.00, 700.00 W"; fi')
    # Every HTTP call succeeds and server info reports the FP8 draft. A Bonus worker's /health answers only
    # once its launch is recorded, so the job cannot stop a worker before it reaches launch_server.
    stub(bin_dir, "curl", f'''url=${{@: -1}}
if [[ -e {harness}/expect-launch && "$url" == */health ]]; then
  port=${{url##*:}}; port=${{port%%/*}}
  case "$port" in 30100) k=prefill-0;; 30101) k=prefill-1;; 30102) k=prefill-2;; 30200) k=decode-0;; esac
  [[ -e {harness}/launch/$k.json ]] || {{ sleep 0.2; exit 7; }}
fi
# The router answers /readiness only once its step has started (it is launched in the background).
if [[ "$url" == */readiness ]]; then [[ -e {harness}/srun/router.json ]] || {{ sleep 0.2; exit 7; }}; fi
[[ "$url" == */health ]] || printf 'curl %s\\n' "$*" >>{harness}/events.log
out=; prev=; for a in "$@"; do [[ "$prev" == -o ]] && out=$a; prev=$a; done
body='{{"speculative_draft_model_quantization":"fp8"}}'
if [[ -z "$out" ]]; then printf '%s' "$body"; elif [[ "$out" != /dev/null ]]; then printf '%s' "$body" >"$out"; fi''')
    # srun: record argv under the step's mount name, then act like the step (workers run until killed).
    stub(bin_dir, "srun", f'''key=other
for a in "$@"; do case "$a" in --container-mounts=*) key=${{a#*/home/}}; key=${{key%%:*}};; esac; done
[[ "$*" == *"rm -rf"* ]] && key=cleanup-$RANDOM$RANDOM
case "$key" in prefill-*|decode-*|cleanup-*) ;; *) printf 'srun %s\\n' "$key" >>{harness}/events.log;; esac
{PYTHON} -c 'import json,sys; json.dump(sys.argv[2:], open(sys.argv[1] + ".tmp", "w")); import os; os.replace(sys.argv[1] + ".tmp", sys.argv[1])' "{harness}/srun/$key.json" "$@"
args=("$@")
for i in "${{!args[@]}}"; do
  case "${{args[$i]}}" in
    */worker.sh)
      script=${{args[$i]}}
      sed -e "s#/opt/sglang/venv/bin/python#{harness}/bin/vpython#g" -e "s#/sys/class/infiniband#{harness}/ib#g" "$script" >"${{script%.sh}}.$key.harness.sh"
      HARNESS_KEY=$key CUDA_VISIBLE_DEVICES=0,1,2,3 exec bash "${{script%.sh}}.$key.harness.sh" "${{args[@]:$((i + 1))}}";;
    */basic-worker.py|*/sglang-router|*/sgl-model-gateway) exec sleep 600;;
    */official-aiperf.sh) echo '{{}}' >"${{args[$((i + 1))]}}/results/score.json"; exit 0;;
    */greedy-probe.py) exit 0;;
  esac
done
exit 0''')
    # The container Python: record the server launch, run anything else for real.
    stub(bin_dir, "vpython", f'''if [[ "$1" == -m && "$2" == sglang.launch_server ]]; then
  {PYTHON} -c 'import json,os,sys; skip={{"PWD","OLDPWD","SHLVL","_","PATH","HOME","LC_CTYPE"}}; json.dump({{"argv": sys.argv[2:], "env": {{k: v for k, v in os.environ.items() if k not in skip and not k.startswith("HARNESS")}}}}, open(sys.argv[1], "w"), indent=1, sort_keys=True)' "{harness}/launch/$HARNESS_KEY.json" "$@"
  exec sleep 600
fi
exec {PYTHON} "$@"''')


def run_sbatch(kind, repo, harness):
    root = harness / "run"
    for name in ("logs", "results", "scripts", "source", "home", "draft"):
        (root / name).mkdir(parents=True)
    (root / "sglang.sqsh").write_text("")
    (root / "draft/model.safetensors").write_text("x")
    model = harness / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (root / "source/model.json").write_text(json.dumps({"model": {"path": str(model)}}))
    package = repo / "submission"
    (harness / "ib/mlx5_0/ports/1").mkdir(parents=True)
    (harness / "ib/mlx5_0/ports/1/state").write_text("4: ACTIVE\n")
    if kind == "bonus-sbatch":
        # What prepare.sh stages: both patched packages, the config, the pinned env files and run/.
        (root / "overlay/sglang/python/sglang").mkdir(parents=True)
        (root / "overlay/deep-gemm/deep_gemm").mkdir(parents=True)
        shutil.copy(package / "config/serving.json", root / "source/firmus-framework.json")
        for name in ("prefill-kernels.env", "decode-kernels.env", "firmus-gpu-sm-clock.json"):
            shutil.copy(package / "config" / name, root / "source" / name)
        for path in (package / "run").iterdir():
            shutil.copy(path, root / "scripts" / path.name)
        clock = json.loads((package / "config/firmus-gpu-sm-clock.json").read_text())
        (harness / "gpus.csv").write_text("".join(f"{i}, {uuid}\n" for i, uuid in enumerate(sorted(clock["gpus"])[:8])))
        (harness / "expect-launch").write_text("")
    else:
        shutil.copy(repo / "configs/firmus-basic.json", root / "source/firmus-framework.json")
        for name in ("basic-worker.py", "check-basic-server-info.py", "collect-eagle-acceptance.py"):
            shutil.copy(repo / "scripts" / name, root / "scripts" / name)
        shutil.copy(package / "basic/submit.sbatch", root / "scripts/submit.sbatch")
        shutil.copy(package / "run/official-aiperf.sh", root / "scripts/official-aiperf.sh")
        (harness / "gpus.csv").write_text("")
        # The host-side readback and EAGLE collector need real servers; record their calls instead.
        stub(harness / "bin", "python3", f'printf "python3 %s\\n" "$*" >>{harness}/events.log')
    env = {"PATH": f"{harness}/bin:/usr/bin:/bin", "HOME": str(harness / "home"), "USER": "tester", "LOGNAME": "tester", "SLURM_SUBMIT_DIR": str(root), "SLURM_JOB_NODELIST": "g[1,4]", "SLURM_JOB_ID": "1"}
    with (harness / "sbatch.out").open("w") as output:
        status = subprocess.run(["bash", str(root / "scripts/submit.sbatch")], cwd=root, env=env, timeout=90, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, check=False).returncode
    if status:
        raise RuntimeError(f"submit.sbatch exited {status}:\n{(harness / 'sbatch.out').read_text()[-3000:]}")
    return root


def run_prepare(kind, repo, harness):
    work = harness / "repo"
    for name in ("submission", "scripts", "configs"):
        shutil.copytree(repo / name, work / name)
    inputs = harness / "inputs"
    image = inputs / "BUILD.sqsh"
    model = inputs / "model"
    tokenizer = inputs / "tokenizer"
    (model / ".cache/huggingface/download").mkdir(parents=True)
    tokenizer.mkdir()
    image.write_bytes(b"image")
    digest = hashlib.sha256(b"image").hexdigest()
    (model / ".cache/huggingface/download/config.json.metadata").write_text("e156cb4efae43fbee1a1ab073f946a1377e6b969\n1\n")
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_moe", "quantization_config": {"quant_method": "fp8"}}))
    shards = [f"model-{i:05d}-of-00024.safetensors" for i in range(1, 25)]
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{i}": shard for i, shard in enumerate(shards)}}))
    for shard in shards:
        (model / shard).write_text("x")
    for name in ("config.json", "generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
        (tokenizer / name).write_text(name)
    log = harness / "events.log"
    bin_dir = harness / "bin"
    # Draft download, image extraction, patch application and the draft rewrite are recorded with the files they would create.
    stub(bin_dir, "uvx", f'printf "uvx %s\\n" "$*" >>{log}; dir=; prev=; for a in "$@"; do [[ "$prev" == --local-dir ]] && dir=$a; prev=$a; done; mkdir -p "$dir"; echo draft >"$dir/model.safetensors"')
    stub(bin_dir, "setsid", 'exec "$@"')
    stub(bin_dir, "unsquashfs", f'printf "unsquashfs %s\\n" "$*" >>{log}; dest=; prev=; for a in "$@"; do [[ "$prev" == -d ]] && dest=$a; prev=$a; done; mkdir -p "$dest/${{@: -1}}"; echo pkg >"$dest/${{@: -1}}/__init__.py"')
    stub(bin_dir, "git", f'if [[ "$1" == apply ]]; then printf "git %s %s\\n" "$(basename "$PWD")" "$*" >>{log}; exit 0; fi; exec /usr/bin/git "$@"')
    stub(bin_dir, "python3", f'if [[ "$1" == */prepare_eagle.py ]]; then printf "python3 %s\\n" "$*" >>{log}; echo \'{json.dumps(PROVENANCE)}\' >"$3/draft-provenance.json"; exit 0; fi; exec {PYTHON} "$@"')
    root = harness / "out/run"
    root.parent.mkdir()
    if kind == "bonus-prepare":
        serving = work / "submission/config/serving.json"
        config = json.loads(serving.read_text())
        config["images"]["sglang"].update(sha256=digest, size_bytes=5)
        serving.write_text(json.dumps(config, indent=2) + "\n")
        command = ["bash", str(work / "submission/run/prepare.sh"), str(root), str(image), str(model), str(tokenizer)]
    else:
        settings = work / "configs/firmus-basic.json"
        config = json.loads(settings.read_text())
        config.update(container_sha256=digest, container_size_bytes=5)
        settings.write_text(json.dumps(config, indent=2) + "\n")
        sidecar = {"tier": "basic", "applied_patches": [], "source_commits": {"sglang": "297e54fd105e974cdf401e5b4d20253a22c16f90", "qwen": config["container_source_commit"]}, "container_artifact": {"sha256": digest, "size_bytes": 5}}
        (inputs / "BUILD.container-sidecar.json").write_text(json.dumps(sidecar))
        command = ["bash", str(work / "submission/basic/prepare.sh"), str(root), str(image), str(model), str(tokenizer), str(settings)]
    result = subprocess.run(command, env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(harness / "home")}, timeout=90, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"prepare.sh exited {result.returncode}:\n{result.stderr[-3000:]}")
    with log.open("a") as events:
        events.write("".join(f"stdout {line}\n" for line in result.stdout.splitlines()))
    return root


def record(kind, repo):
    assert kind in KINDS, kind
    with tempfile.TemporaryDirectory(prefix="submission-harness-") as temporary:
        harness = Path(temporary)
        for name in ("bin", "srun", "launch"):
            (harness / name).mkdir()
        if kind.endswith("sbatch"):
            sbatch_stubs(harness)
            root = run_sbatch(kind, Path(repo), harness)
        else:
            root = run_prepare(kind, Path(repo), harness)
        tree = {}
        for path in sorted(root.rglob("*")):
            relative = str(path.relative_to(root))
            if ".harness." in relative:
                continue
            tree[relative] = "symlink" if path.is_symlink() else hashlib.sha256(path.read_bytes().replace(str(harness).encode(), b"H")).hexdigest() if path.is_file() else "directory"
        events = (harness / "events.log").read_text().splitlines() if (harness / "events.log").exists() else []
        result = {
            "srun": {p.stem: json.loads(p.read_text()) for p in sorted((harness / "srun").glob("*.json")) if not p.stem.startswith("cleanup-")},
            "cleanup_steps": len(list((harness / "srun").glob("cleanup-*.json"))),
            "launch": {p.stem: json.loads(p.read_text()) for p in sorted((harness / "launch").glob("*.json"))},
            "events": events,
            "tree": tree,
        }
        return json.loads(json.dumps(result).replace(str(harness), "H"))


if __name__ == "__main__":
    kind, repo, out = sys.argv[1:4]
    Path(out).write_text(json.dumps(record(kind, repo), indent=1, sort_keys=True) + "\n")
