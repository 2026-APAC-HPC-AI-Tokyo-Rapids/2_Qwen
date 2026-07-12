#!/usr/bin/env python3
"""Render one immutable PJM script from an experiment env file."""
from __future__ import annotations
import argparse, os, shlex
from pathlib import Path

REQUIRED = {"EXPERIMENT_NAME", "MODEL_FAMILY", "MODEL_PATH", "IMAGE", "MODE", "RSCGRP", "GPU_COUNT", "ELAPSE"}

def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"{path}:{lineno}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        if not key.replace("_", "").isalnum() or not key.isupper():
            raise SystemExit(f"{path}:{lineno}: invalid key {key!r}")
        values[key] = value
    missing = REQUIRED - values.keys()
    if missing:
        raise SystemExit(f"missing keys: {', '.join(sorted(missing))}")
    return values

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--result-root", default="/fast/PROJECT/qwen-results")
    args = ap.parse_args()
    exp = args.experiment.resolve()
    cfg = read_env(exp)
    if cfg["MODEL_FAMILY"] != "qwen3.5":
        raise SystemExit("MODEL_FAMILY must be qwen3.5")
    if "qwen3.5" not in cfg["MODEL_PATH"].lower():
        raise SystemExit("MODEL_PATH must point to a Qwen 3.5 checkpoint")
    output = args.output or root / "build" / "jobs" / f"{cfg['EXPERIMENT_NAME']}.pjm"
    replacements = {
        "RSCGRP": cfg["RSCGRP"], "ELAPSE": cfg["ELAPSE"], "GPU_COUNT": cfg["GPU_COUNT"],
        "REPO_DIR_Q": shlex.quote(str(root)), "EXPERIMENT_FILE_Q": shlex.quote(str(exp)),
        "IMAGE_Q": shlex.quote(cfg["IMAGE"]), "MODEL_PATH_Q": shlex.quote(cfg["MODEL_PATH"]),
        "EXPERIMENT_NAME_Q": shlex.quote(cfg["EXPERIMENT_NAME"]),
        "RESULT_ROOT_Q": shlex.quote(args.result_root), "MODE_Q": shlex.quote(cfg["MODE"]),
    }
    text = (root / "jobs/templates/genkai.pjm.in").read_text()
    for key, value in replacements.items(): text = text.replace(f"@{key}@", value)
    if "@" in text:
        raise SystemExit("unresolved template token")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    output.chmod(0o755)
    print(output)

if __name__ == "__main__": main()
