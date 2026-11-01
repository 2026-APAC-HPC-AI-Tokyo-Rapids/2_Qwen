"""One resident torchrun: finish and persist P, then the reserved MegaMoE pass."""
import argparse
import json
import runpy
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--image-sha256', required=True)
    a = p.parse_args()
    config = json.loads((a.root/'source/firmus-framework.json').read_text())['prefill']
    prepack = runpy.run_path(str(Path(__file__).with_name('select-firmus-prefill-fa3-prepack.py')))
    # 69 seconds are reserved before P starts: MegaMoE incl L 40, endurance 6,
    # exit 5, plus 18 seconds of startup/timing uncertainty. Never shorten P's checks.
    prepack['run'](a.root/'results/prefill-fa3-prepack'/a.worker,a.run_id,a.deadline-69,a.image_sha256,p_only=True)
    import gc
    gc.collect()
    timing = runpy.run_path(str(Path(__file__).with_name('run-prefill-megamoe-timing.py')))
    timing['run'](a.root/'results/prefill-megamoe-timing'/a.worker,a.root/'source/megamoe-local-ep1-routing.npz',a.run_id,a.deadline,a.image_sha256,config.get('megamoe_fc1_balance_gate',False))


if __name__ == '__main__':
    main()
