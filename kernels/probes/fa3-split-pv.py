"""Bounded one-GPU correctness and paired timing for a single new FA3 architecture."""
import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location('pipe_probe', Path(__file__).with_name('fa3-prefill-pipe.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def call(lib, args):
    lib.run(0, *args)
    torch.cuda.synchronize()


def compare(control, candidate, lengths, seed, distribution):
    helper.LENGTHS = lengths
    q, k, v, cu = helper.inputs('cuda')
    generator = torch.Generator(device='cuda').manual_seed(seed)
    if distribution == 'random':
        q.copy_((torch.randn(q.shape, device='cuda', generator=generator) * 2).to(q.dtype))
        k.copy_(torch.randn(k.shape, device='cuda', generator=generator).to(k.dtype))
        v.copy_(torch.randn(v.shape, device='cuda', generator=generator).to(v.dtype))
    elif distribution == 'zero':
        q.view(torch.uint8).zero_()
        k.view(torch.uint8).zero_()
    elif distribution == 'peaked':
        q.copy_((q.float() * 4).to(q.dtype))
        k.copy_((k.float() * 4).to(k.dtype))
    args = []
    for _ in range(2):
        out, scales, metadata = helper.buffers('cuda')
        # A sentinel makes a skipped row visible, including on a partial M tile.
        out.view(torch.uint8).fill_(0x7f)
        scales.fill_(float('nan'))
        args.append((q, k, v, cu, out, scales, metadata))
    call(control, args[0])
    call(candidate, args[1])
    outputs_equal = torch.equal(args[0][4].view(torch.uint8), args[1][4].view(torch.uint8))
    rows = sum(lengths)
    scales_equal = torch.equal(args[0][5][:, :rows], args[1][5][:, :rows])
    finite = bool(torch.isfinite(args[1][4].float()).all() and torch.isfinite(args[1][5][:, :rows]).all())
    ref = helper.reference(q, k, v, stride=max(1, max(lengths) // 12))
    errors = [helper.errors(a[4], a[5], ref) for a in args]
    correct = finite and outputs_equal and scales_equal and errors[1]['relative_l2'] <= errors[0]['relative_l2'] * 1.02 + 1e-6
    row = {'lengths': lengths, 'seed': seed, 'distribution': distribution, 'bytes_equal': outputs_equal, 'scales_equal': scales_equal, 'finite': finite, 'errors': errors, 'passed': correct}
    # On a failure, preserve actual mismatching coordinates and magnitudes.
    if not correct:
        mismatch = (args[0][4].view(torch.uint8) != args[1][4].view(torch.uint8)).nonzero()
        row['first_output_mismatches'] = mismatch[:12].cpu().tolist()
        row['mismatching_output_bytes'] = mismatch.shape[0]
    print('NUMERIC ' + json.dumps(row), flush=True)
    return row, args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    control = helper.Library(args.directory / 'control.so')
    candidate = helper.Library(args.directory / 'split-pv.so')
    assert candidate.lib.pf_init() == 0
    assert control.count == candidate.count == 1
    result = {'schema': 'fa3-split-pv-gate/v1', 'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu_executed': True, 'serving_executed': False, 'numeric': [], 'timing': [], 'passed': False}
    save(args.output, result)
    # Tile boundaries, unequal lengths and the actual PP4 pair. Three distributions
    # cover equal maxima, ordinary updates and strongly peaked online softmax.
    checks = [((7, 9), 'random'), ((191, 193), 'random'), ((3994, 3994), 'random'), ((3986, 4027), 'random'), ((193, 385), 'zero'), ((3994, 3994), 'peaked')]
    for index, (lengths, distribution) in enumerate(checks):
        row, _ = compare(control, candidate, lengths, 20261101 + index, distribution)
        result['numeric'].append(row)
        save(args.output, result)
        assert row['passed'], 'Numerical gate failed; no timing was run'
    for lengths in ((3994, 3994), (3986, 4027)):
        helper.LENGTHS = lengths
        q, k, v, cu = helper.inputs('cuda')
        buffers = [(q, k, v, cu, *helper.buffers('cuda')) for _ in range(2)]
        # Repeated independent launches catch mailbox phase or reuse defects.
        call(control, buffers[0])
        for _ in range(300):
            candidate.run(0, *buffers[1])
        torch.cuda.synchronize()
        assert torch.equal(buffers[0][4].view(torch.uint8), buffers[1][4].view(torch.uint8))
        assert torch.equal(buffers[0][5], buffers[1][5])
        graphs = [helper.capture(lib, 0, buf) for lib, buf in zip((control, candidate), buffers)]
        samples = [[], []]
        before = helper.sm_clock()
        for pair in range(20):
            for index in ((0, 1) if pair % 2 == 0 else (1, 0)):
                samples[index].append(helper.measure(graphs[index]))
        # Graph replay must also retain correctness after the timing repetitions.
        assert torch.equal(buffers[0][4].view(torch.uint8), buffers[1][4].view(torch.uint8))
        assert torch.equal(buffers[0][5], buffers[1][5])
        savings = [a-b for a, b in zip(*samples)]
        row = {'lengths': lengths, 'control_us': samples[0], 'candidate_us': samples[1], 'median_saving_us': statistics.median(savings), 'median_ratio': statistics.median(samples[1]) / statistics.median(samples[0]), 'endurance_calls': 300, 'clock_power_before': before, 'clock_power_after': helper.sm_clock()}
        result['timing'].append(row)
        save(args.output, result)
        print('TIMING ' + json.dumps(row), flush=True)
    result.update(passed=True, elapsed_seconds=time.monotonic()-started)
    save(args.output, result)
    print('PASS fa3-split-pv numerical/endurance gates and paired timing completed; this is not a serving result', flush=True)


if __name__ == '__main__':
    main()
