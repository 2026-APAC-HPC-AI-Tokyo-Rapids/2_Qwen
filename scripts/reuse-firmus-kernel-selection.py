#!/usr/bin/env python3
"""Validate a recorded Firmus selection before reusing its five kernel settings."""
import argparse
import copy
import hashlib
import json
import re
import shlex
from pathlib import Path

ENVIRONMENT = {
    'SGLANG_PREFILL_FA3_VERIFY': 'n192',
    'SGLANG_DECODE_FA3_VERIFY': 'p128split2-s3-fp8',
    'DG_SM90_1D2D_DECODE_CONFIG': 'QKV:128,144,5,1,0',
    'SGLANG_DECODE_DENSE_GEMM_QKV': '',
    'SGLANG_DECODE_DENSE_GEMM_O': '',
}


def normalize_config(config):
    normalized = copy.deepcopy(config)
    normalized.get('speculation', {}).pop('draft_quantization', None)
    normalized.pop('reuse_kernel_selection', None)
    for framework, image in normalized['images'].items():
        normalized['images'][framework] = {key: image[key] for key in ('sha256', 'size_bytes')}
    return normalized


def normalized_config_sha256(config):
    return hashlib.sha256(json.dumps(normalize_config(config), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def file_sha(repo, entry):
    require(isinstance(entry, dict) and set(entry) == {'path', 'sha256'}, 'File record must contain path and sha256')
    name = entry['path']
    require(isinstance(name, str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts, 'File path must stay within the repository')
    path = (repo / name).resolve()
    require(path.is_relative_to(repo.resolve()) and path.is_file(), f'Missing repository file: {name}')
    require(valid_sha(entry['sha256']), f'Invalid SHA256: {name}')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], f'File SHA256 mismatch: {name}')


def validate_record(record, config, repo):
    require(isinstance(record, dict) and isinstance(config, dict), 'Record and config must be JSON objects')
    require(record.get('schema') == 'firmus-kernel-selection/v1', 'Unsupported selection schema')
    require(str(record.get('source_run', '')).isdigit() and int(record['source_run']) > 0, 'Missing source run')
    require(isinstance(record.get('source_commit'), str) and re.fullmatch(r'[0-9a-f]{40}', record['source_commit']) is not None, 'Invalid source commit')
    require(record.get('normalized_config_sha256') == normalized_config_sha256(config), 'Normalized runtime config differs from the recorded selection')
    require(valid_sha(record.get('image_sha256')) and record['image_sha256'] == config['images']['sglang']['sha256'], 'SGLang image SHA256 mismatch')
    patches = record.get('patches')
    expected = config.get('sglang_overlay_patches', []) + config.get('deep_gemm_overlay_patches', [])
    require(isinstance(patches, list) and [entry.get('path') if isinstance(entry, dict) else None for entry in patches] == expected, 'Configured patch order differs from the recorded selection')
    for entry in patches:
        file_sha(repo, entry)
    extra = record.get('extra_files')
    require(isinstance(extra, list) and all(isinstance(entry, dict) for entry in extra) and len({entry.get('path') for entry in extra}) == len(extra), 'Invalid or repeated extra file records')
    expert_location = config.get('prefill', {}).get('expert_location')
    require(not expert_location or expert_location in [entry.get('path') for entry in extra], 'Configured expert mapping is missing from extra file records')
    for entry in extra:
        file_sha(repo, entry)
    require(record.get('environment') == ENVIRONMENT, 'Selection environment must contain the five reviewed kernel settings exactly')
    evidence = record.get('evidence_sha256')
    require(isinstance(evidence, dict) and evidence and all(isinstance(name, str) and name and valid_sha(digest) for name, digest in evidence.items()), 'Missing or invalid evidence SHA256 records')


def reuse(record, config, repo, runtime):
    """Validate completely before creating any environment or success marker."""
    validate_record(record, config, Path(repo))
    runtime = Path(runtime)
    source, results = runtime / 'source', runtime / 'results'
    prefill = ['SGLANG_PREFILL_FA3_VERIFY']
    decode = [name for name in ENVIRONMENT if name not in prefill]
    environments = {source / 'prefill-kernels.env': prefill, source / 'decode-kernels.env': decode}
    payload = {**record, 'reused': True, 'no_gpu_selection_rerun': True}
    serialized = json.dumps(payload, indent=2) + '\n'
    source.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    for path, names in environments.items():
        path.write_text(''.join(f'{name}={shlex.quote(record["environment"][name])}\n' for name in names))
    for directory in (source, results):
        (directory / 'reused-kernel-selection.json').write_text(serialized)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    try:
        reused = reuse(json.loads(args.record.read_text()), json.loads(args.config.read_text()), args.repo, args.runtime)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'Kernel selection reuse rejected: {exc}\n')
    print(f'PASS reused kernel selection from run {reused["source_run"]}; no GPU selection rerun')


if __name__ == '__main__':
    main()
