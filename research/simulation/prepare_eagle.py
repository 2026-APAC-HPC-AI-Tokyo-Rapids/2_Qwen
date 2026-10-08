"""Record and adapt the pinned Nebius EAGLE3 config for native SGLang loading."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def prepare(root, evidence):
    config_path = root / 'config.json'
    original = config_path.read_bytes()
    config = json.loads(original)
    if config['architectures'] != ['LlamaForCausalLM'] or config['hidden_size'] != 4096 or config['draft_vocab_size'] != 64000:
        raise ValueError('unexpected pinned draft configuration')
    weights = root / 'model.safetensors'
    with weights.open('rb') as stream:
        header = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    if any(t['dtype'] != ('I32' if k in ('d2t', 't2d') else 'BF16') for k, t in tensors.items()):
        raise ValueError('draft must retain original BF16 weights and integer token maps')
    if tensors['fc.weight']['shape'] != [4096, 12288] or tensors['lm_head.weight']['shape'] != [64000, 4096]:
        raise ValueError('incompatible EAGLE3 target hidden size or vocabulary')
    config['architectures'] = ['LlamaForCausalLMEagle3']
    config['torch_dtype'] = 'bfloat16'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'draft-original-config.json').write_bytes(original)
    (evidence / 'draft-runtime-config.json').write_bytes(config_path.read_bytes())
    (evidence / 'draft-README.md').write_bytes((root / 'README.md').read_bytes())
    digest = hashlib.sha256()
    with weights.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    weight_sha = digest.hexdigest()
    provenance = {'repository': 'nebius/EAGLE3-Qwen3-235B-A22B-Instruct-2507', 'revision': 'ff0f0c0999f6c86b21dfd2f5803eebc4ac712224', 'license': 'CC-BY-4.0', 'weight_sha256': weight_sha, 'weight_bytes': weights.stat().st_size, 'precision': 'BF16', 'parameter_count': sum(math.prod(t['shape']) for t in tensors.values() if t['dtype'] == 'BF16'), 'tensors': tensors, 'adaptation': 'Native SGLang EAGLE3 architecture selector and explicit BF16 dtype only; weights unchanged', 'original_config_sha256': hashlib.sha256(original).hexdigest(), 'runtime_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    (evidence / 'draft-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    prepare(args.root, args.evidence)
