#!/usr/bin/env python3
"""Verify pinned public draft files and adapt their serving configs only."""
import hashlib
import json
import struct
import sys
from pathlib import Path


def prepare(root, evidence, draft):
    if not draft['enabled']:
        raise ValueError(draft['reason'])
    config_path = root / 'config.json'
    original = config_path.read_bytes()
    if hashlib.sha256(original).hexdigest() != draft['original_config_sha256']:
        raise ValueError('draft config SHA-256 mismatch')
    weights = root / 'model.safetensors'
    if weights.stat().st_size != draft['weight_bytes']:
        raise ValueError('draft weight size mismatch')
    digest = hashlib.sha256()
    with weights.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != draft['weight_sha256']:
        raise ValueError('draft weight SHA-256 mismatch')
    with weights.open('rb') as stream:
        header = stream.read(struct.unpack('<Q', stream.read(8))[0])
    if hashlib.sha256(header).hexdigest() != draft['header_sha256']:
        raise ValueError('draft tensor header SHA-256 mismatch')
    config = json.loads(original)
    if draft['name'] == 'nebius':
        config['architectures'] = ['LlamaForCausalLMEagle3']
        config['torch_dtype'] = 'bfloat16'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        adaptation = 'architecture LlamaForCausalLM -> LlamaForCausalLMEagle3; explicit torch_dtype=bfloat16; weights unchanged'
    elif 'transformer_layer_config' in config:
        config = dict(config['transformer_layer_config'], architectures=['LlamaForCausalLMEagle3'], draft_vocab_size=config['draft_vocab_size'], norm_before_residual=config['norm_before_residual'], torch_dtype='bfloat16', dtype='bfloat16', tie_word_embeddings=False, share_target_embedding=False, eagle_config={'use_aux_hidden_state': True, 'eagle_aux_hidden_state_layer_ids': draft['eagle_aux_hidden_state_layer_ids']})
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        adaptation = f"flatten speculators transformer_layer_config; explicit BF16; preserve draft embedding and normalized residual; SGLang eagle_config aux layer ids {draft['eagle_aux_hidden_state_layer_ids']}; weights unchanged"
    else:
        adaptation = 'none; native SGLang config and legacy midlayer loader; weights unchanged'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'draft-original-config.json').write_bytes(original)
    (evidence / 'draft-runtime-config.json').write_bytes(config_path.read_bytes())
    (evidence / 'draft-README.md').write_bytes((root / 'README.md').read_bytes())
    provenance = dict(draft, source_precision='BF16', tensors=json.loads(header), adaptation=adaptation, runtime_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    (evidence / 'draft-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    root, evidence, manifest, name = sys.argv[1:]
    draft = next(item for item in json.loads(Path(manifest).read_text()) if item['name'] == name)
    prepare(Path(root), Path(evidence), draft)
