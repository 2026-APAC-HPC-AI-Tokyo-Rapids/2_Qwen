#!/usr/bin/env python3
"""CPU comparison of real decoder methods; supply pinned source paths, no GPU imports."""
import argparse
import ast
import copy
import types
import unittest
from pathlib import Path

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, hidden, residual=None):
        if residual is not None:
            hidden = hidden + residual
        output = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.eps) * self.weight
        return output if residual is None else (output, hidden)


class MLP(nn.Module):
    def __init__(self, config, *args):
        super().__init__()
        size = config.hidden_size if not isinstance(config, int) else config
        intermediate = config.intermediate_size if not isinstance(config, int) else args[0]
        self.gate = nn.Linear(size, intermediate, bias=False)
        self.up = nn.Linear(size, intermediate, bias=False)
        self.down = nn.Linear(intermediate, size, bias=False)

    def forward(self, hidden):
        return self.down(nn.functional.silu(self.gate(hidden)) * self.up(hidden))


class Attention(nn.Module):
    """Same CPU attention in both layers; isolates decoder residual/norm wiring."""
    def __init__(self, config, *args):
        super().__init__()
        self.head_dim = config.head_dim
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.projection = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.projection(hidden_states)


class ReferenceAttention(Attention):
    def forward(self, hidden_states, **kwargs):
        return super().forward(hidden_states), None, None


class BaseLayer(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)


def extract(source, name, namespace, methods=None):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    node = copy.deepcopy(node)
    node.decorator_list = []
    if methods:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    code = 'from __future__ import annotations\n' + ast.unparse(node)
    scope = dict(namespace)
    exec(compile(code, '<extracted decoder source>', 'exec'), scope)  # noqa: S102
    return scope[name]


def configure(sglang_source, redhat_source, baseline_source):
    global Patched, Reference, Original, EmbeddingModel
    namespace = {'torch': torch, 'nn': nn, 'RMSNorm': RMSNorm, 'LlamaRMSNorm': RMSNorm, 'LlamaDecoderLayer': BaseLayer, 'LlamaMLP': MLP, 'Eagle3Attention': ReferenceAttention, 'QKVParallelLinear': lambda *a, **kw: nn.Identity(), 'add_prefix': lambda a, b: a}
    Patched = extract(sglang_source.read_text(), 'LlamaDecoderLayer', namespace)
    Original = extract(baseline_source.read_text(), 'LlamaDecoderLayer', namespace)
    Reference = extract(redhat_source.read_text(), 'Eagle3DecoderLayer', namespace)

    class EmbeddingBase:
        def set_embed(self, embed):
            self.embedding = embed

    EmbeddingModel = extract(sglang_source.read_text(), 'LlamaForCausalLMEagle3', {'LlamaForCausalLM': EmbeddingBase}, methods={'set_embed'})


def config(flag=None):
    result = types.SimpleNamespace(hidden_size=8, intermediate_size=16, hidden_act='silu', head_dim=4, num_attention_heads=2, num_key_value_heads=1, model_type='llama', rms_norm_eps=1e-6)
    if flag is not None:
        result.norm_before_residual = flag
    return result


class DecoderTests(unittest.TestCase):
    def compare(self, flag):
        torch.manual_seed(2026)
        cfg = config(flag)
        patched = Patched(cfg)
        reference = Reference(cfg, 0, norm_before_residual=bool(flag))
        for name in ('hidden_norm', 'input_layernorm', 'post_attention_layernorm', 'mlp'):
            getattr(reference, name).load_state_dict(getattr(patched, name).state_dict())
        reference.self_attn.projection.load_state_dict(patched.self_attn.projection.state_dict())
        for scale in (0.1, 1, 10):
            with self.subTest(scale=scale):
                hidden = torch.randn(2, 3, cfg.hidden_size) * scale
                embeds = torch.randn_like(hidden)
                expected = reference(torch.cat([embeds, hidden], -1))[0]
                output, residual = patched(None, embeds, hidden, None, None)
                torch.testing.assert_close(output + residual, expected, rtol=1e-6, atol=1e-6)
                if flag is not True:
                    original = Original(cfg)
                    original.load_state_dict(patched.state_dict())
                    old_output, old_residual = original(None, embeds, hidden, None, None)
                    torch.testing.assert_close(output, old_output, rtol=0, atol=0)
                    torch.testing.assert_close(residual, old_residual, rtol=0, atol=0)
                else:
                    reference.norm_before_residual = False
                    old = reference(torch.cat([embeds, hidden], -1))[0]
                    self.assertFalse(torch.allclose(expected, old))
                    reference.norm_before_residual = True

    def test_redhat_true_matches_reference(self):
        self.compare(True)

    def test_false_is_identical_to_original(self):
        self.compare(False)

    def test_missing_is_identical_to_original(self):
        self.compare(None)

    def test_embedding_is_preserved_only_when_requested(self):
        for flag in (None, False, True):
            model = EmbeddingModel()
            model.config = types.SimpleNamespace(**({} if flag is None else {'share_target_embedding': flag}))
            original, target = object(), object()
            model.embedding = original
            model.set_embed(target)
            self.assertIs(model.embedding, original if flag is False else target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sglang-source', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--redhat-source', type=Path, required=True)
    args = parser.parse_args()
    configure(args.sglang_source, args.redhat_source, args.baseline_source)
    unittest.main(argv=[__file__])
