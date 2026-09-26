"""Run CPU torch regression checks on the real patched SGLang methods.

Usage: uv run --with torch ... python integration_ngram_pd.py PATCHED_SOURCE
"""
import ast
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import torch


def methods(path, class_name, names):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return '\n'.join(ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names)


source = Path(sys.argv[1])
namespace = {'torch': torch}
exec('from __future__ import annotations\n' + methods(source / 'python/sglang/srt/speculative/spec_info.py', 'SpeculativeAlgorithm', ['build_disagg_draft_input']), namespace)
info_module = types.ModuleType('sglang.srt.speculative.ngram_info')
info_module.NgramVerifyInput = lambda **kwargs: types.SimpleNamespace(future_indices=None, **kwargs)
algorithm = types.SimpleNamespace(is_eagle=lambda: False, is_ngram=lambda: True)
for overlap in (False, True):
    reqs = [types.SimpleNamespace(rid='a', origin_input_ids=[10, 11], output_ids=[42]), types.SimpleNamespace(rid='b', origin_input_ids=[20, 21], output_ids=[43])]
    batch = types.SimpleNamespace(reqs=reqs, seq_lens=torch.tensor([2, 2]), req_pool_indices=torch.tensor([3, 7]), device='cpu', enable_overlap=overlap, grammar_needs_sync=lambda: False)
    future = Mock()
    with patch.dict(sys.modules, {'sglang.srt.speculative.ngram_info': info_module}):
        info = namespace['build_disagg_draft_input'](algorithm, batch, types.SimpleNamespace(speculative_num_draft_tokens=4), torch.tensor([42, 43]), future)
    assert info.accept_tokens.shape == (8,) and info.accept_tokens.dtype == torch.int32
    assert info.accept_lens.tolist() == [0, 0]
    assert info.new_seq_lens.tolist() == [2, 2]
    assert info.new_seq_lens.data_ptr() != batch.seq_lens.data_ptr()
    assert future.publish.call_count == int(overlap)
    if overlap:
        assert info.future_indices.tolist() == [3, 7]
    batch.spec_info = info
    worker_ns = {}
    worker_source = source / 'python/sglang/srt/speculative/ngram_worker.py'
    exec('from __future__ import annotations\n' + methods(worker_source, 'NGRAMWorker', ['_efficient_concat_last_n', '_prepare_draft_tokens']), worker_ns)
    corpus = Mock()
    corpus.batch_get.return_value = ([0] * 8, [])
    worker = types.SimpleNamespace(draft_token_num=4, enable_overlap=overlap, max_trie_depth=12, ngram_corpus=corpus)
    worker._efficient_concat_last_n = types.MethodType(worker_ns['_efficient_concat_last_n'], worker)
    worker_ns['_prepare_draft_tokens'](worker, batch)
    assert corpus.batch_get.call_args.args == (['a', 'b'], [[10, 11, 42], [20, 21, 43]], [3, 3]), 'PD bonus must appear exactly once'
print('NGRAM PD initial state: overlap/sync and no duplicate bonus passed')
