"""Compare greedy output tokens for identical post-benchmark probe requests."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(baseline, candidate):
    if baseline['prompt_sha256'] != candidate['prompt_sha256']:
        raise ValueError('different probe prompts')
    if len(baseline['responses']) != 8 or len(candidate['responses']) != 8:
        raise ValueError('eight completed probe responses required')
    rows = []
    for index, (left, right) in enumerate(zip(baseline['responses'], candidate['responses'], strict=True)):
        if left['request'] != right['request']:
            raise ValueError('different greedy request parameters')
        a, b = left['response']['choices'][0], right['response']['choices'][0]
        a_tokens, b_tokens = a['logprobs']['content'], b['logprobs']['content']
        if len(a_tokens) != 64 or len(b_tokens) != 64:
            raise ValueError('64 output tokens required for each probe')
        tokens_match = [(t['token'], t.get('bytes')) for t in a_tokens] == [(t['token'], t.get('bytes')) for t in b_tokens]
        rows.append({'prompt_index': index, 'tokens_match': tokens_match, 'text_matches': a['message']['content'] == b['message']['content'], 'finish_reason_matches': a['finish_reason'] == b['finish_reason']})
    return {'schema_version': 'qwen-greedy-parity-comparison/v1', 'all_match': all(all(row[k] for k in ('tokens_match', 'text_matches', 'finish_reason_matches')) for row in rows), 'compared_tokens': 512, 'prompts': rows, 'scope': 'eight generic prompts; not the organizers full-set accuracy check'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    result['source_sha256'] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['all_match'] else 1)
