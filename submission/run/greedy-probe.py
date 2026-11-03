"""Collect post-benchmark greedy outputs through the official gateway."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

PROMPTS = [
    'Explain why the sky appears blue in two sentences.',
    'Write a Python function that returns the sum of the squares in a list.',
    'Continue this sequence and explain the rule: 2, 4, 8, 16, 32.',
    'Translate into Japanese: The train arrives at the station at noon.',
    'List five prime numbers greater than 100, with one number per line.',
    'Describe the difference between a stack and a queue.',
    'Repeat the words red green blue in that order ten times.',
    'Return JSON with keys name, count, enabled and values sample, 3, true.',
]


def main():
    root = Path(sys.argv[1])
    url = sys.argv[2]
    model = sys.argv[3]
    responses = []
    for prompt in PROMPTS:
        body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'repetition_penalty': 1, 'max_tokens': 64, 'min_tokens': 64, 'ignore_eos': True, 'logprobs': True, 'stream': False}
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
        choice = data['choices'][0]
        if not choice.get('logprobs', {}).get('content'):
            raise ValueError('token logprobs required for greedy parity evidence')
        responses.append({'request': body, 'response': data})
    result = {'schema_version': 'qwen-greedy-parity/v1', 'phase': 'after official benchmark', 'prompt_sha256': hashlib.sha256(json.dumps(PROMPTS, ensure_ascii=False).encode()).hexdigest(), 'responses': responses, 'training_data': False}
    (root / 'greedy-parity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
