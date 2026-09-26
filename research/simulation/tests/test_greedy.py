import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('compare_greedy', Path(__file__).resolve().parents[1] / 'compare_greedy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GreedyTests(unittest.TestCase):
    def setUp(self):
        response = {'request': {'temperature': 0}, 'response': {'choices': [{'logprobs': {'content': [{'token': 'x', 'bytes': [120], 'logprob': -.1}] * 64}, 'message': {'content': 'x' * 64}, 'finish_reason': 'length'}]}}
        self.baseline = {'prompt_sha256': 'fixture', 'responses': [copy.deepcopy(response) for _ in range(8)]}

    def test_equal_text_with_different_tokens_is_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        candidate['responses'][0]['response']['choices'][0]['logprobs']['content'][0]['token'] = 'different'
        self.assertFalse(module.compare(self.baseline, candidate)['all_match'])

    def test_float_logprob_variation_does_not_change_token_identity(self):
        candidate = copy.deepcopy(self.baseline)
        candidate['responses'][0]['response']['choices'][0]['logprobs']['content'][0]['logprob'] = -.100001
        self.assertTrue(module.compare(self.baseline, candidate)['all_match'])

    def test_partial_response_is_not_parity_evidence(self):
        candidate = copy.deepcopy(self.baseline)
        candidate['responses'][0]['response']['choices'][0]['logprobs']['content'].pop()
        with self.assertRaisesRegex(ValueError, '64 output tokens'):
            module.compare(self.baseline, candidate)
