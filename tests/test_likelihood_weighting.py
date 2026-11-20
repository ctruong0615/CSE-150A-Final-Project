"""Analytic and exact-inference checks for likelihood weighting."""
import copy
import unittest
from bayesnet.exact_inference import posterior
from bayesnet.likelihood_weighting import sample
from bayesnet.rejection_sampling import sample as rejection
from tests.test_gibbs_sampling import TINY_MODEL, LECTURE_MODEL


class LikelihoodWeightingTests(unittest.TestCase):
    def test_no_evidence_matches_prior_sampling(self):
        result = sample(TINY_MODEL, 'A', samples=10000, seed=17)
        self.assertEqual(result['posterior'], rejection(TINY_MODEL, 'A', samples=10000, seed=17)['posterior'])
        self.assertAlmostEqual(result['effective_sample_size'], 10000)

    def test_child_evidence_reweights_parent(self):
        result = sample(TINY_MODEL, 'A', {'B': 'yes'}, samples=30000, seed=17)
        self.assertAlmostEqual(result['posterior']['yes'], 12/13, delta=0.015)
        self.assertLess(result['effective_sample_size'], 30000)
        self.assertEqual(result['positive_weight_samples'], 30000)

    def test_fixed_root_evidence(self):
        result = sample(TINY_MODEL, 'B', {'A': 'yes'}, samples=20000, seed=17)
        self.assertAlmostEqual(result['posterior']['yes'], 0.8, delta=0.02)
        self.assertAlmostEqual(result['effective_sample_size'], 20000)

    def test_lecture_network(self):
        evidence = {'S': '1', 'G': '1'}
        result = sample(LECTURE_MODEL, 'R', evidence, samples=50000, seed=42)
        self.assertAlmostEqual(result['posterior']['1'], posterior(LECTURE_MODEL, 'R', evidence)['1'], delta=0.02)

    def test_all_zero_weights(self):
        model = copy.deepcopy(TINY_MODEL)
        model['cpts']['A'][0]['probabilities'] = {'no': 1, 'yes': 0}
        result = sample(model, 'B', {'A': 'yes'}, samples=100, seed=42)
        self.assertIsNone(result['posterior'])
        self.assertEqual(result['status'], 'zero_total_weight')
        self.assertEqual(result['effective_sample_size'], 0)

    def test_some_zero_weights(self):
        model = copy.deepcopy(TINY_MODEL)
        model['cpts']['B'][0]['probabilities'] = {'no': 1, 'yes': 0}
        result = sample(model, 'A', {'B': 'yes'}, samples=1000, seed=42)
        self.assertEqual(result['posterior']['yes'], 1)
        self.assertGreater(result['zero_weight_samples'], 0)
        self.assertAlmostEqual(result['effective_sample_size'], result['positive_weight_samples'])

    def test_log_weights_avoid_underflow(self):
        model = copy.deepcopy(TINY_MODEL)
        evidence = {}
        for i in range(4):
            node = f'E{i}'
            model['nodes'].append(node)
            model['states'][node] = ['no', 'yes']
            model['parents'][node] = []
            model['cpts'][node] = [{'parent_values': [], 'probabilities': {'no': 1.0, 'yes': 1e-200}}]
            evidence[node] = 'yes'
        result = sample(model, 'A', evidence, samples=1000, seed=42)
        self.assertEqual(result['status'], 'ok')
        self.assertLess(result['log_total_weight'], -1000)
        self.assertAlmostEqual(result['effective_sample_size'], 1000)

    def test_repeatable_and_no_mutation(self):
        model = copy.deepcopy(TINY_MODEL)
        self.assertEqual(sample(model, 'A', {'B': 'yes'}, seed=9, samples=100), sample(model, 'A', {'B': 'yes'}, seed=9, samples=100))
        self.assertEqual(model, TINY_MODEL)

    def test_invalid_inputs(self):
        for n in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                sample(TINY_MODEL, 'A', samples=n)
        for query, evidence in [('X', {}), ('A', {'A': 'yes'}), ('A', {'B': 'bad'})]:
            with self.assertRaises(ValueError):
                sample(TINY_MODEL, query, evidence)


if __name__ == '__main__':
    unittest.main()
