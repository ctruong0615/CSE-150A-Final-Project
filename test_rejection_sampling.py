"""Checks against analytic probabilities and the project's exact implementation."""
import copy
import unittest

from exact_inference import posterior
from rejection_sampling import sample
from test_gibbs_sampling import TINY_MODEL, LECTURE_MODEL


class RejectionSamplingTests(unittest.TestCase):
    def test_unconditional_matches_prior(self):
        result = sample(TINY_MODEL, 'A', samples=30000, seed=17)
        self.assertEqual(result['accepted_samples'], 30000)
        self.assertAlmostEqual(result['posterior']['yes'], 0.6, delta=0.015)

    def test_child_evidence_updates_parent(self):
        # Bayes: P(A=yes | B=yes) = .6*.8 / (.6*.8 + .4*.1) = 12/13.
        result = sample(TINY_MODEL, 'A', {'B': 'yes'}, samples=50000, seed=17)
        self.assertAlmostEqual(result['posterior']['yes'], 12/13, delta=0.015)
        self.assertAlmostEqual(result['acceptance_rate'], 0.52, delta=0.015)
        self.assertEqual(sum(result['counts'].values()), result['accepted_samples'])

    def test_lecture_network_matches_exact(self):
        evidence = {'S': '1', 'G': '1'}
        result = sample(LECTURE_MODEL, 'R', evidence, samples=50000, seed=42)
        self.assertAlmostEqual(result['posterior']['1'], posterior(LECTURE_MODEL, 'R', evidence)['1'], delta=0.02)

    def test_zero_acceptances_are_not_zero_posterior(self):
        model = copy.deepcopy(TINY_MODEL)
        model['cpts']['A'][0]['probabilities'] = {'no': 1.0, 'yes': 0.0}
        result = sample(model, 'B', {'A': 'yes'}, samples=100, seed=9)
        self.assertIsNone(result['posterior'])
        self.assertEqual(result['status'], 'no_accepted_samples')
        self.assertEqual(result['accepted_samples'], 0)

    def test_seed_repeats_without_mutating_model(self):
        original = copy.deepcopy(TINY_MODEL)
        self.assertEqual(sample(TINY_MODEL, 'A', samples=100, seed=9), sample(TINY_MODEL, 'A', samples=100, seed=9))
        self.assertEqual(TINY_MODEL, original)

    def test_unsorted_nodes_work(self):
        model = copy.deepcopy(TINY_MODEL)
        model['nodes'].reverse()
        self.assertEqual(sample(model, 'A', samples=100, seed=9), sample(TINY_MODEL, 'A', samples=100, seed=9))

    def test_invalid_inputs(self):
        for n in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                sample(TINY_MODEL, 'A', samples=n)
        for query, evidence in [('unknown', {}), ('A', {'A': 'yes'}), ('A', {'B': 'invalid'})]:
            with self.assertRaises(ValueError):
                sample(TINY_MODEL, query, evidence)

    def test_malformed_models_rejected(self):
        model = copy.deepcopy(TINY_MODEL)
        model['cpts']['B'].pop()
        with self.assertRaises(ValueError):
            sample(model, 'A')
        model = copy.deepcopy(TINY_MODEL)
        model['parents']['A'] = ['B']
        with self.assertRaises(ValueError):
            sample(model, 'A')
        model = copy.deepcopy(TINY_MODEL)
        model['cpts']['A'][0]['probabilities']['yes'] = -1
        with self.assertRaises(ValueError):
            sample(model, 'A')


if __name__ == '__main__':
    unittest.main()
