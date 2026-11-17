"""Small Bayesian network checks for Gibbs sampling."""
import unittest

from exact_inference import posterior as exact_posterior
from gibbs_sampling import sample

TINY_MODEL = {
    "nodes": ["A", "B"],
    "states": {"A": ["no", "yes"], "B": ["no", "yes"]},
    "parents": {"A": [], "B": ["A"]},
    "cpts": {
        "A": [{"parent_values": [], "probabilities": {"no": 0.4, "yes": 0.6}}],
        "B": [
            {"parent_values": ["no"], "probabilities": {"no": 0.9, "yes": 0.1}},
            {"parent_values": ["yes"], "probabilities": {"no": 0.2, "yes": 0.8}},
        ],
    },
}

# Gibbs example: C -> R, C -> S, and (S, R) -> G.
LECTURE_MODEL = {
    "nodes": ["C", "R", "S", "G"],
    "states": {node: ["0", "1"] for node in ("C", "R", "S", "G")},
    "parents": {"C": [], "R": ["C"], "S": ["C"], "G": ["S", "R"]},
    "cpts": {
        "C": [{"parent_values": [], "probabilities": {"0": 0.5, "1": 0.5}}],
        "R": [
            {"parent_values": ["0"], "probabilities": {"0": 0.8, "1": 0.2}},
            {"parent_values": ["1"], "probabilities": {"0": 0.2, "1": 0.8}},
        ],
        "S": [
            {"parent_values": ["0"], "probabilities": {"0": 0.9, "1": 0.1}},
            {"parent_values": ["1"], "probabilities": {"0": 0.5, "1": 0.5}},
        ],
        "G": [
            {"parent_values": ["0", "0"], "probabilities": {"0": 0.99, "1": 0.01}},
            {"parent_values": ["0", "1"], "probabilities": {"0": 0.1, "1": 0.9}},
            {"parent_values": ["1", "0"], "probabilities": {"0": 0.1, "1": 0.9}},
            {"parent_values": ["1", "1"], "probabilities": {"0": 0.01, "1": 0.99}},
        ],
    },
}

class GibbsSamplingTests(unittest.TestCase):
    def test_matches_exact_on_tiny_network(self):
        exact = exact_posterior(TINY_MODEL, "B", {"A": "yes"})
        result = sample(TINY_MODEL, "B", {"A": "yes"}, samples=20_000,
                        burn_in=100, seed=17)
        for state in exact:
            self.assertAlmostEqual(result["posterior"][state], exact[state], delta=0.02)

    def test_matches_lecture_example_including_child_factors(self):
        evidence = {"S": "1", "G": "1"}
        exact = exact_posterior(LECTURE_MODEL, "R", evidence)
        result = sample(LECTURE_MODEL, "R", evidence, samples=100_000,
                        burn_in=1_000, seed=42)
        self.assertAlmostEqual(result["posterior"]["1"], exact["1"], delta=0.02)

    def test_evidence_is_clamped_and_seed_repeats(self):
        first = sample(TINY_MODEL, "B", {"A": "no"}, samples=100, seed=9)
        second = sample(TINY_MODEL, "B", {"A": "no"}, samples=100, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["posterior"]["yes"], 0.1, delta=0.12)

    def test_rejects_invalid_sample_settings(self):
        with self.assertRaises(ValueError):
            sample(TINY_MODEL, "B", samples=0)

if __name__ == "__main__":
    unittest.main()
