import unittest

import numpy as np

from cocoonml.array_attention import ArrayAttention
from cocoonml.circuits import circuit_scores, induction_score, previous_token_score


def rows_to(n, target):
    """An n x n causal attention matrix where every position attends fully to target(i) (or itself when target(i) is out of range)."""
    w = np.zeros((n, n))
    for i in range(n):
        j = target(i)
        w[i, j if 0 <= j <= i else i] = 1.0
    return w


class TestTheTwoScores(unittest.TestCase):
    def test_the_previous_token_score_is_one_when_every_position_looks_back_one(self):
        self.assertEqual(previous_token_score(rows_to(6, lambda i: i - 1)), 1.0)
        self.assertEqual(previous_token_score(rows_to(6, lambda i: i)), 0.0)
        self.assertIsNone(previous_token_score(rows_to(1, lambda i: i)))

    def test_the_induction_score_reads_the_position_after_the_previous_occurrence(self):
        tokens = [1, 2, 3, 1, 2, 3]
        induction = rows_to(6, lambda i: {3: 1, 4: 2, 5: 3}.get(i, i))  # positions 3, 4, 5 attend to what followed their earlier occurrence
        self.assertEqual(induction_score(tokens, induction), 1.0)
        self.assertEqual(induction_score(tokens, rows_to(6, lambda i: i)), 0.0)
        self.assertIsNone(induction_score([1, 2, 3], rows_to(3, lambda i: i)))  # no token recurs
        self.assertIsNone(induction_score([1, 1, 1], rows_to(3, lambda i: i)))  # adjacent repeats have nothing to read

    def test_excluded_tokens_are_not_queries(self):
        tokens = [9, 1, 9, 1, 9]  # 9 the separator
        w = rows_to(5, lambda i: {3: 2, 4: 3}.get(i, i))
        self.assertAlmostEqual(induction_score(tokens, w), 2 / 3)  # positions 2 and 4 (the separator) qualify too; position 2 attends to itself
        self.assertEqual(induction_score(tokens, w, exclude={9}), 1.0)  # only position 3 (token 1) is a query
        self.assertIsNone(induction_score([9, 1, 9, 2, 9], w, exclude={9}))


class TestTheModelExposesItsAttention(unittest.TestCase):
    def test_attention_is_causal_and_normalised_per_layer(self):
        model = ArrayAttention(vocab=6, width=8, length=12, seed=1, separator=5, layers=3, scale="derived")
        tokens = [1, 2, 5, 3, 4, 5, 1, 2]
        layers = model.attention(tokens)
        self.assertEqual(len(layers), 3)
        for weights in layers:
            self.assertEqual(weights.shape, (8, 8))
            self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))
            self.assertTrue(np.allclose(np.triu(weights, 1), 0.0))

    def test_circuit_scores_average_over_sequences_per_layer(self):
        model = ArrayAttention(vocab=6, width=8, length=12, seed=1, separator=5, layers=2, scale="derived")
        scores = circuit_scores(model, [[1, 2, 5, 1, 2, 5], [3, 4, 5, 3, 4, 5]], exclude={5})
        self.assertEqual(len(scores["previous_token"]), 2)
        self.assertEqual(len(scores["induction"]), 2)
        for value in scores["previous_token"] + scores["induction"]:
            self.assertTrue(0.0 <= value <= 1.0)
        self.assertEqual(circuit_scores(model, []), {"previous_token": [], "induction": []})


if __name__ == "__main__":
    unittest.main()
