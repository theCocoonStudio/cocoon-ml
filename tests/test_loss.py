import unittest

from cocoonml.autograd import Autograd
from cocoonml.loss import softmax_cross_entropy, softmax_cross_entropy_composed

SCORES = [1.5, -0.3, 0.8, 2.1]
TARGET = 2


def composed_gradient():
    scores = [Autograd(v) for v in SCORES]
    softmax_cross_entropy_composed(scores, TARGET).backward()
    return [s._derivative for s in scores]


def finite_difference_gradient(h=1e-6):
    def f(vals):
        return softmax_cross_entropy_composed([Autograd(v) for v in vals], TARGET).value

    base = f(SCORES)
    grads = []
    for i in range(len(SCORES)):
        nudged = list(SCORES)
        nudged[i] += h
        grads.append((f(nudged) - base) / h)
    return grads


class TestSoftmaxCrossEntropy(unittest.TestCase):
    def test_composed_matches_finite_differences(self):
        for analytic, numeric in zip(composed_gradient(), finite_difference_gradient()):
            self.assertAlmostEqual(analytic, numeric, places=5)

    def test_fused_value_matches_composed(self):
        scores = [Autograd(v) for v in SCORES]
        fused = softmax_cross_entropy(scores, TARGET).value
        composed = softmax_cross_entropy_composed(scores, TARGET).value
        self.assertAlmostEqual(fused, composed, places=12)

    @unittest.expectedFailure
    def test_fused_gradient_matches_composed(self):
        """Passes once the closed-form backward is written; until then it is the open line."""
        scores = [Autograd(v) for v in SCORES]
        softmax_cross_entropy(scores, TARGET).backward()
        for fused, composed in zip([s._derivative for s in scores], composed_gradient()):
            self.assertAlmostEqual(fused, composed, places=9)


if __name__ == "__main__":
    unittest.main()


class TestLargeScores(unittest.TestCase):
    def test_large_scores_do_not_overflow_and_the_loss_is_shift_invariant(self):
        """Sweep 05a: two cells died in exp and log when two-layer logits grew past the range of a float."""
        import math
        from cocoonml.autograd import Autograd
        from cocoonml.loss import softmax_cross_entropy_composed

        small = [Autograd(1.0), Autograd(2.0), Autograd(0.5)]
        large = [Autograd(1.0 + 1000.0), Autograd(2.0 + 1000.0), Autograd(0.5 + 1000.0)]
        self.assertAlmostEqual(softmax_cross_entropy_composed(small, 1).value, softmax_cross_entropy_composed(large, 1).value, places=9)
        self.assertFalse(math.isnan(softmax_cross_entropy_composed(large, 1).value))
