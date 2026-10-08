import random
import unittest

from cocoonml.attention import Attention


class TestAttention(unittest.TestCase):
    def test_forward_shapes(self):
        model = Attention(vocab=4, width=3, length=4, seed=0)
        logits, attended = model.forward([1, 2, 3, 1])
        self.assertEqual(len(logits), 4)
        self.assertEqual(len(logits[0]), 4)
        self.assertEqual(len(attended[2]), 3)

    def test_analytic_gradient_matches_finite_differences(self):
        model = Attention(vocab=3, width=2, length=3, seed=1)
        tokens, targets = [1, 2, 1], [2, 1, 1]
        for p in model.parameters():
            p._derivative = 0.0
        model.loss(tokens, targets).backward()
        checked = 0
        h = 1e-5
        for p in list(model.parameters())[:12]:
            analytic = p._derivative
            saved = p.value
            p.value = saved + h
            up = model.loss(tokens, targets).value
            p.value = saved - h
            down = model.loss(tokens, targets).value
            p.value = saved
            numeric = (up - down) / (2 * h)
            self.assertAlmostEqual(analytic, numeric, places=5)
            checked += 1
        self.assertEqual(checked, 12)

    def test_learns_to_copy_the_previous_token(self):
        """A smoke test of the apparatus, not a result: loss falls on a copy task."""
        rng = random.Random(2)
        model = Attention(vocab=4, width=4, length=4, seed=3)

        def example():
            tokens = [rng.randrange(1, 4) for _ in range(4)]
            targets = [tokens[0]] + tokens[:-1]  # predict the previous token
            return tokens, targets

        batch = [example() for _ in range(6)]
        first = model.train_step(batch, lr=0.5)
        for _ in range(40):
            last = model.train_step(batch, lr=0.5)
        self.assertLess(last, first * 0.7)

    def test_probe_returns_plain_numbers(self):
        model = Attention(vocab=3, width=2, length=3, seed=4)
        rows = model.probe([1, 2, 2])
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(isinstance(x, float) for row in rows for x in row))


if __name__ == "__main__":
    unittest.main()
