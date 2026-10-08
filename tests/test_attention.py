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


class TestWithinArtifactPosition(unittest.TestCase):
    def test_offsets_reset_at_the_separator(self):
        from cocoonml.attention import Attention

        model = Attention(vocab=6, width=2, length=8, seed=0, separator=5)
        self.assertEqual(model.offsets([1, 2, 5, 3, 5, 5, 4, 1]), [0, 1, 0, 1, 0, 0, 1, 2])

    def test_the_other_weights_are_unchanged_by_adding_the_within_embedding(self):
        from cocoonml.attention import Attention

        plain = Attention(vocab=6, width=2, length=4, seed=3)
        within = Attention(vocab=6, width=2, length=4, seed=3, separator=5)
        self.assertEqual([p.value for p in plain.parameters()], [p.value for p in within.parameters()][: sum(1 for _ in plain.parameters())])

    def test_within_gradient_matches_finite_differences(self):
        from cocoonml.attention import Attention

        model = Attention(vocab=6, width=3, length=6, seed=0, separator=5)
        tokens, targets = [1, 2, 5, 3, 4, 5], [2, 5, 3, 4, 5, 5]
        for p in model.parameters():
            p._derivative = 0.0
        model.loss(tokens, targets).backward()
        h = 1e-5
        checked = 0
        for row in model.within[:3]:
            for p in row:
                analytic, saved = p._derivative, p.value
                p.value = saved + h
                up = model.loss(tokens, targets).value
                p.value = saved - h
                down = model.loss(tokens, targets).value
                p.value = saved
                self.assertAlmostEqual(analytic, (up - down) / (2 * h), places=5)
                checked += 1
        self.assertEqual(checked, 9)


class TestTwoLayers(unittest.TestCase):
    def test_one_layer_is_unchanged_and_two_layers_share_its_first_layer(self):
        from cocoonml.attention import Attention

        one = Attention(vocab=6, width=3, length=5, seed=2, separator=5)
        two = Attention(vocab=6, width=3, length=5, seed=2, separator=5, layers=2)
        first = [p.value for p in one.parameters()]
        self.assertEqual([p.value for p in two.parameters()][: len(first)], first)
        self.assertEqual(sum(1 for _ in two.parameters()) - len(first), 3 * 3 * 3)
        tokens = [1, 2, 5, 3, 4]
        l1, a1 = one.forward(tokens)
        l2, a2 = two.forward(tokens)
        self.assertEqual(len(l2), 5)
        self.assertEqual(len(a2[0]), 3)
        self.assertNotAlmostEqual(l1[4][0].value, l2[4][0].value, places=6)

    def test_two_layer_gradient_matches_finite_differences_in_every_block(self):
        from cocoonml.attention import Attention

        model = Attention(vocab=6, width=3, length=6, seed=0, separator=5, layers=2)
        tokens, targets = [1, 2, 5, 3, 4, 5], [2, 5, 3, 4, 5, 5]
        for p in model.parameters():
            p._derivative = 0.0
        model.loss(tokens, targets).backward()
        h = 1e-5
        q2, k2, v2 = model.more[0]
        for p in (model.query[0][0], model.key[1][2], model.value[2][1], q2[0][1], k2[2][2], v2[1][0], model.within[1][1], model.embed[3][0], model.readout[4][2]):
            analytic, saved = p._derivative, p.value
            p.value = saved + h
            up = model.loss(tokens, targets).value
            p.value = saved - h
            down = model.loss(tokens, targets).value
            p.value = saved
            self.assertAlmostEqual(analytic, (up - down) / (2 * h), places=5)
