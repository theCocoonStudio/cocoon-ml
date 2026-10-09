import math
import unittest

from cocoonml.array_attention import ArrayAttention
from cocoonml.attention import Attention

FULL = ([1, 2, 5, 3, 4, 5], [2, 5, 3, 4, 5, 5])
SHORT = ([1, 2, 5, 3], [2, 5, 3, 4])
OTHER = ([3, 5, 1, 2, 5, 4], [5, 1, 2, 5, 4, 1])


def models(seed=0, separator=5, layers=1):
    args = dict(vocab=6, width=3, length=6, seed=seed, separator=separator, layers=layers)
    return Attention(**args), ArrayAttention(**args)


def scalar_loss_and_gradients(scalar, batch):
    """The scalar train_step's loss and gradients without its update."""
    for p in scalar.parameters():
        p._derivative = 0.0
    total = None
    for tokens, targets in batch:
        l = scalar.loss(tokens, targets)
        total = l if total is None else total.add(l)
    total = total / float(len(batch))
    total.backward()
    return total.value, [p._derivative for p in scalar.parameters()]


class TestArrayAttention(unittest.TestCase):
    def assertListsAlmostEqual(self, a, b, places):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=places)

    def test_initial_weights_equal_the_scalar_model(self):
        for seed in (0, 3):
            for separator in (None, 5):
                for layers in (1, 3):
                    scalar, array = models(seed=seed, separator=separator, layers=layers)
                    self.assertEqual(array.flat_parameters(), [p.value for p in scalar.parameters()])

    def test_offsets_match_the_scalar(self):
        tokens = [1, 2, 5, 3, 5, 5, 4, 1]
        scalar = Attention(vocab=6, width=2, length=8, seed=0, separator=5)
        array = ArrayAttention(vocab=6, width=2, length=8, seed=0, separator=5)
        self.assertEqual(array.offsets(tokens), scalar.offsets(tokens))
        self.assertEqual(array.offsets(tokens), [0, 1, 0, 1, 0, 0, 1, 2])

    def test_loss_equals_the_scalar_loss(self):
        for layers in (1, 2, 3):
            for separator in (None, 5):
                scalar, array = models(separator=separator, layers=layers)
                for tokens, targets in (FULL, SHORT):
                    self.assertAlmostEqual(array.loss(tokens, targets), scalar.loss(tokens, targets).value, places=9)

    def test_position_losses_sum_to_the_loss(self):
        scalar, array = models(layers=2)
        for tokens, targets in (FULL, SHORT):
            losses = array.position_losses(tokens, targets)
            self.assertEqual(len(losses), len(tokens))
            self.assertAlmostEqual(sum(losses) / len(losses), array.loss(tokens, targets), places=12)
            logits, _ = scalar.forward(tokens)
            for i, (lg, t) in enumerate(zip(logits, targets)):
                m = max(x.value for x in lg)
                total = sum(math.exp(x.value - m) for x in lg)
                self.assertAlmostEqual(losses[i], -(lg[t].value - m) + math.log(total), places=9)

    def test_gradients_equal_the_scalar_gradients(self):
        tokens, targets = FULL
        for layers in (2, 3):
            scalar, array = models(separator=5, layers=layers)
            for p in scalar.parameters():
                p._derivative = 0.0
            scalar.loss(tokens, targets).backward()
            array.loss_and_gradients([(tokens, targets)])
            self.assertListsAlmostEqual(array.flat_gradients(), [p._derivative for p in scalar.parameters()], places=9)

    def test_batched_gradients_equal_the_scalar_batch(self):
        batch = [FULL, OTHER]
        scalar, array = models(separator=5, layers=2)
        expected_loss, expected_grads = scalar_loss_and_gradients(scalar, batch)
        loss = array.loss_and_gradients(batch)
        self.assertAlmostEqual(loss, expected_loss, places=9)
        self.assertListsAlmostEqual(array.flat_gradients(), expected_grads, places=9)
        scalar_step = scalar.train_step(batch, lr=0.3)
        array_step = array.train_step(batch, lr=0.3)
        self.assertAlmostEqual(array_step, scalar_step, places=9)
        self.assertListsAlmostEqual(array.flat_parameters(), [p.value for p in scalar.parameters()], places=9)

    def test_mixed_length_batch(self):
        _, array = models(separator=5, layers=2)
        full_loss = array.loss_and_gradients([FULL])
        full_grads = array.flat_gradients()
        short_loss = array.loss_and_gradients([SHORT])
        short_grads = array.flat_gradients()
        loss = array.loss_and_gradients([FULL, SHORT])
        self.assertAlmostEqual(loss, (full_loss + short_loss) / 2, places=12)
        self.assertListsAlmostEqual(array.flat_gradients(), [(a + b) / 2 for a, b in zip(full_grads, short_grads)], places=12)

    def test_gradients_match_central_differences(self):
        tokens, targets = FULL
        _, model = models(separator=5, layers=2)
        model.loss_and_gradients([(tokens, targets)])
        grads = model.grads
        q2, k2, v2 = model.more[0]
        dq2, dk2, dv2 = grads["more"][0]
        entries = [
            (model.embed, grads["embed"], (3, 0)),
            (model.embed, grads["embed"], (1, 2)),
            (model.position, grads["position"], (2, 1)),
            (model.query, grads["query"], (0, 0)),
            (model.key, grads["key"], (1, 2)),
            (model.value, grads["value"], (2, 1)),
            (model.readout, grads["readout"], (4, 2)),
            (model.readout, grads["readout"], (0, 1)),
            (model.within, grads["within"], (1, 1)),
            (q2, dq2, (0, 1)),
            (k2, dk2, (2, 2)),
            (v2, dv2, (1, 0)),
        ]
        h = 1e-5
        checked = 0
        for block, grad, index in entries:
            analytic = grad[index]
            saved = block[index]
            block[index] = saved + h
            up = model.loss(tokens, targets)
            block[index] = saved - h
            down = model.loss(tokens, targets)
            block[index] = saved
            self.assertAlmostEqual(analytic, (up - down) / (2 * h), places=5)
            checked += 1
        self.assertEqual(checked, 12)

    def test_probe_equals_the_scalar_probe(self):
        scalar, array = models(separator=5, layers=2)
        for tokens, _ in (FULL, SHORT):
            rows = array.probe(tokens)
            expected = scalar.probe(tokens)
            self.assertEqual(len(rows), len(expected))
            for row, want in zip(rows, expected):
                self.assertListsAlmostEqual(row, want, places=9)
            self.assertTrue(all(isinstance(x, float) for row in rows for x in row))

    def test_forward_shapes(self):
        _, array = models(separator=5, layers=2)
        for tokens, _ in (FULL, SHORT):
            logits, attended = array.forward(tokens)
            self.assertEqual(logits.shape, (len(tokens), 6))
            self.assertEqual(attended.shape, (len(tokens), 3))


if __name__ == "__main__":
    unittest.main()


class TestScale(unittest.TestCase):
    def test_default_scale_is_the_scalar_model_and_a_given_scale_bounds_the_weights(self):
        from cocoonml.attention import Attention
        from cocoonml.array_attention import ArrayAttention

        scalar = Attention(vocab=6, width=3, length=6, seed=5, separator=5, layers=2)
        default = ArrayAttention(vocab=6, width=3, length=6, seed=5, separator=5, layers=2)
        self.assertEqual(default.flat_parameters(), [p.value for p in scalar.parameters()])
        small = ArrayAttention(vocab=6, width=3, length=6, seed=5, separator=5, layers=2, scale=0.1)
        self.assertTrue(all(abs(w) <= 0.1 for w in small.flat_parameters()))
        self.assertNotEqual(small.flat_parameters(), default.flat_parameters())
        self.assertEqual(len(small.flat_parameters()), len(default.flat_parameters()))


class TestDerivedScale(unittest.TestCase):
    def test_the_derived_scale_reads_off_every_count(self):
        import math
        from cocoonml.array_attention import ArrayAttention

        m = ArrayAttention(vocab=6, width=64, length=16, seed=0, separator=5, layers=4, scale="derived")
        bound = lambda a: abs(a).max()
        self.assertLessEqual(bound(m.embed), math.sqrt(3 / (3 * 64)) + 1e-12)  # three embeddings summed
        self.assertLessEqual(bound(m.value), math.sqrt(3 / (4 * 64)) + 1e-12)  # four layers add
        self.assertLessEqual(bound(m.query), math.sqrt(3 / 8) + 1e-12)  # √64 = 8 in the scores
        self.assertLessEqual(bound(m.readout), math.sqrt(3 / 64) + 1e-12)
        self.assertGreater(bound(m.query), bound(m.readout))  # the scores' scale is the loosest
        plain = ArrayAttention(vocab=6, width=64, length=16, seed=0, separator=5, layers=4)
        self.assertEqual(len(m.flat_parameters()), len(plain.flat_parameters()))
