import unittest
from fractions import Fraction

from cocoonml.harness import pack, run
from cocoonml.schema import delta, generate


class TestHarness(unittest.TestCase):
    def test_pack_joins_artifacts_with_the_separator_and_pads(self):
        tokens, targets = pack([(1, 2), (3,)], separator=9, length=7)
        self.assertEqual(tokens, [1, 2, 9, 3, 9, 9, 9])
        self.assertEqual(targets, [2, 9, 3, 9, 9, 9, 9])

    def test_the_apparatus_runs_end_to_end_at_toy_size(self):
        """A smoke test: the pieces fit; nothing about the numbers is asserted beyond their shape."""
        table = generate(forms=3, cuts=2, nonterminals=2, resolution=4, seed=11)
        train_losses, same, after, moved = run(
            table, model_seed=1, width=3, length=6, train_steps=3, batch_size=2, per_context=2,
            lr=0.3, drift_steps=2, bound=Fraction(1, 2), lean=Fraction(1, 2), eval_count=2, seed=5,
        )
        self.assertEqual(len(train_losses), 3)
        self.assertEqual(len(same), 6)
        self.assertEqual(len(after), 6)
        self.assertGreaterEqual(delta(table, moved), 0)


if __name__ == "__main__":
    unittest.main()


class TestMeaning(unittest.TestCase):
    def test_pack_meaning_puts_the_extension_at_the_separator(self):
        from cocoonml.harness import pack_meaning

        tokens, targets = pack_meaning([((1, 2), (0, 1, 1)), ((3,), (1,))], separator=9, length=6, extension_base=10)
        self.assertEqual(tokens, [1, 2, 9, 3, 9, 9])
        self.assertEqual(targets, [2, 9, 12, 9, 11, 9])

    def test_meaning_curve_has_one_entry_per_artifact_in_the_context(self):
        import random
        from cocoonml.attention import Attention
        from cocoonml.harness import meaning_loss_at_separators

        table = generate(forms=3, cuts=2, nonterminals=2, resolution=4, seed=11)
        model = Attention(vocab=20, width=3, length=8, seed=1)
        curve = meaning_loss_at_separators(model, table, count=3, per_context=2, separator=5, length=8, extension_base=10, rng=random.Random(0))
        self.assertGreaterEqual(len(curve), 1)
        self.assertLessEqual(len(curve), 2)
