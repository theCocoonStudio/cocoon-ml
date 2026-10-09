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

        tokens, targets = pack_meaning([((1, 2), (1, 2)), ((3,), (0, 1))], separator=9, length=6, extension_base=10)
        self.assertEqual(tokens, [1, 2, 9, 3, 9, 9])
        self.assertEqual(targets, [2, 9, 16, 9, 10, 9])

    def test_meaning_curve_has_one_entry_per_artifact_in_the_context(self):
        import random
        from cocoonml.attention import Attention
        from cocoonml.harness import meaning_loss_at_separators

        table = generate(forms=3, cuts=2, nonterminals=2, resolution=4, seed=11)
        model = Attention(vocab=20, width=3, length=8, seed=1)
        curve = meaning_loss_at_separators(model, table, count=3, per_context=2, separator=5, length=8, extension_base=10, rng=random.Random(0))
        self.assertGreaterEqual(len(curve), 1)
        self.assertLessEqual(len(curve), 2)


class TestRoundOneFixes(unittest.TestCase):
    def test_pack_meaning_drops_an_artifact_that_does_not_fit_whole(self):
        from cocoonml.harness import pack_meaning

        tokens, targets = pack_meaning([((1, 2), (1, 1)), ((3, 4, 5), (0, 3))], separator=9, length=5, extension_base=10)
        self.assertEqual(tokens, [1, 2, 9, 9, 9])
        self.assertEqual(targets, [2, 9, 13, 9, 9])

    def test_form_loss_by_rank_excludes_padding(self):
        import random
        from cocoonml.attention import Attention
        from cocoonml.harness import form_loss_by_rank

        table = generate(forms=3, cuts=2, nonterminals=2, resolution=4, seed=11)
        model = Attention(vocab=6, width=3, length=8, seed=1)
        curve = form_loss_by_rank(model, table, count=4, per_context=2, separator=5, rng=random.Random(0))
        self.assertLessEqual(len(curve), 2)

    def test_delta_magnitude_grows_with_drift_where_the_flag_delta_saturates(self):
        import random
        from cocoonml.schema import delta, delta_magnitude, step

        table = generate(forms=3, cuts=2, nonterminals=3, resolution=8, seed=1)
        rng = random.Random(0)
        one = step(table, rng, Fraction(1, 8), Fraction(1, 2))
        eight = one
        for _ in range(7):
            eight = step(eight, rng, Fraction(1, 8), Fraction(1, 2))
        self.assertGreaterEqual(delta(table, eight), delta(table, one) * 0.9)
        self.assertGreater(delta_magnitude(table, eight), delta_magnitude(table, one))


class TestFilledContexts(unittest.TestCase):
    """Sweep 02, round one, items 3 and 4: contexts that always fill, curves only where every
    context reaches."""

    def _one_form_table(self):
        import random
        from cocoonml.schema import Leaf

        table = generate(forms=2, cuts=1, nonterminals=1, resolution=8, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 0), Leaf(0, 0)
        for e in node.sisters:
            e.order = Fraction(1)
        return table, random.Random(0)

    def test_a_filled_context_is_full_and_counts_its_artifacts(self):
        from cocoonml.harness import contexts, filled

        table, rng = self._one_form_table()
        tokens, targets, n = filled(table, 9, 8, rng)
        self.assertEqual(n, 4)  # one form and a separator each
        self.assertEqual(tokens[-1], 9)
        self.assertNotEqual(tokens[-2], 9)
        for tokens, _ in contexts(table, 5, 1, 9, 8, rng, fill=True):
            self.assertNotEqual(tokens[-2], 9)

    def test_curves_with_fill_stop_at_the_rank_every_context_reaches(self):
        import random
        from cocoonml.attention import Attention
        from cocoonml.harness import meaning_loss_at_separators

        table, rng = self._one_form_table()
        model = Attention(vocab=14, width=3, length=8, seed=1)
        curve = meaning_loss_at_separators(model, table, count=3, per_context=1, separator=9, length=8, extension_base=10, rng=rng, fill=True)
        self.assertEqual(len(curve), 4)


class TestPairsInTheInput(unittest.TestCase):
    """Sweep 03, round one, item 2: the extension token follows its separator as a token the
    model reads; the target at the separator is still the extension."""

    def test_pack_pairs_puts_the_extension_after_the_separator_and_targets_it_at_the_separator(self):
        from cocoonml.harness import pack_pairs

        tokens, targets = pack_pairs([((1, 2), (1, 1)), ((3,), (0, 1))], separator=9, length=8, extension_base=10)
        self.assertEqual(tokens, [1, 2, 9, 13, 3, 9, 10, 9])
        self.assertEqual(targets, [2, 9, 13, 3, 9, 10, 9, 9])

    def test_pack_pairs_drops_an_artifact_whose_pair_does_not_fit(self):
        from cocoonml.harness import pack_pairs

        tokens, _ = pack_pairs([((1, 2), (1, 1)), ((3, 4), (0, 2))], separator=9, length=7, extension_base=10)
        self.assertEqual(tokens, [1, 2, 9, 13, 9, 9, 9])

    def test_filled_with_pairs_counts_the_extension_slot_and_the_curves_read_the_separators(self):
        import random
        from cocoonml.attention import Attention
        from cocoonml.harness import filled, form_loss_by_rank, meaning_loss_at_separators
        from cocoonml.schema import Leaf

        table = generate(forms=2, cuts=2, nonterminals=1, resolution=8, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 1), Leaf(0, 0)
        for e in node.sisters:
            e.order = Fraction(1)
        rng = random.Random(0)
        tokens, targets, n = filled(table, 9, 10, rng, extension_base=10, pairs=True)
        self.assertEqual(n, 3)  # a form, a separator and an extension token each: nine of ten slots
        self.assertEqual(tokens[-1], 9)  # the tenth slot is padding
        self.assertGreaterEqual(tokens[-2], 10)  # an extension token (the code's value depends on the multiset)
        model = Attention(vocab=16, width=3, length=10, seed=1)
        meaning = meaning_loss_at_separators(model, table, 3, 0, 9, 10, 10, rng, fill=True, pairs=True)
        self.assertEqual(len(meaning), 3)
        form = form_loss_by_rank(model, table, 3, 0, 9, rng, fill=True, extension_base=10, pairs=True)
        self.assertEqual(form, [])  # one-form artifacts have no form inside them to predict


class TestPlateau(unittest.TestCase):
    class Stub:
        def __init__(self, sequence):
            self.sequence, self.calls = sequence, 0

        def train_step(self, batch, lr):
            self.calls += 1
            return self.sequence(self.calls)

    def test_a_flat_loss_stops_after_two_windows_and_a_falling_one_runs_to_the_cap(self):
        from cocoonml.harness import train_until_plateau

        flat = self.Stub(lambda n: 1.0)
        self.assertEqual(len(train_until_plateau(flat, lambda: None, 0.1, window=10, cap=500)[0]), 20)
        halving = self.Stub(lambda n: 2.0 ** -n)
        self.assertEqual(len(train_until_plateau(halving, lambda: None, 0.1, window=10, cap=60)[0]), 60)

    def test_with_a_held_out_evaluation_the_rule_reads_the_evaluations_not_the_batches(self):
        from cocoonml.harness import train_until_plateau

        noisy = self.Stub(lambda n: 1.0 + (0.5 if n % 2 else -0.5))  # batch losses that never settle
        readings = iter([2.0, 1.5, 1.2, 1.19, 1.0])
        losses, evaluations = train_until_plateau(noisy, lambda: None, 0.1, window=10, cap=500, evaluate=lambda: next(readings))
        self.assertEqual(evaluations, [2.0, 1.5, 1.2, 1.19])  # stops when a reading improves by under one percent
        self.assertEqual(len(losses), 40)

    def test_patience_and_minimum_hold_the_rule_back(self):
        from cocoonml.harness import train_until_plateau

        flat = self.Stub(lambda n: 1.0)
        self.assertEqual(len(train_until_plateau(flat, lambda: None, 0.1, window=10, cap=500, patience=3)[0]), 40)
        flat = self.Stub(lambda n: 1.0)
        self.assertEqual(len(train_until_plateau(flat, lambda: None, 0.1, window=10, cap=500, minimum=75)[0]), 80)
        readings = iter([2.0, 1.99, 1.0, 0.999, 0.998, 0.5])
        flat = self.Stub(lambda n: 1.0)
        _, evaluations = train_until_plateau(flat, lambda: None, 0.1, window=10, cap=500, evaluate=lambda: next(readings), patience=2)
        self.assertEqual(evaluations, [2.0, 1.99, 1.0, 0.999, 0.998])  # one stale reading is forgiven; two in a row are not


class TestScrambledPairs(unittest.TestCase):
    def test_scrambling_permutes_the_extension_tokens_and_nothing_else(self):
        import random
        from cocoonml.harness import scramble_extensions

        tokens = [1, 2, 9, 11, 3, 9, 12, 2, 1, 9, 10, 9]
        out = scramble_extensions(tokens, random.Random(3), 10)
        self.assertEqual([t for t in out if t < 10], [t for t in tokens if t < 10])
        self.assertEqual([i for i, t in enumerate(out) if t >= 10], [3, 6, 10])
        self.assertEqual(sorted(t for t in out if t >= 10), [10, 11, 12])
        self.assertEqual(tokens, [1, 2, 9, 11, 3, 9, 12, 2, 1, 9, 10, 9])  # the input is not mutated


class TestExtensionCode(unittest.TestCase):
    def test_the_code_is_injective_and_dense_over_every_multiset_up_to_a_size(self):
        from itertools import combinations_with_replacement
        from cocoonml.harness import extension_code, extension_vocabulary

        for cut_count in (2, 3):
            codes = {}
            for size in range(1, 6):
                for cuts in combinations_with_replacement(range(cut_count), size):
                    vector = tuple(cuts.count(c) for c in range(cut_count))
                    codes[cuts] = extension_code(vector, cut_count)
            self.assertEqual(len(set(codes.values())), len(codes))  # injective
            self.assertEqual(sorted(codes.values()), list(range(extension_vocabulary(cut_count, 5))))  # dense

    def test_the_sum_of_cuts_was_not_injective_and_the_code_is(self):
        from cocoonml.harness import extension_code

        a, b = (2, 1), (4, 1)  # the two extensions sweep 05's token collapsed onto "sum 1": {0,0,1} and {0,0,0,0,1}
        self.assertNotEqual(extension_code(a, 2), extension_code(b, 2))

    def test_the_vocabulary_bounds_every_token_a_table_can_produce(self):
        import random
        from cocoonml.harness import extension_vocabulary, filled
        from cocoonml.schema import max_extension_size

        table = generate(forms=4, cuts=2, nonterminals=3, resolution=4, seed=1)
        base, size = 10, max_extension_size(table)
        self.assertEqual(size, 5)
        self.assertEqual(extension_vocabulary(2, size), 20)
        rng = random.Random(0)
        for _ in range(50):
            tokens, _, _ = filled(table, 9, 40, rng, extension_base=base, pairs=True)
            for t in tokens:
                self.assertLess(t, base + 20)


class TestStream(unittest.TestCase):
    def test_zero_steps_per_context_is_the_stationary_table(self):
        import random
        from fractions import Fraction
        from cocoonml.harness import stream
        from cocoonml.schema import delta_magnitude

        table = generate(forms=4, cuts=2, nonterminals=3, resolution=4, seed=1)
        s = stream(table, random.Random(0), Fraction(1, 4), Fraction(3, 4), Fraction(1, 4), 9, 20, 10, pairs=True, steps_per_context=0)
        for _ in range(20):
            _, _, current = next(s)
            self.assertEqual(delta_magnitude(table, current), 0)

    def test_the_walk_moves_and_stays_within_the_training_radius(self):
        import random
        from fractions import Fraction
        from cocoonml.harness import stream
        from cocoonml.schema import delta_magnitude

        table = generate(forms=4, cuts=2, nonterminals=3, resolution=8, seed=1)
        radius = Fraction(2, 8)
        s = stream(table, random.Random(0), Fraction(3, 8), Fraction(3, 4), radius, 9, 20, 10, pairs=True, steps_per_context=3)
        moved = 0
        for _ in range(300):
            tokens, targets, current = next(s)
            self.assertEqual(len(tokens), 20)
            if delta_magnitude(table, current) > 0:
                moved += 1
            for node, origin in zip(current.nodes, table.nodes):
                self.assertLessEqual(abs(node.share - origin.share), radius + Fraction(1, 8))
                for e, oe in zip(node.sisters, origin.sisters):
                    self.assertLessEqual(abs(e.order - oe.order), radius + Fraction(1, 8))
        self.assertGreater(moved, 100)

    def test_steps_between_contexts_are_a_rate_not_a_metronome(self):
        import random
        from cocoonml.harness import _poisson

        rng = random.Random(0)
        self.assertEqual([_poisson(rng, 0) for _ in range(20)], [0] * 20)
        draws = [_poisson(rng, 1) for _ in range(4000)]
        self.assertGreater(len(set(draws)), 2)  # not always one
        self.assertAlmostEqual(sum(draws) / len(draws), 1.0, delta=0.08)
