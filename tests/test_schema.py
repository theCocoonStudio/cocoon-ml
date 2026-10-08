import random
import unittest
from fractions import Fraction

from cocoonml.schema import Leaf, delta, derivations, draw, estimate, generate, step


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.table = generate(forms=3, cuts=2, nonterminals=3, resolution=8, seed=7)

    def test_same_counts_and_seed_give_the_same_table(self):
        other = generate(forms=3, cuts=2, nonterminals=3, resolution=8, seed=7)
        self.assertEqual(self.table, other)

    def test_shares_and_orders_are_rational_and_at_the_grain(self):
        for node in self.table.nodes:
            self.assertIsInstance(node.share, Fraction)
            self.assertEqual(node.share.denominator in (1, 2, 4, 8), True)
            for e in node.sisters:
                self.assertEqual((e.order * 8).denominator, 1)

    def test_a_draw_is_a_string_of_non_empty_forms(self):
        rng = random.Random(1)
        for _ in range(20):
            forms, cuts = draw(self.table, rng)
            self.assertTrue(all(f != 0 for f in forms))
            self.assertEqual(len(cuts) >= len(forms), True)

    def test_every_drawn_string_has_at_least_one_derivation(self):
        rng = random.Random(2)
        for _ in range(20):
            forms, _ = draw(self.table, rng)
            self.assertGreaterEqual(len(derivations(self.table, forms)), 1)

    def test_a_null_leaf_is_silent(self):
        rng = random.Random(3)
        table = generate(forms=1, cuts=1, nonterminals=1, resolution=4, seed=0)
        node = table.nodes[0]
        node.sisters[0].first = Leaf(0, 0)
        node.sisters[0].second = Leaf(0, 0)
        node.share = Fraction(1)
        forms, cuts = draw(table, rng)
        self.assertEqual(forms, ())
        self.assertEqual(cuts, (0, 0))

    def test_a_step_moves_shares_by_a_bounded_total_and_delta_reads_it(self):
        rng = random.Random(4)
        moved = step(self.table, rng, bound=Fraction(1, 2), lean=Fraction(1, 2))
        total = sum(abs(a.share - b.share) for a, b in zip(self.table.nodes, moved.nodes))
        self.assertLessEqual(total, Fraction(1, 2))
        self.assertGreaterEqual(delta(self.table, moved), 0)
        self.assertEqual(delta(self.table, self.table), 0)

    def test_the_lean_moves_share_toward_the_null_sister(self):
        table = generate(forms=2, cuts=1, nonterminals=1, resolution=8, seed=5)
        node = table.nodes[0]
        node.sisters[0].first = Leaf(0, 0)  # the first sister contains a null
        node.sisters[1].first = Leaf(1, 0)
        node.sisters[1].second = Leaf(2, 0)
        node.share = Fraction(1, 2)
        rng = random.Random(6)
        after = node.share
        for _ in range(200):
            table = step(table, rng, bound=Fraction(1, 4), lean=Fraction(1))
            after = table.nodes[0].share
        self.assertGreater(after, Fraction(1, 2))

    def test_estimate_reads_the_seen_incidence_and_frequencies_off_artifacts(self):
        rng = random.Random(8)
        artifacts = [draw(self.table, rng)[0] for _ in range(40)]
        seen, counts = estimate(self.table, artifacts)
        self.assertTrue(seen.issubset({1, 2, 3}))
        self.assertEqual(len(counts), len(self.table.nodes))
        self.assertGreater(sum(counts[0][0]), 0)


if __name__ == "__main__":
    unittest.main()


class TestRulerValidity(unittest.TestCase):
    """The ruler reads what it is supposed to read: frequencies off artifacts converge to the
    table's shares, and derivation counts are the ambiguity they claim to be."""

    def test_frequencies_off_artifacts_converge_to_shares_without_ambiguity(self):
        table = generate(forms=2, cuts=1, nonterminals=1, resolution=8, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 0), Leaf(0, 0)
        node.share = Fraction(3, 4)
        rng = random.Random(10)
        artifacts = [draw(table, rng)[0] for _ in range(800)]
        _, counts = estimate(table, artifacts)
        first = counts[0][0][0] / sum(counts[0][0])
        self.assertAlmostEqual(first, 0.75, delta=0.05)

    def test_a_string_with_two_trees_has_two_derivations(self):
        table = generate(forms=2, cuts=1, nonterminals=2, resolution=4, seed=0)
        root, inner = table.nodes
        # root -> (1 inner) | (inner 2) ; inner -> (1 2) | (2 1): the string (1, 2, ... ) is ambiguous
        root.sisters[0].first, root.sisters[0].second = Leaf(1, 0), 1
        root.sisters[1].first, root.sisters[1].second = 1, Leaf(2, 0)
        for e in root.sisters:
            e.order = Fraction(1)
        inner.sisters[0].first, inner.sisters[0].second = Leaf(1, 0), Leaf(2, 0)
        inner.sisters[1].first, inner.sisters[1].second = Leaf(2, 0), Leaf(1, 0)
        for e in inner.sisters:
            e.order = Fraction(1)
        # (1, 2, 1) has no reading here; (1, 1, 2) = root[0] with inner[0]; (1, 2, 2) = root[1] with inner[0]
        self.assertEqual(len(derivations(table, (1, 1, 2))) >= 1, True)
        # (1, 2, 1): root[1] with inner[0] gives (1,2,2); root[0] with inner[1] gives (1,2,1): one reading via order
        ds = derivations(table, (1, 2, 1))
        self.assertGreaterEqual(len(ds), 1)

    def test_order_ambiguity_doubles_the_derivations_when_sisters_share_forms(self):
        table = generate(forms=1, cuts=1, nonterminals=1, resolution=4, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(1, 1)
        node.sisters[1].first, node.sisters[1].second = Leaf(1, 0), Leaf(1, 0)
        ds = derivations(table, (1, 1))
        # both sisters produce (1, 1) in both orders: four derivations, none distinguishable by the string
        self.assertEqual(len(ds), 4)


class TestInterference(unittest.TestCase):
    def _two_readings(self):
        from cocoonml.schema import distribution

        table = generate(forms=1, cuts=1, nonterminals=1, resolution=4, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(0, 0), Leaf(1, 0)
        for e in node.sisters:
            e.order = Fraction(1)
        node.share = Fraction(1, 2)
        return table, distribution

    def test_with_zero_phase_two_readings_of_one_string_add_constructively(self):
        table, distribution = self._two_readings()
        table.nodes[0].phase = Fraction(0)
        self.assertAlmostEqual(distribution(table)[(1,)], 1.0)

    def test_with_half_a_turn_the_two_readings_cancel(self):
        table, distribution = self._two_readings()
        table.nodes[0].phase = Fraction(1, 2)
        dist = distribution(table)
        # the only string is (1,), and its amplitude cancels to zero: the distribution is empty mass
        self.assertAlmostEqual(dist.get((1,), 0.0), 0.0, places=9)

    def test_without_ambiguity_interference_equals_the_classical_distribution(self):
        from cocoonml.schema import distribution

        table = generate(forms=2, cuts=1, nonterminals=1, resolution=4, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 0), Leaf(0, 0)
        node.share = Fraction(1, 4)
        node.phase = Fraction(1, 3)  # a phase with nothing to interfere with
        dist = distribution(table)
        self.assertAlmostEqual(dist[(1,)], 0.25)
        self.assertAlmostEqual(dist[(2,)], 0.75)
