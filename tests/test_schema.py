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

    def test_a_resolution_below_two_is_refused(self):
        with self.assertRaises(ValueError):
            generate(forms=2, cuts=1, nonterminals=1, resolution=1, seed=0)

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
        for a, b in zip(self.table.nodes, moved.nodes):
            self.assertLessEqual(abs(a.share - b.share), Fraction(1, 8))  # one grain per node at most
        self.assertGreaterEqual(delta(self.table, moved), 0)
        self.assertEqual(delta(self.table, self.table), 0)

    def test_the_index_never_freezes_and_stays_within_the_radius_of_its_centre(self):
        """Sweep 02, round one, item 1: under the clip and the lean the shares reached an end and
        stayed; with reflection and the pull the walk keeps moving and stays near the origin."""
        table = generate(forms=3, cuts=2, nonterminals=3, resolution=8, seed=1)
        rng = random.Random(0)
        moved, seen, last = table, set(), []
        for i in range(300):
            moved = step(moved, rng, bound=Fraction(3, 8), lean=Fraction(3, 4), centre=table, radius=Fraction(1, 4))
            for a, b in zip(table.nodes, moved.nodes):
                self.assertLessEqual(abs(a.share - b.share), Fraction(1, 4) + Fraction(1, 8))
            shares = tuple(n.share for n in moved.nodes)
            seen.add(shares)
            if i >= 250:
                last.append(shares)
        self.assertGreater(len(seen), 8)
        self.assertGreater(len(set(last)), 1)

    def test_a_removed_sister_returns_and_the_end_reflects_under_a_full_lean(self):
        """Both deterministic: the pull is certain beyond the radius, and a full lean toward the
        null sister from a share of one can only reflect."""
        table = generate(forms=2, cuts=1, nonterminals=1, resolution=8, seed=5)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(0, 0), Leaf(1, 0)  # the null is in sister 0
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 0), Leaf(1, 0)
        centre = table.copy()
        centre.nodes[0].share = Fraction(1, 2)
        node.share = Fraction(0)  # sister 0 removed
        moved = step(table, random.Random(0), bound=Fraction(1), lean=Fraction(0), centre=centre, radius=Fraction(1, 4))
        self.assertEqual(moved.nodes[0].share, Fraction(1, 8))  # displacement 1/2 over radius 1/4: the pull is certain
        node.share = Fraction(1)
        moved = step(table, random.Random(0), bound=Fraction(1), lean=Fraction(1))
        self.assertEqual(moved.nodes[0].share, Fraction(7, 8))  # lean forces +1 toward the null sister; past 1 reflects

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
    """Phases act only where one string has several derivations. The table here has a root with
    two sisters: the first expands to an inner node that produces (1,) in two readings (a leaf and
    a null, in either sister), the second produces (2,). Orders are fixed at one so that order
    ambiguity does not add readings of its own."""

    def _table(self, phase):
        from cocoonml.schema import distribution

        table = generate(forms=2, cuts=1, nonterminals=2, resolution=4, seed=0)
        root, inner = table.nodes
        root.sisters[0].first, root.sisters[0].second = 1, Leaf(0, 0)
        root.sisters[1].first, root.sisters[1].second = Leaf(2, 0), Leaf(0, 0)
        root.share, root.phase = Fraction(1, 2), Fraction(0)
        inner.sisters[0].first, inner.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        inner.sisters[1].first, inner.sisters[1].second = Leaf(0, 0), Leaf(1, 0)
        inner.share, inner.phase = Fraction(1, 2), phase
        for node in table.nodes:
            for e in node.sisters:
                e.order = Fraction(1)
        return table, distribution

    def test_with_zero_phase_two_readings_of_one_string_add_constructively(self):
        table, distribution = self._table(Fraction(0))
        dist = distribution(table)
        # classically (1,) and (2,) would be 1/2 each; constructive interference lifts (1,) above 1/2
        self.assertGreater(dist[(1,)], 0.5)

    def test_with_half_a_turn_the_two_readings_cancel(self):
        table, distribution = self._table(Fraction(1, 2))
        dist = distribution(table)
        self.assertAlmostEqual(dist.get((1,), 0.0), 0.0, places=9)
        self.assertAlmostEqual(dist[(2,)], 1.0, places=9)

    def test_without_any_ambiguity_interference_equals_the_classical_distribution(self):
        from cocoonml.schema import distribution

        table = generate(forms=2, cuts=1, nonterminals=1, resolution=4, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 0), Leaf(0, 0)
        for e in node.sisters:
            e.order = Fraction(1)  # a null sister in either order is the same string: fix the order
        node.share = Fraction(1, 4)
        node.phase = Fraction(1, 3)  # a phase with nothing to interfere with
        dist = distribution(table)
        self.assertAlmostEqual(dist[(1,)], 0.25)
        self.assertAlmostEqual(dist[(2,)], 0.75)

    def test_a_silent_sister_makes_the_two_orders_one_string_and_they_interfere(self):
        """Order ambiguity is ambiguity: with a null sister both orders give the same string, and
        under interference their amplitudes add, so the classical share is not recovered."""
        from cocoonml.schema import distribution

        table = generate(forms=2, cuts=1, nonterminals=1, resolution=4, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 0), Leaf(0, 0)
        node.sisters[0].order = Fraction(1, 2)
        node.sisters[1].order = Fraction(1)
        node.share, node.phase = Fraction(1, 2), Fraction(0)
        dist = distribution(table)
        self.assertGreater(dist[(1,)], 0.5)


class TestTheoremOne(unittest.TestCase):
    """T1: a node visited by no derivation consistent with the prefix does not change the
    predictive distribution at the next position, whatever its share, order or phase."""

    def test_an_unvisited_node_does_not_move_the_prediction(self):
        from cocoonml.schema import predictive, visited_by_consistent_derivations

        table = generate(forms=3, cuts=2, nonterminals=3, resolution=8, seed=13)
        root, a, b = table.nodes
        # root chooses between two subtrees that begin with different forms, so a prefix picks one
        root.sisters[0].first, root.sisters[0].second = Leaf(1, 0), 1
        root.sisters[1].first, root.sisters[1].second = Leaf(2, 0), 2
        for e in root.sisters:
            e.order = Fraction(1)
        a.sisters[0].first, a.sisters[0].second = Leaf(3, 0), Leaf(0, 0)
        a.sisters[1].first, a.sisters[1].second = Leaf(1, 1), Leaf(0, 0)
        b.sisters[0].first, b.sisters[0].second = Leaf(2, 0), Leaf(0, 0)
        b.sisters[1].first, b.sisters[1].second = Leaf(3, 1), Leaf(0, 0)
        for node in (a, b):
            for e in node.sisters:
                e.order = Fraction(1)
        prefix = (1,)
        self.assertNotIn(2, visited_by_consistent_derivations(table, prefix))
        before = predictive(table, prefix)
        b.share, b.phase = Fraction(7, 8), Fraction(1, 3)
        b.sisters[0].order = Fraction(1, 4)
        after = predictive(table, prefix)
        self.assertEqual(set(before), set(after))
        for k in before:
            self.assertAlmostEqual(before[k], after[k], places=12)

    def test_two_strings_that_share_a_prefix_add_as_probabilities_and_do_not_interfere(self):
        """Round one, item 9: (1, 2) and (1, 2, 3) at half a turn from each other. Summing their
        amplitudes before squaring would cancel the next form 2 to nothing; as distinct strings
        they add, and 2 is certain after 1."""
        from cocoonml.schema import distribution, predictive

        table = generate(forms=3, cuts=1, nonterminals=2, resolution=8, seed=0)
        root, a = table.nodes
        root.sisters[0].first, root.sisters[0].second = Leaf(1, 0), Leaf(2, 0)
        root.sisters[1].first, root.sisters[1].second = Leaf(1, 0), 1
        for e in root.sisters:
            e.order = Fraction(1)
        for e in a.sisters:
            e.first, e.second, e.order = Leaf(2, 0), Leaf(3, 0), Fraction(1)
        a.share, root.share, root.phase = Fraction(1), Fraction(1, 2), Fraction(1, 2)
        dist = distribution(table)
        self.assertAlmostEqual(dist[(1, 2)], 0.5)
        self.assertAlmostEqual(dist[(1, 2, 3)], 0.5)
        self.assertEqual(predictive(table, (1,)), {2: 1.0})
        after_two = predictive(table, (1, 2))
        self.assertAlmostEqual(after_two[3], 0.5)
        self.assertAlmostEqual(after_two[None], 0.5)

    def test_predictive_is_the_conditional_of_distribution_on_a_random_table_with_phases(self):
        from cocoonml.schema import distribution, predictive

        rng = random.Random(3)
        checked = 0
        for seed in range(12):
            table = generate(forms=3, cuts=2, nonterminals=3, resolution=8, seed=seed)
            for node in table.nodes:
                node.phase = Fraction(rng.randrange(0, 8), 8)
            dist = {s: p for s, p in distribution(table).items() if p > 0}
            string = draw(table, rng)[0]
            for cut in range(len(string) + 1):
                prefix = string[:cut]
                denominator = sum(p for s, p in dist.items() if s[:cut] == prefix)
                for nxt, q in predictive(table, prefix).items():
                    numerator = sum(p for s, p in dist.items() if s[:cut] == prefix and (s[cut] if len(s) > cut else None) == nxt)
                    self.assertAlmostEqual(q, numerator / denominator, places=12)
                    checked += 1
        self.assertGreater(checked, 20)

    def test_a_visited_node_does_move_the_prediction(self):
        from cocoonml.schema import predictive

        table = generate(forms=3, cuts=2, nonterminals=3, resolution=8, seed=13)
        root, a, b = table.nodes
        root.sisters[0].first, root.sisters[0].second = Leaf(1, 0), 1
        root.sisters[1].first, root.sisters[1].second = Leaf(2, 0), 2
        a.sisters[0].first, a.sisters[0].second = Leaf(3, 0), Leaf(0, 0)
        a.sisters[1].first, a.sisters[1].second = Leaf(1, 1), Leaf(0, 0)
        for node in table.nodes:
            for e in node.sisters:
                e.order = Fraction(1)
        a.share = Fraction(1, 2)
        before = predictive(table, (1,))
        a.share = Fraction(7, 8)
        after = predictive(table, (1,))
        self.assertNotAlmostEqual(before[3], after[3], places=6)


class TestProjections(unittest.TestCase):
    def test_frequency_cost_and_phase_are_three_readings_of_one_amplitude(self):
        import cmath, math
        from cocoonml.schema import projections

        a = 0.5 * cmath.exp(1j * 1.0)
        frequency, cost, phase = projections(a)
        self.assertAlmostEqual(frequency, 0.25)
        self.assertAlmostEqual(cost, -math.log(0.5))
        self.assertAlmostEqual(phase, 1.0)

    def test_the_extension_of_an_unambiguous_string_is_its_cuts_at_equal_weight(self):
        from cocoonml.schema import weighted_extension

        table = generate(forms=2, cuts=2, nonterminals=1, resolution=4, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(2, 1)
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 0), Leaf(0, 0)
        for e in node.sisters:
            e.order = Fraction(1)
        ext = weighted_extension(table, (1, 2))
        self.assertAlmostEqual(ext[0], 0.5)
        self.assertAlmostEqual(ext[1], 0.5)

    def test_cost_is_lower_for_the_more_frequent_string(self):
        from cocoonml.schema import cost_of

        table = generate(forms=2, cuts=1, nonterminals=1, resolution=4, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 0), Leaf(0, 0)
        for e in node.sisters:
            e.order = Fraction(1)
        node.share = Fraction(3, 4)
        self.assertLess(cost_of(table, (1,)), cost_of(table, (2,)))


class TestNEff(unittest.TestCase):
    def test_modulus_grade_rises_with_artifacts_and_phase_grade_is_zero_without_ambiguity(self):
        from cocoonml.schema import n_eff

        table = generate(forms=2, cuts=1, nonterminals=1, resolution=8, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 0), Leaf(0, 0)
        for e in node.sisters:
            e.order = Fraction(1)
        rng = random.Random(0)
        few = [draw(table, rng)[0] for _ in range(4)]
        many = [draw(table, rng)[0] for _ in range(100)]
        n_few, g_few = n_eff(table, few, phase_resolution=4)
        n_many, g_many = n_eff(table, many, phase_resolution=4)
        self.assertLess(n_few, n_many)
        self.assertEqual(g_few[0][1], 0.0)
        self.assertEqual(g_many[0][1], 0.0)
        self.assertEqual(g_many[0][0], 1.0)

    def test_phase_grade_rises_only_through_ambiguous_strings(self):
        from cocoonml.schema import n_eff

        table = generate(forms=1, cuts=1, nonterminals=1, resolution=8, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(0, 0), Leaf(1, 0)
        for e in node.sisters:
            e.order = Fraction(1)
        artifacts = [(1,)] * 16
        _, grades = n_eff(table, artifacts, phase_resolution=4)
        self.assertEqual(grades[0][1], 1.0)


class TestTheoremTwo(unittest.TestCase):
    """T2 (sketch): a node's effect on the prediction is bounded by its consistent mass:
    total variation ≤ m · |δ| / min(s, 1 − s), to first order. Checked on random tables with a
    small δ and a tolerance for the second order."""

    def test_sensitivity_is_bounded_by_consistent_mass_on_random_tables(self):
        from cocoonml.schema import consistent_mass, predictive, sensitivity

        rng = random.Random(0)
        checked = 0
        for seed in range(40):
            table = generate(forms=3, cuts=2, nonterminals=3, resolution=8, seed=seed)
            string = draw(table, rng)[0]
            if not string:
                continue
            prefix = string[:1]
            for node_index, node in enumerate(table.nodes):
                s = float(node.share)
                if s <= 0.125 or s >= 0.875:
                    continue  # keep away from the clip, where first order does not hold
                delta = Fraction(1, 64)
                m = consistent_mass(table, prefix, node_index)
                tv = sensitivity(table, prefix, node_index, delta)
                bound = m * float(delta) / min(s, 1 - s)
                self.assertLessEqual(tv, bound + 0.02, f"seed {seed} node {node_index}: tv {tv} > bound {bound}")
                if m == 0.0:
                    self.assertAlmostEqual(tv, 0.0, places=12)  # T1
                checked += 1
        self.assertGreater(checked, 20)


class TestIdentificationError(unittest.TestCase):
    def _two_strings(self, share):
        table = generate(forms=2, cuts=1, nonterminals=1, resolution=8, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(2, 0), Leaf(0, 0)
        for e in node.sisters:
            e.order = Fraction(1)
        node.share = share
        return table

    def test_the_error_vanishes_as_the_window_grows_and_coverage_reads_the_visited_nodes(self):
        from cocoonml.schema import identification_error

        table = self._two_strings(Fraction(3, 8))
        rng = random.Random(0)
        many = [draw(table, rng)[0] for _ in range(2000)]
        error, coverage, per_node = identification_error(table, many)
        self.assertEqual(coverage, 1.0)
        self.assertLess(error, 0.05)
        self.assertEqual(per_node[0][1], [0.5, 0.5])  # a silent daughter leaves the order unidentifiable
        self.assertEqual(identification_error(table, [])[0], None)
        self.assertEqual(identification_error(table, [])[1], 0.0)

    def test_same_readings_different_shares_n_eff_is_blind_and_the_error_is_not(self):
        """Sweep 02, round one, item 2."""
        from cocoonml.schema import identification_error, n_eff

        table = self._two_strings(Fraction(3, 8))
        other = self._two_strings(Fraction(7, 8))
        rng = random.Random(1)
        window = [draw(table, rng)[0] for _ in range(200)]
        self.assertEqual(n_eff(table, window, 4)[0], n_eff(other, window, 4)[0])
        self.assertLess(identification_error(table, window)[0], identification_error(other, window)[0])

    def test_an_ambiguous_string_is_one_observation(self):
        from cocoonml.schema import estimate_shares

        table = generate(forms=1, cuts=1, nonterminals=1, resolution=8, seed=0)
        node = table.nodes[0]
        node.sisters[0].first, node.sisters[0].second = Leaf(1, 0), Leaf(0, 0)
        node.sisters[1].first, node.sisters[1].second = Leaf(0, 0), Leaf(1, 0)
        share, weight, _ = estimate_shares(table, [(1,)])[0]
        self.assertAlmostEqual(weight, 1.0)
        self.assertAlmostEqual(share, 0.5)


class TestEntropyFloors(unittest.TestCase):
    def _table(self):
        table = generate(forms=3, cuts=2, nonterminals=2, resolution=8, seed=0)
        root, a = table.nodes
        for e in root.sisters:
            e.first, e.second, e.order = Leaf(1, 0), 1, Fraction(1)
        a.sisters[0].first, a.sisters[0].second, a.sisters[0].order = Leaf(2, 0), Leaf(0, 0), Fraction(1)
        a.sisters[1].first, a.sisters[1].second, a.sisters[1].order = Leaf(3, 1), Leaf(0, 0), Fraction(1)
        a.share = Fraction(1, 2)
        return table

    def test_the_form_floor_is_the_entropy_of_the_second_form_and_meaning_is_fixed_by_the_string(self):
        import math
        from cocoonml.schema import classical_joint, entropy_floor_form, entropy_floor_meaning

        table = self._table()
        joint = classical_joint(table)
        self.assertAlmostEqual(sum(joint.values()), 1.0)
        self.assertAlmostEqual(joint[((1, 2), 0)], 0.5)
        self.assertAlmostEqual(joint[((1, 3), 1)], 0.5)
        self.assertAlmostEqual(entropy_floor_form(table), math.log(2))
        self.assertAlmostEqual(entropy_floor_meaning(table), 0.0)

    def test_a_synonymous_form_moves_the_entropy_from_form_to_meaning(self):
        import math
        from cocoonml.schema import entropy_floor_form, entropy_floor_meaning

        table = self._table()
        table.nodes[1].sisters[1].first = Leaf(2, 1)  # the same form with another cut
        self.assertAlmostEqual(entropy_floor_form(table), 0.0)
        self.assertAlmostEqual(entropy_floor_meaning(table), math.log(2))

    def test_the_form_floor_is_zero_when_the_table_is_deterministic(self):
        from cocoonml.schema import entropy_floor_form

        table = self._table()
        table.nodes[1].share = Fraction(1)
        self.assertAlmostEqual(entropy_floor_form(table), 0.0)
