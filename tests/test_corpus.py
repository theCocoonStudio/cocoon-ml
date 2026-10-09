import random
import unittest
from fractions import Fraction

from cocoonml.corpus import distance_from_ball, estimated_distance, pack_window, produce, scramble_order, windows
from cocoonml.schema import generate


def two_tables():
    return [generate(forms=3, cuts=2, nonterminals=2, resolution=8, seed=s) for s in (1, 2)]


class TestCorpus(unittest.TestCase):
    def test_a_corpus_is_finite_ordered_and_carries_its_provenance_apart_from_the_forms(self):
        corpus = produce(two_tables(), random.Random(0), Fraction(1, 8), Fraction(3, 4), Fraction(2, 8), count=60)
        self.assertEqual(len(corpus), 60)
        tables = {a.table for a in corpus.artifacts}
        self.assertEqual(tables, {0, 1})  # both languages produced
        for a in corpus.artifacts:
            self.assertTrue(all(f > 0 for f in a.forms))  # forms only, no null, no extension token
            self.assertEqual(sum(a.extension), sum(a.extension))  # an extension vector travels beside the forms, not in them
            self.assertLess(a.state, len(corpus.walks[a.table]))
        states = [a.state for a in corpus.artifacts if a.table == 0]
        self.assertEqual(states, sorted(states))  # within a language, states succeed in production order
        self.assertGreater(max(states), 0)  # and the walk moved

    def test_zero_steps_and_one_table_is_the_stationary_corner(self):
        corpus = produce(two_tables()[:1], random.Random(0), Fraction(1, 8), Fraction(3, 4), Fraction(2, 8), count=30, steps_mean=0)
        self.assertTrue(all(a.state == 0 and a.table == 0 for a in corpus.artifacts))
        self.assertEqual(len(corpus.walks[0]), 1)

    def test_windows_hold_consecutive_whole_artifacts_mixing_tables_and_states_with_no_extension(self):
        corpus = produce(two_tables(), random.Random(1), Fraction(1, 8), Fraction(3, 4), Fraction(2, 8), count=80)
        ws = windows(corpus, separator=9, length=24)
        self.assertGreater(len(ws), 1)
        covered = 0
        for tokens, targets, used in ws:
            self.assertEqual(len(tokens), 24)
            self.assertEqual(targets[:-1], tokens[1:])
            self.assertTrue(all(t <= 9 for t in tokens))  # forms and the separator only
            self.assertEqual(sum(len(a.forms) + 1 for a in used) <= 24, True)
            covered += len(used)
        self.assertEqual(covered, len(corpus))  # every artifact is in exactly one window
        mixed = [w for w in ws if len({a.table for a in w[2]}) > 1]
        self.assertGreater(len(mixed), 0)  # tables mix within a window

    def test_scrambling_the_order_keeps_the_multiset_and_breaks_the_succession(self):
        corpus = produce(two_tables(), random.Random(2), Fraction(1, 8), Fraction(3, 4), Fraction(2, 8), count=40)
        _, _, used = pack_window(corpus.artifacts, 9, 40)
        shuffled = scramble_order(used, random.Random(0))
        self.assertEqual(sorted(a.forms for a in shuffled), sorted(a.forms for a in used))
        self.assertNotEqual([a.state for a in shuffled], [a.state for a in used])

    def test_distances_the_truth_s_and_the_reader_s(self):
        tables = two_tables()
        corpus = produce(tables[:1], random.Random(3), Fraction(1, 8), Fraction(3, 4), Fraction(2, 8), count=200)
        ball = corpus.walks[0]
        self.assertEqual(distance_from_ball(ball[-1], ball), 0.0)  # a state of the ball is at distance zero from it
        far = ball[-1]
        rng = random.Random(9)
        from cocoonml.schema import step
        for _ in range(60):
            far = step(far, rng, Fraction(1, 8), Fraction(3, 4))  # a free walk away from the ball
        self.assertGreater(distance_from_ball(far, ball), 0.0)
        reading = [a.forms for a in produce([far], random.Random(4), Fraction(1, 8), Fraction(3, 4), None, count=60, steps_mean=0).artifacts]
        d = estimated_distance(tables[0], reading, corpus.forms())
        self.assertIsNotNone(d)
        self.assertGreaterEqual(d, 0.0)
        same = estimated_distance(tables[0], corpus.forms(), corpus.forms())
        self.assertEqual(same, 0.0)


class TestCorpusReaders(unittest.TestCase):
    def test_readers_run_on_corpus_windows_and_the_control_is_zero_for_an_order_blind_reading(self):
        from cocoonml.attention import Attention
        from cocoonml.corpus import form_losses_by_rank, order_control, probe_rows, whole_rank

        tables = two_tables()
        corpus = produce(tables, random.Random(5), Fraction(1, 8), Fraction(3, 4), Fraction(2, 8), count=60)
        ws = windows(corpus, separator=4, length=20)
        model = Attention(vocab=5, width=3, length=20, seed=1, separator=4)
        whole = whole_rank(ws)
        self.assertGreater(whole, 0)
        curve = form_losses_by_rank(model, ws, 4, whole)
        self.assertEqual(len(curve), whole)
        self.assertTrue(all(c > 0 for c in curve))
        control = order_control(model, ws, 4, random.Random(0), whole)
        self.assertEqual(len(control), whole)
        rows = probe_rows(model, ws, corpus, corpus.walks[0] + corpus.walks[1], 4)
        self.assertEqual(len(rows), sum(len(u) for _, _, u in ws))
        self.assertTrue(all(y == 0.0 for _, y in rows))  # every training artifact's state is in the ball
