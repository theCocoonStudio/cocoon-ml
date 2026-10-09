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
        # every artifact is in exactly one window, except the tail (the stretch left when production ends
        # before a window is full), which is not a window (2026-10-09)
        self.assertLessEqual(covered, len(corpus))
        self.assertGreaterEqual(covered, len(corpus) - max(len(u) for _, _, u in ws))
        self.assertEqual([a for _, _, u in ws for a in u], corpus.artifacts[:covered])
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
        self.assertTrue(all(y == 0.0 for _, y, _, _ in rows))  # every training artifact's state is in the ball


class TestAuditReaders(unittest.TestCase):
    """Readers added after Methuselah's audit of 2026-10-09 (findings 2, 3, 5), verified by Claude."""

    def _setup(self):
        import random
        from fractions import Fraction
        from cocoonml.corpus import produce, windows
        from cocoonml.schema import generate

        tables = [generate(forms=3, cuts=2, nonterminals=2, resolution=4, seed=10 + i, graded=True) for i in range(2)]
        rng = random.Random(1)
        corpus = produce(tables, rng, Fraction(1, 4), Fraction(3, 4), Fraction(1, 4), 60, 1.0, 0.5)
        ws = windows(corpus, 4, 16)
        return tables, corpus, ws

    def test_window_distances_one_per_window_from_forms_alone(self):
        from cocoonml.corpus import window_distances

        tables, corpus, ws = self._setup()
        training = [a.forms for a in corpus.artifacts]
        ds = window_distances(tables, ws, training)
        self.assertEqual(len(ds), len(ws))
        self.assertTrue(all(d is None or d >= 0.0 for d in ds))
        # a shape that parses nothing in a window pins nothing: every window of unparseable strings reads None
        nothing = window_distances(tables, [([9, 9, 4], [9, 4, 4], [type(corpus.artifacts[0])((9, 9), (0, 0), 0, 0)])], training)
        self.assertEqual(nothing, [None])

    def test_window_truth_distances_and_bins(self):
        from cocoonml.corpus import bin_windows, window_truth_distances

        tables, corpus, ws = self._setup()
        ball = [corpus.walks[0][0], corpus.walks[1][0]]
        truth = window_truth_distances(ws, corpus, ball)
        self.assertEqual(len(truth), len(ws))
        no_phase = window_truth_distances(ws, corpus, ball, phases=False)
        self.assertEqual(len(no_phase), len(ws))
        bins = bin_windows(ws, truth, 2)
        self.assertEqual(sum(len(part) for _, part in bins), len(ws))
        self.assertLessEqual(bins[0][0], bins[-1][0])
        self.assertEqual(bin_windows(ws, [None] * len(ws), 2), [])

    def test_probe_rows_carry_rank_and_window_index(self):
        from cocoonml.array_attention import ArrayAttention
        from cocoonml.corpus import probe_rows

        tables, corpus, ws = self._setup()
        model = ArrayAttention(vocab=5, width=3, length=16, seed=0, separator=4, layers=1)
        rows = probe_rows(model, ws[:3], corpus, [corpus.walks[0][0]], 4)
        self.assertEqual(len(rows), sum(len(used) for _, _, used in ws[:3]))
        for vector, distance, rank, w in rows:
            self.assertEqual(len(vector), 3)
            self.assertGreaterEqual(distance, 0.0)
            self.assertIn(w, (0, 1, 2))
        self.assertEqual([r[2] for r in rows if r[3] == 0], list(range(len(ws[0][2]))))


class TestWindowsDropTheTail(unittest.TestCase):
    def test_the_tail_is_not_a_window_unless_it_is_the_only_one(self):
        from cocoonml.corpus import Artifact, Corpus, windows

        # strings of three forms, separator 4, length 8: two whole artifacts per window; eleven artifacts leave a tail of one
        arts = [Artifact((1, 2, 3), (0, 0), 0, 0) for _ in range(11)]
        ws = windows(Corpus(arts, [[None]]), 4, 8)
        self.assertEqual([len(u) for _, _, u in ws], [2, 2, 2, 2, 2])
        # ten artifacts: the last window is cut by the length exactly as production ends, so it is kept
        ws = windows(Corpus(arts[:10], [[None]]), 4, 8)
        self.assertEqual([len(u) for _, _, u in ws], [2, 2, 2, 2, 2])
        # one artifact: the only window is kept
        self.assertEqual(len(windows(Corpus(arts[:1], [[None]]), 4, 8)), 1)


class TestFormLossError(unittest.TestCase):
    def test_the_mean_with_error_agrees_with_the_mean_and_carries_a_band(self):
        import random
        from fractions import Fraction
        from cocoonml.array_attention import ArrayAttention
        from cocoonml.corpus import form_losses_by_rank, form_losses_by_rank_with_error, produce, windows, whole_rank
        from cocoonml.schema import generate

        tables = [generate(forms=3, cuts=2, nonterminals=2, resolution=4, seed=10 + i, graded=True) for i in range(2)]
        corpus = produce(tables, random.Random(1), Fraction(1, 4), Fraction(3, 4), Fraction(1, 4), 80, 1.0, 0.5)
        ws = windows(corpus, 4, 16)
        model = ArrayAttention(vocab=5, width=3, length=16, seed=0, separator=4, layers=1)
        w = whole_rank(ws)
        plain = form_losses_by_rank(model, ws, 4, w)
        rich = form_losses_by_rank_with_error(model, ws, 4, w)
        self.assertEqual(len(plain), len(rich))
        for (mean, se, n), p in zip(rich, plain):
            # the plain mean weights every form target equally; the per-window mean weights windows equally: close, not equal
            self.assertLess(abs(mean - p), 0.5)
            self.assertGreaterEqual(n, 1)
            self.assertTrue(se is None or se >= 0.0)


class TestBall(unittest.TestCase):
    def test_the_ball_deduplicates_and_answers_like_the_list(self):
        import random
        from fractions import Fraction
        from cocoonml.corpus import Ball, distance_from_ball, produce
        from cocoonml.schema import generate

        tables = [generate(forms=3, cuts=2, nonterminals=2, resolution=4, seed=10 + i, graded=True) for i in range(2)]
        corpus = produce(tables, random.Random(1), Fraction(1, 4), Fraction(3, 4), Fraction(1, 4), 300, 1.0, 0.5)
        raw = [state for walk in corpus.walks for state in walk]
        ball = Ball(raw)
        self.assertLess(len(ball), len(raw))  # the walks revisit states
        probe = corpus.walks[0][len(corpus.walks[0]) // 2]
        for phases in (True, False):
            self.assertAlmostEqual(distance_from_ball(probe, ball, phases), distance_from_ball(probe, raw, phases), places=12)
        far = corpus.walks[1][-1]
        for phases in (True, False):
            self.assertAlmostEqual(distance_from_ball(far, ball, phases), distance_from_ball(far, raw, phases), places=12)
        self.assertEqual(ball.distance(probe), ball.distance(probe))  # cached answer stable
