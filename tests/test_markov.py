import random
import unittest
from fractions import Fraction

from cocoonml.corpus import Ball, distance_from_ball, produce, windows
from cocoonml.markov import Chain, MarkovBall, _quantise_row, distance, draw, estimated_rows, generate, signature, step, window_distances


class TestTheGenerator(unittest.TestCase):
    def test_quantised_rows_sum_to_one_on_the_grid(self):
        row = _quantise_row([0.3, 0.3, 0.4], 8)
        self.assertEqual(sum(row), 1)
        self.assertTrue(all(x.denominator in (1, 2, 4, 8) for x in row))
        self.assertEqual(_quantise_row([0, 0, 1], 4), (0, 0, 1))

    def test_a_generated_chain_is_on_the_grid_with_an_end_in_every_row(self):
        for seed in range(5):
            c = generate(states=4, resolution=8, seed=seed)
            self.assertEqual(sum(c.start), 1)
            for row in c.rows:
                self.assertEqual(len(row), 5)
                self.assertEqual(sum(row), 1)
                self.assertGreaterEqual(row[-1], Fraction(1, 8))

    def test_the_same_seed_gives_the_same_chain(self):
        self.assertEqual(signature(generate(3, 4, 7)), signature(generate(3, 4, 7)))
        self.assertNotEqual(signature(generate(3, 4, 7)), signature(generate(3, 4, 8)))


class TestDrawAndStep(unittest.TestCase):
    def test_an_artifact_is_a_run_of_the_chain_in_the_forms_one_to_k(self):
        c = generate(4, 8, 3)
        rng = random.Random(1)
        for _ in range(200):
            forms, extension = draw(c, rng)
            self.assertTrue(forms)
            self.assertTrue(all(1 <= f <= 4 for f in forms))
            self.assertEqual(extension, ())
            for a, b in zip(forms, forms[1:]):
                self.assertGreater(c.rows[a - 1][b - 1], 0)  # every transition taken has mass

    def test_a_step_moves_one_grain_per_row_at_most_and_keeps_the_end(self):
        c = generate(4, 8, 3)
        rng = random.Random(2)
        for _ in range(100):
            s = step(c, rng, Fraction(3, 16))
            for a, b in zip((c.start, *c.rows), (s.start, *s.rows)):
                self.assertEqual(sum(b), 1)
                self.assertLessEqual(sum(abs(x - y) for x, y in zip(a, b)) / 2, Fraction(1, 8))
            for row in s.rows:
                self.assertGreaterEqual(row[-1], Fraction(1, 8))
            c = s

    def test_the_pull_keeps_a_walk_near_its_centre_where_a_free_walk_drifts(self):
        centre = generate(4, 8, 3)
        pulled, free = centre, centre
        rng = random.Random(4)
        far_pulled = far_free = 0.0
        for _ in range(400):
            pulled = step(pulled, rng, Fraction(3, 16), centre=centre, radius=Fraction(1, 8))
            free = step(free, rng, Fraction(3, 16))
            far_pulled = max(far_pulled, distance(pulled, centre))
            far_free = max(far_free, distance(free, centre))
        self.assertLess(far_pulled, far_free)
        self.assertLess(far_pulled, 0.3)

    def test_the_distance_is_the_mean_total_variation_and_the_ball_takes_the_least(self):
        a = Chain([Fraction(1)], [[Fraction(1, 2), Fraction(1, 2)]], 2)
        b = Chain([Fraction(1)], [[Fraction(0), Fraction(1)]], 2)
        self.assertAlmostEqual(distance(a, b), (0 + 0.5) / 2)
        self.assertEqual(distance(a, a), 0.0)
        ball = MarkovBall([a, b, a])
        self.assertEqual(len(ball), 2)
        self.assertAlmostEqual(ball.distance(b), 0.0)
        self.assertAlmostEqual(distance_from_ball(Chain([Fraction(1)], [[Fraction(1), Fraction(0)]], 2), ball), 0.25)  # nearest is a, at TV 1/2 on the one row, averaged with the start


class TestTheReaderSideAndTheProduction(unittest.TestCase):
    def test_estimated_rows_recover_the_chain_from_many_strings(self):
        c = generate(3, 8, 5)
        rng = random.Random(6)
        strings = [draw(c, rng)[0] for _ in range(6000)]
        start, rows = estimated_rows(3, strings)
        for est, true in zip(start, c.start):
            self.assertAlmostEqual(est, float(true), delta=0.03)
        for est_row, true_row in zip(rows, c.rows):
            if est_row is not None:
                for est, true in zip(est_row, true_row):
                    self.assertAlmostEqual(est, float(true), delta=0.05)

    def test_produce_runs_in_this_world_and_window_distances_read_the_rows(self):
        chains = [generate(3, 8, 10), generate(3, 8, 11)]
        corpus = produce(chains, random.Random(1), Fraction(3, 16), None, Fraction(1, 8), 300, 1.0, 0.5, draw=draw, step=step)
        self.assertEqual(len(corpus), 300)
        self.assertTrue(all(isinstance(s, Chain) for walk in corpus.walks for s in walk))
        ws = windows(corpus, 4, 32)
        self.assertTrue(ws)
        dists = window_distances(chains, ws, corpus.forms())
        self.assertEqual(len(dists), len(ws))
        self.assertTrue(all(d is None or 0 <= d <= 1 for d in dists))
        ball = MarkovBall(s for walk in corpus.walks for s in walk)
        self.assertGreater(len(ball), 2)
        self.assertEqual(distance_from_ball(corpus.walks[0][0], ball), 0.0)

    def test_the_schema_world_is_the_default_and_unchanged(self):
        from cocoonml.schema import generate as make_table

        tables = [make_table(forms=3, cuts=2, nonterminals=2, resolution=4, seed=10 + i, graded=True) for i in range(2)]
        corpus = produce(tables, random.Random(1), Fraction(2, 8), Fraction(3, 4), None, 20, 0.0, 0.5)
        self.assertEqual(len(corpus), 20)
        self.assertIsInstance(Ball(s for walk in corpus.walks for s in walk), Ball)


if __name__ == "__main__":
    unittest.main()
