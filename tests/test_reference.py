import math
import random
import unittest
from fractions import Fraction

from cocoonml.corpus import produce, windows
from cocoonml.reference import bayes_losses, marginal_losses, string_distribution, switch_probability, token_counts, window_predictives
from cocoonml.schema import generate

FORMS, NTS, RES = 3, 2, 4
SEP = FORMS + 1
BOUND, LEAN = Fraction(NTS, 2 * RES), Fraction(3, 4)


def tables(n=2, seed=1):
    return [generate(forms=FORMS, cuts=2, nonterminals=NTS, resolution=RES, seed=10 * seed + i, graded=True) for i in range(n)]


def toy_windows(n_tables=2, count=300, length=16, steps_mean=0.0, radius=None, seed=1):
    corpus = produce(tables(n_tables), random.Random(seed), BOUND, LEAN, radius, count, steps_mean, 0.5)
    return windows(corpus, SEP, length)


class TestTheGeneratorsPieces(unittest.TestCase):
    def test_the_string_distribution_sums_to_one(self):
        for t in tables():
            self.assertAlmostEqual(sum(string_distribution(t).values()), 1.0)

    def test_the_switch_probability_is_an_odd_poisson_count(self):
        self.assertAlmostEqual(switch_probability(0.5, 2), (1 - math.exp(-1)) / 2)
        self.assertEqual(switch_probability(0.5, 1), 0.0)
        self.assertEqual(switch_probability(0.0, 2), 0.0)


class TestTheExactPredictor(unittest.TestCase):
    def test_the_predictive_sums_to_one_at_every_position(self):
        for tokens, _, _ in toy_windows()[:20]:
            for probs in window_predictives(tokens, SEP, tables(), 0.5):
                self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)

    def test_one_table_and_no_switching_gives_the_conditional_of_the_string_distribution(self):
        (table,) = tables(1)
        dist = string_distribution(table)
        tokens, targets, used = toy_windows(1, count=50, length=64)[0]
        first = used[0].forms
        probs = window_predictives(tokens, SEP, [table], 0.0)
        # inside the first artifact, after its first form: P(next | prefix) over the strings that fit the whole window
        room = 64
        prefix = first[:1]
        fitting = {f: p for f, p in dist.items() if len(f) + 1 <= room and f[:1] == prefix}
        total = sum(fitting.values())
        expected = {}
        for f, p in fitting.items():
            nxt = f[1] if len(f) > 1 else SEP
            expected[nxt] = expected.get(nxt, 0.0) + p / total
        for token, p in expected.items():
            self.assertAlmostEqual(probs[0].get(token, 0.0), p, places=9)

    def test_the_bayes_floor_lies_below_the_marginal_and_the_uniform(self):
        ws = toy_windows(count=400)
        counts = token_counts(ws)
        bayes = [sum(bayes_losses(t, g, SEP, tables(), 0.5)) / len(t) for t, g, _ in ws[:60]]
        marginal = [sum(marginal_losses(g, counts)) / len(t) for t, g, _ in ws[:60]]
        self.assertLess(sum(bayes) / len(bayes), sum(marginal) / len(marginal))
        self.assertLess(sum(marginal) / len(marginal), math.log(FORMS + 1) + 1e-9)

    def test_the_marginal_with_uniform_counts_is_log_of_the_vocabulary(self):
        self.assertAlmostEqual(marginal_losses([1, 2, SEP], {1: 5, 2: 5, 3: 5, SEP: 5})[0], math.log(4))


class TestTheParticleFilter(unittest.TestCase):
    def test_with_steps_it_needs_particles_and_a_generator(self):
        tokens, _, _ = toy_windows()[0]
        with self.assertRaises(ValueError):
            window_predictives(tokens, SEP, tables(), 0.5, steps_mean=1.0, bound=BOUND, lean=LEAN, radius=1)

    def test_with_vanishing_steps_it_agrees_with_the_exact_predictor(self):
        ws = toy_windows()[:6]
        exact = [sum(bayes_losses(t, g, SEP, tables(), 0.5)) / len(t) for t, g, _ in ws]
        rng = random.Random(3)
        sampled = [sum(bayes_losses(t, g, SEP, tables(), 0.5, steps_mean=1e-12, bound=BOUND, lean=LEAN, radius=1, particles=400, rng=rng)) / len(t) for t, g, _ in ws]
        self.assertLess(abs(sum(exact) - sum(sampled)) / len(ws), 0.05)

    def test_with_drift_it_runs_finite_and_deterministic_under_the_seed(self):
        tokens, targets, _ = toy_windows(steps_mean=1.0, radius=1)[0]
        a = bayes_losses(tokens, targets, SEP, tables(), 0.5, steps_mean=1.0, bound=BOUND, lean=LEAN, radius=1, particles=50, rng=random.Random(5))
        b = bayes_losses(tokens, targets, SEP, tables(), 0.5, steps_mean=1.0, bound=BOUND, lean=LEAN, radius=1, particles=50, rng=random.Random(5))
        self.assertEqual(a, b)
        self.assertTrue(all(math.isfinite(x) and x >= 0 for x in a))
        self.assertEqual(len(a), len(tokens))


if __name__ == "__main__":
    unittest.main()


class TestTheMarkovWorld(unittest.TestCase):
    def chains(self, n=2):
        from cocoonml.markov import generate

        return [generate(states=3, resolution=8, seed=20 + i) for i in range(n)]

    def test_one_chain_and_no_steps_gives_its_row_inside_an_artifact(self):
        from cocoonml.markov import draw
        from cocoonml.reference import MarkovWorld

        (chain,) = self.chains(1)
        rng = random.Random(1)
        forms = draw(chain, rng)[0]
        while len(forms) < 2:
            forms = draw(chain, rng)[0]
        tokens = list(forms) + [4] + [4] * (64 - len(forms) - 1)
        probs = window_predictives(tokens, 4, [chain], 0.0, world=MarkovWorld())
        row = chain.rows[forms[0] - 1]
        for state in range(3):
            self.assertAlmostEqual(probs[0].get(state + 1, 0.0), float(row[state]), places=2)  # the fit in the room moves it a hair
        self.assertAlmostEqual(probs[0][4], float(row[-1]), places=2)
        for p in probs:
            self.assertAlmostEqual(sum(p.values()), 1.0, places=9)

    def test_the_floor_lies_below_the_marginal_on_chains_too(self):
        from cocoonml.markov import draw, step
        from cocoonml.reference import MarkovWorld

        chains = self.chains()
        corpus = produce(chains, random.Random(2), Fraction(3, 16), None, None, 300, 0.0, 0.5, draw=draw, step=step)
        ws = windows(corpus, 4, 32)
        counts = token_counts(ws)
        world = MarkovWorld()
        bayes = [sum(bayes_losses(t, g, 4, chains, 0.5, world=world)) / len(t) for t, g, _ in ws[:40]]
        marginal = [sum(marginal_losses(g, counts)) / len(t) for t, g, _ in ws[:40]]
        self.assertLess(sum(bayes) / len(bayes), sum(marginal) / len(marginal))

    def test_the_particle_filter_steps_chains_with_their_own_step(self):
        from cocoonml.markov import draw, step
        from cocoonml.reference import MarkovWorld

        chains = self.chains()
        corpus = produce(chains, random.Random(3), Fraction(3, 16), None, Fraction(1, 8), 60, 1.0, 0.5, draw=draw, step=step)
        tokens, targets, _ = windows(corpus, 4, 32)[0]
        losses = bayes_losses(tokens, targets, 4, chains, 0.5, steps_mean=1.0, bound=Fraction(3, 16), lean=None, radius=Fraction(1, 8), particles=40, rng=random.Random(4), world=MarkovWorld())
        self.assertEqual(len(losses), len(tokens))
        self.assertTrue(all(math.isfinite(x) and x >= 0 for x in losses))
