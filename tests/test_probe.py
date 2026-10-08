import random
import unittest

from cocoonml.probe import fit, predict, r_squared


class TestProbe(unittest.TestCase):
    def test_recovers_a_linear_function_of_the_activations(self):
        rng = random.Random(0)
        xs = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(40)]
        ys = [2.0 * x[0] - 0.5 * x[2] + 0.25 for x in xs]
        w, b = fit(xs, ys)
        self.assertAlmostEqual(w[0], 2.0, places=2)
        self.assertAlmostEqual(w[2], -0.5, places=2)
        self.assertAlmostEqual(b, 0.25, places=2)
        self.assertGreater(r_squared(w, b, xs, ys), 0.999)

    def test_r_squared_is_near_zero_for_noise(self):
        rng = random.Random(1)
        xs = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(200)]
        ys = [rng.uniform(-1, 1) for _ in xs]
        w, b = fit(xs, ys)
        self.assertLess(r_squared(w, b, xs, ys), 0.1)
        self.assertIsInstance(predict(w, b, xs[0]), float)


if __name__ == "__main__":
    unittest.main()
