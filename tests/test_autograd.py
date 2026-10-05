import unittest

from cocoonml.autograd import Autograd


class TestAutograd(unittest.TestCase):
    def test_linear_error_convergence(self):
        """
        Validates that the forward finite difference error shrinks linearly with h (O(h)).
        Proves that error(h) / error(h/2) converges to exactly 2.0 before float precision breaks.
        """
        x_val = 0.5

        # 1. Analytical Gradient
        x = Autograd(x_val)
        out = x.tanh() * x
        out.backward()
        analytical_grad = x._derivative

        # 2. Numerical Gradient Helper
        def f(val):
            temp_x = Autograd(val)
            return (temp_x.tanh() * temp_x).value

        # 3. Compute errors at h and h/2
        h = 1e-4
        num_grad_h = (f(x_val + h) - f(x_val)) / h
        error_h = abs(num_grad_h - analytical_grad)

        h_half = h / 2.0
        num_grad_h_half = (f(x_val + h_half) - f(x_val)) / h_half
        error_h_half = abs(num_grad_h_half - analytical_grad)

        # 4. Prove Linear Convergence (O(h))
        # Since error scales linearly with h, halving h should exactly halve the error.
        error_ratio = error_h / error_h_half

        # We expect the ratio to be very close to 2.0
        self.assertAlmostEqual(
            error_ratio,
            2.0,
            delta=0.01,
            msg=f"Error did not converge linearly. Ratio was {error_ratio}",
        )

    def test_multiple_consumer_accumulation(self):
        """
        Validates that nodes with multiple consumers accumulate gradients (+=).
        If accumulation is broken, df/da will equal 5.0 instead of 10.0.
        """
        a_val, b_val = 2.0, 3.0

        # Analytical Gradient
        a = Autograd(a_val)
        b = Autograd(b_val)

        # f(a, b) = (a + b) * (a + b)
        sum_node = a.add(b)
        out = sum_node * sum_node

        out.backward()

        # Mathematical Proof: f(a, b) = (a + b)^2
        # df/da = 2 * (a + b) * 1 = 2 * 5 = 10.0
        self.assertEqual(a._derivative, 10.0, "Gradient accumulation failed on 'a'")
        self.assertEqual(b._derivative, 10.0, "Gradient accumulation failed on 'b'")

        # Cross-verify with finite differences
        def f_a(val):
            ta = Autograd(val)
            tb = Autograd(b_val)
            return (ta.add(tb) * ta.add(tb)).value

        h = 1e-6
        num_grad_a = (f_a(a_val + h) - f_a(a_val)) / h
        self.assertAlmostEqual(a._derivative, num_grad_a, places=4)


if __name__ == "__main__":
    unittest.main()
