import math
import unittest

import numpy as np

from cocoonml.harness import Copy, train_to_cap
from cocoonml.rules import onset


class Stub:
    """A model whose batch losses follow a sequence and whose one parameter block records the step."""

    def __init__(self, batch_loss=lambda n: 1.0):
        self.batch_loss, self.calls = batch_loss, 0
        self.weights = np.zeros((2, 2))

    def blocks(self):
        return [self.weights]

    def train_step(self, batch, lr):
        self.calls += 1
        self.weights[0, 0] = self.calls
        return self.batch_loss(self.calls)


def held_from(sequence):
    readings = iter(sequence)
    return lambda: next(readings)


ENVELOPE = 0.0378


class TestTrainToCap(unittest.TestCase):
    def test_runs_to_the_cap_with_no_rule_and_keeps_every_loss_and_evaluation(self):
        model = Stub()
        out = train_to_cap(model, lambda: None, 0.1, cap=100, evaluate=held_from([1.3, 1.29, 1.3, 1.29, 1.3, 1.29, 1.3, 1.29, 1.3, 1.29]), onset=lambda ev: onset(ev, ENVELOPE, window=10), window=10)
        self.assertEqual(len(out["losses"]), 100)  # a flat held loss stops nothing
        self.assertEqual(len(out["evaluations"]), 10)
        self.assertIsNone(out["escape_step"])
        self.assertIsNone(out["onset_step"])
        self.assertIsNone(out["copies"]["best"])  # no onset, no best copy: the cell gives the null its marginal copy only
        self.assertEqual(out["copies"]["marginal"].step, 10)
        self.assertEqual(out["copies"]["last"].step, 100)

    def test_the_escape_is_the_first_non_finite_loss_and_the_last_copy_is_the_last_finite_window(self):
        model = Stub(lambda n: 1.0 if n < 35 else float("nan"))
        out = train_to_cap(model, lambda: None, 0.1, cap=100, evaluate=held_from([1.3, 1.29, 1.3]), onset=lambda ev: None, window=10)
        self.assertEqual(out["escape_step"], 35)
        self.assertEqual(len(out["losses"]), 35)
        self.assertEqual(out["copies"]["last"].step, 30)

    def test_a_non_finite_held_evaluation_records_the_escape(self):
        model = Stub()
        out = train_to_cap(model, lambda: None, 0.1, cap=100, evaluate=held_from([1.3, float("nan")]), onset=lambda ev: None, window=10)
        self.assertEqual(out["escape_step"], 20)
        self.assertEqual(len(out["evaluations"]), 2)
        self.assertEqual(out["copies"]["last"].step, 10)

    def test_the_best_copy_is_the_lowest_held_at_or_after_the_onset_and_may_be_the_onset_s_own_window(self):
        held = [1.30, 1.29, 1.20, 1.10, 1.15, 1.05, 1.08, 1.05]  # the onset at the third window (1.20 and 1.10 both past the envelope)
        model = Stub()
        out = train_to_cap(model, lambda: None, 0.1, cap=80, evaluate=held_from(held), onset=lambda ev: onset(ev, ENVELOPE, window=10), window=10)
        self.assertEqual(out["onset_step"], 30)
        self.assertEqual(out["copies"]["marginal"].step, 10)
        self.assertEqual(out["copies"]["best"].step, 60)  # 1.05 first at window 6; the tie at window 8 does not move it
        self.assertEqual(out["copies"]["best"].held, 1.05)
        self.assertEqual(out["copies"]["last"].step, 80)

    def test_the_onset_window_itself_can_be_the_best(self):
        held = [1.30, 1.29, 1.00, 1.10, 1.10]  # the onset's own window holds the lowest reading
        out = train_to_cap(Stub(), lambda: None, 0.1, cap=50, evaluate=held_from(held), onset=lambda ev: onset(ev, ENVELOPE, window=10), window=10)
        self.assertEqual(out["onset_step"], 30)
        self.assertEqual(out["copies"]["best"].step, 30)

    def test_a_predicate_counting_steps_in_other_windows_is_refused(self):
        held = [1.30, 1.29, 1.00, 1.10, 1.10]
        with self.assertRaises(ValueError):
            train_to_cap(Stub(), lambda: None, 0.1, cap=50, evaluate=held_from(held), onset=lambda ev: onset(ev, ENVELOPE), window=10)  # the predicate's default window is 50

    def test_copies_are_snapshots_the_model_s_later_steps_do_not_touch_and_load_back(self):
        model = Stub()
        out = train_to_cap(model, lambda: None, 0.1, cap=30, evaluate=held_from([1.3, 1.3, 1.3]), onset=lambda ev: None, window=10)
        marginal = out["copies"]["marginal"]
        self.assertEqual(marginal.blocks[0][0, 0], 10.0)  # the weights as they were at step 10
        self.assertEqual(model.weights[0, 0], 30.0)
        marginal.load_into(model)
        self.assertEqual(model.weights[0, 0], 10.0)
        self.assertIsInstance(marginal, Copy)


if __name__ == "__main__":
    unittest.main()
