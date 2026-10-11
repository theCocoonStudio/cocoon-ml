import unittest

from cocoonml.run_row import apply_row, ladder, null_readings, onset_check, onset_prediction, replication, report, second_descent_prediction, window_of
from tests.test_rules import FALLING, FLAT, HELD, RISING, cell, reading


def points_for(far, whole=10):
    points = {key: reading((HELD[0], far[1]), far, whole) for key in ("0", "1", "2", "3")}
    points["free"] = reading((HELD[0], []), None, whole)
    return points


def copy(step, far, whole=10):
    return {"step": step, "held_loss": 1.2, "held": reading(HELD, whole=whole), "points": points_for(far, whole), "same": reading(HELD, whole=whole), "probe_r2_truth_distance": None, "probe_r2_rank": None}


def run_cell(width, layers, radius, lr, evaluations, escape=None, onset=None, best_far=RISING, train_steps=None, whole=9):
    """A cap-mode cell as runs/sweep7.py writes it, the curves synthetic: the marginal copy's far bin FLAT, the best copy's `best_far`."""
    c = cell(train_radius=radius, width=width, layers=layers, evaluations=evaluations, whole=whole, far=best_far if onset is not None else FLAT)
    c["run"] = f"sweep-07e-r{radius}-w{width}-lr{lr}"
    c["args"] = ["x", "4", "3", "4", str(width), str(layers), "64", "2", str(radius), "1", "0.5", "42000", "0,1,2,3,free", "2000", "9999", "1", str(lr)]
    c["lr"] = lr
    c["mode"] = "cap"
    c["train_steps"] = train_steps if train_steps is not None else (escape if escape is not None else 50 * len(evaluations))
    c["escape_step"] = escape
    c["onset_step"] = onset
    best_step = None if onset is None else onset + 50
    c["copies"] = {"marginal": copy(50, FLAT, whole), "best": None if onset is None else copy(best_step, best_far, whole), "last": {"step": 50 * len(evaluations), "held_loss": 1.0}}
    c["read_copy"] = "best" if onset is not None else "marginal"
    return c


FLAT_EVALS = [1.30, 1.29, 1.295, 1.29, 1.285, 1.29, 1.29, 1.29]  # no onset under the envelope
LEARNED_EVALS = [1.30, 1.29, 1.20, 1.10, 1.05, 1.0, 0.98, 0.97]  # onset at 150


def flat_07c(radius, width, layers):
    c = cell(train_radius=radius, width=width, layers=layers, far=FLAT)
    c["run"] = f"sweep-07c-r{radius}-w{width}"
    c["args"] = ["x", "4", "3", "4", str(width), str(layers), "64", "2", str(radius), "1", "0.5", "42000", "0,1,2,3,free", "2000", "6000", "1", "0.05"]
    return c


class TestTheLadder(unittest.TestCase):
    def test_concordant_ladder_names_the_next_rate(self):
        cells = [run_cell(96, 7, 1, 0.2, FLAT_EVALS[:2], escape=100), run_cell(96, 7, 1, 0.1, FLAT_EVALS[:8], escape=400, onset=150), run_cell(96, 7, 1, 0.05, LEARNED_EVALS, onset=150), run_cell(96, 7, 1, 0.025, LEARNED_EVALS, onset=250)]
        lad = ladder(cells)
        self.assertEqual((lad["concordant"], lad["discordant"]), (5, 0))
        self.assertEqual(lad["next_rate"], 0.05)
        self.assertTrue(lad["escaped_before_onset"][0.2])
        self.assertFalse(lad["escaped_before_onset"][0.1])  # escaped after its onset: the best copy counts (decision (b))
        self.assertIn("the ladder reads", lad["decision"])

    def test_a_discordant_pair_fires_d(self):
        cells = [run_cell(96, 7, 1, 0.2, FLAT_EVALS[:8], escape=400), run_cell(96, 7, 1, 0.1, FLAT_EVALS[:2], escape=100), run_cell(96, 7, 1, 0.05, LEARNED_EVALS, onset=150)]
        self.assertIn("(d)", ladder(cells)["decision"])

    def test_no_onset_anywhere_is_a_and_names_the_corpus_at_the_pass_bound_size(self):
        small = [run_cell(96, 7, 1, lr, FLAT_EVALS) for lr in (0.1, 0.05)]
        self.assertIn("unread at this budget", ladder(small)["decision"])
        large = [run_cell(192, 9, 1, lr, FLAT_EVALS) for lr in (0.1, 0.05)]
        self.assertIn("unread at this corpus", ladder(large)["decision"])

    def test_every_rung_escaping_before_its_onset_extends_the_ladder_down(self):
        cells = [run_cell(96, 7, 1, 0.2, FLAT_EVALS[:1], escape=30), run_cell(96, 7, 1, 0.1, FLAT_EVALS[:2], escape=80)]
        self.assertIn("extends down one halving", ladder(cells)["decision"])


class TestReplication(unittest.TestCase):
    def three_sizes(self, far_at_largest, radius=1, lr=0.05):
        return [run_cell(96, 7, radius, lr, LEARNED_EVALS, onset=150, best_far=RISING), run_cell(144, 8, radius, lr, LEARNED_EVALS, onset=150, best_far=RISING), run_cell(192, 9, radius, lr, LEARNED_EVALS, onset=150, best_far=far_at_largest)]

    def test_the_null_pools_the_flat_cells_and_every_marginal_copy_at_the_radius(self):
        flat = [flat_07c(1, 144, 8), flat_07c(1, 192, 9), flat_07c(2, 144, 8)]
        null = null_readings(flat, self.three_sizes(RISING), 1)
        self.assertEqual(null["slope_2"], [0.0] * 5)  # two flat cells at radius 1 and three marginal copies, all FLAT

    def test_signs_agreeing_at_every_counting_key_let_the_readings_stand(self):
        rep = replication(self.three_sizes(RISING), [flat_07c(1, 144, 8), flat_07c(1, 192, 9)], 1)
        self.assertEqual(rep["matched_verdicts"][0.05], "agree")
        self.assertIsNone(rep["matched_verdicts"][0.2])  # not comparable: no cell at that rung
        self.assertEqual(rep["best_rung"], {(96, 7): 0.05, (144, 8): 0.05, (192, 9): 0.05})
        self.assertIn("the readings stand", rep["decision"])

    def test_a_sign_differing_across_sizes_is_r4_s_move(self):
        rep = replication(self.three_sizes(FALLING), [flat_07c(1, 144, 8), flat_07c(1, 192, 9)], 1)
        self.assertTrue(rep["matched"][0.05]["slope_2"])  # +0.3 at two sizes against -0.3 at the third, the null at 0
        self.assertIn("R4's move", rep["decision"])

    def test_without_three_comparable_cells_the_comparison_waits(self):
        two = self.three_sizes(RISING)[:2] + [run_cell(192, 9, 1, 0.05, FLAT_EVALS)]  # the largest never left the marginal
        rep = replication(two, [], 1)
        self.assertIsNone(rep["matched"][0.05])
        self.assertIsNone(rep["at_best_rung"])
        self.assertIn("not comparable at this budget", rep["decision"])


class TestThePredictions(unittest.TestCase):
    def test_the_window_convention(self):
        self.assertEqual(window_of(699), 14)
        self.assertEqual(window_of(850), 17)
        self.assertEqual(window_of(1244), 25)

    def test_prediction_three_is_met_by_an_onset_within_three_windows_of_the_crossing(self):
        met = run_cell(144, 8, 1, 0.05, LEARNED_EVALS, onset=850)  # window 17 against the crossing 699 in window 14
        late = run_cell(192, 9, 1, 0.05, LEARNED_EVALS, onset=1500)  # window 30 against 1 243 in window 25
        pred = onset_prediction([met])
        self.assertTrue(pred["fires"])
        self.assertEqual(pred["cells"][met["run"]]["crossing"], 699)
        pred = onset_prediction([met, late])
        self.assertFalse(pred["fires"])
        self.assertTrue(pred["fails_at_largest"])

    def test_prediction_four_reads_the_radius_two_cells_that_reach_four_crossings(self):
        plateau = [1.09, 1.095, 1.088, 1.092, 1.09, 1.091]
        evals = [1.30, 1.29, 1.20, 1.10] + [1.1] * 18 + plateau + [1.007, 1.0] + [1.0] * 10  # the descent at evaluation 29, step 1450
        fires = run_cell(96, 7, 2, 0.05, evals, onset=150)
        pred = second_descent_prediction([fires])
        self.assertEqual(pred["cells"][fires["run"]]["descent_step"], 1450)
        self.assertTrue(pred["cells"][fires["run"]]["within"])
        self.assertTrue(pred["fires"])
        short = run_cell(96, 7, 2, 0.1, LEARNED_EVALS, onset=150, escape=400)
        self.assertTrue(second_descent_prediction([short])["unread"])  # four crossings not reached
        radius_one = run_cell(96, 7, 1, 0.05, evals, onset=150)
        self.assertTrue(second_descent_prediction([radius_one])["unread"])  # reported, not lodged

    def test_the_onset_check_compares_the_harness_to_the_recomputation(self):
        self.assertTrue(onset_check(run_cell(96, 7, 1, 0.05, LEARNED_EVALS, onset=150))["agree"])
        self.assertFalse(onset_check(run_cell(96, 7, 1, 0.05, LEARNED_EVALS, onset=200))["agree"])


class TestTheWholeRow(unittest.TestCase):
    def test_apply_row_and_report_cover_every_test(self):
        cells = [run_cell(w, l, r, lr, LEARNED_EVALS if lr <= 0.05 else FLAT_EVALS[:2], onset=150 if lr <= 0.05 else None, escape=None if lr <= 0.05 else 100) for (w, l) in ((96, 7), (144, 8), (192, 9)) for r in (1, 2) for lr in (0.2, 0.1, 0.05, 0.025)]
        result = apply_row(cells, [flat_07c(1, 144, 8), flat_07c(1, 192, 9), flat_07c(2, 144, 8), flat_07c(2, 192, 9)])
        self.assertEqual(result["cells"], 24)
        self.assertEqual(len(result["ladders"]), 6)
        text = report(result)
        for needle in ("test (1)", "test (2)", "prediction (3)", "prediction (4)", "agrees with the recomputed one"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
