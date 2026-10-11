import unittest

from cocoonml.rules import ENVELOPE, cap_for, concordance, crossing, departures, derived_floor, differ_pairwise, envelope, onset, rank_sign, second_descent


class TestOnsetByTheRunningMedian(unittest.TestCase):
    """Decisions 53, fourth form: the onset's reference is the running median of the preceding evaluations."""

    def test_departures_are_measured_from_the_running_median_and_the_first_has_none(self):
        self.assertEqual(departures([1.3, 1.2, 1.4, 1.0]), [None, 1.3 - 1.2, 1.25 - 1.4, 1.3 - 1.0])

    def test_a_flat_cell_whose_first_evaluation_sits_above_its_plateau_has_no_onset(self):
        flat = [1.31, 1.28, 1.285, 1.275, 1.28, 1.29, 1.28]  # the first evaluation's offset is not a departure of the plateau
        self.assertIsNone(onset(flat, envelope=0.02))  # the first departure (from the first evaluation alone) is the offset; it does not persist
        self.assertTrue(all(d < 0.02 for d in departures(flat)[2:]))

    def test_the_onset_is_the_first_of_two_consecutive_windows_past_the_envelope(self):
        learned = [1.30, 1.28, 1.29, 1.20, 1.10, 1.05]
        self.assertEqual(onset(learned, envelope=0.0378), 200)  # evaluation 4 (step 200) and 5 both exceed
        self.assertIsNone(onset([1.30, 1.28, 1.20, 1.29, 1.28], envelope=0.0378))  # one window alone does not persist

    def test_the_envelope_is_the_largest_departure_over_the_flat_cells(self):
        self.assertAlmostEqual(envelope([[1.3, 1.29, 1.27], [1.3, 1.28, 1.285, 1.26]]), 0.025)  # 1.295 - 1.27 and 1.285 - 1.26, both from the running median
        self.assertIsNone(envelope([[1.3]]))
        self.assertEqual(ENVELOPE, 0.0378)


class TestTheSecondDescent(unittest.TestCase):
    def test_a_descent_after_a_held_plateau_fires_at_its_first_window(self):
        plateau = [1.09, 1.095, 1.088, 1.092, 1.09, 1.091]
        trajectory = [1.3, 1.25, 1.15] + plateau + [1.007, 1.0]
        self.assertEqual(second_descent(trajectory, envelope=0.0378), 50 * 10)

    def test_the_first_descent_does_not_count_as_a_second_one(self):
        falling = [1.3, 1.29, 1.28, 1.2, 1.1, 1.0, 0.9, 0.85, 0.8]  # never six windows held within the envelope
        self.assertIsNone(second_descent(falling, envelope=0.0378))

    def test_the_marginal_plateau_before_the_onset_is_not_a_plateau(self):
        marginal_then_learning = [1.30, 1.29, 1.295, 1.29, 1.285, 1.29, 1.29, 1.2, 1.1, 1.0]  # cell b's shape: flat, then the onset at 400
        self.assertEqual(onset(marginal_then_learning, envelope=0.0378), 400)
        self.assertEqual(second_descent(marginal_then_learning, envelope=0.0378), 400)  # without the onset, the first descent reads as one
        self.assertIsNone(second_descent(marginal_then_learning, envelope=0.0378, after=400))

    def test_a_single_low_window_is_not_a_descent(self):
        self.assertIsNone(second_descent([1.09] * 6 + [1.0, 1.09, 1.09], envelope=0.0378))


class TestTheLadderAndTheSigns(unittest.TestCase):
    def test_concordance_counts_pairs_with_an_escape_and_excludes_ties(self):
        self.assertEqual(concordance({0.2: 100, 0.1: 400, 0.05: None, 0.025: None}), (5, 0))  # the two cap ties are one excluded pair
        self.assertEqual(concordance({0.2: 400, 0.1: 100, 0.05: None, 0.025: None}), (4, 1))
        self.assertEqual(concordance({0.2: 100, 0.1: 100}), (0, 0))
        self.assertEqual(concordance({0.2: None, 0.1: 300}), (0, 1))  # the lower rate escaped and the higher did not

    def test_the_rank_sign_is_the_side_of_the_null(self):
        null = [0.10, 0.12, 0.11]
        self.assertEqual(rank_sign(0.13, null), 1)
        self.assertEqual(rank_sign(0.09, null), -1)
        self.assertEqual(rank_sign(0.11, null), 0)
        self.assertEqual(rank_sign(0.12, null), 0)  # the range's end is inside
        self.assertEqual(rank_sign(5.0, []), 0)

    def test_differ_is_pairwise_with_zeros_abstaining(self):
        self.assertIsNone(differ_pairwise([1, 0, 0]))
        self.assertIs(differ_pairwise([1, 0, 1]), False)
        self.assertIs(differ_pairwise([1, 0, -1]), True)
        self.assertIs(differ_pairwise([1, 1, -1]), True)


class TestTheCrossingAndTheCap(unittest.TestCase):
    def test_the_crossings_at_whole_nine_are_the_record_s(self):
        self.assertEqual([crossing(w, 9, 30.8) for w in (96, 144, 192)], [311, 699, 1243])

    def test_at_whole_eight_the_crossing_is_the_derived_floor_s_cap_row(self):
        floor = derived_floor(distinct_strings=94, suffix_depth=6, mean_string_length=4.4, mean_artifacts_per_window=11.4, effective_count=30.8, cap=6000)
        self.assertAlmostEqual(floor["cap_floor"] / crossing(96, 8, 30.8), 4, delta=0.02)  # the table's cap row is four crossings ("past the crossing"), to rounding

    def test_the_cap_is_the_smaller_of_eight_crossings_and_ten_passes(self):
        self.assertEqual(cap_for(96, 9, 30.8, 3654), 2488)
        self.assertEqual(cap_for(144, 9, 30.8, 3654), 5592)
        self.assertEqual(cap_for(192, 9, 30.8, 3654), 9135)  # the pass bound at radius 1
        self.assertEqual(cap_for(192, 9, 30.8, 3651), 9127)  # at radius 2, the cap the row takes at both


if __name__ == "__main__":
    unittest.main()
