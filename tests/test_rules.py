import unittest

from cocoonml.rules import (
    Ensemble, Vector, apply_rows, cell_at, cell_command, cell_name, corpus_floor_for, derived_floor, host_exceeds,
    log_entry, next_size, parse_cell, past_radius_keys, r1_instability, r2_budget, r3_k_range, r4_disagree, readings,
    round_up_to, sizes,
)

ENSEMBLE = Ensemble(
    forms=4, nonterminals=3, resolution=4, tables=2, seed=1, train_radii=(1, 2), read_radii=("0", "1", "2", "3", "free"),
    steps_mean=1.0, switch_mean=0.5, patience=3, minimum=300, lean="3/4",
)
VECTOR = Vector(width=96, layers=7, length=64, corpus=42000, read_size=2000, cap=6000, lr=0.025)
HELD = ([1.0, 0.9, 0.8, 0.7, 0.6], [0.1, 0.2])
RISING = ([1.0, 1.1, 1.2, 1.3, 1.4], [0.0, -0.05])  # slope +0.3, level +0.8 over HELD, control -0.05
FALLING = ([1.4, 1.3, 1.2, 1.1, 1.0], [0.0, -0.05])  # slope -0.3, level +0.4, control -0.05
FLAT = ([1.0, 1.0, 1.0], [0.0, 0.3])  # slope 0, level +0.2 at rank 2, control +0.3


def curve(form, control, whole=None):
    return {"windows": 10, "whole": len(form) if whole is None else whole, "form_by_rank": list(form), "order_control_by_rank": list(control)}


def reading(near, far=None, whole=None):
    """A READING as runs/sweep7.py writes it: the pooled curves, then up to three bins by distance; the far bin is the last."""
    out = dict(curve(*near, whole=whole), estimated_distance=None, estimated_known=0, truth_distance=None, truth_distance_no_phase=None, by_estimated_distance=[])
    out["by_truth_distance"] = [] if far is None else [dict(distance=0.0, **curve(*near, whole=whole)), dict(distance=1.0, **curve(*far, whole=whole))]
    return out


def cell(train_radius=1, width=96, layers=7, evaluations=(2.0, 1.5, 1.2), hit_cap=False, whole=10, far=RISING, points=None, seconds=712.3):
    """A cell's JSON line with every key runs/sweep7.py writes, the curves synthetic."""
    if points is None:
        points = {key: reading((HELD[0], far[1]), far, whole) for key in ("0", "1", "2", "3")}
        points["free"] = reading((HELD[0], []), None, whole)  # no far bin and no control: skipped by readings()
    return {
        "run": cell_name(train_radius, width), "args": [], "seconds": seconds, "tables": 2, "layers": layers, "width": width, "scale": 0.1768,
        "lr": 0.025, "train_steps": 300, "hit_cap": hit_cap, "evaluations": list(evaluations), "train_first": 2.0, "train_last": 1.0,
        "corpus_artifacts": 42000, "corpus_windows": 3650, "whole": whole, "same": reading(HELD, whole=whole), "held": reading(HELD, whole=whole),
        "points": points, "probe_r2_truth_distance": None, "probe_r2_rank": None,
    }


def three(far_at_largest=RISING):
    return {(1, 96): cell(width=96, layers=7), (1, 144): cell(width=144, layers=8), (1, 192): cell(width=192, layers=9, far=far_at_largest)}


class TestFloorAndSizes(unittest.TestCase):
    def test_round_up_to_a_multiple_of_eight(self):
        self.assertEqual(round_up_to(94, 8), 96)
        self.assertEqual(round_up_to(96, 8), 96)
        self.assertEqual(round_up_to(144, 8), 144)

    def test_the_derived_floor_on_decisions_33_seed_1_two_tables(self):
        floor = derived_floor(distinct_strings=94, suffix_depth=6, mean_string_length=4.4, mean_artifacts_per_window=11.4, effective_count=30.8, cap=6000)
        self.assertEqual(floor["width"], 96)
        self.assertEqual(floor["layers"], 7)
        self.assertEqual(floor["length"], 44)
        self.assertEqual(floor["cap_floor"], 1398)  # ⌈⌈4 · 2 · 96² · 3 / (8 · log₂ 30.8)⌉ / 4⌉
        self.assertEqual(floor["corpus_floor"], 27360)
        self.assertEqual(corpus_floor_for(6000, 11.4), 27360)

    def test_the_three_sizes_and_the_next(self):
        self.assertEqual(sizes(96, 7), [(96, 7), (144, 8), (192, 9)])
        self.assertEqual(next_size(192, 9), (288, 10))


class TestCellCommand(unittest.TestCase):
    def test_argv_follows_sweep7s_usage_order_as_strings(self):
        argv = cell_command("python3", "runs/sweep7.py", "sweep-07-r1-w96", ENSEMBLE, VECTOR, 1)
        self.assertEqual(argv, [
            "python3", "runs/sweep7.py", "sweep-07-r1-w96", "4", "3", "4", "96", "7", "64", "2", "1", "1.0", "0.5",
            "42000", "0,1,2,3,free", "2000", "6000", "1", "0.025", "3", "300", "3/4",
        ])
        self.assertTrue(all(isinstance(x, str) for x in argv))
        self.assertEqual(cell_name(2, 144), "sweep-07-r2-w144")

    def test_cell_at_matches_the_vector_from_the_cells_args(self):
        args = cell_command("python3", "runs/sweep7.py", "sweep-07-r1-w96", ENSEMBLE, VECTOR, 1)[2:]
        self.assertTrue(cell_at({"args": args}, VECTOR, 1))
        self.assertFalse(cell_at({"args": args}, VECTOR, 2))
        self.assertFalse(cell_at({"args": args}, Vector(96, 7, 64, 42000, 2000, 6000, 0.0125), 1))
        self.assertFalse(cell_at({"args": args}, Vector(96, 7, 96, 42000, 2000, 6000, 0.025), 1))
        self.assertFalse(cell_at({"args": []}, VECTOR, 1))
        by_hand = "sweep-07-r1-w96 4 3 4 96 7 64 2 1 1 0.5 42000 0,1,2,3,free 2000 6000 1 0.025 3 300".split()  # runs/README.md's line
        self.assertTrue(cell_at({"args": by_hand}, VECTOR, 1))
        self.assertTrue(cell_at({"args": by_hand[:16]}, Vector(96, 7, 64, 42000, 2000, 6000, 0.1), 1))  # lr absent: sweep7's default


class TestParseCell(unittest.TestCase):
    def test_the_last_json_line_after_progress_lines(self):
        log = 'step 50 loss 1.2\nstep 100 loss 1.1\n{"run": "a", "seconds": 1.0}\n{"run": "b", "seconds": 2.0}\n'
        self.assertEqual(parse_cell(log)["run"], "b")

    def test_a_log_without_a_json_line(self):
        self.assertIsNone(parse_cell("step 50 loss 1.2\nTraceback (most recent call last):\n"))
        self.assertIsNone(parse_cell(""))
        self.assertIsNone(parse_cell('step 50\n{"run": "a", "seconds": 1.'))  # a truncated line is not a JSON line


class TestReadings(unittest.TestCase):
    def test_past_radius_keys(self):
        self.assertEqual(past_radius_keys(1, ("0", "1", "2", "3", "free")), ["2", "3", "free"])
        self.assertEqual(past_radius_keys(3, ("0", "1", "2", "3", "free")), ["free"])

    def test_values_and_signs_on_known_curves(self):
        points = {
            "0": reading((HELD[0], RISING[1]), RISING), "1": reading((HELD[0], RISING[1]), RISING),
            "2": reading((HELD[0], RISING[1]), RISING), "3": reading((HELD[0], FLAT[1]), FLAT), "free": reading((HELD[0], []), None),
        }
        out = readings(cell(points=points), 1)
        self.assertEqual(set(out), {"slope_2", "level_2", "control_2", "slope_3", "level_3", "control_3", "control_held"})
        self.assertAlmostEqual(out["slope_2"][0], 0.3)
        self.assertEqual(out["slope_2"][1], 1)
        self.assertAlmostEqual(out["level_2"][0], 0.8)
        self.assertEqual(out["level_2"][1], 1)
        self.assertAlmostEqual(out["control_2"][0], -0.05)
        self.assertEqual(out["control_2"][1], -1)
        self.assertAlmostEqual(out["slope_3"][0], 0.0)
        self.assertEqual(out["slope_3"][1], 0)
        self.assertAlmostEqual(out["level_3"][0], 0.2)  # rank 2, the last rank the far bin reaches
        self.assertEqual(out["level_3"][1], 1)
        self.assertEqual(out["control_3"][1], 1)
        self.assertAlmostEqual(out["control_held"][0], 0.2)
        self.assertEqual(out["control_held"][1], 1)

    def test_the_band_turns_a_small_value_to_zero(self):
        out = readings(cell(), 1, band=0.1)
        self.assertEqual(out["control_2"][1], 0)
        self.assertEqual(out["slope_2"][1], 1)


class TestRows(unittest.TestCase):
    def test_r1_instability(self):
        fires, detail = r1_instability(cell(evaluations=(1.0, 0.9, 2.0)))
        self.assertTrue(fires)
        self.assertIn("sweep-07-r1-w96", detail)
        self.assertFalse(r1_instability(cell(evaluations=(1.0, 0.9, 1.7)))[0])
        self.assertFalse(r1_instability(cell(evaluations=(1.0,)))[0])

    def test_r2_budget(self):
        self.assertTrue(r2_budget(cell(evaluations=(1.0, 0.9), hit_cap=True))[0])
        self.assertFalse(r2_budget(cell(evaluations=(1.0, 0.9), hit_cap=False))[0])
        self.assertFalse(r2_budget(cell(evaluations=(1.0, 0.995), hit_cap=True))[0])
        self.assertTrue(r2_budget(cell(evaluations=(1.0, 0.995), hit_cap=True), tolerance=0.001)[0])
        self.assertFalse(r2_budget(cell(evaluations=(1.0,), hit_cap=True))[0])

    def test_r3_k_range(self):
        self.assertFalse(r3_k_range(cell(whole=8))[0])
        fires, detail = r3_k_range(cell(whole=7))
        self.assertTrue(fires)
        self.assertIn("corpus", detail)
        short = cell(whole=10)
        short["points"]["3"]["whole"] = 5
        fires, detail = r3_k_range(short)
        self.assertTrue(fires)
        self.assertIn("3", detail)
        self.assertNotIn("corpus", detail)

    def test_r4_agreeing_and_disagreeing_sizes(self):
        cells = three()
        fires, detail = r4_disagree([cells[key] for key in sorted(cells)], 1)
        self.assertFalse(fires)
        self.assertIn("did not hold", detail)
        cells = three(far_at_largest=FALLING)
        fires, detail = r4_disagree([cells[key] for key in sorted(cells)], 1)
        self.assertTrue(fires)
        self.assertIn("slope_2 (+, +, -)", detail)
        self.assertNotIn("level_2", detail)  # +0.8, +0.8, +0.4: the same sign

    def test_host_exceeds_at_the_boundary(self):
        self.assertFalse(host_exceeds(3600.0, 96, 7, 96, 7)[0])
        self.assertTrue(host_exceeds(3600.1, 96, 7, 96, 7)[0])
        self.assertFalse(host_exceeds(1399.0, 96, 7, 144, 8)[0])  # 1399 · 2.25 · 8/7 = 3597
        self.assertTrue(host_exceeds(1401.0, 96, 7, 144, 8)[0])  # 3603
        self.assertTrue(host_exceeds(1399.0, 96, 7, 144, 8, limit_seconds=3000.0)[0])


class TestApplyRows(unittest.TestCase):
    def test_r1_before_r2_when_both_fire(self):
        cells = {(1, 96): cell(evaluations=(1.0, 2.5, 1.0), hit_cap=True)}
        self.assertTrue(r2_budget(cells[(1, 96)])[0])
        row, following, _ = apply_rows(cells, (1,), VECTOR, 11.4)
        self.assertEqual(row, "R1")
        self.assertEqual(following, VECTOR)  # the vector stands; the sizes that fired have their rate halved by the loop (decisions 40)

    def test_r2_doubles_the_cap_and_raises_the_corpus_to_its_floor(self):
        row, following, _ = apply_rows({(1, 96): cell(evaluations=(1.0, 0.9), hit_cap=True)}, (1,), VECTOR, 11.4)
        self.assertEqual(row, "R2")
        self.assertEqual(following, Vector(96, 7, 64, 54720, 2000, 12000, 0.025))
        _, following, _ = apply_rows({(1, 96): cell(evaluations=(1.0, 0.9), hit_cap=True)}, (1,), VECTOR, 1.0)
        self.assertEqual(following.corpus, 42000)  # never lowered

    def test_r3_lengthens_the_window(self):
        row, following, _ = apply_rows({(1, 96): cell(whole=7)}, (1,), VECTOR, 11.4)
        self.assertEqual(row, "R3")
        self.assertEqual(following, Vector(96, 7, 96, 42000, 2000, 6000, 0.025))

    def test_r4_moves_the_sizes_not_the_vector_and_stop_keeps_it(self):
        row, following, detail = apply_rows(three(far_at_largest=FALLING), (1,), VECTOR, 11.4)
        self.assertEqual((row, following), ("R4", None))
        self.assertIn("slope_2", detail)
        self.assertEqual(apply_rows(three(), (1,), VECTOR, 11.4), ("stop", VECTOR, "no row fires"))


class TestLogEntry(unittest.TestCase):
    def test_the_entry_names_the_row_and_the_seconds_and_no_curve_number(self):
        cells = {(1, 96): cell(far=([0.123456, 0.234567, 0.345678, 0.456789, 0.567891], [0.0, -0.054321]), seconds=712.3)}
        row, following, detail = apply_rows(cells, (1,), VECTOR, 11.4)
        entry = log_entry("2026-10-09 18:00:00", VECTOR, [(96, 7)], cells, row, detail, following)
        self.assertTrue(entry.startswith("- 2026-10-09 18:00:00 · vector (96, 7, 64, 42000, 2000, 6000, 0.025)"))
        self.assertIn("sweep-07-r1-w96 712.3 s", entry)
        self.assertIn("row stop: no row fires", entry)
        self.assertIn("next: (96, 7, 64, 42000, 2000, 6000, 0.025)", entry)
        for number in ("0.123456", "0.567891", "0.054321", "0.4443"):
            self.assertNotIn(number, entry)
        self.assertIn("next: stop", log_entry("s", VECTOR, [(96, 7)], cells, "host", "held", None))
        row, _, detail = apply_rows(three(far_at_largest=FALLING), (1,), VECTOR, 11.4)
        entry = log_entry("s", VECTOR, sizes(96, 7), three(far_at_largest=FALLING), row, detail, VECTOR)
        self.assertIn("row R4: held over sweep-07-r1-w96, sweep-07-r1-w144, sweep-07-r1-w192: slope_2 (+, +, -)", entry)
        self.assertNotIn("0.3", entry)


if __name__ == "__main__":
    unittest.main()


class TestRatePerSize(unittest.TestCase):
    """Decisions 40: the rate is per size; R1 names the cells and the loop halves their sizes' rates."""

    def test_rate_of_falls_back_to_the_vector(self):
        from cocoonml.rules import rate_of

        rates = {(144, 8): 0.0125}
        self.assertEqual(rate_of(rates, VECTOR), 0.025)
        self.assertEqual(rate_of(rates, Vector(144, 8, 64, 42000, 2000, 6000, 0.025)), 0.0125)
        self.assertEqual(rate_of({}, VECTOR), 0.025)

    def test_cell_command_and_cell_at_use_the_size_rate(self):
        from cocoonml.rules import cell_at, cell_command

        big = Vector(144, 8, 64, 42000, 2000, 6000, 0.025)
        rates = {(144, 8): 0.0125}
        argv = cell_command("python3", "runs/sweep7.py", "sweep-07-r1-w144", ENSEMBLE, big, 1, rates)
        self.assertEqual(argv[18], "0.0125")  # python, script, name, forms, nts, res, width, layers, length, tables, radius, steps, switch, corpus, read radii, read size, cap, seed, lr
        cell = {"args": argv[2:]}
        self.assertTrue(cell_at(cell, big, 1, rates))
        self.assertFalse(cell_at(cell, big, 1))  # without the rates the vector's 0.025 is expected

    def test_r1_fires_on_a_diverged_cell_and_names_its_size(self):
        from cocoonml.rules import r1_instability, sizes_that_fired_r1

        cell = {"run": "sweep-07-r1-w192", "width": 192, "layers": 9, "diverged": True, "evaluations": []}
        fine = {"run": "sweep-07-r1-w96", "width": 96, "layers": 7, "diverged": False, "evaluations": [1.0, 0.9]}
        self.assertTrue(r1_instability(cell)[0])
        self.assertFalse(r1_instability(fine)[0])
        self.assertEqual(sizes_that_fired_r1({(1, 192): cell, (1, 96): fine}), [(192, 9)])

    def test_cell_name_takes_a_prefix(self):
        from cocoonml.rules import cell_name

        self.assertEqual(cell_name(2, 144, "sweep-07b"), "sweep-07b-r2-w144")
