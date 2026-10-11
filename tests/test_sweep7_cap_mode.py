import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOY = ["smoke-07-cap", "3", "2", "4", "3", "2", "16", "2", "1", "1", "0.5", "40", "0,2,free", "20", "120", "1", "0.3", "3", "5", "3/4"]


class TestSweep7CapMode(unittest.TestCase):
    """runs/sweep7.py in mode "cap" (decisions 53) at toy size: the shape of its JSON line."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")
        done = subprocess.run([sys.executable, os.path.join(ROOT, "runs", "sweep7.py"), *TOY, "cap", cls.tmp.name], capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
        cls.returncode, cls.stderr = done.returncode, done.stderr
        lines = [l for l in done.stdout.splitlines() if l.startswith('{"run"')]
        cls.line = json.loads(lines[-1]) if lines else None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_cell_ran_to_its_cap_or_its_escape_with_no_rule(self):
        self.assertEqual(self.returncode, 0, self.stderr)
        line = self.line
        self.assertEqual(line["mode"], "cap")
        self.assertTrue(line["train_steps"] == 120 or line["escape_step"] is not None)
        self.assertEqual(len(line["batch_losses"]), line["train_steps"])  # every batch loss written
        self.assertEqual(len(line["evaluations"]), line["train_steps"] // 50)

    def test_the_copies_are_written_and_the_read_copy_is_named(self):
        copies = self.line["copies"]
        self.assertEqual(copies["marginal"]["step"], 50)
        self.assertIn(self.line["read_copy"], ("marginal", "best"))
        if copies["best"] is None:
            self.assertIsNone(self.line["onset_step"])
            self.assertEqual(self.line["read_copy"], "marginal")
        else:
            self.assertGreaterEqual(copies["best"]["step"], self.line["onset_step"])
            self.assertEqual(self.line["read_copy"], "best")
        for name in ("marginal", "best"):
            if copies[name] is not None:
                self.assertEqual(set(copies[name]["points"]), {"0", "2", "free"})  # the readers ran on this copy
        self.assertNotIn("points", copies["last"])  # the last copy is kept and not read
        for name, entry in copies.items():
            if entry is not None:
                self.assertIsInstance(entry["held_loss"], float)  # the copy's held loss, beside its held reading
        for name, entry in copies.items():
            if entry is not None:
                self.assertTrue(os.path.exists(entry["file"]), entry["file"])
                self.assertEqual(len(np.load(entry["file"]).files), 7 + 3)  # six blocks, the within block, and one more layer's three

    def test_the_top_level_readings_are_the_read_copy_s(self):
        chosen = self.line["copies"][self.line["read_copy"]]
        self.assertEqual(self.line["points"], chosen["points"])
        self.assertEqual(self.line["held"], chosen["held"])  # the held reading of the read copy
        self.assertEqual(self.line["same"], chosen["same"])

    def test_plateau_mode_is_unchanged_in_shape(self):
        env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")
        done = subprocess.run([sys.executable, os.path.join(ROOT, "runs", "sweep7.py"), *TOY], capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        line = json.loads([l for l in done.stdout.splitlines() if l.startswith('{"run"')][-1])
        self.assertEqual(line["mode"], "plateau")
        self.assertNotIn("copies", line)
        self.assertIn("points", line)
        self.assertEqual(len(line["circuits"]["induction"]), 2)  # the circuit scores per layer, both modes


if __name__ == "__main__":
    unittest.main()
