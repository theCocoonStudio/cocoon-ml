import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOY = ["smoke-08", "3", "8", "8", "2", "16", "2", "1", "1", "0.5", "40", "0,2,free", "20", "120", "1", "0.3", "3", "5", "1.0", "0.5"]


class TestSweep8(unittest.TestCase):
    """runs/sweep8.py, the Markov-chain world through the model and readers, at toy size."""

    def run_it(self, *extra):
        env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")
        done = subprocess.run([sys.executable, os.path.join(ROOT, "runs", "sweep8.py"), *TOY, *extra], capture_output=True, text=True, env=env, cwd=ROOT, timeout=180)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads([l for l in done.stdout.splitlines() if l.startswith('{"run"')][-1])

    def test_cap_mode_carries_the_copies_the_circuits_and_the_two_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            line = self.run_it("cap", tmp, "20")
        self.assertEqual(line["world"], "markov")
        self.assertEqual((line["states"], line["resolution"]), (3, 8))
        self.assertTrue(line["train_steps"] == 120 or line["escape_step"] is not None)
        self.assertEqual(set(line["points"]), {"0", "2", "free"})
        self.assertEqual(len(line["circuits"]["induction"]), 2)
        ref = line["reference"]
        self.assertEqual(ref["floor_estimate"], 20)  # particles, since the training walk steps
        self.assertLess(ref["floor"], ref["marginal"])
        self.assertEqual(len(ref["floor_per_window"]), len(ref["marginal_per_window"]))

    def test_with_no_steps_the_floor_is_exact(self):
        toy = list(TOY)
        toy[8] = "0"  # steps_mean
        env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")
        done = subprocess.run([sys.executable, os.path.join(ROOT, "runs", "sweep8.py"), *toy], capture_output=True, text=True, env=env, cwd=ROOT, timeout=180)
        self.assertEqual(done.returncode, 0, done.stderr)
        line = json.loads([l for l in done.stdout.splitlines() if l.startswith('{"run"')][-1])
        self.assertEqual(line["mode"], "plateau")
        self.assertEqual(line["reference"]["floor_estimate"], "exact")
        self.assertLess(line["reference"]["floor"], line["reference"]["marginal"])


if __name__ == "__main__":
    unittest.main()
