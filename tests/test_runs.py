import glob
import os
import py_compile
import unittest


class TestRuns(unittest.TestCase):
    """Every sweep, summariser and smoke script under runs/ compiles: the scripts that produced the
    numbers each review file argues against are part of the record and must at least parse."""

    def test_every_run_script_compiles(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scripts = sorted(glob.glob(os.path.join(here, "runs", "*.py")))
        self.assertTrue(scripts, "runs/ holds no scripts")
        for path in scripts:
            with self.subTest(script=os.path.basename(path)):
                py_compile.compile(path, doraise=True)
