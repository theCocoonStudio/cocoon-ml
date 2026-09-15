import unittest

import cocoonml.autograd  # noqa: F401  (the package imports; real tests come with the code)


class TestScaffold(unittest.TestCase):
    def test_imports(self):
        self.assertTrue(True)
