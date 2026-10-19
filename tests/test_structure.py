"""The repository's structure as a test: what a script can decide from the tracked files alone (2026-10-19).

No record of a session or a chat is ever tracked here (the records live in the private records repository; a
private session reached a public branch of another repository on 2026-10-17 and nothing refused it), and no
project .npmrc (npm reads it over the user's config, and the sandbox's start would run install scripts).
The records-guard workflow refuses the same at the push; this refuses it at the merge.
"""
import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"^(records|chat|exports)/|(^|/)export-(claude|gemini)\.md$|(^|/)\d{4}-\d{2}-\d{2}-session-[0-9a-f]{8}\.(md|jsonl)$|(^|/)\.npmrc$")


class Structure(unittest.TestCase):
    def tracked(self):
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
        return [p for p in out.split("\n") if p]

    def test_no_record_and_no_npmrc_is_tracked(self):
        hits = [p for p in self.tracked() if FORBIDDEN.search(p)]
        self.assertEqual(hits, [], "a record or a project .npmrc is tracked; neither ever belongs in a public repository")

    def test_the_code_owners_file_names_the_owner_for_the_workflows(self):
        text = (ROOT / "CODEOWNERS").read_text()
        self.assertRegex(text, r"(?m)^/\.github/\s+@theCocoonStudio$")
