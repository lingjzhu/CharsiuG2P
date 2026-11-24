# -*- coding: utf-8 -*-
"""agentg2p.fst: reading OpenFst models and 1-best decoding.

Run: python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AGENTG2P_OFFLINE", "1")  # tests never download
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentg2p.fst import FSTModel  # noqa: E402
from fst_fixtures import ISYMS, OSYMS, STATES, all_words, brute_force, write_fst  # noqa: E402


class ToyModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "toy.fst"
        write_fst(cls.path, 0, STATES, ISYMS, OSYMS)
        cls.model = FSTModel.from_openfst(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reads_structure(self):
        m = self.model
        self.assertEqual(len(m.final), 3)
        self.assertEqual(len(m.ilabel), sum(len(a) for _, a in STATES))
        self.assertEqual(m.isyms, ISYMS)
        self.assertEqual(list(m.n_eps), [1, 0, 1])
        # Arcs sorted by input label within each state, epsilons first.
        for s in range(3):
            labels = list(m.ilabel[m.state_ptr[s]:m.state_ptr[s + 1]])
            self.assertEqual(labels, sorted(labels))

    def test_hand_checked_paths(self):
        d = self.model.decode
        self.assertEqual(d("ab"), ["x", "z"])        # a:x then b:z after "a" beats the a|b chunk
        self.assertEqual(d("a"), ["x"])              # negative backoff weight is taken
        self.assertEqual(d("b"), ["y"])
        self.assertEqual(d("abc"), ["x", "z"])       # c maps to "_" (deleted)
        self.assertIsNone(d("c"))                    # only deletions: no pronunciation, like phonetisaurus
        self.assertEqual(d("}"), ["}"])              # reserved character, escaped like training
        self.assertEqual(d("a b"), ["x", "z", "y"])  # space -> placeholder grapheme
        self.assertIsNone(d(""))

    def test_unknown_characters_are_dropped(self):
        # phonetisaurus-g2pfst skips characters missing from the symbol table.
        self.assertEqual(self.model.decode("ad"), ["x"])
        self.assertEqual(self.model.decode("a\u2603b"), ["x", "z"])
        self.assertIsNone(self.model.decode("dd"))

    def test_cost(self):
        m = self.model
        self.assertEqual(m.decode("ab"), ["x", "z"])
        self.assertAlmostEqual(m.cost("ab", ["x", "z"]), m.last_cost, places=5)
        self.assertAlmostEqual(m.cost("ab", ["x", "z"]), 1.4, places=5)
        self.assertAlmostEqual(m.cost("ab", ["x", "y"]), 2.3, places=5)  # the a|b chunk path
        self.assertIsNone(m.cost("ab", ["z"]))
        for word in all_words("abc", 4):
            hyp = m.decode(word)
            self.assertAlmostEqual(m.cost(word, hyp or []), m.last_cost, places=5, msg=word)

    def test_matches_brute_force(self):
        for word in all_words("abc}", 5):
            self.assertEqual(self.model.decode(word), brute_force(word), word)

    def test_npz_round_trip(self):
        out = Path(self.tmp.name) / "toy.npz"
        self.model.save(out)
        loaded = FSTModel.load(out)
        for word in all_words("abc", 4):
            self.assertEqual(loaded.decode(word), self.model.decode(word))
        self.assertEqual(loaded.graphemes(), {"a", "b", "c", "\ue000", "\u2581"})


class RealModelTest(unittest.TestCase):
    """A real phonetisaurus-train model, if one is given in $AGENTG2P_TEST_FST
    (the AgentG2P research repository has test/temp_model.fst)."""

    def test_loads_and_decodes(self):
        path = Path(os.environ.get("AGENTG2P_TEST_FST", ROOT / "tests" / "temp_model.fst"))
        if not path.exists():
            self.skipTest("no real model given")
        m = FSTModel.from_openfst(path)
        self.assertEqual(len(m.final), 3603)
        self.assertEqual(m.decode("Chaka"), ["t͡ʃ", "a", "k", "a"])
        # Unknown characters are dropped, as phonetisaurus-g2pfst does.
        self.assertEqual(m.decode("Cha\u2603ka"), m.decode("Chaka"))


if __name__ == "__main__":
    unittest.main()
