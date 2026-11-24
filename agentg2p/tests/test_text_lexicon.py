# -*- coding: utf-8 -*-
"""agentg2p.lexicon and agentg2p.text: parsing, lookup, tokenization, segmentation."""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AGENTG2P_OFFLINE", "1")  # tests never download
sys.path.insert(0, str(ROOT / "src"))

from agentg2p.lexicon import Lexicon, parse_line  # noqa: E402
from agentg2p.text import Segmenter, split_spans  # noqa: E402


class ParseTest(unittest.TestCase):
    def test_mfa_line_with_scores_and_variants(self):
        word, variants = parse_line("a\t0.99 0.13 1.48 0.93 aː,0.01 0.21 1.0 1.0 a\n")
        self.assertEqual(word, "a")
        self.assertEqual(variants, [["aː"], ["a"]])

    def test_wikipron_variants(self):
        self.assertEqual(parse_line("'n\tm,n,ŋ,ə n"), ("'n", [["m"], ["n"], ["ŋ"], ["ə", "n"]]))

    def test_junk_is_dropped(self):
        for line in ("[bracketed]\t0.99 0.83 2.11 0.29 spn", "<unk>\tspn", "word\t", "no tab here"):
            self.assertIsNone(parse_line(line), line)
        self.assertEqual(parse_line("hm\tspn,h m"), ("hm", [["h", "m"]]))

    def test_nfc(self):
        word, variants = parse_line("café\té")
        self.assertEqual(word, "café")
        self.assertEqual(variants, [["é"]])


class LexiconTest(unittest.TestCase):
    def setUp(self):
        self.lex = Lexicon.from_dict({
            "morgen": [["m", "ɔ", "ʁ", "ɡ", "ə", "n"]], "Haus": [["h", "aʊ", "s"]],
            "don't": [["d", "oʊ", "n", "t"]], "ice cream": [["aɪ", "s", "k", "ɹ", "iː", "m"]],
            "a": [["aː"], ["a"]]})

    def test_case_candidates(self):
        self.assertEqual(self.lex.find("Morgen"), "morgen")
        self.assertEqual(self.lex.find("MORGEN"), "morgen")
        self.assertEqual(self.lex.find("haus"), "Haus")
        self.assertIsNone(self.lex.find("hause"))

    def test_apostrophes(self):
        self.assertEqual(self.lex.find("don’t"), "don't")
        self.assertEqual(self.lex.lookup("Don’t"), [["d", "oʊ", "n", "t"]])

    def test_variants_and_stats(self):
        self.assertEqual(self.lex.lookup("a"), [["aː"], ["a"]])
        self.assertEqual(self.lex.max_words, 2)


class SplitTest(unittest.TestCase):
    def test_round_trip(self):
        for text in ("Guten Morgen, Welt!", "don't  stop-it 'cause", "日本語のテキスト。", "a‌b c—d", ""):
            self.assertEqual("".join(s for s, _ in split_spans(text)), text)

    def test_words(self):
        spans = [s for s, w in split_spans("Hello, it's well-known (really).") if w]
        self.assertEqual(spans, ["Hello", "it's", "well-known", "really"])
        # Combining marks stay inside the word (Devanagari vowel signs, ZWNJ in Persian).
        self.assertEqual([s for s, w in split_spans("नमस्ते दुनिया") if w], ["नमस्ते", "दुनिया"])
        self.assertEqual([s for s, w in split_spans("می‌خواهم") if w], ["می‌خواهم"])
        self.assertEqual([s for s, w in split_spans("abc123") if w], ["abc"])


class SegmenterTest(unittest.TestCase):
    def setUp(self):
        self.lex = Lexicon.from_dict({
            w: [["x"]] for w in ["ice", "cream", "ice cream", "'cause", "well", "known",
                                 "日本", "日本語", "語", "本", "東京", "に", "行く", "Việt Nam", "Việt"]})
        self.seg = Segmenter(self.lex)

    def words(self, text):
        return [(t.text, t.form) for t in self.seg.units(text) if t.is_word]

    def test_multiword_entries(self):
        self.assertEqual(self.words("I like ice cream"),
                         [("I", None), ("like", None), ("ice cream", "ice cream")])
        self.assertEqual(self.words("Việt Nam"), [("Việt Nam", "Việt Nam")])
        self.assertEqual(self.words("ice  cream"), [("ice", "ice"), ("cream", "cream")])  # two spaces

    def test_edge_apostrophes_and_hyphens(self):
        self.assertEqual(self.words("'cause"), [("'cause", "'cause")])
        toks = self.seg.units("dogs' well-known")
        self.assertEqual([(t.text, t.is_word) for t in toks],
                         [("dogs", True), ("'", False), (" ", False), ("well", True), ("-", False),
                          ("known", True)])

    def test_unsegmented_scripts(self):
        self.assertEqual(self.words("東京に行く"), [("東京", "東京"), ("に", "に"), ("行く", "行く")])
        self.assertEqual(self.words("日本語"), [("日本語", "日本語")])
        # Unknown characters stay together and go to the model.
        self.assertEqual(self.words("日本猫犬に"), [("日本", "日本"), ("猫犬", None), ("に", "に")])

    def test_without_lexicon(self):
        seg = Segmenter(None)
        self.assertEqual([t.text for t in seg.units("東京に well-known") if t.is_word],
                         ["東京に", "well", "known"])


if __name__ == "__main__":
    unittest.main()
