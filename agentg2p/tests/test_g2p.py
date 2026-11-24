# -*- coding: utf-8 -*-
"""agentg2p.G2P end to end on a toy data bundle, language resolution, CLI, ByT5 encoding."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AGENTG2P_OFFLINE", "1")  # tests never download
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agentg2p  # noqa: E402
from agentg2p import G2P, byt5  # noqa: E402
from agentg2p.assets import Catalog  # noqa: E402
from agentg2p.cli import main as cli_main  # noqa: E402
from agentg2p.fst import FSTModel  # noqa: E402
from agentg2p.lexicon import write_lexicon  # noqa: E402
from fst_fixtures import ISYMS, OSYMS, STATES, write_fst  # noqa: E402


def make_bundle(root: Path) -> None:
    write_fst(root / "toy.fst", 0, STATES, ISYMS, OSYMS)
    FSTModel.from_openfst(root / "toy.fst").save(root / "fst" / "toy" / "abc.npz")
    write_lexicon({"cab": [["k", "a", "b"], ["k", "æ", "b"]], "ba ba": [["b", "a", "b", "a"]]},
                  root / "lexicons" / "toy" / "abc.tsv.gz")
    write_lexicon({"Hund": [["h", "ʊ", "n", "t"]]}, root / "lexicons" / "toy" / "deu.tsv.gz")

    def entry(name, lang, words, backend, fst, scores, source_tag="toy"):
        return {"id": f"toy/{name}", "source": "toy", "name": name, "lang": lang,
                "source_tag": source_tag, "words": words, "lexicon": f"lexicons/toy/{name}.tsv.gz",
                "alphabet": "abc " if name == "abc" else "Hdnu", "backend": backend, "fst": fst,
                "scores": scores, "notes": []}

    manifest = {"format": 1, "models": {"byt5-pooled": {"path": "byt5/missing", "kind": "byt5"}},
                "dictionaries": [
                    entry("abc", "abc", 2, "fst", "fst/toy/abc.npz", {"fst": {"per": 0.1}}),
                    entry("deu", "deu", 1, "byt5-pooled", None, {"byt5-pooled": {"per": 0.2}}),
                    entry("deu_small", "deu", 0, "fst", None, {}),
                    entry("eng_uk", "eng-uk", 0, "fst", None, {}),
                    entry("cmn", "cmn", 0, "fst", None, {}),
                    entry("arpa", "eng-us", 99, "fst", None, {}, source_tag="mfa-arpa"),
                    entry("eng_us", "eng-us", 5, "fst", None, {}),
                ]}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class G2PTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_bundle(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_lookup_then_fst(self):
        g = G2P("abc", data_dir=self.root)
        toks = g("cab ab, ba ba!")
        words = [(t.text, t.phones, t.source) for t in toks if t.is_word]
        self.assertEqual(words, [("cab", ["k", "a", "b"], "lexicon"), ("ab", ["x", "z"], "fst"),
                                 ("ba ba", ["b", "a", "b", "a"], "lexicon")])
        self.assertEqual(toks[0].variants, [["k", "a", "b"], ["k", "æ", "b"]])
        self.assertEqual(g.transcribe("cab ab"), "kab xz")
        self.assertEqual(g.transcribe("cab ab", phone_sep=" ", word_sep=" | "), "k a b | x z")
        self.assertEqual(g.word("AB"), ["x", "z"])   # uppercase mapped to the dictionary's lowercase
        self.assertEqual(g.batch(["cab", "b", "zzz"]), [["k", "a", "b"], ["y"], None])
        self.assertEqual(g.variants("cab"), [["k", "a", "b"], ["k", "æ", "b"]])
        self.assertEqual(g.transcribe("zzz"), "?")

    def test_model_only(self):
        g = G2P("abc", data_dir=self.root, use_lexicon=False)
        self.assertEqual(g.word("ab"), ["x", "z"])

    def test_unknown_characters_dropped(self):
        g = G2P("abc", data_dir=self.root)
        self.assertEqual(g.word("abd"), ["x", "z"])

    def test_byt5_backend_missing_falls_back(self):
        g = G2P("deu", data_dir=self.root)
        self.assertEqual(g.info.id, "toy/deu")  # largest German dictionary is the default
        self.assertEqual(g.backends, ["byt5-pooled", "fst"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            toks = g("Hund Katze")
        self.assertEqual([(t.phones, t.source) for t in toks if t.is_word],
                         [(["h", "ʊ", "n", "t"], "lexicon"), (None, None)])

    def test_model_form_uses_alphabet(self):
        g = G2P("deu", data_dir=self.root)
        self.assertEqual(g._model_form("HUND"), "Hund")  # H kept upper, rest lowered
        g = G2P("abc", data_dir=self.root)
        self.assertEqual(g._model_form("ABC"), "abc")

    def test_language_resolution(self):
        cat = Catalog(self.root)
        self.assertEqual(cat.resolve("de").id, "toy/deu")
        self.assertEqual(cat.resolve("German").id, "toy/deu")
        self.assertEqual(cat.resolve("ger").id, "toy/deu")
        self.assertEqual(cat.resolve("en-GB").id, "toy/eng_uk")
        self.assertEqual(cat.resolve("en_US").id, "toy/eng_us")   # ARPAbet is never the default
        self.assertEqual(cat.resolve("zh").id, "toy/cmn")         # macrolanguage -> Mandarin
        self.assertEqual(cat.resolve("en").id, "toy/eng_us")      # "eng" alone: largest IPA variety
        self.assertEqual(cat.resolve(dictionary="toy/deu_small").id, "toy/deu_small")
        self.assertEqual(cat.resolve("deu_small").id, "toy/deu_small")
        with self.assertRaises(KeyError):
            cat.resolve("xx-unknown")

    def test_backend_override(self):
        g = G2P("abc", data_dir=self.root, backend="byt5")
        self.assertEqual(g.backends, ["byt5-pooled", "fst"])
        with self.assertRaises(ValueError):
            G2P("abc", data_dir=self.root, backend="nope")

    def test_module_api_and_cli(self):
        self.assertIn("deu", agentg2p.languages(data_dir=self.root))
        self.assertEqual([d.id for d in agentg2p.dictionaries("de", data_dir=self.root)],
                         ["toy/deu", "toy/deu_small"])
        self.assertEqual(agentg2p.transcribe("cab ab", "abc", data_dir=self.root), "kab xz")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli_main(["-l", "abc", "--data-dir", str(self.root), "cab", "ab"])
        self.assertEqual((code, out.getvalue()), (0, "kab xz\n"))
        out = io.StringIO()
        stdin, sys.stdin = sys.stdin, io.StringIO("cab\nab\n")
        try:
            with contextlib.redirect_stdout(out):
                cli_main(["-l", "abc", "--data-dir", str(self.root), "--words"])
        finally:
            sys.stdin = stdin
        self.assertEqual(out.getvalue(), "cab\tk a b\tlexicon\nab\tx z\tfst\n")


class ByT5EncodingTest(unittest.TestCase):
    def test_matches_training_inputs(self):
        """Same ids as the training code: bytes of "<lang><source>:" + word, each + 3, no EOS."""
        ids = byt5.encode_input("eng-us", "wikipron-broad", "naïve")
        expected = [b + 3 for b in "<eng-us><wikipron-broad>:".encode()] + [b + 3 for b in "naïve".encode()]
        self.assertEqual(ids, expected)
        self.assertEqual(ids[:3], [ord("<") + 3, ord("e") + 3, ord("n") + 3])
        self.assertNotIn(1, ids)  # no EOS on the source
        self.assertEqual(byt5.decode_output([ord("a") + 3, 35, ord("b") + 3, 1, 50]), "a\x20b")
        self.assertEqual(byt5.max_new_tokens(10), 72)
        self.assertEqual(byt5.max_new_tokens(1000), 512)


if __name__ == "__main__":
    unittest.main()
