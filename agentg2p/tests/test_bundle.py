# -*- coding: utf-8 -*-
"""Integration checks on a full data folder (set AGENTG2P_TEST_BUNDLE; skipped otherwise)."""

import json
import os
import sys
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AGENTG2P_OFFLINE", "1")  # tests never download
sys.path.insert(0, str(ROOT / "src"))

import agentg2p  # noqa: E402
from agentg2p import byt5  # noqa: E402

BUNDLE = Path(os.environ.get("AGENTG2P_TEST_BUNDLE", ROOT / "no-bundle"))


@unittest.skipUnless((BUNDLE / "manifest.json").exists(), "set AGENTG2P_TEST_BUNDLE to a full data folder")
class BundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))

    def test_manifest_integrity(self):
        models = self.manifest["models"]
        self.assertEqual(set(models), {"byt5-pooled", "byt5-mfa", "byt5-wiktionary", "byt5-charsiu"})
        for m in models.values():
            self.assertTrue((BUNDLE / m["path"] / "model.safetensors").is_file())
        ids = set()
        for d in self.manifest["dictionaries"]:
            self.assertNotIn(d["id"], ids)
            ids.add(d["id"])
            self.assertTrue((BUNDLE / d["lexicon"]).is_file(), d["id"])
            if d["fst"]:
                self.assertTrue((BUNDLE / d["fst"]).is_file(), d["id"])
            self.assertTrue(d["backend"] == "fst" or d["backend"] in models, d["id"])
            if d["backend"] == "fst":
                self.assertIsNotNone(d["fst"], d["id"])
            self.assertIn(d["backend"], d["scores"], d["id"])
            self.assertGreater(d["words"], 0)
        self.assertEqual(len(ids), 717)

    def test_language_resolution(self):
        expect = {"de": "deu", "German": "deu", "en-US": "eng-us", "en-GB": "eng-uk", "pt-BR": "por-br",
                  "zh": "cmn", "ja": "jpn", "ko": "kor", "ru": "rus", "fr": "fra", "es": "spa", "hi": "hin"}
        for code, tag in expect.items():
            g = agentg2p.load(code, data_dir=BUNDLE)
            self.assertTrue(g.info.lang == tag or g.info.lang.startswith(tag + "-"), (code, g.info.id))
            self.assertNotIn(g.info.source_tag, {"mfa-arpa", "mfa-jamo"})

    def test_lookup_and_models(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = agentg2p.G2P(dictionary="wiktionary/deu_latn_broad", data_dir=BUNDLE)
            word = next(w for w in g.lexicon.words() if w.isalpha() and len(w) > 4)
            tok = g(word)[0]
            self.assertEqual((tok.source, tok.phones), ("lexicon", g.lexicon.get(word)[0]))
            fst = agentg2p.G2P(dictionary="wiktionary/deu_latn_broad", backend="fst", data_dir=BUNDLE)
            tok = [t for t in fst("Schmetterlingsflügelchen") if t.is_word][0]
            self.assertEqual(tok.source, "fst")
            self.assertTrue(tok.phones)
            if byt5.available():
                nn = agentg2p.G2P(dictionary="wiktionary/deu_latn_broad", backend="byt5", data_dir=BUNDLE)
                tok = [t for t in nn("Schmetterlingsflügelchen") if t.is_word][0]
                self.assertEqual(tok.source, "byt5")
                self.assertTrue(tok.phones)

    def test_every_language_transcribes(self):
        texts = {"en": "hello world", "de": "Guten Morgen", "ja": "東京に行きたい", "zh": "我们去北京",
                 "ar": "مرحبا بالعالم", "th": "สวัสดีครับ", "vi": "Tôi là người Việt Nam", "ko": "안녕하세요"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for lang, text in texts.items():
                toks = [t for t in agentg2p.load(lang, data_dir=BUNDLE)(text) if t.is_word]
                self.assertTrue(toks, lang)
                self.assertTrue(all(t.phones for t in toks), (lang, [(t.text, t.phones) for t in toks]))


if __name__ == "__main__":
    unittest.main()
