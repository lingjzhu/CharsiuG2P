# -*- coding: utf-8 -*-
"""The desktop window's server (agentg2p ui): API results on the toy bundle, request checks, CLI."""

import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AGENTG2P_OFFLINE", "1")  # tests never download
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentg2p import G2P, cli  # noqa: E402
from agentg2p.ui import server as ui_server  # noqa: E402
from agentg2p.ui import start_server  # noqa: E402
from test_g2p import make_bundle  # noqa: E402

BUNDLE = Path(os.environ.get("AGENTG2P_TEST_BUNDLE", ROOT / "no-bundle"))


class ToyBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_bundle(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class ApiTest(ToyBundle):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.api = ui_server.Api(cls.root)

    def test_languages_and_dictionaries(self):
        langs = {l["tag"]: l for l in self.api.languages()}
        self.assertEqual(langs["deu"]["name"], "German")
        self.assertEqual(langs["deu"]["alias"], "de")
        self.assertEqual(langs["eng-uk"]["name"], "English (UK)")
        dicts = self.api.dictionaries("deu")
        self.assertEqual([d["id"] for d in dicts], ["toy/deu", "toy/deu_small"])
        self.assertTrue(dicts[0]["default"])
        self.assertEqual(sum(d["default"] for d in dicts), 1)

    def test_transcribe_marks_sources_and_flags(self):
        rows = self.api.transcribe("toy/abc", "cab abd zzz 9")["rows"]
        got = [(r["kind"], r["text"], r["phones"], r["source"]) for r in rows]
        self.assertEqual(got, [("word", "cab", ["k", "a", "b"], "lexicon"), ("word", "abd", ["x", "z"], "fst"),
                               ("word", "zzz", None, None), ("number", "9", None, None)])
        self.assertIsNone(rows[0]["flag"])
        self.assertEqual(rows[0]["variants"], [["k", "a", "b"], ["k", "æ", "b"]])
        self.assertEqual(rows[1]["flag"], "d dropped: not in this dictionary's alphabet")
        self.assertTrue(rows[2]["flag"].startswith("No transcription"))
        self.assertEqual(rows[3]["flag"], "Numbers are not read out")

    def test_output_is_the_library_output(self):
        text = "cab ab, ba ba!"
        g2p = G2P(dictionary="toy/abc", data_dir=self.root)
        expected = [(t.text, t.phones, t.source) for t in g2p.tokens(text) if t.is_word]
        got = [(r["text"], r["phones"], r["source"]) for r in self.api.transcribe("toy/abc", text)["rows"]]
        self.assertEqual(got, expected)
        words = ["cab", "b", "zzz", "AB"]
        rows = self.api.words("toy/abc", words)["rows"]
        self.assertEqual([r["text"] for r in rows], words)
        self.assertEqual([r["phones"] for r in rows], g2p.batch(words))

    def test_compare(self):
        res = self.api.compare("abc", "ab")
        self.assertEqual(len(res["rows"]), 1)
        self.assertEqual(res["rows"][0]["variants"], [["x", "z"]])
        self.assertEqual(res["rows"][0]["source"], "fst")

    def test_user_errors(self):
        with self.assertRaises(ui_server.UserError):
            self.api.download("nope/nothing")
        with self.assertRaises(ui_server.UserError) as cm:
            self.api.download("toy/eng_uk")  # files missing, nothing to download
        self.assertIn("eng_uk is not available to download", str(cm.exception))
        with self.assertRaises(ui_server.UserError):
            self.api.transcribe("toy/abc", "x" * (ui_server.MAX_TEXT + 1))

    def test_prefs_and_save(self):
        old = ui_server.PREFS_PATH
        with tempfile.TemporaryDirectory() as tmp:
            ui_server.PREFS_PATH = Path(tmp) / "ui.json"
            try:
                self.assertEqual(self.api.prefs(), {})
                self.api.save_prefs({"lang": "deu", "recent": ["deu"]})
                self.assertEqual(self.api.prefs()["lang"], "deu")
            finally:
                ui_server.PREFS_PATH = old
        self.assertEqual(self.api.save("x.tsv", "a\tb\n"), {"download": True})


class ServerTest(ToyBundle):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.server = start_server(0, cls.root)
        cls.base = cls.server.url

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        super().tearDownClass()

    def request(self, path, data=None, headers=None):
        req = urllib.request.Request(self.base + path.lstrip("/"), data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read()

    def call(self, name, args, token=None):
        return self.request(f"api/{name}", json.dumps(args).encode(),
                            {"Content-Type": "application/json",
                             "X-Agentg2p-Token": self.server.token if token is None else token})

    def test_index_carries_the_token(self):
        status, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(self.server.token.encode(), body)

    def test_api_needs_the_token(self):
        self.assertEqual(self.call("info", {}, token="wrong")[0], 403)
        status, body = self.call("transcribe", {"dictionary": "toy/abc", "text": "cab"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["result"]["rows"][0]["phones"], ["k", "a", "b"])

    def test_unknown_calls_and_bad_requests(self):
        self.assertEqual(self.call("__init__", {})[0], 404)
        self.assertEqual(self.call("transcribe", {"nonsense": 1})[0], 400)
        status, body = self.call("download", {"dictionary": "nope/nothing"})
        self.assertEqual(status, 400)
        self.assertIn("nope/nothing", json.loads(body)["error"])

    def test_static_files_stay_inside(self):
        self.assertEqual(self.request("/app.js")[0], 200)
        self.assertEqual(self.request("/fonts/charis-sil-ipa.woff2")[0], 200)
        self.assertEqual(self.request("/../server.py")[0], 404)
        self.assertEqual(self.request("/%2e%2e/server.py")[0], 404)


class CliTest(unittest.TestCase):
    def test_ui_subcommand_is_routed(self):
        import agentg2p.ui as ui
        called = []
        old = ui.main
        ui.main = lambda argv: called.append(argv) or 0
        try:
            self.assertEqual(cli.main(["ui", "--browser"]), 0)
        finally:
            ui.main = old
        self.assertEqual(called, [["--browser"]])


@unittest.skipUnless((BUNDLE / "manifest.json").exists(), "set AGENTG2P_TEST_BUNDLE to a full data folder")
class RealBundleTest(unittest.TestCase):
    def test_dropped_letters_are_flagged(self):
        api = ui_server.Api(BUNDLE)
        rows = {r["text"]: r for r in api.transcribe("mfa/german_mfa", "Der Zug nach Łódź")["rows"]}
        self.assertEqual(rows["Zug"]["source"], "lexicon")
        self.assertIn("Ł ó ź dropped", rows["Łódź"]["flag"])
        names = {l["tag"]: l["name"] for l in api.languages()}
        self.assertEqual(names["acw"], "Hijazi Arabic")
        self.assertFalse([t for t, n in names.items() if n == t], "every language has an English name")


if __name__ == "__main__":
    unittest.main()
