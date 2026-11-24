# -*- coding: utf-8 -*-
"""Downloading data from a release: a toy bundle packed by tools/make_release.py, served via file://."""

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentg2p import G2P, remote  # noqa: E402
from agentg2p.assets import _catalog  # noqa: E402
from agentg2p.lexicon import write_lexicon  # noqa: E402
from test_g2p import make_bundle  # noqa: E402


class RemoteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.bundle, self.release, self.data = root / "bundle", root / "release", root / "data"
        self.bundle.mkdir()
        make_bundle(self.bundle)
        # make_release needs a ByT5 folder for every model in the manifest.
        model_dir = self.bundle / "byt5" / "missing"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        manifest = json.loads((self.bundle / "manifest.json").read_text())
        manifest["models"]["byt5-pooled"]["condition"] = "pooled"
        for d in manifest["dictionaries"]:  # resolution-only entries get a placeholder lexicon
            if not (self.bundle / d["lexicon"]).exists():
                write_lexicon({"x": [["x"]]}, self.bundle / d["lexicon"])
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_release.py"), "--bundle", str(self.bundle),
                        "--out", str(self.release), "--tag", "test"], check=True, capture_output=True)
        self.env = {k: os.environ.get(k) for k in ("AGENTG2P_DATA_URL", "AGENTG2P_OFFLINE", "AGENTG2P_DATA")}
        os.environ["AGENTG2P_DATA_URL"] = self.release.as_uri()
        os.environ.pop("AGENTG2P_OFFLINE", None)
        os.environ["AGENTG2P_DATA"] = str(self.data)
        _catalog.cache_clear()

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _catalog.cache_clear()
        self.tmp.cleanup()

    def test_first_use_downloads_what_is_needed(self):
        self.assertFalse(self.data.exists())
        g = G2P("abc", data_dir=self.data)
        self.assertEqual(g.transcribe("cab ab"), "kab xz")
        self.assertTrue((self.data / "manifest.json").is_file())
        self.assertTrue((self.data / "lexicons" / "toy" / "abc.tsv.gz").is_file())
        self.assertTrue((self.data / "fst" / "toy" / "abc.npz").is_file())
        self.assertFalse((self.data / "lexicons" / "toy" / "deu.tsv.gz").exists())  # not needed yet
        self.assertFalse((self.data / "byt5").exists())  # the FST handled every word

    def test_download_ahead_of_time(self):
        remote.download("de", dictionaries=["toy/abc"], data_dir=self.data, neural=False, quiet=True)
        self.assertTrue((self.data / "lexicons" / "toy" / "deu.tsv.gz").is_file())
        self.assertTrue((self.data / "lexicons" / "toy" / "abc.tsv.gz").is_file())
        os.environ["AGENTG2P_OFFLINE"] = "1"  # everything needed is local now
        self.assertEqual(G2P("abc", data_dir=self.data).word("ab"), ["x", "z"])

    def test_checksum_mismatch_is_refused(self):
        path = self.release / "dict__toy__abc.tar"
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaises(Exception) as ctx:
            G2P("abc", data_dir=self.data)
        self.assertIn("SHA-256", str(ctx.exception))
        self.assertFalse((self.data / "lexicons" / "toy" / "abc.tsv.gz").exists())

    def test_offline(self):
        os.environ["AGENTG2P_OFFLINE"] = "1"
        with self.assertRaises(Exception) as ctx:
            G2P("abc", data_dir=self.data)
        self.assertIn("AGENTG2P_OFFLINE", str(ctx.exception))

    def test_unsafe_archive_member_is_refused(self):
        evil = Path(self.tmp.name) / "evil.tar"
        with tarfile.open(evil, "w") as tar:
            info = tarfile.TarInfo("../outside.txt")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
        self.data.mkdir()
        with self.assertRaises(remote.DownloadError):
            remote._safe_extract(evil, self.data)
        self.assertFalse((Path(self.tmp.name) / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
