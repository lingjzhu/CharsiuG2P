# -*- coding: utf-8 -*-
"""The local server behind the agentg2p window: static files plus a small JSON API.

It listens on 127.0.0.1 only. Every /api call must carry the per-run token that
the server writes into index.html, and a Host header naming this server, so
neither other local programs nor web pages open in a browser can drive it.
"""

import json
import os
import secrets
import threading
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

from .. import __version__, remote
from .. import byt5 as byt5_mod
from ..assets import Catalog, DataNotFound, DictionaryInfo, get_catalog
from ..g2p import G2P
from ..langcodes import CODE_NAMES, ISO1_TO_3
from .langnames import REF_NAMES

STATIC = Path(__file__).parent / "static"
PREFS_PATH = Path(os.environ.get("AGENTG2P_UI_PREFS", Path.home() / ".config" / "agentg2p" / "ui.json"))

NOTATIONS = {"mfa": "MFA phone set", "mfa-cv": "MFA phone set", "mfa-arpa": "ARPAbet",
             "mfa-jamo": "Hangul jamo", "mfa-pinyin": "Pinyin", "mfa-prosodylab": "Prosodylab",
             "wikipron-broad": "WikiPron broad IPA", "wikipron-narrow": "WikiPron narrow IPA"}
VARIETIES = {"us": "US", "uk": "UK", "in": "India", "ng": "Nigeria", "nonnative": "non-native",
             "tw": "Taiwan", "cn": "China", "erhua": "Erhua", "br": "Brazil", "pt": "Portugal",
             "es": "Spain", "la": "Latin America", "mx": "Mexico", "qc": "Quebec",
             "hokkien": "Hokkien", "teochew": "Teochew", "dhaka": "Dhaka", "rarh": "Rarh",
             "nw": "North Wales", "sw": "South Wales", "e": "Eastern", "w": "Western",
             "clas": "Classical", "eccl": "Ecclesiastical", "ara": "Arabic script",
             "c": "Central", "n": "Northern", "s": "Southern"}
ISO1 = {three: two for two, three in ISO1_TO_3.items()}
SOURCES = {"mfa": "MFA", "wiktionary": "WikiPron", "charsiu": "CharsiuG2P"}
MAX_TEXT = 200_000
MAX_WORDS = 100_000


def language_name(tag: str) -> str:
    base, _, variety = tag.partition("-")
    name = CODE_NAMES.get(base) or REF_NAMES.get(base) or base
    return f"{name} ({VARIETIES.get(variety, variety)})" if variety else name


def notation(d: DictionaryInfo) -> str:
    if d.source_tag.startswith("charsiu"):
        return "CharsiuG2P IPA"
    return NOTATIONS.get(d.source_tag, d.source_tag)


def model_label(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return "FST" if name == "fst" else "ByT5"


class UserError(Exception):
    """A problem the person can act on: a plain message, plus technical detail."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


class Api:
    """Every method takes and returns JSON-able values; the handler dispatches to them."""

    PUBLIC = ("info", "languages", "dictionaries", "transcribe", "words", "compare", "download",
              "prefs", "save_prefs", "save")

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.window = None          # set by the launcher when a native window exists
        self._g2p: Dict[tuple, G2P] = {}
        self._lock = threading.RLock()

    # ---- helpers -----------------------------------------------------------------

    def _catalog(self) -> Catalog:
        try:
            return get_catalog(self.data_dir)
        except DataNotFound as e:
            raise UserError("The dictionary list could not be loaded. agentg2p downloads it from "
                            "GitHub on first use, so check your connection and try again.", str(e)) from e

    def _info(self, d: DictionaryInfo) -> dict:
        catalog = self._catalog()
        score = d.scores.get(d.backend, {})
        size = catalog.manifest.get("release", {}).get("assets", {}).get(d.asset or "", {}).get("size")
        return {"id": d.id, "name": d.name, "lang": d.lang, "language": language_name(d.lang),
                "source": SOURCES.get(d.source, d.source), "notation": notation(d),
                "words": d.words, "model": model_label(d.backend), "per": score.get("per"),
                "n_test": score.get("n_test"), "installed": catalog.installed(d), "size": size}

    def _resolve(self, dictionary: str) -> DictionaryInfo:
        d = self._catalog().by_id.get(dictionary)
        if d is None:
            raise UserError(f"There is no dictionary called {dictionary}.")
        return d

    def _get(self, dictionary: str, backend: str = "auto", use_lexicon: bool = True) -> G2P:
        key = (dictionary, backend, bool(use_lexicon))
        if key not in self._g2p:
            try:
                self._g2p[key] = G2P(dictionary=dictionary, backend=backend,
                                     use_lexicon=bool(use_lexicon), data_dir=self.data_dir)
            except DataNotFound as e:
                raise self._download_error(self._resolve(dictionary), e) from e
        return self._g2p[key]

    @staticmethod
    def _download_error(d: DictionaryInfo, e: Exception) -> UserError:
        if not d.asset:
            return UserError(f"{d.name} is not available to download, so it can only be used once "
                             "its files are in the data folder.", str(e))
        if remote.offline():
            return UserError(f"{d.name} is not downloaded yet, and downloads are turned off "
                             "(AGENTG2P_OFFLINE is set).", str(e))
        return UserError(f"{d.name} could not be downloaded. Check your internet connection and "
                         "try again.", str(e))

    @staticmethod
    def _flag(g2p: G2P, text: str, source: Optional[str], phones) -> Optional[str]:
        if not phones:
            return "No transcription: no model gave an output for this word"
        if source == "lexicon":
            return None
        form = g2p._model_form(text)
        unknown = list(dict.fromkeys(c for c in form if not c.isspace() and c not in g2p._alphabet))
        if not unknown or not g2p._alphabet:
            return None
        letters = " ".join(unknown)
        if source == "fst":
            return f"{letters} dropped: not in this dictionary's alphabet"
        return f"{letters} never appear in this dictionary, so this guess is less reliable"

    def _notes(self, g2p: G2P, backend: str) -> List[str]:
        wanted = g2p.backends[0]
        if wanted != "fst" and not byt5_mod.available():
            return ["ByT5 is not installed, so the FST transcribes words that are not in the "
                    "dictionary. Install it with: pip install 'agentg2p[neural]'"]
        return []

    def _row(self, g2p: G2P, text: str, phones, variants, source) -> dict:
        return {"kind": "word", "text": text, "phones": phones or None, "variants": variants or [],
                "source": source, "flag": self._flag(g2p, text, source, phones)}

    # ---- API ---------------------------------------------------------------------

    def info(self) -> dict:
        return {"version": __version__, "data_release": remote.DATA_RELEASE,
                "byt5": byt5_mod.available(), "window": self.window is not None}

    def languages(self) -> List[dict]:
        catalog = self._catalog()
        counts: Dict[str, int] = {}
        for d in catalog.dictionaries:
            counts[d.lang] = counts.get(d.lang, 0) + 1
        out = [{"tag": t, "name": language_name(t), "count": n, "alias": ISO1.get(t.partition("-")[0])}
               for t, n in counts.items()]
        return sorted(out, key=lambda r: (r["name"] == r["tag"], r["name"].casefold(), r["tag"]))

    def dictionaries(self, lang: Optional[str] = None) -> List[dict]:
        catalog = self._catalog()
        if lang:
            found = catalog.candidates(lang)
            return [dict(self._info(d), default=(k == 0)) for k, d in enumerate(found)]
        defaults = {catalog.candidates(t)[0].id for t in catalog.languages()}
        return [dict(self._info(d), default=d.id in defaults) for d in catalog.dictionaries]

    def transcribe(self, dictionary: str, text: str, backend: str = "auto",
                   use_lexicon: bool = True) -> dict:
        if len(text) > MAX_TEXT:
            raise UserError(f"The text is longer than {MAX_TEXT:,} characters. "
                            "Use the word list for long inputs.")
        g2p = self._get(dictionary, backend, use_lexicon)
        rows = []
        for tok in g2p.tokens(text):
            if tok.is_word:
                rows.append(self._row(g2p, tok.text, tok.phones, tok.variants, tok.source))
            elif any(unicodedata.category(c) == "Nd" for c in tok.text):
                for num in "".join(c if unicodedata.category(c) == "Nd" else " "
                                   for c in tok.text).split():
                    rows.append({"kind": "number", "text": num, "phones": None, "variants": [],
                                 "source": None, "flag": "Numbers are not read out"})
        return {"dictionary": self._info(g2p.info), "rows": rows, "notes": self._notes(g2p, backend)}

    def words(self, dictionary: str, words: List[str], backend: str = "auto",
              use_lexicon: bool = True) -> dict:
        if len(words) > MAX_WORDS:
            raise UserError(f"The list has more than {MAX_WORDS:,} words. Split it into smaller files.")
        g2p = self._get(dictionary, backend, use_lexicon)
        found = [g2p.lexicon.lookup(w) if g2p.lexicon is not None else None for w in words]
        missing = [k for k, v in enumerate(found) if not v]
        results: List = [None] * len(missing)
        sources: List = [None] * len(missing)
        if missing:
            g2p._predict_into([words[k] for k in missing], results, sources)
        predicted = {k: (p, s) for k, p, s in zip(missing, results, sources)}
        rows = []
        for k, w in enumerate(words):
            if found[k]:
                rows.append(self._row(g2p, w, found[k][0], found[k], "lexicon"))
            else:
                phones, src = predicted[k]
                rows.append(self._row(g2p, w, phones, [phones] if phones else [], src))
        return {"dictionary": self._info(g2p.info), "rows": rows, "notes": self._notes(g2p, backend)}

    def compare(self, lang: str, word: str) -> dict:
        word = word.strip()
        catalog = self._catalog()
        rows = []
        for d in catalog.candidates(lang):
            row = {"dictionary": self._info(d), "variants": [], "source": None, "flag": None}
            if row["dictionary"]["installed"] and word:
                g2p = self._get(d.id)
                variants = g2p.lexicon.lookup(word) if g2p.lexicon is not None else None
                if variants:
                    row.update(variants=variants, source="lexicon")
                else:
                    res, src = [None], [None]
                    g2p._predict_into([word], res, src)
                    row.update(variants=[res[0]] if res[0] else [], source=src[0],
                               flag=self._flag(g2p, word, src[0], res[0]))
            rows.append(row)
        return {"word": word, "language": language_name(normalize(catalog, lang)), "rows": rows}

    def download(self, dictionary: str) -> dict:
        d = self._resolve(dictionary)
        try:
            self._catalog().ensure_dictionary(d)
        except DataNotFound as e:
            raise self._download_error(d, e) from e
        return self._info(d)

    def prefs(self) -> dict:
        try:
            return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_prefs(self, prefs: dict) -> dict:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PREFS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(prefs, ensure_ascii=False), encoding="utf-8")
        tmp.replace(PREFS_PATH)
        return {"ok": True}

    def save(self, name: str, content: str) -> dict:
        """Asks where to save an export (native window); the browser downloads it itself."""
        if self.window is None:
            return {"download": True}
        import webview
        kind = getattr(getattr(webview, "FileDialog", None), "SAVE", None) or webview.SAVE_DIALOG
        chosen = self.window.create_file_dialog(kind, save_filename=name)
        if not chosen:
            return {"cancelled": True}
        path = Path(chosen if isinstance(chosen, str) else chosen[0])
        path.write_text(content, encoding="utf-8")
        return {"saved": str(path)}


def normalize(catalog: Catalog, lang: str) -> str:
    found = catalog.candidates(lang)
    return found[0].lang if found else lang


class Handler(BaseHTTPRequestHandler):
    server: "UIServer"

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._send(403, b"forbidden", "text/plain")
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_text(encoding="utf-8").replace(
                "{{TOKEN}}", self.server.token)
            csp = ("default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
                   "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'")
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8",
                              {"Content-Security-Policy": csp})
        target = (STATIC / path.lstrip("/")).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".woff2": "font/woff2", ".svg": "image/svg+xml",
                 ".txt": "text/plain; charset=utf-8"}.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), ctype)

    def do_POST(self) -> None:
        if not self._host_ok() or not secrets.compare_digest(
                self.headers.get("X-Agentg2p-Token", ""), self.server.token):
            return self._send(403, b'{"error": "forbidden"}', "application/json")
        name = self.path.split("?", 1)[0].removeprefix("/api/")
        if name not in Api.PUBLIC:
            return self._send(404, b'{"error": "unknown call"}', "application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            args = json.loads(self.rfile.read(length) or b"{}")
            with self.server.api._lock:
                result = getattr(self.server.api, name)(**args)
            code, body = 200, {"result": result}
        except UserError as e:
            code, body = 400, {"error": str(e), "detail": e.detail}
        except (TypeError, ValueError, KeyError) as e:
            code, body = 400, {"error": f"Bad request: {e}"}
        except Exception as e:  # shown in the window instead of a silent failure
            code, body = 500, {"error": f"Something went wrong: {type(e).__name__}: {e}"}
        self._send(code, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, api: Api, port: int = 0):
        super().__init__(("127.0.0.1", port), Handler)
        self.api = api
        self.token = secrets.token_urlsafe(24)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"
