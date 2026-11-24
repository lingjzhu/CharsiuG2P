# -*- coding: utf-8 -*-
"""The G2P pipeline: dictionary lookup first, then a trained model.

For each word the dictionary is tried first (forgiving about case and
apostrophes). Words it lacks go to the dictionary's backend, which the build
chose per dictionary as the model with the lower test PER in the benchmarks:
the Phonetisaurus FST trained on that dictionary, or a ByT5 model (pooled or
per-source) prompted with the dictionary's language and source tags. If the
chosen backend is unavailable or gives no output, the other one is tried.
"""

import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from . import byt5 as byt5_mod
from .assets import Catalog, DictionaryInfo, get_catalog
from .fst import FSTModel
from .lexicon import Lexicon, nfc
from .text import Segmenter, Token

_lock = threading.Lock()


@lru_cache(maxsize=32)
def _load_fst(path: str) -> FSTModel:
    return FSTModel.load(path)


@lru_cache(maxsize=64)
def _load_lexicon(path: str) -> Lexicon:
    return Lexicon.load(path)


_byt5_cache: Dict[tuple, "byt5_mod.ByT5Model"] = {}


def _load_byt5(path: str, device: Optional[str]) -> "byt5_mod.ByT5Model":
    key = (path, device)
    with _lock:
        if key not in _byt5_cache:
            _byt5_cache[key] = byt5_mod.ByT5Model(path, device=device)
        return _byt5_cache[key]


class G2P:
    """Grapheme-to-phoneme conversion for one dictionary (language + convention).

    >>> g2p = G2P("deu")                    # or "de", "German", dictionary="mfa/german_mfa"
    >>> g2p.transcribe("Guten Morgen")      # IPA string, words separated by spaces
    >>> g2p("Guten Morgen")                 # list of Token(text, phones, source, ...)
    >>> g2p.word("Morgen")                  # phones of one word

    backend: "auto" (the build's choice for this dictionary), "fst", "byt5"
    (the build's ByT5 choice, else pooled), or a model name such as
    "byt5-pooled". ``use_lexicon=False`` skips the dictionary (model only).
    """

    def __init__(self, lang: Optional[str] = None, *, dictionary: Optional[str] = None,
                 source: Optional[str] = None, backend: str = "auto", use_lexicon: bool = True,
                 data_dir: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        self.catalog: Catalog = get_catalog(data_dir)
        self.info: DictionaryInfo = self.catalog.resolve(lang, dictionary, source)
        self.catalog.ensure_dictionary(self.info)  # downloads lexicon + FST on first use
        self.lexicon: Optional[Lexicon] = (
            _load_lexicon(str(self.catalog.path(self.info.lexicon))) if use_lexicon else None)
        self.segmenter = Segmenter(self.lexicon)
        self.device = device
        self._alphabet = frozenset(self.info.alphabet)
        self.backends = self._backend_order(backend)
        self._warned = set()

    def __repr__(self) -> str:
        return f"G2P({self.info.id!r}, lang={self.info.lang!r}, backends={self.backends})"

    # ---- backends -------------------------------------------------------------

    def _byt5_choice(self) -> str:
        if self.info.backend.startswith("byt5"):
            return self.info.backend
        scored = [k for k in self.info.scores if k.startswith("byt5") and k in self.catalog.models]
        if scored:
            return min(scored, key=lambda k: self.info.scores[k].get("per", 9e9))
        return "byt5-pooled"

    def _backend_order(self, backend: str) -> List[str]:
        if backend == "auto":
            first = self.info.backend
        elif backend == "byt5":
            first = self._byt5_choice()
        elif backend == "fst" or backend in self.catalog.models:
            first = backend
        else:
            raise ValueError(f"unknown backend {backend!r}: use auto, fst, byt5 or one of "
                             f"{sorted(self.catalog.models)}")
        second = self._byt5_choice() if first == "fst" else "fst"
        return [first, second]

    def _usable(self, name: str) -> bool:
        if name == "fst":
            return bool(self.info.fst) and self.catalog.path(self.info.fst).is_file()
        if name not in self.catalog.models:
            return False
        if not byt5_mod.available():
            if "torch" not in self._warned:
                self._warned.add("torch")
                warnings.warn("torch/transformers not installed; using the FST for words not in "
                              "the dictionary (pip install 'agentg2p[neural]' for ByT5)")
            return False
        return self.catalog.ensure_model(name)  # downloads the model on first use

    def _model_form(self, word: str) -> str:
        """The word as the models saw such words in training.

        Per character: kept if the dictionary uses it, else its lowercase (or
        uppercase) form if the dictionary uses that. So "Morgen" becomes
        "morgen" for a lowercase dictionary but stays "Morgen" for one with
        capitalized nouns, and case-less marks like Abkhaz Ӏ are left alone.
        """
        word = nfc(word)
        alphabet = self._alphabet
        if not alphabet:
            return word
        out = []
        for ch in word:
            if ch in alphabet or ch == " ":
                out.append(ch)
                continue
            low, up = ch.lower(), ch.upper()
            if all(c in alphabet for c in low):
                out.append(low)
            elif all(c in alphabet for c in up):
                out.append(up)
            else:
                out.append(ch)
        return nfc("".join(out))

    def _run_backend(self, name: str, forms: List[str]) -> List[Optional[List[str]]]:
        if name == "fst":
            fst = _load_fst(str(self.catalog.path(self.info.fst)))
            return [fst.decode(f) for f in forms]
        model = _load_byt5(str(self.catalog.path(self.catalog.models[name]["path"])), self.device)
        return model.predict([(self.info.lang, self.info.source_tag, f) for f in forms])

    def predict(self, words: Iterable[str]) -> List[Optional[List[str]]]:
        """Model output only (no dictionary), with the backend fallback order."""
        words = list(words)
        results: List[Optional[List[str]]] = [None] * len(words)
        sources: List[Optional[str]] = [None] * len(words)
        self._predict_into(words, results, sources)
        return results

    def _predict_into(self, words, results, sources) -> None:
        forms = [self._model_form(w) for w in words]
        todo = [k for k in range(len(words)) if forms[k]]
        for name in self.backends:
            # Checked lazily: a fallback model is only downloaded if some word needs it.
            if not todo:
                break
            if not self._usable(name):
                continue
            uniq = list(dict.fromkeys(forms[k] for k in todo))
            out = dict(zip(uniq, self._run_backend(name, uniq)))
            label = "fst" if name == "fst" else "byt5"
            for k in todo:
                if out.get(forms[k]):
                    results[k], sources[k] = out[forms[k]], label
            todo = [k for k in todo if results[k] is None]

    # ---- public API -------------------------------------------------------------

    def tokens(self, text: str) -> List[Token]:
        """Splits ``text`` and transcribes every word token."""
        tokens = self.segmenter.units(nfc(text))
        pending = []
        for tok in tokens:
            if not tok.is_word:
                continue
            variants = self.lexicon.get(tok.form) if (self.lexicon is not None and tok.form) else None
            if variants:
                tok.variants, tok.phones, tok.source = variants, variants[0], "lexicon"
            else:
                tok.form = None
                pending.append(tok)
        if pending:
            results: List[Optional[List[str]]] = [None] * len(pending)
            sources: List[Optional[str]] = [None] * len(pending)
            self._predict_into([t.text for t in pending], results, sources)
            for tok, phones, src in zip(pending, results, sources):
                tok.phones, tok.source = phones, src
                tok.form = self._model_form(tok.text)
                tok.variants = [phones] if phones else []
        return tokens

    __call__ = tokens

    def transcribe(self, text: str, phone_sep: str = "", word_sep: str = " ",
                   keep_punctuation: bool = False, unknown: str = "?") -> str:
        """The text's pronunciation as one string.

        Each word's phones are joined with ``phone_sep`` (default: none, so IPA
        reads naturally; use " " to keep phone boundaries), and words with
        ``word_sep``. Untranscribable words become ``unknown``.
        """
        parts: List[str] = []
        for tok in self.tokens(text):
            if tok.is_word:
                parts.append(phone_sep.join(tok.phones) if tok.phones else unknown)
            elif keep_punctuation and tok.text.strip():
                parts.append(tok.text.strip())
        return word_sep.join(parts)

    def word(self, word: str) -> Optional[List[str]]:
        """Phones for a single word (dictionary first, then the model)."""
        return self.batch([word])[0]

    def variants(self, word: str) -> List[List[str]]:
        """Every dictionary variant of ``word``, or the model's single guess."""
        if self.lexicon is not None:
            found = self.lexicon.lookup(word)
            if found:
                return found
        phones = self.predict([word])[0]
        return [phones] if phones else []

    def batch(self, words: Iterable[str]) -> List[Optional[List[str]]]:
        """Phones for many single words; model calls are batched."""
        words = list(words)
        results: List[Optional[List[str]]] = [None] * len(words)
        missing = []
        for k, w in enumerate(words):
            found = self.lexicon.lookup(w) if self.lexicon is not None else None
            if found:
                results[k] = found[0]
            else:
                missing.append(k)
        if missing:
            sub: List[Optional[List[str]]] = [None] * len(missing)
            srcs: List[Optional[str]] = [None] * len(missing)
            self._predict_into([words[k] for k in missing], sub, srcs)
            for k, phones in zip(missing, sub):
                results[k] = phones
        return results
