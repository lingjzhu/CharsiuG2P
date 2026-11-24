# -*- coding: utf-8 -*-
"""Splitting running text into the units a pronunciation dictionary knows.

1. ``split_spans``: text -> word spans and the rest (spaces, punctuation, digits).
   A word is a run of letters and combining marks, plus apostrophes, hyphens,
   middle dots, ZWJ/ZWNJ and Hebrew geresh inside it.
2. ``Segmenter.units``: word spans -> lookup units, using the lexicon:
   - several words that form one dictionary entry ("ice cream", Vietnamese
     syllables) are merged, longest match first;
   - a span with an edge apostrophe or hyphen keeps it only if the dictionary
     has that form ('cause), otherwise the mark becomes punctuation;
   - an unknown hyphenated compound is split at its hyphens;
   - scripts written without spaces (Han, kana, Thai, Lao, Khmer, Myanmar) are
     segmented into dictionary words by dynamic programming: the fewest
     characters left unmatched, then the fewest pieces. Runs of unmatched
     characters stay together and go to the model.
"""

import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .lexicon import APOSTROPHES, Lexicon

HYPHENS = "-‐‑"
JOINERS = APOSTROPHES + HYPHENS + "·׳״‌‍"
MAX_SEGMENT_CHARS = 16

_UNSEGMENTED_RANGES = (
    (0x0E00, 0x0EFF),    # Thai, Lao
    (0x1000, 0x109F),    # Myanmar
    (0x1780, 0x17FF),    # Khmer
    (0x19E0, 0x19FF),    # Khmer symbols
    (0x3040, 0x30FF),    # Hiragana, Katakana
    (0x31F0, 0x31FF),    # Katakana phonetic extensions
    (0x3400, 0x4DBF),    # CJK extension A
    (0x4E00, 0x9FFF),    # CJK unified ideographs
    (0xF900, 0xFAFF),    # CJK compatibility ideographs
    (0xFF66, 0xFF9F),    # half-width katakana
    (0x20000, 0x323AF),  # CJK extensions B-H
)


def is_unsegmented(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _UNSEGMENTED_RANGES)


def is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LM"


@dataclass
class Token:
    """One piece of the input text.

    ``phones`` is None for spaces, punctuation and digits, and for words no
    backend could transcribe. ``source`` says where the phones came from:
    "lexicon", "fst" or "byt5". ``form`` is the dictionary entry used, or the
    string sent to the model.
    """
    text: str
    is_word: bool
    phones: Optional[List[str]] = None
    source: Optional[str] = None
    form: Optional[str] = None
    variants: List[List[str]] = field(default_factory=list)


def split_spans(text: str) -> List[Tuple[str, bool]]:
    """(substring, is_word) pieces that concatenate back to ``text``."""
    spans: List[Tuple[str, bool]] = []
    i, n = 0, len(text)
    while i < n:
        if is_word_char(text[i]) or (text[i] in JOINERS and i + 1 < n and is_word_char(text[i + 1])):
            j = i + 1
            while j < n and (is_word_char(text[j]) or (text[j] in JOINERS and j + 1 < n
                                                        and (is_word_char(text[j + 1]) or text[j + 1] in JOINERS))):
                j += 1
            # A trailing joiner (dogs') is kept for now; Segmenter decides.
            if j < n and text[j] in APOSTROPHES:
                j += 1
            spans.append((text[i:j], True))
            i = j
        else:
            j = i + 1
            while j < n and not is_word_char(text[j]) and not (
                    text[j] in JOINERS and j + 1 < n and is_word_char(text[j + 1])):
                j += 1
            spans.append((text[i:j], False))
            i = j
    return spans


class Segmenter:
    def __init__(self, lexicon: Optional[Lexicon]):
        self.lexicon = lexicon

    def _known(self, s: str) -> bool:
        return self.lexicon is not None and self.lexicon.find(s) is not None

    def units(self, text: str) -> List[Token]:
        """Tokens for ``text``; word tokens still need phones."""
        spans = split_spans(text)
        tokens: List[Token] = []
        i = 0
        while i < len(spans):
            piece, is_word = spans[i]
            if not is_word:
                tokens.append(Token(piece, False))
                i += 1
                continue
            merged = self._merge_words(spans, i)
            if merged is not None:
                form, end = merged
                tokens.append(Token("".join(s for s, _ in spans[i:end]), True, form=form))
                i = end
                continue
            tokens.extend(self._split_word(piece))
            i += 1
        return tokens

    def _merge_words(self, spans, i) -> Optional[Tuple[str, int]]:
        """Longest run of words, separated by single spaces, that is one dictionary entry."""
        if self.lexicon is None or self.lexicon.max_words < 2:
            return None
        words = [spans[i][0]]
        best = None
        k = i + 1
        while k + 1 < len(spans) and len(words) < self.lexicon.max_words:
            sep, sep_is_word = spans[k]
            nxt, nxt_is_word = spans[k + 1]
            if sep_is_word or not nxt_is_word or sep.strip() or len(sep) != 1 or "\n" in sep:
                break
            words.append(nxt)
            k += 2
            form = self.lexicon.find(" ".join(words))
            if form is not None:
                best = (form, k)
        return best

    def _split_word(self, span: str) -> List[Token]:
        if self._known(span):
            return [Token(span, True, form=self.lexicon.find(span))]
        # Edge joiners only stay if the dictionary has that form.
        lead = len(span) - len(span.lstrip(JOINERS))
        trail = len(span) - len(span.rstrip(JOINERS))
        if lead or trail:
            for a, b in ((lead, 0), (0, trail), (lead, trail)):
                core = span[a:len(span) - b]
                if core and (a or b) and self._known(core):
                    out = [Token(span[:a], False)] if a else []
                    out.append(Token(core, True, form=self.lexicon.find(core)))
                    if b:
                        out.append(Token(span[len(span) - b:], False))
                    return out
            core = span[lead:len(span) - trail]
            out = [Token(span[:lead], False)] if lead else []
            out.extend(self._split_word(core) if core else [])
            if trail:
                out.append(Token(span[len(span) - trail:], False))
            return out
        if any(h in span for h in HYPHENS):
            out: List[Token] = []
            part = ""
            for ch in span:
                if ch in HYPHENS:
                    if part:
                        out.extend(self._split_word(part))
                    out.append(Token(ch, False))
                    part = ""
                else:
                    part += ch
            if part:
                out.extend(self._split_word(part))
            return out
        if self.lexicon is not None and any(is_unsegmented(c) for c in span):
            return [Token(p, True, form=self.lexicon.find(p)) for p in self.segment(span)]
        return [Token(span, True)]

    def segment(self, span: str) -> List[str]:
        """Dictionary-based segmentation: fewest unmatched characters, then fewest pieces."""
        n = len(span)
        max_len = min(MAX_SEGMENT_CHARS, self.lexicon.max_chars if self.lexicon else 1)
        inf = (n + 1, n + 1)
        best = [inf] * (n + 1)
        back: List[Optional[Tuple[int, bool]]] = [None] * (n + 1)
        best[0] = (0, 0)
        for i in range(n):
            if best[i] == inf:
                continue
            unknown, pieces = best[i]
            cand = (unknown + 1, pieces + 1)
            if cand < best[i + 1]:
                best[i + 1], back[i + 1] = cand, (i, False)
            for j in range(i + 1, min(n, i + max_len) + 1):
                if self._known(span[i:j]):
                    cand = (unknown, pieces + 1)
                    if cand < best[j]:
                        best[j], back[j] = cand, (i, True)
        pieces_rev: List[Tuple[str, bool]] = []
        j = n
        while j > 0:
            i, known = back[j]
            pieces_rev.append((span[i:j], known))
            j = i
        out: List[str] = []
        pending = ""
        for piece, known in reversed(pieces_rev):
            if known:
                if pending:
                    out.append(pending)
                    pending = ""
                out.append(piece)
            else:
                pending += piece
        if pending:
            out.append(pending)
        return out
