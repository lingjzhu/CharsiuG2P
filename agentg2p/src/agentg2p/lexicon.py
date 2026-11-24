# -*- coding: utf-8 -*-
"""Pronunciation dictionaries: parsing the raw TSVs and looking words up.

Raw dictionaries (MFA_PD, Wiktionary_PD, Charsiu_PD) have one ``word<TAB>pron``
line per word. Variants are separated by commas, and MFA prefixes each variant
with four probabilities. ``parse_line`` turns such a line into the word and its
variants. The benchmarks score the first variant, so it is the primary one.

The bundled lexicon files written by the build use one line per word:
``word<TAB>variant1<TAB>variant2...``, with space-separated phones.
"""

import gzip
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

# MFA's leading pronunciation and silence probabilities (same pattern as
# test/benchmark_fst_all.py:load_and_split).
_MFA_SCORES = re.compile(r"^(\d+(?:\.\d+)?\s+){4}")
# MFA's markup entries such as [bracketed], [laughter], <unk>, <eps>, <cutoff>.
_JUNK_WORD = re.compile(r"^(\[.*\]|<.*>)$")
# MFA's spoken-noise and silence "pronunciations".
_JUNK_PRONS = {("spn",), ("sil",), ("sp",)}

APOSTROPHES = "'’ʼ‘"


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def parse_line(line: str) -> Optional[Tuple[str, List[List[str]]]]:
    """Returns (word, variants) for a raw dictionary line, or None to skip it.

    The word and every phone are NFC-normalized. Variants keep their order and
    duplicates are removed.
    """
    parts = line.strip().split("\t")
    if len(parts) != 2:
        return None
    word = nfc(parts[0].strip())
    if not word or _JUNK_WORD.match(word):
        return None
    variants: List[List[str]] = []
    for chunk in parts[1].split(","):
        chunk = _MFA_SCORES.sub("", chunk.strip())
        phones = [nfc(p) for p in chunk.split()]
        if phones and tuple(phones) not in _JUNK_PRONS and phones not in variants:
            variants.append(phones)
    if not variants:
        return None
    return word, variants


def read_raw_dictionary(path: Union[str, Path]) -> Dict[str, List[List[str]]]:
    """Parses a raw dictionary TSV into word -> variants (words in file order)."""
    entries: Dict[str, List[List[str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            word, variants = parsed
            known = entries.setdefault(word, [])
            for v in variants:
                if v not in known:
                    known.append(v)
    return entries


def write_lexicon(entries: Dict[str, List[List[str]]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for word, variants in entries.items():
            f.write(word + "\t" + "\t".join(" ".join(v) for v in variants) + "\n")
    tmp.replace(path)


def _case_candidates(word: str) -> Iterator[str]:
    yield word
    lower = word.lower()
    yield lower
    if word:
        yield word[0].upper() + word[1:].lower()
    yield word.casefold()


class Lexicon:
    """word -> pronunciation variants, with forgiving lookup.

    ``lookup`` tries the word as given, then lowercase, capitalized and
    casefolded forms, then the same with typographic apostrophes replaced by
    the one the dictionary uses.
    """

    def __init__(self, entries: Dict[str, str]):
        # Values are the tab-joined variants, split on lookup to save memory.
        self._entries = entries
        self.max_words = 1   # most space-separated tokens in one entry
        self.max_chars = max(map(len, entries), default=1)  # longest entry, in characters
        counts = dict.fromkeys(APOSTROPHES, 0)
        for word in entries:
            if " " in word:
                self.max_words = max(self.max_words, word.count(" ") + 1)
            if not word.isalpha():
                for a in APOSTROPHES:
                    if a in word:
                        counts[a] += 1
        # The apostrophe the dictionary uses most often; typographic ones in input map to it.
        self._apostrophe = max(counts, key=counts.get) if any(counts.values()) else None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Lexicon":
        entries: Dict[str, str] = {}
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                word, _, rest = line.rstrip("\n").partition("\t")
                if word and rest:
                    entries[word] = rest
        return cls(entries)

    @classmethod
    def from_dict(cls, entries: Dict[str, Iterable[Iterable[str]]]) -> "Lexicon":
        return cls({w: "\t".join(" ".join(v) for v in vs) for w, vs in entries.items()})

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, word: str) -> bool:
        return word in self._entries

    def words(self) -> Iterator[str]:
        return iter(self._entries)

    def get(self, word: str) -> Optional[List[List[str]]]:
        """Exact lookup only."""
        raw = self._entries.get(word)
        if raw is None:
            return None
        return [v.split(" ") for v in raw.split("\t")]

    def find(self, word: str) -> Optional[str]:
        """The dictionary form that ``lookup`` would use for ``word``, if any."""
        word = nfc(word)
        for cand in _case_candidates(word):
            if cand in self._entries:
                return cand
        if self._apostrophe is not None:
            fixed = "".join(self._apostrophe if c in APOSTROPHES else c for c in word)
            if fixed != word:
                for cand in _case_candidates(fixed):
                    if cand in self._entries:
                        return cand
        return None

    def lookup(self, word: str) -> Optional[List[List[str]]]:
        """All variants for ``word`` (first = primary), or None if absent."""
        form = self.find(word)
        return None if form is None else self.get(form)

    def lowercase_share(self) -> float:
        """Share of entries with letters whose form equals its lowercase form."""
        cased = same = 0
        for word in self._entries:
            if word.lower() != word.upper():
                cased += 1
                same += word == word.lower()
        return same / cased if cased else 1.0
