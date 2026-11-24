# -*- coding: utf-8 -*-
"""Finding the data bundle and choosing a dictionary for a language.

The data bundle is a folder with manifest.json, lexicons/, fst/ and byt5/. It
is looked for in, in order: the ``data_dir`` argument, $AGENTG2P_DATA and
~/.cache/agentg2p. If there is none, the manifest is downloaded from the GitHub
release (see remote.py), and dictionaries and models follow on first use.
"""

import json
import os
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

from . import remote
from .langcodes import MACRO_MEMBERS, name_of, normalize_code

ENV_VAR = "AGENTG2P_DATA"
# Source tags whose phones are not IPA, so they are never a language's default.
NON_IPA_SOURCES = {"mfa-arpa", "mfa-jamo", "mfa-pinyin", "mfa-prosodylab"}
_VARIETY_ALIASES = {"gb": "uk", "419": "la", "tw": "tw", "hk": "hk"}


class DataNotFound(FileNotFoundError):
    pass


def candidate_dirs(explicit: Optional[Union[str, Path]] = None) -> List[Path]:
    dirs = []
    if explicit:
        dirs.append(Path(explicit))
    if os.environ.get(ENV_VAR):
        dirs.append(Path(os.environ[ENV_VAR]))
    dirs.append(Path.home() / ".cache" / "agentg2p")
    return dirs


def find_data_dir(explicit: Optional[Union[str, Path]] = None) -> Path:
    tried = candidate_dirs(explicit)
    for d in tried:
        if (d / "manifest.json").is_file():
            return d
        if explicit:
            break
    target = Path(explicit) if explicit else remote.default_data_dir()
    if not remote.offline():
        try:
            return remote.fetch_manifest(target).parent
        except remote.DownloadError as e:
            raise DataNotFound(f"agentg2p data not found and the download failed: {e}") from e
    raise DataNotFound(
        "agentg2p data not found (looked for manifest.json in: " + ", ".join(str(d) for d in tried)
        + "). Unset AGENTG2P_OFFLINE to download it, or point $AGENTG2P_DATA at a data folder.")


@dataclass
class DictionaryInfo:
    id: str
    source: str
    name: str
    lang: str
    source_tag: str
    words: int
    lexicon: str
    backend: str
    alphabet: str = ""
    fst: Optional[str] = None
    asset: Optional[str] = None
    scores: Dict[str, dict] = field(default_factory=dict)
    fst_train_args: List[str] = field(default_factory=list)
    fst_coverage: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def language(self) -> str:
        return name_of(self.lang)

    def summary(self) -> str:
        best = self.scores.get(self.backend, {})
        per = f"{best['per']:.2%}" if "per" in best else "n/a"
        return (f"{self.id} [{self.lang}, {self.source_tag}] {self.words} words, "
                f"backend {self.backend} (test PER {per})")


class Catalog:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "manifest.json", encoding="utf-8") as f:
            self.manifest = json.load(f)
        known = set(DictionaryInfo.__dataclass_fields__)
        self.dictionaries: List[DictionaryInfo] = [
            DictionaryInfo(**{k: v for k, v in d.items() if k in known})
            for d in self.manifest["dictionaries"]]
        self.by_id = {d.id: d for d in self.dictionaries}
        self.by_name = {d.name: d for d in self.dictionaries}
        self.models: Dict[str, dict] = self.manifest.get("models", {})

    def path(self, rel: str) -> Path:
        return self.data_dir / rel

    # ---- installed data -------------------------------------------------------

    def installed(self, d: DictionaryInfo) -> bool:
        return self.path(d.lexicon).is_file() and (not d.fst or self.path(d.fst).is_file())

    def model_installed(self, name: str) -> bool:
        return (self.path(self.models[name]["path"]) / "model.safetensors").is_file()

    def ensure_dictionary(self, d: DictionaryInfo) -> None:
        """Downloads the dictionary's lexicon and FST if they are missing."""
        if self.installed(d):
            return
        if not d.asset:
            raise DataNotFound(f"{d.id}: files missing from {self.data_dir} and no download is listed")
        try:
            remote.fetch_asset(d.asset, self.data_dir, self.manifest)
        except remote.DownloadError as e:
            raise DataNotFound(f"{d.id}: {e}") from e

    def ensure_model(self, name: str) -> bool:
        """Downloads a ByT5 model if missing; False if it can't be had."""
        if self.model_installed(name):
            return True
        asset = self.models[name].get("asset")
        if not asset or remote.offline():
            return False
        try:
            remote.fetch_asset(asset, self.data_dir, self.manifest)
        except remote.DownloadError as e:
            warnings.warn(f"could not download ByT5 model {name}: {e}")
            return False
        return True

    def languages(self) -> List[str]:
        return sorted({d.lang for d in self.dictionaries})

    def _by_tag(self, tag: str) -> List[DictionaryInfo]:
        exact = [d for d in self.dictionaries if d.lang == tag]
        if exact:
            return exact
        return [d for d in self.dictionaries if d.lang.startswith(tag + "-")]

    def candidates(self, lang: str) -> List[DictionaryInfo]:
        """Dictionaries for a language name or code, best default first."""
        tag = normalize_code(lang)
        base, _, variety = tag.partition("-")
        if variety:
            tag = f"{base}-{_VARIETY_ALIASES.get(variety, variety)}"
        found = self._by_tag(tag)
        if not found and base in MACRO_MEMBERS:
            for member in MACRO_MEMBERS[base]:
                found = self._by_tag(f"{member}-{variety}" if variety else member)
                if found:
                    break
        if not found and variety:
            found = self.candidates(base)
        return sorted(found, key=self._rank)

    @staticmethod
    def _rank(d: DictionaryInfo):
        return (d.name.endswith("_filtered"), d.source_tag in NON_IPA_SOURCES,
                "inventory" in " ".join(d.notes), -d.words, d.id)

    def resolve(self, lang: Optional[str] = None, dictionary: Optional[str] = None,
                source: Optional[str] = None) -> DictionaryInfo:
        if dictionary:
            d = self.by_id.get(dictionary) or self.by_name.get(dictionary)
            if d is None:
                raise KeyError(f"unknown dictionary {dictionary!r}; see agentg2p.dictionaries()")
            return d
        if not lang:
            raise ValueError("give a language (e.g. 'deu', 'de' or 'German') or a dictionary id")
        if lang in self.by_id or lang in self.by_name:
            return self.by_id.get(lang) or self.by_name[lang]
        found = self.candidates(lang)
        if source:
            found = [d for d in found if source in (d.source, d.source_tag)]
        if not found:
            raise KeyError(f"no dictionary for language {lang!r}"
                           + (f" with source {source!r}" if source else "")
                           + "; see agentg2p.languages()")
        return found[0]


@lru_cache(maxsize=8)
def _catalog(path: str) -> Catalog:
    return Catalog(Path(path))


def get_catalog(data_dir: Optional[Union[str, Path]] = None) -> Catalog:
    return _catalog(str(find_data_dir(data_dir).resolve()))
