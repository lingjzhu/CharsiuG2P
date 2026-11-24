# -*- coding: utf-8 -*-
"""agentg2p: grapheme-to-phoneme conversion for 700+ pronunciation dictionaries.

Words are looked up in the dictionary first. Words not in it are transcribed by
the model that scored better on that dictionary's held-out test set: a
Phonetisaurus FST (decoded in pure Python + numpy) or a ByT5 model (needs the
optional torch + transformers). Data is downloaded from the GitHub release on
first use (or ahead of time with ``agentg2p.download("de")``).

    import agentg2p
    agentg2p.transcribe("Guten Morgen", "de")       # 'ɡuːtən mɔʁɡən' (example)
    g2p = agentg2p.G2P("German")
    g2p("Guten Morgen")                              # tokens with phones and source
    agentg2p.languages()                             # language tags with a dictionary
    agentg2p.dictionaries("eng")                     # every English dictionary
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

from .assets import DataNotFound, DictionaryInfo, find_data_dir, get_catalog
from .remote import download
from .fst import FSTModel
from .g2p import G2P
from .lexicon import Lexicon
from .text import Token

__version__ = "0.1.2"
__all__ = ["G2P", "Token", "DictionaryInfo", "FSTModel", "Lexicon", "DataNotFound", "load",
           "transcribe", "languages", "dictionaries", "find_data_dir", "download"]


@lru_cache(maxsize=16)
def _cached(lang, dictionary, source, backend, data_dir) -> G2P:
    return G2P(lang, dictionary=dictionary, source=source, backend=backend, data_dir=data_dir)


def load(lang: Optional[str] = None, *, dictionary: Optional[str] = None, source: Optional[str] = None,
         backend: str = "auto", data_dir: Optional[Union[str, Path]] = None) -> G2P:
    """A cached G2P for a language (code or English name) or a dictionary id."""
    return _cached(lang, dictionary, source, backend, str(data_dir) if data_dir else None)


def transcribe(text: str, lang: Optional[str] = None, **kwargs) -> str:
    """One-call transcription; keyword arguments go to ``load`` and ``G2P.transcribe``."""
    load_keys = {"dictionary", "source", "backend", "data_dir"}
    g2p = load(lang, **{k: v for k, v in kwargs.items() if k in load_keys})
    return g2p.transcribe(text, **{k: v for k, v in kwargs.items() if k not in load_keys})


def languages(data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Language tags (ISO 639-3, sometimes with a variety) that have a dictionary."""
    return get_catalog(data_dir).languages()


def dictionaries(lang: Optional[str] = None,
                 data_dir: Optional[Union[str, Path]] = None) -> List[DictionaryInfo]:
    """Every dictionary, or those for one language (the default first)."""
    catalog = get_catalog(data_dir)
    return catalog.candidates(lang) if lang else list(catalog.dictionaries)
