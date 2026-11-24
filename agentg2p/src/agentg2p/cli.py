# -*- coding: utf-8 -*-
"""Command line: agentg2p -l LANG [TEXT ...]  (reads stdin when no TEXT is given).

Examples:
  agentg2p -l de "Guten Morgen"
  agentg2p -l eng-us --words < words.txt          # word<TAB>phones<TAB>source per line
  agentg2p --list                                 # language tags
  agentg2p --list eng                             # dictionaries for one language
  agentg2p --download de fr                       # fetch data ahead of time (else on first use)
  agentg2p ui                                     # open the desktop window (--browser: in a browser)
"""

import argparse
import sys

from . import __version__, dictionaries, languages, remote
from .assets import DataNotFound, get_catalog
from .remote import DownloadError, download
from .g2p import G2P


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["ui"]:
        from .ui import main as ui_main
        return ui_main(argv[1:])
    p = argparse.ArgumentParser(prog="agentg2p", description="Grapheme-to-phoneme conversion.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("text", nargs="*", help="text to transcribe (default: read lines from stdin)")
    p.add_argument("-l", "--lang", help="language: ISO code (de, deu, eng-us) or English name")
    p.add_argument("-d", "--dictionary", help="dictionary id, e.g. wiktionary/deu_latn_broad")
    p.add_argument("-s", "--source", help="prefer a source: mfa, wiktionary, charsiu, or a source tag")
    p.add_argument("-b", "--backend", default="auto", help="auto, fst, byt5 or a model name")
    p.add_argument("--words", action="store_true", help="input lines are words; print word, phones, source")
    p.add_argument("--phone-sep", default="", help="between phones of a word (default: none)")
    p.add_argument("--no-lexicon", action="store_true", help="model only, skip the dictionary")
    p.add_argument("--list", nargs="?", const="", metavar="LANG",
                   help="list language tags, or the dictionaries for LANG")
    p.add_argument("--download", nargs="+", metavar="LANG",
                   help="download the default dictionary (and ByT5 model, if torch is installed) for LANG")
    p.add_argument("--download-all", action="store_true", help="download every dictionary (about 1.5 GB)")
    p.add_argument("--data-dir", help="data folder (default: $AGENTG2P_DATA, else ~/.cache/agentg2p)")
    p.add_argument("--version", action="version",
                   version=f"agentg2p {__version__} (data {remote.DATA_RELEASE})")
    args = p.parse_args(argv)

    try:
        if args.download or args.download_all:
            download(*(args.download or []), all_dictionaries=args.download_all, data_dir=args.data_dir)
            return 0
        if args.list is not None:
            if args.list:
                catalog = get_catalog(args.data_dir)
                for d in dictionaries(args.list, data_dir=args.data_dir):
                    print(d.summary() + ("  [downloaded]" if catalog.installed(d) else ""))
            else:
                for tag in languages(data_dir=args.data_dir):
                    print(tag)
            return 0
        if not args.lang and not args.dictionary:
            p.error("give --lang or --dictionary")
        g2p = G2P(args.lang, dictionary=args.dictionary, source=args.source, backend=args.backend,
                  use_lexicon=not args.no_lexicon, data_dir=args.data_dir)
    except (DataNotFound, DownloadError, KeyError, ValueError) as e:
        print(f"agentg2p: {e}", file=sys.stderr)
        return 2

    lines = [" ".join(args.text)] if args.text else (line.rstrip("\n") for line in sys.stdin)
    if args.words:
        words = [w.strip() for w in lines if w.strip()]
        for w, tok in zip(words, (g2p.tokens(w) for w in words)):
            word_toks = [t for t in tok if t.is_word]
            phones = " ".join(" ".join(t.phones or ["?"]) for t in word_toks)
            src = ",".join(sorted({t.source or "none" for t in word_toks}))
            print(f"{w}\t{phones}\t{src}")
        return 0
    for line in lines:
        print(g2p.transcribe(line, phone_sep=args.phone_sep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
