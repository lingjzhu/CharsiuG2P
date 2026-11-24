# agentg2p

Grapheme-to-phoneme (G2P) conversion for about 350 languages: give it text, get
pronunciations, in IPA for most dictionaries.

```python
import agentg2p

agentg2p.transcribe("Guten Morgen", "de")
# 'ɡuːtʰn̩ mɔʁŋ'
```

For every word, agentg2p:

1. **looks it up** in a pronunciation dictionary (717 dictionaries from MFA, WikiPron
   and CharsiuG2P);
2. if the word isn't there, **predicts it** with the model that was more accurate for
   that dictionary: a Phonetisaurus FST trained on it, or a ByT5 neural model.

## Install

```bash
pip install git+https://github.com/extreme-coder/agentg2p-lib
```

This needs only numpy, and handles dictionary lookup plus the FST models.

To also use the ByT5 models (better for some languages, such as Korean), install the
`neural` extra, which adds torch and transformers:

```bash
pip install "agentg2p[neural] @ git+https://github.com/extreme-coder/agentg2p-lib"
```

Without the extra, words that are not in the dictionary always go to the FST.

## Use

```python
import agentg2p

g2p = agentg2p.G2P("German")                 # language: code ("de", "deu") or English name
for tok in g2p("Guten Morgen, Welt!"):
    if tok.is_word:
        print(tok.text, tok.phones, tok.source)   # source: lexicon | fst | byt5

g2p.word("Morgen")                           # ['m', 'ɔ', 'ʁ', 'ŋ']
g2p.variants("Morgen")                       # every pronunciation in the dictionary
g2p.batch(["Haus", "Häuschen"])              # many words at once
g2p.transcribe("Guten Morgen", phone_sep=" ", word_sep=" | ")

agentg2p.languages()                         # every language tag
agentg2p.dictionaries("eng")                 # all English dictionaries, default first
agentg2p.G2P(dictionary="wiktionary/eng_latn_us_broad")   # pick a specific dictionary
agentg2p.G2P("fra", backend="fst")           # force a model: fst, byt5, byt5-pooled, ...
```

From the command line:

```bash
agentg2p -l de "Guten Morgen"
agentg2p -l en-US --words < words.txt        # word<TAB>phones<TAB>source per line
agentg2p --list                              # language tags
agentg2p --list eng                          # dictionaries for one language, with test error rates
```

## Desktop window

```bash
agentg2p ui                 # open the agentg2p window
agentg2p ui --browser       # the same interface in your default browser
```

The window does everything above without code. It has four views:

- **Transcribe:** type or paste text. Each word gets its phones and its source (dictionary, FST or ByT5). Words that lost letters the model doesn't know are flagged, and so are numbers, which are not read out.
- **Word list:** open a TXT, CSV or TSV file and export the results as TSV.
- **Compare:** one word in every dictionary of the language, with all variants.
- **Dictionaries:** all 717 dictionaries, with their notation, size, model and test error rate; download any of them here.

The window runs a small server on `127.0.0.1` that only it can use, and keeps its preferences in `~/.config/agentg2p/ui.json`. On Linux, pywebview also needs GTK or Qt; without them, `agentg2p ui` opens your browser.

## Data

The dictionaries and models (1.6 GB in total) are on the
[v0.1.0 release](https://github.com/extreme-coder/agentg2p-lib/releases/tag/v0.1.0).
agentg2p **downloads only what it needs, the first time it needs it**: the default German
dictionary is about 17 MB, and a ByT5 model is 29 MB. Each file is checked against its
SHA-256 hash.

- Data goes to `~/.cache/agentg2p`. Set `AGENTG2P_DATA` to use another folder.
- To download ahead of time: `agentg2p --download de fr ja`, or `agentg2p --download-all`
  for everything. From Python: `agentg2p.download("de", "fr")`.
- `AGENTG2P_OFFLINE=1` turns downloading off.

## Languages and dictionaries

- **Naming a language:** a language can be given as an ISO 639-3 tag as used by the
  dictionaries (`deu`, `eng-us`, `cmn-tw`), an ISO 639-1 code (`de`, `en-GB`, `pt-BR`), an
  ISO 639-2/B code (`ger`) or an English name (`German`, `Mandarin`). Macrolanguages fall
  back to their members (`zh` → Mandarin, `ar` → Arabic varieties, `no` → Bokmål/Nynorsk).
- **Several dictionaries per language:** most languages have more than one, each with its
  own transcription convention, for example MFA's phone set versus WikiPron broad or
  narrow IPA.
- **The default:** the largest IPA dictionary for the language. ARPAbet, jamo and pinyin
  dictionaries are never the default.
- **Choosing one:** use `dictionary=` or `source=` when the convention matters. Lookup and
  models always use the same dictionary, so the output stays in one convention.

## How text is handled

- **Words** are runs of letters and combining marks, with apostrophes, hyphens and ZWJ/ZWNJ
  inside them. Digits and punctuation are kept as they are, but not transcribed.
- **Multi-word dictionary entries** ("ice cream", Vietnamese syllable groups) are matched
  longest first.
- **Hyphenated compounds** that are not in the dictionary are split at the hyphens.
- **Scripts written without spaces** (Chinese, Japanese, Thai, Lao, Khmer, Burmese) are
  segmented into dictionary words. Unknown stretches go to the model.
- **Case:** before a word goes to a model, its case is matched to the dictionary's
  alphabet. `Morgen` becomes `morgen` for a lowercase dictionary, but stays `Morgen` for one
  that capitalizes nouns.

## Accuracy

- **Dictionary words** get the dictionary's own pronunciation.
- **Other words:** each dictionary uses the model with the lower phone error rate (PER)
  on that dictionary's held-out 10% test set. The median test PER of the chosen model is
  0.73% for MFA, 10.2% for WikiPron and 1.2% for Charsiu dictionaries. This is slightly
  optimistic, because the same test set chose the model.
- **Per-dictionary numbers:** shown by `agentg2p --list LANG`, and stored in the manifest.
- **The FST decoder** is pure Python + numpy, so the Phonetisaurus binaries (Linux-only)
  are not needed. On 81,628 test words across all 717 models, it gives the same output as
  `phonetisaurus-g2pfst`. The only differences are 411 words where two outputs have the
  same path cost, which the two decoders break differently.
- **The ByT5 backend** matches the original evaluation code on 6,000 of 6,000 test words.
- **Speed (one CPU core):** the FST takes about 0.2 ms per word, and ByT5 about 60 ms per
  word (batched, 5 beams).

## Limitations

- Numbers, abbreviations and symbols are not expanded.
- Words are transcribed one at a time, without context. A heteronym (*read*, *lead*) gets
  the dictionary's first pronunciation, and there are no changes across word boundaries.
- The test PER is measured on held-out dictionary words. Names and loanwords are usually
  harder.
- You have to say which language the text is in; there is no language detection.

## Data sources and licences

The bundled data comes from [MFA](https://mfa-models.readthedocs.io/),
[WikiPron](https://github.com/CUNY-CL/wikipron) (Wiktionary) and
[CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P), and the models are trained on it.
See [DATA_LICENSE.md](https://github.com/extreme-coder/agentg2p-lib/blob/main/DATA_LICENSE.md). The code is MIT-licensed ([LICENSE](https://github.com/extreme-coder/agentg2p-lib/blob/main/LICENSE)).

The desktop window bundles two fonts, [Archivo](https://github.com/Omnibus-Type/Archivo) and [Charis SIL](https://software.sil.org/charis/), under the SIL Open Font License; the licence texts are in [`src/agentg2p/ui/static/fonts/`](https://github.com/extreme-coder/agentg2p-lib/tree/main/src/agentg2p/ui/static/fonts). Its English language names come from the [ISO 639-3 code tables](https://iso639-3.sil.org/code_tables/download_tables) published by SIL International.

## Development

```bash
pip install -e .
python -m unittest discover -s tests
AGENTG2P_TEST_BUNDLE=~/.cache/agentg2p python -m unittest discover -s tests   # with a full data folder
```

Releasing (code to PyPI, data to GitHub Releases): see [RELEASING.md](https://github.com/extreme-coder/agentg2p-lib/blob/main/RELEASING.md).
Changes: [CHANGELOG.md](https://github.com/extreme-coder/agentg2p-lib/blob/main/CHANGELOG.md).
