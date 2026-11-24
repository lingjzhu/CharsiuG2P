# Changelog

The code version is `agentg2p.__version__`. Data releases are separate GitHub releases;
the code downloads from the one named in `agentg2p.remote.DATA_RELEASE` (currently
`v0.1.0`).

## 0.1.2 (unreleased)

- New desktop window, `agentg2p ui`, for people who don't code:
  - four views: Transcribe, Word list (TXT/CSV/TSV in, TSV out), Compare (one word in every dictionary of a language) and Dictionaries (browse and download);
  - every word shows its source (dictionary, FST or ByT5), and the window flags letters a model drops and numbers that are not read out;
  - `agentg2p ui --browser` opens it in a browser instead.

  pywebview is a new dependency. On Linux, pywebview also needs GTK or Qt; without them, `agentg2p ui` opens the browser.
- Packaging for PyPI:
  - SPDX licence metadata (`MIT`), with `LICENSE` and `DATA_LICENSE.md` shipped as licence files;
  - a single version source;
  - Python version classifiers and project URLs;
  - the source distribution now includes the tests, test fixtures and release tool.
- `agentg2p --version` prints the code and data versions.
- README links are absolute, so they work on PyPI.
- CI builds the sdist and wheel and checks them (`twine check --strict`). It then installs
  the wheel into a clean environment, and runs the test suite from the unpacked sdist.
- New `publish` workflow, which uploads to PyPI or TestPyPI with trusted publishing (see
  `RELEASING.md`).

## 0.1.1 (2026-11-19)

- Data downloads failed with `CERTIFICATE_VERIFY_FAILED` on Pythons that have no usable
  system certificates, such as python.org's macOS installer. They now fall back to
  certifi's CA bundle. certifi is a new dependency.

## 0.1.0 (2026-11-19)

- First release:
  - dictionary lookup, then the FST or ByT5 backend per dictionary;
  - a pure-Python Phonetisaurus decoder;
  - text segmentation and a CLI.
- Data release `v0.1.0`: 717 dictionaries and 4 ByT5 models, downloaded on first use.
