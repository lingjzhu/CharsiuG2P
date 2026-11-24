# Releasing agentg2p

Code and data are released separately:

- **Code** goes to PyPI as the `agentg2p` package. The version is in
  `src/agentg2p/__init__.py` (`__version__`).
- **Data** (dictionaries and models) is attached to a GitHub release, currently
  [`v0.1.0`](https://github.com/extreme-coder/agentg2p-lib/releases/tag/v0.1.0). The code
  downloads from the release named in `src/agentg2p/remote.py` (`DATA_RELEASE`).

## One-time setup for PyPI (trusted publishing, no API token)

1. Create an account on <https://pypi.org> and turn on two-factor authentication. Doing the
   same on <https://test.pypi.org> is optional but recommended.
2. On PyPI, open *Your account → Publishing → Add a new pending publisher*
   (<https://pypi.org/manage/account/publishing/>) and enter:
   - PyPI project name: `agentg2p`
   - Owner: `extreme-coder`
   - Repository name: `agentg2p-lib`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

   For TestPyPI, add the same publisher there with environment `testpypi`.
3. On GitHub, open *Settings → Environments* and create the environments `pypi` and
   `testpypi`. Optionally, add yourself as a required reviewer on `pypi`, so every upload
   waits for a click.

`.github/workflows/publish.yml` uploads only from these environments, through
`pypa/gh-action-pypi-publish`. GitHub vouches for the workflow run, so no secret is stored
in the repository.

## Releasing a new code version

1. **Version and notes:** bump `__version__` in `src/agentg2p/__init__.py`, and move the
   "unreleased" notes in `CHANGELOG.md` under the new version and today's date.
2. **First PyPI release only:** in `README.md`, change the install lines to
   ```bash
   pip install agentg2p
   pip install "agentg2p[neural]"     # with the ByT5 models (torch + transformers)
   ```
   and keep `pip install git+https://github.com/extreme-coder/agentg2p-lib` as "latest from
   GitHub".
3. **Check locally:**
   ```bash
   python -m pip install build twine
   python -m build                      # dist/agentg2p-X.Y.Z.tar.gz and the .whl
   python -m twine check --strict dist/*
   python -m unittest discover -s tests
   ```
   CI does the same on every push (the `package` job in `tests.yml`): it also installs the
   wheel into a clean environment and runs the tests from the unpacked sdist.
4. **Commit and push** to `main`, and wait for CI to pass.
5. **Optional dry run on TestPyPI:** go to *Actions → publish → Run workflow* and choose
   `testpypi`. Then check the result in a clean environment:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ agentg2p
   ```
6. **Publish:** create a GitHub release with the tag `vX.Y.Z`, matching `__version__` exactly:
   ```bash
   gh release create vX.Y.Z --title "agentg2p vX.Y.Z" --notes-file <notes> --latest=false
   ```
   Publishing the release starts `publish.yml`, which builds, checks, and uploads to PyPI.
   A release whose tag doesn't match the package version (a data-only release) is built but
   not uploaded. Use `--latest=false` while `v0.1.0` holds the data, so the "Latest" badge
   stays on the release that has the data files.
7. **Verify:** `pip install agentg2p==X.Y.Z` in a fresh environment, then run
   `agentg2p --version` and `agentg2p -l de "Guten Morgen"`.

A PyPI version can't be replaced once it is uploaded (it can only be yanked). If something
is wrong, release `X.Y.(Z+1)`.

Manual fallback (with a PyPI API token): `python -m build && python -m twine upload dist/*`.

## Releasing new data

1. Build the data bundle in the AgentG2P research project. It needs `manifest.json`,
   `lexicons/`, `fst/` and `byt5/`.
2. Pack the bundle:
   `python tools/make_release.py --bundle <bundle dir> --out <release dir> --tag vA.B.C`
3. Upload it as a draft, then publish once every asset is up:
   ```bash
   gh release create vA.B.C --draft --title "agentg2p data vA.B.C" --notes-file <notes>
   gh release upload vA.B.C <release dir>/*      # in batches of about 60 files
   gh release edit vA.B.C --draft=false
   ```
4. Point the code at the new data: set `DATA_RELEASE = "vA.B.C"` in `src/agentg2p/remote.py`,
   update the `Data` URL in `pyproject.toml`, and release a new code version (above). Older
   code versions keep using their own data release.

Each asset must stay under GitHub's 2 GB limit; today the largest is 49 MB.
