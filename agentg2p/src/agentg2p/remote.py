# -*- coding: utf-8 -*-
"""Downloading the data bundle from the GitHub release.

The release holds ``manifest.json`` plus one archive per dictionary
(``dict__<source>__<name>.tar``: its lexicon and FST) and one per ByT5 model
(``byt5__<condition>.tar``). The manifest lists every archive with its size and
SHA-256. Archives are fetched only when needed, checked, and unpacked into the
data folder (default ``~/.cache/agentg2p``).

Environment:
- AGENTG2P_DATA: data folder.
- AGENTG2P_OFFLINE=1: never download.
- AGENTG2P_DATA_URL: base URL of the release files (default: the GitHub release).
"""

import hashlib
import json
import os
import shutil
import ssl
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional, Union

REPO = "extreme-coder/agentg2p-lib"
DATA_RELEASE = "v0.1.0"
USER_AGENT = "agentg2p-downloader"


class DownloadError(RuntimeError):
    pass


def offline() -> bool:
    return os.environ.get("AGENTG2P_OFFLINE", "").strip() not in ("", "0", "false", "no")


def base_url() -> str:
    return os.environ.get("AGENTG2P_DATA_URL") or \
        f"https://github.com/{REPO}/releases/download/{DATA_RELEASE}"


def default_data_dir() -> Path:
    env = os.environ.get("AGENTG2P_DATA")
    return Path(env) if env else Path.home() / ".cache" / "agentg2p"


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(f"agentg2p: {msg}", file=sys.stderr, flush=True)


def _urlopen(req):
    """Opens with the system's certificates, then certifi's if those fail to verify.

    python.org's macOS Python, for one, has no usable system certificates until
    "Install Certificates.command" is run; certifi covers that case, and trying the
    system store first keeps corporate CAs working.
    """
    try:
        return urllib.request.urlopen(req, timeout=60)
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ssl.SSLCertVerificationError):
            raise
        try:
            import certifi
        except ImportError:
            raise e from None
        context = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(req, timeout=60, context=context)


def _fetch(name: str, dest: Path, quiet: bool = False, size: Optional[int] = None) -> None:
    if offline():
        raise DownloadError(f"{name} is not downloaded and AGENTG2P_OFFLINE is set")
    url = f"{base_url().rstrip('/')}/{name}"
    what = f" ({size / 1e6:.1f} MB)" if size else ""
    _log(f"downloading {name}{what}", quiet)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with _urlopen(req) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"could not download {url}: {e}") from e
    tmp.replace(dest)


def fetch_manifest(data_dir: Optional[Union[str, Path]] = None, quiet: bool = False) -> Path:
    """Downloads manifest.json into the data folder (replacing an older one)."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    path = data_dir / "manifest.json"
    _fetch("manifest.json", path, quiet)
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        path.unlink()
        raise DownloadError(f"downloaded manifest.json is not valid JSON: {e}") from e
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(archive: Path, data_dir: Path) -> List[str]:
    """Unpacks regular files only, refusing absolute paths and '..'."""
    written = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            p = PurePosixPath(member.name)
            if member.isdir():
                continue
            if not member.isfile() or p.is_absolute() or ".." in p.parts:
                raise DownloadError(f"{archive.name}: refusing archive member {member.name!r}")
            target = data_dir.joinpath(*p.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            tmp = target.with_name(target.name + ".part")
            with open(tmp, "wb") as f:
                shutil.copyfileobj(src, f)
            tmp.replace(target)
            written.append(str(p))
    return written


def fetch_asset(name: str, data_dir: Union[str, Path], manifest: dict, quiet: bool = False) -> None:
    """Downloads one release archive, checks its SHA-256 and unpacks it."""
    data_dir = Path(data_dir)
    info = manifest.get("release", {}).get("assets", {}).get(name)
    if info is None:
        raise DownloadError(f"{name} is not part of this data release")
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        archive = Path(tmp) / name
        _fetch(name, archive, quiet, info.get("size"))
        digest = _sha256(archive)
        if digest != info["sha256"]:
            raise DownloadError(f"{name}: SHA-256 mismatch (got {digest}, expected {info['sha256']})")
        _safe_extract(archive, data_dir)


def download(*langs: str, dictionaries: Iterable[str] = (), all_dictionaries: bool = False,
             neural: Optional[bool] = None, data_dir: Optional[Union[str, Path]] = None,
             quiet: bool = False) -> Path:
    """Downloads data ahead of time (otherwise it is fetched on first use).

    ``langs``: languages (code or name); each gets its default dictionary.
    ``dictionaries``: dictionary ids. ``all_dictionaries``: everything (about 1.5 GB).
    ``neural``: also fetch the ByT5 models these dictionaries use; by default
    only if torch and transformers are installed.
    Returns the data folder.
    """
    from . import byt5
    from .assets import Catalog
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    if not (data_dir / "manifest.json").is_file():
        fetch_manifest(data_dir, quiet)
    catalog = Catalog(data_dir)
    if all_dictionaries:
        chosen = list(catalog.dictionaries)
    else:
        chosen = [catalog.resolve(lang) for lang in langs] + [catalog.resolve(dictionary=d) for d in dictionaries]
    if neural is None:
        neural = byt5.available()
    models = set()
    for d in chosen:
        if not catalog.installed(d):
            fetch_asset(d.asset, data_dir, catalog.manifest, quiet)
        if neural and d.backend in catalog.models:
            models.add(d.backend)
    for name in sorted(models if not all_dictionaries else (catalog.models if neural else ())):
        if not catalog.model_installed(name):
            fetch_asset(catalog.models[name]["asset"], data_dir, catalog.manifest, quiet)
    _log(f"data ready in {data_dir}", quiet)
    return data_dir
