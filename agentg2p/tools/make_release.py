#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packs a built data bundle into GitHub release assets.

Input: a bundle folder (manifest.json, lexicons/, fst/, byt5/), as built by the
AgentG2P research repository (src/agentg2p_build/bundle.py).
Output folder:
- dict__<source>__<name>.tar: one dictionary's lexicon and FST;
- byt5__<condition>.tar: one ByT5 model;
- manifest.json: the bundle manifest, with each dictionary's and model's
  "asset" and a "release" section listing every asset's size and SHA-256.
The tars are deterministic (sorted, zero mtimes and owners).

Usage: python tools/make_release.py --bundle <bundle dir> --out <release dir> --tag v0.1.0
Then:  gh release create v0.1.0 <release dir>/* (in batches; see README "Releasing").
"""

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

REPO = "extreme-coder/agentg2p-lib"


def add_file(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    data = path.read_bytes()
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def make_tar(out: Path, files) -> dict:
    with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as tar:
        for src, arcname in sorted(files, key=lambda f: f[1]):
            add_file(tar, src, arcname)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    return {"size": out.stat().st_size, "sha256": digest}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    bundle, out = Path(args.bundle), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assets = {}

    for k, d in enumerate(manifest["dictionaries"]):
        name = f"dict__{d['source']}__{d['name']}.tar"
        files = [(bundle / d["lexicon"], d["lexicon"])]
        if d.get("fst"):
            files.append((bundle / d["fst"], d["fst"]))
        for src, _ in files:
            if not src.is_file():
                sys.exit(f"missing {src}")
        assets[name] = make_tar(out / name, files)
        d["asset"] = name
        if k % 100 == 0:
            print(f"[{k + 1}/{len(manifest['dictionaries'])}] {name}", flush=True)

    for model_name, m in manifest["models"].items():
        folder = bundle / m["path"]
        name = f"byt5__{m['condition']}.tar"
        files = [(f, f"{m['path']}/{f.name}") for f in sorted(folder.iterdir()) if f.is_file()]
        assets[name] = make_tar(out / name, files)
        m["asset"] = name

    manifest["release"] = {"repo": REPO, "tag": args.tag, "assets": assets}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(a["size"] for a in assets.values())
    print(f"{len(assets)} assets + manifest.json, {total / 1e9:.2f} GB, largest "
          f"{max(a['size'] for a in assets.values()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
