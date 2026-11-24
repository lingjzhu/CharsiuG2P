# -*- coding: utf-8 -*-
"""Phonetisaurus G2P models without the Phonetisaurus binaries.

Phonetisaurus trains a joint grapheme-phoneme n-gram model and converts it to an
OpenFst transducer (phonetisaurus-arpa2wfst): input labels are grapheme chunks,
output labels phone chunks, multi-symbol chunks are joined with "|", "_" marks a
deleted grapheme, and n-gram backoff arcs are input-epsilon arcs. Decoding
(phonetisaurus-g2pfst, 1-best) composes a linear acceptor of the word's chunks
with the model and takes the shortest path in the tropical semiring.

This module reads the model file (OpenFst "vector" format, "standard" arcs) and
runs that shortest path directly: a Viterbi pass over (word position, model
state), with epsilon arcs relaxed within a position. Path weights are summed in
float32 like OpenFst's TropicalWeight, so ties and near-ties resolve the same
way. Only numpy is needed, so it runs on macOS and Windows too.

Models are converted once to a compact .npz (``FSTModel.save``) that loads fast.
"""

import struct
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

FST_MAGIC = 2125659606
SYMTAB_MAGIC = 2125658996
_HAS_ISYMBOLS, _HAS_OSYMBOLS, _IS_ALIGNED = 1, 2, 4
TIE = "|"
SKIP = "_"
SPACE_PLACEHOLDER = "\u2581"  # spaces in multi-word entries were encoded this way for training
# Characters Phonetisaurus reserves, as escaped for training by
# the training scripts (same table). Harmless for models trained
# without escaping: those never saw these characters.
RESERVED = {"}": "\ue000", "|": "\ue001", "_": "\ue002", "\xa0": "\ue003", " ": SPACE_PLACEHOLDER}
_UNRESERVED = {v: k for k, v in RESERVED.items() if k != " "}
_INF = float("inf")
_ARC_DTYPE = np.dtype([("ilabel", "<i4"), ("olabel", "<i4"), ("weight", "<f4"), ("nextstate", "<i4")])


def _compact(a: np.ndarray) -> np.ndarray:
    """The narrowest signed integer type that holds every value of ``a``."""
    for dt in (np.int8, np.int16, np.int32):
        if a.size == 0 or (int(a.min()) >= np.iinfo(dt).min and int(a.max()) <= np.iinfo(dt).max):
            return a.astype(dt)
    return a.astype(np.int64)


def _f32(x: float) -> float:
    """Rounds to float32 (an exact float64 sum of two float32 values, rounded, is their float32 sum)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def unpack(self, fmt: str):
        vals = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += struct.calcsize(fmt)
        return vals

    def string(self) -> str:
        (n,) = self.unpack("<i")
        s = self.data[self.pos:self.pos + n].decode("utf-8")
        self.pos += n
        return s

    def symbol_table(self) -> List[str]:
        (magic,) = self.unpack("<i")
        if magic != SYMTAB_MAGIC:
            raise ValueError("bad symbol table magic number")
        self.string()  # table name
        self.unpack("<q")  # available key
        (size,) = self.unpack("<q")
        table: Dict[int, str] = {}
        for _ in range(size):
            sym = self.string()
            (key,) = self.unpack("<q")
            table[key] = sym
        out = [""] * (max(table) + 1 if table else 0)
        for k, v in table.items():
            out[k] = v
        return out


class FSTModel:
    """A Phonetisaurus model held as flat arrays, arcs sorted by (state, ilabel)."""

    def __init__(self, start: int, final: np.ndarray, state_ptr: np.ndarray, n_eps: np.ndarray,
                 ilabel: np.ndarray, olabel: np.ndarray, weight: np.ndarray, nextstate: np.ndarray,
                 isyms: Sequence[str], osyms: Sequence[str]):
        self.start = int(start)
        # Integer arrays keep whatever width they come in (save() stores the smallest
        # that fits); weights stay float32 so path sums match OpenFst.
        self.final = np.asarray(final, dtype=np.float32)
        self.state_ptr = np.asarray(state_ptr)
        self.n_eps = np.asarray(n_eps)
        self.ilabel = np.asarray(ilabel)
        self.olabel = np.asarray(olabel)
        self.weight = np.asarray(weight, dtype=np.float32)
        self.nextstate = np.asarray(nextstate)
        self.isyms = list(isyms)
        self.osyms = list(osyms)
        self._prepare()

    # ---- loading and saving -------------------------------------------------

    @classmethod
    def from_openfst(cls, path: Union[str, Path]) -> "FSTModel":
        """Reads a model written by phonetisaurus-train (model.fst)."""
        data = Path(path).read_bytes()
        r = _Reader(data)
        (magic,) = r.unpack("<i")
        if magic != FST_MAGIC:
            raise ValueError(f"{path}: not an OpenFst binary file")
        fst_type, arc_type = r.string(), r.string()
        if fst_type != "vector" or arc_type != "standard":
            raise ValueError(f"{path}: unsupported FST type {fst_type}/{arc_type}")
        _version, flags = r.unpack("<ii")
        _props, start, n_states, _n_arcs = r.unpack("<Qqqq")
        if flags & _IS_ALIGNED:
            raise ValueError(f"{path}: aligned FST files are not supported")
        isyms = r.symbol_table() if flags & _HAS_ISYMBOLS else None
        osyms = r.symbol_table() if flags & _HAS_OSYMBOLS else None
        if isyms is None or osyms is None:
            raise ValueError(f"{path}: the model has no symbol tables")

        final = np.empty(n_states, dtype=np.float32)
        counts = np.empty(n_states, dtype=np.int64)
        arc_offsets = np.empty(n_states, dtype=np.int64)
        pos = r.pos
        unpack = struct.unpack_from
        for s in range(n_states):
            fw, n = unpack("<fq", data, pos)
            final[s] = fw
            counts[s] = n
            arc_offsets[s] = pos + 12
            pos += 12 + 16 * n
        if pos != len(data):
            raise ValueError(f"{path}: {len(data) - pos} unexpected trailing bytes")

        total = int(counts.sum())
        # State records are 12 + 16n bytes, so every arc record starts at the same
        # offset modulo 4: one aligned int32 view reads all four fields.
        base = int(arc_offsets[0]) % 4 if n_states else 0
        words = np.frombuffer(data, dtype="<i4", offset=base, count=(len(data) - base) // 4)
        floats = words.view("<f4")
        first = np.cumsum(counts) - counts
        idx = (np.repeat((arc_offsets - base) // 4, counts)
               + 4 * (np.arange(total, dtype=np.int64) - np.repeat(first, counts)))
        ilabel, olabel = words[idx], words[idx + 1]
        weight, nextstate = floats[idx + 2], words[idx + 3]
        del idx
        state_of_arc = np.repeat(np.arange(n_states, dtype=np.int64), counts)
        # Sort arcs by ilabel within each state (input-epsilon arcs come first).
        order = np.lexsort((ilabel, state_of_arc))
        ilabel, olabel, weight, nextstate = ilabel[order], olabel[order], weight[order], nextstate[order]
        state_of_arc = state_of_arc[order]
        state_ptr = np.zeros(n_states + 1, dtype=np.int64)
        state_ptr[1:] = np.cumsum(counts)
        n_eps = np.bincount(state_of_arc[ilabel == 0], minlength=n_states)
        if total and ilabel.min() < 0:
            raise ValueError(f"{path}: negative input labels")
        return cls(start, final, state_ptr, n_eps, ilabel, olabel, weight, nextstate, isyms, osyms)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        # Compressed, smallest integer widths: about 30% of the OpenFst file, loads in ~0.1 s.
        np.savez_compressed(
            tmp, start=np.int64(self.start), final=self.final, state_ptr=_compact(self.state_ptr),
            n_eps=_compact(self.n_eps), ilabel=_compact(self.ilabel), olabel=_compact(self.olabel),
            weight=self.weight, nextstate=_compact(self.nextstate),
            isyms=np.array(self.isyms, dtype=object), osyms=np.array(self.osyms, dtype=object))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FSTModel":
        """Loads an .npz written by ``save``, or a raw OpenFst model.fst."""
        path = Path(path)
        if path.suffix != ".npz":
            return cls.from_openfst(path)
        with np.load(path, allow_pickle=True) as z:
            return cls(int(z["start"]), z["final"], z["state_ptr"], z["n_eps"], z["ilabel"],
                       z["olabel"], z["weight"], z["nextstate"], z["isyms"].tolist(), z["osyms"].tolist())

    # ---- decoding -----------------------------------------------------------

    def _prepare(self) -> None:
        # Single graphemes and multi-grapheme chunks ("a|b" -> ("a", "b")).
        self._single: Dict[str, int] = {}
        self._clusters: Dict[Tuple[str, ...], int] = {}
        self._max_cluster = 1
        # Every symbol string; phonetisaurus-g2pfst drops input characters not in it.
        self._table = set(self.isyms[1:])
        for i, sym in enumerate(self.isyms):
            # "_" and "|" are reserved: no arc reads them, so words containing them
            # have no path, as in phonetisaurus-g2pfst.
            if i == 0 or not sym or sym in (TIE, SKIP, "<eps>"):
                continue
            if TIE in sym and sym != TIE:
                parts = tuple(sym.split(TIE))
                self._clusters[parts] = i
                self._max_cluster = max(self._max_cluster, len(parts))
            else:
                self._single[sym] = i
        # Output symbol id -> phones ("e|k" -> ["e", "k"]); eps, "|" and "_" emit nothing.
        self._out: List[List[str]] = []
        for i, sym in enumerate(self.osyms):
            if i == 0 or sym in ("", TIE, SKIP, "<eps>"):
                self._out.append([])
            else:
                self._out.append(["".join(_UNRESERVED.get(c, c) for c in p)
                                  for p in sym.split(TIE) if p])
        # Python lists are much faster than numpy scalars in the inner loop.
        self._ptr = self.state_ptr.tolist()
        self._neps = self.n_eps.tolist()
        self._final = self.final.tolist()

    def graphemes(self) -> set:
        """Every single grapheme the model knows."""
        return set(self._single)

    def _arcs(self, state: int, label: int) -> Tuple[int, int]:
        lo, hi = self._ptr[state], self._ptr[state + 1]
        if hi - lo <= 8:
            il = self.ilabel
            a = lo
            while a < hi and il[a] < label:
                a += 1
            b = a
            while b < hi and il[b] == label:
                b += 1
            return a, b
        seg = self.ilabel[lo:hi]
        return lo + int(np.searchsorted(seg, label, "left")), lo + int(np.searchsorted(seg, label, "right"))

    def _closure(self, layer: Dict[int, tuple], pos: int) -> None:
        """Relaxes input-epsilon (backoff) arcs within one word position."""
        weight, nextstate, ptr, neps = self.weight, self.nextstate, self._ptr, self._neps
        queue = deque(layer)
        queued = set(layer)
        while queue:
            s = queue.popleft()
            queued.discard(s)
            n = neps[s]
            if not n:
                continue
            c = layer[s][0]
            lo = ptr[s]
            for a in range(lo, lo + n):
                t = int(nextstate[a])
                nc = _f32(c + float(weight[a]))
                old = layer.get(t)
                if old is None or nc < old[0]:
                    layer[t] = (nc, (pos, s, a))
                    if t not in queued:
                        queue.append(t)
                        queued.add(t)

    def decode(self, word: str) -> Optional[List[str]]:
        """1-best phones for ``word``, or None if the model has no path for it.

        Like phonetisaurus-g2pfst, characters missing from the model's symbol
        table are dropped before decoding (checked against the binary on the
        build's validation words).
        """
        chars = [c for c in (RESERVED.get(c, c) for c in word) if c in self._table]
        n = len(chars)
        if n == 0:
            return None
        # Arcs of the word acceptor: (label, end position) from each position.
        steps: List[List[Tuple[int, int]]] = []
        for i in range(n):
            out = []
            sid = self._single.get(chars[i])
            if sid is not None:
                out.append((sid, i + 1))
            for j in range(2, min(self._max_cluster, n - i) + 1):
                cid = self._clusters.get(tuple(chars[i:i + j]))
                if cid is not None:
                    out.append((cid, i + j))
            steps.append(out)

        weight, nextstate = self.weight, self.nextstate
        layers: List[Dict[int, tuple]] = [dict() for _ in range(n + 1)]
        layers[0][self.start] = (0.0, None)
        for i in range(n + 1):
            layer = layers[i]
            if not layer:
                continue
            self._closure(layer, i)
            if i == n:
                break
            items = list(layer.items())
            for label, j in steps[i]:
                target = layers[j]
                for s, (c, _) in items:
                    lo, hi = self._arcs(s, label)
                    for a in range(lo, hi):
                        t = int(nextstate[a])
                        nc = _f32(c + float(weight[a]))
                        old = target.get(t)
                        if old is None or nc < old[0]:
                            target[t] = (nc, (i, s, a))

        best_state, best_cost = None, _INF
        final = self._final
        for s, (c, _) in layers[n].items():
            fw = final[s]
            if fw != _INF:
                total = _f32(c + fw)
                if total < best_cost:
                    best_state, best_cost = s, total
        if best_state is None:
            return None

        labels = []
        pos, s = n, best_state
        back = layers[pos][s][1]
        while back is not None:
            prev_pos, prev_state, arc = back
            labels.append(int(self.olabel[arc]))
            pos, s = prev_pos, prev_state
            back = layers[pos][s][1]
        phones: List[str] = []
        for lab in reversed(labels):
            phones.extend(self._out[lab])
        self.last_cost = best_cost
        # A best path that deletes every grapheme prints an empty pronunciation, which
        # phonetisaurus-apply reports as no pronunciation.
        return phones or None

    def cost(self, word: str, phones: Sequence[str]) -> Optional[float]:
        """Cost (-log probability, float32 sums) of the best path that reads
        ``word`` and outputs exactly ``phones``, or None if there is none.

        ``decode`` returns a minimum-cost output, so two outputs with equal
        cost are equally good answers (OpenFst breaks such ties by heap order).
        """
        chars = [c for c in (RESERVED.get(c, c) for c in word) if c in self._table]
        n, target = len(chars), list(phones)
        if n == 0:
            return None
        steps = []
        for i in range(n):
            out = []
            sid = self._single.get(chars[i])
            if sid is not None:
                out.append((sid, i + 1))
            for j in range(2, min(self._max_cluster, n - i) + 1):
                cid = self._clusters.get(tuple(chars[i:i + j]))
                if cid is not None:
                    out.append((cid, i + j))
            steps.append(out)
        best: Dict[Tuple[int, int, int], float] = {(0, self.start, 0): 0.0}
        queue = deque(best)
        while queue:
            key = queue.popleft()
            pos, s, k = key
            c = best[key]
            arcs = list(range(self._ptr[s], self._ptr[s] + self._neps[s]))
            ends = [pos] * len(arcs)
            if pos < n:
                for label, j in steps[pos]:
                    lo, hi = self._arcs(s, label)
                    arcs.extend(range(lo, hi))
                    ends.extend([j] * (hi - lo))
            for a, j in zip(arcs, ends):
                emitted = self._out[int(self.olabel[a])]
                if target[k:k + len(emitted)] != emitted:
                    continue
                nk = (j, int(self.nextstate[a]), k + len(emitted))
                nc = _f32(c + float(self.weight[a]))
                if nk not in best or nc < best[nk]:
                    best[nk] = nc
                    queue.append(nk)
        totals = [_f32(c + self._final[s]) for (pos, s, k), c in best.items()
                  if pos == n and k == len(target) and self._final[s] != _INF]
        return min(totals) if totals else None
