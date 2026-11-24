# -*- coding: utf-8 -*-
"""Writes small OpenFst binary models (the format phonetisaurus-train produces)."""

import itertools
import math
import struct
from typing import List, Tuple

FST_MAGIC = 2125659606
SYMTAB_MAGIC = 2125658996


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<i", len(b)) + b


def _symtab(name: str, syms: List[str]) -> bytes:
    out = struct.pack("<i", SYMTAB_MAGIC) + _string(name) + struct.pack("<qq", len(syms), len(syms))
    for k, s in enumerate(syms):
        out += _string(s) + struct.pack("<q", k)
    return out


def write_fst(path, start: int, states: List[Tuple[float, List[Tuple[int, int, float, int]]]],
              isyms: List[str], osyms: List[str]) -> None:
    """states[s] = (final weight or inf, [(ilabel, olabel, weight, nextstate), ...])."""
    n_arcs = sum(len(a) for _, a in states)
    data = struct.pack("<i", FST_MAGIC) + _string("vector") + _string("standard")
    data += struct.pack("<ii", 2, 3) + struct.pack("<Qqqq", 0, start, len(states), n_arcs)
    data += _symtab("isyms", isyms) + _symtab("osyms", osyms)
    for final, arcs in states:
        data += struct.pack("<fq", final, len(arcs))
        for il, ol, w, nx in arcs:
            data += struct.pack("<iifi", il, ol, w, nx)
    with open(path, "wb") as f:
        f.write(data)


INF = float("inf")
# A toy joint n-gram model: state 0 is the <s> history, 1 the unigram (backoff)
# state, 2 the history after "a". Backoff from 2 has a negative weight.
ISYMS = ["<eps>", "|", "_", "a", "b", "c", "a|b", "\ue000", "\u2581"]
OSYMS = ["<eps>", "|", "_", "x", "y", "z", "x|y", "\ue000"]
STATES = [
    (INF, [(0, 0, 0.5, 1), (3, 3, 1.0, 2)]),
    (0.3, [(3, 3, 2.0, 1), (4, 4, 2.0, 1), (5, 2, 1.0, 1), (6, 6, 1.5, 1), (7, 7, 0.7, 1),
           (8, 5, 0.2, 1)]),
    (INF, [(4, 5, 0.1, 1), (0, 0, -0.2, 1)]),
]


def brute_force(word: str, isyms=ISYMS, osyms=OSYMS, states=STATES, start=0):
    """Best output by enumerating every path (eps arcs at most 3 in a row)."""
    sym = {s: i for i, s in enumerate(isyms)}
    chars = [{"}": "\ue000", " ": "\u2581"}.get(c, c) for c in word]
    best = (math.inf, None)

    def walk(state, pos, cost, out, eps_run):
        nonlocal best
        if pos == len(chars):
            fw = states[state][0]
            if fw != INF and cost + fw < best[0]:
                best = (cost + fw, out)
        for il, ol, w, nx in states[state][1]:
            if il == 0:
                if eps_run < 3:
                    walk(nx, pos, cost + w, out + [ol], eps_run + 1)
                continue
            label = isyms[il]
            parts = label.split("|") if "|" in label and label != "|" else [label]
            if chars[pos:pos + len(parts)] == parts:
                walk(nx, pos + len(parts), cost + w, out + [ol], 0)

    if chars:
        walk(start, 0, 0.0, [], 0)
    if best[1] is None:
        return None
    phones = []
    for ol in best[1]:
        s = osyms[ol]
        if ol == 0 or s in ("|", "_"):
            continue
        phones.extend("}" if p == "\ue000" else p for p in s.split("|"))
    return phones or None  # all graphemes deleted: phonetisaurus reports no pronunciation


def all_words(alphabet: str, max_len: int):
    for n in range(1, max_len + 1):
        for t in itertools.product(alphabet, repeat=n):
            yield "".join(t)
