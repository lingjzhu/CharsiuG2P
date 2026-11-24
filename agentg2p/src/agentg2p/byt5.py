# -*- coding: utf-8 -*-
"""ByT5 G2P models (needs the optional torch + transformers dependencies).

The models are word-level ByT5 models (7.2M parameters, trained from scratch
following CharsiuG2P) from the AgentG2P project. Inputs must match training exactly:
the UTF-8 bytes of "<lang><source_tag>:" plus the word, each byte + 3, and no
EOS. Decoding matches the training project's evaluation: 5 beams, no sampling, at most
min(512, 4 * input length + 32) new tokens; outputs are bytes + 3 up to EOS,
split on spaces into phones.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

PAD_ID, EOS_ID = 0, 1
BYTE_OFFSET = 3
GEN_CAP = 512


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def encode_input(lang: str, source_tag: str, word: str) -> List[int]:
    prefix = f"<{lang}><{source_tag}>:".encode("utf-8")
    return [b + BYTE_OFFSET for b in prefix + word.encode("utf-8")]


def decode_output(ids: Sequence[int]) -> str:
    out = bytearray()
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i >= BYTE_OFFSET:
            out.append(i - BYTE_OFFSET)
    return out.decode("utf-8", errors="ignore")


def max_new_tokens(input_len: int, cap: int = GEN_CAP) -> int:
    return min(cap, 4 * input_len + 32)


class ByT5Model:
    def __init__(self, model_dir: Union[str, Path], device: Optional[str] = None,
                 beams: int = 5, batch_size: int = 64):
        if not available():
            raise ImportError("ByT5 needs torch and transformers: pip install 'agentg2p[neural]'")
        import torch
        from transformers import T5ForConditionalGeneration
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
        except (ImportError, AttributeError):
            pass
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = T5ForConditionalGeneration.from_pretrained(str(model_dir)).to(self.device).eval()
        self.beams = beams
        self.batch_size = batch_size

    def predict(self, items: Sequence[Tuple[str, str, str]]) -> List[Optional[List[str]]]:
        """Phones for each (lang, source_tag, word); None where the output is empty."""
        torch = self.torch
        encoded = [encode_input(lang, tag, word) for lang, tag, word in items]
        order = sorted(range(len(items)), key=lambda k: len(encoded[k]))
        out: List[Optional[List[str]]] = [None] * len(items)
        with torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                chunk = order[start:start + self.batch_size]
                width = max(len(encoded[k]) for k in chunk)
                ids = torch.full((len(chunk), width), PAD_ID, dtype=torch.long)
                for row, k in enumerate(chunk):
                    ids[row, :len(encoded[k])] = torch.tensor(encoded[k], dtype=torch.long)
                ids = ids.to(self.device)
                gen = self.model.generate(input_ids=ids, attention_mask=(ids != PAD_ID).long(),
                                          num_beams=self.beams, do_sample=False,
                                          max_new_tokens=max_new_tokens(width))
                for row, k in enumerate(chunk):
                    phones = decode_output(gen[row].tolist()[1:]).split()
                    out[k] = phones or None
        return out
