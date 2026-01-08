from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional
import json


SPECIAL_TOKENS = ["<PAD>", "<START>", "<END>"]
PAD, START, END = 0, 1, 2


@dataclass
class SmilesCharTokenizer:
    """Simple character level tokenizer with fixed special token indices."""
    char_to_idx: Dict[str, int]
    idx_to_char: Dict[int, str]

    @classmethod
    def from_smiles(cls, smiles: Sequence[str]) -> "SmilesCharTokenizer":
        chars = sorted(set("".join(smiles)))
        char_to_idx: Dict[str, int] = {SPECIAL_TOKENS[0]: PAD, SPECIAL_TOKENS[1]: START, SPECIAL_TOKENS[2]: END}
        # start normal chars after specials
        for c in chars:
            if c in char_to_idx:
                continue
            char_to_idx[c] = len(char_to_idx)
        idx_to_char = {i: c for c, i in char_to_idx.items()}
        return cls(char_to_idx=char_to_idx, idx_to_char=idx_to_char)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_idx)

    def encode(self, smi: str, max_len: int) -> List[int]:
        seq = [START] + [self.char_to_idx.get(ch, PAD) for ch in smi] + [END]
        seq = seq[:max_len]
        if len(seq) < max_len:
            seq = seq + [PAD] * (max_len - len(seq))
        return seq

    def encode_many(self, smiles: Sequence[str], max_len: int):
        import numpy as np
        return np.asarray([self.encode(s, max_len) for s in smiles], dtype="int32")

    def decode(self, idxs: Sequence[int], stop_at_end: bool = True) -> str:
        out = []
        for i in idxs:
            if stop_at_end and i == END:
                break
            if i in (PAD, START, END):
                continue
            out.append(self.idx_to_char.get(int(i), ""))
        return "".join(out)

    def to_json(self, path: str) -> None:
        payload = {"char_to_idx": self.char_to_idx}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "SmilesCharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        char_to_idx = payload["char_to_idx"]
        # normalize keys (json keeps them as strings)
        char_to_idx = {str(k): int(v) for k, v in char_to_idx.items()}
        # ensure specials exist with correct indices
        char_to_idx[SPECIAL_TOKENS[0]] = PAD
        char_to_idx[SPECIAL_TOKENS[1]] = START
        char_to_idx[SPECIAL_TOKENS[2]] = END
        idx_to_char = {i: c for c, i in char_to_idx.items()}
        return cls(char_to_idx=char_to_idx, idx_to_char=idx_to_char)
