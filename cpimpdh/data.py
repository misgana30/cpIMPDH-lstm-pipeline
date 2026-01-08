from __future__ import annotations

from typing import List, Tuple, Union, Dict
import numpy as np

from .tokenizer import SmilesCharTokenizer

def load_smiles_txt(path: str) -> List[str]:
    smiles: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                smiles.append(s)
    return smiles

def max_smiles_len(smiles: List[str]) -> int:
    return max((len(s) for s in smiles), default=0)

def encode_smiles(
    smiles: List[str],
    tokenizer_or_char_to_idx: Union[SmilesCharTokenizer, Dict[str, int]],
    max_len: int,
) -> np.ndarray:
    if isinstance(tokenizer_or_char_to_idx, SmilesCharTokenizer):
        tok = tokenizer_or_char_to_idx
        seqs = tok.encode_many(smiles, max_len=max_len)
        return np.asarray(seqs, dtype=np.int32)

    # Backward-compatible path (char_to_idx dict)
    char_to_idx = tokenizer_or_char_to_idx
    PAD = char_to_idx.get("<PAD>", 0)
    out = []
    for s in smiles:
        seq = [char_to_idx["<START>"]] + [char_to_idx[ch] for ch in s] + [char_to_idx["<END>"]]
        if len(seq) < max_len:
            seq = seq + [PAD] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        out.append(seq)
    return np.asarray(out, dtype=np.int32)

def make_xy(seqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = seqs[:, :-1]
    y = seqs[:, 1:]
    return x, y
