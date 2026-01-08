from __future__ import annotations

from typing import Optional
import numpy as np


def sample_from_probs(probs: np.ndarray, temperature: float = 1.0, rng: Optional[np.random.Generator] = None) -> int:
    if rng is None:
        rng = np.random.default_rng()
    probs = np.asarray(probs, dtype=np.float64)
    # avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    if temperature <= 0:
        # greedy
        return int(np.argmax(probs))
    logits = np.log(probs) / float(temperature)
    exp = np.exp(logits - np.max(logits))
    p = exp / np.sum(exp)
    return int(rng.choice(len(p), p=p))


def generate_smiles_keras(
    model,
    tokenizer,
    max_len: int,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> str:
    if rng is None:
        rng = np.random.default_rng()

    PAD = tokenizer.char_to_idx["<PAD>"]
    START = tokenizer.char_to_idx["<START>"]
    END = tokenizer.char_to_idx["<END>"]

    sequence = [START]
    out_chars = []

    for _ in range(max_len - 1):
        padded = sequence + [PAD] * (max_len - len(sequence))
        x = np.array(padded, dtype="int32").reshape(1, -1)

        y = model.predict(x, verbose=0)[0]              # [L, vocab]
        step = len(sequence) - 1
        probs = y[step]
        idx = sample_from_probs(probs, temperature=temperature, rng=rng)

        if idx in (END, PAD):
            break
        out_chars.append(tokenizer.idx_to_char.get(idx, ""))
        sequence.append(idx)

    return "".join(out_chars)


def generate_many(
    model,
    tokenizer,
    n: int,
    max_len: int,
    temperature: float = 1.0,
    seed: Optional[int] = None,
):
    rng = np.random.default_rng(seed)
    return [generate_smiles_keras(model, tokenizer, max_len=max_len, temperature=temperature, rng=rng) for _ in range(n)]
