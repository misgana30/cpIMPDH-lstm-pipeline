from __future__ import annotations

from typing import List, Optional
import tensorflow as tf
from rdkit import Chem

from .tokenizer import SmilesCharTokenizer
from .generation import generate_many

def _is_valid(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None

class SmilesGenerationCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        tokenizer: SmilesCharTokenizer,
        max_len: int,
        temps: Optional[List[float]] = None,
        num_samples: int = 200,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.temps = temps or [0.7]
        self.num_samples = int(num_samples)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for t in self.temps:
            smi = generate_many(self.model, self.tokenizer, n=self.num_samples, max_len=self.max_len, temperature=float(t))
            valid = [s for s in smi if _is_valid(s)]
            validity = (len(valid) / max(1, len(smi))) * 100.0
            uniq = len(set(valid)) / max(1, len(valid)) * 100.0
            print(f"\n[epoch {epoch+1}] temp={t:.2f} | validity={validity:.1f}% | uniqueness={uniq:.1f}% | n_valid={len(valid)}")
