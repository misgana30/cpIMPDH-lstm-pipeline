#!/usr/bin/env python3
"""
CLI: fine-tune a pretrained Keras SMILES model on a cpIMPDH-focused SMILES list and generate a library.

Example:
python scripts/finetune_and_generate.py \
  --pretrained best_march.h5 \
  --finetune_smiles augmented_smiles_only.csv \
  --outdir generated \
  --strategy disc_lr \
  --n 10000 \
  --temperature 0.6 \
  --max_len 181
"""
from __future__ import annotations

import argparse, os, json
import numpy as np
import tensorflow as tf

from cpimpdh.tokenizer import SmilesCharTokenizer
from cpimpdh.finetune import finetune_full_unfreeze, finetune_discriminative_lr
from cpimpdh.generation import generate_many
from cpimpdh.metrics import calc_metrics, morgan_fp


def load_smiles(path: str):
    smiles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split(",")[0]  # safe for one-col csv-ish files
            if s:
                smiles.append(s)
    return smiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True, help="Path to pretrained .h5 Keras model")
    ap.add_argument("--finetune_smiles", required=True, help="Path to SMILES file (one per line or first column)")
    ap.add_argument("--outdir", default="generated")
    ap.add_argument("--strategy", choices=["full", "disc_lr"], default="disc_lr")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_len", type=int, default=181)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pretrained = tf.keras.models.load_model(args.pretrained, compile=False)

    ft_smiles = load_smiles(args.finetune_smiles)
    tokenizer = SmilesCharTokenizer.from_smiles(ft_smiles)

    encoded = tokenizer.encode_many(ft_smiles, args.max_len)
    X_ft, y_ft = encoded[:, :-1], encoded[:, 1:]

    if args.strategy == "full":
        res = finetune_full_unfreeze(pretrained, X_ft, y_ft)
        tag = "full_unfreeze"
    else:
        res = finetune_discriminative_lr(pretrained, X_ft, y_ft)
        tag = "disc_lr"

    model = res.model

    if args.save_model:
        model.save(os.path.join(args.outdir, f"{tag}.h5"))

    L = int(model.input_shape[1])
    gen = generate_many(model, tokenizer, n=args.n, max_len=L, temperature=args.temperature, seed=args.seed)
    out_smi = os.path.join(args.outdir, f"{tag}_T{args.temperature}.smi")
    with open(out_smi, "w", encoding="utf-8") as f:
        f.write("\n".join(gen))

    train_fps = [morgan_fp(s) for s in ft_smiles]
    train_fps = [fp for fp in train_fps if fp is not None]
    metrics = calc_metrics(gen, train_fps=train_fps)
    metrics["strategy"] = tag
    metrics["temperature"] = args.temperature

    with open(os.path.join(args.outdir, f"{tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", out_smi)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
