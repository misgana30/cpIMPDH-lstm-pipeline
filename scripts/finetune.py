#!/usr/bin/env python
from __future__ import annotations

import argparse, os
import numpy as np
import tensorflow as tf

from cpimpdh.tokenizer import SmilesCharTokenizer
from cpimpdh.data import load_smiles_txt, encode_smiles, make_xy
from cpimpdh.finetune import finetune_full_unfreeze, finetune_discriminative_lr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True, help="Path to pretrained Keras .h5 model")
    ap.add_argument("--finetune_smiles", required=True, help="SMILES .txt or .csv (one SMILES per line/row) for fine-tuning")
    ap.add_argument("--outdir", default="outputs_finetune")
    ap.add_argument("--strategy", choices=["full_ft", "disc_lr"], default="full_ft")
    ap.add_argument("--max_len", type=int, default=181, help="Max token length used during training data encoding (incl specials)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    ap.add_argument("--layer_lrs", type=str, default="1e-5,3e-5,1e-4", help="Disc-LR comma-separated LRs (bottom->top)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.pretrained):
        raise SystemExit(f"Error: Pretrained model not found: {args.pretrained}")
    if not os.path.exists(args.finetune_smiles):
        raise SystemExit(f"Error: Fine-tuning SMILES file not found: {args.finetune_smiles}")

    # Load finetune SMILES
    try:
        if args.finetune_smiles.lower().endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(args.finetune_smiles)
            if "smiles" in df.columns:
                ft_smiles = df["smiles"].dropna().astype(str).tolist()
            else:
                ft_smiles = df.iloc[:,0].dropna().astype(str).tolist()
        else:
            ft_smiles = load_smiles_txt(args.finetune_smiles)
    except Exception as e:
        raise SystemExit(f"Error loading fine-tuning SMILES: {e}")

    if not ft_smiles:
        raise SystemExit("No SMILES found for fine-tuning.")

    try:
        tokenizer = SmilesCharTokenizer.from_smiles(ft_smiles)
        tokenizer.to_json(os.path.join(args.outdir, "vocab.json"))
    except Exception as e:
        raise SystemExit(f"Error creating tokenizer: {e}")

    try:
        model = tf.keras.models.load_model(args.pretrained, compile=False)
    except Exception as e:
        raise SystemExit(f"Error loading pretrained model: {e}")

    # Encode SMILES for fine-tuning
    from cpimpdh.data import encode_smiles, make_xy
    seqs = encode_smiles(ft_smiles, tokenizer, max_len=args.max_len)
    X_ft, y_ft = make_xy(seqs)

    if args.strategy == "full_ft":
        result = finetune_full_unfreeze(
            pretrained=model,
            X_ft=X_ft,
            y_ft=y_ft,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
        result.model.save(os.path.join(args.outdir, "best_finetuned_full_ft.h5"))
    else:
        lrs = [float(x) for x in args.layer_lrs.split(",")]
        if len(lrs) >= 2:
            lr_low, lr_high = lrs[0], lrs[-1]
        else:
            lr_low, lr_high = lrs[0] if lrs else 1e-5, args.lr
        result = finetune_discriminative_lr(
            pretrained=model,
            X_ft=X_ft,
            y_ft=y_ft,
            lr_low=lr_low,
            lr_high=lr_high,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
        result.model.save(os.path.join(args.outdir, "best_finetuned_disc_lr.h5"))

    print("Done.")

if __name__ == "__main__":
    main()
