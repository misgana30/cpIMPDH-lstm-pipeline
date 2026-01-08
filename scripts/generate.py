#!/usr/bin/env python
from __future__ import annotations

import argparse, os
import pandas as pd
import tensorflow as tf

from cpimpdh.tokenizer import SmilesCharTokenizer
from cpimpdh.generation import generate_many

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to Keras .h5 model")
    ap.add_argument("--vocab", required=True, help="Path to vocab.json (from tokenizer.to_json)")
    ap.add_argument("--n", type=int, default=10000, help="Number of SMILES to generate")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_len", type=int, default=181, help="Max token length including specials")
    ap.add_argument("--out", default="generated_smiles.csv")
    args = ap.parse_args()

    tok = SmilesCharTokenizer.from_json(args.vocab)
    model = tf.keras.models.load_model(args.model, compile=False)

    L = int(model.input_shape[1])
    smiles = generate_many(model, tok, n=args.n, max_len=L, temperature=args.temperature)
    df = pd.DataFrame({"smiles": smiles})
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
