#!/usr/bin/env python
from __future__ import annotations

import argparse, os
import tensorflow as tf

from cpimpdh.data import load_smiles_txt, encode_smiles, make_xy
from cpimpdh.tokenizer import SmilesCharTokenizer
from cpimpdh.model import build_pretrained_lstm
from cpimpdh.train import compile_model, fit
from cpimpdh.callbacks import SmilesGenerationCallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles_txt", required=True, help="Path to SMILES .txt (one per line)")
    ap.add_argument("--outdir", default="outputs_pretrain", help="Output directory")
    ap.add_argument("--max_len", type=int, default=181, help="Max token length including <START>/<END>/<PAD>")
    ap.add_argument("--embedding_dim", type=int, default=300)
    ap.add_argument("--lstm_units", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--callback_samples", type=int, default=200)
    ap.add_argument("--callback_temp", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.smiles_txt):
        raise SystemExit(f"Error: Input file not found: {args.smiles_txt}")

    try:
        smiles = load_smiles_txt(args.smiles_txt)
    except Exception as e:
        raise SystemExit(f"Error loading SMILES file: {e}")

    if not smiles:
        raise SystemExit("No SMILES found in input file.")

    # Tokenizer
    tok = SmilesCharTokenizer.from_smiles(smiles)
    vocab_path = os.path.join(args.outdir, "vocab.json")
    tok.to_json(vocab_path)

    # Data
    seqs = encode_smiles(smiles, tok, max_len=args.max_len)
    x, y = make_xy(seqs)

    # Model
    model = build_pretrained_lstm(
        vocab_size=tok.vocab_size,  # Property, not method
        max_len=args.max_len - 1,  # X length (teacher forcing)
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
    )
    model = compile_model(model, learning_rate=args.lr)

    # Callback (optional)
    cb = SmilesGenerationCallback(tok, max_len=args.max_len, temps=[args.callback_temp], num_samples=args.callback_samples)

    best_path = os.path.join(args.outdir, "best_pretrained.h5")
    history = fit(
        model,
        x, y,
        out_path=best_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[cb],
    )

    print(f"Saved best model to: {best_path}")
    print(f"Saved vocab to: {vocab_path}")

if __name__ == "__main__":
    main()
