# cpIMPDH SMILES LSTM Pipeline (Pretrain + Fine-tune)

A reproducible, reviewer-friendly codebase for:
1. **Pretraining** a character-level LSTM on large SMILES corpora
2. **Fine-tuning** on cpIMPDH-focused compounds using:
   - **FULL FT** (full unfreeze, low learning rate)
   - **Disc-LR** (discriminative learning rates)
3. **Generation** of new SMILES + basic RDKit evaluation

## Folder structure
- `cpimpdh/` : reusable modules (tokenizer, model, training, fine-tuning, metrics)
- `scripts/` : command-line entry points
- `notebooks/` : optional (keep demo notebooks only)

## Install
```bash
pip install -r requirements.txt
```

> Tip: if you see NumPy / RDKit binary issues, pin NumPy:
> `numpy<2`

## 1) Pretrain
```bash
python scripts/pretrain.py   --smiles_txt ro5_fil_smiles.txt   --outdir outputs_pretrain   --max_len 181   --epochs 50
```

Outputs:
- `outputs_pretrain/best_pretrained.h5`
- `outputs_pretrain/vocab.json`

## 2) Fine-tune (FULL FT or Disc-LR)
```bash
# FULL FT
python scripts/finetune.py   --pretrained outputs_pretrain/best_pretrained.h5   --finetune_smiles augmented_smiles_only.csv   --outdir outputs_finetune_full   --strategy full_ft

# Disc-LR
python scripts/finetune.py   --pretrained outputs_pretrain/best_pretrained.h5   --finetune_smiles augmented_smiles_only.csv   --outdir outputs_finetune_disc   --strategy disc_lr   --layer_lrs 1e-5,3e-5,1e-4
```

## 3) Generate
```bash
python scripts/generate.py   --model outputs_finetune_full/best_finetuned_full_ft.h5   --vocab outputs_finetune_full/vocab.json   --n 10000   --temperature 0.7   --out gen.csv
```

## Notes
- This repo is **Open-Source-tool friendly** (no Schrödinger-specific code).
- Keep large datasets and trained weights out of Git history (`.gitignore` already helps).

## License
MIT (edit if you prefer)
