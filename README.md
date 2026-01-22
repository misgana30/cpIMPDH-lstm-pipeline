# cpIMPDH SMILES LSTM Pipeline 


## Install
```bash
pip install -r requirements.txt
```

> Tip: if you see NumPy / RDKit binary issues, pin NumPy:
> `numpy<2`

## 1) Pretrain
```bash
python scripts/pretrain.py   --smiles_txt datasets/chembl_ro5_smiles.txt   --outdir outputs_pretrain   --max_len 181   --epochs 50
```



## 2) Fine-tune (FULL FT or Disc-LR)
```bash
# FULL FT
python scripts/finetune.py   --pretrained outputs_pretrain/best_pretrained.h5   --finetune_smiles datasets/augmented_smiles_only.csv   --outdir outputs_finetune_full   --strategy full_ft

# Disc-LR
python scripts/finetune.py   --pretrained outputs_pretrain/best_pretrained.h5   --finetune_smiles datasets/augmented_smiles_only.csv   --outdir outputs_finetune_disc   --strategy disc_lr   --layer_lrs 1e-5,3e-5,1e-4
```

## 3) Generate
```bash
python scripts/generate.py   --model Models/Full_FT.h5   --vocab outputs_finetune_full/vocab.json   --n 10000   --temperature 0.7   --out gen.csv
```


