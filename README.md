<div align="center">
  
  # 🧬 cpIMPDH-LSTM Pipeline
  
  **de novo molecular generation and focused library design for *Cryptosporidium parvum* IMPDH inhibitors**

  [![Python](https://img.shields.io/badge/Python-3.9%2B-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](https://python.org)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-%23FF6F00?style=flat-square&logo=TensorFlow)](https://tensorflow.org)
  [![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=flat-square&logo=Keras)](https://keras.io)
  [![RDKit](https://img.shields.io/badge/RDKit-2024-3949AB?style=flat-square&logo=rdkit)](https://rdkit.org)
  [![License](https://img.shields.io/badge/License-CC0--1.0-blue?style=flat-square)](LICENSE)

</div>

---



---



---

## 📦 Installation

```bash
git clone https://github.com/misgana30/cpIMPDH-lstm-pipeline.git
cd cpIMPDH-lstm-pipeline
pip install .
```

Or install with pip directly:

```bash
pip install git+https://github.com/misgana30/cpIMPDH-lstm-pipeline.git
```

### Dependencies

| Package | Minimum Version |
|---------|----------------|
| Python | ≥ 3.9 |
| TensorFlow | ≥ 2.15.0 |
| RDKit | 2024+ |
| scikit-learn | ≥ 1.0 |
| SciPy | ≥ 1.7 |
| NumPy | < 2 |
| Pandas | ≥ 1.0 |
| Matplotlib | ≥ 3.0 |
| Joblib | ≥ 1.0 |
| tqdm | ≥ 4.0 |
| h5py | ≥ 3.0 |
| shap | ≥ 0.44 (optional, for QSAR model interpretation) |

---

## 🚀 Usage

### 1. Pretrain on ChEMBL

```bash
python scripts/pretrain.py \
  --smiles_txt datasets/chembl_ro5_smiles.txt \
  --outdir outputs_pretrain \
  --max_len 181 \
  --epochs 50 \
  --batch_size 512 \
  --lr 1e-3
```

### 2. Fine-tune on target SMILES

```bash
python scripts/finetune.py \
  --pretrained Models/chembl_pretrained.h5 \
  --finetune_smiles datasets/IMPDH_augmented_smiles_only.txt \
  --outdir outputs_finetune \
  --max_len 181 \
  --strategy full_ft \
  --epochs 20
```

Available fine-tuning strategies:

| Strategy | Description |
|----------|-------------|
| `full_ft` | Unfreeze all layers, train with exponential LR decay |
| `disc_lr` | Discriminative layer-wise learning rates (low → high) |

### 3. Generate novel molecules

```bash
python scripts/generate.py \
  --model Models/chembl_pretrained.h5 \
  --vocab outputs_pretrain/vocab.json \
  --n 10000 \
  --temperature 0.7 \
  --out generated_smiles.csv
```

### 4. Fine-tune + Generate (end-to-end)

```bash
python scripts/finetune_and_generate.py \
  --pretrained Models/chembl_pretrained.h5 \
  --finetune_smiles datasets/IMPDH_augmented_smiles_only.txt \
  --outdir generated \
  --strategy disc_lr \
  --n 10000 \
  --temperature 0.6 \
  --epochs 8
```

---

## 🧪 Python API

```python
from cpimpdh.tokenizer import SmilesCharTokenizer
from cpimpdh.model import build_pretrained_lstm, load_pretrained
from cpimpdh.data import load_smiles_txt, encode_smiles, make_xy
from cpimpdh.train import compile_model, fit
from cpimpdh.generation import generate_many
from cpimpdh.metrics import calc_metrics

# Load and tokenize SMILES
smiles = load_smiles_txt("datasets/chembl_ro5_smiles.txt")
tokenizer = SmilesCharTokenizer.from_smiles(smiles)

# Build and train model
seqs = encode_smiles(smiles, tokenizer, max_len=181)
x, y = make_xy(seqs)
model = build_pretrained_lstm(vocab_size=tokenizer.vocab_size, max_len=181)
model = compile_model(model)
history = fit(model, x, y, epochs=30, out_model_path="best_model.h5")

# Generate molecules
generated = generate_many(model, tokenizer, n=1000, max_len=model.input_shape[1])

# Calculate molecular metrics
metrics = calc_metrics(generated)
print(metrics)


```

---



The same workflow is available from the command line:

```bash
python -m cpimpdh.qsar datasets/finetune_pic50.txt --plot comparison.png --shap
```

(`finetune_pic50.txt` is a two-column, whitespace-separated `SMILES pIC50` file
you derive once from the CSV above with `load_smiles_pic50_txt`/a small export
script — this repo does not ship a pre-augmented pIC50 file, deliberately, so
there is no risk of accidentally training on augmented/leaked data.)

---

## 📁 Project Structure

```
cpIMPDH-lstm-pipeline/
├── cpimpdh/                    # Core Python package
│   ├── tokenizer.py            # SMILES character-level tokenizer
│   ├── model.py                # LSTM model builder + Keras compat loader
│   ├── data.py                 # SMILES I/O and encoding utilities
│   ├── train.py                # Training loop with callbacks
│   ├── generation.py           # Temperature-controlled SMILES generation
│   ├── metrics.py              # Molecular property calculations
│   ├── finetune.py             # Transfer learning strategies
│   ├── qsar.py                 # scaffold-split QSAR: benchmark, CV, y-scrambling, AD, SHAP
│   └── callbacks.py            # Training callbacks (generation monitor)
├── scripts/
│   ├── pretrain.py             # Pretraining on large SMILES corpus
│   ├── finetune.py             # Fine-tuning on target molecules
│   ├── generate.py             # Generate SMILES from trained model
│   └── finetune_and_generate.py# Combined fine-tune + generation pipeline
├── datasets/
│   ├── chembl_ro5_smiles.txt   # ~1,200 ChEMBL Rule-of-5 SMILES (pretrain)
│   ├── IMPDH_augmented_smiles_only.txt      # Augmented 236 unique cpIMPDH-targeted SMILES (for LSTM fine-tuning only)
│   └── finetune_IMPDH_with_chembl_id.csv # the same 236 compounds with ChEMBL IDs and raw IC50 (nM);
│                                          
├── Models/
│   ├── chembl_pretrained.h5    # Pretrained ChEMBL model (vocab: 50)
│   ├── DiscLR.h5               # Discriminative LR fine-tuned model
│   └── Full_FT.h5              # Full fine-tuned model
├── pyproject.toml
├── setup.py
└── requirements.txt
```

---

## 📊 Molecular Metrics

Generated molecules are evaluated on:

| Metric | Description |
|--------|-------------|
| **Validity %** | Fraction of valid RDKit-parsable SMILES |
| **Uniqueness %** | Fraction of unique valid molecules |
| **Scaffold %** | Fraction of unique Murcko scaffolds |
| **QED** | Quantitative Estimate of Drug-likeness |
| **MW** | Molecular weight (Da) |
| **cLogP** | Calculated octanol/water partition coefficient |
| **SA Score** | Synthetic accessibility score |
| **Novelty** | 1 − max(Tanimoto similarity to training set) |

---



Featurization uses **Morgan fingerprints** (radius=2, 2048 bits) with optional **standard scaling**.



## 📄 License

This project is licensed under the [CC0-1.0 License](LICENSE) 
