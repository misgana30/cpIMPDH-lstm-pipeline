from __future__ import annotations

from typing import Iterable, List, Dict, Optional, Sequence
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED, Descriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold


def is_valid_smiles(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)


def murcko_scaffolds(smiles: Sequence[str]) -> List[str]:
    scafs = set()
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if not m:
            continue
        core = MurckoScaffold.GetScaffoldForMol(m)
        if core:
            scafs.add(Chem.MolToSmiles(core))
    return sorted(scafs)


def try_sa_score(mol) -> Optional[float]:
    try:
        from rdkit.Chem import RDConfig
        import os, sys
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def calc_metrics(smiles: Sequence[str], train_fps: Optional[Sequence] = None) -> Dict[str, float]:
    smiles = list(smiles)
    valid = [s for s in smiles if is_valid_smiles(s)]
    uniq = set(valid)
    scafs = murcko_scaffolds(valid)

    qed_vals, mw_vals, logp_vals, sa_vals, novelty_vals = [], [], [], [], []

    for s in valid:
        m = Chem.MolFromSmiles(s)
        if not m:
            continue
        qed_vals.append(QED.qed(m))
        mw_vals.append(Descriptors.MolWt(m))
        logp_vals.append(Crippen.MolLogP(m))

        sa = try_sa_score(m)
        if sa is not None:
            sa_vals.append(sa)

        if train_fps is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, list(train_fps))
            novelty_vals.append(1.0 - max(sims) if sims else 1.0)

    out = {
        "n_total": float(len(smiles)),
        "n_valid": float(len(valid)),
        "validity_%": 100.0 * len(valid) / len(smiles) if smiles else 0.0,
        "uniqueness_%": 100.0 * len(uniq) / len(valid) if valid else 0.0,
        "scaffold_%": 100.0 * len(scafs) / len(valid) if valid else 0.0,
        "qed_mean": float(np.mean(qed_vals)) if qed_vals else 0.0,
        "mw_mean": float(np.mean(mw_vals)) if mw_vals else 0.0,
        "clogp_mean": float(np.mean(logp_vals)) if logp_vals else 0.0,
        "sa_mean": float(np.mean(sa_vals)) if sa_vals else 0.0,
        "novelty_mean": float(np.mean(novelty_vals)) if novelty_vals else 0.0,
    }
    return out
