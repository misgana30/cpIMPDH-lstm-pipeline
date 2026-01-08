from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

import joblib


# -----------------------------
# Featurization
# -----------------------------
def get_morgan_fingerprint(
    smiles: str, radius: int = 2, nbits: int = 2048
) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_list_to_fps(
    smiles: Iterable[str], radius: int = 2, nbits: int = 2048
) -> Tuple[np.ndarray, List[int]]:
    fps = []
    kept = []
    for i, s in enumerate(smiles):
        fp = get_morgan_fingerprint(s, radius=radius, nbits=nbits)
        if fp is None:
            continue
        fps.append(fp)
        kept.append(i)
    if not fps:
        raise ValueError("No valid SMILES were featurized (all invalid?).")
    return np.asarray(fps, dtype=np.float32), kept


# -----------------------------
# Benchmarking models
# -----------------------------
@dataclass
class ModelMetrics:
    mse: float
    r2: float


@dataclass
class ModelFitResult:
    name: str
    model: object
    metrics: ModelMetrics


@dataclass
class BenchmarkResult:
    results: Dict[str, ModelFitResult]
    best_model_name: str
    scaler: Optional[StandardScaler]
    kept_indices: List[int]
    X_test: np.ndarray
    y_test: np.ndarray


def default_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression (RBF)": SVR(kernel="rbf"),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        ),
    }


def benchmark_regressors(
    smiles: List[str],
    y: np.ndarray,
    *,
    radius: int = 2,
    nbits: int = 2048,
    scale: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    models: Optional[Dict[str, object]] = None,
) -> BenchmarkResult:
    if models is None:
        models = default_models(random_state=random_state)

    X, kept = smiles_list_to_fps(smiles, radius=radius, nbits=nbits)
    y_kept = np.asarray(y, dtype=np.float32)[kept]

    scaler = None
    X_use = X
    if scale:
        scaler = StandardScaler()
        X_use = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_use, y_kept, test_size=test_size, random_state=random_state
    )

    results: Dict[str, ModelFitResult] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = float(mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        results[name] = ModelFitResult(name=name, model=model, metrics=ModelMetrics(mse=mse, r2=r2))

    best_name = min(results.keys(), key=lambda k: results[k].metrics.mse)

    return BenchmarkResult(
        results=results,
        best_model_name=best_name,
        scaler=scaler,
        kept_indices=kept,
        X_test=X_test,
        y_test=y_test,
    )


# -----------------------------
# Plotting / Saving
# -----------------------------
def plot_model_comparison(
    benchmark: BenchmarkResult,
    *,
    out_path: Optional[str] = None,
    show: bool = False,
    title: str = "Comparison of Regression Models",
):
    import matplotlib.pyplot as plt  # local import

    y_test = benchmark.y_test
    X_test = benchmark.X_test

    plt.figure(figsize=(8, 6))
    for name, fit in benchmark.results.items():
        y_pred = fit.model.predict(X_test)
        plt.scatter(
            y_test,
            y_pred,
            alpha=0.6,
            label=f"{name} (R2={fit.metrics.r2:.2f})",
        )

    best_fit = benchmark.results[benchmark.best_model_name]
    y_pred_best = best_fit.model.predict(X_test)
    plt.scatter(y_test, y_pred_best, label=f"Best Model ({benchmark.best_model_name})", s=50)

    mn, mx = float(np.min(y_test)), float(np.max(y_test))
    plt.plot([mn, mx], [mn, mx], "k--", lw=2)
    plt.xlabel("Actual pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if out_path:
        # Ensure extension
        if not (out_path.lower().endswith(".png") or out_path.lower().endswith(".pdf")):
            out_path = out_path + ".png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------
# Prediction helper
# -----------------------------
def predict_pic50(
    smiles_list: List[str],
    trained_model,
    *,
    radius: int = 2,
    nbits: int = 2048,
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    X, _ = smiles_list_to_fps(smiles_list, radius=radius, nbits=nbits)
    if scaler is not None:
        X = scaler.transform(X)
    return trained_model.predict(X)


# -----------------------------
# Convenience I/O
# -----------------------------
def load_augmented_smiles_pic50_txt(path: str) -> Tuple[List[str], np.ndarray]:
    smiles = []
    y = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            s, v = parts[0], parts[1]
            try:
                y_val = float(v)
            except ValueError:
                continue
            smiles.append(s)
            y.append(y_val)
    if not smiles:
        raise ValueError(f"No valid (SMILES, pIC50) pairs found in: {path}")
    return smiles, np.asarray(y, dtype=np.float32)


def save_model(path: str, model) -> None:
    joblib.dump(model, path)
