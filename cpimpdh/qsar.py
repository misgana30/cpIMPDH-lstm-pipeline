

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

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


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    """Canonical Bemis-Murcko scaffold SMILES for one molecule (None if invalid)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold is not None else ""


def smiles_list_to_scaffolds(smiles: Iterable[str]) -> List[Optional[str]]:
    return [get_murcko_scaffold(s) for s in smiles]


# -----------------------------
# Scaffold-based train/test split
# -----------------------------
def scaffold_train_test_split(
    scaffolds: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bemis-Murcko scaffold split. Compounds are grouped by scaffold, and whole
    scaffold groups (never individual compounds) are assigned to train or
    test, so no scaffold is shared between the two sets.

    Scaffold groups are sorted largest-first and assigned to the TEST set
    until the target test size is reached; the remaining, smaller/singleton
    scaffold groups go to the training set. This means the test set can be
    dominated by the single largest scaffold group in the data -- this
    behavior is intentional (it matches the split used for the manuscript)
    and is disclosed as a limitation there, not a bug to "fix" by reversing
    the assignment order.

    Returns
    -------
    train_idx, test_idx : np.ndarray of integer indices into `scaffolds`
    """
    n = len(scaffolds)
    target_test_n = round(n * test_size)

    groups: Dict[str, List[int]] = {}
    for i, sc in enumerate(scaffolds):
        groups.setdefault(sc, []).append(i)

    rng = np.random.RandomState(random_state)
    group_keys = list(groups.keys())
    # Shuffle first so that ties in group size are broken reproducibly but
    # without a fixed alphabetical bias.
    rng.shuffle(group_keys)
    group_keys.sort(key=lambda k: len(groups[k]), reverse=True)

    test_idx: List[int] = []
    train_idx: List[int] = []
    for k in group_keys:
        members = groups[k]
        if len(test_idx) < target_test_n:
            test_idx.extend(members)
        else:
            train_idx.extend(members)

    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


# -----------------------------
# Benchmarking models
# -----------------------------
@dataclass
class ModelMetrics:
    mse: float
    mae: float
    r2: float
    spearman_rho: float
    cv_r2_mean: float = float("nan")
    cv_r2_std: float = float("nan")
    cv_mae_mean: float = float("nan")
    cv_mae_std: float = float("nan")
    cv_pooled_rmse: float = float("nan")


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
    scaffolds: List[str]
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def default_models(random_state: int = 42) -> Dict[str, object]:
    """
    Default model set for the Table 1 comparison. The SVR hyperparameters
    (C=30, gamma=0.0005, epsilon=0.1) are the values selected by
    GroupKFold cross-validation inside the training set and are the ones
    reported as the final model in the manuscript.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression (RBF)": SVR(kernel="rbf", C=30, gamma=0.0005, epsilon=0.1),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        ),
    }


def _clone_fresh(model):
    from sklearn.base import clone
    return clone(model)


def group_kfold_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Scaffold-grouped k-fold cross-validation. `groups` should be an integer
    (or hashable) scaffold id per row of X/y, so folds never split a
    scaffold across train and validation.
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_r2, fold_mae = [], []
    pooled_true, pooled_pred = [], []

    for tr_idx, va_idx in gkf.split(X, y, groups):
        m = _clone_fresh(model)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[va_idx])
        fold_r2.append(r2_score(y[va_idx], pred))
        fold_mae.append(mean_absolute_error(y[va_idx], pred))
        pooled_true.extend(y[va_idx].tolist())
        pooled_pred.extend(pred.tolist())

    pooled_rmse = float(np.sqrt(mean_squared_error(pooled_true, pooled_pred)))
    return {
        "cv_r2_mean": float(np.mean(fold_r2)),
        "cv_r2_std": float(np.std(fold_r2)),
        "cv_mae_mean": float(np.mean(fold_mae)),
        "cv_mae_std": float(np.std(fold_mae)),
        "cv_pooled_rmse": pooled_rmse,
    }


def benchmark_regressors(
    smiles: List[str],
    y: np.ndarray,
    *,
    radius: int = 2,
    nbits: int = 2048,
    scale: bool = True,
    test_size: float = 0.2,
    cv_splits: int = 5,
    random_state: int = 42,
    models: Optional[Dict[str, object]] = None,
) -> BenchmarkResult:
    """
    Fit and compare regressors using a Bemis-Murcko scaffold split plus
    scaffold-grouped cross-validation. `smiles`/`y` must be the unique
    compounds (no SMILES augmentation) to avoid leakage.
    """
    if models is None:
        models = default_models(random_state=random_state)

    X, kept = smiles_list_to_fps(smiles, radius=radius, nbits=nbits)
    y_kept = np.asarray(y, dtype=np.float32)[kept]
    smiles_kept = [smiles[i] for i in kept]
    scaffolds = smiles_list_to_scaffolds(smiles_kept)

    scaler = None
    X_use = X
    if scale:
        scaler = StandardScaler()
        X_use = scaler.fit_transform(X)

    train_idx, test_idx = scaffold_train_test_split(
        scaffolds, test_size=test_size, random_state=random_state
    )
    X_train, X_test = X_use[train_idx], X_use[test_idx]
    y_train, y_test = y_kept[train_idx], y_kept[test_idx]

    # Integer group ids for GroupKFold, restricted to the training set.
    train_scaffolds = [scaffolds[i] for i in train_idx]
    scaffold_to_id = {sc: gid for gid, sc in enumerate(sorted(set(train_scaffolds)))}
    train_groups = np.asarray([scaffold_to_id[sc] for sc in train_scaffolds])

    results: Dict[str, ModelFitResult] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = float(mean_squared_error(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        rho = float(spearmanr(y_test, y_pred).correlation)

        cv_stats = group_kfold_cv(model, X_train, y_train, train_groups, n_splits=cv_splits)

        results[name] = ModelFitResult(
            name=name,
            model=model,
            metrics=ModelMetrics(mse=mse, mae=mae, r2=r2, spearman_rho=rho, **cv_stats),
        )

    best_name = max(results.keys(), key=lambda k: results[k].metrics.r2)

    return BenchmarkResult(
        results=results,
        best_model_name=best_name,
        scaler=scaler,
        kept_indices=kept,
        scaffolds=scaffolds,
        train_idx=train_idx,
        test_idx=test_idx,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def print_table1(benchmark: BenchmarkResult) -> None:
    """Print a Table-1-style summary: test R2/MAE, CV R2/MAE, Spearman rho."""
    header = f"{'Model':32s} {'Test R2':>8s} {'Test MAE':>9s} {'CV R2':>8s} {'CV MAE':>8s} {'Spearman rho':>13s}"
    print(header)
    print("-" * len(header))
    for name, fit in benchmark.results.items():
        m = fit.metrics
        print(
            f"{name:32s} {m.r2:8.3f} {m.mae:9.3f} {m.cv_r2_mean:8.3f} "
            f"{m.cv_mae_mean:8.3f} {m.spearman_rho:13.3f}"
        )


# -----------------------------
# Y-scrambling
# -----------------------------
def y_scrambling_test(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_permutations: int = 100,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Y-scrambling / y-randomization test. The observed scaffold-grouped CV R2
    is compared against the CV R2 obtained after randomly permuting the
    training labels `n_permutations` times. A model that is genuinely
    learning structure-activity signal, rather than overfitting noise,
    should show observed >> scrambled.
    """
    observed = group_kfold_cv(model, X_train, y_train, groups, n_splits=n_splits)["cv_r2_mean"]

    rng = np.random.RandomState(random_state)
    scrambled_scores = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_train)
        score = group_kfold_cv(model, X_train, y_perm, groups, n_splits=n_splits)["cv_r2_mean"]
        scrambled_scores.append(score)

    scrambled_scores = np.asarray(scrambled_scores)
    p_value = float(np.mean(scrambled_scores >= observed))

    return {
        "observed_cv_r2": float(observed),
        "scrambled_cv_r2_mean": float(scrambled_scores.mean()),
        "scrambled_cv_r2_max": float(scrambled_scores.max()),
        "scrambled_cv_r2_std": float(scrambled_scores.std()),
        "n_permutations": n_permutations,
        "p_value": p_value,
    }


# -----------------------------
# Applicability domain
# -----------------------------
def _tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise Tanimoto similarity between two sets of binary fingerprints
    (rows = compounds, cols = bits). Returns an (len(A), len(B)) matrix.
    """
    A = A.astype(bool)
    B = B.astype(bool)
    intersection = A.astype(np.float32) @ B.astype(np.float32).T
    a_counts = A.sum(axis=1, keepdims=True).astype(np.float32)
    b_counts = B.sum(axis=1, keepdims=True).astype(np.float32).T
    union = a_counts + b_counts - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return sim


def mean_top_k_tanimoto(X_query: np.ndarray, X_ref: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean Tanimoto similarity of each query fingerprint to its k nearest
    neighbors in X_ref (fingerprints, not scaled features)."""
    sim = _tanimoto_matrix(X_query, X_ref)
    k = min(k, X_ref.shape[0])
    top_k = np.sort(sim, axis=1)[:, -k:]
    return top_k.mean(axis=1)


def applicability_domain_cutoff(
    X_train: np.ndarray,
    scaffolds_train: List[str],
    k: int = 5,
    percentile: float = 5.0,
) -> float:
    """
    Derive the AD cutoff as the given percentile (default 5th) of the
    mean top-k Tanimoto similarity, computed leave-one-scaffold-out within
    the training set: for each compound, neighbors are drawn only from
    OTHER scaffold groups, so the statistic reflects similarity to
    chemically distinct training compounds rather than to near-duplicates
    from its own scaffold.
    """
    n = X_train.shape[0]
    scaffolds_arr = np.asarray(scaffolds_train)
    stats = np.zeros(n)
    for i in range(n):
        mask = scaffolds_arr != scaffolds_arr[i]
        ref = X_train[mask]
        if ref.shape[0] == 0:
            stats[i] = np.nan
            continue
        stats[i] = mean_top_k_tanimoto(X_train[i : i + 1], ref, k=k)[0]
    stats = stats[~np.isnan(stats)]
    return float(np.percentile(stats, percentile))


def evaluate_applicability_domain(
    X_query: np.ndarray,
    X_train: np.ndarray,
    cutoff: float,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (mean_top_k_similarity, in_domain_bool) for each query compound."""
    sims = mean_top_k_tanimoto(X_query, X_train, k=k)
    return sims, sims >= cutoff


# -----------------------------
# SHAP interpretation
# -----------------------------
def shap_permutation_importance(
    model,
    X: np.ndarray,
    nbits: int = 2048,
    max_evals: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Permutation SHAP (model-agnostic) on the fitted model, computed over the
    given feature matrix (use the full 236-compound set to match the
    manuscript). Returns the mean |SHAP value| per bit and the ranked top
    features (with sign of the mean signed SHAP value, so a reviewer can see
    whether a bit is activity-promoting or activity-reducing).

    NOTE: permutation SHAP over a 2048-bit fingerprint is slow (minutes to
    hours for 236 compounds, depending on `max_evals` and hardware) -- this
    is inherent to the method on high-dimensional binary features, not a
    bug. Leave `max_evals` unset to use SHAP's own "auto" budget, or pass a
    smaller integer for a quicker, lower-fidelity pass while iterating.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "shap is required for this function: pip install shap"
        ) from e

    background = shap.sample(X, min(100, X.shape[0]), random_state=random_state)

    explainer = shap.explainers.Permutation(
        model.predict, background, seed=random_state
    )
    shap_values = explainer(X, max_evals=(max_evals if max_evals is not None else "auto"))

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    mean_signed = shap_values.values.mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    top_features = [
        {
            "bit": int(idx),
            "mean_abs_shap": float(mean_abs[idx]),
            "mean_signed_shap": float(mean_signed[idx]),
        }
        for idx in order
    ]

    return {
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
        "top_features": top_features,
    }


# -----------------------------
# Plotting / Saving
# -----------------------------
def plot_model_comparison(
    benchmark: BenchmarkResult,
    *,
    out_path: Optional[str] = None,
    show: bool = False,
    title: str = "Comparison of Regression Models (scaffold-held-out test set)",
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
            label=f"{name} (test R2={fit.metrics.r2:.2f}, CV R2={fit.metrics.cv_r2_mean:.2f})",
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


def predict_pic50_with_ad(
    smiles_list: List[str],
    trained_model,
    *,
    X_train_fp: np.ndarray,
    ad_cutoff: float,
    radius: int = 2,
    nbits: int = 2048,
    scaler: Optional[StandardScaler] = None,
    k: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Predict pIC50 and flag whether each query compound falls inside the
    applicability domain of the training set. Predictions for out-of-domain
    compounds should be treated as unreliable, per the manuscript.
    """
    X_fp, kept = smiles_list_to_fps(smiles_list, radius=radius, nbits=nbits)
    sims, in_domain = evaluate_applicability_domain(X_fp, X_train_fp, ad_cutoff, k=k)

    X_use = X_fp
    if scaler is not None:
        X_use = scaler.transform(X_fp)
    preds = trained_model.predict(X_use)

    return {
        "kept_indices": np.asarray(kept),
        "predicted_pic50": preds,
        "mean_top_k_tanimoto": sims,
        "in_domain": in_domain,
    }


# -----------------------------
# Convenience I/O
# -----------------------------
def load_smiles_pic50_txt(path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load (SMILES, pIC50) pairs, one per line, whitespace-separated. Point
    this at the 236 UNIQUE fine-tuning compounds only -- do not feed it the
    SMILES-augmented set used for LSTM fine-tuning, since augmentation
    produces multiple SMILES per molecule and would leak across any split.
    """
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


# Backward-compatible alias for the old name.
load_augmented_smiles_pic50_txt = load_smiles_pic50_txt


def save_model(path: str, model) -> None:
    joblib.dump(model, path)


# -----------------------------
# Example end-to-end usage
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scaffold-split QSAR benchmark for cpIMPDH pIC50 prediction."
    )
    parser.add_argument(
        "data_path",
        help="Whitespace-separated (SMILES, pIC50) file, one unique compound per line.",
    )
    parser.add_argument("--plot", default=None, help="Optional path to save the comparison plot.")
    parser.add_argument("--shap", action="store_true", help="Also run permutation SHAP on the SVR model.")
    args = parser.parse_args()

    smiles, y = load_smiles_pic50_txt(args.data_path)
    bench = benchmark_regressors(smiles, y)

    print_table1(bench)
    print(f"\nBest model by test R2: {bench.best_model_name}")

    svr_fit = bench.results["Support Vector Regression (RBF)"]
    train_scaffolds = [bench.scaffolds[i] for i in bench.train_idx]
    scaffold_to_id = {sc: gid for gid, sc in enumerate(sorted(set(train_scaffolds)))}
    train_groups = np.asarray([scaffold_to_id[sc] for sc in train_scaffolds])

    ys = y_scrambling_test(svr_fit.model, bench.X_train, bench.y_train, train_groups)
    print(
        f"\nY-scrambling (SVR): observed CV R2 = {ys['observed_cv_r2']:.3f}, "
        f"scrambled mean = {ys['scrambled_cv_r2_mean']:.3f}, "
        f"max = {ys['scrambled_cv_r2_max']:.3f}, p = {ys['p_value']:.4f}"
    )

    cutoff = applicability_domain_cutoff(bench.X_train, train_scaffolds)
    sims, in_domain = evaluate_applicability_domain(bench.X_test, bench.X_train, cutoff)
    print(
        f"\nApplicability domain cutoff = {cutoff:.3f}; "
        f"{in_domain.sum()}/{len(in_domain)} test compounds in-domain "
        f"({100 * in_domain.mean():.1f}%)"
    )

    if args.plot:
        plot_model_comparison(bench, out_path=args.plot)

    if args.shap:
        X_all, _ = smiles_list_to_fps(smiles)
        if bench.scaler is not None:
            X_all = bench.scaler.transform(X_all)
        shap_res = shap_permutation_importance(svr_fit.model, X_all)
        print("\nTop 10 SHAP bits (SVR):")
        for feat in shap_res["top_features"][:10]:
            print(
                f"  bit {feat['bit']:5d}  mean|SHAP| = {feat['mean_abs_shap']:.4f}  "
                f"sign = {'+' if feat['mean_signed_shap'] >= 0 else '-'}"
            )
