"""
Synthetic Dataset Generator for Selective Prediction Experiments
=================================================================
Generates controlled binary classification datasets varying:
  - tau    : class prior (= minority class proportion = decision threshold)
  - d      : class separability (mean difference in std units, i.e. effect size)
  - n      : total sample size

Output structure:
  synthetic_data/
    tau_0.10_d_0.5_n_300/
        X.csv   (features, no header)
        y.csv   (binary labels, single column)
    tau_0.10_d_0.5_n_1000/
        ...
    ...

Usage:
    python synthetic_generator.py
    
    Or import and call generate_all_conditions() / generate_single_condition()
"""

import numpy as np
import pandas as pd
import os
from itertools import product


# ─────────────────────────────────────────────
# EXPERIMENT GRID
# ─────────────────────────────────────────────

TAU_VALUES          = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
SEPARABILITY_VALUES = [0.5, 1.0, 2.0, 3.0]   # mean difference in std units (Cohen's d)
N_VALUES            = [300, 1000]
N_FEATURES          = 20    # fixed across all conditions
RANDOM_SEED         = 42
OUTPUT_DIR          = "../data/synthetic_data"


# ─────────────────────────────────────────────
# CORE GENERATOR
# ─────────────────────────────────────────────

def generate_single_condition(
    tau: float,
    d: float,
    n: int,
    n_features: int = N_FEATURES,
    seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a single synthetic binary classification dataset.

    Parameters
    ----------
    tau : float
        Minority class proportion in (0, 0.5].
        Also interpreted as the decision threshold for downstream evaluation.
        Class 1 (progressor/minority) has proportion tau.
        Class 0 (stable/majority)    has proportion 1 - tau.

    d : float
        Class separability: distance between class means in units of
        standard deviation (Cohen's d). 
        d = 0.5 → heavy overlap (hard task)
        d = 1.0 → moderate overlap
        d = 2.0 → good separation
        d = 3.0 → near-complete separation (easy task)

    n : int
        Total number of samples.

    n_features : int
        Number of features. All features are informative to varying degrees.
        First feature carries the full signal; remaining features add
        structured noise to avoid trivial separability at high d.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : pd.DataFrame, shape (n, n_features)
    y : pd.Series, shape (n,), dtype int, values in {0, 1}
    
    Notes
    -----
    - Class 0 (majority): each feature ~ N(0, 1)
    - Class 1 (minority): feature_1 ~ N(d, 1), remaining ~ N(d * decay, 1)
      where decay decreases exponentially so signal concentrates in
      fewer features as expected in real clinical data.
    - This avoids the degenerate case where all features are equally
      informative, which would make the task unrealistically easy at d >= 2.
    """
    rng = np.random.default_rng(seed)

    n_minority = max(1, int(round(n * tau)))
    n_majority = n - n_minority

    # Signal decay across features: first feature carries full signal,
    # subsequent features carry exponentially less
    decay = np.exp(-0.5 * np.arange(n_features))  # shape (n_features,)

    # Class 0 (majority/stable): centered at 0
    X0 = rng.normal(loc=0.0, scale=1.0, size=(n_majority, n_features))

    # Class 1 (minority/progressor): centered at d * decay per feature
    means_1 = d * decay  # shape (n_features,)
    X1 = rng.normal(loc=means_1, scale=1.0, size=(n_minority, n_features))

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)]).astype(int)

    # Shuffle to avoid any order dependence in downstream pipeline
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="label")


def condition_dirname(tau: float, d: float, n: int) -> str:
    """Consistent directory name for a condition."""
    return f"tau_{tau:.2f}_d_{d:.1f}_n_{n}"


def condition_paths(
    tau: float,
    d: float,
    n: int,
    output_dir: str = OUTPUT_DIR
) -> tuple[str, str]:
    """Return (X_path, y_path) for a given condition."""
    folder = os.path.join(output_dir, condition_dirname(tau, d, n))
    return os.path.join(folder, "X.csv"), os.path.join(folder, "y.csv")


def generate_all_conditions(
    tau_values: list       = TAU_VALUES,
    separability_values: list = SEPARABILITY_VALUES,
    n_values: list         = N_VALUES,
    n_features: int        = N_FEATURES,
    seed: int              = RANDOM_SEED,
    output_dir: str        = OUTPUT_DIR,
    verbose: bool          = True
) -> pd.DataFrame:
    """
    Generate and save all conditions in the experiment grid.

    Returns
    -------
    manifest : pd.DataFrame
        Table with columns [tau, d, n, n_minority, n_majority, X_path, y_path]
        for easy iteration in the classification pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    conditions = list(product(tau_values, separability_values, n_values))
    n_total = len(conditions)

    for i, (tau, d, n) in enumerate(conditions):
        folder = os.path.join(output_dir, condition_dirname(tau, d, n))
        os.makedirs(folder, exist_ok=True)

        # Use a deterministic but varied seed per condition
        condition_seed = seed + int(tau * 1000) + int(d * 100) + n
        X, y = generate_single_condition(tau, d, n, n_features, condition_seed)

        x_path = os.path.join(folder, "X.csv")
        y_path = os.path.join(folder, "y.csv")

        X.to_csv(x_path, index=False)
        y.to_csv(y_path, index=False)

        n_minority = int(y.sum())
        n_majority = len(y) - n_minority

        records.append({
            "tau":        tau,
            "d":          d,
            "n":          n,
            "n_minority": n_minority,
            "n_majority": n_majority,
            "actual_tau": round(n_minority / len(y), 4),
            "X_path":     x_path,
            "y_path":     y_path,
        })

        if verbose:
            print(
                f"[{i+1:>3}/{n_total}] tau={tau:.2f}  d={d:.1f}  "
                f"n={n}  minority={n_minority}  majority={n_majority}"
            )

    manifest = pd.DataFrame(records)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    if verbose:
        print(f"\nDone. {n_total} conditions saved to '{output_dir}/'")
        print(f"Manifest saved to '{manifest_path}'")

    return manifest


# ─────────────────────────────────────────────
# CONVENIENCE: LOAD A CONDITION
# ─────────────────────────────────────────────

def load_condition(
    tau: float,
    d: float,
    n: int,
    output_dir: str = OUTPUT_DIR
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a previously generated condition from disk.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    x_path, y_path = condition_paths(tau, d, n, output_dir)
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()
    return X, y


def load_from_manifest_row(row: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load directly from a manifest row.
    
    Usage
    -----
    manifest = pd.read_csv("synthetic_data/manifest.csv")
    for _, row in manifest.iterrows():
        X, y = load_from_manifest_row(row)
        # pass X_path and y_path to your pipeline
    """
    X = pd.read_csv(row["X_path"])
    y = pd.read_csv(row["y_path"]).squeeze()
    return X, y


# ─────────────────────────────────────────────
# SANITY CHECKS
# ─────────────────────────────────────────────

def verify_condition(tau: float, d: float, n: int, output_dir: str = OUTPUT_DIR):
    """Print a quick sanity check for a single condition."""
    X, y = load_condition(tau, d, n, output_dir)
    n_min = y.sum()
    n_maj = len(y) - n_min
    actual_tau = n_min / len(y)

    print(f"\nCondition: tau={tau:.2f}, d={d:.1f}, n={n}")
    print(f"  Shape:          X={X.shape}, y={y.shape}")
    print(f"  Class counts:   majority={n_maj}, minority={n_min}")
    print(f"  Actual tau:     {actual_tau:.4f} (target: {tau:.4f})")
    print(f"  Feature means (class 0): {X[y==0].mean().values[:3].round(3)} ...")
    print(f"  Feature means (class 1): {X[y==1].mean().values[:3].round(3)} ...")
    print(f"  Expected mean diff (f1): {d:.2f}, observed: "
          f"{(X[y==1]['feature_1'].mean() - X[y==0]['feature_1'].mean()):.3f}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    manifest = generate_all_conditions()

    print("\n── Manifest preview ──")
    print(manifest.to_string(index=False))

    # Sanity check on a few representative conditions
    verify_condition(tau=0.10, d=0.5,  n=300)   # hard: severe imbalance, low sep
    verify_condition(tau=0.10, d=3.0,  n=300)   # severe imbalance, high sep
    verify_condition(tau=0.50, d=1.0,  n=1000)  # balanced, moderate sep (control)