"""
Panel perturbation stability analysis.

NOT a bootstrap with coverage guarantees. A stability measure that
quantifies how sensitive biological conclusions are to panel composition.

Procedure:
1. From panel S (m genes), create B sub-panels by dropping a fraction of genes.
2. For each sub-panel, re-impute and re-run CCI.
3. For each LR pair, compute stability (how consistent the result is).
"""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Callable

from .utils import get_spatial_edges


def panel_perturbation_stability(
    adata,
    panel_mask: np.ndarray,
    impute_fn: Callable,
    lr_pairs: pd.DataFrame,
    n_perturbations: int = 50,
    drop_fraction: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Panel perturbation stability analysis for CCI inference.

    Parameters
    ----------
    adata : ad.AnnData
        Spatial data.
    panel_mask : np.ndarray
        Boolean mask of measured genes.
    impute_fn : callable
        Function (adata, mask) -> Lambda_hat.
    lr_pairs : pd.DataFrame
        Columns: 'ligand', 'receptor'.
    n_perturbations : int
        Number of sub-panel perturbations.
    drop_fraction : float
        Fraction of panel genes to drop per perturbation.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Stability metrics for each LR pair.
    """
    rng = np.random.RandomState(seed)
    measured_indices = np.where(panel_mask)[0]
    m = len(measured_indices)
    n_drop = int(drop_fraction * m)

    gene_names = list(adata.var_names)
    edges_i, edges_j = get_spatial_edges(adata)

    # Collect scores across perturbations
    all_scores = {}

    for b in range(n_perturbations):
        # Create sub-panel
        keep_idx = rng.choice(m, size=m - n_drop, replace=False)
        sub_measured = measured_indices[keep_idx]
        sub_mask = np.zeros(len(panel_mask), dtype=bool)
        sub_mask[sub_measured] = True

        # Impute
        Lambda_hat_b = impute_fn(adata, sub_mask)

        # Compute CCI scores
        for _, row in lr_pairs.iterrows():
            ligand, receptor = row["ligand"], row["receptor"]
            if ligand not in gene_names or receptor not in gene_names:
                continue

            pair_key = f"{ligand}_{receptor}"
            l_idx = gene_names.index(ligand)
            r_idx = gene_names.index(receptor)

            score = np.mean(
                Lambda_hat_b[edges_i, l_idx] * Lambda_hat_b[edges_j, r_idx]
            )

            if pair_key not in all_scores:
                all_scores[pair_key] = []
            all_scores[pair_key].append(score)

        if (b + 1) % 10 == 0:
            print(f"  Perturbation {b + 1}/{n_perturbations}")

    # Compute stability metrics
    stability_results = []
    for pair_key in all_scores:
        scores = np.array(all_scores[pair_key])
        parts = pair_key.split("_", 1)
        ligand, receptor = parts[0], parts[1]

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        cv = std_score / (np.abs(mean_score) + 1e-10)

        stability_results.append(
            {
                "ligand": ligand,
                "receptor": receptor,
                "mean_score": mean_score,
                "std_score": std_score,
                "cv_score": cv,
                "stability": 1.0 - np.clip(cv, 0, 1),
                "robustness": np.mean(scores > 0),
            }
        )

    return pd.DataFrame(stability_results)
