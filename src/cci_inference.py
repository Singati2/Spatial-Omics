"""Cell-cell interaction inference: naive and calibrated."""

import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.stats.multitest import multipletests

from .utils import normalize_counts, get_spatial_edges


def compute_cci_scores(
    adata,
    lr_pairs: pd.DataFrame,
    layer: str = "counts",
    normalize: bool = True,
    n_perms: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute cell-cell interaction scores for all ligand-receptor pairs.

    For each LR pair (l, r), the interaction score across spatial neighbors:

        T_lr = (1/|E|) * sum_{(i,j) in E} x_il * x_jr

    where E is the set of spatially adjacent cell pairs.

    Parameters
    ----------
    adata : ad.AnnData
        Spatial data with spatial neighbor graph.
    lr_pairs : pd.DataFrame
        Columns: 'ligand', 'receptor'.
    layer : str
        Layer to use for expression values.
    normalize : bool
        Whether to normalize to CP10K.
    n_perms : int
        Number of permutations for p-value computation.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        CCI results with columns: ligand, receptor, score, p_value,
        p_adjusted, significant, effect_size, etc.
    """
    rng = np.random.RandomState(seed)

    # Get expression
    if layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float64)

    if normalize:
        X = normalize_counts(X, target_sum=1e4)

    gene_names = list(adata.var_names)
    edges_i, edges_j = get_spatial_edges(adata)
    n_edges = len(edges_i)

    results = []

    for _, row in lr_pairs.iterrows():
        ligand, receptor = row["ligand"], row["receptor"]

        if ligand not in gene_names or receptor not in gene_names:
            continue

        l_idx = gene_names.index(ligand)
        r_idx = gene_names.index(receptor)

        ligand_expr = X[:, l_idx]
        receptor_expr = X[:, r_idx]

        # Observed interaction score
        observed_score = np.mean(ligand_expr[edges_i] * receptor_expr[edges_j])

        # Permutation test
        perm_scores = np.zeros(n_perms)
        for p in range(n_perms):
            perm_idx = rng.permutation(len(ligand_expr))
            perm_scores[p] = np.mean(
                ligand_expr[perm_idx[edges_i]] * receptor_expr[edges_j]
            )

        p_value = (np.sum(perm_scores >= observed_score) + 1) / (n_perms + 1)

        null_mean = perm_scores.mean()
        null_std = perm_scores.std()
        effect_size = (observed_score - null_mean) / null_std if null_std > 0 else 0

        results.append(
            {
                "ligand": ligand,
                "receptor": receptor,
                "score": observed_score,
                "p_value": p_value,
                "effect_size": effect_size,
                "null_mean": null_mean,
                "null_std": null_std,
                "ligand_mean_expr": ligand_expr.mean(),
                "receptor_mean_expr": receptor_expr.mean(),
            }
        )

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        _, results_df["p_adjusted"], _, _ = multipletests(
            results_df["p_value"], method="fdr_bh"
        )
        results_df["significant"] = results_df["p_adjusted"] < 0.05

    return results_df
