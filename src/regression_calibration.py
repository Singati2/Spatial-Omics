"""
Regression calibration for bias correction in CCI statistics.

Core insight: Spatial imputation methods introduce spatially correlated
reconstruction errors. Because CCI tests measure spatial co-expression,
these correlated errors inflate CCI statistics, generating false positives.

If lambda_hat = lambda + e, where e has spatial correlation, then:

    E[lambda_hat_il * lambda_hat_jr] = lambda_il * lambda_jr + Cov(e_il, e_jr)

The bias term Cov(e_il, e_jr) is positive for graph-based imputation methods
that share information across spatial neighbors.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from statsmodels.stats.multitest import multipletests

from .utils import to_dense, normalize_counts, get_spatial_edges


def estimate_gene_mse(
    adata,
    Lambda_hat: np.ndarray,
    panel_mask: np.ndarray,
    n_components: int = 30,
) -> np.ndarray:
    """
    Estimate per-gene reconstruction MSE.

    For measured genes: direct comparison with true counts.
    For unmeasured genes: predicted via a meta-model trained on measured genes,
    using mean expression, variance, and latent-space predictability (R^2).

    Parameters
    ----------
    adata : ad.AnnData
        Original data with counts in layers['counts'].
    Lambda_hat : np.ndarray
        Imputed expression matrix.
    panel_mask : np.ndarray
        Boolean mask of measured genes.
    n_components : int
        Number of PCA components for latent structure.

    Returns
    -------
    np.ndarray
        Estimated MSE for each gene, shape (n_genes,).
    """
    X_true = to_dense(adata.layers["counts"])
    X_true_norm = normalize_counts(X_true)

    G = X_true.shape[1]
    mse = np.zeros(G)

    # Measured genes: direct MSE
    measured_idx = np.where(panel_mask)[0]
    for g in measured_idx:
        mse[g] = np.mean((X_true_norm[:, g] - Lambda_hat[:, g]) ** 2)

    # Latent structure via PCA on imputed measured genes
    X_measured_imputed = Lambda_hat[:, panel_mask]
    K = min(n_components, len(measured_idx) - 1)
    pca = PCA(n_components=K)
    U = pca.fit_transform(X_measured_imputed)

    # R^2 for each measured gene
    r2_measured = np.zeros(len(measured_idx))
    for i in range(len(measured_idx)):
        pred = U @ (U.T @ X_measured_imputed[:, i]) / len(U)
        ss_res = np.sum((X_measured_imputed[:, i] - pred) ** 2)
        ss_tot = np.sum(
            (X_measured_imputed[:, i] - X_measured_imputed[:, i].mean()) ** 2
        )
        r2_measured[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Fit MSE prediction model: MSE ~ f(mean_expr, var, R^2)
    measured_features = np.column_stack(
        [
            np.log1p(X_true_norm[:, measured_idx].mean(axis=0)),
            np.log1p(X_true_norm[:, measured_idx].var(axis=0)),
            r2_measured,
        ]
    )

    mse_model = Ridge(alpha=1.0)
    mse_model.fit(measured_features, np.log1p(mse[measured_idx]))

    # Predict MSE for unmeasured genes
    unmeasured_idx = np.where(~panel_mask)[0]

    r2_unmeasured = np.zeros(len(unmeasured_idx))
    for i, g in enumerate(unmeasured_idx):
        pred = U @ (U.T @ Lambda_hat[:, g]) / len(U)
        ss_res = np.sum((Lambda_hat[:, g] - pred) ** 2)
        ss_tot = np.sum((Lambda_hat[:, g] - Lambda_hat[:, g].mean()) ** 2)
        r2_unmeasured[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    unmeasured_features = np.column_stack(
        [
            np.log1p(Lambda_hat[:, unmeasured_idx].mean(axis=0)),
            np.log1p(Lambda_hat[:, unmeasured_idx].var(axis=0)),
            r2_unmeasured,
        ]
    )

    mse[unmeasured_idx] = np.expm1(mse_model.predict(unmeasured_features))
    mse = np.clip(mse, 0, None)

    return mse


def regression_calibrated_cci(
    adata,
    Lambda_hat: np.ndarray,
    mse: np.ndarray,
    lr_pairs: pd.DataFrame,
    n_perms: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Regression-calibrated CCI inference.

    Corrects the bias in CCI statistics caused by spatially correlated
    imputation errors.

    The bias for LR pair (l, r) is estimated as:

        bias_lr = rho_lr * sqrt(MSE_l * MSE_r)

    where rho_lr captures spatial error correlation, estimated via
    Moran's I of the expression values.

    Parameters
    ----------
    adata : ad.AnnData
        Spatial data with neighbor graph.
    Lambda_hat : np.ndarray
        Imputed expression matrix.
    mse : np.ndarray
        Per-gene MSE estimates from estimate_gene_mse().
    lr_pairs : pd.DataFrame
        Columns: 'ligand', 'receptor'.
    n_perms : int
        Permutations for p-value.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Calibrated CCI results with bias estimates and corrected scores.
    """
    rng = np.random.RandomState(seed)
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

        ligand_expr = Lambda_hat[:, l_idx]
        receptor_expr = Lambda_hat[:, r_idx]

        # Naive score
        naive_score = np.mean(ligand_expr[edges_i] * receptor_expr[edges_j])

        # --- REGRESSION CALIBRATION ---
        mse_l = mse[l_idx]
        mse_r = mse[r_idx]

        # Moran's I for spatial autocorrelation of each gene
        ligand_centered = ligand_expr - ligand_expr.mean()
        spatial_autocorr_l = np.mean(
            ligand_centered[edges_i] * ligand_centered[edges_j]
        ) / (np.var(ligand_expr) + 1e-10)

        receptor_centered = receptor_expr - receptor_expr.mean()
        spatial_autocorr_r = np.mean(
            receptor_centered[edges_i] * receptor_centered[edges_j]
        ) / (np.var(receptor_expr) + 1e-10)

        # Cross-gene spatial error correlation estimate
        rho_lr = np.sqrt(np.abs(spatial_autocorr_l) * np.abs(spatial_autocorr_r))

        # Bias estimate
        bias_estimate = rho_lr * np.sqrt(mse_l * mse_r)

        # Corrected score
        corrected_score = naive_score - bias_estimate

        # Variance including correction uncertainty
        var_naive = (
            np.var(ligand_expr[edges_i] * receptor_expr[edges_j]) / n_edges
        )
        var_correction = (mse_l * mse_r) / n_edges
        var_total = var_naive + var_correction

        # Permutation test on corrected score
        perm_scores = np.zeros(n_perms)
        for p in range(n_perms):
            perm_idx = rng.permutation(len(ligand_expr))
            perm_scores[p] = (
                np.mean(ligand_expr[perm_idx[edges_i]] * receptor_expr[edges_j])
                - bias_estimate
            )

        p_value = (np.sum(perm_scores >= corrected_score) + 1) / (n_perms + 1)

        results.append(
            {
                "ligand": ligand,
                "receptor": receptor,
                "naive_score": naive_score,
                "bias_estimate": bias_estimate,
                "corrected_score": corrected_score,
                "p_value": p_value,
                "mse_ligand": mse_l,
                "mse_receptor": mse_r,
                "spatial_autocorr_ligand": spatial_autocorr_l,
                "spatial_autocorr_receptor": spatial_autocorr_r,
                "rho_lr": rho_lr,
                "var_total": var_total,
            }
        )

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        _, results_df["p_adjusted"], _, _ = multipletests(
            results_df["p_value"], method="fdr_bh"
        )
        results_df["significant"] = results_df["p_adjusted"] < 0.05

    return results_df
