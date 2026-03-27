"""Unit tests for CCI inference and regression calibration."""

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse


def make_test_adata(n_cells=500, n_genes=100, seed=42):
    """Create a minimal test AnnData with spatial structure."""
    rng = np.random.RandomState(seed)

    # Random count matrix
    X = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    X = sparse.csr_matrix(X)

    # Random spatial coordinates
    coords = rng.rand(n_cells, 2) * 100

    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obsm["spatial"] = coords
    adata.layers["counts"] = X.copy()

    # Build spatial neighbors via KNN (simple for testing)
    from sklearn.neighbors import kneighbors_graph

    adj = kneighbors_graph(coords, n_neighbors=6, mode="connectivity")
    adata.obsp["spatial_connectivities"] = adj

    return adata


def test_compute_cci_scores():
    """Test that CCI computation runs and returns expected columns."""
    from src.cci_inference import compute_cci_scores

    adata = make_test_adata()

    lr_pairs = pd.DataFrame(
        {"ligand": ["gene_0", "gene_2"], "receptor": ["gene_1", "gene_3"]}
    )

    results = compute_cci_scores(adata, lr_pairs, n_perms=100)

    assert len(results) == 2
    assert "score" in results.columns
    assert "p_value" in results.columns
    assert "p_adjusted" in results.columns
    assert "significant" in results.columns
    assert all(results["p_value"] >= 0)
    assert all(results["p_value"] <= 1)
    print("test_compute_cci_scores PASSED")


def test_estimate_gene_mse():
    """Test MSE estimation for measured and unmeasured genes."""
    from src.regression_calibration import estimate_gene_mse

    adata = make_test_adata()
    X = adata.layers["counts"].toarray()

    # Simulate imputation: add noise to true values
    rng = np.random.RandomState(42)
    Lambda_hat = X.astype(np.float64) + rng.normal(0, 1, X.shape)
    Lambda_hat = np.clip(Lambda_hat, 0, None)

    panel_mask = np.zeros(adata.n_vars, dtype=bool)
    panel_mask[:50] = True  # first 50 genes measured

    mse = estimate_gene_mse(adata, Lambda_hat, panel_mask)

    assert mse.shape == (adata.n_vars,)
    assert all(mse >= 0)
    # Measured genes should have computable MSE
    assert np.sum(mse[panel_mask] > 0) > 0
    print("test_estimate_gene_mse PASSED")


def test_regression_calibrated_cci():
    """Test that calibration reduces or preserves scores."""
    from src.regression_calibration import (
        estimate_gene_mse,
        regression_calibrated_cci,
    )

    adata = make_test_adata()
    X = adata.layers["counts"].toarray().astype(np.float64)

    rng = np.random.RandomState(42)
    Lambda_hat = X + rng.normal(0, 2, X.shape)
    Lambda_hat = np.clip(Lambda_hat, 0, None)

    panel_mask = np.zeros(adata.n_vars, dtype=bool)
    panel_mask[:50] = True

    mse = estimate_gene_mse(adata, Lambda_hat, panel_mask)

    lr_pairs = pd.DataFrame(
        {"ligand": ["gene_0", "gene_2"], "receptor": ["gene_1", "gene_3"]}
    )

    results = regression_calibrated_cci(
        adata, Lambda_hat, mse, lr_pairs, n_perms=100
    )

    assert "naive_score" in results.columns
    assert "corrected_score" in results.columns
    assert "bias_estimate" in results.columns
    # Bias should be non-negative
    assert all(results["bias_estimate"] >= 0)
    # Corrected score should be <= naive score
    assert all(results["corrected_score"] <= results["naive_score"] + 1e-10)
    print("test_regression_calibrated_cci PASSED")


def test_panel_simulation():
    """Test panel simulation strategies."""
    from src.panel_simulation import simulate_sparse_panel

    adata = make_test_adata()

    lr_pairs = pd.DataFrame(
        {"ligand": ["gene_0", "gene_2"], "receptor": ["gene_1", "gene_3"]}
    )

    mask = simulate_sparse_panel(adata, m=30, lr_pairs=lr_pairs, strategy="mixed")

    assert mask.dtype == bool
    assert mask.sum() == 30
    print("test_panel_simulation PASSED")


if __name__ == "__main__":
    test_compute_cci_scores()
    test_estimate_gene_mse()
    test_regression_calibrated_cci()
    test_panel_simulation()
    print("\nAll tests PASSED")
