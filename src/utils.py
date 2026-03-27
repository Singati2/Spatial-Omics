"""Shared utilities for the spatial omics measurement error framework."""

import numpy as np
from scipy import sparse
from typing import Optional


def to_dense(X) -> np.ndarray:
    """Convert sparse matrix to dense numpy array."""
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X, dtype=np.float64)


def normalize_counts(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Normalize count matrix to counts per target_sum (default CP10K)."""
    X = to_dense(X).astype(np.float64)
    lib_size = X.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1
    return X / lib_size * target_sum


def get_spatial_edges(adata) -> tuple:
    """Extract spatial neighbor edges from AnnData."""
    adj = adata.obsp["spatial_connectivities"]
    adj_coo = sparse.coo_matrix(adj)
    return adj_coo.row, adj_coo.col


def log_normalize(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Log-normalize a count matrix."""
    return np.log1p(normalize_counts(X, target_sum))
