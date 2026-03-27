"""Simulate sparse gene panels from full spatial omics data."""

import scanpy as sc
import numpy as np
import pandas as pd
from typing import Optional


def simulate_sparse_panel(
    adata,
    m: int = 300,
    lr_pairs: Optional[pd.DataFrame] = None,
    strategy: str = "mixed",
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate a sparse gene panel of size m.

    Parameters
    ----------
    adata : ad.AnnData
        Full spatial omics dataset.
    m : int
        Number of genes in the panel.
    lr_pairs : pd.DataFrame, optional
        LR pair table with columns 'ligand', 'receptor'.
    strategy : str
        Panel design strategy:
        - 'random': uniform random selection
        - 'mixed': 50% HVG + 30% LR genes + 20% random (mimics real panels)
        - 'lr_rich': maximize LR pair coverage
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_genes,).
    """
    rng = np.random.RandomState(seed)
    G = adata.n_vars
    gene_names = np.array(adata.var_names)

    if strategy == "random":
        selected = rng.choice(G, size=min(m, G), replace=False)
        mask = np.zeros(G, dtype=bool)
        mask[selected] = True

    elif strategy == "mixed":
        # Compute HVGs
        adata_tmp = adata.copy()
        sc.pp.normalize_total(adata_tmp, target_sum=1e4)
        sc.pp.log1p(adata_tmp)
        sc.pp.highly_variable_genes(adata_tmp, n_top_genes=min(m, G))
        hvg_mask = adata_tmp.var["highly_variable"].values
        hvg_indices = np.where(hvg_mask)[0]

        # LR genes
        if lr_pairs is not None:
            lr_genes = set(lr_pairs["ligand"]) | set(lr_pairs["receptor"])
            lr_indices = [i for i, g in enumerate(gene_names) if g in lr_genes]
        else:
            lr_indices = []

        # Allocate: 50% HVG, 30% LR, 20% random
        n_hvg = int(0.5 * m)
        n_lr = int(0.3 * m)

        selected = set()

        # HVGs
        hvg_pick = rng.choice(
            hvg_indices, size=min(n_hvg, len(hvg_indices)), replace=False
        )
        selected.update(hvg_pick)

        # LR genes
        lr_remaining = [i for i in lr_indices if i not in selected]
        if lr_remaining:
            lr_pick = rng.choice(
                lr_remaining, size=min(n_lr, len(lr_remaining)), replace=False
            )
            selected.update(lr_pick)

        # Random fill
        remaining = [i for i in range(G) if i not in selected]
        n_fill = m - len(selected)
        if n_fill > 0 and remaining:
            random_pick = rng.choice(
                remaining, size=min(n_fill, len(remaining)), replace=False
            )
            selected.update(random_pick)

        mask = np.zeros(G, dtype=bool)
        for idx in selected:
            mask[idx] = True

    elif strategy == "lr_rich":
        # Maximize LR pair coverage
        if lr_pairs is None:
            raise ValueError("lr_pairs required for lr_rich strategy")

        lr_genes = set(lr_pairs["ligand"]) | set(lr_pairs["receptor"])
        lr_indices = [i for i, g in enumerate(gene_names) if g in lr_genes]

        selected = set(lr_indices[:m])

        # Fill remaining with random
        remaining = [i for i in range(G) if i not in selected]
        n_fill = m - len(selected)
        if n_fill > 0 and remaining:
            random_pick = rng.choice(
                remaining, size=min(n_fill, len(remaining)), replace=False
            )
            selected.update(random_pick)

        mask = np.zeros(G, dtype=bool)
        for idx in selected:
            mask[idx] = True

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Report LR coverage
    if lr_pairs is not None:
        measured_genes = set(gene_names[mask])
        lr_covered = lr_pairs[
            lr_pairs["ligand"].isin(measured_genes)
            & lr_pairs["receptor"].isin(measured_genes)
        ]
        lr_partial = lr_pairs[
            lr_pairs["ligand"].isin(measured_genes)
            | lr_pairs["receptor"].isin(measured_genes)
        ]
        print(f"Panel size: {mask.sum()}")
        print(f"LR pairs fully measured: {len(lr_covered)} / {len(lr_pairs)}")
        print(f"LR pairs partially measured: {len(lr_partial)} / {len(lr_pairs)}")
        print(
            f"LR pairs needing imputation: "
            f"{len(lr_pairs) - len(lr_covered)} / {len(lr_pairs)}"
        )

    return mask
