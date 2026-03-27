"""Reference-based imputation via Tangram."""

import numpy as np
import anndata as ad
from scipy import sparse


def run_tangram(
    adata_spatial: ad.AnnData,
    adata_ref: ad.AnnData,
    panel_mask: np.ndarray,
    num_epochs: int = 500,
    device: str = "cpu",
) -> np.ndarray:
    """
    Reference-based imputation using Tangram.

    Maps scRNA-seq cells to spatial locations via optimal transport,
    then transfers genome-wide expression.

    Parameters
    ----------
    adata_spatial : ad.AnnData
        Spatial data with counts in adata.layers['counts'].
    adata_ref : ad.AnnData
        scRNA-seq reference data.
    panel_mask : np.ndarray
        Boolean mask of measured genes.
    num_epochs : int
        Training epochs for the mapping.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray
        Imputed expression matrix (cells x genes), shape matches adata_spatial.
    """
    import tangram as tg

    measured_genes = list(adata_spatial.var_names[panel_mask])

    # Prepare data
    tg.pp_adatas(adata_ref, adata_spatial, genes=measured_genes)

    # Map cells to space
    ad_map = tg.map_cells_to_space(
        adata_ref,
        adata_spatial,
        mode="cells",
        density_prior="rna_count_based",
        num_epochs=num_epochs,
        device=device,
    )

    # Project to get genome-wide predictions
    ad_ge = tg.project_genes(ad_map, adata_ref)

    Lambda_hat = ad_ge.X
    if sparse.issparse(Lambda_hat):
        Lambda_hat = Lambda_hat.toarray()

    return Lambda_hat.astype(np.float64)
