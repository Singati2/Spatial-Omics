"""Deep generative imputation via gimVI (scvi-tools)."""

import numpy as np
import anndata as ad


def run_gimvi(
    adata_spatial: ad.AnnData,
    adata_ref: ad.AnnData,
    panel_mask: np.ndarray,
    max_epochs: int = 200,
) -> np.ndarray:
    """
    Deep generative imputation using gimVI.

    Joint VAE model for spatial + scRNA-seq data. Models count data
    with negative binomial likelihood and provides uncertainty estimates.

    Parameters
    ----------
    adata_spatial : ad.AnnData
        Spatial data.
    adata_ref : ad.AnnData
        scRNA-seq reference.
    panel_mask : np.ndarray
        Boolean mask of measured genes.
    max_epochs : int
        Training epochs.

    Returns
    -------
    np.ndarray
        Imputed expression matrix (cells x genes).
    """
    import scvi

    # Setup anndata objects
    scvi.model.GIMVI.setup_anndata(adata_spatial)
    scvi.model.GIMVI.setup_anndata(adata_ref)

    model = scvi.model.GIMVI(adata_spatial, adata_ref)
    model.train(max_epochs=max_epochs)

    # Get imputed values for all genes
    _, imputed = model.get_imputed_values(normalized=True)

    return imputed.astype(np.float64)
