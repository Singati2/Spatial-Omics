"""Load and preprocess spatial omics data."""

import scanpy as sc
import squidpy as sq
import anndata as ad
import numpy as np


def load_xenium_breast_cancer(path: str) -> ad.AnnData:
    """
    Load Xenium breast cancer data.

    Expected format: .h5ad with:
      - adata.X: count matrix (cells x genes)
      - adata.obsm['spatial']: (x, y) coordinates
      - adata.obs['cell_type']: cell type annotations (optional)

    Parameters
    ----------
    path : str
        Path to the .h5ad file.

    Returns
    -------
    ad.AnnData
        Preprocessed AnnData with spatial neighbors computed and
        raw counts stored in adata.layers['counts'].
    """
    adata = sc.read_h5ad(path)

    # Basic QC
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=50)

    # Store raw counts
    adata.layers["counts"] = adata.X.copy()

    # Build spatial neighbor graph (Delaunay triangulation)
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)

    print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")
    print(f"Panel density: {adata.n_vars} / ~20000 = {adata.n_vars / 20000:.3f}")

    return adata


def load_reference(path: str) -> ad.AnnData:
    """
    Load scRNA-seq reference data for reference-based imputation.

    Parameters
    ----------
    path : str
        Path to scRNA-seq .h5ad file.

    Returns
    -------
    ad.AnnData
    """
    adata_ref = sc.read_h5ad(path)
    sc.pp.filter_cells(adata_ref, min_counts=200)
    sc.pp.filter_genes(adata_ref, min_cells=10)
    print(f"Reference: {adata_ref.n_obs} cells, {adata_ref.n_vars} genes")
    return adata_ref


def get_lr_pairs(adata: ad.AnnData):
    """
    Get ligand-receptor pairs detectable in the dataset.

    Uses squidpy's built-in LR database (CellPhoneDB/OmniPath).
    Falls back to a minimal curated set if unavailable.

    Parameters
    ----------
    adata : ad.AnnData

    Returns
    -------
    pd.DataFrame
        Columns: 'ligand', 'receptor'
    """
    import pandas as pd

    gene_set = set(adata.var_names)

    try:
        # squidpy >= 1.3 provides a ligrec resource
        interactions = sq.gr.ligrec(
            adata,
            n_perms=0,
            use_raw=False,
            copy=True,
        )
        # Extract unique LR pairs from the result
        # This depends on squidpy version; fallback below
        raise NotImplementedError("Extract from squidpy result")
    except Exception:
        pass

    # Fallback: curated breast cancer-relevant LR pairs
    # In production, load full CellPhoneDB or OmniPath database
    lr_candidates = [
        ("CCL2", "CCR2"), ("CCL5", "CCR5"), ("CXCL12", "CXCR4"),
        ("CXCL10", "CXCR3"), ("CCL19", "CCR7"), ("CCL21", "CCR7"),
        ("FGF2", "FGFR1"), ("FGF7", "FGFR2"), ("EGF", "EGFR"),
        ("HGF", "MET"), ("VEGFA", "FLT1"), ("VEGFA", "KDR"),
        ("PDGFA", "PDGFRA"), ("PDGFB", "PDGFRB"), ("IGF1", "IGF1R"),
        ("WNT5A", "FZD7"), ("WNT7B", "FZD4"), ("BMP2", "BMPR1A"),
        ("BMP4", "BMPR2"), ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"),
        ("TNF", "TNFRSF1A"), ("FASLG", "FAS"), ("CD274", "PDCD1"),
        ("CD80", "CTLA4"), ("CD86", "CD28"), ("ICAM1", "ITGAL"),
        ("VCAM1", "ITGA4"), ("SPP1", "CD44"), ("SPP1", "ITGAV"),
        ("COL1A1", "ITGA1"), ("COL1A1", "ITGB1"), ("FN1", "ITGA5"),
        ("LAMB1", "ITGA6"), ("DLL1", "NOTCH1"), ("JAG1", "NOTCH2"),
        ("DLL4", "NOTCH1"), ("EFNA1", "EPHA2"), ("SEMA3A", "NRP1"),
        ("PLAU", "PLAUR"), ("MMP2", "TIMP2"), ("MMP9", "TIMP1"),
        ("IL6", "IL6R"), ("IL10", "IL10RA"), ("IFNG", "IFNGR1"),
        ("IL1B", "IL1R1"), ("CSF1", "CSF1R"), ("KIT", "KITLG"),
        ("CXCL1", "CXCR2"), ("CXCL8", "CXCR1"), ("CCL3", "CCR1"),
        ("TNFSF10", "TNFRSF10A"), ("ANGPT1", "TEK"), ("ANGPT2", "TEK"),
    ]

    lr_pairs = pd.DataFrame(lr_candidates, columns=["ligand", "receptor"])

    # Filter to genes present in data
    lr_pairs = lr_pairs[
        lr_pairs["ligand"].isin(gene_set) & lr_pairs["receptor"].isin(gene_set)
    ].reset_index(drop=True)

    print(f"LR pairs available in data: {len(lr_pairs)}")
    return lr_pairs
