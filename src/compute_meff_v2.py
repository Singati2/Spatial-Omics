"""
Compute m_i, rho_i (Moran's I), and m_eff — v2 (manual loading, handles duplicate genes)
"""
import os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
import h5py
import warnings
warnings.filterwarnings('ignore')

base_dir = "/Users/ganeshshiwakoti/Desktop/Biostatistics/E-MTAB-13530"
results = []

h5_files = sorted([f for f in os.listdir(base_dir) if f.endswith('-filtered_feature_bc_matrix.h5')])

for h5_file in h5_files:
    sample_name = h5_file.replace('-filtered_feature_bc_matrix.h5', '')
    spatial_dir = os.path.join(base_dir, f"{sample_name}_spatial", "spatial")
    h5_path = os.path.join(base_dir, h5_file)
    pos_file = os.path.join(spatial_dir, 'tissue_positions_list.csv')

    if not os.path.exists(pos_file):
        print(f"  SKIP {sample_name}: no positions file")
        continue

    try:
        # Load positions
        pos = pd.read_csv(pos_file, header=None)
        pos.columns = ['barcode', 'in_tissue', 'row', 'col', 'px_y', 'px_x']
        pos = pos[pos['in_tissue'] == 1].copy()
        pos = pos.set_index('barcode')
        m_i = len(pos)

        if m_i < 50:
            print(f"  SKIP {sample_name}: only {m_i} spots")
            continue

        # Load h5 expression
        with h5py.File(h5_path, 'r') as f:
            grp = f['matrix']
            data = grp['data'][:]
            indices = grp['indices'][:]
            indptr = grp['indptr'][:]
            shape = grp['shape'][:]
            barcodes = [b.decode() for b in grp['barcodes'][:]]
            gene_names = [g.decode() for g in grp['features']['name'][:]]

        X = sparse.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()

        # Make gene names unique
        from collections import Counter
        counts = Counter(gene_names)
        seen = {}
        unique_genes = []
        for g in gene_names:
            if counts[g] > 1:
                n = seen.get(g, 0)
                unique_genes.append(f"{g}_{n}")
                seen[g] = n + 1
            else:
                unique_genes.append(g)

        # Create AnnData
        import anndata as ad
        adata = ad.AnnData(X=X)
        adata.obs_names = barcodes
        adata.var_names = unique_genes

        # Filter to in-tissue spots
        common = [b for b in adata.obs_names if b in pos.index]
        adata = adata[common].copy()
        coords = pos.loc[common, ['px_x', 'px_y']].values.astype(float)
        adata.obsm['spatial'] = coords
        m_i = adata.n_obs

        if m_i < 50:
            print(f"  SKIP {sample_name}: {m_i} in-tissue spots")
            continue

        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Build 6-NN graph manually
        adj = kneighbors_graph(coords, n_neighbors=6, mode='connectivity')
        adj = (adj + adj.T)
        adj[adj > 1] = 1
        adata.obsp['spatial_connectivities'] = adj
        adata.obsp['spatial_distances'] = kneighbors_graph(coords, n_neighbors=6, mode='distance')

        # Find immune genes
        immune_genes = ['CD3E', 'CD3D', 'CD8A', 'CD8B', 'CD68', 'CD14', 'PTPRC', 'CD4', 'FOXP3', 'MS4A1']
        available = [g for g in immune_genes if g in adata.var_names]

        if len(available) == 0:
            # Use top variable genes
            sc.pp.highly_variable_genes(adata, n_top_genes=20, flavor='seurat_v3',
                                         layer=None, subset=False)
            available = list(adata.var_names[adata.var['highly_variable']][:5])

        # Compute Moran's I manually for robustness
        from scipy.sparse import issparse
        W_mat = adata.obsp['spatial_connectivities']
        if issparse(W_mat):
            W_mat = W_mat.toarray()

        S0 = W_mat.sum()
        n = W_mat.shape[0]
        morans_list = []

        for gene in available[:5]:
            if gene not in adata.var_names:
                continue
            g_idx = list(adata.var_names).index(gene)
            x = adata.X[:, g_idx]
            if issparse(x):
                x = x.toarray().flatten()
            else:
                x = np.array(x).flatten()

            xbar = x.mean()
            xdev = x - xbar
            ss = np.sum(xdev**2)
            if ss == 0:
                continue

            num = 0.0
            for i in range(n):
                for j in range(n):
                    if W_mat[i, j] > 0:
                        num += xdev[i] * xdev[j]

            I_val = (n / S0) * (num / ss)
            morans_list.append(I_val)

        if len(morans_list) == 0:
            # Fallback: compute on total counts
            x = np.array(adata.X.sum(axis=1)).flatten()
            xbar = x.mean()
            xdev = x - xbar
            ss = np.sum(xdev**2)
            if ss > 0:
                num = float(xdev @ W_mat @ xdev)
                I_val = (n / S0) * (num / ss)
                morans_list = [I_val]

        rho_i = float(np.mean(morans_list)) if morans_list else 0.0

        # Compute effective sample size
        rho_clamped = max(0.001, min(rho_i, 0.999))
        DEFF = 1 + (m_i - 1) * rho_clamped
        m_eff = m_i / DEFF

        # Determine type
        parts = sample_name.split('_')
        is_tumor = len(parts) > 1 and parts[1].startswith('T')
        patient_id = parts[0]

        results.append({
            'sample': sample_name,
            'patient': patient_id,
            'type': 'tumor' if is_tumor else 'normal/adjacent',
            'm_i': m_i,
            'rho_i': round(rho_i, 4),
            'DEFF': round(DEFF, 1),
            'm_eff': round(m_eff, 1),
            'genes_used': ', '.join(available[:5]),
        })

        print(f"  OK {sample_name}: m={m_i}, rho={rho_i:.4f}, DEFF={DEFF:.1f}, m_eff={m_eff:.1f}")

    except Exception as e:
        print(f"  ERROR {sample_name}: {e}")
        import traceback
        traceback.print_exc()
        # Continue to next sample
        continue

if not results:
    print("NO RESULTS. Check errors above.")
    sys.exit(1)

df = pd.DataFrame(results)
df = df.sort_values(['patient', 'type', 'sample'])
df.to_csv(os.path.join(base_dir, 'meff_results.csv'), index=False)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

tumor = df[df['type'] == 'tumor']
normal = df[df['type'] != 'tumor']

print(f"\nTotal sections: {len(df)} ({len(tumor)} tumor, {len(normal)} normal/adjacent)")

print("\n--- TUMOR SECTIONS ---")
if len(tumor) > 0:
    print(f"  m_i:   {tumor['m_i'].min()} – {tumor['m_i'].max()} (mean {tumor['m_i'].mean():.0f})")
    print(f"  rho_i: {tumor['rho_i'].min():.4f} – {tumor['rho_i'].max():.4f} (mean {tumor['rho_i'].mean():.4f})")
    print(f"  DEFF:  {tumor['DEFF'].min():.1f} – {tumor['DEFF'].max():.1f} (mean {tumor['DEFF'].mean():.1f})")
    print(f"  m_eff: {tumor['m_eff'].min():.1f} – {tumor['m_eff'].max():.1f} (mean {tumor['m_eff'].mean():.1f})")

print("\n--- NORMAL/ADJACENT SECTIONS ---")
if len(normal) > 0:
    print(f"  m_i:   {normal['m_i'].min()} – {normal['m_i'].max()} (mean {normal['m_i'].mean():.0f})")
    print(f"  rho_i: {normal['rho_i'].min():.4f} – {normal['rho_i'].max():.4f} (mean {normal['rho_i'].mean():.4f})")
    print(f"  m_eff: {normal['m_eff'].min():.1f} – {normal['m_eff'].max():.1f} (mean {normal['m_eff'].mean():.1f})")

print("\n" + "=" * 80)
print("GO / NO-GO DECISION")
print("=" * 80)
if len(tumor) > 0:
    mean_meff = tumor['m_eff'].mean()
    mean_mi = tumor['m_i'].mean()
    ratio = mean_meff / mean_mi
    print(f"\n  Mean m_i (tumor):   {mean_mi:.0f} spots")
    print(f"  Mean m_eff (tumor): {mean_meff:.1f} effective spots")
    print(f"  Ratio m_eff/m_i:    {ratio:.4f}")
    print(f"  Effective reduction: {(1-ratio)*100:.1f}%")
    print()
    if mean_meff < 20:
        print("  >>> VERDICT: STRONG EFFECT. Paper exists. GO. <<<")
    elif mean_meff < 100:
        print("  >>> VERDICT: MODERATE EFFECT. Paper viable. <<<")
    elif mean_meff < 500:
        print("  >>> VERDICT: WEAK EFFECT. Reconsider framing. <<<")
    else:
        print("  >>> VERDICT: NO MEANINGFUL EFFECT. Pivot needed. <<<")

print("\n--- FULL TABLE ---")
print(df.to_string(index=False))
