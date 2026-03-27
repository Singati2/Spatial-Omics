"""GNN-based spatial imputation using graph convolutional networks."""

import numpy as np
import anndata as ad
from scipy import sparse

from ..utils import log_normalize


def run_spatial_gnn(
    adata_spatial: ad.AnnData,
    panel_mask: np.ndarray,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cpu",
) -> np.ndarray:
    """
    GNN-based imputation using spatial graph structure.

    Builds a spatial graph from cell coordinates, uses measured genes as
    node features, propagates information through GCN message passing,
    and predicts unmeasured genes.

    Parameters
    ----------
    adata_spatial : ad.AnnData
        Spatial data with counts in layers['counts'] and spatial neighbors.
    panel_mask : np.ndarray
        Boolean mask of measured genes.
    hidden_dim : int
        Hidden layer dimension.
    latent_dim : int
        Latent representation dimension.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray
        Imputed expression matrix (cells x genes).
    """
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv

    # Prepare data
    X_counts = adata_spatial.layers["counts"]
    if sparse.issparse(X_counts):
        X_counts = X_counts.toarray()

    X_norm = log_normalize(X_counts)
    X_measured = X_norm[:, panel_mask]
    X_full = X_norm

    # Build edge index from spatial neighbors
    adj = adata_spatial.obsp["spatial_connectivities"]
    adj_coo = sparse.coo_matrix(adj)
    edge_index = torch.tensor(
        np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long
    ).to(device)

    m = X_measured.shape[1]
    G = X_full.shape[1]

    x_tensor = torch.tensor(X_measured, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(X_full, dtype=torch.float32).to(device)

    # GCN autoencoder
    class SpatialGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(m, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, G),
            )

        def forward(self, x, edge_idx):
            h = torch.relu(self.conv1(x, edge_idx))
            h = self.conv2(h, edge_idx)
            return self.decoder(h)

    model = SpatialGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train: self-supervised loss on measured genes
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_tensor, edge_index)
        loss = nn.functional.mse_loss(pred[:, panel_mask], y_tensor[:, panel_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"    GNN epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}")

    # Predict full matrix
    model.eval()
    with torch.no_grad():
        Lambda_hat = model(x_tensor, edge_index).cpu().numpy()

    # Convert from log-space
    Lambda_hat = np.expm1(np.clip(Lambda_hat, 0, 15))

    return Lambda_hat.astype(np.float64)
