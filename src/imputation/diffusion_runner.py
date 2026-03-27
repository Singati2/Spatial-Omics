"""Diffusion-based imputation for spatial transcriptomics."""

import numpy as np
import anndata as ad
from scipy import sparse

from ..utils import log_normalize


def run_diffusion(
    adata_spatial: ad.AnnData,
    panel_mask: np.ndarray,
    timesteps: int = 100,
    hidden_dim: int = 512,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """
    Diffusion-based imputation (conditional denoising diffusion model).

    Conditions on measured genes, iteratively denoises unmeasured genes
    from Gaussian noise through learned reverse diffusion.

    Parameters
    ----------
    adata_spatial : ad.AnnData
        Spatial data with counts in layers['counts'].
    panel_mask : np.ndarray
        Boolean mask of measured genes.
    timesteps : int
        Number of diffusion steps.
    hidden_dim : int
        Hidden layer dimension.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Training batch size.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray
        Imputed expression matrix (cells x genes).
    """
    import torch
    import torch.nn as nn

    # Prepare data
    X_counts = adata_spatial.layers["counts"]
    if sparse.issparse(X_counts):
        X_counts = X_counts.toarray()

    X_norm = log_normalize(X_counts)
    X_measured = X_norm[:, panel_mask]
    X_full = X_norm

    N, m_genes = X_measured.shape
    G = X_full.shape[1]
    unmeasured_mask = ~panel_mask
    n_unmeasured = unmeasured_mask.sum()

    # Diffusion schedule
    betas = np.linspace(1e-4, 0.02, timesteps)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)

    # Conditional denoiser network
    class ConditionalDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(m_genes + n_unmeasured + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_unmeasured),
            )

        def forward(self, x_noisy, x_cond, t):
            t_embed = t.unsqueeze(-1) if t.dim() == 1 else t
            inp = torch.cat([x_cond, x_noisy, t_embed], dim=-1)
            return self.net(inp)

    dev = torch.device(device)
    model = ConditionalDenoiser().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_cond = torch.tensor(X_measured, dtype=torch.float32, device=dev)
    X_target = torch.tensor(X_full[:, unmeasured_mask], dtype=torch.float32, device=dev)

    # Train
    model.train()
    actual_batch = min(batch_size, N)
    for epoch in range(epochs):
        perm = torch.randperm(N)[:actual_batch]
        x0 = X_target[perm]
        cond = X_cond[perm]

        # Random timestep
        t = torch.randint(0, timesteps, (actual_batch,), device=dev)
        t_norm = t.float() / timesteps

        # Forward diffusion: add noise
        alpha_bar_t = torch.tensor(
            alpha_bar[t.cpu().numpy()], dtype=torch.float32, device=dev
        ).unsqueeze(-1)
        noise = torch.randn_like(x0)
        x_noisy = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        # Predict noise
        pred_noise = model(x_noisy, cond, t_norm)
        loss = nn.functional.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"    Diffusion epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}")

    # Reverse diffusion (sampling)
    model.eval()
    with torch.no_grad():
        x_t = torch.randn(N, n_unmeasured, device=dev)
        for t in reversed(range(timesteps)):
            t_batch = torch.full((N,), t / timesteps, device=dev)
            pred_noise = model(x_t, X_cond, t_batch)

            alpha_t = alphas[t]
            alpha_bar_t_val = alpha_bar[t]

            x_t = (1 / np.sqrt(alpha_t)) * (
                x_t - (betas[t] / np.sqrt(1 - alpha_bar_t_val)) * pred_noise
            )

            if t > 0:
                x_t += np.sqrt(betas[t]) * torch.randn_like(x_t)

        imputed_unmeasured = x_t.cpu().numpy()

    # Reconstruct full matrix
    Lambda_hat = np.zeros_like(X_full)
    Lambda_hat[:, panel_mask] = X_measured
    Lambda_hat[:, unmeasured_mask] = imputed_unmeasured

    # Convert from log-space
    Lambda_hat = np.expm1(np.clip(Lambda_hat, 0, 15))

    return Lambda_hat.astype(np.float64)
