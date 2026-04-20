"""
Lightweight MLP VAE for multiqubit shadow fingerprints.

Input: (batch, n_features=36) float32 in [-1, 1]  — shadow Pauli expectations
Encoder: n_features → hidden → hidden//2 → (μ, logvar)  [latent_dim]
Decoder: latent_dim → hidden//2 → hidden → n_features   [linear, MSE loss]

Loss: MSE reconstruction + β·KL. No Sigmoid — features are continuous in [-1,1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpVAE(nn.Module):
    def __init__(self, seq_len: int, latent_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        h = hidden
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(h // 2, latent_dim)
        self.fc_logvar = nn.Linear(h // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h // 2),
            nn.LayerNorm(h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, seq_len),
            # No Sigmoid: shadow features are continuous in [-1,1], use MSE loss
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len)  →  (recon, μ, logvar)"""
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) → μ: (batch, latent_dim)  [no reparameterize]"""
        h = self.encoder(x)
        return self.fc_mu(h)


def mlp_vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE + β·KL. MSE averaged over batch and features; KL averaged over batch."""
    rec = F.mse_loss(recon, x)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return rec + beta * kld, rec, kld
