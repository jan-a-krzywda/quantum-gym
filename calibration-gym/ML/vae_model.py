"""1D convolutional VAE for per-(shot, qubit) τ sweeps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1d_out_length(
    length_in: int, kernel: int = 3, stride: int = 2, padding: int = 1
) -> int:
    return (length_in + 2 * padding - kernel) // stride + 1


def _transpose_out_length(
    length_in: int,
    kernel: int = 3,
    stride: int = 2,
    padding: int = 1,
    output_padding: int = 0,
) -> int:
    return (length_in - 1) * stride - 2 * padding + kernel + output_padding


def _pick_output_padding(length_in: int, target: int) -> int:
    """Choose ``output_padding`` in ``{0, 1}`` so transpose conv hits ``target`` length."""
    for op in (0, 1):
        if _transpose_out_length(length_in, output_padding=op) == target:
            return op
    raise ValueError(
        f"Cannot match sequence length: transpose from len={length_in} to {target} "
        "with kernel=3, stride=2, padding=1 and output_padding in {{0,1}}"
    )


class QubitConvVAE(nn.Module):
    """
    Convolutional VAE on sequences ``[batch, 1, n_tau]``.

    Encoder/decoder widths match the user's reference architecture; lengths are derived
    from ``n_tau`` so jobs with e.g. 50 delay steps (``tau_ns_from_indices(1, 51)``) work
    without hand-tuning 256 = 32×8 flats.
    """

    def __init__(self, seq_len: int, latent_dim: int = 2):
        super().__init__()
        if seq_len < 4:
            raise ValueError(f"seq_len must be >= 4 for this architecture, got {seq_len}")

        self.seq_len = int(seq_len)
        self.latent_dim = int(latent_dim)

        l1 = _conv1d_out_length(self.seq_len)
        l2 = _conv1d_out_length(l1)
        self._enc_spatial = l2
        flat_dim = 32 * l2

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, flat_dim)
        op1 = _pick_output_padding(l2, l1)
        op2 = _pick_output_padding(l1, self.seq_len)

        # Conv stack only; ``decoder_input`` maps z → channels×time (see ``forward``).
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, l2)),
            nn.ConvTranspose1d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=op1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=op2
            ),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        mu, logvar = self.fc_mu(encoded), self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        h = self.decoder_input(z)
        reconstructed = self.decoder(h)
        return reconstructed, mu, logvar


def vae_loss(
    reconstructed_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes VAE loss: sum over features/sequence, average over batch.
    """
    # 1. Reconstruction Loss: Sum over sequence length (dim 1 and 2), average over batch
    rec_per_sample = F.binary_cross_entropy(reconstructed_x, x, reduction="none")
    rec = rec_per_sample.sum(dim=(1, 2)).mean()

    # 2. KL Divergence: Sum over latent dimensions, average over batch
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = kld_per_sample.mean()

    total = rec + float(beta) * kld
    return total, rec, kld
