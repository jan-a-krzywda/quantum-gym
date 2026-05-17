"""
World Model - MID tier: Autoregressive VAE.

Architecture:
  Encoder : BiGRU on (basis_emb, outcome_one_hot) → μ, log_σ² (latent_dim)
  Latent  : z ~ N(μ, σ²) via reparameterization trick
  Decoder : Autoregressive GRU conditioned on z (via initial hidden state)
            P(s_i | s_{<i}, b_i, z)

Training  : ELBO = E_q[log P(S|B,z)] - β·KL(q||N(0,I))
EFE       : sample z ~ prior, decode autoregressively, measure stabilizer parity variance.

Spec: world_model_spec_mid.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


BASIS_MAP = {"X": 0, "Y": 1, "Z": 2}


# ─────────────────────────────────────────────────────────────────────────────
# Encoder: BiGRU captures global parity structure bidirectionally
# ─────────────────────────────────────────────────────────────────────────────

class BiGRUEncoder(nn.Module):
    def __init__(self, n_qubits: int, embed_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.basis_emb = nn.Embedding(3, embed_dim)
        # outcome one-hot: 2 classes (binary)
        self.bigru = nn.GRU(
            input_size=embed_dim + 2,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.mu_head = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, bases: torch.Tensor, outcomes: torch.Tensor):
        """
        bases:    (B, N) long
        outcomes: (B, N) float → cast to long for one_hot
        returns: mu (B, latent_dim), log_var (B, latent_dim)
        """
        basis_emb = self.basis_emb(bases)                             # (B, N, embed_dim)
        out_oh = F.one_hot(outcomes.long(), num_classes=2).float()    # (B, N, 2)
        x = torch.cat([basis_emb, out_oh], dim=-1)                   # (B, N, embed_dim+2)
        _, h = self.bigru(x)                                          # h: (2, B, hidden_dim)
        h_cat = torch.cat([h[0], h[1]], dim=-1)                      # (B, 2*hidden_dim)
        return self.mu_head(h_cat), self.logvar_head(h_cat)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder: Autoregressive GRU, z injected as initial hidden state
# ─────────────────────────────────────────────────────────────────────────────

class AutoregressiveDecoder(nn.Module):
    def __init__(self, n_qubits: int, embed_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.basis_emb = nn.Embedding(3, embed_dim)
        self.z_to_h0 = nn.Linear(latent_dim, hidden_dim)
        # GRU input: basis_emb (embed_dim) + prev outcome one-hot (3 classes: 0,1,START=2)
        self.gru = nn.GRU(input_size=embed_dim + 3, hidden_size=hidden_dim, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, bases: torch.Tensor, outcomes: torch.Tensor, z: torch.Tensor):
        """
        Teacher-forced decode for training.
        bases:    (B, N) long
        outcomes: (B, N) float
        z:        (B, latent_dim)
        returns:  (B, N) logits
        """
        B, N = bases.shape
        h0 = torch.tanh(self.z_to_h0(z)).unsqueeze(0)                 # (1, B, hidden_dim)
        start = torch.full((B, 1), 2, dtype=torch.long, device=bases.device)
        shifted = torch.cat([start, outcomes[:, :-1].long()], dim=1)  # (B, N)
        prev_enc = F.one_hot(shifted, num_classes=3).float()           # (B, N, 3)
        basis_emb = self.basis_emb(bases)                              # (B, N, embed_dim)
        gru_in = torch.cat([basis_emb, prev_enc], dim=-1)
        hs, _ = self.gru(gru_in, h0)
        return self.output_head(hs).squeeze(-1)                        # (B, N)

    @torch.no_grad()
    def sample(self, bases: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Batched autoregressive sampling conditioned on z.
        bases: (B, N) long
        z:     (B, latent_dim)
        returns: (B, N) long {0,1}
        """
        B, N = bases.shape
        device = bases.device
        h = torch.tanh(self.z_to_h0(z)).unsqueeze(0)                  # (1, B, hidden_dim)
        prev = torch.full((B, 1), 2, dtype=torch.long, device=device)
        bits = []
        for i in range(N):
            emb = self.basis_emb(bases[:, i:i+1])                     # (B, 1, embed_dim)
            enc = F.one_hot(prev, num_classes=3).float()              # (B, 1, 3)
            out, h = self.gru(torch.cat([emb, enc], dim=-1), h)
            prob = torch.sigmoid(self.output_head(out).squeeze(-1))   # (B, 1)
            bit = torch.bernoulli(prob).long()
            bits.append(bit.squeeze(-1))
            prev = bit
        return torch.stack(bits, dim=1)                                # (B, N)


# ─────────────────────────────────────────────────────────────────────────────
# Full VAE model
# ─────────────────────────────────────────────────────────────────────────────

class AutoregressiveVAE(nn.Module):
    def __init__(
        self,
        n_qubits: int = 6,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.encoder = BiGRUEncoder(n_qubits, embed_dim, hidden_dim, latent_dim)
        self.decoder = AutoregressiveDecoder(n_qubits, embed_dim, hidden_dim, latent_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(self, bases: torch.Tensor, outcomes: torch.Tensor):
        """Returns (logits, mu, log_var) for ELBO computation."""
        mu, log_var = self.encoder(bases, outcomes)
        z = self.reparameterize(mu, log_var)
        logits = self.decoder(bases, outcomes, z)
        return logits, mu, log_var

    @torch.no_grad()
    def sample(self, bases: torch.Tensor) -> torch.Tensor:
        """
        Sample from prior z ~ N(0,I), decode autoregressively.
        bases: (B, N) long  — already batched to n_cands × n_efe_samples
        returns: (B, N) long
        """
        z = torch.randn(bases.shape[0], self.latent_dim, device=bases.device)
        return self.decoder.sample(bases, z)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class MidAgent:
    """
    Active Inference agent using AutoregressiveVAE as world model.
    EFE = stabilizer parity variance across samples decoded from prior z.
    """
    MODEL_NAME = "Mid (Autoregressive VAE)"

    def __init__(
        self,
        n_qubits: int = 6,
        stabilizers: List[str] = None,
        lr: float = 1e-3,
        beta: float = 1.0,
    ):
        self.n_qubits = n_qubits
        self.stabilizers = stabilizers or []
        self.beta = beta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoregressiveVAE(n_qubits).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._buf_bases: List[torch.Tensor] = []
        self._buf_outcomes: List[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # Belief update: ELBO = reconstruction - β·KL
    # ------------------------------------------------------------------ #

    def update_beliefs(
        self,
        shots: List[Tuple[List[str], List[int]]],
        epochs: int = 5,
        batch_size: int = 64,
    ) -> None:
        for basis, outcome in shots:
            self._buf_bases.append(
                torch.tensor([BASIS_MAP[b] for b in basis], dtype=torch.long)
            )
            self._buf_outcomes.append(
                torch.tensor(outcome, dtype=torch.float)
            )

        n = len(self._buf_bases)
        if n < 2:
            return

        X = torch.stack(self._buf_bases).to(self.device)
        Y = torch.stack(self._buf_outcomes).to(self.device)
        bs = min(batch_size, n)

        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                self.optimizer.zero_grad()
                logits, mu, log_var = self.model(X[idx], Y[idx])
                recon = F.binary_cross_entropy_with_logits(logits, Y[idx])
                kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                (recon + self.beta * kl).backward()
                self.optimizer.step()

    # ------------------------------------------------------------------ #
    # EFE helpers
    # ------------------------------------------------------------------ #

    def _stab_variance(self, basis_str: List[str], samples: np.ndarray) -> float:
        total = 0.0
        for stab in self.stabilizers:
            if not all(p == "I" or p == b for p, b in zip(stab, basis_str)):
                continue
            mask = np.array([p != "I" for p in stab])
            if not mask.any():
                continue
            parities = samples[:, mask].sum(axis=1) % 2
            total += float(np.var(np.where(parities == 0, 1.0, -1.0)))
        return total

    # ------------------------------------------------------------------ #
    # Batch selection (vectorised prior sampling)
    # ------------------------------------------------------------------ #

    def select_batch(
        self,
        candidate_bases: List[List[str]],
        batch_size: int = 20,
        n_efe_samples: int = 50,
    ) -> List[List[str]]:
        self.model.eval()
        n_cands = len(candidate_bases)

        all_b = torch.tensor(
            [[BASIS_MAP[b] for b in bs] for bs in candidate_bases],
            dtype=torch.long, device=self.device,
        )
        # (n_cands * n_efe_samples, N)
        all_b_rep = all_b.repeat_interleave(n_efe_samples, dim=0)

        with torch.no_grad():
            samples_all = self.model.sample(all_b_rep).cpu().numpy()

        samples_all = samples_all.reshape(n_cands, n_efe_samples, self.n_qubits)

        efe = np.array([
            self._stab_variance(candidate_bases[j], samples_all[j])
            for j in range(n_cands)
        ], dtype=np.float32)

        efe += 1e-6 * np.random.rand(n_cands)
        efe -= efe.max()
        probs = np.exp(efe)
        probs /= probs.sum()

        indices = np.random.choice(n_cands, size=batch_size, replace=False, p=probs)
        return [candidate_bases[i] for i in indices]

    # ------------------------------------------------------------------ #
    # Stabilizer fidelity metric
    # ------------------------------------------------------------------ #

    def compute_stabilizer_fidelity(self, n_samples: int = 300) -> float:
        self.model.eval()
        evs = []
        for stab in self.stabilizers:
            basis_str = [p if p != "I" else "Z" for p in stab]
            b_t = torch.tensor(
                [[BASIS_MAP[b] for b in basis_str]], dtype=torch.long, device=self.device
            ).repeat(n_samples, 1)
            with torch.no_grad():
                samples = self.model.sample(b_t).cpu().numpy()
            mask = np.array([p != "I" for p in stab])
            parities = samples[:, mask].sum(axis=1) % 2
            ev = float(np.mean(np.where(parities == 0, 1.0, -1.0)))
            evs.append(ev)
        return float(np.mean([(1.0 + ev) / 2.0 for ev in evs]))
