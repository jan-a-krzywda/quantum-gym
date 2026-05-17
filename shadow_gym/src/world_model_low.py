"""
World Model - LOW tier: Autoregressive NQS (exact likelihood, no latent space).

Architecture: GRU autoregressively models P(s_0,...,s_{N-1} | b_0,...,b_{N-1}).
No compression. Every qubit outcome depends on all previous outcomes via GRU hidden state.
EFE = stabilizer variance across hallucinated bitstrings.

Spec: world_model_spec_low.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


BASIS_MAP = {"X": 0, "Y": 1, "Z": 2}


class AutoregressiveNQS(nn.Module):
    """GRU-based NQS. Models P(outcomes | basis) with no latent compression."""

    def __init__(self, n_qubits: int = 6, embed_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.n_qubits = n_qubits
        # Basis token: 0=X, 1=Y, 2=Z
        self.basis_embedding = nn.Embedding(3, embed_dim)
        # Previous outcome token: 0, 1, or 2 (START)
        self.gru = nn.GRU(input_size=embed_dim + 3, hidden_size=hidden_dim, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, bases: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced forward pass for training.
        bases:    (B, N) long
        outcomes: (B, N) float {0, 1}
        returns:  (B, N) logits for BCEWithLogitsLoss
        """
        B, N = bases.shape
        # Shift outcomes right; prepend START token (value=2)
        start = torch.full((B, 1), 2, dtype=torch.long, device=bases.device)
        shifted = torch.cat([start, outcomes[:, :-1].long()], dim=1)  # (B, N)
        prev_enc = F.one_hot(shifted, num_classes=3).float()           # (B, N, 3)
        basis_emb = self.basis_embedding(bases)                        # (B, N, embed_dim)
        gru_in = torch.cat([basis_emb, prev_enc], dim=-1)
        hidden_states, _ = self.gru(gru_in)
        return self.output_head(hidden_states).squeeze(-1)             # (B, N)

    @torch.no_grad()
    def sample(self, bases: torch.Tensor) -> torch.Tensor:
        """
        Batched autoregressive sampling.
        bases: (B, N) long  — B can be n_candidates × n_efe_samples
        returns: (B, N) long {0, 1}
        """
        B, N = bases.shape
        device = bases.device
        hidden = None
        prev = torch.full((B, 1), 2, dtype=torch.long, device=device)  # START
        bits = []
        for i in range(N):
            emb = self.basis_embedding(bases[:, i:i+1])          # (B, 1, embed_dim)
            enc = F.one_hot(prev, num_classes=3).float()         # (B, 1, 3)
            out, hidden = self.gru(torch.cat([emb, enc], dim=-1), hidden)
            prob = torch.sigmoid(self.output_head(out).squeeze(-1))  # (B, 1)
            bit = torch.bernoulli(prob).long()
            bits.append(bit.squeeze(-1))
            prev = bit
        return torch.stack(bits, dim=1)  # (B, N)


class LowAgent:
    """
    Active Inference agent using AutoregressiveNQS as world model.
    EFE = variance of stabilizer parities over hallucinated samples.
    """
    MODEL_NAME = "Low (Autoregressive NQS)"

    def __init__(self, n_qubits: int = 6, stabilizers: List[str] = None, lr: float = 1e-3):
        self.n_qubits = n_qubits
        self.stabilizers = stabilizers or []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoregressiveNQS(n_qubits).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self._buf_bases: List[torch.Tensor] = []
        self._buf_outcomes: List[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # Belief update
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
                logits = self.model(X[idx], Y[idx])
                self.criterion(logits, Y[idx]).backward()
                self.optimizer.step()

    # ------------------------------------------------------------------ #
    # EFE helpers
    # ------------------------------------------------------------------ #

    def _stab_variance(self, basis_str: List[str], samples: np.ndarray) -> float:
        """Parity variance of each compatible stabilizer, summed."""
        total = 0.0
        for stab in self.stabilizers:
            if not all(p == "I" or p == b for p, b in zip(stab, basis_str)):
                continue
            mask = np.array([p != "I" for p in stab])
            if not mask.any():
                continue
            parities = samples[:, mask].sum(axis=1) % 2
            evs = np.where(parities == 0, 1.0, -1.0)
            total += float(np.var(evs))
        return total

    # ------------------------------------------------------------------ #
    # Batch selection (vectorised)
    # ------------------------------------------------------------------ #

    def select_batch(
        self,
        candidate_bases: List[List[str]],
        batch_size: int = 20,
        n_efe_samples: int = 50,
    ) -> List[List[str]]:
        self.model.eval()
        n_cands = len(candidate_bases)

        # Build (n_cands × n_efe_samples, N) basis tensor in one shot
        all_b = torch.tensor(
            [[BASIS_MAP[b] for b in bs] for bs in candidate_bases],
            dtype=torch.long, device=self.device,
        )
        all_b_rep = all_b.repeat_interleave(n_efe_samples, dim=0)

        with torch.no_grad():
            samples_all = self.model.sample(all_b_rep).cpu().numpy()  # (n_cands*K, N)

        samples_all = samples_all.reshape(n_cands, n_efe_samples, self.n_qubits)

        efe = np.array([
            self._stab_variance(candidate_bases[j], samples_all[j])
            for j in range(n_cands)
        ], dtype=np.float32)

        # Break ties; softmax; sample without replacement
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
        """
        For each stabilizer K_i, pick a compatible basis, sample bitstrings,
        compute mean eigenvalue <K_i>. Fidelity = mean_i (1 + <K_i>) / 2.
        Perfect model → 1.0, random model → 0.5.
        """
        self.model.eval()
        evs = []
        for stab in self.stabilizers:
            # Compatible basis: use Pauli where non-I, else Z
            basis_str = [p if p != "I" else "Z" for p in stab]
            b_t = torch.tensor(
                [[BASIS_MAP[b] for b in basis_str]], dtype=torch.long, device=self.device
            ).repeat(n_samples, 1)
            with torch.no_grad():
                samples = self.model.sample(b_t).cpu().numpy()  # (n_samples, N)
            mask = np.array([p != "I" for p in stab])
            parities = samples[:, mask].sum(axis=1) % 2
            ev = float(np.mean(np.where(parities == 0, 1.0, -1.0)))
            evs.append(ev)
        return float(np.mean([(1.0 + ev) / 2.0 for ev in evs]))
