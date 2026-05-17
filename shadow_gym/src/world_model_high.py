"""
World Model - HIGH tier: Pure MLP VAE (Free Energy Principle).

Spec: world_model_spec_high.md

Architecture
------------
  Encoder  MLP: cat(one_hot(basis), binary_outcomes) → μ, log_σ²   [parallel]
  Latent   z ~ N(μ, σ²I)
  Decoder  MLP: cat(z, one_hot(basis))  →  N independent Bernoulli logits  [parallel]

Why NOT MADE
------------
  MADE enforces an explicit token-by-token ordering that is physically
  arbitrary for qubit measurements.  The pure MLP VAE already captures
  correlations *through the latent variable*:

      p(s | b) = ∫ p(z) ∏_i p(s_i | b, z) dz

  Even though p(s|b,z) factorises, marginalising over z produces a proper
  mixture model that CAN represent multi-body parity correlations — provided
  the latent space is not collapsed by an over-strong KL term.

Key fix: β_max << 1
-------------------
  With β_max = 1 the KL dominates and forces all z → N(0,I), collapsing
  distinct quantum sectors into a single latent point.  The decoder then
  outputs p_i = 0.5 for every qubit → stabilizer EV = 0 → fidelity stuck
  at 0.5.  Reducing β_max to ~0.01 keeps the latent sectors distinguishable
  while still regularising against overfitting.

Training  : VFE = β · KL(q‖p) − E_q[log P(S|B,z)]   with linear β warm-up
EFE       : Epistemic (mean latent σ² from re-encoding decoder samples)
            + Aleatoric (Shannon entropy of decoder probabilities)
            as specified in world_model_spec_high.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


BASIS_MAP = {"X": 0, "Y": 1, "Z": 2}


# ─────────────────────────────────────────────────────────────────────────────
# Encoder  –  parallel MLP
# ─────────────────────────────────────────────────────────────────────────────

class MLPEncoder(nn.Module):
    """
    Maps (one_hot(basis), binary_outcomes) → (μ, log_σ²).

    Input  : 3·N  (one-hot basis)  +  N  (outcomes ∈ {0,1})
    Output : latent_dim  for each of μ and log_σ²
    """

    def __init__(self, n_qubits: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = 3 * n_qubits + n_qubits
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, basis_oh: torch.Tensor, outcomes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([basis_oh, outcomes], dim=-1))
        return self.mu_head(h), self.logvar_head(h)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder  –  parallel MLP (no autoregression, no MADE)
# ─────────────────────────────────────────────────────────────────────────────

class MLPDecoder(nn.Module):
    """
    Maps cat(z, one_hot(basis)) → N independent Bernoulli logits.

    A single forward pass produces all N qubit predictions simultaneously.
    Qubit correlations are captured implicitly through z, not through an
    explicit autoregressive factorisation.
    """

    def __init__(self, n_qubits: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = latent_dim + 3 * n_qubits
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_qubits),          # raw logits, no activation
        )

    def forward(self, z: torch.Tensor, basis_oh: torch.Tensor) -> torch.Tensor:
        """Returns (B, N) logits for N independent Bernoullis."""
        return self.net(torch.cat([z, basis_oh], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Full VAE
# ─────────────────────────────────────────────────────────────────────────────

class PureVAE(nn.Module):
    """
    Pure MLP VAE — both encoder and decoder are fully parallel MLPs.

    Forward   : (bases, outcomes) → (logits, μ, log_σ²)
    vfe_loss  : VFE = β·KL + BCE_recon
    sample    : z ~ N(0,I), decode → Bernoulli samples
    efe_score : epistemic (latent σ²) + aleatoric (decoder entropy)
    """

    def __init__(
        self,
        n_qubits:   int = 6,
        latent_dim: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_qubits   = n_qubits
        self.latent_dim = latent_dim
        self.encoder = MLPEncoder(n_qubits, latent_dim, hidden_dim)
        self.decoder = MLPDecoder(n_qubits, latent_dim, hidden_dim)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _basis_oh(bases: torch.Tensor) -> torch.Tensor:
        """(B, N) long → (B, 3N) float one-hot."""
        return F.one_hot(bases, num_classes=3).float().view(bases.shape[0], -1)

    def _reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    # ── forward / loss ───────────────────────────────────────────────────

    def forward(
        self, bases: torch.Tensor, outcomes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boh            = self._basis_oh(bases)
        mu, logvar     = self.encoder(boh, outcomes)
        z              = self._reparam(mu, logvar)
        logits         = self.decoder(z, boh)
        return logits, mu, logvar

    def vfe_loss(
        self, bases: torch.Tensor, outcomes: torch.Tensor, beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VFE  =  β · KL(q‖p)  +  BCE_recon

        Reconstruction uses BCE because outcomes are binary {0,1}.
        KL is the analytical Gaussian-vs-standard-Gaussian formula.

        Returns (vfe, recon, kl) — all scalar tensors.
        """
        logits, mu, logvar = self(bases, outcomes)

        # BCE reconstruction: correct for binary outcomes ∈ {0,1}
        recon = F.binary_cross_entropy_with_logits(
            logits, outcomes, reduction="mean"
        )

        # KL: -½ Σ (1 + log σ² - μ² - σ²)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        return recon + beta * kl, recon, kl

    # ── sampling ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, bases: torch.Tensor) -> torch.Tensor:
        """
        Sample binary measurement outcomes for given bases from the prior.

        bases  : (B, N) long
        returns: (B, N) float ∈ {0.0, 1.0}
        """
        boh    = self._basis_oh(bases)
        z      = torch.randn(bases.shape[0], self.latent_dim, device=bases.device)
        logits = self.decoder(z, boh)
        return torch.bernoulli(torch.sigmoid(logits))

    # ── EFE scoring (spec §3) ─────────────────────────────────────────────

    @torch.no_grad()
    def efe_score(
        self, bases: torch.Tensor, n_samples: int = 40
    ) -> torch.Tensor:
        """
        EFE per candidate basis  =  epistemic  +  aleatoric.

        Epistemic
            Sample outcomes from the decoder, re-encode them to get q(z|s,b),
            and measure the mean latent variance σ².  High σ² indicates the
            model is still uncertain about the quantum sector → high info gain.

        Aleatoric
            Shannon entropy of the decoder's output probabilities,
            averaged over prior samples and qubits.
            H[p] = −p log p − (1−p) log(1−p)

        bases  : (B, N) long
        returns: (B,)   float EFE scores
        """
        B   = bases.shape[0]
        boh = self._basis_oh(bases)                              # (B, 3N)

        # Tile bases over n_samples prior draws
        boh_rep = boh.repeat_interleave(n_samples, dim=0)        # (B·K, 3N)
        z       = torch.randn(B * n_samples, self.latent_dim, device=bases.device)
        logits  = self.decoder(z, boh_rep)                       # (B·K, N)
        probs   = torch.sigmoid(logits)                          # (B·K, N)

        # Aleatoric: mean Shannon entropy of decoder probabilities
        eps       = 1e-7
        h         = -(probs * (probs + eps).log()
                      + (1 - probs) * (1 - probs + eps).log())  # (B·K, N)
        aleatoric = h.view(B, n_samples, -1).mean(dim=[1, 2])   # (B,)

        # Epistemic: re-encode decoder samples → latent variance σ²
        s_hat        = torch.bernoulli(probs)                    # (B·K, N)
        _, logvar_q  = self.encoder(boh_rep, s_hat)              # (B·K, latent)
        var_q        = logvar_q.exp().view(B, n_samples, -1)     # (B, K, latent)
        epistemic    = var_q.mean(dim=[1, 2])                    # (B,)

        return epistemic + aleatoric                             # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class HighAgent:
    """
    Active Inference agent backed by the Pure MLP VAE world model.

    Changes from the previous MADE-VAE version
    -------------------------------------------
    * MADE decoder removed — pure parallel MLP decoder (simpler, faster).
    * β_max  = 0.01  (not 1.0) — prevents latent collapse, the main cause
      of the 0.5–0.6 fidelity ceiling.
    * EFE    = latent epistemic σ² + decoder aleatoric entropy (per spec §3).
    * Gradient clipping (max_norm=1.0) — stabilises early training.
    """

    MODEL_NAME = "High (Pure VAE)"

    def __init__(
        self,
        n_qubits:    int   = 6,
        stabilizers: list  = None,
        lr:          float = 1e-3,
        latent_dim:  int   = 8,
        hidden_dim:  int   = 128,
        beta_max:    float = 0.01,   # ← key fix: was 1.0
        beta_warmup: int   = 20,     # update_beliefs calls to reach β_max
    ):
        self.n_qubits    = n_qubits
        self.stabilizers = stabilizers or []
        self.beta_max    = beta_max
        self.beta_warmup = beta_warmup
        self._step       = 0         # counts update_beliefs() calls

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = PureVAE(
            n_qubits=n_qubits, latent_dim=latent_dim, hidden_dim=hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self._buf_bases:    List[torch.Tensor] = []
        self._buf_outcomes: List[torch.Tensor] = []

    # ── KL schedule ──────────────────────────────────────────────────────

    @property
    def beta(self) -> float:
        """Linear warm-up: β goes 0 → β_max over `beta_warmup` update calls."""
        if self.beta_warmup <= 0:
            return self.beta_max
        return min(self.beta_max, self.beta_max * self._step / max(1, self.beta_warmup))

    # ── Belief update: minimise VFE ───────────────────────────────────────

    def update_beliefs(
        self,
        shots:      List[Tuple[List[str], List[int]]],
        epochs:     int = 5,
        batch_size: int = 64,
    ) -> None:
        """
        Append new shots to replay buffer and train the VAE for `epochs` epochs.

        shots : list of (basis_list, outcome_list)
                e.g. (["X","Z","Y"], [0, 1, 0])
        """
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

        X  = torch.stack(self._buf_bases).to(self.device)     # (n, N)
        Y  = torch.stack(self._buf_outcomes).to(self.device)  # (n, N)
        bs = min(batch_size, n)
        β  = self.beta
        self._step += 1

        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                self.optimizer.zero_grad()
                vfe, _, _ = self.model.vfe_loss(X[idx], Y[idx], beta=β)
                vfe.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

    # ── Basis selection via EFE ───────────────────────────────────────────

    def select_batch(
        self,
        candidate_bases: List[List[str]],
        batch_size:      int = 20,
        n_efe_samples:   int = 40,
    ) -> List[List[str]]:
        """
        Score every candidate basis by EFE (latent σ² + decoder entropy)
        and return `batch_size` bases sampled proportional to exp(EFE).
        """
        self.model.eval()
        n_cands = len(candidate_bases)

        all_b = torch.tensor(
            [[BASIS_MAP[b] for b in bs] for bs in candidate_bases],
            dtype=torch.long, device=self.device,
        )                                                         # (n_cands, N)

        with torch.no_grad():
            efe = self.model.efe_score(all_b, n_samples=n_efe_samples).cpu().numpy()

        # Softmax sampling with tiny noise for tie-breaking
        efe += 1e-6 * np.random.rand(n_cands)
        efe -= efe.max()
        probs  = np.exp(efe)
        probs /= probs.sum()

        indices = np.random.choice(n_cands, size=batch_size, replace=False, p=probs)
        return [candidate_bases[i] for i in indices]

    # ── Stabilizer fidelity metric ────────────────────────────────────────

    def compute_stabilizer_fidelity(self, n_samples: int = 300) -> float:
        """
        Estimate F_stab = (1/n) Σ_i (1 + <K_i>) / 2  ∈ [0, 1].

        For each stabilizer K_i, sample from the model in the compatible
        basis and compute the parity expectation value.
        """
        self.model.eval()
        evs = []
        for stab in self.stabilizers:
            basis_str = [p if p != "I" else "Z" for p in stab]
            b_t = torch.tensor(
                [[BASIS_MAP[b] for b in basis_str]],
                dtype=torch.long, device=self.device,
            ).repeat(n_samples, 1)                              # (n_samples, N)

            with torch.no_grad():
                samples = self.model.sample(b_t).cpu().numpy().astype(int)

            mask     = np.array([p != "I" for p in stab])
            parities = samples[:, mask].sum(axis=1) % 2
            ev       = float(np.mean(np.where(parities == 0, 1.0, -1.0)))
            evs.append(ev)

        if not evs:
            return 0.5
        return float(np.mean([(1.0 + ev) / 2.0 for ev in evs]))
