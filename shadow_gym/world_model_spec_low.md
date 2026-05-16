# Neural Active Inference for 17-Qubit Tomography (PyTorch)

## Overview
This module implements a Neural Quantum State (NQS) using an Autoregressive Recurrent Neural Network (RNN). It models the joint probability distribution of the 17-qubit outcomes conditioned on the measurement basis, replacing the explicit tracking of expectation values with a highly scalable generative world model.

## 1. The Generative World Model (Autoregressive NQS)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Set

class AutoregressiveNQS(nn.Module):
    """
    Neural Quantum State (NQS) modeled via an autoregressive GRU.
    Learns the conditional distribution P(Outcomes | Basis).
    """
    def __init__(self, n_qubits: int = 17, embed_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Embed the measurement bases: 0->X, 1->Y, 2->Z
        self.basis_embedding = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)
        
        # GRU input: Basis Embedding (embed_dim) + Previous Outcome (3 dims: 0, 1, or 2 for START token)
        self.gru = nn.GRU(input_size=embed_dim + 3, hidden_size=hidden_dim, batch_first=True)
        
        # Output layer predicts the logit for P(s_i = 1)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, bases: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
        """
        Calculates logits for parallel teacher-forced training.
        bases: (Batch, N_Qubits) - integers 0, 1, 2
        outcomes: (Batch, N_Qubits) - integers 0, 1
        """
        B, N = bases.shape
        
        # Shift outcomes right by 1 to create the autoregressive input (predict i from i-1)
        # Prepend a 'START' token (value = 2) for the 0th qubit
        start_tokens = torch.full((B, 1), 2, dtype=torch.long, device=bases.device)
        shifted_outcomes = torch.cat([start_tokens, outcomes[:, :-1]], dim=1)
        
        # One-hot encode the previous outcomes (0, 1, or 2)
        prev_outcome_encoded = F.one_hot(shifted_outcomes, num_classes=3).float()
        
        # Embed the current measurement basis
        basis_embedded = self.basis_embedding(bases)
        
        # Combine and pass through GRU
        gru_in = torch.cat([basis_embedded, prev_outcome_encoded], dim=-1)
        hidden_states, _ = self.gru(gru_in)
        
        # Predict probability of outcome == 1
        logits = self.output_head(hidden_states).squeeze(-1)
        return logits

    @torch.no_grad()
    def sample(self, basis: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Autoregressively 'hallucinates' K synthetic bitstrings for a given basis.
        Used to calculate Expected Free Energy (Uncertainty).
        basis: (1, N_Qubits)
        """
        device = basis.device
        basis_batched = basis.repeat(n_samples, 1) # (n_samples, N_Qubits)
        
        synthetic_outcomes = torch.zeros((n_samples, self.n_qubits), dtype=torch.long, device=device)
        
        # Initial state
        hidden = None
        prev_outcome = torch.full((n_samples, 1), 2, dtype=torch.long, device=device) # START token
        
        for i in range(self.n_qubits):
            curr_basis = basis_batched[:, i:i+1] # (n_samples, 1)
            
            # Prepare step input
            basis_emb = self.basis_embedding(curr_basis)
            prev_out_enc = F.one_hot(prev_outcome, num_classes=3).float()
            step_in = torch.cat([basis_emb, prev_out_enc], dim=-1)
            
            # Step the GRU
            out, hidden = self.gru(step_in, hidden)
            logit = self.output_head(out).squeeze(-1)
            
            # Sample the outcome from the predicted Bernoulli distribution
            prob = torch.sigmoid(logit)
            sampled_bit = torch.bernoulli(prob).long()
            
            synthetic_outcomes[:, i] = sampled_bit.squeeze(-1)
            prev_outcome = sampled_bit # Feed forward to next step
            
        return synthetic_outcomes



class NeuralActiveInferenceAgent:
    """
    Coordinates the NQS training and Expected Free Energy (EFE) evaluation.
    """
    def __init__(self, n_qubits: int = 17, target_stabilizers: List[str] = None, lr: float = 0.01):
        self.n_qubits = n_qubits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoregressiveNQS(n_qubits).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.target_stabilizers = target_stabilizers if target_stabilizers else []
        self.shadow_buffer_bases = []
        self.shadow_buffer_outcomes = []
        
        # Loss function: Binary Cross Entropy
        self.criterion = nn.BCEWithLogitsLoss()

    def update_beliefs(self, new_shots: List[Tuple[List[str], List[int]]], epochs: int = 5, batch_size: int = 64):
        """
        Supervised training phase: updates the world model on all physical data collected so far.
        """
        # 1. Parse and append new shots to the replay buffer
        basis_map = {"X": 0, "Y": 1, "Z": 2}
        for basis, outcome in new_shots:
            if "D" in basis: continue # Skip diagonal mid-circuit shots for generative training
            b_tensor = torch.tensor([basis_map[b] for b in basis], dtype=torch.long)
            o_tensor = torch.tensor(outcome, dtype=torch.float)
            self.shadow_buffer_bases.append(b_tensor)
            self.shadow_buffer_outcomes.append(o_tensor)

        # Convert to full dataset tensors
        X_bases = torch.stack(self.shadow_buffer_bases).to(self.device)
        Y_outs = torch.stack(self.shadow_buffer_outcomes).to(self.device)
        dataset_size = len(X_bases)

        # 2. Train the NQS on the replay buffer
        self.model.train()
        for epoch in range(epochs):
            # Shuffle data
            permutation = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                b_batch, o_batch = X_bases[indices], Y_outs[indices]
                
                self.optimizer.zero_grad()
                logits = self.model(b_batch, o_batch)
                loss = self.criterion(logits, o_batch)
                loss.backward()
                self.optimizer.step()

    def _calculate_stabilizer_variance(self, basis_str: List[str], synthetic_samples: torch.Tensor) -> float:
        """
        Evaluates Epistemic Uncertainty via the Parity Trick on hallucinated samples.
        """
        total_variance = 0.0
        synthetic_samples_np = synthetic_samples.cpu().numpy()
        
        for stab in self.target_stabilizers:
            # Check compatibility: if the target stabilizer cannot be measured in this basis, skip
            if not all((p == "I" or p == b) for p, b in zip(stab, basis_str)):
                continue
                
            # Extract parity for compatible stabilizers
            mask = np.array([p != "I" for p in stab])
            if not mask.any(): continue
                
            # Sum the bits where Pauli != I, modulo 2 determines parity (+1 or -1 eigenvalue)
            relevant_bits = synthetic_samples_np[:, mask]
            parities = np.sum(relevant_bits, axis=1) % 2
            eigenvalues = np.where(parities == 0, 1.0, -1.0)
            
            # The higher the variance, the less the network "knows" about this stabilizer
            total_variance += np.var(eigenvalues)
            
        return total_variance

    def select_batch(self, candidate_bases: List[List[str]], n_samples: int = 100, batch_size: int = 20) -> List[List[str]]:
        """
        Action Phase: Evaluates candidate bases using the Generative Model's uncertainty.
        """
        self.model.eval()
        basis_map = {"X": 0, "Y": 1, "Z": 2}
        efe_scores = []
        
        for basis_str in candidate_bases:
            # Format basis for the network
            b_tensor = torch.tensor([basis_map[b] for b in basis_str], dtype=torch.long, device=self.device).unsqueeze(0)
            
            # 1. Model hallucinates K likely outcomes for this specific measurement
            synthetic_samples = self.model.sample(b_tensor, n_samples)
            
            # 2. Calculate variance (Expected Free Energy) of target observables
            variance = self._calculate_stabilizer_variance(basis_str, synthetic_samples)
            efe_scores.append(variance)
            
        # 3. Softmax sampling or greedy selection of the bases with the highest uncertainty
        efe_tensor = torch.tensor(efe_scores)
        
        # Hardware Softmax trick across candidate bases
        probs = F.softmax(efe_tensor, dim=0).numpy()
        selected_indices = np.random.choice(len(candidate_bases), size=batch_size, p=probs, replace=False)
        
        return [candidate_bases[i] for i in selected_indices]