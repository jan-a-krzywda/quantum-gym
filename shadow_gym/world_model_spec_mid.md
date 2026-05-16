# Specification: world_model_spec_mid.md (Autoregressive VAE)

## Overview
This "Mid" architecture implements an Autoregressive Variational Autoencoder (Seq2Seq VAE). It compresses the 17-qubit quantum state into a low-dimensional latent space $z$ (for interpretability and manifold mapping) while utilizing an autoregressive RNN decoder to maintain the exact sequential likelihood generation required to respect quantum parity rules.

## 1. Architecture details
* **Encoder (Bidirectional GRU):** Takes the measurement basis $B$ and physical outcome $S$. Processes the sequence bidirectionally to capture global parity rules, projecting the final hidden state into a latent distribution $\mathcal{N}(\mu, \sigma^2)$.
* **Latent Space ($z$):** A low-dimensional continuous vector (e.g., $d=8$ or $d=16$) sampled via the reparameterization trick: $z = \mu + \epsilon \cdot \sigma$. This acts as the compressed "World Model".
* **Decoder (Autoregressive GRU):** Conditioned on the latent vector $z$ and the basis $B$. Generates the predicted outcome $\hat{S}$ sequentially: $P_{\theta}(S | B, z) = \prod_{i=0}^{16} P_{\theta}(s_i | s_{<i}, b_i, z)$.

## 2. Training Objective: The ELBO + Exact Likelihood
The model is trained to maximize the Evidence Lower Bound (ELBO):
$$\mathcal{L} = \mathbb{E}_{q_{\phi}(z | S, B)} [\log P_{\theta}(S | B, z)] - \beta \cdot D_{KL}(q_{\phi}(z | S, B) || \mathcal{N}(0, I))$$
* The first term is the Reconstruction Loss (Binary Cross Entropy from the autoregressive decoder).
* The second term is the KL divergence, forcing the quantum states to organize into a continuous manifold. $\beta$ controls the bottleneck strictness.

## 3. Action Phase: Expected Free Energy (EFE) via Latent Decoding
1. For a candidate basis $B$, sample $K$ vectors from the prior (or the aggregated posterior).
2. Decode these $K$ latent vectors into synthetic bitstrings using the Autoregressive Decoder.
3. Calculate the variance of the target stabilizers across these synthetic strings. High variance indicates high EFE.