import torch
import numpy as np

def generate_ou_process(batch_size, seq_len, dt=0.01):
    """
    Simulates a 2D Ornstein-Uhlenbeck (O-U) process using the Euler-Maruyama method.
    dX_t = -Theta * X_t * dt + Sigma * dW_t
    """
    # Parameters
    # Mean reversion matrix
    Theta = torch.tensor([[1.0, 0.0],
                          [0.0, 1.0]])

    # Volatility matrix (non-diagonal to ensure cross-correlation)
    Sigma = torch.tensor([[0.5, 0.2],
                          [0.2, 0.5]])

    # Initialize the time series
    X = torch.zeros((batch_size, seq_len, 2))

    # Initial state (random or zero, using zero here)
    X[:, 0, :] = 0.0

    for t in range(1, seq_len):
        # Brownian motion increment (dW)
        dW = torch.randn((batch_size, 2)) * torch.sqrt(torch.tensor(dt))

        # Current state
        X_curr = X[:, t-1, :]

        # Drift term: -Theta * X_t * dt
        drift = -torch.matmul(X_curr, Theta.T) * dt

        # Diffusion term: Sigma * dW_t
        diffusion = torch.matmul(dW, Sigma.T)

        # Update state
        X[:, t, :] = X_curr + drift + diffusion

    # Normalize the output to roughly [-1, 1]
    # We can normalize each batch separately or globally. Global normalization per feature:
    max_vals, _ = torch.max(torch.abs(X), dim=1, keepdim=True)
    # Avoid division by zero
    max_vals = torch.clamp(max_vals, min=1e-5)
    X_normalized = X / max_vals

    return X_normalized

if __name__ == "__main__":
    # Quick test
    X = generate_ou_process(batch_size=2, seq_len=100)
    print("Shape:", X.shape)
    print("Min:", X.min().item(), "Max:", X.max().item())
