import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data import generate_ou_process
from quantum_reservoir import run_quantum_reservoir
from decoder import ClassicalDecoder

def calculate_1d_acf(x, max_lag):
    """
    Calculates the 1D Autocorrelation Function (ACF) for a batch of sequences.
    x: torch tensor of shape (batch_size, seq_len, features)
    Returns: torch tensor of shape (batch_size, max_lag, features)
    """
    batch_size, seq_len, features = x.shape
    acf = torch.zeros(batch_size, max_lag, features, device=x.device)

    # Subtract mean along sequence dimension for each batch and feature
    mean = torch.mean(x, dim=1, keepdim=True)
    x_centered = x - mean

    # Calculate variance (lag=0) to normalize
    var = torch.sum(x_centered * x_centered, dim=1) + 1e-8

    for lag in range(max_lag):
        if lag == 0:
            acf[:, lag, :] = 1.0
        else:
            # Shift sequences
            x_t = x_centered[:, :-lag, :]
            x_t_plus_lag = x_centered[:, lag:, :]
            # Compute covariance
            cov = torch.sum(x_t * x_t_plus_lag, dim=1)
            acf[:, lag, :] = cov / var

    return acf

def acf_mse_loss(preds, targets, max_lag=10):
    """
    Calculates MSE between the ACF of predictions and targets.
    """
    pred_acf = calculate_1d_acf(preds, max_lag)
    target_acf = calculate_1d_acf(targets, max_lag)
    return nn.MSELoss()(pred_acf, target_acf)

def train(epochs=100, batch_size=4, seq_len=50, lambda_val=0.5, use_hardware=False):
    # Initialize the decoder model
    model = ClassicalDecoder()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    mse_criterion = nn.MSELoss()

    print("Generating dataset...")
    # Generate train and validation data
    # (Since the quantum reservoir is slow to simulate, we generate a static dataset)
    X_train = generate_ou_process(batch_size=32, seq_len=seq_len)

    print("Running quantum reservoir encoder (this might take a minute)...")
    # c1 and c2 can be tuned. We use pi/2
    encoded_train = run_quantum_reservoir(X_train, c1=np.pi/2, c2=np.pi/2, use_hardware=use_hardware)

    dataset = torch.utils.data.TensorDataset(encoded_train, X_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_traj_loss = 0.0
        total_acf_loss = 0.0

        for batch_encoded, batch_targets in dataloader:
            optimizer.zero_grad()

            # Forward pass (decoder only)
            preds = model(batch_encoded)

            # Trajectory Loss
            loss_traj = mse_criterion(preds, batch_targets)

            # Statistical Loss (ACF)
            # Use max_lag up to seq_len // 2
            loss_acf = acf_mse_loss(preds, batch_targets, max_lag=min(10, seq_len//2))

            # Total Loss
            loss = loss_traj + lambda_val * loss_acf

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_traj_loss += loss_traj.item()
            total_acf_loss += loss_acf.item()

        num_batches = len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {total_loss/num_batches:.4f} "
                  f"(Traj: {total_traj_loss/num_batches:.4f}, "
                  f"ACF: {total_acf_loss/num_batches:.4f})")

    # Save the model
    torch.save(model.state_dict(), "decoder.pth")
    print("Model saved to decoder.pth")

    # Save a sample for plotting
    torch.save(encoded_train[0], "sample_encoded.pt")
    torch.save(X_train[0], "sample_target.pt")
    model.eval()
    with torch.no_grad():
        sample_pred = model(encoded_train[0:1])[0]
    torch.save(sample_pred, "sample_pred.pt")

if __name__ == "__main__":
    # pass use_hardware=True to connect to Tuna-9 (will fallback to AerSimulator if not authenticated)
    train(epochs=50, use_hardware=True)
