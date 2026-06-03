import torch
import pytest
from data import generate_ou_process
from decoder import ClassicalDecoder
from train import calculate_1d_acf
from quantum_reservoir import run_quantum_reservoir
import numpy as np

def test_data_generation():
    batch_size = 2
    seq_len = 10
    X = generate_ou_process(batch_size, seq_len)

    assert X.shape == (batch_size, seq_len, 2)
    # Check normalization
    assert torch.all(X >= -1.0) and torch.all(X <= 1.0)

def test_classical_decoder():
    model = ClassicalDecoder()
    batch_size = 2
    seq_len = 10
    input_dim = 6

    dummy_input = torch.zeros((batch_size, seq_len, input_dim))
    out = model(dummy_input)

    assert out.shape == (batch_size, seq_len, 2)

def test_calculate_1d_acf():
    batch_size = 2
    seq_len = 20
    features = 2
    max_lag = 5

    # Create a simple tensor
    x = torch.randn(batch_size, seq_len, features)
    acf = calculate_1d_acf(x, max_lag)

    assert acf.shape == (batch_size, max_lag, features)
    # The ACF at lag 0 should be approximately 1
    assert torch.allclose(acf[:, 0, :], torch.ones(batch_size, features), atol=1e-5)

def test_quantum_reservoir_simulation():
    batch_size = 1
    seq_len = 5
    X = generate_ou_process(batch_size, seq_len)

    # Test with simulator
    out = run_quantum_reservoir(X, use_hardware=False)

    assert out.shape == (batch_size, seq_len, 6)
    # Values should be 0 or 1
    assert torch.all((out == 0) | (out == 1))
