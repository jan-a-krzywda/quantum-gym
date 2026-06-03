# StoQastiQ

A hybrid Quantum-Classical model to simulate and decode a 2D Ornstein-Uhlenbeck (O-U) stochastic process.

## Architecture

1. **Classical Data Generation** (`data.py`):
   Simulates a 2D O-U process using the Euler-Maruyama method. The output is a time-series tensor of shape `(batch_size, sequence_length, 2)`. The generated trajectory has cross-correlation between the two dimensions.

2. **Quantum Reservoir Encoder** (`quantum_reservoir.py`):
   A 9-qubit quantum reservoir implemented with Qiskit. It acts as an encoder. Qubits are arranged in a 3x3 grid, and data is re-uploaded at each time step. Measured qubits collapse to a binary sequence, and are *not* reset between steps.
   By default, the `use_hardware` parameter attempts to connect to the `Tuna-9` hardware via Quantum Inspire (`QIProvider`). If unauthenticated, it safely falls back to a Qiskit `AerSimulator`.

3. **Classical Decoder** (`decoder.py`):
   A PyTorch-based sequence decoder consisting of:
   - An RC Embedding linear layer to project the binary quantum output to a continuous latent space.
   - An LSTM layer.
   - A linear Readout layer to reconstruct the 2D O-U process.

4. **Training and Losses** (`train.py`):
   Trains the classical decoder to reconstruct the O-U process from the quantum reservoir's "zebra plot" output. The loss is a combination of:
   - MSE Trajectory Loss (`nn.MSELoss`)
   - Custom 1D Autocorrelation (ACF) MSE Loss

5. **Visualization** (`plot.py`):
   Plots the generated O-U process, the 6-bit "zebra plot" output from the quantum reservoir, and the reconstructed trajectory compared to the target.

## Running the project

```bash
# 1. Generate data, run the reservoir, train the decoder, and save sample data
python train.py

# 2. Visualize the results
python plot.py
```
