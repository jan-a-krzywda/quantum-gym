import torch
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_quantuminspire.qi_provider import QIProvider
from qiskit_aer import AerSimulator
import warnings

def build_reservoir_circuit(x_seq, c1=1.0, c2=1.0):
    """
    Builds the quantum reservoir circuit for a single sequence.
    x_seq: numpy array of shape (seq_len, 2)
    c1, c2: scaling constants
    """
    seq_len = x_seq.shape[0]
    num_qubits = 9
    measured_qubits = [0, 1, 2, 3, 5, 6]
    num_measured = len(measured_qubits)

    # We need a classical bit for each measured qubit at each time step
    qc = QuantumCircuit(num_qubits, seq_len * num_measured)

    # 3x3 grid nearest-neighbor pairs for CZ gates
    # Grid:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    cz_pairs = [
        (0, 1), (1, 2), # Horizontal row 0
        (3, 4), (4, 5), # Horizontal row 1
        (6, 7), (7, 8), # Horizontal row 2
        (0, 3), (3, 6), # Vertical col 0
        (1, 4), (4, 7), # Vertical col 1
        (2, 5), (5, 8)  # Vertical col 2
    ]

    for t in range(seq_len):
        x1_t, x2_t = x_seq[t]

        # 1. Apply Rx and Ry on measured qubits
        for q in measured_qubits:
            qc.rx(c1 * x1_t, q)
            qc.ry(c2 * x2_t, q)

        # 2. Apply a layer of CZ gates
        for pair in cz_pairs:
            qc.cz(pair[0], pair[1])

        # 3. Measurement (No reset!)
        # The classical register index needs to account for the time step
        cbits = [t * num_measured + i for i in range(num_measured)]
        qc.measure(measured_qubits, cbits)

    return qc

def run_quantum_reservoir(X, c1=1.0, c2=1.0, use_hardware=False):
    """
    Runs the quantum reservoir encoder on a batch of sequences.
    X: torch tensor of shape (batch_size, seq_len, 2)
    Returns: torch tensor of shape (batch_size, seq_len, 6)
    """
    batch_size, seq_len, _ = X.shape
    num_measured = 6

    # Try to connect to real hardware if requested
    backend = None
    if use_hardware:
        try:
            provider = QIProvider()
            backend = provider.get_backend(name='Tuna-9')
            print("Successfully connected to Tuna-9 hardware backend.")
        except Exception as e:
            warnings.warn(f"Failed to connect to Tuna-9 hardware: {e}. Falling back to AerSimulator.")

    if backend is None:
        backend = AerSimulator()

    binary_outputs = torch.zeros((batch_size, seq_len, num_measured))

    # Convert batch to numpy for qiskit
    X_np = X.numpy()

    circuits = []
    for i in range(batch_size):
        qc = build_reservoir_circuit(X_np[i], c1, c2)
        circuits.append(qc)

    # Transpile (using optimization_level=3 as in reference tuna script)
    tqc = transpile(circuits, backend, optimization_level=3)

    # Run simulation or hardware. We use shots=1 because the sequence itself is a single trajectory,
    # and we want to sample the specific quantum state collapse sequence.
    job = backend.run(tqc, shots=1, memory=True)
    result = job.result()

    for i in range(batch_size):
        # We used shots=1, memory=True means we get a list of bitstrings, length 1
        bitstring = result.get_memory(i)[0] # e.g. "101011 001101 ..." (but qiskit outputs without spaces)

        # Qiskit bitstring reads right-to-left.
        # The first measurement (t=0) corresponds to the classical bits [0...5],
        # which are at the RIGHT end of the string.
        for t in range(seq_len):
            for m_idx in range(num_measured):
                cbit_idx = t * num_measured + m_idx
                # Right-to-left mapping
                char = bitstring[-1 - cbit_idx]
                binary_outputs[i, t, m_idx] = 1.0 if char == '1' else 0.0

    return binary_outputs

if __name__ == "__main__":
    from data import generate_ou_process
    X = generate_ou_process(batch_size=2, seq_len=10)
    out = run_quantum_reservoir(X, c1=np.pi/2, c2=np.pi/2, use_hardware=True)
    print("Input shape:", X.shape)
    print("Output shape:", out.shape)
    print("Output snapshot:\n", out[0, :3, :])
