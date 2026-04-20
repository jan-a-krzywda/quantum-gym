"""
Multiqubit zebra-style fingerprint generator (pure simulation, no hardware).

For each quantum state |ψ⟩, sweeps Ry(θ) on all qubits over n_theta angles,
samples n_shots Z-basis measurements, and stacks results into a binary fingerprint.

Fingerprint layout — qubit-major (mirrors fid_data_io.stack_to_vae_tensors):
    [q0@θ₀..θ_{n-1}, q1@θ₀..θ_{n-1}, ..., q_{Q-1}@θ₀..θ_{n-1}]
    shape: (n_shots, n_qubits * n_theta), uint8

This makes each qubit's angular sweep a contiguous stripe — same "persona" idea
as the hardware data, so the same QubitConvVAE(seq_len=n_qubits*n_theta) applies.

Requires: qiskit, qiskit-aer (or qiskit.quantum_info Statevector, which is pure Python).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import partial_trace

try:
    from qiskit.quantum_info import Statevector
    from qiskit import QuantumCircuit

    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_THETA_DEFAULT = 40
N_SHOTS_DEFAULT = 1000
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_qiskit() -> None:
    if not _HAS_QISKIT:
        raise ImportError("qiskit required: pip install qiskit")


# ---------------------------------------------------------------------------
# State preparation helpers
# ---------------------------------------------------------------------------


def prepare_zero_state(n_qubits: int) -> "Statevector":
    """Return |00…0⟩."""
    _require_qiskit()
    qc = QuantumCircuit(n_qubits)
    return Statevector(qc)


def prepare_plus_state(n_qubits: int) -> "Statevector":
    """Return |++…+⟩ = H^⊗n |00…0⟩."""
    _require_qiskit()
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    return Statevector(qc)


def prepare_hypergraph_3q() -> "Statevector":
    """
    3-qubit hypergraph state: CCZ(0,1,2) · H³ |000⟩.

    CCZ applies −1 phase when all three qubits are |1⟩, making this the
    canonical 3-uniform hypergraph state on the triangle {0,1,2}.
    """
    _require_qiskit()
    qc = QuantumCircuit(3)
    qc.h([0, 1, 2])
    qc.ccz(0, 1, 2)
    return Statevector(qc)


def prepare_ghz_state(n_qubits: int) -> "Statevector":
    """Return GHZ = (|00…0⟩ + |11…1⟩) / √2."""
    _require_qiskit()
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    return Statevector(qc)


# Map name → factory (no-arg callables for 3-qubit baselines)
BASELINE_STATES_3Q: Dict[str, callable] = {
    "|000>": lambda: prepare_zero_state(3),
    "|+++>": lambda: prepare_plus_state(3),
    "GHZ":   lambda: prepare_ghz_state(3),
    "hypergraph_CCZ": prepare_hypergraph_3q,
}


# ---------------------------------------------------------------------------
# Core fingerprint generator
# ---------------------------------------------------------------------------


def fingerprint_from_statevector(
    sv: "Statevector",
    n_qubits: int,
    n_theta: int = N_THETA_DEFAULT,
    n_shots: int = N_SHOTS_DEFAULT,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Zebra-style fingerprint from a Qiskit Statevector.

    For each of n_theta angles θ ∈ [0, π), applies Ry(θ) to all qubits
    simultaneously, samples n_shots Z-basis outcomes, and unpacks per-qubit bits.

    Parameters
    ----------
    sv        : input Statevector (any n_qubits)
    n_qubits  : must match sv dimension (2**n_qubits)
    n_theta   : number of rotation angles (default 40)
    n_shots   : single-shot samples per angle (default 1000)
    rng       : numpy Generator for reproducibility

    Returns
    -------
    fingerprint : ndarray (n_shots, n_qubits * n_theta), uint8
        Layout: [q0@θ₀..θ_{n-1}, q1@θ₀..θ_{n-1}, ...]
    """
    _require_qiskit()
    if rng is None:
        rng = np.random.default_rng()

    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    fingerprint = np.zeros((n_shots, n_qubits * n_theta), dtype=np.uint8)

    for k, theta in enumerate(thetas):
        # Ry(theta) on all qubits simultaneously
        rot_qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            rot_qc.ry(float(theta), q)
        rotated = sv.evolve(rot_qc)

        # Probabilities over 2^n_qubits computational basis states
        probs = rotated.probabilities()  # shape (2**n_qubits,)

        # Sample n_shots outcomes (integers 0..2^n_qubits-1)
        outcomes = rng.choice(len(probs), size=n_shots, p=probs)

        # Unpack bits: Qiskit little-endian → qubit q = bit q of outcome integer
        for q in range(n_qubits):
            fingerprint[:, q * n_theta + k] = (outcomes >> q) & 1

    return fingerprint


def fingerprint_from_circuit(
    qc: "QuantumCircuit",
    n_theta: int = N_THETA_DEFAULT,
    n_shots: int = N_SHOTS_DEFAULT,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Convenience: circuit → Statevector → fingerprint."""
    _require_qiskit()
    sv = Statevector(qc)
    return fingerprint_from_statevector(
        sv, qc.num_qubits, n_theta=n_theta, n_shots=n_shots, rng=rng
    )


# ---------------------------------------------------------------------------
# Transition pair generation (for RL transition dataset)
# ---------------------------------------------------------------------------

# Action set for 3-qubit system
ACTION_NAMES_3Q = [
    "H0", "H1", "H2",
    "T0", "T1", "T2",
    "CZ01", "CZ12"
]


def _apply_action(sv: "Statevector", action: str) -> "Statevector":
    """Apply named action gate to a Statevector and return new Statevector."""
    _require_qiskit()
    n_qubits = int(round(np.log2(len(sv.data))))
    qc = QuantumCircuit(n_qubits)
    if action.startswith("H"):
        qc.h(int(action[1]))
    elif action.startswith("T"):
        qc.t(int(action[1]))
    elif action.startswith("CZ"):
        q0, q1 = int(action[2]), int(action[3])
        qc.cz(q0, q1)
    else:
        raise ValueError(f"Unknown action: {action!r}")
    return sv.evolve(qc)


def apply_action(sv: "Statevector", action: str) -> "Statevector":
    """Public wrapper: apply one named gate action to sv."""
    return _apply_action(sv, action)


def make_transition_pairs(
    base_states: Dict[str, "Statevector"],
    action_names: Sequence[str] = ACTION_NAMES_3Q,
    n_theta: int = N_THETA_DEFAULT,
    n_shots: int = N_SHOTS_DEFAULT,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    For each base state and each action, fingerprint both |ψ⟩ and a(|ψ⟩).

    Returns dict with keys:
        'state_names'   : list of N base state names
        'action_names'  : list of A action names
        'fingerprints'  : ndarray (N, n_shots, n_qubits*n_theta) — base state fps
        'next_fps'      : ndarray (N, A, n_shots, n_qubits*n_theta) — post-action fps
        'n_theta'       : int
        'n_shots'       : int
    """
    _require_qiskit()
    if rng is None:
        rng = np.random.default_rng()

    state_names = list(base_states.keys())
    N = len(state_names)
    A = len(action_names)

    # Infer n_qubits from first state
    first_sv = next(iter(base_states.values()))
    n_qubits = int(round(np.log2(len(first_sv.data))))
    seq_len = n_qubits * n_theta

    fingerprints = np.zeros((N, n_shots, seq_len), dtype=np.uint8)
    next_fps = np.zeros((N, A, n_shots, seq_len), dtype=np.uint8)

    for i, name in enumerate(state_names):
        sv = base_states[name]
        fingerprints[i] = fingerprint_from_statevector(
            sv, n_qubits, n_theta=n_theta, n_shots=n_shots, rng=rng
        )
        for j, action in enumerate(action_names):
            sv_next = _apply_action(sv, action)
            next_fps[i, j] = fingerprint_from_statevector(
                sv_next, n_qubits, n_theta=n_theta, n_shots=n_shots, rng=rng
            )

    return {
        "state_names": state_names,
        "action_names": list(action_names),
        "fingerprints": fingerprints,
        "next_fps": next_fps,
        "n_theta": n_theta,
        "n_shots": n_shots,
        "n_qubits": n_qubits,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_fingerprint_npz(
    path: Path,
    fingerprints: np.ndarray,
    *,
    state_names: Optional[Sequence[str]] = None,
    action_names: Optional[Sequence[str]] = None,
    next_fps: Optional[np.ndarray] = None,
    n_theta: int = N_THETA_DEFAULT,
    n_shots: int = N_SHOTS_DEFAULT,
    n_qubits: int = 3,
    **extra,
) -> None:
    """Save fingerprint dataset to .npz."""
    arrays: Dict[str, np.ndarray] = {
        "fingerprints": fingerprints,
        "n_theta": np.array(n_theta),
        "n_shots": np.array(n_shots),
        "n_qubits": np.array(n_qubits),
    }
    if state_names is not None:
        arrays["state_names"] = np.array(state_names)
    if action_names is not None:
        arrays["action_names"] = np.array(action_names)
    if next_fps is not None:
        arrays["next_fps"] = next_fps
    for k, v in extra.items():
        arrays[k] = np.asarray(v)
    np.savez_compressed(path, **arrays)


def load_fingerprint_npz(path: Path) -> Dict:
    """Load .npz saved by save_fingerprint_npz. Returns plain dict."""
    raw = np.load(path, allow_pickle=True)
    out: Dict = {}
    for k in raw.files:
        v = raw[k]
        if v.ndim == 0:
            out[k] = v.item()
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _load_zebra_style():
    """Import zebra_gif_2d_style from quantum_code (best-effort)."""
    qc_dir = str(_REPO_ROOT / "quantum_code")
    if qc_dir not in sys.path:
        sys.path.insert(0, qc_dir)
    try:
        from zebra_plot_style import zebra_gif_2d_style
        return zebra_gif_2d_style
    except ImportError:
        return None


def plot_zebra_fingerprint(
    fingerprint: np.ndarray,
    n_qubits: int,
    n_theta: int = N_THETA_DEFAULT,
    *,
    title: str = "",
    axes=None,
    max_shots: int = 200,
    thetas: Optional[np.ndarray] = None,
):
    """
    Zebra imshow of fingerprint — one panel per qubit.

    fingerprint : (n_shots, n_qubits * n_theta)
    Returns (fig, axes).
    """
    import matplotlib.pyplot as plt

    zebra_style_fn = _load_zebra_style()
    if zebra_style_fn is not None:
        style = zebra_style_fn(n_qubits, n_theta)
        figsize = style["figsize"]
    else:
        figsize = (min(9.8, 3.0 * n_qubits + 1.0), 5.0)

    n_show = min(len(fingerprint), max_shots)
    data = fingerprint[:n_show]

    if axes is None:
        fig, axs = plt.subplots(1, n_qubits, figsize=figsize, sharey=True)
        if n_qubits == 1:
            axs = [axs]
    else:
        axs = list(axes)
        fig = axs[0].get_figure()

    if thetas is None:
        thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)

    for q, ax in enumerate(axs):
        panel = data[:, q * n_theta: (q + 1) * n_theta]  # (n_show, n_theta)
        ax.imshow(
            panel,
            aspect="auto",
            cmap="binary",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            origin="upper",
            extent=[thetas[0], thetas[-1], n_show, 0],
        )
        ax.set_title(f"q{q}", fontsize=10)
        ax.set_xlabel("θ (rad)", fontsize=9)
        if q == 0:
            ax.set_ylabel("shot", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig, axs


def plot_state_fingerprints(
    named_fingerprints: Dict[str, np.ndarray],
    n_qubits: int,
    n_theta: int = N_THETA_DEFAULT,
    max_shots: int = 100,
) -> "plt.Figure":
    """
    Grid of zebra panels: one row per state, one column group per qubit.

    named_fingerprints : {state_name: (n_shots, n_qubits*n_theta)}
    """
    import matplotlib.pyplot as plt

    names = list(named_fingerprints.keys())
    n_states = len(names)
    fig, axes = plt.subplots(
        n_states,
        n_qubits,
        figsize=(3.0 * n_qubits, 2.5 * n_states),
        sharey="row",
        sharex="col",
        squeeze=False,
    )
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)

    for row, name in enumerate(names):
        fp = named_fingerprints[name]
        n_show = min(len(fp), max_shots)
        data = fp[:n_show]
        for q in range(n_qubits):
            ax = axes[row, q]
            panel = data[:, q * n_theta: (q + 1) * n_theta]
            ax.imshow(
                panel,
                aspect="auto",
                cmap="binary",
                vmin=0,
                vmax=1,
                interpolation="nearest",
                origin="upper",
                extent=[thetas[0], thetas[-1], n_show, 0],
            )
            if row == 0:
                ax.set_title(f"q{q}", fontsize=9)
            if q == 0:
                ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=50, va="center")
            if row == n_states - 1:
                ax.set_xlabel("θ (rad)", fontsize=8)

    fig.tight_layout()
    return fig


def compute_entanglement_entropy(sv: "Statevector", subsystem: Sequence[int]) -> float:
    """Compute the von Neumann entropy of the reduced density matrix."""
    reduced_dm = partial_trace(sv, subsystem)  # Directly use partial_trace on Statevector
    eigenvalues = np.real(np.linalg.eigvalsh(reduced_dm.data))
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-12))  # Avoid log(0)
    return entropy
