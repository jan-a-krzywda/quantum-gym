"""
Phase 1: Reverse-compilation via Backward Disentanglement Beam Search.

Strategy:
  1. Start from GHZ target state.
  2. Search backward: apply inverse ansatz layers to disentangle.
  3. Score by purity = Tr(ρ₀²) + Tr(ρ₁²) + Tr(ρ₂²).  Max = 3.0 (fully separable).
  4. Stop when purity > 2.99 (state is product).
  5. Invert sequence → guaranteed forward GHZ-prep circuit.

Backward layer structure (inverse of forward ansatz):
    CZ(0,1), CZ(1,2)  →  Rz(-phi_q) Ry(-theta_q) per qubit

Forward layer (used in inverted circuit and in build_ansatz_circuit):
    Ry(theta_q) Rz(phi_q) per qubit  →  CZ(0,1), CZ(1,2)

Parameters per round: [theta_0, phi_0, theta_1, phi_1, theta_2, phi_2]  (6 total)
Action space: [-ANGLE_INCREMENT, 0, +ANGLE_INCREMENT] per parameter  →  3^6 = 729 branches.
Angles accumulate: angles_{n+1} = angles_n + delta_{n+1}.

Usage:
    from beam_search import beam_search, ANGLE_INCREMENT
    trajectories = beam_search(beam_width=50)
    # returns forward prep circuits sorted by GHZ fidelity descending
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path
from typing import List

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_QUBITS        = 3
MAX_ROUNDS      = 20
DEVICE_STEP     = np.pi / 36         # device-native quantization (5 degrees)
SEARCH_STEP     = np.pi / 18         # beam-search step (10 degrees = 2 device steps)
                                     # π/2 = 9 × π/18 exactly → can reach optimal GHZ angle
ANGLE_INCREMENT = SEARCH_STEP        # alias used throughout beam search
CONNECTIVITY    = [(0, 1), (1, 2)]   # linear topology
PURITY_THRESH   = 2.99               # fully separable at 3.0

# Parameters per layer: [ry_0, ry_1, ry_2, cr_01, cr_12] = 5
# Forward ansatz:  Ry(ry_q) per qubit → CR(cr_01, 0→1) → CR(cr_12, 1→2)
# Backward layer:  CR(-cr_12, 1,2) → CR(-cr_01, 0,1) → Ry(-ry_q) per qubit
#
# Why CR(θ) for entanglement?
#   CR(θ) = controlled-Ry(θ): rotates the target qubit conditioned on the control.
#   For small θ it is near-identity → smooth Fubini-Study displacements ∝ θ. ✓
#   CR(θ)† = CR(-θ), so backward inversion is exact.
#   Unlike CPhase/CRz (diagonal), CR mixes |0⟩↔|1⟩ amplitudes, which is necessary
#   to build X-type GHZ correlations (|000⟩+|111⟩)/√2.
#   CR is native on the device (listed as CR, CRk).
#
# Why SEARCH_STEP = π/18?
#   The optimal single-qubit angle is π/2 = 9 × π/18, so the search can hit it exactly.
#   π/18 is also 2 × DEVICE_STEP, so circuits are still hardware-compatible.
N_PARAMS        = 5  # [ry_0, ry_1, ry_2, cr_01, cr_12]


def _require_qiskit():
    if not _HAS_QISKIT:
        raise ImportError("qiskit required: pip install qiskit")


def _quantize(angles: np.ndarray, step: float = DEVICE_STEP) -> np.ndarray:
    """Round angles to the nearest multiple of step (device-native quantization for hardware export)."""
    return np.round(angles / step) * step


def build_ansatz_circuit(angle_schedule: np.ndarray) -> "QuantumCircuit":
    """
    Forward ansatz circuit from |000>.

    angle_schedule : (n_rounds, 5) — [ry_0, ry_1, ry_2, cr_01, cr_12] per row.
    Each round:
        Ry(ry_q) on each qubit q
        CR(cr_01, control=0, target=1)   — controlled-Ry rotation
        CR(cr_12, control=1, target=2)

    CR(θ) = CRy(θ) mixes amplitudes (unlike CPhase which is diagonal),
    enabling X-type GHZ correlations. Small θ → near-identity → smooth manifold.
    """
    _require_qiskit()
    n_rounds = angle_schedule.shape[0]
    qc = QuantumCircuit(N_QUBITS)
    for r in range(n_rounds):
        row = angle_schedule[r]          # (5,)
        for q in range(N_QUBITS):
            qc.ry(float(row[q]), q)
        for k, (q0, q1) in enumerate(CONNECTIVITY):
            qc.cry(float(row[N_QUBITS + k]), q0, q1)   # CR = controlled-Ry
    return qc


def build_backward_circuit(sv_start: "Statevector", angle_schedule: np.ndarray) -> "Statevector":
    """
    Apply backward (inverse) layers to an existing statevector.

    Inverse of one forward layer:
        CR(-cr_12, 1, 2) → CR(-cr_01, 0, 1) → Ry(-ry_q) per qubit
    CR(θ)† = CR(-θ), so negating the angle gives the exact inverse.
    """
    _require_qiskit()
    sv = sv_start
    n_rounds = angle_schedule.shape[0]
    for r in range(n_rounds):
        row = angle_schedule[r]          # (5,)
        layer = QuantumCircuit(N_QUBITS)
        # Reverse CR gates (reversed order, negated angles)
        for k, (q0, q1) in reversed(list(enumerate(CONNECTIVITY))):
            layer.cry(-float(row[N_QUBITS + k]), q0, q1)
        # Reverse Ry gates (negated angles)
        for q in range(N_QUBITS):
            layer.ry(-float(row[q]), q)
        sv = sv.evolve(layer)
    return sv


def ghz_target_sv() -> "Statevector":
    """(|000> + |111>) / sqrt(2)."""
    _require_qiskit()
    qc = QuantumCircuit(N_QUBITS)
    qc.h(0)
    for q in range(N_QUBITS - 1):
        qc.cx(q, q + 1)
    return Statevector(qc)


# ---------------------------------------------------------------------------
# Purity scoring
# ---------------------------------------------------------------------------

def compute_purity_score(sv: "Statevector") -> float:
    """
    Purity score = Tr(ρ₀²) + Tr(ρ₁²) + Tr(ρ₂²).

    Max = 3.0 (fully separable product state).
    GHZ state = 1.5 (maximally entangled).
    """
    _require_qiskit()
    score = 0.0
    for q in range(N_QUBITS):
        others = [i for i in range(N_QUBITS) if i != q]
        rho_q = partial_trace(sv, others)
        rho_mat = np.asarray(rho_q.data)
        score += float(np.real(np.trace(rho_mat @ rho_mat)))
    return score


def extract_bloch_ry(sv: "Statevector") -> np.ndarray:
    """
    Extract the Ry angle for each qubit from a (near-)product Statevector.

    For a single-qubit state |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩, the Ry
    angle is θ = arccos(<Z>).  We ignore the azimuthal φ since the beam search
    will optimize the CPhase angles.

    Returns array of shape (5,): [ry_0, ry_1, ry_2, 0, 0]
    (CPhase angles initialized to zero — to be optimized by beam search).
    """
    _require_qiskit()
    angles = np.zeros(N_PARAMS)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    for q in range(N_QUBITS):
        others = [i for i in range(N_QUBITS) if i != q]
        rho_q = partial_trace(sv, others).data
        z = float(np.real(np.trace(rho_q @ Z)))
        angles[q] = float(np.arccos(np.clip(z, -1.0, 1.0)))
    return _quantize(angles)


# ---------------------------------------------------------------------------
# Beam search (backward)
# ---------------------------------------------------------------------------

def beam_search(
    beam_width: int = 50,
    max_rounds: int = MAX_ROUNDS,
    angle_increment: float = ANGLE_INCREMENT,
    purity_thresh: float = PURITY_THRESH,
    n_samples: int = 10,
    verbose: bool = True,
) -> List[dict]:
    """
    Forward beam search from |000> toward GHZ.

    Each step applies one ansatz layer: Ry(ry_q) → CPhase(cp_01, 0,1) → CPhase(cp_12, 1,2)
    and scores by GHZ fidelity.  Small parameter changes → small Fubini-Study displacements.
    We collect all intermediate states so the dataset contains smooth trajectories through
    Hilbert space.

    CPhase(θ) is continuously parameterized, giving smooth manifold geometry. ✓

    Returns trajectories sorted by final GHZ fidelity descending. Each dict:
        "angle_schedule"   : (n_rounds, 5) float64 — forward cumulative angles
        "actions"          : (n_rounds, 5) float64 — per-round angle deltas
        "distance_history" : list[float]            — trace distance from GHZ at each step
        "fidelity"         : float                  — final GHZ fidelity
    """
    _require_qiskit()
    from qiskit.quantum_info import Statevector as _SV
    ghz_sv = ghz_target_sv()
    zero_sv = _SV.from_label('0' * N_QUBITS)

    deltas_1d    = [-angle_increment, 0.0, angle_increment]
    all_combos   = np.array(list(product(range(3), repeat=N_PARAMS)), dtype=np.int8)  # (243, 5)
    delta_matrix = np.array(deltas_1d, dtype=np.float64)[all_combos]                  # (243, 5)

    def _fidelity(sv) -> float:
        from qiskit.quantum_info import state_fidelity as _sf
        return float(_sf(sv, ghz_sv))

    def _dist(fid: float) -> float:
        return float(np.sqrt(max(0.0, 1.0 - fid)))

    # --- Initialize forward beams from |000> ---
    # Two-pool initialization:
    #   Pool A (50%): small Ry + small CR  → smooth short-step exploration
    #   Pool B (50%): small Ry + CR near π → jump straight to near-GHZ entanglement
    # This ensures the search finds high-fidelity circuits quickly while
    # also populating the smooth-trajectory region of Hilbert space.
    beams = []
    n_pool_b = beam_width // 2
    for i in range(beam_width):
        ry_raw = np.random.uniform(-np.pi/9, np.pi/9, size=(1, N_QUBITS))
        if i < n_pool_b:
            # Pool A: small CR angles for smooth exploration
            cr_raw = np.random.uniform(-np.pi/9, np.pi/9, size=(1, len(CONNECTIVITY)))
        else:
            # Pool B: CR near π (optimal for GHZ), with small noise
            cr_raw = np.pi + np.random.uniform(-np.pi/9, np.pi/9, size=(1, len(CONNECTIVITY)))
        raw = np.column_stack([ry_raw, cr_raw])
        init_angles = _quantize(raw, angle_increment)
        from qiskit import QuantumCircuit as _QC
        layer = _QC(N_QUBITS)
        row = init_angles[0]
        for q in range(N_QUBITS):
            layer.ry(float(row[q]), q)
        for k, (q0, q1) in enumerate(CONNECTIVITY):
            layer.cry(float(row[N_QUBITS + k]), q0, q1)   # CR gate
        sv = zero_sv.evolve(layer)
        fid = _fidelity(sv)
        beams.append({
            "sched_f":        init_angles,
            "sv":             sv,
            "fid_history":    [0.0, fid],
            "_fidelity":      fid,
        })

    beams.sort(key=lambda x: -x["_fidelity"])
    global_best = list(beams)

    t0 = time.time()
    best_fid = global_best[0]["_fidelity"]

    if verbose:
        print(f"  Forward round 1/{max_rounds}  beams={len(beams)}  "
              f"best_fidelity={best_fid:.4f}  t={time.time()-t0:.1f}s")

    for round_idx in range(1, max_rounds):
        candidates = []
        for beam in beams:
            last_angles = beam["sched_f"][-1]   # (3,)

            if n_samples < len(delta_matrix):
                idx = np.random.choice(len(delta_matrix), size=n_samples, replace=False)
                sampled_deltas = delta_matrix[idx]
            else:
                sampled_deltas = delta_matrix

            for combo_deltas in sampled_deltas:
                new_angles_row = last_angles + combo_deltas   # (3,)
                new_sched_f = np.vstack([beam["sched_f"], new_angles_row])

                # Apply one new forward layer on top of current sv
                from qiskit import QuantumCircuit as _QC
                layer = _QC(N_QUBITS)
                for q in range(N_QUBITS):
                    layer.ry(float(new_angles_row[q]), q)
                for k, (q0, q1) in enumerate(CONNECTIVITY):
                    layer.cry(float(new_angles_row[N_QUBITS + k]), q0, q1)   # CR gate
                new_sv = beam["sv"].evolve(layer)

                fid = _fidelity(new_sv)
                candidates.append({
                    "sched_f":     new_sched_f,
                    "sv":          new_sv,
                    "fid_history": beam["fid_history"] + [fid],
                    "_fidelity":   fid,
                })

        candidates.sort(key=lambda x: -x["_fidelity"])

        # Deduplicate by fidelity
        unique_candidates = []
        seen = set()
        for c in candidates:
            key = round(c["_fidelity"], 5)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        candidates = unique_candidates

        # Hall of fame
        global_best.extend(candidates[:beam_width])
        global_best.sort(key=lambda x: -x["_fidelity"])
        unique_gb = []
        seen_gb = set()
        for b in global_best:
            key = round(b["_fidelity"], 5)
            if key not in seen_gb:
                seen_gb.add(key)
                unique_gb.append(b)
        global_best = unique_gb[:beam_width]

        # 80/20 exploration
        if len(candidates) > beam_width:
            n_top  = int(beam_width * 0.8)
            n_rand = beam_width - n_top
            beams  = candidates[:n_top]
            rest   = candidates[n_top:]
            if n_rand > 0 and len(rest) > 0:
                ri = np.random.choice(len(rest), size=min(n_rand, len(rest)), replace=False)
                beams.extend([rest[i] for i in ri])
            beams.sort(key=lambda x: -x["_fidelity"])
        else:
            beams = candidates

        best_fid = global_best[0]["_fidelity"]

        if verbose:
            print(f"  Forward round {round_idx + 1}/{max_rounds}  "
                  f"beams={len(beams)}  best_fidelity={best_fid:.4f}  "
                  f"t={time.time()-t0:.1f}s")

        if best_fid >= 0.999:
            if verbose:
                print(f"  Early stop: fidelity {best_fid:.4f} >= 0.999")
            break

    if verbose:
        print(f"Forward search done in {time.time()-t0:.1f}s. "
              f"Best fidelity={best_fid:.4f}.")

    # --- Package results ---
    result = []
    for b in global_best:
        sched_f   = b["sched_f"]       # (n_rounds, 3)
        n_rounds  = len(sched_f)
        fid_hist  = b["fid_history"]   # length n_rounds + 1 (includes step 0)

        actions_f = np.zeros_like(sched_f)
        actions_f[0] = sched_f[0]
        if n_rounds > 1:
            actions_f[1:] = np.diff(sched_f, axis=0)

        dist_history = [_dist(f) for f in fid_hist[1:]]   # one per round
        final_fidelity = float(state_fidelity(
            Statevector(build_ansatz_circuit(sched_f)), ghz_sv
        ))

        result.append({
            "angle_schedule":   sched_f,
            "actions":          actions_f,
            "distance_history": dist_history,
            "fidelity":         final_fidelity,
        })

    # Sort by best GHZ fidelity descending
    result.sort(key=lambda x: -x["fidelity"])

    if verbose:
        print(f"Top-20 forward circuits:")
        for i, t in enumerate(result[:20]):
            print(f"  [{i}] fidelity={t['fidelity']:.6f}  distance={t['distance_history'][-1]:.6f}  rounds={len(t['angle_schedule'])}")

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_trajectories(path: Path, trajectories: List[dict]) -> None:
    """Save trajectories as .npz (forward format for generate_dataset.py)."""
    n = len(trajectories)
    max_rounds = max(len(t["angle_schedule"]) for t in trajectories)

    angle_schedules = np.zeros((n, max_rounds, N_PARAMS), dtype=np.float64)
    actions         = np.zeros((n, max_rounds, N_PARAMS), dtype=np.float64)
    fidelities      = np.array([t["fidelity"] for t in trajectories])         # (n,)

    dist_histories = np.zeros((n, max_rounds), dtype=np.float64)
    lengths         = np.zeros(n, dtype=np.int32)
    for i, t in enumerate(trajectories):
        l = len(t["angle_schedule"])
        lengths[i] = l
        angle_schedules[i, :l] = t["angle_schedule"]
        actions[i, :l]         = t["actions"]
        dh = np.array(t["distance_history"])
        dist_histories[i, :len(dh)] = dh

    np.savez_compressed(
        path,
        angle_schedules=angle_schedules,
        actions=actions,
        distance_histories=dist_histories,
        fidelities=fidelities,
        lengths=lengths,
        n_params=np.array(N_PARAMS),
        angle_increment=np.array(ANGLE_INCREMENT),
        n_qubits=np.array(N_QUBITS),
    )


def load_trajectories(path: Path) -> List[dict]:
    """Load trajectories saved by save_trajectories."""
    raw = np.load(path, allow_pickle=False)
    angle_schedules = raw["angle_schedules"]
    actions         = raw["actions"]
    dist_histories  = raw["distance_histories"]
    fidelities      = raw["fidelities"]
    purity_histories = raw["purity_histories"] if "purity_histories" in raw.files \
        else np.zeros_like(dist_histories)
        
    if "lengths" in raw.files:
        lengths = raw["lengths"]
    else:
        lengths = [angle_schedules.shape[1]] * len(angle_schedules)
        
    n = len(angle_schedules)
    return [
        {
            "angle_schedule":   angle_schedules[i, :lengths[i]],
            "actions":          actions[i, :lengths[i]],
            "distance_history": list(dist_histories[i, :lengths[i]]),
            "fidelity":         float(fidelities[i]),
            "purity_history":   list(purity_histories[i, :lengths[i]+1]),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: Backward disentanglement beam search")
    p.add_argument("--beam-width",      type=int,   default=50)
    p.add_argument("--max-rounds",      type=int,   default=MAX_ROUNDS)
    p.add_argument("--angle-increment", type=float, default=ANGLE_INCREMENT)
    p.add_argument("--purity-thresh",   type=float, default=PURITY_THRESH)
    p.add_argument("--n-samples",       type=int,   default=10,
                   help="Random action samples per beam (default: 10)")
    p.add_argument("--out",             type=Path,  default=None,
                   help="Output .npz path (default: data/beam_trajectories.npz)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    trajectories = beam_search(
        beam_width=args.beam_width,
        max_rounds=args.max_rounds,
        angle_increment=args.angle_increment,
        purity_thresh=args.purity_thresh,
        n_samples=args.n_samples,
        verbose=verbose,
    )

    out_path = args.out
    if out_path is None:
        data_dir = Path(__file__).resolve().parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_path = data_dir / "beam_trajectories.npz"

    save_trajectories(out_path, trajectories)
    size_mb = out_path.stat().st_size / 1e6

    if verbose:
        print(f"\nSaved {len(trajectories)} forward circuits → {out_path}  ({size_mb:.2f} MB)")
        for i, t in enumerate(trajectories[:20]):
            print(f"  [{i}] fidelity={t['fidelity']:.6f}  "
                  f"distance={t['distance_history'][-1]:.6f}  "
                  f"rounds={len(t['angle_schedule'])}")


if __name__ == "__main__":
    main()
