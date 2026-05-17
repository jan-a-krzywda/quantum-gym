"""
Master benchmark: Active Learning loop comparing 3 neural world models.

Loop per model
--------------
For step = 1 … MAX_STEPS:
  a. Agent selects 20 bases via Expected Free Energy.
  b. QuantumEnvironment simulates those measurements.
  c. Shots added to replay buffer.
  d. Model trained for TRAIN_EPOCHS on full buffer.
  e. Every EVAL_EVERY steps: compute and log Stabilizer Fidelity.

Metric: Stabilizer Fidelity vs. Total Quantum Shots taken.
  stab_fidelity = mean_i (1 + <K_i>_model) / 2   ∈ [0, 1]
  random model → ~0.5, perfect model → 1.0

Config
------
N_QUBITS   = 6   (3^6 = 729 candidate bases — tractable)
MAX_STEPS  = 80  → 80 × 20 = 1600 shots per model
BATCH_SIZE = 20  shots per active learning step
"""
import sys
import os
import random
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as iproduct

# Make sure the package root is on the path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shadow_gym.src.quantum_environment import QuantumEnvironment
from shadow_gym.src.utils import cluster_state_stabilizers
from shadow_gym.src.world_model_low import LowAgent
from shadow_gym.src.world_model_mid import MidAgent
from shadow_gym.src.world_model_high import HighAgent

# ── Hyperparameters ──────────────────────────────────────────────────────────
N_QUBITS = 5
BATCH_SIZE = 10          # shots acquired per active-learning step
MAX_STEPS = 100           # total active-learning steps
EVAL_EVERY = 5          # evaluate stabilizer fidelity every N steps
TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 64
N_EFE_SAMPLES = 40       # hallucinated samples per candidate for EFE
N_FIDELITY_SAMPLES = 400 # samples used when evaluating stabilizer fidelity
LEARNING_RATE = 1e-3
SEED = 42

# For n_qubits > ENUMERATE_THRESHOLD, enumerate all 3^N is intractable.
# Instead, build CANDIDATE_POOL_SIZE stabilizer-compatible bases per step.
ENUMERATE_THRESHOLD = 7   # 3^7 = 2187 — still manageable
CANDIDATE_POOL_SIZE  = 800 # size of subsampled candidate pool for large n_qubits

# ── Colours / styles ─────────────────────────────────────────────────────────
PALETTE = {
    "Low (Autoregressive NQS)":  ("#e74c3c", "o-"),
    "Mid (Autoregressive VAE)":  ("#2ecc71", "s-"),
    "High (MADE VAE)":           ("#3498db", "^-"),
    "Random (baseline)":         ("#95a5a6", "x--"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Random-selection baseline (no active inference)
# ─────────────────────────────────────────────────────────────────────────────

class RandomAgent:
    """Uniform random basis selection — no learning of EFE."""
    MODEL_NAME = "Random (baseline)"

    def __init__(self, n_qubits: int = 6, stabilizers=None, lr: float = 1e-3):
        from shadow_gym.src.world_model_low import LowAgent
        self._inner = LowAgent(n_qubits=n_qubits, stabilizers=stabilizers, lr=lr)
        self.MODEL_NAME = "Random (baseline)"

    def update_beliefs(self, shots, epochs=5, batch_size=64):
        self._inner.update_beliefs(shots, epochs, batch_size)

    def select_batch(self, candidate_bases, batch_size=20, n_efe_samples=40):
        indices = np.random.choice(len(candidate_bases), size=batch_size, replace=False)
        return [candidate_bases[i] for i in indices]

    def compute_stabilizer_fidelity(self, n_samples=400):
        return self._inner.compute_stabilizer_fidelity(n_samples)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate pool helpers
# ─────────────────────────────────────────────────────────────────────────────

def _all_bases(n_qubits: int):
    """Enumerate all 3^n bases (only called when n_qubits <= ENUMERATE_THRESHOLD)."""
    return [list(c) for c in iproduct(["X", "Y", "Z"], repeat=n_qubits)]


def _sample_candidate_pool(stabilizers, n_qubits: int, n_candidates: int):
    """
    Sample n_candidates stabilizer-compatible bases.
    For each candidate: pick a random stabilizer, keep its non-I positions,
    randomise the I positions.  Guarantees every candidate is compatible with
    at least one target stabilizer, providing a useful EFE signal.
    """
    paulis = ["X", "Y", "Z"]
    candidates = set()
    max_iter = n_candidates * 20
    for _ in range(max_iter):
        if len(candidates) >= n_candidates:
            break
        stab = random.choice(stabilizers)
        basis = tuple(p if p != "I" else random.choice(paulis) for p in stab)
        candidates.add(basis)
    # Top up with purely random bases if needed
    while len(candidates) < n_candidates:
        candidates.add(tuple(random.choice(paulis) for _ in range(n_qubits)))
    return [list(b) for b in list(candidates)[:n_candidates]]


def get_candidate_pool(stabilizers, n_qubits: int, regen: bool = False):
    """
    Return candidate basis pool.
    Small n_qubits (≤ ENUMERATE_THRESHOLD): enumerate all 3^n.
    Large n_qubits: sample CANDIDATE_POOL_SIZE stabilizer-compatible bases.
    """
    if n_qubits <= ENUMERATE_THRESHOLD:
        return _all_bases(n_qubits)
    return _sample_candidate_pool(stabilizers, n_qubits, CANDIDATE_POOL_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# Single model benchmark run
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(agent_class, env, stabilizers, label, seed):
    np.random.seed(seed)
    random.seed(seed)
    agent = agent_class(
        n_qubits=N_QUBITS,
        stabilizers=stabilizers,
        lr=LEARNING_RATE,
    )

    shots_at_eval = []
    fidelity_at_eval = []
    total_shots = 0
    t0 = time.time()

    # Pre-enumerate small n_qubits; re-sample each step for large n_qubits
    static_pool = (N_QUBITS <= ENUMERATE_THRESHOLD)
    if static_pool:
        pool = get_candidate_pool(stabilizers, N_QUBITS)
        print(f"  candidate pool: all {len(pool)} bases (3^{N_QUBITS})")
    else:
        print(f"  candidate pool: {CANDIDATE_POOL_SIZE} stabilizer-compatible bases per step "
              f"(3^{N_QUBITS}={3**N_QUBITS:,} total bases)")

    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")

    for step in range(1, MAX_STEPS + 1):
        # ── (a) Candidate pool ───────────────────────────────────────────
        if not static_pool:
            pool = get_candidate_pool(stabilizers, N_QUBITS, regen=True)

        # ── (b) Select batch of bases via EFE ────────────────────────────
        selected = agent.select_batch(
            pool,
            batch_size=BATCH_SIZE,
            n_efe_samples=N_EFE_SAMPLES,
        )

        # ── (c) Simulate measurements ────────────────────────────────────
        shots = env.sample_classical(n_shots=BATCH_SIZE, bases=selected)
        total_shots += BATCH_SIZE

        # ── (d) Update world model ───────────────────────────────────────
        agent.update_beliefs(shots, epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH_SIZE)

        # ── (e) Evaluate stabilizer fidelity ─────────────────────────────
        if step % EVAL_EVERY == 0:
            fid = agent.compute_stabilizer_fidelity(n_samples=N_FIDELITY_SAMPLES)
            shots_at_eval.append(total_shots)
            fidelity_at_eval.append(fid)
            elapsed = time.time() - t0
            print(
                f"  step {step:3d}/{MAX_STEPS} | shots {total_shots:5d} "
                f"| StabFid {fid:.4f} | {elapsed:5.1f}s"
            )

    return np.array(shots_at_eval), np.array(fidelity_at_eval)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for label, (shots, fid) in results.items():
        colour, ls = PALETTE.get(label, ("#333333", "-"))
        ax.plot(shots, fid, ls, color=colour, label=label,
                linewidth=2, markersize=5, alpha=0.9)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.35, linewidth=1.2)
    ax.set_xlabel("Total quantum shots", fontsize=12)
    ax.set_ylabel("Stabilizer fidelity", fontsize=12)
    ax.set_title(
        f"Active Inference: Stabilizer Fidelity vs. Shots\n"
        f"({N_QUBITS}-qubit cluster state, batch={BATCH_SIZE})",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # ── Rate-of-learning panel: fidelity gain per shot ──────────────────
    ax2 = axes[1]
    for label, (shots, fid) in results.items():
        if len(shots) < 2:
            continue
        colour, ls = PALETTE.get(label, ("#333333", "-"))
        # Δfidelity / Δshots
        dshots = np.diff(shots)
        dfid = np.diff(fid)
        mid_shots = 0.5 * (shots[:-1] + shots[1:])
        rate = dfid / dshots
        ax2.plot(mid_shots, rate, ls, color=colour, label=label,
                 linewidth=1.5, markersize=4, alpha=0.8)
    ax2.axhline(0, color="black", linestyle=":", alpha=0.3, linewidth=1)
    ax2.set_xlabel("Total quantum shots", fontsize=12)
    ax2.set_ylabel("ΔFidelity / ΔShots (learning rate)", fontsize=12)
    ax2.set_title("Sample Efficiency (higher is better)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    threshold = 0.80
    print(f"\n{'='*62}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*62}")
    header = f"{'Model':<30} {'Final Fid':>10} {'Shots→{:.0f}%'.format(threshold*100):>12}"
    print(header)
    print("-" * 62)
    for label, (shots, fid) in results.items():
        final = fid[-1] if len(fid) else float("nan")
        above = shots[fid >= threshold]
        shots_to = int(above[0]) if len(above) else None
        s = f"{str(shots_to):>12}" if shots_to else f"{'N/A':>12}"
        print(f"{label:<30} {final:>10.4f} {s}")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Benchmark config: N_QUBITS={N_QUBITS}, MAX_STEPS={MAX_STEPS}, "
          f"BATCH_SIZE={BATCH_SIZE}, SEED={SEED}")

    # ── Environment ──────────────────────────────────────────────────────
    env = QuantumEnvironment(n_data=N_QUBITS)
    env.prepare_cluster_state()

    stabilizers = cluster_state_stabilizers(N_QUBITS)
    print(f"\n{N_QUBITS}-qubit cluster stabilizers:")
    for s in stabilizers:
        print(f"  {s}")

    # ── Models to benchmark ──────────────────────────────────────────────
    models = [
        (LowAgent,    "Low (Autoregressive NQS)"),
        (MidAgent,    "Mid (Autoregressive VAE)"),
        (HighAgent,   "High (MADE VAE)"),
        (RandomAgent, "Random (baseline)"),
    ]

    results = {}
    for agent_class, label in models:
        shots, fid = run_benchmark(
            agent_class, env, stabilizers, label, seed=SEED
        )
        results[label] = (shots, fid)

    # ── Output ───────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "benchmark_results.png")
    plot_results(results, out_path)
    print_summary(results)


if __name__ == "__main__":
    main()
