import sys
import os
import numpy as np
import optuna
import logging

# Ensure we can import from the shadow_gym package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from shadow_gym.src import QuantumEnvironment, ShadowProcessor, ActiveInferenceAgent
from shadow_gym.src.utils import fidelity

# Reduce Optuna verbosity so it doesn't flood the console
optuna.logging.set_verbosity(optuna.logging.INFO)

# ---------------------------------------------------------
# Search Configuration
# ---------------------------------------------------------
N_QUBITS = 4
N_SHOTS = 500       # Smaller budget per trial to speed up search
BATCH_SIZE = 20
N_TRIALS = 50       # Number of parameter combinations to try

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function. Suggests a set of hyperparameters,
    runs the Active Inference agent, and returns the final state fidelity.
    """
    # 1. Sample hyperparameters from the search space
    alpha = trial.suggest_float('alpha', 0.5, 3.0)
    chi_stabilizer = trial.suggest_float('chi_stabilizer', 1.0, 100.0, log=True)
    temperature = trial.suggest_float('temperature', 0.01, 5.0, log=True)
    ucb_c = trial.suggest_float('ucb_c', 0.1, 5.0)

    # 2. Initialize Environment and Target State (Noisy 4-qubit Cluster)
    env = QuantumEnvironment(N_QUBITS)
    env.prepare_cluster_state(depolarizing_p=0.2)
    rho_true = env.rho_true
    proc = ShadowProcessor(N_QUBITS)

    # 3. Initialize Agent with suggested parameters
    agent = ActiveInferenceAgent(
        n_qubits=N_QUBITS,
        alpha=alpha,
        chi_stabilizer=chi_stabilizer,
        temperature=temperature,
        ucb_c=ucb_c
    )

    # 4. Run the Active Inference Loop
    all_shots = []
    while len(all_shots) < N_SHOTS:
        this_batch = min(BATCH_SIZE, N_SHOTS - len(all_shots))
        bases = agent.select_batch(this_batch)
        batch_shots = env.sample_classical(this_batch, bases=bases)
        all_shots.extend(batch_shots)
        agent.update(batch_shots)

    # 5. Reconstruct State and Calculate Objective (Fidelity)
    rho_est = proc.reconstruct(all_shots, project=True)
    return fidelity(rho_est, rho_true)

if __name__ == "__main__":
    print(f"Starting Optuna Hyperparameter Search ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize", study_name="ai_agent_tuning")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    print("\n" + "="*50)
    print(f"OPTIMIZATION COMPLETE - Best Fidelity: {study.best_value:.4f}")
    print("="*50)
    print("Best Parameters to use in act2_benchmark.ipynb:")
    for key, value in study.best_params.items():
        print(f"  {key} = {value:.4f}")
    print("="*50)