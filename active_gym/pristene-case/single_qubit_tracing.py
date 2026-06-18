import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ── 1. The Environment (OU Process) ──────────────────────────────────────

class SingleQubitOUEnvironment:
    """
    Simulates a single qubit undergoing a Ramsey sequence where the 
    detuning phase (dphi) drifts according to an Ornstein-Uhlenbeck process.
    """
    def __init__(self, phi0=0.0, tc=100.0, sigma=0.02, dt=1.0):
        self.phi0 = phi0
        self.tc = tc
        self.sigma = sigma
        self.dt = dt
        self.current_dphi = phi0

    def step(self, action_theta: float) -> tuple[int, float]:
        """
        Applies the action, calculates measurement probability, and drifts the noise.
        P(1) = sin^2((action + dphi) / 2)
        """
        # 1. Calculate measurement probability based on current dphi
        total_phase = action_theta + self.current_dphi
        p_1 = np.sin(total_phase / 2.0) ** 2
        
        # 2. Collapse the state (Single Shot)
        measurement = 1 if np.random.rand() < p_1 else 0
        
        # 3. OU Process Drift for the *next* step
        drift = (self.phi0 - self.current_dphi) * (self.dt / self.tc)
        diffusion = self.sigma * np.sqrt(self.dt) * np.random.randn()
        
        true_dphi_used = self.current_dphi
        self.current_dphi += drift + diffusion
        
        return measurement, true_dphi_used

# ── 2. Optimized Mamba/Recurrent Tracker ─────────────────────────────────

class EfficientMockMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x, h_gru=None):
        # Natively carries hidden state context across iterations to prevent launch lag
        out, h_gru = self.gru(x, h_gru)
        return out, h_gru

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar.clamp(-10, 10))
    return mu + torch.randn_like(std) * std

def kl_normal(mu1, logvar1, mu2, logvar2):
    var1, var2 = logvar1.exp(), logvar2.exp()
    return 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / var2 - 1).sum(dim=-1)

class MambaPhaseTracker(nn.Module):
    """
    RSSM tracking the hidden dphi parameter using single-shot binary data.
    """
    def __init__(self, mamba_dim=64, latent_dim=16):
        super().__init__()
        
        # Encoders
        self.obs_encoder = nn.Linear(1, mamba_dim)
        self.action_encoder = nn.Linear(1, mamba_dim)
        
        # Optimized Temporal Dynamics
        self.mamba = EfficientMockMambaBlock(mamba_dim)
        
        # Bayesian Distributions
        self.prior = nn.Sequential(nn.Linear(mamba_dim + mamba_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim * 2))
        self.posterior = nn.Sequential(nn.Linear(mamba_dim + mamba_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim * 2))
        
        # Generative Bottleneck Projection
        self.s_proj = nn.Linear(latent_dim, mamba_dim)
        self.dphi_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.h_init = nn.Parameter(torch.randn(1, 1, mamba_dim) * 0.05)

    def forward(self, obs_stream: torch.Tensor, actions: torch.Tensor):
        B, T, _ = obs_stream.shape
        obs_emb = self.obs_encoder(obs_stream.float())
        act_emb = self.action_encoder(actions.float())
        
        h = self.h_init.expand(B, -1, -1)
        h_gru = None  # Persistent context container across the unrolled loops
        
        dphi_preds = []
        kl_loss = torch.zeros(B, device=obs_stream.device)
        
        for t in range(T):
            o_t = obs_emb[:, t]
            a_t = act_emb[:, t]
            
            # Prior: p(s_t | h_{t-1}, a_t)
            p_params = self.prior(torch.cat([h.squeeze(1), a_t], dim=-1))
            p_mu, p_lv = p_params.chunk(2, dim=-1)
            
            # Posterior: q(s_t | h_{t-1}, o_t)
            q_params = self.posterior(torch.cat([h.squeeze(1), o_t], dim=-1))
            q_mu, q_lv = q_params.chunk(2, dim=-1)
            
            # Latent parameter sampling
            s_t = reparameterize(q_mu, q_lv)
            dphi_pred = self.dphi_head(s_t)
            dphi_preds.append(dphi_pred)
            
            # Loosened KL floor to unlock early parameter exploration
            kl_loss += torch.clamp(kl_normal(q_mu, q_lv, p_mu, p_lv), min=0.1)
            
            # State Update via context-carrying Mamba/GRU block
            h_input = h + self.s_proj(s_t).unsqueeze(1) + a_t.unsqueeze(1)
            h, h_gru = self.mamba(h_input, h_gru)
            
        return torch.stack(dphi_preds, dim=1), (kl_loss / T)

    def compute_loss(self, obs_stream: torch.Tensor, actions: torch.Tensor, beta=0.01):
        # 1. Forward pass to extract sequence history
        dphi_preds, kl_loss = self.forward(obs_stream, actions)
        
        # 2. FIX: Causal Temporal Target Alignment
        # Predicted dphi at time t combined with upcoming action at time t+1 
        # reconstructs the outcome probability of observation at time t+1.
        pred_dphi_slice = dphi_preds[:, :-1, :]
        next_action_slice = actions[:, 1:, :]
        target_obs_slice = obs_stream[:, 1:, :]
        
        # Physics Transition Equation
        total_phase = next_action_slice + pred_dphi_slice
        hat_prob_1 = torch.sin(total_phase / 2.0) ** 2
        
        # 3. Vectorized Binary Cross Entropy Execution
        # Added eps clamping to prevent log(0) NaN anomalies
        hat_prob_1 = torch.clamp(hat_prob_1, min=1e-6, max=1.0 - 1e-6)
        bce_loss = F.binary_cross_entropy(hat_prob_1, target_obs_slice.float(), reduction="none")
        recon_loss = bce_loss.mean()
        
        # Balanced scaling between Reconstruction surprise and Complexity costs
        total_vfe = recon_loss + beta * kl_loss.mean()
        
        return total_vfe, dphi_preds

# ── 3. Diagnostic Plotting ───────────────────────────────────────────────

def plot_tracking_quality_with_losses(true_dphis, dphi_preds_np, actions, observations, model):
    """
    Generates a 4-panel diagnostic dashboard tracking the true phase, residuals,
    applied actions, and the explicit structural breakdown of VFE (BCE vs. KL).
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Align causal timelines
    aligned_trues = np.array(true_dphis[1:])
    aligned_preds = dphi_preds_np[:-1]
    steps = np.arange(len(aligned_trues))
    
    # 2. Extract step-by-step KL and BCE via an evaluation pass
    obs_t = torch.tensor(observations, dtype=torch.float32).unsqueeze(0)
    act_t = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        # Evaluate out parameters directly
        out = model.forward(obs_t, act_t)
        dphi_preds_tensor, _ = out
        
        # Physics transition reconstruction
        pred_dphi_slice = dphi_preds_tensor[:, :-1, :]
        next_action_slice = act_t[:, 1:, :]
        target_obs_slice = obs_t[:, 1:, :]
        
        total_phase = next_action_slice + pred_dphi_slice
        hat_prob_1 = torch.clamp(torch.sin(total_phase / 2.0) ** 2, min=1e-6, max=1.0-1e-6)
        
        # Component Extraction
        bce_per_step = F.binary_cross_entropy(hat_prob_1, target_obs_slice.float(), reduction="none").squeeze().numpy()
        
        # Re-compute posteriors and priors to extract un-clamped KL values
        obs_emb = model.obs_encoder(obs_t.float())
        act_emb = model.action_encoder(act_t.float())
        h = model.h_init.expand(1, -1, -1)
        h_gru = None
        kl_per_step = []
        
        for t in range(len(aligned_trues)):
            o_t = obs_emb[:, t]
            a_t = act_emb[:, t]
            p_params = model.prior(torch.cat([h.squeeze(1), a_t], dim=-1))
            q_params = model.posterior(torch.cat([h.squeeze(1), o_t], dim=-1))
            p_mu, p_lv = p_params.chunk(2, dim=-1)
            q_mu, q_lv = q_params.chunk(2, dim=-1)
            
            kl_val = kl_normal(q_mu, q_lv, p_mu, p_lv).item()
            kl_per_step.append(kl_val)
            
            s_t = reparameterize(q_mu, q_lv)
            h, h_gru = model.mamba(h + model.s_proj(s_t).unsqueeze(1) + a_t.unsqueeze(1), h_gru)

    # 3. Canvas Composition
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Panel 0: Phase Trajectory Tracking
    axs[0].plot(steps, aligned_trues, label="True $d\phi$ (OU Drift)", color="black", alpha=0.8)
    axs[0].plot(steps, aligned_preds, label="Mamba-RSSM Prediction", color="crimson", linestyle="--", alpha=0.9)
    axs[0].set_ylabel("Phase (rad)")
    axs[0].set_title("Stochastic Phase Tracking Performance")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    # Panel 1: Tracking Residual Errors
    residuals = aligned_preds - aligned_trues
    axs[1].plot(steps, residuals, color="purple", alpha=0.7)
    axs[1].axhline(0, color="black", linestyle=":", alpha=0.5)
    axs[1].set_ylabel("Error (rad)")
    axs[1].set_title(f"Residual Errors (MAE: {np.mean(np.abs(residuals)):.4f} rad)")
    axs[1].grid(True, alpha=0.3)

    # Panel 2: The Variational Free Energy Breakdown (BCE vs KL)
    axs[2].plot(steps, bce_per_step, color="steelblue", alpha=0.5, label="BCE (Reconstruction Surprise)")
    axs[2].plot(steps, kl_per_step, color="green", alpha=0.8, label="KL Divergence (Complexity Penalty)")
    axs[2].set_ylabel("Loss Magnitude")
    axs[2].set_title("Variational Component Losses Across Trajectory Timeline")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)
    
    # Panel 3: Input Probes and Quantum Measurements
    actions_np = np.array(actions[1:]).flatten()
    obs_np = np.array(observations[1:]).flatten()
    axs[3].plot(steps, actions_np, color="gray", alpha=0.3, label="Applied Action $\\theta$")
    axs[3].scatter(steps[obs_np == 1], actions_np[obs_np == 1], color="orange", s=8, label="Shot: 1", alpha=0.6)
    axs[3].scatter(steps[obs_np == 0], actions_np[obs_np == 0], color="royalblue", s=8, label="Shot: 0", alpha=0.6)
    axs[3].set_ylabel("Probe Angle (rad)")
    axs[3].set_xlabel("Shot Sequence Index (Time)")
    axs[3].legend(loc="upper right")
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mamba_kl_collapse_diagnostics.png", dpi=300)
    plt.show()

# ── 4. Optimized Validation Execution ────────────────────────────────────

def main():
    device = torch.device("cpu")
    seq_len = 2000
    
    # Environment Setup
    env = SingleQubitOUEnvironment(phi0=np.pi/8, tc=400.0, sigma=0.05)
    model = MambaPhaseTracker().to(device)
    
    # Speed Optimization: Pre-compiles the dependency graph and fusses computational loops
    model = torch.compile(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    print("Generating OU Process Trajectory...")
    true_dphis, observations, actions = [], [], []
    action_sequence = [np.pi/2]*seq_len  # Static action probe (theta=0) to isolate the tracking of the OU process
    
    for a in action_sequence:
        bit, true_dphi = env.step(a)
        observations.append([bit])
        true_dphis.append(true_dphi)
        actions.append([a])
        
    obs_t = torch.tensor(observations, dtype=torch.float32).unsqueeze(0).to(device) 
    act_t = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(device)      
    
    print("Training Mamba-RSSM Tracker...")
    epochs = 10
    target_beta = 0.01  # Low KL weight to prioritize reconstruction and tracking performance
    for epoch in range(epochs):
        current_beta = target_beta * min(1.0, epoch / 50.0)
        optimizer.zero_grad()
        loss, dphi_preds = model.compute_loss(obs_t, act_t, beta=current_beta)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | VFE Loss: {loss.item():.4f}")

    dphi_preds_np = dphi_preds.squeeze().detach().cpu().numpy()
    mae = np.mean(np.abs(dphi_preds_np[:-1] - np.array(true_dphis[1:])))
    print(f"\nFinal Causal Tracking MAE: {mae:.4f} radians")
    
    # Generate diagnostic plot
    plot_tracking_quality_with_losses(true_dphis, dphi_preds_np, actions, observations, model)


if __name__ == "__main__":
    main()
