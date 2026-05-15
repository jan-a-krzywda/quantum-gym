import torch
import torch.nn.functional as F
import numpy as np

def active_inference_search(
    world_model, 
    vae_decoder, 
    mu_start, 
    ghz_shadow, 
    action_space,
    depth=5, 
    beam_width=10, 
    uncertainty_penalty=0.1,
    uncertainty_threshold=0.5
):
    """
    Active Inference planner balancing:
    1. Pragmatic Value: Exploiting the model to reach the GHZ state.
    2. Epistemic Value: Hypothesis testing highly uncertain regions on real hardware.
    """
    # Beam format: (score, mu_state, trajectory, uncertainty_accumulated)
    beam = [(0.0, mu_start, [], 0.0)]
    
    best_hypothesis = None  # (info_gain, trajectory, mu_predicted)
    
    for step in range(depth):
        next_beam = []
        
        for score, mu, traj, uncert in beam:
            # Evaluate all possible discrete actions from this state
            for action in action_space:
                # 1. Predict next state and epistemic uncertainty
                # (Assuming world model outputs mean and logvar/variance)
                mu_next, logvar_next = world_model.predict_with_uncertainty(mu, action)
                step_uncertainty = torch.exp(logvar_next).mean().item()
                
                # 2. Decode to shadow and calculate fidelity (Pragmatic Reward)
                shadow_pred = vae_decoder(mu_next)
                fidelity = F.cosine_similarity(shadow_pred, ghz_shadow).item()
                
                new_uncert = uncert + step_uncertainty
                
                # 3. Hypothesis Testing (Epistemic Value)
                # If the uncertainty crosses our safety threshold, this region is unknown.
                # We propose this partial path as a hardware verification run.
                if step_uncertainty > uncertainty_threshold:
                    info_gain = step_uncertainty
                    if best_hypothesis is None or info_gain > best_hypothesis[0]:
                        best_hypothesis = (info_gain, traj + [action], mu_next)
                    # Do not continue planning deeper into the unknown fog.
                    continue
                
                # 4. Active Inference Score (Exploitation)
                active_inference_score = fidelity - (uncertainty_penalty * new_uncert)
                
                next_beam.append((
                    active_inference_score, 
                    mu_next, 
                    traj + [action], 
                    new_uncert
                ))
                
        # Sort by best pragmatic score and prune
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]
        
        # Early stopping if we hit the target with high confidence
        if len(beam) > 0 and beam[0][0] > 0.95: 
            break
            
    # --- Decision Making ---
    best_pragmatic = beam[0] if len(beam) > 0 else None
    
    # If our pragmatic path isn't confident enough, but we found a highly uncertain boundary,
    # we execute a "Hypothesis Test" to collapse the uncertainty and improve our model.
    if best_hypothesis is not None:
        if best_pragmatic is None or best_pragmatic[0] < 0.85:
            expected_gain, hyp_traj, hyp_mu = best_hypothesis
            print(f"[Epistemic] High uncertainty detected (Info Gain: {expected_gain:.3f}).")
            print(f"            Verifying hypothesis on hardware: {hyp_traj}")
            return hyp_traj, "verify"
            
    if best_pragmatic is not None:
        best_score, best_mu, best_traj, best_uncert = best_pragmatic
        print(f"[Pragmatic] Plan found! Expected Score: {best_score:.3f}, Uncertainty: {best_uncert:.3f}")
        return best_traj, "execute"
        
    print("[Failure] No valid plan or hypothesis found.")
    return [], "fail"

# --- Usage Example ---
# 1. Load VAE and World Model
# 2. Measure |000> on hardware, encode to mu_start
# 3. Plan the best sequence of actions
# best_actions, action_type = active_inference_search(world_model, vae.decoder, mu_start, ghz_target, action_space)
# 4. Execute best_actions on the real device.
# 5. Observe the new real mu.
# 6. If action_type == "verify", use the real outcome to update/retrain the world model!
