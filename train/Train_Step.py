# ===================================================================================

# ===================================================================================
 
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Latent Knowledge Distillation
def train_step(ILDC_model, full_batch, past_len, diff_steps=1, config=None):
    """
    Executes the forward passes for Student-Teacher LKD.
    Returns the loss tensor to be backpropagated by the external training loop.
    """
  
    device = full_batch.device
    B, Total_S = full_batch.shape
    future_len = Total_S - past_len -1

    context_tokens = full_batch[:, :past_len]

    # Assuming full_batch contains the full sequence including the target for the last token
    current_tokens = full_batch[:, past_len:-1]
    labels = full_batch[:, past_len+1:]
    
    # Model Inference
    train_output, T_latent_states = ILDC_model(
        input_ids=current_tokens, 
        context_ids=context_tokens,
        active_kv_caches= None,
        active_compressed_kv=None, 
        latent_states= None, # fresh new diffusion
        start_step = 0, 
        diff_steps = diff_steps, # how many diffusion steps to be cycled
        start_pos = 0
        
    )

    # Train Outputs  
    teacher_output = train_output["teacher"]
    # Shapes: [B, S, V] and [B, S, H]
    logits_teacher = teacher_output["logits"] 
    h_teacher = teacher_output["hidden"]

    student_output = train_output["student"]
    # Shapes: [B, T, S, V] and [B, T, S, H]
    student_logits = student_output["T_logits"] 
    student_h = student_output["T_hidden"]

    _, T, S, V = student_logits.shape
    _, _, _, H = student_h.shape



    # Reward Policy creation...     
    # ---------------------------------------------------------
    # 1. BAYESIAN STEP SCHEDULE (0.0 to 1.0)
    # ---------------------------------------------------------
    steps_tensor = torch.arange(T, device=device).float()
    if T > 1:
        progress = (steps_tensor / (T - 1)).view(1, T, 1, 1) # [1, T, 1, 1]
    else:
        # If T=1, default to 1.0 (Hard Ground truth) or 0.0 (Smoothed teacher). 
        # Usually, a 1-step diffusion targets the final hard state.
        progress = torch.ones((1, T, 1, 1), device=device, dtype=torch.float)
    
    # Temperature Annealing Schedule (e.g., 2.0 -> 0.1)
    # 0.1 is mathematically sharp enough to behave exactly like 0.0.
    tau_start, tau_end = 1.0, 0.1
    # Exponential decay for temperature annealing
    tau_t = tau_start * ((tau_end / tau_start) ** progress)

    # ---------------------------------------------------------
    # 2. DYNAMIC MORPHING TARGET DISTRIBUTION
    # ---------------------------------------------------------
    # A. Expand Teacher Logits: [B, T, S, V]
    logits_teacher_exp = logits_teacher.unsqueeze(1).expand(-1, T, -1, -1)
    
    # B. Apply Annealed Temperature to Teacher
    # Early steps: very smooth. Late steps: highly sharp.
    P_teacher = F.softmax(logits_teacher_exp / tau_t, dim=-1)
    
    # C. Get One-Hot Hard Labels
    labels_exp = labels.unsqueeze(1).expand(B, T, S)
    P_labels = F.one_hot(labels_exp, num_classes=V).float()
    
    # D. Morph the Target 
    # At t=0 (progress=0), target is 100% Smoothed Teacher.
    # At t=T (progress=1), target is 100% Hard Ground Truth.
    P_target = (1.0 - progress) * P_teacher + progress * P_labels
    
    # ---------------------------------------------------------
    # 3. UNIFIED LOGIT LOSS
    # ---------------------------------------------------------
    # We DO NOT apply temperature to the student. By keeping student temp=1.0, 
    # it is forced to naturally learn to predict sharp probabilities by the final step.
    student_log_P = F.log_softmax(student_logits, dim=-1)
    
    # Because P_target ends up as a one-hot vector at the final step,
    # this KL Divergence call mathematically calculates Exact Cross-Entropy at step T!
    # unified_logit_loss = F.kl_div(student_log_P, P_target, reduction='none')
    # Instead of: F.kl_div(log_probs, targets)
    # Use:
    eps = 1e-7
    unified_logit_loss = (P_target * (torch.log(P_target + eps) - student_log_P))

    unified_logit_loss = unified_logit_loss.sum(dim=-1).mean()

    # ---------------------------------------------------------
    # 4. LATENT LOSS (Still Required for Diffusion Geometry)
    # ---------------------------------------------------------
    h_teacher_exp = h_teacher.unsqueeze(1).expand(-1, T, -1, -1)
    
    # MSE on hidden states ensures the DM Tower learns to reconstruct the continuous KV cache
    mse_loss = F.mse_loss(student_h, h_teacher_exp, reduction='none').mean(dim=-1)
    
    # # Progressive Refinement: penalize if latent distance INCREASES between steps
    # dist_to_teacher = mse_loss.mean(dim=(0, 2))
    # improvement_loss = F.relu(dist_to_teacher[1:] - dist_to_teacher[:-1]).mean()

    # Final Loss
    total_loss = (
        unified_logit_loss + 
        (config.lambda_latent * mse_loss.mean())) 
        # (config.lambda_refine * improvement_loss)
        # )

    # --- TEMPORARY CALIBRATION PRINT ---
    loss_items = {
        "logit_val": unified_logit_loss.item(),
        "mse_val": mse_loss.mean().item(),
        # "refine_val": improvement_loss.item()
    }
    return total_loss, loss_items




# class TrainingConfig:
#     def __init__(self):
#         self.model_id = "google/gemma-3-1b-it"
#         self.dataset_name = "wikitext"
#         self.dataset_config = "wikitext-2-raw-v1"
#         self.past_len = 2048    
#         self.future_len = 512   
#         self.batch_size = 2     
#         self.gradient_accumulation_steps = 1
#         self.learning_rate = 1e-5
#         self.epochs = 10
#         self.checkpoint_dir = "checkpoints"
#         self.log_every = 1      
#         self.save_every = 100   
#         self.lambda_latent = 0.4
#         self.lambda_refine = 9.0
#         self.diff_steps = 2

# train_config = TrainingConfig()

# from ILDC_Model import ILDC
# import os
# import sys

# device = "cpu"
# ILDC_model = ILDC(device = device)
# ILDC_model.to(torch.bfloat16)
# ILDC_model.to(device)

# with torch.no_grad():
#     full_batch = torch.randint(1, 26400, (3, 2561)).to(device)
#     past_len =  2048
#     train_loss, loss_items = train_step(ILDC_model, full_batch, past_len, 2, train_config)
#     print("Total Loss:", train_loss)
#     print("Loss Items:", loss_items)