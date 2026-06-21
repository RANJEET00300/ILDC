import torch
import torch.nn.functional as F


# Latent Knowledge Distillation
# ===================================================================================


def train_step(
    ILDC_model: torch.nn.Module,
    full_batch: torch.Tensor,
    config=None,
) -> tuple[torch.Tensor, dict]:
    """
    Executes the Full Back-Propagation Through Time (BPTT) LKD forward pass.

    Runs the Diffusion Compressor for multiple continuous steps within the graph.
    Applies the Bayesian Step Schedule and Dynamic Morphing Target Distribution
    across the entire temporal dimension simultaneously.

    Args:
        ILDC_model: The wrapper model containing AR and DC architectures.
        full_batch (torch.Tensor): Shape (B, Total_S). The full sequence of tokens.
        config: TrainingConfig object.

    Returns:
        Tuple of (total_loss_tensor, loss_items_dictionary).
    """

    device = full_batch.device
    B, Total_S = full_batch.shape

    Tdiff_steps = config.total_diffusion_steps
    past_len = config.past_len

    context_tokens = full_batch[:, :past_len]

    # Assuming full_batch contains the full sequence including the target for the last token
    current_tokens = full_batch[:, past_len:-1]
    labels = full_batch[:, past_len + 1 :]

    # Model Inference
    train_output, T_latent_states = ILDC_model.bptt_forward(
        input_ids=current_tokens,
        context_ids=context_tokens,
        active_kv_caches=None,
        active_compressed_kv=None,
        start_pos=0,
        Tdiff_steps=Tdiff_steps - 1,  # how many diffusion steps to be cycled
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
        progress = (steps_tensor / (T - 1)).view(1, T, 1, 1)  # [1, T, 1, 1]
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
    P_target = P_target.clamp(min=0.0, max=1.0)

    # ---------------------------------------------------------
    # 3. UNIFIED LOGIT LOSS
    # ---------------------------------------------------------
    # We DO NOT apply temperature to the student. By keeping student temp=1.0,
    # it is forced to naturally learn to predict sharp probabilities by the final step.
    student_log_P = F.log_softmax(student_logits, dim=-1)

    # Because P_target ends up as a one-hot vector at the final step,
    # this KL Divergence call mathematically calculates Exact Cross-Entropy at step T!
    eps = 1e-7
    unified_logit_loss = P_target * (torch.log(P_target + eps) - student_log_P)

    unified_logit_loss = unified_logit_loss.sum(dim=-1).mean()

    # ---------------------------------------------------------
    # 4. LATENT LOSS (Still Required for Diffusion Geometry)
    # ---------------------------------------------------------
    h_teacher_exp = h_teacher.unsqueeze(1).expand(-1, T, -1, -1)

    # MSE on hidden states ensures the DM Tower learns to reconstruct the continuous KV cache
    mse_loss = F.mse_loss(student_h, h_teacher_exp, reduction="none").mean(dim=-1)

    # Final Loss
    total_loss = unified_logit_loss + (config.lambda_latent * mse_loss.mean())

    # --- TEMPORARY CALIBRATION PRINT ---
    loss_items = {"logit_val": unified_logit_loss.item(), "mse_val": mse_loss.mean().item()}
    return total_loss, loss_items
