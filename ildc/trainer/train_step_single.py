import torch
import torch.nn.functional as F


# ============================================================================
# Latent Knowledge Distillation
def train_step(
    ILDC_model: torch.nn.Module,
    full_batch: torch.Tensor,
    start_step: int = 0,
    training_step: int = 0,
    config=None,
) -> tuple[torch.Tensor, dict]:
    """
    Executes the Latent Knowledge Distillation (LKD) single-step block forward pass.

    Extracts the uncompressed Target Knowledge from the Teacher and calculates the
    Unified Logit Loss against the Student conditioned on the generated compressed KV cache.

    Args:
        ILDC_model: The wrapper model containing AR and DC architectures.
        full_batch (torch.Tensor): Shape (B, Total_S). The full sequence of tokens.
        start_step (int): The current diffusion block starting step.
        training_step (int): The absolute timestep index for Bayesian scheduling.
        config: TrainingConfig object.

    Returns:
        Tuple of (total_loss_tensor, loss_items_dictionary).
    """

    B, Total_S = full_batch.shape
    past_len = config.past_len

    T = config.total_diffusion_steps  # total_block_diffusions
    diff_steps = config.diff_steps

    turn = training_step  # current state of diffusion step to be trained

    context_tokens = full_batch[:, :past_len]

    # Assuming full_batch contains the full sequence including the target for the last token
    current_tokens = full_batch[:, past_len:-1]
    labels = full_batch[:, past_len + 1 :]

    # Model Inference
    train_output, _ = ILDC_model(
        input_ids=current_tokens,
        context_ids=context_tokens,
        active_kv_caches=None,
        active_compressed_kv=None,
        start_pos=0,  # compressed_kv positions....
        start_step=start_step,  # this will be the starting position of diffusion-step
        diff_steps=diff_steps,
    )

    # ================================================================

    # Train Outputs
    teacher_output = train_output["teacher"]
    student_output = train_output["student"]

    # Shapes: [B, S, V] and [B, S, H]
    logits_teacher = teacher_output["logits"]
    h_teacher = teacher_output["hidden"]

    logits_student = student_output["logits"]
    h_student = student_output["hidden"]

    _, _, V = logits_student.shape

    # Train Policy ...
    # ---------------------------------------------------------
    # 1. BAYESIAN STEP SCHEDULE (0.0 to 1.0)
    # ---------------------------------------------------------
    # If T=1, default to 1.0 (Hard Ground truth) or 0.0 (Smoothed teacher).

    if T > 1:
        progress = turn / (T - 1)
    else:
        # 1-step diffusion targets the final hard state.
        progress = 1

    # Temperature Annealing Schedule (e.g., 2.0 -> 0.1)
    # 0.1 is mathematically sharp enough to behave exactly like 0.0.
    tau_start, tau_end = 1.0, 0.1
    # Exponential decay for temperature annealing
    tau_t = tau_start * ((tau_end / tau_start) ** progress)

    # ---------------------------------------------------------
    # 2. DYNAMIC MORPHING TARGET DISTRIBUTION
    # ---------------------------------------------------------

    # B. Apply Annealed Temperature to Teacher
    # Early steps: very smooth. Late steps: highly sharp.
    P_teacher = F.softmax(logits_teacher / tau_t, dim=-1)

    # C. Get One-Hot Hard Labels
    P_labels = F.one_hot(labels, num_classes=V).float()

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
    student_log_P = F.log_softmax(logits_student, dim=-1)

    # Because P_target ends up as a one-hot vector at the final step,
    # so KL Divergence call mathematically calculates Exact Cross-Entropy at step T!

    eps = 1e-7
    log_P_target = torch.log(P_target + eps)

    loss_elements = P_target * (log_P_target - student_log_P)
    unified_logit_loss = loss_elements.sum(dim=-1).mean()

    # print("loss_elements Contains NaN:", torch.isnan(loss_elements).any().item())

    # ---------------------------------------------------------
    # 4. LATENT LOSS (for Diffusion Geometry)
    # ---------------------------------------------------------

    # MSE on hidden states ensures the DM Tower learns to reconstruct the continuous KV cache
    mse_loss = F.mse_loss(h_student, h_teacher, reduction="none")

    # Final Loss
    total_loss = unified_logit_loss + (config.lambda_latent * mse_loss.mean())

    # --- TEMPORARY CALIBRATION PRINT ---
    loss_items = {"logit_val": unified_logit_loss.item(), "mse_val": mse_loss.mean().item()}
    return total_loss, loss_items
