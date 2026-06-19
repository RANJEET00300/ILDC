# # ===================================================================================
  
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ============================================================================
# # Latent Knowledge Distillation
# def train_step(
#     ILDC_model, full_batch, past_len, full_active_pass= None,  prev_ground_latent_states= None, 
#     start_step:int=0, diff_steps: int = 2,
#     ):
#     """
#     Executes the forward passes for Student-Teacher LKD.
#     Returns the loss tensor to be backpropagated by the external training loop.
    
#     prev_ground_latent_states: tensor (previous latent output of diffusion step): this is ground-truth after training...
#                                we move one step, diffuse and our current ground-truth latent would build...and 
#                                use it for the current step training

#     start_step: int (The step where the diffusion would be made)
#     diff_steps: int (Count of diffusion steps to proceed)
#     """
  
#     device = full_batch.device
#     B, Total_S = full_batch.shape
#     future_len = Total_S - past_len - 1

#     context_tokens = full_batch[:, :past_len]

#     # Assuming full_batch contains the full sequence including the target for the last token
#     current_tokens = full_batch[:, past_len:-1]
#     labels = full_batch[:, past_len+1:]
    
#     # Model Inference
#     train_output, full_active_pass, T_latent_states, prev_ground_latent_states = ILDC_model(
#         input_ids=current_tokens, 
#         context_ids=context_tokens,
#         active_kv_caches= None,
#         active_compressed_kv=None, 
#         start_pos = 0,  # compressed_kv positions....

#         start_step = start_step,   # this will be the starting position of diffusion-step
#         diff_steps = diff_steps, 

#         full_active_pass = full_active_pass,   # if full kv cache exists, do not call teacher,...just compress and pass to student
#         prev_ground_latent_states= prev_ground_latent_states 
        
#     )

#     # Add the labels to pass to training policy:
#     train_output["labels"] = labels

#     return train_output, full_active_pass, T_latent_states, prev_ground_latent_states




# def train_policy(train_output, training_step, total_diff_steps, config):

#     T = total_diff_steps # total_block_diffusions
#     turn = training_step  # current state of diffusion step to be trained

#     # Train Outputs  
#     teacher_output = train_output["teacher"]
#     student_output = train_output["student"]
#     labels = train_output["labels"]

#     # Shapes: [B, S, V] and [B, S, H]
#     logits_teacher = teacher_output["logits"] 
#     h_teacher = teacher_output["hidden"]

#     logits_student = student_output["logits"] 
#     h_student = student_output["hidden"]

#     _, _, V = logits_student.shape
   

#     # print("\n=== LOGIT SCALE DIAGNOSTIC ===")
#     # print(f"Teacher Logits | Max: {logits_teacher.max().item():.4f} | Min: {logits_teacher.min().item():.4f} | Std: {logits_teacher.std().item():.4f}")
#     # print(f"Student Logits | Max: {logits_student.max().item():.4f} | Min: {logits_student.min().item():.4f} | Std: {logits_student.std().item():.4f}")
#     # print("==============================\n")

#     # Train Policy ...     
#     # ---------------------------------------------------------
#     # 1. BAYESIAN STEP SCHEDULE (0.0 to 1.0)
#     # ---------------------------------------------------------
#     # If T=1, default to 1.0 (Hard Ground truth) or 0.0 (Smoothed teacher).

#     if T > 1:
#         progress = (turn / (T - 1))
#     else: 
#         # 1-step diffusion targets the final hard state.
#         progress = 1
    
#     # Temperature Annealing Schedule (e.g., 2.0 -> 0.1)
#     # 0.1 is mathematically sharp enough to behave exactly like 0.0.
#     tau_start, tau_end = 1.0, 0.1
#     # Exponential decay for temperature annealing
#     tau_t = tau_start * ((tau_end / tau_start) ** progress)
    
#     # ---------------------------------------------------------
#     # 2. DYNAMIC MORPHING TARGET DISTRIBUTION
#     # ---------------------------------------------------------
    
#     # B. Apply Annealed Temperature to Teacher
#     # Early steps: very smooth. Late steps: highly sharp.
#     P_teacher = F.softmax(logits_teacher / tau_t, dim=-1)
    
#     # C. Get One-Hot Hard Labels
#     P_labels = F.one_hot(labels, num_classes=V).float()
    
#     # D. Morph the Target 
#     # At t=0 (progress=0), target is 100% Smoothed Teacher.
#     # At t=T (progress=1), target is 100% Hard Ground Truth.
#     P_target = (1.0 - progress) * P_teacher + progress * P_labels
#     P_target.clamp(min=0.0, max=1.0)
   
#     # ---------------------------------------------------------
#     # 3. UNIFIED LOGIT LOSS
#     # ---------------------------------------------------------
#     # We DO NOT apply temperature to the student. By keeping student temp=1.0, 
#     # it is forced to naturally learn to predict sharp probabilities by the final step.
#     student_log_P = F.log_softmax(logits_student, dim=-1)
    
#     # Because P_target ends up as a one-hot vector at the final step,
#     # so KL Divergence call mathematically calculates Exact Cross-Entropy at step T!
    
#     eps = 1e-5
#     log_P_target = torch.log(P_target + eps)
    
    
#     loss_elements = (P_target * (log_P_target - student_log_P))
#     unified_logit_loss = loss_elements.sum(dim=-1).mean()

#     # print("loss_elements Contains NaN:", torch.isnan(loss_elements).any().item())

#     # ---------------------------------------------------------
#     # 4. LATENT LOSS (for Diffusion Geometry)
#     # ---------------------------------------------------------
    
#     # MSE on hidden states ensures the DM Tower learns to reconstruct the continuous KV cache
#     mse_loss = F.mse_loss(h_student, h_teacher, reduction='none')
    
    
#     # Final Loss
#     total_loss = (
#         unified_logit_loss + 
#         (config.lambda_latent * mse_loss.mean())) 
       

#     # --- TEMPORARY CALIBRATION PRINT ---
#     loss_items = {
#         "logit_val": unified_logit_loss.item(),
#         "mse_val": mse_loss.mean().item()
#     }
#     return total_loss, loss_items




# class TrainingConfig:
#     def __init__(self):
#         self.model_id = "google/gemma-3-1b-it"
#         self.dataset_name = "wikitext"
#         self.dataset_config = "wikitext-2-raw-v1"
#         self.past_len = 2048    
#         self.future_len = 512   
#         self.batch_size = 2     
#         self.gradient_accumulation_steps = 32
#         self.learning_rate = 1e-4
#         self.epochs = 10
#         self.checkpoint_dir = "checkpoints"
#         self.log_every = 1      
#         self.save_every = 100   
#         self.lambda_latent = 100.0
#         self.diff_steps = 2
#         self.total_diffusion_steps = 8

# train_config = TrainingConfig()

# from ILDC_Model import ILDC
# import os
# import sys

# device = "cuda"
# ILDC_model = ILDC(device = device)
# ILDC_model.to(torch.bfloat16)
# ILDC_model.to(device)



# with torch.no_grad():
#     full_batch = torch.randint(1, 26400, (1, 2561)).to(device)
#     past_len =  2048
#     full_active_pass = None
#     prev_ground_latent_states = None 
#     Tdiff_steps = train_config.total_diff_steps
#     diff_steps = train_config.diff_steps
#     start_step = 0
#     training_step = 0

#     for _ in range((Tdiff_steps//diff_steps)):

#         print("Passing Diffusion Step:", start_step)
#         train_output, full_active_pass, T_latent_states, prev_ground_latent_states = train_step(
#             ILDC_model= ILDC_model,
#             full_batch= full_batch, 
#             past_len=past_len, 
#             full_active_pass= full_active_pass,  
#             prev_ground_latent_states= prev_ground_latent_states, 
#             start_step= start_step, 
#             diff_steps= diff_steps
#         )

#         start_step += diff_steps
#         training_step = start_step -1

        
#         # print(train_output)
#         train_loss, loss_items = train_policy(train_output, training_step, Tdiff_steps, train_config)
        

#         print("Total Loss:", train_loss)
#         print("Loss Items:", loss_items)



# ===================================================================================
  
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Latent Knowledge Distillation
def train_step(
    ILDC_model, full_batch, 
    start_step:int=0,
    training_step: int=0, config = None
    ):
    """
    Executes the forward passes for Student-Teacher LKD.
    Returns the loss tensor to be backpropagated by the external training loop.

    start_step: int (The step where the diffusion would be made)
    diff_steps: int (Count of diffusion steps to proceed)
    """
  
    device = full_batch.device
    B, Total_S = full_batch.shape
    past_len = config.past_len
    T = config.total_diffusion_steps # total_block_diffusions
    diff_steps = config.diff_steps
    turn = training_step  # current state of diffusion step to be trained

    future_len = Total_S - past_len - 1

    context_tokens = full_batch[:, :past_len]

    # Assuming full_batch contains the full sequence including the target for the last token
    current_tokens = full_batch[:, past_len:-1]
    labels = full_batch[:, past_len+1:]
    
    # Model Inference
    train_output, _ = ILDC_model(
        input_ids=current_tokens, 
        context_ids=context_tokens,
        active_kv_caches= None,
        active_compressed_kv=None, 
        start_pos = 0,  # compressed_kv positions....

        start_step = start_step,   # this will be the starting position of diffusion-step
        diff_steps = diff_steps
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
        progress = (turn / (T - 1))
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
    P_target.clamp(min=0.0, max=1.0)
   
    # ---------------------------------------------------------
    # 3. UNIFIED LOGIT LOSS
    # ---------------------------------------------------------
    # We DO NOT apply temperature to the student. By keeping student temp=1.0, 
    # it is forced to naturally learn to predict sharp probabilities by the final step.
    student_log_P = F.log_softmax(logits_student, dim=-1)
    
    # Because P_target ends up as a one-hot vector at the final step,
    # so KL Divergence call mathematically calculates Exact Cross-Entropy at step T!
    
    eps = 1e-5
    log_P_target = torch.log(P_target + eps)
    
    
    loss_elements = (P_target * (log_P_target - student_log_P))
    unified_logit_loss = loss_elements.sum(dim=-1).mean()

    # print("loss_elements Contains NaN:", torch.isnan(loss_elements).any().item())

    # ---------------------------------------------------------
    # 4. LATENT LOSS (for Diffusion Geometry)
    # ---------------------------------------------------------
    
    # MSE on hidden states ensures the DM Tower learns to reconstruct the continuous KV cache
    mse_loss = F.mse_loss(h_student, h_teacher, reduction='none')
    
    
    # Final Loss
    total_loss = (
        unified_logit_loss + 
        (config.lambda_latent * mse_loss.mean())) 
       

    # --- TEMPORARY CALIBRATION PRINT ---
    loss_items = {
        "logit_val": unified_logit_loss.item(),
        "mse_val": mse_loss.mean().item()
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
#         self.gradient_accumulation_steps = 32
#         self.learning_rate = 1e-4
#         self.epochs = 10
#         self.checkpoint_dir = "checkpoints"
#         self.log_every = 1      
#         self.save_every = 100   
#         self.lambda_latent = 0.1
#         self.diff_steps = 2
#         self.total_diffusion_steps = 8

# train_config = TrainingConfig()

# from ILDC_Model import ILDC
# import os
# import sys

# device = "cuda"
# ILDC_model = ILDC(device = device)
# ILDC_model.to(torch.bfloat16)
# ILDC_model.to(device)



# with torch.no_grad():
#     full_batch = torch.randint(1, 26400, (1, 2561)).to(device)
#     past_len =  2048
#     Tdiff_steps = train_config.total_diff_steps
#     diff_steps = train_config.diff_steps
#     start_step = 0
#     training_step = 0

#     for _ in range((Tdiff_steps//diff_steps)):
#         training_step = start_step + 1
#         print("Passing Diffusion Step:", start_step)
#         train_loss, loss_items  = train_step(
#             ILDC_model= ILDC_model,
#             full_batch= full_batch, 
#             start_step= start_step, 
#             training_step = training_step,
#             config = train_config
#         )

#         start_step += diff_steps
        
        
#         print("Total Loss:", train_loss)
#         print("Loss Items:", loss_items)
