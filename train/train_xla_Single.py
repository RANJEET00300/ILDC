import os
os.environ["XLA_USE_SPMD"] = "1" # Required for SPMD

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModelForCausalLM

from model.ILDC_Model import ILDC
from train.Train_Step_Single import train_step, train_policy
from dataset.data_loader import get_wikipedia_batches

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr 
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np

"""
ILDC Stage 1 Warm-up Training — PyTorch XLA (TPU v5e-8)
=======================================================
Policy Gradient (RL) Version
"""

class TrainingConfig:
    def __init__(self):
        self.model_id = "google/gemma-3-1b-it"
        self.dataset_name = "wikitext"
        self.dataset_config = "wikitext-2-raw-v1"
        self.checkpoint_dir = "/kaggle/working/checkpoints/dc_model_final.pt"

        self.past_len = 2048    
        self.future_len = 512   
        self.full_len = 2561
        
        self.learning_rate = 1e-4
        self.lambda_latent = 0.1

        self.epochs = 10
        self.log_every = 1      
        self.save_every = 100 
    
        self.batch_size = 1     
        self.gradient_accumulation_steps = 16
        self.total_diffusion_steps = 4 
        self.diff_steps = 2 # hard training with Block, 2 steps per/

        self.N_batches = 16384


def detach_state(state):
    """Recursively detaches tensors to sever the computation graph for TBPTT."""
    if state is None:
        return None
    elif isinstance(state, torch.Tensor):
        return state.detach()
    elif isinstance(state, list):
        return [detach_state(x) for x in state]
    elif isinstance(state, tuple):
        return tuple(detach_state(x) for x in state)
    elif isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    return state


        
def train_fn(index):
    config = TrainingConfig()
    chunk_size = config.past_len + config.future_len + 1
    
    device = torch_xla.device()     
    world_size = xr.world_size()    
    rank = xr.global_ordinal()      
    
    xm.master_print("=" * 60)
    xm.master_print("  ILDC Stage Training (PyTorch XLA / TPU)")
    xm.master_print(f"  World size: {world_size} TPU cores")
    xm.master_print("=" * 60)
    

    batch_size = config.batch_size     
    grad_accum_steps = config.gradient_accumulation_steps
    Tdiff_steps = config.total_diffusion_steps 
    diff_steps = config.diff_steps  # hard training with Block, 2 steps per/
    full_len = config.full_len
    past_len = config.past_len

    # effective batch-size, break them into rolls and blocks...for training
    eff_batch_size = batch_size * grad_accum_steps

    dataloader = get_wikipedia_batches(
        tokenizer_path=config.model_id, 
        batch_size=eff_batch_size, 
        required_length=config.full_len, 
        max_batches= config.N_batches  
    )
    mp_dataloader = pl.MpDeviceLoader(dataloader, device)
    
    xm.master_print("Loading ILDC Model...")
    ILDC_model = ILDC(config.checkpoint_dir, device)
    ILDC_model.to(torch.bfloat16)
    ILDC_model.to(device)


    ILDC_model.dc_model.train()  
    optimizer = torch.optim.AdamW([{'params': ILDC_model.dc_model.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}])
    
    if rank == 0: 
        parent_dir = os.path.dirname(config.checkpoint_dir)
        os.makedirs(parent_dir, exist_ok=True)


    # ==============================================================================
    # 2. SCHEDULER SETUP (COSINE + WARMUP)
    # ==============================================================================
    
    total_optim_steps = (config.N_batches // grad_accum_steps) * config.epochs
    warmup_steps = int(total_optim_steps * 0.02) # 2% of training is warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps
    )

    # ==============================================================================
    # 3. OBSERVABILITY TRACKERS
    # ==============================================================================
    global_step = 0
    running_loss = 0.0
    running_logit = 0.0
    running_mse = 0.0

    # ==============================================================================
    # 4. THE XLA TRAINING LOOP
    # ==============================================================================
    xm.master_print(f"Starting Training! Total Opt Steps: {total_optim_steps} | Warmup: {warmup_steps}")

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        
        for batch_idx, full_batch in enumerate(mp_dataloader):
            
            full_batch = full_batch["input_ids"]

            diff_batch = full_batch.reshape(grad_accum_steps, batch_size, full_len)

            # 1. Initialize the lists with None for each gradient accumulation batch
            full_active_pass_B = [None] * grad_accum_steps
            prev_ground_latent_states_B = [None] * grad_accum_steps

            start_step = 0
            training_step = 0

            for _ in range((Tdiff_steps//diff_steps)):

                for b in range(grad_accum_steps):

                    input_batch = diff_batch[b, :, :]

                    # --- A. Forward Pass ---
                    train_output, full_active_pass, _, prev_ground_latent_states = train_step(
                        ILDC_model= ILDC_model,
                        full_batch= input_batch, 
                        past_len= past_len, 
                        full_active_pass= full_active_pass_B[b],  
                        prev_ground_latent_states= prev_ground_latent_states_B[b], 
                        start_step= start_step, 
                        diff_steps= diff_steps
                    )

                    # Store the values for the next diffusion steps....
                    full_active_pass_B[b] = detach_state(full_active_pass)
                    prev_ground_latent_states_B[b] = detach_state(prev_ground_latent_states)


                    # print(train_output)
                    training_step = start_step + 1
                    train_loss, loss_items = train_policy(train_output, training_step, Tdiff_steps, config)

                    # --- B. Gradient Accumulation Scaling ---
                    # We divide the loss so the accumulated gradients equal a true large batch
                    loss_to_backward = train_loss / grad_accum_steps
                    loss_to_backward.backward()
                    

                    # Update running trackers (Detached from graph for memory safety)
                    running_loss += train_loss.detach().item()
                    running_logit += loss_items["logit_val"]
                    running_mse += loss_items["mse_val"]
                

                # --- C. Optimization Step (grad_accum_steps) ---
                # print("Optimizer Step UP")
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(ILDC_model.parameters(), max_norm=1.0)
                
                # Optimizer Step 
                xm.optimizer_step(optimizer)
                
                
                # Scheduler Step & Zero Grad
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # move to next diffusion step
                start_step += diff_steps

                # =============================================================



                
            # --- D. Observability & Logging ---
            if global_step % config.log_every == 0:
                # Average the metrics over the logging window
                log_window = (config.log_every * grad_accum_steps * Tdiff_steps // diff_steps )
                avg_loss = running_loss / log_window
                avg_log  = running_logit / log_window
                avg_mse  = running_mse / log_window
                
                current_dm_lr = optimizer.param_groups[0]['lr']

                # Calling .item() here forces a TPU sync, which is perfectly safe 
                # because we only do it every config.log_every steps!
                xm.master_print(
                    f"Epoch {epoch} | Step {global_step}/{total_optim_steps} | "
                    f"Loss: {avg_loss.item():.4f} "
                    f"(Logit: {avg_log:.4f}, MSE: {avg_mse:.4f}) | "
                    f"LR_DM: {current_dm_lr:.2e}"
                )
                
                # Reset trackers
                running_loss = 0.0
                running_logit = 0.0
                running_mse = 0.0

            # --- E. Saving Checkpoints ---
            if global_step > 0 and global_step % config.save_every == 0:
                checkpoint_path = os.path.join(config.checkpoint_dir, f"dc_model_step_{global_step}.pt")
                # xm.save ensures only the master TPU node writes to disk
                xm.save(ILDC_model.dc_model.state_dict(), checkpoint_path)
                xm.master_print(f"--> Saved Checkpoint to {checkpoint_path}")


    xm.master_print("Training complete!")
    xm.save(ILDC_model.dc_model.state_dict(), os.path.join(config.checkpoint_dir, "dc_model_final.pt"))


def main():
    os.environ.pop('TPU_PROCESS_ADDRESSES', None)
    os.environ.pop('CLOUD_TPU_TASK_ID', None)
    os.environ["PJRT_DEVICE"] = "TPU"
    
    print("Launching PyTorch XLA training on Kaggle TPU...")
    xmp.spawn(train_fn, nprocs=None, start_method='spawn')

if __name__ == "__main__":
    main()

