# ===================================================================================
  

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModelForCausalLM

from ILDC_Model import ILDC
from Train_Step import train_step
from data_loader_xla import get_dataloader_xla

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
        self.past_len = 2048    
        self.future_len = 512   
        self.batch_size = 1     
        self.gradient_accumulation_steps = 16
        self.learning_rate = 1e-4
        self.epochs = 10
        self.checkpoint_dir = "checkpoints"
        self.log_every = 1      
        self.save_every = 100   
        self.lambda_latent = 0.1
        self.lambda_refine = 9.0
        self.diffusion_steps = 2

        
def train_fn(index):
    config = TrainingConfig()
    chunk_size = config.past_len + config.future_len + 1
    
    device = torch_xla.device()     
    world_size = xr.world_size()    
    rank = xr.global_ordinal()      
    
    xm.master_print("=" * 60)
    xm.master_print("  ILDC Stage 1 Warm-up Training (PyTorch XLA / TPU)")
    xm.master_print("  Mode: RL Policy Gradient (Zero AR Backprop)")
    xm.master_print(f"  World size: {world_size} TPU cores")
    xm.master_print("=" * 60)
    
    dataloader, sampler = get_dataloader_xla(
        model_id=config.model_id, dataset_name=config.dataset_name, 
        dataset_config=config.dataset_config, chunk_size=chunk_size,
        batch_size=config.batch_size, world_size=world_size, rank=rank
    )
    mp_dataloader = pl.MpDeviceLoader(dataloader, device)
    
    xm.master_print("Loading ILDC Model...")
    ILDC_model = ILDC(config.checkpoint_dir, device)
    ILDC_model.to(torch.bfloat16)
    ILDC_model.to(device)


    ILDC_model.dc_model.train()  
    optimizer = torch.optim.AdamW([{'params': ILDC_model.dc_model.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}])
    
    if rank == 0: os.makedirs(config.checkpoint_dir, exist_ok=True)
    # ==============================================================================
    # 2. SCHEDULER SETUP (COSINE + WARMUP)
    # ==============================================================================
    grad_accum_steps = config.gradient_accumulation_steps # e.g., 16 or 32
    total_batches = len(mp_dataloader)
    total_optim_steps = (total_batches // grad_accum_steps) * config.epochs
    warmup_steps = int(total_optim_steps * 0.05) # 5% of training is warmup

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
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        
        for batch_idx, full_batch in enumerate(mp_dataloader):
            
            # --- A. Forward Pass ---
            train_loss, loss_items = train_step(
                ILDC_model, full_batch, 
                config.past_len, 
                config.diffusion_steps, 
                config
            )
            
            # --- B. Gradient Accumulation Scaling ---
            # We divide the loss so the accumulated gradients equal a true large batch
            loss_to_backward = train_loss / grad_accum_steps
            loss_to_backward.backward()

            # Update running trackers (Detached from graph for memory safety)
            running_loss += train_loss.detach()
            running_logit += loss_items["logit_val"]
            running_mse += loss_items["mse_val"]

            # --- C. Optimization Step (Only every 'grad_accum_steps') ---
            if (batch_idx + 1) % grad_accum_steps == 0:
                
                # 1. Gradient Clipping (MANDATORY for Diffusion loops)
                torch.nn.utils.clip_grad_norm_(ILDC_model.parameters(), max_norm=1.0)
                
                # 2. XLA Optimizer Step (Handles TPU graph sync natively)
                xm.optimizer_step(optimizer)
                
                # 3. Scheduler Step & Zero Grad
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # --- D. Observability & Logging ---
                if global_step % config.log_every == 0:
                    # Average the metrics over the logging window
                    avg_loss = running_loss / (config.log_every * grad_accum_steps)
                    avg_log  = running_logit / (config.log_every * grad_accum_steps)
                    avg_mse  = running_mse / (config.log_every * grad_accum_steps)
                    
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
                    xm.save(ILDC_model.state_dict(), checkpoint_path)
                    xm.master_print(f"--> Saved Checkpoint to {checkpoint_path}")

    xm.master_print("Training complete!")
    xm.save(ILDC_model.state_dict(), os.path.join(config.checkpoint_dir, "dc_model_final.pt"))

def main():
    os.environ.pop('TPU_PROCESS_ADDRESSES', None)
    os.environ.pop('CLOUD_TPU_TASK_ID', None)
    os.environ["PJRT_DEVICE"] = "TPU"
    
    print("Launching PyTorch XLA training on Kaggle TPU...")
    xmp.spawn(train_fn, nprocs=None, start_method='spawn')

if __name__ == "__main__":
    main()

