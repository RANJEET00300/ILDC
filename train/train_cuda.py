import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# Append parent dir to sys.path to import model components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.MoT_model import ILDC
from Train_Step import train_step
from dataset.data_loader import get_dataloader


class TrainingConfig:
    def __init__(self):
        self.model_id = "google/gemma-3-1b-it"
        self.dataset_name = "wikitext"
        self.dataset_config = "wikitext-2-raw-v1"
        self.past_len = 2048   # N context to compress
        self.future_len = 512  # Target tokens
        self.batch_size = 2
        self.gradient_accumulation_steps = 4
        self.learning_rate = 1e-5
        self.epochs = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = "checkpoints"
        self.log_every = 1      
        self.save_every = 100 
        self.lambda_latent = 0.4
        self.lambda_refine = 9.0
        self.diffusion_steps = 8

def main():
    config = TrainingConfig()
    chunk_size = config.past_len + config.future_len + 1
    
    print("--- ILDC Stage 1 Warm-up Training ---")
    print(f"Device: {config.device}")
    print(f"Context Split: {config.past_len} drop -> Compress, {config.future_len} target")
    print(f"Batch Size: {config.batch_size} (Grad Accum: {config.gradient_accumulation_steps})")
    
    # 1. Dataset
    dataloader = get_dataloader(
        model_id=config.model_id,
        dataset_name=config.dataset_name, 
        dataset_config=config.dataset_config,
        chunk_size=chunk_size,
        batch_size=config.batch_size
    )
    
    
    ILDC_model = ILDC(use_compressor=True)
    ILDC_model.base_model.to(torch.bfloat16)
    ILDC_model.dm_tower.to(torch.bfloat16)
    ILDC_model.to(device)
    for param in ILDC_model.base_model.parameters(): param.requires_grad = False
    ILDC_model.train()  
    
    optimizer = torch.optim.AdamW(ILDC_model.dm_tower.parameters(), lr=config.learning_rate)
    
    # 4. Training Loop
    print("Starting Training Loop...")
    global_step = 0
    optimizer.zero_grad()
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    for epoch in range(config.epochs):
        for batch_idx, full_batch in enumerate(dataloader):
            batch = batch.to(config.device)
            
            # Forward pass and Loss computation with Mixed Precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = train_step(ILDC_model, full_batch, config.past_len, config)

                # Scale loss by accumulation steps
                loss = loss / config.gradient_accumulation_steps
            
            # Backprop
            loss.backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.log_every == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Total Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")
                
                if global_step > 0 and global_step % save_every == 0:
                    print(f"Saving checkpoint at step {global_step}...")
                    torch.save(ILDC_model.dm_tower.state_dict(), os.path.join(config.checkpoint_dir, f"dm_tower_epoch_step_{global_step}.pt"))

    print("Training complete!")
    torch.save(ILDC_model.dm_tower.state_dict(), os.path.join(config.checkpoint_dir, "dm_tower_final.pt"))

     
if __name__ == "__main__":
    main()
