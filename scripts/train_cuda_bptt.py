# ===================================================================================

import os
import torch
from transformers import get_cosine_schedule_with_warmup

from ildc.models.ildc_model import ILDC
from ildc.trainer.train_step_bptt import train_step
from ildc.data.data_loader import get_wikipedia_batches

"""
ILDC Training — PyTorch CUDA
=======================================================
"""


class TrainingConfig:
    def __init__(self):
        self.model_id = "google/gemma-3-1b-it"
        self.dataset_name = "wikimedia/wikipedia"
        self.dataset_config = "20241101.en"
        self.checkpoint_dir = "./checkpoints"

        self.past_len = 2048
        self.future_len = 512
        self.full_len = 2561

        self.learning_rate = 1e-4
        self.lambda_latent = 0.1

        self.log_every = 1
        self.save_every = 100

        self.batch_size = 1
        self.gradient_accumulation_steps = 32
        self.total_diffusion_steps = 8
        self.diff_steps = 2  # hard training with Block, 2 steps per/

        self.N_batches = 16284
        self.epochs = self.total_diffusion_steps - self.diff_steps + 1


def train_fn():
    config = TrainingConfig()

    # 1. SETUP CUDA DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  ILDC Stage 1 Warm-up Training (PyTorch CUDA / GPU)")
    print(f"  Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU Model: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # 2. DATALOADER (Standard PyTorch iteration)
    dataloader = get_wikipedia_batches(
        tokenizer_path=config.model_id,
        batch_size=config.batch_size,
        required_length=config.full_len,
        max_batches=config.N_batches,
    )

    print("Loading ILDC Model...")
    ILDC_model = ILDC(config.checkpoint_dir, device)
    ILDC_model.to(torch.bfloat16)
    ILDC_model.to(device)

    ILDC_model.dc_model.train()
    optimizer = torch.optim.AdamW(
        [{"params": ILDC_model.dc_model.parameters(), "lr": 1e-4, "weight_decay": 0.01}]
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ==============================================================================
    # 2. SCHEDULER SETUP (COSINE + WARMUP)
    # ==============================================================================
    grad_accum_steps = config.gradient_accumulation_steps  # e.g., 16 or 32
    total_optim_steps = (config.N_batches // grad_accum_steps) * config.epochs
    warmup_steps = int(total_optim_steps * 0.05)  # 5% of training is warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps
    )

    # ==============================================================================
    # 3. OBSERVABILITY TRACKERS
    # ==============================================================================
    global_step = 0
    running_loss = 0.0
    running_logit = 0.0
    running_mse = 0.0

    # ==============================================================================
    # 4. THE CUDA TRAINING LOOP
    # ==============================================================================
    print(f"Starting Training! Total Opt Steps: {total_optim_steps} | Warmup: {warmup_steps}")

    for epoch in range(config.epochs):
        optimizer.zero_grad()

        for batch_idx, full_batch in enumerate(dataloader):
            # Move the batch explicitly to the CUDA device
            input_ids = full_batch["input_ids"].to(device)

            # --- A. Forward Pass ---
            # Note: Changed `config.diffusion_steps` to `config.total_diffusion_steps`
            # to match the attributes available in TrainingConfig
            train_loss, loss_items = train_step(ILDC_model, input_ids, config)

            # --- B. Gradient Accumulation Scaling ---
            # We divide the loss so the accumulated gradients equal a true large batch
            loss_to_backward = train_loss / grad_accum_steps
            loss_to_backward.backward()

            # Update running trackers (Using .item() for memory safety on CUDA)
            running_loss += train_loss.item()
            running_logit += loss_items["logit_val"]
            running_mse += loss_items["mse_val"]

            # --- C. Optimization Step (Only every 'grad_accum_steps') ---
            if (batch_idx + 1) % grad_accum_steps == 0:
                # 1. Gradient Clipping
                torch.nn.utils.clip_grad_norm_(ILDC_model.parameters(), max_norm=1.0)

                # 2. Standard PyTorch Optimizer Step
                optimizer.step()

                # 3. Scheduler Step & Zero Grad
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # --- D. Observability & Logging ---
                if global_step % config.log_every == 0:
                    # Average the metrics over the logging window
                    avg_loss = running_loss / (config.log_every * grad_accum_steps)
                    avg_log = running_logit / (config.log_every * grad_accum_steps)
                    avg_mse = running_mse / (config.log_every * grad_accum_steps)

                    current_dm_lr = optimizer.param_groups[0]["lr"]

                    print(
                        f"Epoch {epoch} | Step {global_step}/{total_optim_steps} | "
                        f"Loss: {avg_loss:.4f} "
                        f"(Logit: {avg_log:.4f}, MSE: {avg_mse:.4f}) | "
                        f"LR_DM: {current_dm_lr:.2e}"
                    )

                    # Reset trackers
                    running_loss = 0.0
                    running_logit = 0.0
                    running_mse = 0.0

                # --- E. Saving Checkpoints ---
                if global_step > 0 and global_step % config.save_every == 0:
                    checkpoint_path = os.path.join(
                        config.checkpoint_dir, f"dc_model_step_{global_step}.pt"
                    )
                    torch.save(ILDC_model.dc_model.state_dict(), checkpoint_path)
                    print(f"--> Saved Checkpoint to {checkpoint_path}")

    print("Training complete!")
    torch.save(
        ILDC_model.dc_model.state_dict(), os.path.join(config.checkpoint_dir, "dc_model_final.pt")
    )


def main():
    print("Launching PyTorch CUDA training on GPU...")
    train_fn()


if __name__ == "__main__":
    main()
