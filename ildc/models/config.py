import torch


class ModelConfig:
    def __init__(self):
        self.dtype = torch.bfloat16
        self.vocab_size = 262144  # vocab_size
        self.hidden_size = 1152
        self.intermediate_size = 6912
        self.num_hidden_layers = 26
        self.num_attention_heads = 4
        self.num_key_value_heads = 1
        self.head_dim = 256
        self.rms_norm_eps = 1e-06
        self.rope_theta = 1000000
        self.rope_local_base_freq = 10000
        self.window_size = 8192  # SWA Window Size
        self.X_factor = 16  # Compression ratio for Strided RoPE


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
