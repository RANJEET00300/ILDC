import torch
import torch.nn as nn
import torch.nn.functional as F


from .helper_fn import RMSNorm, apply_rotary_pos_emb, get_timestep_embedding
from .config import ModelConfig
from .ar_model import MLP


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies Adaptive LayerNorm (adaLN) modulation.

    Dynamically scales and shifts the normalized hidden states
    based on the current diffusion timestep condition.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiT_Block(nn.Module):
    """
    Diffusion Transformer Block.

    Contains two novel attention mechanisms:
    1. Dual Cross-Attention (daGPAM): Attends to LLM KV caches using positive/negative
       queries and a bounded learnable skew (S).
    2. Cog Self-Attention: Latents attend to each other using sign * softmax(|scores|)
       to allow negative spatial correlations.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

        # Self Attention in DiT: use the existing weights from the LLM
        self.self_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.self_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.self_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.self_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # 1. RMSNorm normalizes the scale drift caused by sum(|a|) = 1
        self.cog_norm = RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)

        # mlp layers
        self.mlp = MLP(config)

        # Cross Attention in DiT (attends to LLM KV-caches)| New weights
        self.max_skew = 1.0
        self.cross_q_pos = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.relu = nn.ReLU()
        self.cross_q_neg = nn.Linear(
            self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False
        )

        self.qc_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Learnable Skew Parameter per head (initialized to 0)
        # Shape: [num_heads]
        self.skew_logits = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

        self.cross_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # 1. Adaptive LayerNorms for standard DiT modulation (Timestep/Class)
        self.norm1 = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False, eps=config.rms_norm_eps
        )
        self.norm2 = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False, eps=config.rms_norm_eps
        )
        self.norm3 = nn.LayerNorm(
            self.hidden_size, elementwise_affine=False, eps=config.rms_norm_eps
        )

        # Modulation Layer (adaLN)
        # Maps the conditioning vector 'c' to 9 modulation parameters:
        # 3 shifts, 3 scales, 3 gates (for self-attn, cross-attn, and MLP)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_size, 9 * self.hidden_size, bias=True)
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Zero-out the adaLN modulation layers for stable training."""
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_kv: torch.Tensor,
        Cond_Vector: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single Diffusion step in the DiT tower.

        Args:
            hidden_states (torch.Tensor): Shape (B, seq_len, hidden_size). The noisy latents.
            latent_kv (torch.Tensor): Shape (B, prev_latent_len, 2, num_kv_heads, head_dim).
                Self-Attention KV cache of the latents.
            Cond_Vector (torch.Tensor): Shape (B, hidden_size). Timestep conditioning vector.
            position_ids (torch.Tensor): Shape (1, seq_len). Strided positions for the latents.
            kv_cache (torch.Tensor): Shape (B, active_kv_len, 2, num_kv_heads, head_dim).
                Uncompressed LLM KV cache to compress.

        Returns:
            Tuple of updated hidden_states and the generated latent_kv.
        """

        bsz, q_len, _ = hidden_states.size()

        # 1. Generate modulation parameters from global conditioning 'c'
        # Chunk into 9 pieces for our 3 sub-blocks (shift, scale, gate for each)
        modulations = self.adaLN_modulation(Cond_Vector).chunk(9, dim=1)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulations

        # -----------------------------------------
        # 2. Cross-Attention Block (Compression <- LLM KV)
        # -----------------------------------------
        normed_h_cross = modulate(self.norm2(hidden_states), shift_cross, scale_cross)
        q_cross_pos_raw = self.cross_q_pos(normed_h_cross)
        q_cross_pos = q_cross_pos_raw.view(bsz, q_len, self.num_heads, self.head_dim)

        q_cross_neg_raw = self.cross_q_neg(self.relu(q_cross_pos_raw))
        q_cross_neg = q_cross_neg_raw.view(bsz, q_len, self.num_heads, self.head_dim)

        q_cross_pos = self.qc_norm(q_cross_pos).transpose(1, 2)
        q_cross_neg = self.qc_norm(q_cross_neg).transpose(1, 2)

        q_cross_pos, _ = apply_rotary_pos_emb(
            q_cross_pos, None, position_ids, self.rope_theta, self.head_dim
        )
        q_cross_neg, _ = apply_rotary_pos_emb(
            q_cross_neg, None, position_ids, self.rope_theta, self.head_dim
        )

        cond_k, cond_v = kv_cache[:, :, 0, ...], kv_cache[:, :, 1, ...]
        cond_k, cond_v = cond_k.transpose(1, 2), cond_v.transpose(1, 2)

        num_queries_per_kv_cross = self.num_heads // cond_k.shape[1]
        cond_k = (
            cond_k.unsqueeze(2)
            .expand(-1, -1, num_queries_per_kv_cross, -1, -1)
            .reshape(bsz, self.num_heads, cond_k.size(2), self.head_dim)
        )

        cond_v = (
            cond_v.unsqueeze(2)
            .expand(-1, -1, num_queries_per_kv_cross, -1, -1)
            .reshape(bsz, self.num_heads, cond_v.size(2), self.head_dim)
        )

        # here we implement the daGPAM cross-Attention:

        # 3. Calculate Dual Attention Scores
        scores_pos = (q_cross_pos @ cond_k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores_neg = (q_cross_neg @ cond_k.transpose(-2, -1)) / (self.head_dim**0.5)

        attn_pos = F.softmax(scores_pos, dim=-1)
        attn_neg = F.softmax(scores_neg, dim=-1)

        # 4. Bounded Learnable Skew (S)
        # Sigmoid keeps it positive, max_skew prevents out-of-distribution explosion
        S = torch.sigmoid(self.skew_logits) * self.max_skew

        # 5. Affine Combination (Guaranteed to sum to 1)
        attn_combined = (1 + S) * attn_pos - S * attn_neg

        # 6. Apply to raw LLM Values
        cross_attn_out = attn_combined @ cond_v

        cross_attn_out = cross_attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        # Add residual with gating
        hidden_states = hidden_states + gate_cross.unsqueeze(1) * self.cross_o(cross_attn_out)

        # -----------------------------------------
        # 3. Self-Attention Block (Latent <-> Latent)
        # -----------------------------------------
        normed_hstates = modulate(self.norm1(hidden_states), shift_msa, scale_msa)

        q = self.self_q(normed_hstates).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.self_k(normed_hstates).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.self_v(normed_hstates).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        q_self = self.q_norm(q).transpose(1, 2)  # [B, H, Q_len, D]
        k_self = self.k_norm(k).transpose(1, 2)
        v_self = v.transpose(1, 2)

        # Only rotate the parts that look at eachothers
        q_self, k_self = apply_rotary_pos_emb(
            q_self, k_self, position_ids, self.rope_theta, self.head_dim
        )

        # Save the pure generated KV for Diffusion model
        # Shape: (B, K_hidden_len, 2, num_heads, head_dim)
        generated_kv = torch.stack([k_self, v_self], dim=2)
        output_kv = generated_kv.transpose(1, 3)

        # if previous latent kv are available:
        prev_k_self, prev_v_self = latent_kv[:, :, 0, ...], latent_kv[:, :, 1, ...]
        prev_k_self, prev_v_self = prev_k_self.transpose(1, 2), prev_v_self.transpose(1, 2)

        k_self = torch.cat([prev_k_self, k_self], dim=2)
        v_self = torch.cat([prev_v_self, v_self], dim=2)

        # Self-Attention (Spatial structure)
        num_queries_per_kv_self = self.num_heads // self.num_kv_heads
        k_self = (
            k_self.unsqueeze(2)
            .expand(-1, -1, num_queries_per_kv_self, -1, -1)
            .reshape(bsz, self.num_heads, k_self.size(2), self.head_dim)
        )

        v_self = (
            v_self.unsqueeze(2)
            .expand(-1, -1, num_queries_per_kv_self, -1, -1)
            .reshape(bsz, self.num_heads, v_self.size(2), self.head_dim)
        )

        # here we implement the cog  self-Attention:
        # Raw scores
        scores = (q_self @ k_self.transpose(-2, -1)) / (self.head_dim**0.5)

        # 2. Cog Attention Math: Sign * Softmax(|Scores|)
        abs_scores = torch.abs(scores)
        attn_magnitude = F.softmax(abs_scores, dim=-1)
        attn_weights = torch.sign(scores) * attn_magnitude

        # 3. Apply attention to values
        self_attn_out = attn_weights @ v_self

        self_attn_out = self_attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        # 4. Stabilize output scale for diffusion and gate it
        self_attn_out = self.cog_norm(self_attn_out)

        # Add residual with gating
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * self.self_o(self_attn_out)

        # -----------------------------------------
        # 4. MLP Block
        # -----------------------------------------
        normed_h_mlp = modulate(self.norm3(hidden_states), shift_mlp, scale_mlp)

        mlp_out = self.mlp(normed_h_mlp)

        # Add residual with gating
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * mlp_out

        return hidden_states, output_kv  # Compressed KVs from the diffusion


# Latent Diffusion KV Compression Model:
# BPTT; early stopping based....not a const 8 steps diffusion...
# put a head on latent-states with stop/continue binary prediction...if they say stop..
# stop else continue..till the max diffusion steps....
# put mask/light; if mask, kill the latent, is light, add another latent...


class CompDiTModel(nn.Module):
    """
    The Core Latent Diffusion KV Compressor.

    Wraps the DiT blocks. Orchestrates the iterative denoising process
    over a sequence of latent tokens, conditioned on the LLM's uncompressed KV cache.
    """

    def __init__(self):
        super().__init__()
        self.config = ModelConfig()

        self.X_factor = self.config.X_factor

        self.comp_layers = nn.ModuleList(
            [DiT_Block(self.config, i) for i in range(self.config.num_hidden_layers)]
        )

        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Timestep conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.SiLU(),
            nn.Linear(self.config.intermediate_size, self.config.hidden_size),
        )

    def forward(
        self,
        latent_states: torch.Tensor,
        Latent_KVs: torch.Tensor,
        kv_caches: torch.Tensor,
        timestep: int,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the full Diffusion sequence at a specific timestep.

        Args:
            latent_states (torch.Tensor): Shape (B, K_latent_len, hidden_size). Noisy latents.
            Latent_KVs (torch.Tensor, optional):
                Self-attention KV cache of latents from prior steps.
            kv_caches (torch.Tensor):
                Shape (B, layers, seq_len, 2, heads, dim). LLM uncompressed KVs.
            timestep (int): The current diffusion step.
            start_pos (int, optional): Spatial start position to anchor the Strided RoPE.

        Returns:
            Tuple of the refined latent_states and updated Latent_KVs.
        """

        _, K_latent_len, _ = latent_states.shape
        B, num_layer, seq_len, _, num_heads, head_dim = kv_caches.shape

        # Get device and dtype from the KVs (e.g., bfloat16)
        device = kv_caches.device
        target_dtype = kv_caches.dtype

        # Generate strided position-ids
        # This ensures that noise token 'i' is associated with position 'i * X_factor'
        # Example: If X_factor=2, IDs are start_pos + [0, 2, 4, 6, ...] from start_pos
        # TODO: need to implement the position_id profiling,.. where do we need and
        # what type of profile...exponential decaying, uniform, ...
        position_ids = torch.arange(
            start_pos,
            start_pos + K_latent_len * self.X_factor,
            step=self.X_factor,
            device=device,
            dtype=torch.long,
        )

        raw_t_emb = get_timestep_embedding(B, timestep, self.config.hidden_size, device)

        # 2. CAST to match hidden_states (bfloat16) before passing to Linear layers
        t_emb = self.time_embed(raw_t_emb.to(target_dtype))

        # Expand for Batch dimension: [1, K_latent_len]
        # This matches the shape expected by apply_rotary_pos_emb_DiT
        position_ids = position_ids.unsqueeze(0)

        # Add noise to latent states + spread the variance|mean: TODO
        # Create the input Gaussian noise natively in the correct dtype
        # Shape: [Batch, Sequence_Length, Hidden_Size]
        noisy_states = torch.randn(
            B, K_latent_len, self.config.hidden_size, device=device, dtype=target_dtype
        )

        latent_states = 0.8 * latent_states + 0.6 * noisy_states  # 0.8**2 + 0.6**2 = 1

        # (Layers, Batch, K_latent_len, 2, Heads, Dim)
        if Latent_KVs is None:
            Latent_KVs = torch.randn(
                (B, num_layer, K_latent_len, 2, num_heads, head_dim),
                device=device,
                dtype=target_dtype,
            )

        for i, layer in enumerate(self.comp_layers):
            # Pass the corresponding layer's KV cache from the AR model
            latent_states, latent_kv = layer(
                latent_states, Latent_KVs[:, i, ...], t_emb, position_ids, kv_caches[:, i, ...]
            )
            Latent_KVs[:, i] = latent_kv
        # as we use latent kvs as compressed kvs for the ar_model
        return latent_states, Latent_KVs


"""
We are asking CompDiTModel to produce best latent vector which will act as query
which then going to look at the LLM KVs and produce best latent KVS as compressed KVS to
be used by LLM for further autoregression;

I doubt it!

Instead; we should keep the prev. step latent KVs and tell the diffusion model to reason
onto itself what he know and what he need to know more which will produce best KVs to act
as compressed KVs...

Just Implemented...
"""
