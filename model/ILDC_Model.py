import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# ==========================================
# 1. THE ILDC based LLM ARCHITECTURE
"""
A Standard Auto-regressive model LLM; based on Gemma-3-1b
Modified to support ILDC (Iterative Latent Diffusion for Continuous KV Compression)
"""
# ==========================================
class Config:
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
        self.X_factor = 16       # Compression ratio for Strided RoPE

        
        
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # Force float32 for math stability, then strictly cast back to input dtype
        x_f32 = x.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(x.dtype)) * (1.0 + self.weight)

        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


def apply_rotary_pos_emb(q, k, position_ids, rope_theta, head_dim):
    # Dynamically determine the reference tensor to pull the device and dtype
    ref_tensor = q if q is not None else k
    
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=ref_tensor.device) / head_dim))
    t = position_ids[0].to(torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    # STRICT CAST TO MATCH DTYPE (bfloat16)
    cos = emb.cos().to(ref_tensor.dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(ref_tensor.dtype).unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Only apply to Q if Q is actually provided
    q_embed = None
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        
    # Only apply to K if K is actually provided
    k_embed = None
    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
    return q_embed, k_embed
    



# ===========================================================================================================

# ===========================================================================================================



# Grouped Query Multihead Attention:
class GQMHAttention_AR(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config 
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    
    def forward(self, hidden_states, position_ids, attention_mask=None, active_kv=None, compressed_kv=None):
        """
        hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size) 
                    [Tokens to be processed] 
        
        active_kv: torch.Tensor of shape (
                                    B, active_kv_len, 2, num_kv_heads, head_dim
                                ) [KVs from previous forward passes]
                                
        compressed_kv: torch.tensor of shape (
                                    B, K_hidden_len, 2, num_kv_heads, head_dim
                                ) from DiT Tower (compressed)
        """
        
        B, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, position_ids, self.rope_theta, self.head_dim)

        # The new KV values that will be appended to the cache
        generated_kv = torch.stack([k, v], dim=2)
        new_kv = generated_kv.transpose(1, 3) # Shape: (B, q_len, 2, num_kv_heads, head_dim)

        # Build full K and V | from preceeding tokens
        full_k, full_v = k, v

        # Active KV-Cache from previous AR tokens: uncompressed + Compressed
        if active_kv is not None:
            active_k, active_v = active_kv[:, :, 0, ...], active_kv[:, :, 1, ...]
            active_k, active_v = active_k.transpose(1,2), active_v.transpose(1,2)
            full_k = torch.cat([active_k, full_k], dim=2)
            full_v = torch.cat([active_v, full_v], dim=2)

        # Prepend compressed latents from the diffusion model
        if compressed_kv is not None:
            comp_k, comp_v = compressed_kv[:, :, 0, ...], compressed_kv[:, :, 1, ...]
            comp_k, comp_v = comp_k.transpose(1,2), comp_v.transpose(1,2)
            full_k = torch.cat([comp_k, full_k], dim=2)
            full_v = torch.cat([comp_v, full_v], dim=2)

        # Repeat K/V for Grouped Query Attention
        full_k = full_k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        full_v = full_v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn_output = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.o_proj(attn_output), new_kv




    
class LLM_block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = GQMHAttention_AR(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_ids, attention_mask=None, active_kv=None, compressed_kv=None):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x, new_kv = self.self_attn(
            x, position_ids, attention_mask, 
            active_kv, compressed_kv)
        
        x = self.post_attention_layernorm(x)
        hidden_states = residual + x

        residual = hidden_states
        x = self.pre_feedforward_layernorm(hidden_states)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        hidden_states = residual + x

        return hidden_states, new_kv






# ===========================================================================================================

# ===========================================================================================================




class CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=0)
        self.gen_layers = nn.ModuleList([LLM_block(self.config, i) for i in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, active_kv_caches=None, compressed_kv_caches=None):
        """
        input_ids: torch.Tensor of shape (batch_size, seq_len) 
                    [Tokens to be processed] 
        
        active_kv_caches: torch.Tensor of shape (
                                    B, num_layer, active_kv_len, 2, num_kv_heads, head_dim
                                ) [KVs from previous forward passes] if not compressed
                                
        compressed_kv_caches: torch.tensor of shape (
                                    B, num_layer, K_hidden_state, 2, num_kv_heads, head_dim
                                ) from DiT Tower 
        """
        
        B, seq_len = input_ids.shape
        
        past_len = active_kv_caches[0].shape[1] if active_kv_caches is not None else 0
        comp_len = compressed_kv_caches[0].shape[1] if compressed_kv_caches is not None else 0

        # Calculate start position based on X_factor compression if compressed KVs exist
        if comp_len > 0:
            start_pos = (comp_len * self.config.X_factor) + past_len
        else:
            start_pos = past_len
        
        position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        hidden_states = self.embed_tokens(input_ids)
        
        # Standard Gemma scaling
        hidden_states = hidden_states * math.sqrt(self.config.hidden_size)

        causal_mask = None
        if seq_len > 1:
            full_k_len = comp_len + past_len + seq_len
            pad_len = comp_len + past_len
            min_dtype = torch.finfo(hidden_states.dtype).min
            causal_mask = torch.full((seq_len, seq_len), fill_value=min_dtype, device=input_ids.device, dtype=hidden_states.dtype)
            causal_mask.triu_(diagonal=1)
            
            if pad_len > 0:
                past_mask = torch.zeros((seq_len, pad_len), device=input_ids.device, dtype=hidden_states.dtype)
                causal_mask = torch.cat([past_mask, causal_mask], dim=1)
                
            causal_mask = causal_mask.view(1, 1, seq_len, full_k_len)
       
        # --- FIX: ALLOCATE TENSOR FOR THE ENTIRE ACCUMULATED KV LENGTH ---
        total_active_len = past_len + seq_len
        new_active_kv_caches = torch.empty(
            (B, self.config.num_hidden_layers, total_active_len, 2, self.config.num_key_value_heads, self.config.head_dim),
            device=input_ids.device, dtype=hidden_states.dtype
        )
        
        for i, layer in enumerate(self.gen_layers):
            layer_active_kv = active_kv_caches[:, i, ...] if active_kv_caches is not None else None
            layer_comp_kv = compressed_kv_caches[:, i, ...] if compressed_kv_caches is not None else None
            
            hidden_states, new_kv = layer(
                hidden_states, 
                position_ids, 
                attention_mask=causal_mask, 
                active_kv=layer_active_kv, 
                compressed_kv=layer_comp_kv
            )

            # --- FIX: CONCATENATE HISTORY WITH NEW KV ---
            if layer_active_kv is not None:
                updated_layer_kv = torch.cat([layer_active_kv, new_kv], dim=1)
            else:
                updated_layer_kv = new_kv
            
            new_active_kv_caches[:, i, ...] = updated_layer_kv  
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "active_kv_caches": new_active_kv_caches
        } 






# ===========================================================================================================

# ===========================================================================================================


def modulate(x, shift, scale):
    """Applies adaptive LayerNorm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)




class DiT_Block(nn.Module):
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
        self.cross_q_neg = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False)

        self.qc_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Learnable Skew Parameter per head (initialized to 0)
        # Shape: [num_heads]
        self.skew_logits = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

        self.cross_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
  
  
        # 1. Adaptive LayerNorms for standard DiT modulation (Timestep/Class)
        self.norm1 = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=config.rms_norm_eps)
        self.norm2 = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=config.rms_norm_eps)
        self.norm3 = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=config.rms_norm_eps)


        # Modulation Layer (adaLN)
        # Maps the conditioning vector 'c' to 9 modulation parameters:
        # 3 shifts, 3 scales, 3 gates (for self-attn, cross-attn, and MLP)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 9 * self.hidden_size, bias=True)
        )
        
        self.initialize_weights()



    def initialize_weights(self):
        """Zero-out the adaLN modulation layers for stable training."""
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)



    def forward(self, hidden_states, Cond_Vector, position_ids, kv_cache):

        """
        hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size) 
                    [Latent Tokens] 
        
        kv_cache: torch.Tensor of shape (
                                    B, active_kv_len, 2, num_heads, head_dim
                                ) [KVs from prior AR forward passes]
        """

        bsz, q_len, _ = hidden_states.size()

        # 1. Generate modulation parameters from global conditioning 'c'
        # Chunk into 9 pieces for our 3 sub-blocks (shift, scale, gate for each)
        modulations = self.adaLN_modulation(Cond_Vector).chunk(9, dim=1)
        (shift_msa, scale_msa, gate_msa, 
         shift_cross, scale_cross, gate_cross, 
         shift_mlp, scale_mlp, gate_mlp) = modulations


        # ===============================================================================
       
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

        q_cross_pos, _ = apply_rotary_pos_emb(q_cross_pos, None, position_ids, self.rope_theta, self.head_dim)
        q_cross_neg, _ = apply_rotary_pos_emb(q_cross_neg, None, position_ids, self.rope_theta, self.head_dim)
    

        cond_k, cond_v = kv_cache[:, :, 0, ...], kv_cache[:, :, 1, ...]
        cond_k, cond_v = cond_k.transpose(1,2),  cond_v.transpose(1,2)
       
    
        # Repeat cond_kv to match DiT heads
        cond_k = cond_k.repeat_interleave(self.num_heads // cond_k.shape[1], dim=1)
        cond_v = cond_v.repeat_interleave(self.num_heads // cond_v.shape[1], dim=1)
        
        # here we implement the daGPAM cross-Attention:

        # 3. Calculate Dual Attention Scores
        scores_pos = (q_cross_pos @ cond_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores_neg = (q_cross_neg @ cond_k.transpose(-2, -1)) / (self.head_dim ** 0.5)

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



        # =========================================================================================

        # -----------------------------------------
        # 3. Self-Attention Block (Latent <-> Latent)
        # -----------------------------------------
        normed_hstates = modulate(self.norm1(hidden_states), shift_msa, scale_msa)
      
        q = self.self_q(normed_hstates).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.self_k(normed_hstates).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.self_v(normed_hstates).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        q_self = self.q_norm(q).transpose(1, 2) # [B, H, Q_len, D]
        k_self = self.k_norm(k).transpose(1, 2)
        v_self = v.transpose(1, 2)
        
        # Only rotate the parts that look at eachothers
        q_self, k_self = apply_rotary_pos_emb(q_self, k_self, position_ids, self.rope_theta, self.head_dim)
    
        # Save the pure generated KV for Diffusion model
        # Shape: (B, K_hidden_len, 2, num_heads, head_dim)
        generated_kv = torch.stack([k_self, v_self], dim=2)
        output_kv = generated_kv.transpose(1,3)

        # Self-Attention (Spatial structure)
        # Expand K/V for GQA
        k_self = k_self.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v_self = v_self.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        

        # here we implement the cog  self-Attention:
        # Raw scores
        scores = (q_self @ k_self.transpose(-2, -1)) / (self.head_dim ** 0.5)

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

        return hidden_states, output_kv # Compressed KVs from the diffusion





# ===========================================================================================================

# ===========================================================================================================

def get_timestep_embedding(B, timestep, embedding_dim, device):
    timesteps = torch.full((B,), timestep*100, dtype=torch.long, device=device)
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


# Latent Diffusion KV Compression Model: 
# BPTT; early stopping based....not a const 8 steps diffusion...

class CompDiTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        
        self.X_factor =self.config.X_factor 

        self.comp_layers = nn.ModuleList([DiT_Block(self.config, i) for i in range(self.config.num_hidden_layers)])

        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        
        # Timestep conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.SiLU(),
            nn.Linear(self.config.intermediate_size, self.config.hidden_size)
        )


    def forward(self, latent_states, kv_caches, timestep, start_pos=0):

        """
        latent_states: torch.Tensor of shape (batch_size, K_latent_len, hidden_size) 
                    [Latent Tokens of Diffusion] 
        
        kv_caches: torch.Tensor of shape (
                                    B, num_hidden_layers, seq_len, 2, num_heads, head_dim
                                ) [KVs from AR forward passes]
        
        timestep: int [the step of diffusion]
        start_pos: int [Start-position of uncompressed KVs in KV-caches which we are going to compress]
        """

        B, K_latent_len, _ = latent_states.shape
        B, num_layer, seq_len, _, num_heads, head_dim = kv_caches.shape
        
        # Get device and dtype from the KVs (e.g., bfloat16)
        device = kv_caches.device
        target_dtype = kv_caches.dtype
        
        # Generate strided position-ids
        # This ensures that noise token 'i' is associated with position 'i * X_factor'
        # Example: If X_factor=2, IDs are start_pos + [0, 2, 4, 6, ...] from start_pos
        # need to implement the possition_id profiling,..... where do we need and what type of profile...
        # exponential decaying, uniform, ...TODO
        position_ids = torch.arange(
            start_pos, 
            start_pos + K_latent_len * self.X_factor, 
            step= self.X_factor, 
            device=device, 
            dtype=torch.long
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
            B, 
            K_latent_len, 
            self.config.hidden_size, 
            device=device, 
            dtype=target_dtype
        )
        latent_states = latent_states + noisy_states

        
        # (Layers, Batch, K_latent_len, 2, Heads, Dim)
        Compressed_KVs = torch.empty(
                (num_layer, B, K_latent_len, 2, num_heads, head_dim), 
                device=device, 
                dtype=target_dtype
            ) 

        for i, layer in enumerate(self.comp_layers):
            # Pass the corresponding layer's KV cache from the AR model
            latent_states, ckv = layer(latent_states, t_emb, position_ids, kv_caches[:, i, ...])
            Compressed_KVs[i] = ckv

        # (Batch, Layers, K_latent_len, Heads, Dim)
        Compressed_KVs = Compressed_KVs.transpose(0, 1).contiguous()

        return latent_states, Compressed_KVs 
 




# ===========================================================================================================

# ===========================================================================================================

def load_models(checkpoint_dir, device):

    print(f"Using device: {device}")
    print("Loading ILDC Architecture...")

    checkpoint = f"{checkpoint_dir}/dc_model_final.pt"
    
    model_id = "google/gemma-3-1b-it"
    # Instantiate models
    ar_model = CausalLM().to(torch.bfloat16)
    dc_model = CompDiTModel().to(torch.bfloat16)
    
    print("Downloading and mapping HF weights...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    gemma_state_dict = {}
    for key, value in hf_model.state_dict().items():
        new_key = key[6:] if key.startswith("model.") else key
        if "rotary_emb" not in new_key:
            gemma_state_dict[new_key] = value
    
    ar_model.load_state_dict(gemma_state_dict, strict=False)
        
    # We use ar_model mlp weight and freeze that, only train other weights
    if checkpoint is None:
        print("⚠️ Checkpoint not provided! Instantiating with untrained compressor weights")

        # Iterate over the ModuleLists to load the MLPs layer-by-layer
        for dc_layer, ar_layer in zip(dc_model.comp_layers, ar_model.gen_layers):
            dc_layer.mlp.load_state_dict(ar_layer.mlp.state_dict())

    elif os.path.exists(checkpoint): # Fixed from checkpoint_path
        print(f"Loading trained compressor weights from {checkpoint}...")
        state_dict = torch.load(checkpoint, map_location="cpu")
        dc_model.load_state_dict(state_dict)
        
    else:
        print(f"⚠️ Checkpoint '{checkpoint}' not found! Using untrained compressor weights for test.")

        # Iterate over the ModuleLists to load the MLPs layer-by-layer
        for dc_layer, ar_layer in zip(dc_model.comp_layers, ar_model.gen_layers):
            dc_layer.mlp.load_state_dict(ar_layer.mlp.state_dict())


    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Weights loaded successfully!")

    return ar_model.to(device), dc_model.to(device)



class ILDC(nn.Module):
    def __init__(self, checkpoint_path=None, device="cpu"):
        super().__init__()

        self.config = Config()
        self.device = device

        # Load Baseline Gemma 3 1B
        model_id = "google/gemma-3-1b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Instantiate and Load weights
        self.ar_model, self.dc_model = load_models(checkpoint_path, device)
        
        # Freeze base LLM weights initially
        for param in self.ar_model.parameters():
            param.requires_grad = False

        # Freeze base mlp weights initially
        for dc_layer in self.dc_model.comp_layers:
            for param in dc_layer.mlp.parameters():
                param.requires_grad = False




    # ===================================================================================
    # Compressing the KV-cache with Diffusion at an timestep with given laten and KVs 
    # Latent KV Compression Model

    def kv_compress(self, latent_states, kv_caches, start_step, diff_steps:int=1, start_pos=0):   
        """
        latent_states: torch.tensor of shape (B, K_latent, latent_size) [Diffusion Latent States]

        kv_caches: torch.Tensor of shape (
                                    B, num_hidden_layers, seq_len, 2, num_heads, head_dim
                                ) [KVs Compressed and Uncompressed Both]

        start_step: int [the timestep of input Latent-step if has been processed]
        diff_steps: int [Diffusion steps to go]
        start_pos: int [Start-position of uncompressed KVs in KV-caches which we are going to compress]

        Return: Latent States & New compressed KVs
        """

        B, num_layer, seq_len, _, num_heads, H = kv_caches.shape
        _, K_latent_len, latent_size = latent_states.shape 

        device = self.device
        target_dtype = self.config.dtype
      

        # 🔴 Ensure ALL incoming KV caches are strictly BFloat16
        safe_kv_caches = kv_caches.to(target_dtype)
        
        # The latent-states and compressed-kvs will be collect on each diffusion steps and will be given to AR model for student generation pass
        # Pre-allocate tensors Shape: (Steps, Batch, K_latent_len, Latent)
        all_latent_states = torch.empty(
            (diff_steps, B, K_latent_len, latent_size), 
            device=device, dtype=target_dtype
        )
        
        # Determine KV shapes (Adjust these based on your dc_model output)
        num_layers = self.config.num_hidden_layers
        
        all_compressed_kvs = torch.empty(
            (diff_steps, B, num_layers, K_latent_len, 2, num_heads, H),
            device=device, dtype=target_dtype
        )


        # 3. Diffusion Loop
        for t in range(diff_steps):
            timestep = t + start_step
            
            # Dit Model Pass
            # h: (B, K_latent_len, latent_size), kv: (B, layers, K_latent_len, 2, heads, head_dim)
            l, kv = self.dc_model(latent_states, safe_kv_caches, timestep, start_pos)
            
            # Direct assignment to pre-allocated memory
            all_latent_states[t] = l
            all_compressed_kvs[t] = kv
            
            # Update latent_states for next iteration
            latent_states = l 


        # 4. Reshape to (Batch, Steps, ...) 
        output_state = all_latent_states.transpose(0, 1).contiguous()

        output_compressed_kvs = all_compressed_kvs.transpose(0, 1).contiguous()

        return output_state, output_compressed_kvs





    # =================================================================================================

    def forward(
        self, input_ids, context_ids=None, 
        active_kv_caches= None, active_compressed_kv= None, 
        latent_states= None,
        start_step = 0, diff_steps=1, start_pos=None
        ): 
        """
        Forward pass mainly used for Student Pass during training.
        input_ids: the preceeding token which will be in native embedding 
        context_ids: the preceeding token which will be compressed before feeding to AR
        """
        X_factor = self.config.X_factor
        target_dtype = self.config.dtype

        # upcoming token for training
        B , future_len = input_ids.shape
        current_ids = input_ids

        # current preceeding token in context
        current_len = 0
        if context_ids is not None:
            _, current_len = context_ids.shape
            current_ids = torch.cat([context_ids, current_ids], dim=1)

       
        # effective seq-len to be compressed
        eff_seq_len = current_len

        if active_kv_caches is not None:
            _, _, KV_len, _, _, _ = active_compressed_kv.shape
            eff_seq_len += KV_len
            dropped_kvs =  active_kv_caches

        K_latent_len = (eff_seq_len + X_factor -1)// X_factor

        # Our start-pos for compression will start after the cmpressed kvs for efficient compression
        # if both the compressed_kv and start-pos are given,...we assume is as mistake and overwrite the start-pos
        if active_compressed_kv is not None:
            _, _, CompKV_len, _, _, _ = active_compressed_kv.shape
            if start_pos is not None:
                print(f"Overwriting start-pos with Comp_KV seq len: {CompKV_len}!")
            start_pos = CompKV_len
        
        if start_pos is None:
            start_pos = 0


        # ===================================== Teacher ==============================================================
         

        
        # 1. TEACHER PASS (No Gradients)
        with torch.no_grad():
            # Process the dropped context to populate its KV cache
            teacher_outputs = self.ar_model(
                input_ids = current_ids,   # combined input_ids + context_ids
                active_kv_caches=active_kv_caches,
                compressed_kv_caches=active_compressed_kv
            )

            # Teacher truth for the future tokens
            teacher_hidden_states = teacher_outputs["hidden_states"]
            teacher_output_logits = teacher_outputs["logits"]
            teacher_active_kv_caches = teacher_outputs["active_kv_caches"]

            # logit and hidden-states for future training
            h_teacher = teacher_hidden_states[:, -int(future_len):, :] # [B, S, H]
            logits_teacher = teacher_output_logits[:, -int(future_len):, :]       # [B, S, V]

            # total uncompressed kvs, from input kV_caches and the new uncompressed context KVs to be compressed
            # TODO: Needs to understand the ordering of Concatings of KVs all the way from begining...
            uncomp_context_kvs = teacher_active_kv_caches[:, :, :int(eff_seq_len), ... ]
    


        # ============================ KV Compression==============================================================
        # Totals KVs on which to conditioned for compression: Compressed_KVs and Uncompressed KVs too!
        if active_compressed_kv is not None:
            cond_context_kvs = torch.cat([active_compressed_kv, uncomp_context_kvs], dim=2)
        else: 
            cond_context_kvs = uncomp_context_kvs


        if latent_states is None:
            # if No latent_states, we assume, it is the first step and so we just put a noisy latent
            # and also make start_step = 0
            start_step = 0
            latent_states = torch.randn(
                B, 
                K_latent_len, 
                self.config.hidden_size, 
                device=self.device, 
                dtype=target_dtype
            )

        # Compress the extracted KVs using the DM tower 
        T_latent_states, T_newcompressed_kv_caches = self.kv_compress(latent_states, cond_context_kvs, start_step, diff_steps, start_pos)
            
        B, T, L, Comp_Seq, _, H, D = T_newcompressed_kv_caches.shape
        
        # Concat new and previous compressed KVs
        if active_compressed_kv is not None:
            T_active_compressed_kv = active_compressed_kv.unsqueeze(1).expand(B, T, L, CompKV_len, 2, H, D)
            T_compressed_kv_caches = torch.cat([T_active_compressed_kv, T_newcompressed_kv_caches], dim=3)
        else:
            T_compressed_kv_caches = T_newcompressed_kv_caches

        # =============================== Student ===========================================================

        device = input_ids.device
        
        # Process the target tokens utilizing the compressed KVs for historical context
        # As the DM_Tower goes with diffusion steps, and we are conditions the LLM 
        # on each diffusion step, we flatten then along step dim and pass the same input_ids
        # T times with each step having its own compressed_kv_caches for different steps of same forward pass.

        # Expand and Flatten input_ids: [B, S] -> [B, T, S] -> [B*T, S]
        _, S = input_ids.shape
        input_ids_flat = input_ids.unsqueeze(1).expand(B, T, S).reshape(B * T, S)

        _, _, _, Comp_L, _, _, _   = T_compressed_kv_caches.shape
        
        # Flatten KV caches: [B, T, L, S, 2, H, D] -> [B*T, L, S, 2, H, D]
        compressed_kv_flat = T_compressed_kv_caches.reshape(B * T, L, Comp_L, 2, H, D)

        
        student_outputs = self.ar_model(
            input_ids = input_ids_flat, 
            active_kv_caches=None,  # As we have compressed all the KV caches
            compressed_kv_caches=compressed_kv_flat
        )
        
        # Shapes: [B*T, S, V] and [B*T, S, H]
        student_logits = student_outputs["logits"] 
        h_student = student_outputs["hidden_states"]

        _, S, V = student_logits.shape
        _, _, H = h_student.shape

        # Shapes: [B, T, S, V] and [B, T, S, H]
        student_logits = student_logits.reshape(B, T, S, V)
        student_h = h_student.reshape(B, T, S, H)

        # ============================= Output =============================================================
        
        train_output = {
            "teacher": {
                "hidden": h_teacher,  #(B, S, V)
                "logits": logits_teacher
            },
            "student":{
                "T_hidden": student_h, #(B, T, S, V)
                "T_logits": student_logits 
            }
        }


        return train_output, T_latent_states




    # ===================================================================================
    # Auto-regressive generation
        
    def generate_text(self, prompt, max_new_tokens=1024, temperature=0.7, top_k=50, top_p=0.9):
        self.eval()
        device = next(self.parameters()).device

        prompt_str = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        
        input_len = list(input_ids.shape)
        
        compressed_kvs = None

        if input_len[1] >= 1024:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                x_factor = 16
                # ---------------------------------------------------------
                # STEP 1: Extract Raw KV Cache of the past
                # ---------------------------------------------------------
                print("\n" + "="*50)
                print(" STEP 1: EXTRACTING & COMPRESSING KV CACHES")
                print("="*50)
                K_latent_len = (input_len[1] + x_factor -1) // x_factor
                
                print("Passing long context through AR model to extract raw KVs...")
                past_outputs = self.ar_model(input_ids)
                uncompressed_kvs = past_outputs["active_kv_caches"]
                
                print(f"🟢 Original KV Cache Shape: {uncompressed_kvs.shape}")

                # ---------------------------------------------------------
                # STEP 2: Compress the KV Caches via DC
                # ---------------------------------------------------------
                print(f"\nPassing raw KVs through DC Model for 16x compression...")
                # if No latent_states, we assume, it is the first step and so we just put a noisy latent
                # and also make start_step = 0
                start_step = 0
                latent_states = torch.randn(
                    1, 
                    K_latent_len, 
                    1152, 
                    device=device, 
                    dtype=torch.bfloat16
                )

                # Compress the extracted KVs using the DM tower 
                T_latent_states, T_compressed_kvs = self.kv_compress(latent_states, uncompressed_kvs, start_step, 2, 0)
            
                compressed_kvs = T_compressed_kvs[:, -1, ...]
                ck_shape = compressed_kvs.shape
                print(f"🔵 Compressed KV Cache Shape : {compressed_kvs.shape}")
                print(f"   -> Successfully compressed to just {ck_shape[2]} tokens of memory space!")


                # refreshes input_ids to just a single bos:
                input_ids = torch.tensor([[2]]).to(device)


        
        # Auto Regressive generations:

        # EOS and Turn-end tokens for Gemma
        eos_token_ids = [self.tokenizer.eos_token_id, 106] 

        print(f"\nUser: {prompt}\nModel: ", end="", flush=True)

        active_kv_caches = None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.ar_model(
                    input_ids, 
                    active_kv_caches=active_kv_caches, 
                    compressed_kv_caches=compressed_kvs
                )
                
            logits = outputs["logits"]
            active_kv_caches = outputs["active_kv_caches"]
            
            # Get the logits for the last token in the sequence
            next_token_logits = logits[:, -1, :]
            
            # --- MULTINOMIAL SAMPLING LOGIC ---
            
            # 1. Apply Temperature
            if temperature != 1.0 and temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                
            # 2. Apply Top-K filtering
            if top_k > 0:
                # Find the top k values and their indices
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                # Mask out everything below the k-th highest value
                indices_to_remove = next_token_logits < top_k_values[:, -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # 3. Apply Top-P (Nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1, dtype=torch.float32), dim=-1)

                # Remove tokens with cumulative probability above the threshold (top_p)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter the mask back to the original indices
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            # 4. Convert to probabilities (use float32 for numerical stability)
            probs = F.softmax(next_token_logits, dim=-1, dtype=torch.float32)

            # 5. Sample the next token
            if temperature == 0.0:
                # Fallback to greedy if temperature is strictly 0
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Multinomial sample
                next_token = torch.multinomial(probs, num_samples=1)
                
            # ----------------------------------

            # Only pass the new token for the next iteration
            input_ids = next_token  
            
            # Decode and print dynamically
            print(self.tokenizer.decode([next_token.item()]), end="", flush=True)

            # Check for End of Sequence
            if next_token.item() in eos_token_ids:
                break
                
        print("\n")







# device = 'cpu'


# ILDC_model = ILDC(device = device)
# ILDC_model.to(torch.bfloat16)

# AR_model = ILDC_model.ar_model

# with torch.no_grad(): 
#     # [B, Seq_len]
#     # input_ids = None
#     input_ids = torch.randint(1, 26400, (1, 16))
    
#     # [B, Layers, Seq_len, KV, heads, Head_dim]
#     active_kv_caches = None
#     # active_kv_caches = torch.randn(1, 26, 1024, 2, 1, 256).to(torch.bfloat16)

#     # compressed_kv_caches = None
#     compressed_kv_caches = torch.randn(1, 26, 15, 2, 1, 256).to(torch.bfloat16)
#     outputs = AR_model(input_ids, active_kv_caches, compressed_kv_caches)
#     print(outputs["hidden_states"].shape)
#     print(outputs["active_kv_caches"].shape)



# DC_model = CompDiTModel().to(torch.bfloat16)

# with torch.no_grad(): 
#     # [B, Layers, Seq_len, KV, heads, Head_dim]
#     kv_caches = torch.randn(3, 26, 1024, 2, 1, 256).to(torch.bfloat16)
#     latent_states = torch.randn(3, 64, 1152).to(torch.bfloat16)
#     A, B = DC_model(latent_states, kv_caches, 1)
#     print(A, B)



# with torch.no_grad():

#     output = ILDC_model(
#         input_ids = torch.randint(1, 26400, (3, 512)).to(device), 
#         context_ids=torch.randint(1, 26400, (3, 2048)).to(device), 
#         active_kv_caches= torch.randn(3, 26, 1024, 2, 1, 256).to(torch.bfloat16).to(device), 
#         active_compressed_kv= torch.randn(3, 26, 768, 2, 1, 256).to(torch.bfloat16).to(device), 
#         latent_states = torch.randn(3, 192, 1152).to(torch.bfloat16).to(device),
#         start_step = 0, 
#         diff_steps=1, 
#         start_pos=768  # as we already have 768 compressed kvs
#     )

#     print(output)


# hidden_fact = "The secret code to bypass the mainframe is 'OMEGA-77'."
# filler_text = "The system logs show normal operational status with minor fluctuations in the thermal array. " * 100
# question = "What's the secret code of the mainframe? Hello"
# prompt = filler_text + hidden_fact + filler_text 

# ILDC_model.generate_text(prompt)
