import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


from .helper_fn import RMSNorm, apply_rotary_pos_emb, get_timestep_embedding
from .config import ModelConfig


# ===========================================================================================================

# ===========================================================================================================


# MLP: will be used in both AR and DC   
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))




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
        self.config = ModelConfig()
        
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


