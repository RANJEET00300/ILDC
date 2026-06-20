import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from .helper_fn import RMSNorm, apply_rotary_pos_emb
from .config import ModelConfig


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with Gated-GELU activation.

    Used in both the Auto-Regressive (AR) and Diffusion Compressor (DC) models.
    Maps hidden states to an intermediate size, applies a gating mechanism
    using GELU, and projects back to the hidden size.
    """

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GQMHAttention_AR(nn.Module):
    """
    Grouped Query Multi-Head Attention (GQMHAttention) tailored for ILDC.

    This modified attention mechanism seamlessly concatenates historical
    uncompressed KV caches and compressed latent KV caches (from the DiT tower)
    ahead of the current sequence's KV cache.
    """

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        active_kv: torch.Tensor = None,
        compressed_kv: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Grouped Query Attention.

        Args:
            hidden_states (torch.Tensor): Shape (B, seq_len, hidden_size). Tokens to process.
            position_ids (torch.Tensor): Shape (B, seq_len). Sequence positions for RoPE.
            attention_mask (torch.Tensor, optional): Shape (1, 1, seq_len, total_kv_len).
                                                    Causal mask.
            active_kv (torch.Tensor, optional):
                Shape (B, active_kv_len, 2, num_kv_heads, head_dim).
                Uncompressed KV cache from previous forward passes.

            compressed_kv (torch.Tensor, optional):
                Shape (B, K_hidden_len, 2, num_kv_heads, head_dim).
                Compressed latent KV cache from the Diffusion tower.

        Returns:
            Tuple of:
                - attn_output (torch.Tensor): Shape (B, seq_len, hidden_size).
                                            Projected attention outputs.
                - new_kv (torch.Tensor): Shape (B, seq_len, 2, num_kv_heads, head_dim).
                                            Current tokens' KV to be cached.
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
        new_kv = generated_kv.transpose(1, 3)  # Shape: (B, q_len, 2, num_kv_heads, head_dim)

        # Build full K and V | from preceeding tokens
        full_k, full_v = k, v

        # Active KV-Cache from previous AR tokens: uncompressed + Compressed
        if active_kv is not None:
            active_k, active_v = active_kv[:, :, 0, ...], active_kv[:, :, 1, ...]
            active_k, active_v = active_k.transpose(1, 2), active_v.transpose(1, 2)
            full_k = torch.cat([active_k, full_k], dim=2)
            full_v = torch.cat([active_v, full_v], dim=2)

        # Prepend compressed latents from the diffusion model
        if compressed_kv is not None:
            comp_k, comp_v = compressed_kv[:, :, 0, ...], compressed_kv[:, :, 1, ...]
            comp_k, comp_v = comp_k.transpose(1, 2), comp_v.transpose(1, 2)
            full_k = torch.cat([comp_k, full_k], dim=2)
            full_v = torch.cat([comp_v, full_v], dim=2)

        num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Shape goes from (B, num_kv_heads, total_seq_len, head_dim)
        # -> (B, num_kv_heads, num_queries_per_kv, total_seq_len, head_dim)
        # -> (B, num_heads, total_seq_len, head_dim)
        full_k = (
            full_k.unsqueeze(2)
            .expand(-1, -1, num_queries_per_kv, -1, -1)
            .reshape(B, self.num_heads, full_k.size(2), self.head_dim)
        )

        full_v = (
            full_v.unsqueeze(2)
            .expand(-1, -1, num_queries_per_kv, -1, -1)
            .reshape(B, self.num_heads, full_v.size(2), self.head_dim)
        )

        attn_output = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.o_proj(attn_output), new_kv


class LLM_block(nn.Module):
    """
    Standard Auto-Regressive Transformer Block.

    Contains pre-norm GQMHAttention and pre-norm MLP layers.
    Routes active and compressed KV caches specifically to the attention module.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = GQMHAttention_AR(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states, position_ids, attention_mask=None, active_kv=None, compressed_kv=None
    ):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x, new_kv = self.self_attn(x, position_ids, attention_mask, active_kv, compressed_kv)

        x = self.post_attention_layernorm(x)
        hidden_states = residual + x

        residual = hidden_states
        x = self.pre_feedforward_layernorm(hidden_states)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        hidden_states = residual + x

        return hidden_states, new_kv


class CausalLM(nn.Module):
    """
    Auto-Regressive (AR) Language Model.

    This acts as both the Teacher (generating uncompressed context targets)
    and the Student (generating future tokens conditioned on compressed latents).
    """

    def __init__(self):
        super().__init__()
        self.config = ModelConfig()

        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size, padding_idx=0
        )
        self.gen_layers = nn.ModuleList(
            [LLM_block(self.config, i) for i in range(self.config.num_hidden_layers)]
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        active_kv_caches: torch.Tensor = None,
        compressed_kv_caches: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass for the Causal LM.

        Args:
            input_ids (torch.Tensor): Shape (B, seq_len). Input token IDs.
            active_kv_caches (torch.Tensor, optional): Shape
                (B, num_layer, active_kv_len, 2, num_kv_heads, head_dim).
                Uncompressed KV cache from previous steps.
            compressed_kv_caches (torch.Tensor, optional): Shape
                (B, num_layer, K_hidden_state, 2, num_kv_heads, head_dim).
                Compressed KV cache from the DiT Tower.

        Returns:
            dict containing:
                - 'logits': Output predictions over the vocabulary.
                - 'hidden_states': Final transformer hidden states.
                - 'active_kv_caches': The newly constructed active KV cache.
        """

        B, seq_len = input_ids.shape

        past_len = active_kv_caches[0].shape[1] if active_kv_caches is not None else 0
        comp_len = compressed_kv_caches[0].shape[1] if compressed_kv_caches is not None else 0

        # Calculate start position based on X_factor compression if compressed KVs exist
        if comp_len > 0:
            start_pos = (comp_len * self.config.X_factor) + past_len
        else:
            start_pos = past_len

        position_ids = torch.arange(
            start_pos, start_pos + seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)

        # Standard Gemma scaling
        hidden_states = hidden_states * math.sqrt(self.config.hidden_size)

        causal_mask = None
        if seq_len > 1:
            full_k_len = comp_len + past_len + seq_len
            pad_len = comp_len + past_len
            min_dtype = torch.finfo(hidden_states.dtype).min
            causal_mask = torch.full(
                (seq_len, seq_len),
                fill_value=min_dtype,
                device=input_ids.device,
                dtype=hidden_states.dtype,
            )
            causal_mask.triu_(diagonal=1)

            if pad_len > 0:
                past_mask = torch.zeros(
                    (seq_len, pad_len), device=input_ids.device, dtype=hidden_states.dtype
                )
                causal_mask = torch.cat([past_mask, causal_mask], dim=1)

            causal_mask = causal_mask.view(1, 1, seq_len, full_k_len)

        total_active_len = past_len + seq_len
        new_active_kv_caches = torch.empty(
            (
                B,
                self.config.num_hidden_layers,
                total_active_len,
                2,
                self.config.num_key_value_heads,
                self.config.head_dim,
            ),
            device=input_ids.device,
            dtype=hidden_states.dtype,
        )

        for i, layer in enumerate(self.gen_layers):
            layer_active_kv = active_kv_caches[:, i, ...] if active_kv_caches is not None else None
            layer_comp_kv = (
                compressed_kv_caches[:, i, ...] if compressed_kv_caches is not None else None
            )

            hidden_states, new_kv = layer(
                hidden_states,
                position_ids,
                attention_mask=causal_mask,
                active_kv=layer_active_kv,
                compressed_kv=layer_comp_kv,
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
            "active_kv_caches": new_active_kv_caches,
        }
