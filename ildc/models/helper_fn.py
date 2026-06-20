import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input tensor along the last dimension using the root mean square,
    providing a learnable scale parameter initialized to zeros. Follows the Gemma
    convention where the final scale is (1.0 + weight).

    Args:
        dim (int): The feature dimension to normalize.
        eps (float): A small value added to the denominator for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Normalized tensor of the same shape and dtype as input.
        """
        # Force float32 for math stability, then strictly cast back to input dtype
        x_f32 = x.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(x.dtype)) * (1.0 + self.weight)


def apply_rotary_pos_emb(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    rope_theta: float,
    head_dim: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Applies Rotary Position Embeddings (RoPE) to query and key tensors (On-the-fly compute).

    This function explicitly maps 1D position IDs to trigonometric frequencies
    and applies them dynamically. This computes inverse frequencies on every call.

    Args:
        q (torch.Tensor | None): Query tensor of shape [batch, num_heads, seq_len, head_dim].
        k (torch.Tensor | None): Key tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        position_ids (torch.Tensor): Tensor of shape [batch, seq_len] containing sequence positions.
        rope_theta (float): Base for the frequency exponent (e.g., 10000.0 or 1000000.0).
        head_dim (int): Dimensionality of the attention heads.

    Returns:
        Tuple containing the rotated query and key tensors.
    """
    ref_tensor = q if q is not None else k
    if ref_tensor is None:
        return None, None

    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=ref_tensor.device) / head_dim)
    )
    t = position_ids[0].to(torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos().to(ref_tensor.dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(ref_tensor.dtype).unsqueeze(0).unsqueeze(0)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None

    return q_embed, k_embed


class CachedRotaryPosEmb(nn.Module):
    """
    Optimized Rotary Position Embedding (RoPE) Module.

    Precomputes and caches the `inv_freq` tensor upon initialization to save
    computation cycles during the forward pass. This is especially useful for
    standard generation, while strided position IDs can still dynamically select
    their respective frequency outer products.
    """

    def __init__(self, head_dim: int, rope_theta: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        # Precompute the inverse frequencies and register as buffer (moves with model device)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, q: Optional[torch.Tensor], k: Optional[torch.Tensor], position_ids: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        ref_tensor = q if q is not None else k
        if ref_tensor is None:
            return None, None

        # Use cached inv_freq
        t = position_ids[0].to(torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().to(ref_tensor.dtype).unsqueeze(0).unsqueeze(0)
        sin = emb.sin().to(ref_tensor.dtype).unsqueeze(0).unsqueeze(0)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
        k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None

        return q_embed, k_embed


def get_timestep_embedding(
    batch_size: int, timestep: int, embedding_dim: int, device: torch.device
) -> torch.Tensor:
    """
    Generates standard sinusoidal timestep embeddings for the Diffusion model.

    Transforms a scalar timestep into a high-dimensional continuous embedding
    using sine and cosine frequencies.

    Args:
        batch_size (int): The batch size to expand the embedding for.
        timestep (int): The current diffusion step.
        embedding_dim (int): The target dimensionality of the embedding.
        device (torch.device): The device to place the tensor on.

    Returns:
        torch.Tensor: Timestep embedding of shape [batch_size, embedding_dim].
    """
    timesteps = torch.full((batch_size,), timestep * 100, dtype=torch.long, device=device)
    half_dim = embedding_dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)

    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb
