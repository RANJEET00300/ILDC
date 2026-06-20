import torch
import torch.nn as nn
import math


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


def apply_rotary_pos_emb(q, k, position_ids, rope_theta, head_dim):
    # Dynamically determine the reference tensor to pull the device and dtype
    ref_tensor = q if q is not None else k

    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=ref_tensor.device) / head_dim)
    )
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


def get_timestep_embedding(B, timestep, embedding_dim, device):
    timesteps = torch.full((B,), timestep * 100, dtype=torch.long, device=device)
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb
