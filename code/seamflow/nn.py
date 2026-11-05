"""Pure-PyTorch building blocks used by SeamFlow.

Norms run in float32 and cast back to the input dtype. Linear layers cast
their (float32) weights to the activation dtype, so a float32 model runs in
bfloat16 when fed bfloat16 activations.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


class Linear(nn.Linear):
    """``nn.Linear`` that computes in the input's dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


def linear(in_features: int, out_features: int, bias: bool = False) -> Linear:
    return Linear(in_features, out_features, bias=bias)


def rms_norm(
    x: torch.Tensor, scale: torch.Tensor | float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    """RMSNorm over the last dim; ``scale`` broadcasts against ``x``."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    if isinstance(scale, torch.Tensor):
        y = y * scale.float()
    elif scale != 1.0:
        y = y * scale
    return y.to(x.dtype)


def add_rms_norm(
    residual: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor | float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``residual + rms_norm(x) * scale``."""
    return residual + rms_norm(x, scale, eps)


def qk_rmsnorm(qk: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Per-head RMSNorm. ``qk``: ``[..., nh, hd]``, ``scales``: ``[nh]``."""
    return rms_norm(qk, scales[:, None])


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """``x[..., :D] * silu(x[..., D:])``."""
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.scale, self.eps)


class FixedFourier(nn.Module):
    """Scalar -> ``num_channels`` cosine features with linearly spaced freqs."""

    def __init__(self, num_channels: int, freq_start=0.01, freq_end=10.0):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.linspace(freq_start, freq_end, num_channels),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float().outer(self.freqs.float())
        return (y.cos() * math.sqrt(2)).to(x.dtype)


class FourierEmbedder(nn.Module):
    """NeRF-style sin/cos embedding with log-spaced frequencies.

    ``[..., input_dim] -> [..., input_dim * (2 * num_freqs + 1)]``.
    """

    def __init__(self, num_freqs: int = 8, input_dim: int = 3):
        super().__init__()
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_freqs, dtype=torch.float32),
            persistent=False,
        )
        self.out_dim = input_dim * (2 * num_freqs + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((x, embed.sin(), embed.cos()), dim=-1)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense attention on ``[B, L, nh, hd]`` tensors."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
    )
    return out.transpose(1, 2)


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Block-diagonal attention over packed ``[S, nh, hd]`` sequences.

    Uses ``flash_attn`` when it is installed and the inputs are half precision
    on CUDA; otherwise runs SDPA per sequence.
    """
    if (
        flash_attn_varlen_func is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    ):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=scale,
        )

    bounds = cu_seqlens.tolist()
    outs = [
        attention(q[None, s:e], k[None, s:e], v[None, s:e], scale)[0]
        for s, e in zip(bounds[:-1], bounds[1:])
    ]
    return torch.cat(outs, dim=0)


def farthest_point_sample(
    points: torch.Tensor,
    num_samples: int,
    random_start: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Farthest point sampling. ``points``: ``[B, N, 3]`` -> indices ``[B, K]``."""
    B, N, _ = points.shape
    pts = points.float()
    batch = torch.arange(B, device=pts.device)
    if random_start:
        cur = torch.randint(N, (B,), device=pts.device, generator=generator)
    else:
        cur = torch.zeros(B, dtype=torch.long, device=pts.device)

    idx = torch.empty(B, num_samples, dtype=torch.long, device=pts.device)
    min_dist = torch.full((B, N), float("inf"), device=pts.device)
    for i in range(num_samples):
        idx[:, i] = cur
        d = (pts - pts[batch, cur][:, None]).square().sum(-1)
        min_dist = torch.minimum(min_dist, d)
        cur = min_dist.argmax(dim=-1)
    return idx
