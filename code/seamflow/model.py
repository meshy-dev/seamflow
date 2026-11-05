"""SeamFlow: rectified-flow seam prediction with an MMDiT backbone.

Each mesh edge is one token carrying Fourier(v0), Fourier(v1) and the current
flow state ``x_t``. A point cloud sampled from the surface is encoded into
``num_latents`` (+1 face-count) condition tokens. Edge and surface tokens are
processed jointly by an MMDiT (multi-stream layers with modality-specific MLPs,
then fused layers), and the model predicts one velocity scalar per edge.
Integrating the flow from Gaussian noise (t=0) to t=1 gives a per-edge score;
positive means seam.

Module attribute names match the released checkpoints, so do not rename them.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from seamflow.nn import (
    FixedFourier,
    FourierEmbedder,
    RMSNorm,
    add_rms_norm,
    attention,
    farthest_point_sample,
    linear,
    qk_rmsnorm,
    rms_norm,
    swiglu,
    varlen_attention,
)

EDGE, SURFACE = 0, 1
TOKEN_TYPES = ("EDGE", "SURFACE")


@dataclass
class SeamFlowConfig:
    # MMDiT backbone
    num_heads: int = 16
    head_dim: int = 64
    mlp_mult: float = 4.0
    multi_stream_layers: int = 4
    fused_layers: int = 28
    mapping_dim: int = 768
    time_fourier_dim: int = 256
    fourier_freqs: int = 8
    # Surface encoder
    num_latents: int = 1024
    condition_dim: int = 1024
    condition_heads: int = 8
    condition_layers: int = 8
    point_feats: int = 3
    face_count_control: bool = True
    # Number of surface points fed to the encoder
    point_size: int = 40960

    @property
    def model_dim(self) -> int:
        return self.num_heads * self.head_dim

    @classmethod
    def from_json(cls, path: str | Path) -> SeamFlowConfig:
        return cls(**json.loads(Path(path).read_text()))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def _round_up(x: int, to: int) -> int:
    return (x + to - 1) // to * to


# ---------------------------------------------------------------------------
# Surface encoder
# ---------------------------------------------------------------------------


class SelfAttentionLayer(nn.Module):
    """QK-RMSNorm self-attention + SwiGLU MLP, both with RMS-gated residuals."""

    def __init__(self, nh: int, hd: int, ff_hidden: int):
        super().__init__()
        D = nh * hd
        self.q_scales = nn.Parameter(torch.ones(nh))
        self.to_qkv = linear(D, D * 3)
        self.qk_bias = nn.Parameter(torch.zeros(2, nh, hd))
        self.attn_to_out = linear(D, D)
        self.attn_gate = nn.Parameter(torch.zeros(D))
        self.to_ff = linear(D, ff_hidden * 2)
        self.ff_to_out = linear(ff_hidden, D)
        self.ff_gate = nn.Parameter(torch.zeros(D))
        self.nh, self.hd = nh, hd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).unflatten(-1, (3, self.nh, self.hd))
        qk, v = qkv[..., :2, :, :], qkv[..., 2, :, :]
        qk = qk_rmsnorm(qk + self.qk_bias.to(qk.dtype), self.q_scales)
        q, k = qk.unbind(-3)
        c = attention(q, k, v).flatten(-2)
        x = add_rms_norm(x, self.attn_to_out(c), self.attn_gate)
        ff = self.ff_to_out(swiglu(self.to_ff(x)))
        return add_rms_norm(x, ff, self.ff_gate)


class CrossAttentionLayer(nn.Module):
    """Cross-attention from ``x`` to ``y`` + SwiGLU MLP."""

    def __init__(self, nh: int, hd: int, ff_hidden: int):
        super().__init__()
        D = nh * hd
        self.q_scales = nn.Parameter(torch.ones(nh))
        self.to_q = linear(D, D)
        self.to_kv = linear(D, D * 2)
        self.attn_to_out = linear(D, D)
        self.attn_gate = nn.Parameter(torch.zeros(D))
        self.to_ff = linear(D, ff_hidden * 2)
        self.ff_to_out = linear(ff_hidden, D)
        self.ff_gate = nn.Parameter(torch.zeros(D))
        self.nh, self.hd = nh, hd

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q = qk_rmsnorm(self.to_q(x).unflatten(-1, (self.nh, self.hd)), self.q_scales)
        kv = self.to_kv(y).unflatten(-1, (2, self.nh, self.hd))
        k = qk_rmsnorm(kv[..., 0, :, :], self.q_scales)
        v = kv[..., 1, :, :]
        c = attention(q, k, v).flatten(-2)
        x = add_rms_norm(x, self.attn_to_out(c), self.attn_gate)
        ff = self.ff_to_out(swiglu(self.to_ff(x)))
        return add_rms_norm(x, ff, self.ff_gate)


class SurfaceEncoder(nn.Module):
    """Point cloud ``[B, N, 6]`` (xyz + normal) -> ``[B, 1 + L, D]`` tokens.

    FPS picks ``num_latents`` query points, which cross-attend to the full
    cloud and then pass through a self-attention stack. A face-count token is
    prepended when ``face_count_control`` is on.
    """

    def __init__(self, cfg: SeamFlowConfig):
        super().__init__()
        self.num_latents = cfg.num_latents
        self.point_feats = cfg.point_feats
        cd = cfg.condition_dim
        hd = cd // cfg.condition_heads

        self.fourier_embedder = FourierEmbedder(cfg.fourier_freqs, input_dim=3)
        self.cond_proj = linear(
            self.fourier_embedder.out_dim + cfg.point_feats, cd, bias=True
        )
        self.pc_feats_cross_attn = CrossAttentionLayer(cfg.condition_heads, hd, cd * 3)
        self.pc_feats_transformer = nn.ModuleList(
            SelfAttentionLayer(cfg.condition_heads, hd, cd * 3)
            for _ in range(cfg.condition_layers)
        )
        self.cond_post_norm = RMSNorm(cd)
        self.cond_dim_proj = (
            linear(cd, cfg.model_dim) if cd != cfg.model_dim else None
        )
        self.face_count_proj = (
            nn.Sequential(linear(1, cd, bias=True), nn.SiLU(), linear(cd, cd, bias=True))
            if cfg.face_count_control
            else None
        )

    def forward(
        self,
        surface: torch.Tensor,
        face_count: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        pc, feats = surface[..., :3], surface[..., 3:]
        B = pc.shape[0]
        idx = farthest_point_sample(pc, self.num_latents, generator=generator)
        batch = torch.arange(B, device=pc.device)[:, None]

        query = self.fourier_embedder(pc[batch, idx])
        data = self.fourier_embedder(pc)
        if self.point_feats > 0:
            query = torch.cat([query, feats[batch, idx]], dim=-1)
            data = torch.cat([data, feats], dim=-1)
        if dtype is not None:
            query, data = query.to(dtype), data.to(dtype)

        cond = self.pc_feats_cross_attn(self.cond_proj(query), self.cond_proj(data))
        for layer in self.pc_feats_transformer:
            cond = layer(cond)
        cond = self.cond_post_norm(cond)

        if self.face_count_proj is not None and face_count is not None:
            fc = torch.log(face_count.float() + 1.0)[:, None] / math.log(1e5)
            if dtype is not None:
                fc = fc.to(dtype)
            cond = torch.cat([self.face_count_proj(fc)[:, None], cond], dim=1)

        if self.cond_dim_proj is not None:
            cond = self.cond_dim_proj(cond)
        return cond


# ---------------------------------------------------------------------------
# MMDiT backbone
# ---------------------------------------------------------------------------


class EdgeEmbedder(nn.Module):
    """``(v0, v1, x_t)`` -> ``model_dim``."""

    def __init__(self, model_dim: int, num_freqs: int):
        super().__init__()
        self.fourier = FourierEmbedder(num_freqs, input_dim=3)
        self.proj = linear(2 * self.fourier.out_dim + 1, model_dim, bias=True)

    def forward(self, edges: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        v0, v1 = edges[..., :3], edges[..., 3:]
        return self.proj(torch.cat([self.fourier(v0), self.fourier(v1), x_t], dim=-1))


class MultiTaskFeedForward(nn.Module):
    """AdaLN-modulated SwiGLU MLP with separate weights per token type."""

    def __init__(self, dim: int, hidden: int, cond_dim: int, n_tasks: int):
        super().__init__()
        self.cond_to_scales = linear(cond_dim, dim * n_tasks, bias=True)
        self.cond_to_gates = linear(cond_dim, dim * n_tasks, bias=True)
        self.in_proj = linear(dim, hidden * 2 * n_tasks)
        self.out_proj = linear(hidden, dim * n_tasks)
        self.n_tasks = n_tasks

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, token_types: torch.Tensor
    ) -> torch.Tensor:
        x_norm = rms_norm(x)
        order = torch.argsort(token_types)
        inverse = torch.argsort(order)
        x_sorted, cond_sorted, types_sorted = x_norm[order], cond[order], token_types[order]

        chunks = [
            p.chunk(self.n_tasks)
            for p in (
                self.cond_to_scales.weight,
                self.cond_to_scales.bias,
                self.cond_to_gates.weight,
                self.cond_to_gates.bias,
                self.in_proj.weight,
                self.out_proj.weight,
            )
        ]
        outs, gates = [], []
        start = 0
        for tid in range(self.n_tasks):
            count = int((types_sorted == tid).sum().item())
            if count == 0:
                continue
            s_w, s_b, g_w, g_b, in_w, out_w = (c[tid].type_as(x) for c in chunks)
            xt = x_sorted[start : start + count]
            ct = cond_sorted[start : start + count]
            h = xt * F.linear(ct, s_w, s_b)
            h = F.linear(swiglu(F.linear(h, in_w)), out_w)
            outs.append(h)
            gates.append(F.linear(ct, g_w, g_b))
            start += count

        out = rms_norm(torch.cat(outs)) * torch.cat(gates)
        return x + out[inverse]


class AttentionPreBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, cond_dim: int):
        super().__init__()
        attn_dim = n_heads * head_dim
        self.cond_to_scales = linear(cond_dim, dim, bias=True)
        self.to_qk = linear(dim, attn_dim * 2, bias=True)
        self.to_v = linear(dim, attn_dim)
        self.q_scales = nn.Parameter(torch.ones(n_heads))
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x_norm = F.rms_norm(x, x.shape[-1:]) * self.cond_to_scales(cond)
        qk = self.to_qk(x_norm).unflatten(-1, (2, self.n_heads, -1))
        qk = qk_rmsnorm(qk, self.q_scales)
        v = self.to_v(x_norm).unflatten(-1, (self.n_heads, -1))
        return qk, v


class AttentionPostBlock(nn.Module):
    def __init__(self, n_heads: int, head_dim: int, dim: int, cond_dim: int):
        super().__init__()
        self.cond_to_gates = linear(cond_dim, dim, bias=True)
        self.attn_out = linear(n_heads * head_dim, dim)

    def forward(self, x, attn_out, cond):
        out = self.attn_out(attn_out)
        return x + F.rms_norm(out, out.shape[-1:]) * self.cond_to_gates(cond)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, cond_dim: int):
        super().__init__()
        self.pre = AttentionPreBlock(dim, n_heads, head_dim, cond_dim)
        self.post = AttentionPostBlock(n_heads, head_dim, dim, cond_dim)

    def forward(self, x, cond, attn_args):
        qk, v = self.pre(x, cond)
        out = varlen_attention(qk[:, 0], qk[:, 1], v, *attn_args).flatten(-2)
        return self.post(x, out, cond)


class MultiStreamLayer(nn.Module):
    """Joint attention + modality-specific MLP."""

    def __init__(self, dim, n_heads, head_dim, mlp_dim, cond_dim, n_tasks):
        super().__init__()
        self.attn = AttentionBlock(dim, n_heads, head_dim, cond_dim)
        self.mlp = MultiTaskFeedForward(dim, mlp_dim, cond_dim, n_tasks)

    def forward(self, x, cond, token_types, attn_args):
        x = self.attn(x, cond, attn_args)
        return self.mlp(x, cond, token_types)


class FusedLayer(nn.Module):
    """Parallel attention + MLP shared by all tokens."""

    def __init__(self, dim: int, n_heads: int, head_dim: int, mlp_dim: int):
        super().__init__()
        attn_dim = n_heads * head_dim
        self.scales = nn.Parameter(torch.ones(dim))
        self.qk_scales = nn.Parameter(torch.ones(n_heads))
        self.to_qk = linear(dim, attn_dim * 2, bias=True)
        self.to_v = linear(dim, attn_dim)
        self.to_mlp = linear(dim, mlp_dim * 2)
        self.mlp_out = linear(mlp_dim, dim)
        self.attn_out = linear(attn_dim, dim)
        self.gates = nn.Parameter(torch.zeros(dim))
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, attn_args) -> torch.Tensor:
        x_norm = rms_norm(x, self.scales)
        qk = qk_rmsnorm(
            self.to_qk(x_norm).unflatten(-1, (2, self.n_heads, -1)), self.qk_scales
        )
        v = self.to_v(x_norm).unflatten(-1, (self.n_heads, -1))
        attn = varlen_attention(qk[:, 0], qk[:, 1], v, *attn_args).flatten(-2)
        out = self.mlp_out(swiglu(self.to_mlp(x_norm))) + self.attn_out(attn)
        return add_rms_norm(x, out, self.gates)


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.out_proj = linear(dim, out_dim, bias=True)
        self.cond_to_scales = linear(cond_dim, dim, bias=True)

    def forward(self, x, cond):
        return self.out_proj(F.rms_norm(x, x.shape[-1:]) * self.cond_to_scales(cond))


class EdgeFlowTransformer(nn.Module):
    """MMDiT over packed ``[edges_b, surface_b]`` sequences."""

    def __init__(self, cfg: SeamFlowConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.model_dim
        mlp_dim = _round_up(int(D * cfg.mlp_mult), 256)

        self.to_cond = nn.Sequential(
            FixedFourier(cfg.time_fourier_dim),
            linear(cfg.time_fourier_dim, cfg.mapping_dim),
            nn.SiLU(),
        )
        self.edge_embedder = EdgeEmbedder(D, cfg.fourier_freqs)
        # The surface encoder already outputs model_dim tokens.
        self.surface_in_proj = nn.Identity()
        self.multi_blocks = nn.ModuleList(
            MultiStreamLayer(
                D, cfg.num_heads, cfg.head_dim, mlp_dim, cfg.mapping_dim, len(TOKEN_TYPES)
            )
            for _ in range(cfg.multi_stream_layers)
        )
        self.fused_blocks = nn.ModuleList(
            FusedLayer(D, cfg.num_heads, cfg.head_dim, mlp_dim)
            for _ in range(cfg.fused_layers)
        )
        self.final = FinalLayer(D, 1, cfg.mapping_dim)

    def forward(
        self,
        edges: torch.Tensor,  # [N, 6]
        x_t: torch.Tensor,  # [N, 1]
        timestep: torch.Tensor,  # [B]
        n_edges: list[int],  # per-sample edge counts, sum == N
        surface_tokens: torch.Tensor,  # [B, L, D]
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        device = edges.device
        B, L = surface_tokens.shape[:2]
        dtype = dtype or surface_tokens.dtype

        cond_b = self.to_cond(timestep).to(dtype)
        edge_tok = self.edge_embedder(edges, x_t).to(dtype)
        surf_tok = self.surface_in_proj(surface_tokens.to(dtype))

        tokens, conds, types, seqlens = [], [], [], []
        start = 0
        for b, n in enumerate(n_edges):
            tokens += [edge_tok[start : start + n], surf_tok[b]]
            types += [
                torch.full((n,), EDGE, dtype=torch.int8, device=device),
                torch.full((L,), SURFACE, dtype=torch.int8, device=device),
            ]
            conds.append(cond_b[b : b + 1].expand(n + L, -1))
            seqlens.append(n + L)
            start += n

        x = torch.cat(tokens)
        cond = torch.cat(conds)
        token_types = torch.cat(types)
        cu_seqlens = torch.tensor([0] + seqlens, device=device).cumsum(0).to(torch.int32)
        attn_args = (cu_seqlens, max(seqlens), 1.0 / self.cfg.head_dim)

        for layer in self.multi_blocks:
            x = layer(x, cond, token_types, attn_args)
        for layer in self.fused_blocks:
            x = layer(x, attn_args)
        return self.final(x, cond)[token_types == EDGE]


# ---------------------------------------------------------------------------
# Full model + sampler
# ---------------------------------------------------------------------------


class SeamFlowModel(nn.Module):
    def __init__(self, cfg: SeamFlowConfig | None = None):
        super().__init__()
        self.config = cfg or SeamFlowConfig()
        self.surface_encoder = SurfaceEncoder(self.config)
        self.transformer = EdgeFlowTransformer(self.config)

    @classmethod
    def from_pretrained(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> SeamFlowModel:
        """Load from a directory with ``config.json`` + ``model.safetensors``."""
        from safetensors.torch import load_file

        path = Path(path)
        cfg_path = path / "config.json"
        cfg = SeamFlowConfig.from_json(cfg_path) if cfg_path.exists() else None
        model = cls(cfg)
        state = load_file(str(path / "model.safetensors"))
        model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
        return model.eval().to(device)

    def encode_surface(
        self,
        surface: torch.Tensor,
        face_count: torch.Tensor,
        dtype: torch.dtype | None = torch.bfloat16,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.surface_encoder(surface, face_count, dtype, generator)

    def forward(
        self,
        edges: torch.Tensor,  # [N, 6]
        x_t: torch.Tensor,  # [N]
        t: float,
        surface_tokens: torch.Tensor,  # [B, L, D]
        n_edges: list[int] | None = None,
        dtype: torch.dtype | None = torch.bfloat16,
    ) -> torch.Tensor:
        """Velocity ``[N]`` at flow time ``t`` in ``[0, 1)``."""
        n_edges = n_edges or [edges.shape[0]]
        # The timestep shares the surface tokens' dtype, as in training.
        timestep = surface_tokens.new_full((surface_tokens.shape[0],), float(t))
        v = self.transformer(edges, x_t[:, None], timestep, n_edges, surface_tokens, dtype)
        return v[:, 0].float()

    @torch.no_grad()
    def sample(
        self,
        edges: torch.Tensor,  # [E, 6]
        surface: torch.Tensor,  # [1, P, 6]
        face_count: torch.Tensor,  # [1]
        n_steps: int = 50,
        method: Literal["euler", "ddpm"] = "euler",
        dtype: torch.dtype | None = torch.bfloat16,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate the flow from noise (t=0) to t=1. Returns scores ``[E]``.

        Edges with a positive score are predicted seams.
        """
        surface_tokens = self.encode_surface(surface, face_count, dtype, generator)
        x = torch.randn(
            edges.shape[0], device=edges.device, dtype=torch.float32, generator=generator
        )
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = step * dt
            v = self(edges, x, t, surface_tokens, dtype=dtype)
            if method == "euler" or step == 0:
                x = x + dt * v
            elif method == "ddpm":
                x0 = x + (1 - t) * v
                t_next = t + dt
                sigma = math.sqrt(((1 - t) / t) ** 2 - ((1 - t_next) / t_next) ** 2)
                mean = (((1 - t) * t_next - t * (1 - t_next)) * x0 + (1 - t_next) * x) / (
                    1 - t
                )
                noise = torch.randn(
                    x.shape, device=x.device, dtype=x.dtype, generator=generator
                )
                x = (mean / t_next + sigma * noise) * t_next
            else:
                raise ValueError(f"unknown sampling method: {method}")
        return x
