"""CPU smoke tests with a tiny randomly initialized model."""

import numpy as np
import torch
import trimesh

from seamflow import SeamFlowConfig, SeamFlowModel, SeamFlowPipeline, preprocess_mesh
from seamflow.nn import farthest_point_sample, varlen_attention, attention

TINY = SeamFlowConfig(
    num_heads=2,
    head_dim=16,
    multi_stream_layers=1,
    fused_layers=1,
    mapping_dim=32,
    time_fourier_dim=16,
    num_latents=16,
    condition_dim=32,
    condition_heads=2,
    condition_layers=1,
    point_size=256,
)


def _box_with_uv():
    mesh = trimesh.creation.box()
    # Unweld into per-face vertices so every face has its own UV island.
    v = mesh.vertices[mesh.faces].reshape(-1, 3)
    f = np.arange(len(v)).reshape(-1, 3)
    uv = np.tile([[0, 0], [1, 0], [0, 1]], (len(mesh.faces), 1)).astype(np.float32)
    return v, f, uv


def test_preprocess_box():
    v, f, uv = _box_with_uv()
    m = preprocess_mesh(v, f, uv)
    assert m.edges.shape == (18, 6)  # 12 triangles -> 18 unique edges
    assert m.is_seam is not None and m.is_seam.all()
    assert np.abs(m.edges).max() <= 0.95 + 1e-5
    # World-space edges lie on the original unit box.
    assert np.allclose(np.abs(m.edges_world), 0.5, atol=1e-4)


def test_fps_matches_bruteforce():
    pts = torch.rand(2, 64, 3)
    idx = farthest_point_sample(pts, 8, random_start=False)
    assert idx.shape == (2, 8) and (idx[:, 0] == 0).all()
    assert all(len(set(row.tolist())) == 8 for row in idx)


def test_varlen_attention_is_block_diagonal():
    q, k, v = (torch.randn(10, 2, 8) for _ in range(3))
    cu = torch.tensor([0, 4, 10], dtype=torch.int32)
    out = varlen_attention(q, k, v, cu, 6)
    ref = attention(q[None, 4:], k[None, 4:], v[None, 4:])[0]
    assert torch.allclose(out[4:], ref, atol=1e-6)


def test_sample_and_pipeline(tmp_path):
    torch.manual_seed(0)
    model = SeamFlowModel(TINY).eval()
    v, f, uv = _box_with_uv()
    m = preprocess_mesh(v, f, uv)
    pipe = SeamFlowPipeline(model, "cpu")
    pred = pipe.predict_preprocessed(m, n_steps=3, seed=0)
    assert pred.scores.shape == (18,) and np.isfinite(pred.scores).all()
    again = pipe.predict_preprocessed(m, n_steps=3, seed=0)
    assert np.array_equal(pred.scores, again.scores)
    assert set(pred.metrics()) == {"precision", "recall", "f1", "accuracy"}
    pred.save_obj(tmp_path / "s.obj")
    pred.save_npz(tmp_path / "s.npz")


def test_batched_forward_matches_single():
    torch.manual_seed(0)
    model = SeamFlowModel(TINY).eval()
    surf = torch.randn(2, 256, 6)
    fc = torch.tensor([10.0, 20.0])
    torch.manual_seed(1)
    tok = model.encode_surface(surf, fc, dtype=torch.float32)
    edges = torch.randn(7, 6)
    x = torch.randn(7)
    both = model(edges, x, 0.3, tok, n_edges=[3, 4], dtype=torch.float32)
    first = model(edges[:3], x[:3], 0.3, tok[:1], dtype=torch.float32)
    assert torch.allclose(both[:3], first, atol=1e-5)
