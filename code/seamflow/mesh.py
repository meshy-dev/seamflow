"""Mesh loading and preprocessing.

Mirrors the training data pipeline:

1. Normalize the mesh bbox to ``[-0.95, 0.95]``.
2. Weld vertices on a 2^16 grid and extract unique undirected edges. Seam
   labels (if the mesh has UVs) are edges that are boundaries or carry more
   than one UV pair.
3. Apply the canonical eval transform (a 180 degree turn about Y) and
   re-normalize to ``[-0.95, 0.95]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh

XYZ_RESOLUTION = 2**16
UV_RESOLUTION = 2**16
# Canonical eval transform; rotation angles are (rng * 2 - 1) * pi.
_EVAL_RNG = np.array([0.5, 0.0, 0.5, 0.5, 0.5, 0.5])


@dataclass
class MeshInput:
    """A preprocessed mesh, ready for the model."""

    edges: np.ndarray  # [E, 6] float32, endpoints in model space
    vertices: np.ndarray  # [V, 3] float32, mesh vertices in model space
    faces: np.ndarray  # [F, 3] int64
    face_count: int
    # Edge endpoints in the input mesh's coordinates, for export.
    edges_world: np.ndarray  # [E, 6] float32
    # Seam labels from the input UVs, or None if the mesh has no usable UVs.
    is_seam: np.ndarray | None  # [E] bool


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load all geometry of a scene file into one triangle mesh.

    Returns ``(vertices [V, 3], faces [F, 3], uvs [V, 2] or None)``. Node
    transforms are applied. ``uvs`` is None unless every sub-mesh has UVs.
    """
    scene = trimesh.load(path, force="scene")
    vertices, faces, uvs = [], [], []
    offset = 0
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        mesh = trimesh.Trimesh(geom.vertices, geom.faces, process=False, validate=False)
        mesh.apply_transform(transform)
        vertices.append(np.asarray(mesh.vertices))
        faces.append(np.asarray(mesh.faces) + offset)
        uv = getattr(geom.visual, "uv", None)
        uvs.append(None if uv is None else np.asarray(uv))
        offset += len(mesh.vertices)

    if not vertices:
        raise ValueError(f"no triangle geometry in {path}")
    has_uv = all(u is not None and len(u) == len(v) for u, v in zip(uvs, vertices))
    return (
        np.concatenate(vertices),
        np.concatenate(faces),
        np.concatenate(uvs)[:, :2] if has_uv else None,
    )


def normalize_vertices(vertices: np.ndarray, scale: float = 1.0):
    """Center the bbox and scale its longest side to ``2 * scale``.

    Returns ``(normalized, center, factor)`` with
    ``normalized = (vertices - center) * factor``.
    """
    bbmin, bbmax = vertices.min(axis=0), vertices.max(axis=0)
    center = (bbmin + bbmax) / 2
    factor = 2 * scale / (bbmax - bbmin).max()
    return (vertices - center) * factor, center, factor


def rng2trs(rng: np.ndarray) -> np.ndarray:
    """Map 6 numbers in [0, 1] to a 4x4 rotate-then-scale matrix."""
    rx, ry, rz = (rng[:3] * 2 - 1) * np.pi
    s = rng[3:] * 0.5 + 0.75
    c, n = np.cos, np.sin
    rot_x = np.array([[1, 0, 0, 0], [0, c(rx), n(rx), 0], [0, -n(rx), c(rx), 0], [0, 0, 0, 1]])
    rot_y = np.array([[c(ry), 0, n(ry), 0], [0, 1, 0, 0], [-n(ry), 0, c(ry), 0], [0, 0, 0, 1]])
    rot_z = np.array([[c(rz), n(rz), 0, 0], [-n(rz), c(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return rot_z @ rot_y @ rot_x @ np.diag([s[0], s[1], s[2], 1.0])


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points, np.ones((len(points), 1), points.dtype)], axis=-1)
    return (homo @ transform.T)[:, :3]


def transform_and_normalize(vertices: np.ndarray, transform: np.ndarray, scale: float):
    """Apply ``transform``, re-normalize the bbox, and return the combined matrix."""
    out = apply_transform(vertices, transform)
    out, center, factor = normalize_vertices(out, scale)
    combined = transform.copy()
    combined[:3, 3] -= center * factor
    combined[:3, :3] *= factor
    return out, combined


def weld_vertices(vertices: np.ndarray, faces: np.ndarray, resolution: int = XYZ_RESOLUTION):
    """Merge vertices that fall into the same cell of a grid over ``[-1, 1]^3``.

    Returns ``(welded_vertices, quantized_faces)``; welded vertices are cell
    centers sorted by quantized position.
    """
    q = np.floor((vertices + 1) * 0.5 * resolution).astype(np.int32)
    uq, old_to_new = np.unique(q, axis=0, return_inverse=True)
    welded = ((uq + 0.5) / resolution * 2 - 1).astype(np.float32)
    return welded, old_to_new.reshape(-1).astype(np.int32)[faces]


def extract_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    remove_degenerate: bool = True,
):
    """Unique undirected edges of the welded mesh, with UV-seam flags.

    Returns ``(endpoints [2E, 3], is_boundary [E], has_multi_uv [E])``.
    """
    welded, q_faces = weld_vertices(vertices, faces)
    q_uv = np.floor(uv * UV_RESOLUTION).astype(np.int32)

    order = np.argsort(q_faces, axis=-1)
    faces_sorted = np.take_along_axis(faces, order, axis=-1)
    q_faces = np.take_along_axis(q_faces, order, axis=-1)
    if remove_degenerate:
        keep = (q_faces[:, 0] < q_faces[:, 1]) & (q_faces[:, 1] < q_faces[:, 2])
    else:
        keep = np.ones(len(q_faces), dtype=bool)

    # Per face corner: (welded vertex id, quantized u, quantized v).
    keys = np.concatenate((q_faces[:, :, None], q_uv[faces_sorted]), axis=-1)[keep]
    edge_keys = np.stack(
        (keys[:, [0, 1]], keys[:, [1, 2]], keys[:, [0, 2]]), axis=1
    ).reshape(-1, 6)

    vertex_pairs, pair_count = np.unique(edge_keys[:, [0, 3]], axis=0, return_counts=True)
    uv_edges = np.unique(edge_keys, axis=0)
    _, uv_count = np.unique(uv_edges[:, [0, 3]], axis=0, return_counts=True)

    endpoints = welded[vertex_pairs].reshape(-1, 3)
    return endpoints.astype(np.float32), pair_count == 1, uv_count > 1


def preprocess_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray | None = None,
    remove_degenerate: bool = True,
) -> MeshInput:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("empty mesh")

    vertices, center, factor = normalize_vertices(vertices, 0.95)

    has_uv = (
        uvs is not None
        and len(uvs) == len(vertices)
        and np.all(np.isfinite(uvs))
    )
    uv = np.asarray(uvs, np.float32)[:, :2] if has_uv else np.zeros((len(vertices), 2), np.float32)

    endpoints, is_boundary, multi_uv = extract_edges(vertices, faces, uv, remove_degenerate)
    if len(endpoints) == 0:
        raise ValueError("mesh has no non-degenerate edges")

    vertices_t, transform = transform_and_normalize(vertices, rng2trs(_EVAL_RNG), 0.95)
    edges = apply_transform(endpoints, transform).reshape(-1, 6)
    edges_world = (endpoints / factor + center).reshape(-1, 6)

    return MeshInput(
        edges=edges.astype(np.float32),
        vertices=vertices_t.astype(np.float32),
        faces=faces,
        face_count=len(faces),
        edges_world=edges_world.astype(np.float32),
        is_seam=(is_boundary | multi_uv) if has_uv else None,
    )


def load_and_preprocess(path: str, remove_degenerate: bool = True) -> MeshInput:
    return preprocess_mesh(*load_mesh(path), remove_degenerate=remove_degenerate)


def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int = 40960,
    extrude_prob: float = 0.3,
    pos_noise: float = 0.02,
    normal_noise: float = 0.03,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample ``[num_points, 6]`` surface points + face normals.

    Defaults reproduce the augmentation the model saw during training and
    validation: with probability ``extrude_prob`` the samples are split into
    two shells pushed slightly outward/inward along the normal, then Gaussian
    noise is added to positions and normals.
    """
    rng = np.random.default_rng(seed)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    pool = 499712 // 2
    points, face_idx = trimesh.sample.sample_surface(mesh, pool, seed=rng.integers(2**31))
    surface = np.concatenate([points, mesh.face_normals[face_idx]], axis=1).astype(np.float32)

    if rng.random() < extrude_prob:
        dist = rng.choice([1 / 128, 1 / 256, 1 / 512, 1 / 768])
        half = num_points // 2
        outer = surface[rng.choice(pool, half, replace=False)]
        inner = surface[rng.choice(pool, half, replace=False)]
        inner[:, 3:] *= -1
        outer[:, :3] += outer[:, 3:] * dist * 0.5
        inner[:, :3] += inner[:, 3:] * dist * 0.5
        surface = np.concatenate([outer, inner])
    else:
        surface = surface[rng.choice(pool, num_points, replace=False)]

    surface[:, :3] += rng.standard_normal((len(surface), 3)).astype(np.float32) * pos_noise
    surface[:, 3:] += rng.standard_normal((len(surface), 3)).astype(np.float32) * normal_noise
    surface[:, 3:] /= np.linalg.norm(surface[:, 3:], axis=-1, keepdims=True) + 1e-6
    return torch.from_numpy(surface)
