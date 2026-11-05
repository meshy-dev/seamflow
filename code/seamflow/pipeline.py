"""End-to-end seam prediction: mesh file in, scored edges out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from seamflow.mesh import MeshInput, load_and_preprocess, sample_surface
from seamflow.model import SeamFlowModel


@dataclass
class SeamPrediction:
    edges: np.ndarray  # [E, 6] edge endpoints in the input mesh's coordinates
    scores: np.ndarray  # [E] flow output; > threshold means seam
    threshold: float
    gt_seam: np.ndarray | None  # [E] labels from the input UVs, if any

    @property
    def is_seam(self) -> np.ndarray:
        return self.scores > self.threshold

    def metrics(self) -> dict[str, float] | None:
        """Precision / recall / F1 / accuracy against the input UV seams."""
        if self.gt_seam is None:
            return None
        pred, gt = self.is_seam, self.gt_seam
        tp = int((pred & gt).sum())
        precision = tp / max(int(pred.sum()), 1)
        recall = tp / max(int(gt.sum()), 1)
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "accuracy": float((pred == gt).mean()),
        }

    def save_obj(self, path: str | Path, mask: np.ndarray | None = None) -> None:
        """Write the predicted seam edges as OBJ line elements."""
        sel = self.edges[self.is_seam if mask is None else mask]
        lines = []
        for i, e in enumerate(sel):
            lines.append(f"v {e[0]:.6f} {e[1]:.6f} {e[2]:.6f}")
            lines.append(f"v {e[3]:.6f} {e[4]:.6f} {e[5]:.6f}")
            lines.append(f"l {2 * i + 1} {2 * i + 2}")
        Path(path).write_text("\n".join(lines) + "\n")

    def save_npz(self, path: str | Path) -> None:
        data = {"edges": self.edges, "scores": self.scores, "is_seam": self.is_seam}
        if self.gt_seam is not None:
            data["gt_seam"] = self.gt_seam
        np.savez_compressed(path, **data)


class SeamFlowPipeline:
    def __init__(self, model: SeamFlowModel, device: str | torch.device):
        self.model = model
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls, path: str | Path, device: str | torch.device | None = None
    ) -> SeamFlowPipeline:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(SeamFlowModel.from_pretrained(path, device), device)

    @torch.no_grad()
    def predict_preprocessed(
        self,
        mesh: MeshInput,
        n_steps: int = 50,
        method: Literal["euler", "ddpm"] = "euler",
        threshold: float = 0.0,
        seed: int | None = 0,
        surface_augment: bool = True,
    ) -> SeamPrediction:
        aug = {} if surface_augment else {"extrude_prob": 0.0, "pos_noise": 0.0, "normal_noise": 0.0}
        surface = sample_surface(
            mesh.vertices,
            mesh.faces,
            num_points=self.model.config.point_size,
            seed=seed,
            **aug,
        )
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        scores = self.model.sample(
            edges=torch.from_numpy(mesh.edges).to(self.device),
            surface=surface[None].to(self.device),
            face_count=torch.tensor([float(mesh.face_count)], device=self.device),
            n_steps=n_steps,
            method=method,
            dtype=dtype,
            generator=generator,
        )
        return SeamPrediction(
            edges=mesh.edges_world,
            scores=scores.cpu().numpy(),
            threshold=threshold,
            gt_seam=mesh.is_seam,
        )

    def __call__(self, mesh_path: str | Path, **kwargs) -> SeamPrediction:
        """Predict UV seams for a mesh file (.glb / .gltf / .obj / ...).

        Keyword arguments go to :meth:`predict_preprocessed`.
        """
        return self.predict_preprocessed(load_and_preprocess(str(mesh_path)), **kwargs)
