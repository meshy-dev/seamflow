"""SeamFlow: UV seam prediction for triangle meshes with rectified flow."""

from seamflow.mesh import MeshInput, load_and_preprocess, preprocess_mesh
from seamflow.model import SeamFlowConfig, SeamFlowModel
from seamflow.pipeline import SeamFlowPipeline, SeamPrediction

__version__ = "0.1.0"

__all__ = [
    "MeshInput",
    "SeamFlowConfig",
    "SeamFlowModel",
    "SeamFlowPipeline",
    "SeamPrediction",
    "load_and_preprocess",
    "preprocess_mesh",
]
