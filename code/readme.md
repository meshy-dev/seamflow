# SeamFlow

SeamFlow predicts UV seams on triangle meshes. It treats every mesh edge as a
token and uses rectified flow to map Gaussian noise to a per-edge seam score.
The scores are conditioned on a point cloud sampled from the surface.

This repository contains inference code and a checkpoint converter. It needs
only PyTorch, NumPy, trimesh and safetensors.

## Installation

```bash
pip install -e .
# Optional: faster varlen attention on CUDA
pip install -e ".[flash]"
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.4. A CUDA GPU is recommended.

## Weights

The weights are on Hugging Face at
[meshy-ai-team/seamflow](https://huggingface.co/meshy-ai-team/seamflow):

```bash
huggingface-cli download meshy-ai-team/seamflow --local-dir checkpoints
```

This creates `checkpoints/seamflow_sf2k_c12k_s14_0.940M/`, which holds
`config.json` and `model.safetensors`. Any directory with these two files works
as `--ckpt`.

## Usage

### Command line

```bash
seamflow-infer --ckpt checkpoints/seamflow_sf2k_c12k_s14_0.940M mesh.glb other.obj meshes_dir/ -o outputs/
```

For each mesh, this writes:

- `<name>_seams.obj`: the predicted seam edges as OBJ line elements, in the
  input mesh's coordinates. Load it next to the original mesh to view the seams.
- `<name>.npz`: `edges [E, 6]`, `scores [E]`, `is_seam [E]` and, when the
  input has UVs, `gt_seam [E]`.

If the input mesh already has UVs, the command also prints precision, recall
and F1 against the seams those UVs imply.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--steps` | 50 | Number of flow integration steps |
| `--method` | `euler` | Sampler: `euler` (deterministic ODE) or `ddpm` (stochastic) |
| `--threshold` | 0.0 | An edge with a score above this value is a seam |
| `--seed` | 0 | Seed for point sampling, FPS and the initial noise |
| `--no-augment` | off | Sample a clean point cloud (see below) |

### Python

```python
from seamflow import SeamFlowPipeline

pipe = SeamFlowPipeline.from_pretrained("checkpoints/seamflow_sf2k_c12k_s14_0.940M")
pred = pipe("mesh.glb", n_steps=50, seed=0)

pred.scores    # [E] per-edge score
pred.is_seam   # [E] bool
pred.edges     # [E, 6] edge endpoints (x0 y0 z0 x1 y1 z1)
pred.save_obj("mesh_seams.obj")
```

To bring your own mesh arrays, call
`seamflow.preprocess_mesh(vertices, faces, uvs=None)` and then
`pipe.predict_preprocessed(...)`.

## How it works

1. **Preprocessing.** The mesh is normalized to `[-0.95, 0.95]`. Vertices are
   welded on a 2¹⁶ grid, so vertices split by UVs or normals merge back
   together. Unique undirected edges are then extracted. Finally the mesh is
   rotated into the canonical orientation used in training.
2. **Surface condition.** 40,960 points with normals are sampled from the
   surface. 1,024 of them are chosen by farthest point sampling as queries.
   The queries cross-attend to the full cloud and pass through 8
   self-attention layers. A token that encodes the face count is prepended.
3. **MMDiT.** Each edge token embeds Fourier(v0), Fourier(v1) and the flow
   state `x_t`. Edge and surface tokens are processed jointly by
   4 multi-stream layers (shared attention, a separate MLP per modality) and
   28 fused layers. The flow time `t` modulates the layers through AdaLN. The
   model has no positional encoding beyond the Fourier features.
4. **Sampling.** The ODE is integrated from `x_0 ~ N(0, I)` at `t = 0` to
   `t = 1`. Edges with `x_1 > 0` are seams.

By default the surface point cloud is sampled with the training-time
augmentation. Gaussian noise is added to positions (σ = 0.02) and normals
(σ = 0.03). With probability 0.3, the points are also split into two shells
offset slightly along the normal. These defaults match what the model saw in
training and validation. `--no-augment` turns them off.

## Converting a training checkpoint

```bash
python scripts/convert_checkpoint.py /path/to/ckpt_dir checkpoints/seamflow [--dtype bfloat16]
```

This reads a `torch.distributed.checkpoint` directory, keeps only the model
weights, and writes `model.safetensors` and `config.json`.

## Tests

```bash
pip install -e ".[dev]"
pytest tests
```

## Citation

This code accompanies
[SeamFlow: Structure-Aware Flow Matching on Edge Probabilities for Artist-Like UV Unwrapping](https://arxiv.org/abs/2609.04751).

```bibtex
@inproceedings{zhao2026seamflow,
  title         = {SeamFlow: Structure-Aware Flow Matching on Edge Probabilities for Artist-Like UV Unwrapping},
  author        = {Zhao, Yuming and Xian, Zangyueyang and Zhang, Qijian and Liang, Rendong and Jia, Qin and He, Ying and Hou, Junhui},
  booktitle     = {SIGGRAPH Asia 2026},
  year          = {2026},
}
```

