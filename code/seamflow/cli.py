"""Command-line inference.

    seamflow-infer --ckpt checkpoints/seamflow mesh.glb [more.obj | dir/ ...] -o out/

For each mesh, writes ``<name>_seams.obj`` (seam edges as OBJ lines, in the
input mesh's coordinates) and ``<name>.npz`` (all edges and their scores).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from seamflow.pipeline import SeamFlowPipeline

MESH_EXTS = {".glb", ".gltf", ".obj", ".ply", ".off", ".stl", ".fbx"}


def _collect(inputs: list[str]) -> list[Path]:
    paths = []
    for item in map(Path, inputs):
        if item.is_dir():
            paths += sorted(p for p in item.rglob("*") if p.suffix.lower() in MESH_EXTS)
        else:
            paths.append(item)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="mesh files or directories")
    ap.add_argument("--ckpt", required=True, help="dir with config.json + model.safetensors")
    ap.add_argument("-o", "--output", default="outputs")
    ap.add_argument("--steps", type=int, default=50, help="flow integration steps")
    ap.add_argument("--method", choices=["euler", "ddpm"], default="euler")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true", help="sample a clean surface point cloud")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pipe = SeamFlowPipeline.from_pretrained(args.ckpt, args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in _collect(args.inputs):
        start = time.perf_counter()
        try:
            pred = pipe(
                path,
                n_steps=args.steps,
                method=args.method,
                threshold=args.threshold,
                seed=args.seed,
                surface_augment=not args.no_augment,
            )
        except ValueError as e:
            print(f"[skip] {path}: {e}")
            continue
        pred.save_obj(out_dir / f"{path.stem}_seams.obj")
        pred.save_npz(out_dir / f"{path.stem}.npz")
        info = {"edges": len(pred.scores), "seams": int(pred.is_seam.sum())}
        if (m := pred.metrics()) is not None:
            info |= {k: round(v, 4) for k, v in m.items()}
        print(f"{path.name}: {json.dumps(info)} ({time.perf_counter() - start:.1f}s)")


if __name__ == "__main__":
    main()
