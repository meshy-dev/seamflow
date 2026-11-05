"""Convert a training checkpoint (torch.distributed.checkpoint) to safetensors.

    python scripts/convert_checkpoint.py /path/to/ckpt_0041.610M checkpoints/seamflow

Writes ``model.safetensors`` (model weights only, optimizer state dropped) and
``config.json``. The model state is expected under ``train_helper.model_0``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from seamflow.model import SeamFlowConfig, SeamFlowModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="DCP checkpoint directory (contains .metadata)")
    ap.add_argument("output", help="output directory")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--prefix", default="train_helper.model_0")
    args = ap.parse_args()

    config = SeamFlowConfig()
    state = SeamFlowModel(config).state_dict()

    # Nest the template the same way the training checkpoint does.
    nested: dict = {}
    node = nested
    *parents, leaf = args.prefix.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = state
    dcp.load(nested, checkpoint_id=args.ckpt)

    dtype = getattr(torch, args.dtype)
    tensors = {k: v.to(dtype).contiguous() for k, v in state.items()}

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    config.to_json(out / "config.json")
    n = sum(v.numel() for v in tensors.values())
    print(f"wrote {len(tensors)} tensors ({n / 1e6:.1f}M params) to {out}")


if __name__ == "__main__":
    main()
