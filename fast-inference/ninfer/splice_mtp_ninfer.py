#!/usr/bin/env python3
# [2026-08-30 | INFO | splice_mtp_ninfer | Rewrite mtp/* objects in an existing
# .ninfer artifact from a trained mtp.*.safetensors overlay, copying every other
# object byte-for-byte. Avoids a full requant + convert of the 27B backbone.
#
# Usage (ninfer-convert image, ninfer source at /ninfer):
#   python3 /work/splice_mtp_ninfer.py \
#     --artifact /models/qwen3_8_27b_nvfp4.ninfer \
#     --official /hf \
#     --mtp-override /out/mtp.chat.safetensors \
#     --out /models/qwen3_8_27b_nvfp4.chat-mtp.ninfer

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open


class OverlayReader:
    """ShardReader-compatible get() that prefers overlay tensors."""

    def __init__(self, official_dir: Path, overlay: dict[str, torch.Tensor]):
        sys.path.insert(0, "/ninfer")
        from tools.convert.common.safetensors import ShardReader

        self._inner = ShardReader(official_dir)
        self._overlay = overlay

    def get(self, name: str) -> torch.Tensor:
        if name in self._overlay:
            return self._overlay[name]
        return self._inner.get(name)

    def has(self, name: str) -> bool:
        return name in self._overlay or self._inner.has(name)

    def close(self) -> None:
        self._inner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--mtp-override", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, "/ninfer")
    from tools.artifact.container import (
        Artifact,
        ArtifactWriter,
        ResourceObject,
        ResourceSpec,
        TensorObject,
        TensorSpec,
    )
    from tools.convert.common.quantize import pick_device
    from tools.convert.qwen3_6.common import conversion as family_conversion
    from tools.convert.qwen3_8_27b import inventory_nvfp4 as inventory
    from tools.convert.qwen3_8_27b.convert_nvfp4 import _materialize_official

    overlay: dict[str, torch.Tensor] = {}
    with safe_open(str(args.mtp_override), framework="pt") as f:
        for k in f.keys():
            overlay[k] = f.get_tensor(k).contiguous()
    print(f"overlay tensors: {len(overlay)}", flush=True)

    device = pick_device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    inv_by_name = {spec.name: spec for spec in inventory.OBJECT_SPECS}

    with Artifact(args.artifact) as src, OverlayReader(args.official, overlay) as reader:
        writer_specs = []
        for obj in src.objects:
            if isinstance(obj, TensorObject):
                writer_specs.append(TensorSpec(obj.name, obj.shape, obj.format, obj.layout))
            elif isinstance(obj, ResourceObject):
                writer_specs.append(ResourceSpec(obj.name, obj.encoding, obj.bytes))
            else:
                raise TypeError(f"unknown object {obj}")
        with ArtifactWriter(args.out, src.identity, writer_specs) as writer:
            for i, obj in enumerate(src.objects, start=1):
                if obj.name.startswith("mtp/") and isinstance(obj, TensorObject):
                    inv = inv_by_name[obj.name]
                    tensor = _materialize_official(inv, reader, {})
                    payload = family_conversion.encode_tensor_payload(tensor, inv, device)
                    del tensor
                    writer.write(obj.name, payload)
                    print(f"[{i}/{len(src.objects)}] REENC {obj.name}", flush=True)
                else:
                    payload = bytes(src.payload(obj.name))
                    writer.write(obj.name, payload)
                    print(f"[{i}/{len(src.objects)}] COPY  {obj.name}", flush=True)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
