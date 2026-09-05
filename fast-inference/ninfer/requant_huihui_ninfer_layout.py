#!/usr/bin/env python3
# [2026-08-27 | INFO | requant_huihui_ninfer_layout | Requantize Huihui BF16 into the exact
# quantized-checkpoint layout ninfer's convert_nvfp4.py expects as its second source.
#
# Semantics verified bit-exact against unsloth/Qwen3.8-27B-NVFP4 reference tensors:
#   FP8 row-scaled:  codes F8_E4M3 [n,k], scales BF16 [n,1]; w = code * scale
#                    (row amax saturates code to 448; scale = amax/448 in BF16)
#   NVFP4:           w = e2m1(nibble) * group_scale(F8_E4M3, per 16) / global_scale(F32)
#                    global_scale = 2688 / tensor_amax (gate+up use JOINT amax => shared global)
#                    low nibble = first element; group scale <= 448; nibble 8 (-0) unused
import argparse
import json
import re
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file, safe_open

FP8_MLP_LAYERS = set(range(56, 64))
VOCAB, HIDDEN, INTERMEDIATE = 248320, 5120, 17408

FP8_E4M3_MAX = 448.0
NVFP4_E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def classify(name: str):
    # [2026-08-27 | INFO | requant_huihui_ninfer_layout | names here are full tensor
    # names (module.weight); recipe sources are module paths. The MTP head and vision
    # tower stay BF16 in the registered layout - only main text layers quantize.
    module = name[: -len(".weight")] if name.endswith(".weight") else name
    if module.startswith(("mtp.", "model.visual")):
        return None
    if module == "lm_head":
        return "fp8"
    m = re.match(r"model\.language_model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj$", module)
    if m:
        return "fp8" if int(m.group(1)) in FP8_MLP_LAYERS else "nvfp4"
    if re.search(r"self_attn\.(q|k|v|o)_proj$", module):
        return "fp8"
    if re.search(r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$", module):
        return "fp8"
    return None


def quantize_fp8_row(w: torch.Tensor):
    """Row-scaled symmetric E4M3FN; scale stored BF16, code = round(w/scale)."""
    w32 = w.to(torch.float32)
    amax = w32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = (amax / FP8_E4M3_MAX).to(torch.bfloat16)
    codes = (w32 / scale.float()).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return codes, scale  # [n,k] F8_E4M3, [n,1] BF16


def nvfp4_global_scale(joint_amax: float) -> torch.Tensor:
    # unsloth stores power-of-two-friendly integers: 2688/amax rounded how?
    # verified: global=6400 for amax=0.42 -> 2688/0.42 = 6400 exactly.
    # We use exact float32 value; converter only requires finite positive.
    return torch.tensor([2688.0 / max(joint_amax, 1e-30)], dtype=torch.float32)


def quantize_nvfp4(w: torch.Tensor, global_scale: torch.Tensor):
    n, k = w.shape
    assert k % 16 == 0, f"k={k} not divisible by 16"
    w32 = w.to(torch.float32)
    gs = global_scale.item()

    groups = w32.reshape(n, k // 16, 16)
    # dequant: w = e2m1 * group_scale / global  =>  e2m1*group_scale = w*global
    # group_scale = amax_group * global / 6, must fit E4M3 (<=448)
    target = groups * gs                      # scaled domain
    g_amax = target.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    group_scale_f = (g_amax / 6.0).clamp(1e-30, 448.0)
    group_scale = group_scale_f.to(torch.float8_e4m3fn).float()

    norm = target / group_scale.clamp(min=1e-30)  # in [0, 6] nominally
    codes_idx = _round_e2m1_index(norm).reshape(n, k)  # flatten groups
    packed = _pack_fp4_idx(codes_idx)              # U8 [n, k/2], low nibble first
    return packed, group_scale.to(torch.float8_e4m3fn).reshape(n, k // 16)


def _round_e2m1_index(x: torch.Tensor) -> torch.Tensor:
    """Map signed float to e2m1 code index 0..15 (bit3 = sign). No -0 (index 8)."""
    grid = torch.tensor(NVFP4_E2M1_GRID, dtype=torch.float32, device=x.device)
    ax = x.abs().clamp(max=6.0)
    idx = torch.argmin((ax.unsqueeze(-1) - grid).abs(), dim=-1).to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    return (idx | (sign.to(torch.uint8) << 3)).to(torch.uint8)


def _pack_fp4_idx(idx: torch.Tensor) -> torch.Tensor:
    n, k = idx.shape
    lo = idx[:, 0::2] & 0xF
    hi = idx[:, 1::2] & 0xF
    return (lo | (hi << 4)).contiguous()


def load_unsloth_scalars(path: Path):
    data = path.read_bytes()
    out = {}
    off = 0
    while off < len(data):
        (nlen,) = struct.unpack_from("<Q", data, off); off += 8
        name = data[off:off+nlen].decode(); off += nlen
        (dlen,) = struct.unpack_from("<Q", data, off); off += 8
        out[name] = data[off:off+dlen]; off += dlen
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input-scales", type=Path, default=None,
                    help="unsloth scalar bundle for borrowing input_global_scale values")
    ap.add_argument("--mtp-override", type=Path, default=None,
                    help="safetensors file whose mtp.* tensors replace the source. "
                         "Use to splice a fine-tuned draft head into the artifact.")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    src, out = args.model, args.out
    out.mkdir(parents=True, exist_ok=True)

    mtp_overrides: dict[str, torch.Tensor] = {}
    if args.mtp_override:
        # [2026-08-28 | INFO | requant_huihui_ninfer_layout | mtp.* overrides come from
        # a separate safetensors (e.g. train_mtp_chat.py output). Keys must exactly
        # match the HF names (mtp.fc.weight, mtp.layers.0.self_attn.q_proj.weight, ...).
        from safetensors import safe_open as _safe_open  # local alias
        with _safe_open(str(args.mtp_override), framework="pt") as f:
            for k in f.keys():
                if not k.startswith("mtp."):
                    raise ValueError(f"mtp-override contains non-mtp key: {k}")
                mtp_overrides[k] = f.get_tensor(k)
        print(f"mtp-override: {len(mtp_overrides)} tensors from {args.mtp_override}", flush=True)

    borrowed = {}
    if args.input_scales:
        blob = load_unsloth_scalars(args.input_scales)
        for name, raw in blob.items():
            if name.endswith(".input_global_scale"):
                key = name[: -len(".input_global_scale")]
                borrowed[key] = struct.unpack("<f", raw)[0]

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    for f in list(src.glob("*.json")) + list(src.glob("*.jinja")) + list(src.glob("*.txt")):
        if f.name != "model.safetensors.index.json":
            (out / f.name).write_bytes(f.read_bytes())

    new_weight_map = {}
    shard_tensors: dict[str, dict[str, torch.Tensor]] = {}
    written_shards: set[str] = set()

    # [2026-08-27 | INFO | requant_huihui_ninfer_layout | Derived quant fields (codes,
    # scales, packed nibbles, scalars) go into ONE dedicated shard; BF16 pass-through
    # tensors keep their original shard. This keeps the rewritten index trivially
    # consistent and avoids partial-flush ordering hazards entirely.
    DERIVED_SHARD = "quant-derived-00001.safetensors"

    def add_tensor(name: str, tensor: torch.Tensor, derived: bool):
        shard = DERIVED_SHARD if derived else weight_map[name]
        shard_tensors.setdefault(shard, {})[name] = tensor.cpu().contiguous().clone()
        new_weight_map[name] = shard

    def flush_completed_shards():
        # [2026-08-27 | INFO | requant_huihui_ninfer_layout | write a pass-through shard
        # as soon as every source tensor it owns has been visited, bounding peak RAM;
        # the derived shard is only written at the end
        remaining_sources = set()
        for src_name, shard in weight_map.items():
            if src_name not in visited:
                remaining_sources.add(shard)
        for shard in sorted(shard_tensors):
            if shard == DERIVED_SHARD or shard in written_shards:
                continue
            if shard in remaining_sources:
                continue  # may still receive tensors
            save_file(shard_tensors[shard], out / shard, metadata={"format": "pt"})
            print(f"wrote {shard} ({len(shard_tensors[shard])} tensors)", flush=True)
            written_shards.add(shard)
            del shard_tensors[shard]

    stats = {"fp8": 0, "nvfp4": 0, "direct": 0}

    # ---- pass 1: compute joint amax for gate/up pairs ----
    # keys are module paths (e.g. ...layers.3.mlp.gate_proj) matching recipe sources
    wgs: dict[str, torch.Tensor] = {}
    gate_up_amax: dict[str, float] = {}
    down_amax: dict[str, float] = {}
    print("pass 1: scanning amax for nvfp4 layers", flush=True)
    for shard in shards:
        with safe_open(src / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                module = name[: -len(".weight")]
                if classify(name) != "nvfp4":
                    continue
                t = f.get_slice(name)
                amax = 0.0
                # chunked amax to bound memory
                n = t.get_shape()[0]
                for beg in range(0, n, 2048):
                    chunk = t[beg:beg+2048].to(torch.float32)
                    amax = max(amax, float(chunk.abs().max()))
                    del chunk
                if module.endswith("gate_proj") or module.endswith("up_proj"):
                    pair = module.rsplit(".", 1)[0]
                    gate_up_amax[pair] = max(gate_up_amax.get(pair, 0.0), amax)
                else:
                    down_amax[module] = amax
        print(f"  {shard} scanned", flush=True)
    for pair, amax in gate_up_amax.items():
        g = nvfp4_global_scale(amax)
        wgs[f"{pair}.gate_proj"] = g
        wgs[f"{pair}.up_proj"] = g
    for module, amax in down_amax.items():
        wgs[module] = nvfp4_global_scale(amax)
    print(f"pass 1 complete: {len(wgs)} global scales", flush=True)

    # ---- pass 2: quantize + write ----
    visited: set[str] = set()
    for shard in shards:
        with safe_open(src / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                # Splice in the trained MTP tensor if we have an override for it.
                if name in mtp_overrides:
                    tensor = mtp_overrides[name].clone()
                else:
                    tensor = f.get_tensor(name)
                visited.add(name)
                module = name[: -len(".weight")]
                kind = classify(name)
                if kind == "fp8":
                    w = tensor.to(dev, non_blocking=True)
                    codes, scale = quantize_fp8_row(w)
                    add_tensor(f"{module}.weight", codes.to("cpu"), derived=True)
                    add_tensor(f"{module}.weight_scale", scale.reshape(-1, 1).to("cpu"), derived=True)
                    stats["fp8"] += 1
                    del w, codes, scale
                elif kind == "nvfp4":
                    w = tensor.to(dev, non_blocking=True)
                    packed, gscales = quantize_nvfp4(w, wgs[module])
                    add_tensor(f"{module}.weight_packed", packed, derived=True)
                    add_tensor(f"{module}.weight_scale", gscales, derived=True)
                    add_tensor(f"{module}.weight_global_scale", wgs[module].reshape(1).cpu(), derived=True)
                    igs = torch.tensor([borrowed.get(module, 1.0)], dtype=torch.float32)
                    add_tensor(f"{module}.input_global_scale", igs, derived=True)
                    stats["nvfp4"] += 1
                    del w, packed, gscales
                else:
                    add_tensor(name, tensor, derived=False)
                    stats["direct"] += 1
                del tensor
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"processed {shard}: {stats}", flush=True)
        flush_completed_shards()

    for shard_name in sorted(shard_tensors):
        save_file(shard_tensors[shard_name], out / shard_name, metadata={"format": "pt"})
        print(f"wrote {shard_name} ({len(shard_tensors[shard_name])} tensors)", flush=True)

    # config.json: mixed-precision quantization_config matching _validate_quantized_config
    cfg = json.loads((out / "config.json").read_text())
    cfg["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "input_activations": {
                    "dynamic": True, "group_size": None, "num_bits": 8,
                    "scale_dtype": None, "strategy": "token", "symmetric": True, "type": "float",
                },
                "output_activations": None,
                "targets": [
                    "re:.*self_attn\\.(q|k|v|o)_proj$",
                    "re:.*linear_attn\\.(in_proj_qkv|in_proj_z|out_proj)$",
                    "re:.*lm_head",
                    "re:.*layers\\.(56|57|58|59|60|61|62|63)\\.mlp\\.(gate|up|down)_proj$",
                ],
                "weights": {
                    "dynamic": False, "group_size": None, "num_bits": 8,
                    "scale_dtype": None, "strategy": "channel", "symmetric": True, "type": "float",
                },
            },
            "group_1": {
                "format": "nvfp4-pack-quantized",
                "input_activations": {
                    "dynamic": "local", "group_size": 16, "num_bits": 4,
                    "scale_dtype": "torch.float8_e4m3fn", "strategy": "tensor_group",
                    "symmetric": True, "type": "float",
                },
                "output_activations": None,
                "targets": ["re:.*mlp\\.(gate|up|down)_proj$"],
                "weights": {
                    "dynamic": False, "group_size": 16, "num_bits": 4,
                    "scale_dtype": "torch.float8_e4m3fn", "strategy": "tensor_group",
                    "symmetric": True, "type": "float",
                },
            },
        },
        "format": "mixed-precision",
        "ignore": ["re:.*visual.*", "re:.*conv1d.*", "re:.*mtp.*",
                   "model.language_model.embed_tokens", "model.language_model.layers.0.input_layernorm"],
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    idx = {"metadata": index.get("metadata", {}), "weight_map": new_weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
    print(f"DONE stats={stats}", flush=True)


if __name__ == "__main__":
    main()
