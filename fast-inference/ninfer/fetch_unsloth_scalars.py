#!/usr/bin/env python3
# [2026-08-27 | INFO | fetch_unsloth_scalars | Fetch only the F32 scalar tensors
# (weight_global_scale / input_global_scale) and a few reference tensors from the
# unsloth/Qwen3.8-27B-NVFP4 checkpoint via HTTP range requests, without downloading
# the ~16 GB of packed weights.
#
# safetensors format: first 8 bytes = little-endian u64 header length, then a JSON
# header mapping tensor names -> {dtype, shape, data_offsets}. We fetch the header
# once, then range-fetch only the byte ranges we need.
import json
import struct
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/resolve/main"
OUT = Path("/tmp/unsloth_scalars")
OUT.mkdir(parents=True, exist_ok=True)


def fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def fetch_file_header(fname: str) -> dict:
    url = f"{BASE}/{fname}"
    # first 8 bytes: header length
    hdr_len = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    header = json.loads(fetch_range(url, 8, 8 + hdr_len - 1))
    return {"url": url, "header": header, "hdr_len": hdr_len}


def fetch_tensors(fname: str, names: set[str], header_info: dict) -> dict[str, bytes]:
    header = header_info["header"]
    out = {}
    data_base = 8 + header_info["hdr_len"]
    for name in names:
        meta = header.get(name)
        if meta is None:
            continue
        s, e = meta["data_offsets"]
        out[name] = fetch_range(header_info["url"], data_base + s, data_base + e - 1)
    return out


def main():
    index = json.loads(urllib.request.urlopen(f"{BASE}/model.safetensors.index.json").read())
    weight_map = index["weight_map"]

    # which shard holds each wanted tensor
    wanted_scalars = set()
    for name, shard in weight_map.items():
        if name.endswith(".weight_global_scale") or name.endswith(".input_global_scale"):
            wanted_scalars.add(name)
    print(f"scalar tensors wanted: {len(wanted_scalars)}")

    # one reference nvfp4 weight (small: k_proj? no - reference mlp weight_packed of layer 0)
    # and one reference fp8 weight with scale for semantic validation
    refs = {
        "model.language_model.layers.0.mlp.gate_proj.weight_packed",
        "model.language_model.layers.0.mlp.gate_proj.weight_scale",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight_scale",
    }

    by_shard: dict[str, set[str]] = {}
    for name in wanted_scalars | refs:
        shard = weight_map.get(name)
        if shard:
            by_shard.setdefault(shard, set()).add(name)

    result: dict[str, bytes] = {}
    for shard in sorted(by_shard):
        info = fetch_file_header(shard)
        got = fetch_tensors(shard, by_shard[shard], info)
        result.update(got)
        print(f"{shard}: fetched {len(got)} tensors")

    with open(OUT / "scalars.bin", "wb") as f:
        for name in sorted(result):
            f.write(struct.pack("<Q", len(name)) + name.encode() + struct.pack("<Q", len(result[name])) + result[name])

    # quick sanity print
    for name in sorted(wanted_scalars)[:4]:
        if name in result:
            val = struct.unpack("<f", result[name])[0]
            print(f"{name} = {val:.6g}")

    print(f"saved {len(result)} tensors to {OUT/'scalars.bin'}")


if __name__ == "__main__":
    main()
