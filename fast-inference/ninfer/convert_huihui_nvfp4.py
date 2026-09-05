#!/usr/bin/env python3
# [2026-08-27 | INFO | convert_huihui_nvfp4 | Driver that runs the upstream ninfer
# NVFP4 converter against the Huihui abliterated sources. The upstream converter is
# used unmodified for ALL numeric work; we only relax two closed-world checks that
# pin it to the two official repositories:
#   1. recipe_nvfp4.BASE_REPOSITORY / QUANTIZED_REPOSITORY constants (report-only)
#   2. the nvfp4 global-scale integer-word validation (ours are exact floats)
# Everything else (object plan, encoding, draft head, report) is stock behavior.
#
# Usage (inside the ninfer-convert image, repo at /work/ninfer):
#   python3 tools/convert_huihui_nvfp4.py \
#     --official /models/huihui-bf16 \
#     --quantized /models/huihui-quant \
#     --out /models/artifacts/qwen3_8_27b_nvfp4.ninfer
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from tools.convert.qwen3_8_27b import convert_nvfp4  # noqa: E402
from tools.convert.qwen3_8_27b import recipe_nvfp4 as recipe  # noqa: E402


def patch_repositories(official_name: str, quantized_name: str) -> None:
    # [2026-08-27 | INFO | convert_huihui_nvfp4 | Point provenance constants at our
    # sources; the converter embeds these in the conversion report only.
    recipe.BASE_REPOSITORY = official_name
    recipe.QUANTIZED_REPOSITORY = quantized_name
    convert_nvfp4.recipe = recipe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official", required=True, type=Path)
    ap.add_argument("--quantized", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    patch_repositories("huihui-ai/Huihui-Qwen3.8-27B-abliterated",
                       "local/huihui-nvfp4-requant")

    report = convert_nvfp4.convert(
        args.official, args.quantized, args.out, device=args.device
    )
    print(f"conversion report: {report}")


if __name__ == "__main__":
    main()
