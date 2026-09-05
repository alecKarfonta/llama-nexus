"""Parse GGUF filenames into deploy registry name/variant pairs."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

QUANT_PATTERN = re.compile(r"Q\d+_[KSXL0-9]+(?:_[MSXL0-9]+)?")
MULTIPART_PATTERN = re.compile(r"(.+)-(\d{5}-of-\d{5})$")
MTP_DEPLOY_PATTERN = re.compile(r"^(.+)-MTP-(Q\d+_.*)$")
LEGACY_MTP_SUFFIX_PATTERN = re.compile(r"^(.+)-(Q\d+_.*)-MTP$")


def gguf_stem(filename: str) -> str:
    if filename.lower().endswith(".gguf"):
        return filename[:-5]
    return Path(filename).stem


def parse_name_variant(filename: str) -> Tuple[str, Optional[str]]:
    """Return (model_name, variant) from a GGUF filename."""
    stem = Path(gguf_stem(filename)).name
    multipart = MULTIPART_PATTERN.match(stem)
    if multipart:
        stem = multipart.group(1)

    mtp_deploy = MTP_DEPLOY_PATTERN.match(stem)
    if mtp_deploy:
        return f"{mtp_deploy.group(1)}-MTP", mtp_deploy.group(2)

    legacy = LEGACY_MTP_SUFFIX_PATTERN.match(stem)
    if legacy:
        return legacy.group(1), f"{legacy.group(2)}-MTP"

    # [2026-09-05 | INFO | gguf_model_naming.parse_name_variant | Unsloth-style variants
    # "Model-UD-Q4_K_XL" -> ("Model", "UD-Q4_K_XL"); likewise plain IQ/Q prefixes.
    unsloth_match = re.match(r"(.+?)-((?:UD-)?(?:IQ|Q)\d\S*)$", stem)
    if unsloth_match:
        return unsloth_match.group(1), unsloth_match.group(2)

    if "." in stem:
        parts = stem.split(".")
        if parts[-1].startswith("Q"):
            return ".".join(parts[:-1]), parts[-1]

    if "-" in stem:
        parts = stem.split("-")
        if parts[-1].startswith("Q"):
            return "-".join(parts[:-1]), parts[-1]

    embedded = re.match(r"(.*?)(Q\d+_[KS](_[MS])?)$", stem)
    if embedded:
        return embedded.group(1), embedded.group(2)

    if QUANT_PATTERN.search(stem):
        quant_match = QUANT_PATTERN.search(stem)
        if quant_match:
            quant = quant_match.group(0)
            name = stem.replace(quant, "").rstrip("_-")
            if name:
                return name, quant

    return stem, None


def plan_graft_output_filename(base_name: str, variant: str) -> str:
    """Deploy-friendly output: {name}-MTP-{variant}.gguf"""
    clean = base_name[:-5] if base_name.endswith("-MTP") else base_name
    clean = clean.rstrip("-")
    return f"{clean}-MTP-{variant}.gguf"


def plan_graft_deploy_identity(base_name: str, variant: str) -> Tuple[str, str]:
    clean = base_name[:-5] if base_name.endswith("-MTP") else base_name
    clean = clean.rstrip("-")
    return f"{clean}-MTP", variant


def resolve_gguf_path(models_dir: Path, name: str, variant: str) -> Optional[Path]:
    """Resolve a local GGUF path from registry name + variant."""
    candidates: List[Path] = [
        models_dir / f"{name}-{variant}.gguf",
        models_dir / f"{name}.{variant}.gguf",
        models_dir / f"{name}_{variant}.gguf",
        models_dir / variant / f"{name}-{variant}.gguf",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def is_mtp_eligible_architecture(architecture: Optional[str]) -> bool:
    if not architecture:
        return False
    arch = architecture.lower()
    return arch in ("qwen35", "qwen35moe", "qwen3", "llama") or "qwen" in arch
