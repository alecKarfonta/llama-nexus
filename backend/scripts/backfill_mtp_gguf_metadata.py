#!/usr/bin/env python3
"""One-shot backfill: scan local GGUF files and persist MTP metadata in .metadata/."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root or backend/
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from modules.gguf_metadata import gguf_header_path_for_filename, scan_gguf_mtp  # noqa: E402


def _parse_name_variant(filename: str):
    import re

    stem = filename[:-5] if filename.lower().endswith(".gguf") else Path(filename).stem
    multipart_match = re.match(r"(.+)-(\d{5}-of-\d{5})$", stem)
    if multipart_match:
        stem = multipart_match.group(1)
    if "." in stem:
        parts = stem.split(".")
        if parts[-1].startswith("Q"):
            return ".".join(parts[:-1]), parts[-1]
    if "-" in stem:
        parts = stem.split("-")
        if parts[-1].startswith("Q"):
            return "-".join(parts[:-1]), parts[-1]
    return stem, None


def main() -> int:
    models_dir = Path(os.getenv("LLAMACPP_MODELS_DIR", "/home/llamacpp/models"))
    metadata_dir = models_dir / ".metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    scanned = 0
    updated = 0
    mtp_capable_count = 0

    seen_keys: set[str] = set()
    for entry in sorted(models_dir.rglob("*.gguf")):
        name = entry.name
        if "-00001-of-" not in name and "-of-" in name and name.endswith(".gguf"):
            # Skip non-first multipart shards
            import re

            if re.search(r"-\d{5}-of-\d{5}\.gguf$", name) and "-00001-of-" not in name:
                continue

        model_name, variant = _parse_name_variant(name)
        if not model_name:
            continue
        variant = variant or "unknown"
        model_key = f"{model_name}:{variant}"
        if model_key in seen_keys:
            continue
        seen_keys.add(model_key)

        header = gguf_header_path_for_filename(models_dir, name)
        if not header.is_file():
            continue

        scanned += 1
        info = scan_gguf_mtp(header)
        if info.mtp_capable:
            mtp_capable_count += 1

        metadata_file = metadata_dir / f"{model_name}-{variant}.json"
        existing: dict = {}
        if metadata_file.is_file():
            try:
                existing = json.loads(metadata_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}

        existing.update(
            {
                "model_name": model_name,
                "variant": variant,
                "filename": name,
                "mtp_capable": info.mtp_capable,
                "mtp_nextn_layers": info.mtp_nextn_layers,
                "mtp_scanned_at": datetime.now().isoformat(),
            }
        )
        if info.architecture:
            existing["gguf_architecture"] = info.architecture
        if info.error:
            existing["mtp_scan_error"] = info.error
        else:
            existing.pop("mtp_scan_error", None)

        metadata_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        updated += 1
        print(
            f"{model_name}-{variant}: mtp_capable={info.mtp_capable} "
            f"layers={info.mtp_nextn_layers} arch={info.architecture or '?'}"
        )

    print(f"Done. scanned={scanned} updated={updated} mtp_capable={mtp_capable_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
