"""MTP head graft preflight and path planning."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.gguf_metadata import scan_gguf_mtp
from modules.gguf_model_naming import (
    is_mtp_eligible_architecture,
    plan_graft_deploy_identity,
    plan_graft_output_filename,
)


@dataclass
class GraftPreflightResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    architecture: Optional[str] = None
    target_block_count: Optional[int] = None
    donor_block_count: Optional[int] = None
    donor_nextn: Optional[int] = None
    extra_tensor_count: int = 0
    planned_output_filename: Optional[str] = None
    planned_deploy_name: Optional[str] = None
    planned_deploy_variant: Optional[str] = None
    disk_required_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "architecture": self.architecture,
            "target_block_count": self.target_block_count,
            "donor_block_count": self.donor_block_count,
            "donor_nextn": self.donor_nextn,
            "extra_tensor_count": self.extra_tensor_count,
            "planned_output_filename": self.planned_output_filename,
            "planned_deploy_name": self.planned_deploy_name,
            "planned_deploy_variant": self.planned_deploy_variant,
            "disk_required_bytes": self.disk_required_bytes,
        }


def _gguf_field(reader, key: str):
    field_obj = reader.get_field(key)
    return field_obj.contents() if field_obj else None


def preflight_graft(
    target_path: Path,
    donor_path: Path,
    *,
    target_name: str,
    target_variant: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> GraftPreflightResult:
    result = GraftPreflightResult(ok=False)

    if not target_path.is_file():
        result.errors.append(f"Target GGUF not found: {target_path}")
        return result
    if not donor_path.is_file():
        result.errors.append(f"Donor GGUF not found: {donor_path}")
        return result

    target_info = scan_gguf_mtp(target_path)
    donor_info = scan_gguf_mtp(donor_path)

    if target_info.mtp_capable:
        result.errors.append("Target already has MTP heads (mtp_capable=true)")
    if not donor_info.mtp_capable:
        result.errors.append("Donor has no MTP heads (mtp_capable=false)")

    result.architecture = target_info.architecture or donor_info.architecture
    if target_info.architecture and donor_info.architecture:
        if target_info.architecture != donor_info.architecture:
            result.errors.append(
                f"Architecture mismatch: target={target_info.architecture}, "
                f"donor={donor_info.architecture}"
            )

    if not is_mtp_eligible_architecture(result.architecture):
        result.warnings.append(
            f"Architecture {result.architecture!r} is not in the known MTP-eligible list; "
            "graft may still work if block layout matches."
        )

    try:
        from gguf import GGUFReader

        tr = GGUFReader(str(target_path))
        sr = GGUFReader(str(donor_path))
        arch = _gguf_field(tr, "general.architecture") or result.architecture
        if not arch:
            result.errors.append("Could not read general.architecture from target")
            return result

        tb = int(_gguf_field(tr, f"{arch}.block_count") or 0)
        sb = int(_gguf_field(sr, f"{arch}.block_count") or 0)
        sn = int(_gguf_field(sr, f"{arch}.nextn_predict_layers") or 0)
        result.target_block_count = tb
        result.donor_block_count = sb
        result.donor_nextn = sn

        if sn <= 0:
            result.errors.append("Donor nextn_predict_layers is missing or zero")
        elif tb != sb - sn:
            result.errors.append(
                f"Block layout mismatch: target block_count={tb}, "
                f"donor block_count={sb}, nextn={sn} (expected target={sb - sn})"
            )

        prefix = f"blk.{tb}."
        extra = [t.name for t in sr.tensors if t.name.startswith(prefix)]
        result.extra_tensor_count = len(extra)
        if not extra:
            result.errors.append(f"No donor tensors with prefix {prefix!r}")
    except Exception as exc:
        result.errors.append(f"GGUF preflight failed: {exc}")

    out_name = plan_graft_output_filename(target_name, target_variant)
    deploy_name, deploy_variant = plan_graft_deploy_identity(target_name, target_variant)
    result.planned_output_filename = out_name
    result.planned_deploy_name = deploy_name
    result.planned_deploy_variant = deploy_variant

    models_dir = target_path.parent
    out = output_path or (models_dir / out_name)
    target_size = target_path.stat().st_size
    if out.is_file():
        result.disk_required_bytes = target_size
    else:
        # Output ~= target weights + MTP block copied from donor (typically <1GB).
        result.disk_required_bytes = target_size + max(400_000_000, donor_path.stat().st_size // 20)

    if out.is_file() and not overwrite:
        result.errors.append(f"Output already exists: {out.name} (set overwrite=true to replace)")

    import shutil

    usage = shutil.disk_usage(models_dir)
    if result.disk_required_bytes > usage.free:
        result.errors.append(
            f"Insufficient disk space: need ~{result.disk_required_bytes // (1024**3)}GB, "
            f"free {usage.free // (1024**3)}GB"
        )

    result.ok = len(result.errors) == 0
    return result
