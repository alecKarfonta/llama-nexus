"""Read GGUF file metadata (header/KV only) for model registry features such as MTP detection."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_SIZES = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}


class GgufMetadataError(Exception):
    """Raised when a GGUF file cannot be parsed."""


@dataclass(frozen=True)
class MtpGgufInfo:
    mtp_capable: bool
    mtp_nextn_layers: int
    architecture: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mtp_capable": self.mtp_capable,
            "mtp_nextn_layers": self.mtp_nextn_layers,
            "architecture": self.architecture,
            "error": self.error,
        }


def gguf_header_path_for_filename(models_dir: Path, filename: str) -> Path:
    """Resolve the GGUF path to read metadata from (part 1 for multi-part files)."""
    import re

    path = models_dir / filename
    match = re.match(r"(.+)-\d{5}-of-(\d{5})\.gguf$", filename, flags=re.IGNORECASE)
    if match:
        first_name = f"{match.group(1)}-00001-of-{match.group(2)}.gguf"
        first_path = models_dir / first_name
        if first_path.is_file():
            return first_path
    return path


def read_gguf_metadata(path: Union[str, Path], *, max_kv: int = 4096) -> Dict[str, Any]:
    """Read all KV metadata from a GGUF file without loading tensor data."""
    file_path = Path(path)
    with file_path.open("rb") as handle:
        return _read_gguf_metadata_from_handle(handle, max_kv=max_kv)


def _read_gguf_metadata_from_handle(handle, *, max_kv: int) -> Dict[str, Any]:
    header = handle.read(4)
    if len(header) != 4:
        raise GgufMetadataError("file too short for GGUF magic")
    magic = struct.unpack("<I", header)[0]
    if magic != GGUF_MAGIC:
        raise GgufMetadataError("invalid GGUF magic")

    version = _read_u32(handle)
    if version not in (2, 3):
        raise GgufMetadataError(f"unsupported GGUF version {version}")

    n_tensors = _read_u64(handle)
    n_kv = _read_u64(handle)
    if n_kv > max_kv:
        raise GgufMetadataError(f"too many KV pairs ({n_kv})")

    metadata: Dict[str, Any] = {}
    for _ in range(n_kv):
        key = _read_string(handle)
        value_type = _read_u32(handle)
        is_array = False
        count = 1
        if value_type == GGUF_TYPE_ARRAY:
            is_array = True
            value_type = _read_u32(handle)
            count = _read_u64(handle)
        value = _read_value(handle, value_type, count, is_array)
        metadata[key] = value

    # n_tensors is unused for MTP detection; caller only needs metadata.
    _ = n_tensors
    return metadata


def detect_mtp_from_metadata(metadata: Dict[str, Any]) -> MtpGgufInfo:
    """Derive MTP capability from GGUF KV metadata (`{arch}.nextn_predict_layers`)."""
    architecture = metadata.get("general.architecture")
    if isinstance(architecture, bytes):
        architecture = architecture.decode("utf-8", errors="replace")
    if architecture is not None:
        architecture = str(architecture)

    layers = 0
    if architecture:
        layers = _coerce_uint(metadata.get(f"{architecture}.nextn_predict_layers"))
    if layers <= 0:
        for key, raw in metadata.items():
            if str(key).endswith(".nextn_predict_layers"):
                layers = _coerce_uint(raw)
                if layers > 0:
                    break

    return MtpGgufInfo(
        mtp_capable=layers > 0,
        mtp_nextn_layers=layers,
        architecture=architecture,
    )


def scan_gguf_mtp(path: Union[str, Path]) -> MtpGgufInfo:
    """Scan a local GGUF file for MTP (NextN) heads."""
    try:
        metadata = read_gguf_metadata(path)
        return detect_mtp_from_metadata(metadata)
    except GgufMetadataError as exc:
        return MtpGgufInfo(mtp_capable=False, mtp_nextn_layers=0, error=str(exc))
    except OSError as exc:
        return MtpGgufInfo(mtp_capable=False, mtp_nextn_layers=0, error=str(exc))


def _coerce_uint(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        return _coerce_uint(value[0])
    try:
        parsed = int(value)
        return max(0, parsed)
    except (TypeError, ValueError):
        return 0


def _read_u32(handle) -> int:
    data = handle.read(4)
    if len(data) != 4:
        raise GgufMetadataError("unexpected EOF reading u32")
    return struct.unpack("<I", data)[0]


def _read_u64(handle) -> int:
    data = handle.read(8)
    if len(data) != 8:
        raise GgufMetadataError("unexpected EOF reading u64")
    return struct.unpack("<Q", data)[0]


def _read_string(handle) -> str:
    length = _read_u64(handle)
    if length > 10_000_000:
        raise GgufMetadataError("string length too large")
    raw = handle.read(length)
    if len(raw) != length:
        raise GgufMetadataError("unexpected EOF reading string")
    return raw.decode("utf-8", errors="replace")


def _read_value(handle, value_type: int, count: int, is_array: Any):
    if is_array:
        if value_type == GGUF_TYPE_STRING:
            return [_read_string(handle) for _ in range(count)]
        element_size = _SCALAR_SIZES.get(value_type)
        if element_size is None:
            raise GgufMetadataError(f"unsupported array element type {value_type}")
        raw = handle.read(element_size * count)
        if len(raw) != element_size * count:
            raise GgufMetadataError("unexpected EOF reading array")
        return _decode_scalar_array(value_type, raw, count)

    if value_type == GGUF_TYPE_STRING:
        return _read_string(handle)
    element_size = _SCALAR_SIZES.get(value_type)
    if element_size is None:
        raise GgufMetadataError(f"unsupported value type {value_type}")
    raw = handle.read(element_size)
    if len(raw) != element_size:
        raise GgufMetadataError("unexpected EOF reading scalar")
    return _decode_scalar(value_type, raw)


def _decode_scalar(value_type: int, raw: bytes):
    if value_type == GGUF_TYPE_UINT8:
        return raw[0]
    if value_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", raw)[0]
    if value_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", raw)[0]
    if value_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", raw)[0]
    if value_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", raw)[0]
    if value_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", raw)[0]
    if value_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", raw)[0]
    if value_type == GGUF_TYPE_BOOL:
        return raw[0] != 0
    if value_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", raw)[0]
    if value_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", raw)[0]
    if value_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", raw)[0]
    raise GgufMetadataError(f"cannot decode scalar type {value_type}")


def _decode_scalar_array(value_type: int, raw: bytes, count: int):
    return [_decode_scalar(value_type, raw[i * _SCALAR_SIZES[value_type]:(i + 1) * _SCALAR_SIZES[value_type]]) for i in range(count)]
