"""Tests for GGUF MTP metadata detection and HF hints."""
import importlib.util
import struct
import sys
import tempfile
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND / rel_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_gguf = _load_module("gguf_metadata_test", "modules/gguf_metadata.py")
_hints = _load_module("mtp_hf_hints_test", "modules/mtp_hf_hints.py")

GGUF_MAGIC = _gguf.GGUF_MAGIC
GGUF_TYPE_STRING = _gguf.GGUF_TYPE_STRING
GGUF_TYPE_UINT32 = _gguf.GGUF_TYPE_UINT32
GgufMetadataError = _gguf.GgufMetadataError
detect_mtp_from_metadata = _gguf.detect_mtp_from_metadata
gguf_header_path_for_filename = _gguf.gguf_header_path_for_filename
read_gguf_metadata = _gguf.read_gguf_metadata
scan_gguf_mtp = _gguf.scan_gguf_mtp
MTP_GGUF_CAVEAT = _hints.MTP_GGUF_CAVEAT
enrich_repo_files_response = _hints.enrich_repo_files_response
filename_suggests_mtp = _hints.filename_suggests_mtp
repo_likely_mtp_converted = _hints.repo_likely_mtp_converted

def _write_string(handle, value: str) -> None:
    encoded = value.encode("utf-8")
    handle.write(struct.pack("<Q", len(encoded)))
    handle.write(encoded)


def _write_kv_uint32(handle, key: str, value: int) -> None:
    _write_string(handle, key)
    handle.write(struct.pack("<I", GGUF_TYPE_UINT32))
    handle.write(struct.pack("<I", value))


def _write_kv_string(handle, key: str, value: str) -> None:
    _write_string(handle, key)
    handle.write(struct.pack("<I", GGUF_TYPE_STRING))
    _write_string(handle, value)


def write_minimal_gguf(path: Path, kvs: list) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<I", GGUF_MAGIC))
        handle.write(struct.pack("<I", 3))
        handle.write(struct.pack("<Q", 0))
        handle.write(struct.pack("<Q", len(kvs)))
        for item in kvs:
            key, vtype, val = item
            if vtype == GGUF_TYPE_STRING:
                _write_kv_string(handle, key, val)
            elif vtype == GGUF_TYPE_UINT32:
                _write_kv_uint32(handle, key, val)
            else:
                raise ValueError(f"unsupported test kv type {vtype}")


def test_mtp_gguf_detected():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qwen-MTP-Q4_K_M.gguf"
        write_minimal_gguf(
            path,
            [
                ("general.architecture", GGUF_TYPE_STRING, "qwen35"),
                ("qwen35.nextn_predict_layers", GGUF_TYPE_UINT32, 3),
            ],
        )
        info = scan_gguf_mtp(path)
        assert info.mtp_capable is True
        assert info.mtp_nextn_layers == 3
        assert info.architecture == "qwen35"


def test_non_mtp_gguf_zero_layers():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qwen-Q4_K_M.gguf"
        write_minimal_gguf(
            path,
            [
                ("general.architecture", GGUF_TYPE_STRING, "qwen35"),
                ("qwen35.nextn_predict_layers", GGUF_TYPE_UINT32, 0),
            ],
        )
        info = scan_gguf_mtp(path)
        assert info.mtp_capable is False
        assert info.mtp_nextn_layers == 0


def test_multipart_header_path_uses_part_one():
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        first = models_dir / "Model-Q6_K-00001-of-00003.gguf"
        second = models_dir / "Model-Q6_K-00002-of-00003.gguf"
        write_minimal_gguf(
            first,
            [
                ("general.architecture", GGUF_TYPE_STRING, "qwen35"),
                ("qwen35.nextn_predict_layers", GGUF_TYPE_UINT32, 2),
            ],
        )
        second.write_bytes(b"GGUF-part2-placeholder")
        resolved = gguf_header_path_for_filename(models_dir, "Model-Q6_K-00002-of-00003.gguf")
        assert resolved == first
        info = scan_gguf_mtp(resolved)
        assert info.mtp_capable is True


def test_corrupt_header_returns_error():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.gguf"
        path.write_bytes(b"NOTGGUF!!!!")
        info = scan_gguf_mtp(path)
        assert info.mtp_capable is False
        assert info.error
        try:
            read_gguf_metadata(path)
            assert False, "expected GgufMetadataError"
        except GgufMetadataError:
            pass


def test_detect_mtp_fallback_key_suffix():
    meta = {"llama.nextn_predict_layers": 1}
    info = detect_mtp_from_metadata(meta)
    assert info.mtp_capable is True
    assert info.mtp_nextn_layers == 1


def test_hf_repo_and_filename_hints():
    assert repo_likely_mtp_converted("unsloth/Qwen3.6-27B-MTP-GGUF")
    assert filename_suggests_mtp("Qwen3.6-27B-MTP-Q4_K_M.gguf")
    payload = enrich_repo_files_response(
        "unsloth/Qwen3.6-27B-MTP-GGUF",
        ["Qwen3.6-27B-Q4_K_M.gguf", "Qwen3.6-27B-MTP-Q6_K.gguf"],
    )
    assert payload["mtp"]["repo_likely_converted"] is True
    assert MTP_GGUF_CAVEAT in payload["mtp"]["caveat"]
    assert "Qwen3.6-27B-MTP-Q6_K.gguf" in payload["file_hints"]


if __name__ == "__main__":
    test_mtp_gguf_detected()
    test_non_mtp_gguf_zero_layers()
    test_multipart_header_path_uses_part_one()
    test_corrupt_header_returns_error()
    test_detect_mtp_fallback_key_suffix()
    test_hf_repo_and_filename_hints()
    print("all tests passed")
