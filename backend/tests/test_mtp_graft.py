"""Tests for MTP graft naming and preflight."""
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


_gguf = _load_module("gguf_metadata_graft_test", "modules/gguf_metadata.py")
_naming = _load_module("gguf_model_naming_test", "modules/gguf_model_naming.py")
_graft = _load_module("mtp_graft_test", "modules/mtp_graft.py")
_hints = _load_module("mtp_hf_hints_graft_test", "modules/mtp_hf_hints.py")

GGUF_MAGIC = _gguf.GGUF_MAGIC
GGUF_TYPE_STRING = _gguf.GGUF_TYPE_STRING
GGUF_TYPE_UINT32 = _gguf.GGUF_TYPE_UINT32
parse_name_variant = _naming.parse_name_variant
plan_graft_output_filename = _naming.plan_graft_output_filename
plan_graft_deploy_identity = _naming.plan_graft_deploy_identity
preflight_graft = _graft.preflight_graft
suggest_donor_for_target = _hints.suggest_donor_for_target


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
        for key, vtype, val in kvs:
            if vtype == GGUF_TYPE_STRING:
                _write_kv_string(handle, key, val)
            elif vtype == GGUF_TYPE_UINT32:
                _write_kv_uint32(handle, key, val)
            else:
                raise ValueError(f"unsupported test kv type {vtype}")


def test_parse_mtp_deploy_filename():
    name, variant = parse_name_variant("Qwen3.6-35B-A3B-uncensored-heretic-MTP-Q6_K.gguf")
    assert name == "Qwen3.6-35B-A3B-uncensored-heretic-MTP"
    assert variant == "Q6_K"


def test_plan_graft_output_filename():
    assert (
        plan_graft_output_filename("Qwen3.6-35B-A3B-uncensored-heretic", "Q6_K")
        == "Qwen3.6-35B-A3B-uncensored-heretic-MTP-Q6_K.gguf"
    )
    deploy_name, deploy_variant = plan_graft_deploy_identity(
        "Qwen3.6-35B-A3B-uncensored-heretic", "Q6_K"
    )
    assert deploy_name == "Qwen3.6-35B-A3B-uncensored-heretic-MTP"
    assert deploy_variant == "Q6_K"


def test_preflight_graft_block_layout():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "target-Q6_K.gguf"
        donor = root / "donor-Q8_0.gguf"
        write_minimal_gguf(
            target,
            [
                ("general.architecture", GGUF_TYPE_STRING, "qwen35moe"),
                ("qwen35moe.block_count", GGUF_TYPE_UINT32, 40),
            ],
        )
        write_minimal_gguf(
            donor,
            [
                ("general.architecture", GGUF_TYPE_STRING, "qwen35moe"),
                ("qwen35moe.block_count", GGUF_TYPE_UINT32, 41),
                ("qwen35moe.nextn_predict_layers", GGUF_TYPE_UINT32, 1),
            ],
        )
        result = preflight_graft(
            target,
            donor,
            target_name="target",
            target_variant="Q6_K",
        )
        assert result.target_block_count == 40
        assert result.donor_block_count == 41
        assert result.donor_nextn == 1
        assert result.planned_output_filename == "target-MTP-Q6_K.gguf"
        assert result.ok is False
        assert any("No donor tensors" in e for e in result.errors)


def test_preflight_rejects_mtp_target():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "target-MTP-Q6_K.gguf"
        donor = root / "donor-Q8_0.gguf"
        write_minimal_gguf(
            target,
            [
                ("general.architecture", GGUF_TYPE_STRING, "qwen35moe"),
                ("qwen35moe.block_count", GGUF_TYPE_UINT32, 40),
                ("qwen35moe.nextn_predict_layers", GGUF_TYPE_UINT32, 1),
            ],
        )
        write_minimal_gguf(
            donor,
            [
                ("general.architecture", GGUF_TYPE_STRING, "qwen35moe"),
                ("qwen35moe.block_count", GGUF_TYPE_UINT32, 41),
                ("qwen35moe.nextn_predict_layers", GGUF_TYPE_UINT32, 1),
            ],
        )
        result = preflight_graft(
            target,
            donor,
            target_name="target-MTP",
            target_variant="Q6_K",
        )
        assert result.ok is False
        assert any("already has MTP" in e for e in result.errors)


def test_suggest_donor_for_qwen_moe():
    data = suggest_donor_for_target(target_name="Qwen3.6-35B-A3B-uncensored-heretic")
    assert data["family"] == "qwen3.6-moe"
    assert len(data["suggested_donors"]) >= 1
    assert "unsloth/Qwen3.6-35B-A3B-MTP-GGUF" in data["suggested_donors"][0]["repo_id"]
