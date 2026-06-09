"""Unit tests for llama.cpp build tag parsing (MTP gating)."""
import importlib.util
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _load_build_info():
    spec = importlib.util.spec_from_file_location(
        "llamacpp_build_info_test",
        _BACKEND / "modules" / "llamacpp_build_info.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


_bi = _load_build_info()
MTP_MIN_BUILD_NUMBER = _bi.MTP_MIN_BUILD_NUMBER
build_info_from_tag_and_version = _bi.build_info_from_tag_and_version
mtp_supported_for_build = _bi.mtp_supported_for_build
parse_llamacpp_build_number = _bi.parse_llamacpp_build_number
parse_llguidance_enabled = _bi.parse_llguidance_enabled


def test_parse_build_tag():
    assert parse_llamacpp_build_number("b9193") == 9193
    assert parse_llamacpp_build_number("b8250") == 8250
    assert parse_llamacpp_build_number("build 9496") == 9496
    assert parse_llamacpp_build_number("unknown") is None


def test_mtp_supported_threshold():
    assert mtp_supported_for_build(9193) is True
    assert mtp_supported_for_build(9192) is False
    assert mtp_supported_for_build(None) is False
    assert MTP_MIN_BUILD_NUMBER == 9193


def test_build_info_payload():
    info = build_info_from_tag_and_version(
        "b9193",
        "version: 1a68ec937 (b9193)",
    )
    assert info["tag"] == "b9193"
    assert info["build_number"] == 9193
    assert info["mtp_supported"] is True
    assert info["grammar_gbnf_supported"] is True
    assert info["llguidance_enabled"] is False
    assert info["cli_version"].startswith("version:")


def test_llguidance_enabled_flag():
    assert parse_llguidance_enabled("true") is True
    assert parse_llguidance_enabled("false") is False
    info = build_info_from_tag_and_version("b9193", llguidance_enabled=True)
    assert info["llguidance_enabled"] is True


if __name__ == "__main__":
    test_parse_build_tag()
    test_mtp_supported_threshold()
    test_build_info_payload()
    test_llguidance_enabled_flag()
    print("all tests passed")
