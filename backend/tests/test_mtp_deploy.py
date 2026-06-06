"""Tests for MTP deployment validation and log parsing."""
import importlib.util
import sys
import types
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_build = _load("modules/llamacpp_build_info.py", "llamacpp_build_info_test")
_log = _load("modules/mtp_log_parser.py", "mtp_log_parser_test")
# Avoid importing backend/modules/__init__.py when loading mtp_deploy
if "modules" not in sys.modules:
    sys.modules["modules"] = types.ModuleType("modules")
sys.modules["modules.llamacpp_build_info"] = _build
_deploy = _load("modules/mtp_deploy.py", "mtp_deploy_test")

validate_mtp_deployment = _deploy.validate_mtp_deployment
normalize_mtp_config = _deploy.normalize_mtp_config
mtp_env_from_config = _deploy.mtp_env_from_config
parse_mtp_stats_from_log_line = _log.parse_mtp_stats_from_log_line
MTP_MIN_BUILD_NUMBER = _build.MTP_MIN_BUILD_NUMBER


def test_normalize_mtp_defaults():
    cfg = normalize_mtp_config({})
    assert cfg["enabled"] is False
    assert cfg["draft_n_max"] == 3


def test_validate_rejects_non_mtp_model():
    config = {
        "mtp": {"enabled": True},
        "performance": {"parallel_slots": 1},
    }
    errors, warnings = validate_mtp_deployment(
        config,
        model_mtp_capable=False,
        llamacpp_build_number=9193,
    )
    assert any("mtp_capable" in e for e in errors)
    assert warnings == []


def test_validate_rejects_old_build():
    errors, _ = validate_mtp_deployment(
        {"mtp": {"enabled": True}, "performance": {"parallel_slots": 1}},
        model_mtp_capable=True,
        llamacpp_build_number=8250,
    )
    assert any("b9193" in e or str(MTP_MIN_BUILD_NUMBER) in e for e in errors)


def test_validate_warns_parallel_slots():
    _, warnings = validate_mtp_deployment(
        {"mtp": {"enabled": True}, "performance": {"parallel_slots": 4}},
        model_mtp_capable=True,
        llamacpp_build_number=9193,
    )
    assert any("parallel_slots" in w for w in warnings)


def test_mtp_env_mirror():
    env = mtp_env_from_config({"mtp": {"enabled": True, "draft_n_max": 4}})
    assert env["MTP_ENABLED"] == "true"
    assert env["MTP_DRAFT_N_MAX"] == "4"


def test_log_parser_acceptance_line():
    line = "draft acceptance rate = 0.76482 ( 3483 accepted / 4554 generated)"
    stats = parse_mtp_stats_from_log_line(line)
    assert stats is not None
    assert stats["acceptance_rate"] == 0.76482
    assert stats["tokens_accepted"] == 3483
    assert stats["tokens_drafted"] == 4554


if __name__ == "__main__":
    test_normalize_mtp_defaults()
    test_validate_rejects_non_mtp_model()
    test_validate_rejects_old_build()
    test_validate_warns_parallel_slots()
    test_mtp_env_mirror()
    test_log_parser_acceptance_line()
    print("all tests passed")
