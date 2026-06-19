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
apply_mtp_workload_profile = _deploy.apply_mtp_workload_profile
clamp_mtp_for_model_capability = _deploy.clamp_mtp_for_model_capability
chat_template_kwargs_cli_value = _deploy.chat_template_kwargs_cli_value
mtp_env_from_config = _deploy.mtp_env_from_config
parse_mtp_stats_from_log_line = _log.parse_mtp_stats_from_log_line
MTP_MIN_BUILD_NUMBER = _build.MTP_MIN_BUILD_NUMBER


def test_normalize_mtp_defaults():
    cfg = normalize_mtp_config({})
    assert cfg["enabled"] is False
    assert cfg["draft_n_max"] == 3
    assert cfg["workload_profile"] == "chat"


def test_apply_agent_profile_sets_n8_and_thinking_off():
    cfg = apply_mtp_workload_profile(
        {"mtp": {"workload_profile": "agent", "enabled": True}}
    )
    assert cfg["mtp"]["draft_n_max"] == 8
    assert cfg["server"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert cfg["server"]["cache_reuse"] == 1024
    assert cfg["performance"]["parallel_slots"] == 1


def test_apply_chat_profile_sets_n2():
    cfg = apply_mtp_workload_profile({"mtp": {"workload_profile": "chat"}})
    assert cfg["mtp"]["draft_n_max"] == 2
    assert cfg["mtp"]["enabled"] is True
    assert cfg["server"]["chat_template_kwargs"] == {"preserve_thinking": True}


def test_chat_profile_does_not_override_disabled_mtp():
    cfg = apply_mtp_workload_profile(
        {"mtp": {"workload_profile": "chat", "enabled": False}}
    )
    assert cfg["mtp"]["enabled"] is False
    assert cfg["mtp"]["draft_n_max"] == 2


def test_custom_profile_preserves_user_draft_n_max():
    cfg = apply_mtp_workload_profile(
        {
            "mtp": {
                "workload_profile": "custom",
                "enabled": True,
                "draft_n_max": 5,
            }
        }
    )
    assert cfg["mtp"]["draft_n_max"] == 5


def test_validate_agent_warns_parallel_slots():
    _, warnings = validate_mtp_deployment(
        {
            "mtp": {"workload_profile": "agent", "enabled": True},
            "performance": {"parallel_slots": 4},
        },
        model_mtp_capable=True,
        llamacpp_build_number=9193,
    )
    assert any("Agent workload" in w for w in warnings)
    assert any("overridden to 1" in w for w in warnings)


def test_chat_template_kwargs_cli_value():
    assert (
        chat_template_kwargs_cli_value({"chat_template_kwargs": {"enable_thinking": False}})
        == '{"enable_thinking":false}'
    )


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


def test_clamp_disables_mtp_for_non_capable_model():
    cfg = clamp_mtp_for_model_capability(
        {"mtp": {"enabled": True, "workload_profile": "chat"}},
        model_mtp_capable=False,
    )
    assert cfg["mtp"]["enabled"] is False


def test_clamp_leaves_mtp_when_capable():
    cfg = clamp_mtp_for_model_capability(
        {"mtp": {"enabled": True}},
        model_mtp_capable=True,
    )
    assert cfg["mtp"]["enabled"] is True


def test_validate_rejects_old_build():
    errors, _ = validate_mtp_deployment(
        {"mtp": {"enabled": True}, "performance": {"parallel_slots": 1}},
        model_mtp_capable=True,
        llamacpp_build_number=8250,
    )
    assert any("b9193" in e or str(MTP_MIN_BUILD_NUMBER) in e for e in errors)


def test_validate_warns_parallel_slots():
    _, warnings = validate_mtp_deployment(
        {
            "mtp": {"enabled": True, "workload_profile": "custom"},
            "performance": {"parallel_slots": 4},
        },
        model_mtp_capable=True,
        llamacpp_build_number=9193,
    )
    assert any("parallel_slots" in w for w in warnings)


def test_mtp_env_mirror():
    env = mtp_env_from_config(
        {"mtp": {"enabled": True, "workload_profile": "custom", "draft_n_max": 4}}
    )
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
    test_apply_agent_profile_sets_n8_and_thinking_off()
    test_apply_chat_profile_sets_n2()
    test_custom_profile_preserves_user_draft_n_max()
    test_validate_agent_warns_parallel_slots()
    test_chat_template_kwargs_cli_value()
    test_validate_rejects_non_mtp_model()
    test_validate_rejects_old_build()
    test_validate_warns_parallel_slots()
    test_mtp_env_mirror()
    test_log_parser_acceptance_line()
    print("all tests passed")
