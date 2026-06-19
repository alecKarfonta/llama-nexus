"""
Regression tests for Docker GPU attachment in LlamaCPPManager.

Background: requesting specific host GPUs (e.g. host indices 1,3) while also
setting CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES to those *host* indices
caused the container to see only a subset of the attached GPUs. The NVIDIA
Container Toolkit remaps the requested host GPUs to internal indices 0..N-1,
so setting host-index env vars conflicts with the remapping. See
`LlamaCPPManager._docker_gpu_attachment` for the full explanation.
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None

_BACKEND = Path(__file__).resolve().parents[1]


def _make_manager():
    """Load LlamaCPPManager with stubbed heavy dependencies."""
    # Stub dependencies that pull in docker / databases / etc.
    for mod_name in [
        "enhanced_logger",
        "docker",
        "httpx",
        "psutil",
        "huggingface_hub",
        "fastapi",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = ModuleType(mod_name)

    sys.modules["enhanced_logger"].enhanced_logger = MagicMock()
    sys.modules["fastapi"].HTTPException = type("HTTPException", (Exception,), {})
    sys.modules["huggingface_hub"].hf_hub_url = lambda *a, **k: ""
    sys.modules["huggingface_hub"].HfApi = MagicMock

    docker_stub = sys.modules["docker"]
    docker_stub.types = MagicMock()
    # Capture what the manager passes into DeviceRequest.
    captured = {"device_requests": []}

    def _device_request(**kwargs):
        captured["device_requests"].append(kwargs)
        return kwargs

    docker_stub.types.DeviceRequest = _device_request

    # Stub submodules the manager imports at import time.
    for sub in [
        "modules",
        "modules.managers",
        "modules.managers.base",
        "modules.llamacpp_build_info",
        "modules.mtp_deploy",
        "modules.mtp_log_parser",
    ]:
        if sub not in sys.modules:
            m = ModuleType(sub)
            if "." in sub:
                parent_name, leaf = sub.rsplit(".", 1)
                setattr(sys.modules[parent_name], leaf, m)
            sys.modules[sub] = m

    sys.modules["modules.managers.base"].DOCKER_AVAILABLE = False
    sys.modules["modules.managers.base"].docker_client = None
    sys.modules["modules.llamacpp_build_info"].build_info_from_tag_and_version = lambda *a, **k: {}
    sys.modules["modules.llamacpp_build_info"].parse_llguidance_enabled = lambda *a, **k: False
    for fn_name in [
        "apply_mtp_workload_profile",
        "chat_template_kwargs_cli_value",
        "clamp_mtp_for_model_capability",
        "mtp_env_from_config",
        "mtp_enabled_in_config",
        "normalize_mtp_config",
        "validate_mtp_deployment",
    ]:
        setattr(sys.modules["modules.mtp_deploy"], fn_name, lambda *a, **k: (a[0] if a else {}))
    sys.modules["modules.mtp_log_parser"].parse_mtp_stats_from_log_line = lambda *a, **k: None

    spec = importlib.util.spec_from_file_location(
        "_llamacpp_manager_test",
        _BACKEND / "modules" / "managers" / "llamacpp_manager.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    # Instantiate without running __init__'s side effects beyond what we need.
    mgr = object.__new__(module.LlamaCPPManager)
    return mgr, captured, docker_stub


def test_specific_devices_does_not_set_host_index_env():
    """CUDA_VISIBLE_DEVICES must NOT be set to host indices when using DeviceRequest."""
    mgr, captured, docker_stub = _make_manager()
    result = mgr._docker_gpu_attachment("1,3")

    env = result["env"]
    assert "CUDA_VISIBLE_DEVICES" not in env, (
        "Setting CUDA_VISIBLE_DEVICES to host indices conflicts with the toolkit's "
        "internal remapping (host 1,3 -> internal 0,1) and silently shrinks the "
        "visible GPU set."
    )
    assert "NVIDIA_VISIBLE_DEVICES" not in env, (
        "Same remapping conflict as CUDA_VISIBLE_DEVICES."
    )
    # The toolkit still needs these:
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    # DeviceRequest was created with the host indices.
    assert len(result["device_requests"]) == 1
    dr = result["device_requests"][0]
    assert dr["driver"] == "nvidia"
    assert dr["device_ids"] == ["1", "3"]

    # CLI form for --gpus uses host indices (quoted, as Docker requires).
    assert result["gpus_cli"] == '"device=1,3"'


def test_all_devices_path_unchanged():
    """'all' must not set host-index env vars and must use count=-1."""
    mgr, _, _ = _make_manager()
    result = mgr._docker_gpu_attachment("all")

    assert "CUDA_VISIBLE_DEVICES" not in result["env"]
    assert "NVIDIA_VISIBLE_DEVICES" not in result["env"]
    assert result["gpus_cli"] == "all"
    assert len(result["device_requests"]) == 1


def test_single_device_also_skips_env():
    """Even a single host index must not be propagated as CUDA_VISIBLE_DEVICES."""
    mgr, _, _ = _make_manager()
    result = mgr._docker_gpu_attachment("0")
    assert "CUDA_VISIBLE_DEVICES" not in result["env"]
    assert "NVIDIA_VISIBLE_DEVICES" not in result["env"]
    assert result["device_requests"][0]["device_ids"] == ["0"]
    assert result["gpus_cli"] == '"device=0"'


if __name__ == "__main__":
    test_specific_devices_does_not_set_host_index_env()
    test_all_devices_path_unchanged()
    test_single_device_also_skips_env()
    print("all GPU attachment tests passed")
