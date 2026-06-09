"""Tests for chat completion proxy passthrough contract."""
import importlib.util
import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_spec = importlib.util.spec_from_file_location(
    "chat_proxy_test", _BACKEND / "modules/chat_proxy.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CHAT_PROXY_PASSTHROUGH_KEYS = _mod.CHAT_PROXY_PASSTHROUGH_KEYS
parse_chat_proxy_body = _mod.parse_chat_proxy_body
chat_proxy_body_unchanged = _mod.chat_proxy_body_unchanged
missing_passthrough_keys = _mod.missing_passthrough_keys


def test_passthrough_keys_include_tools_and_grammar():
    assert "tools" in CHAT_PROXY_PASSTHROUGH_KEYS
    assert "tool_choice" in CHAT_PROXY_PASSTHROUGH_KEYS
    assert "response_format" in CHAT_PROXY_PASSTHROUGH_KEYS
    assert "grammar" in CHAT_PROXY_PASSTHROUGH_KEYS


def test_proxy_body_unchanged():
    body = json.dumps(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "weather"}}],
            "tool_choice": "auto",
            "response_format": {"type": "json_schema", "json_schema": {"schema": {}}},
            "grammar": "root ::= object",
        }
    ).encode()
    forwarded = chat_proxy_body_unchanged(body)
    assert forwarded == body
    parsed = parse_chat_proxy_body(forwarded)
    for key in CHAT_PROXY_PASSTHROUGH_KEYS:
        assert key in parsed


def test_missing_passthrough_keys_detects_drops():
    original = {
        "tools": [{"type": "function"}],
        "grammar": "root ::= object",
        "messages": [],
    }
    forwarded = {"messages": []}
    missing = missing_passthrough_keys(original, forwarded)
    assert "tools" in missing
    assert "grammar" in missing


if __name__ == "__main__":
    test_passthrough_keys_include_tools_and_grammar()
    test_proxy_body_unchanged()
    test_missing_passthrough_keys_detects_drops()
    print("all tests passed")
