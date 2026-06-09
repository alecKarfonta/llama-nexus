#!/usr/bin/env python3
"""
MTP / Phase 0 integration smoke tests.

Validates a running llama-nexus stack and repo unit tests. Use after rebuilding
llamacpp-api (b9193+) and backend-api with MTP changes.

Usage:
  python3 scripts/smoke_mtp.py
  python3 scripts/smoke_mtp.py --unit-only
  BACKEND_URL=http://127.0.0.1:8700 LLAMACPP_URL=http://127.0.0.1:8600 python3 scripts/smoke_mtp.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

_REPO = Path(__file__).resolve().parent.parent
MTP_MIN_BUILD = 9193


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    detail: str = ""


def _http_json(
    method: str,
    url: str,
    *,
    body: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 60.0,
) -> tuple[int, Optional[dict], str]:
    hdrs = {"Accept": "application/json", **(headers or {})}
    data = None
    if body is not None:
        hdrs["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), raw
            except json.JSONDecodeError:
                return resp.status, None, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw), raw
        except json.JSONDecodeError:
            return exc.code, None, raw
    except urllib.error.URLError as exc:
        return 0, None, str(exc.reason)


def run_unit_tests() -> Check:
    tests = [
        _REPO / "backend/tests/test_llamacpp_build_info.py",
        _REPO / "backend/tests/test_gguf_mtp.py",
        _REPO / "backend/tests/test_mtp_deploy.py",
        _REPO / "backend/tests/test_mtp_bench_lib.py",
    ]
    failed = []
    for path in tests:
        r = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(_REPO),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            failed.append(path.name)
    if failed:
        return Check("unit_tests", "FAIL", f"failed: {', '.join(failed)}")
    return Check("unit_tests", "PASS", f"{len(tests)} modules OK")


def check_backend_health(backend_url: str) -> Check:
    code, data, raw = _http_json("GET", f"{backend_url.rstrip('/')}/health")
    if code == 200 and isinstance(data, dict) and data.get("status") == "up":
        return Check("backend_health", "PASS", str(data.get("service", {})))
    return Check("backend_health", "FAIL", f"HTTP {code}: {raw[:200]}")


def check_service_status(backend_url: str) -> List[Check]:
    checks: List[Check] = []
    code, data, raw = _http_json("GET", f"{backend_url.rstrip('/')}/api/v1/service/status")
    if code != 200 or not isinstance(data, dict):
        checks.append(Check("service_status", "FAIL", f"HTTP {code}: {raw[:200]}"))
        return checks

    running = data.get("running")
    checks.append(
        Check("service_status", "PASS" if running else "WARN", f"running={running}")
    )

    build_num = data.get("llamacpp_build_number")
    mtp_supported = data.get("mtp_supported")
    build_info = data.get("llamacpp_build") or {}

    if build_num is None and not build_info:
        checks.append(
            Check(
                "build_version_api",
                "WARN",
                "llamacpp_build not in status — rebuild/restart backend-api with MTP Phase 0 code",
            )
        )
    else:
        num = build_num or build_info.get("build_number")
        if num is not None and int(num) >= MTP_MIN_BUILD:
            checks.append(
                Check(
                    "build_version_api",
                    "PASS",
                    f"build_number={num}, mtp_supported={mtp_supported}",
                )
            )
        else:
            checks.append(
                Check(
                    "build_version_api",
                    "FAIL",
                    f"build_number={num} (need >={MTP_MIN_BUILD})",
                )
            )

    health = data.get("llamacpp_health") or {}
    if running and health.get("healthy"):
        checks.append(Check("llamacpp_health_proxy", "PASS", "healthy"))
    elif running:
        checks.append(Check("llamacpp_health_proxy", "WARN", str(health)))
    return checks


def check_llamacpp_health(llamacpp_url: str) -> Check:
    code, data, raw = _http_json("GET", f"{llamacpp_url.rstrip('/')}/health")
    if code == 200:
        return Check("llamacpp_health", "PASS", raw[:80])
    return Check("llamacpp_health", "FAIL", f"HTTP {code}: {raw[:120]}")


def check_chat_completions(llamacpp_url: str, api_key: str) -> Check:
    code, data, raw = _http_json(
        "POST",
        f"{llamacpp_url.rstrip('/')}/v1/chat/completions",
        body={
            "messages": [{"role": "user", "content": "Reply with exactly: smoke ok"}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if code != 200 or not isinstance(data, dict):
        return Check("chat_completions", "FAIL", f"HTTP {code}: {raw[:300]}")
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    fingerprint = data.get("system_fingerprint", "")
    build_match = re.search(r"b(\d+)", fingerprint)
    detail = f"content={content!r}, fingerprint={fingerprint}"
    if "smoke ok" not in content.lower():
        return Check("chat_completions", "FAIL", detail)
    if build_match:
        bn = int(build_match.group(1))
        if bn < MTP_MIN_BUILD:
            return Check(
                "chat_completions",
                "WARN",
                f"{detail} — inference build b{bn} < b{MTP_MIN_BUILD}; rebuild llamacpp-api",
            )
    return Check("chat_completions", "PASS", detail)


def check_embeddings(embed_url: str, api_key: str) -> Check:
    code, data, raw = _http_json(
        "POST",
        f"{embed_url.rstrip('/')}/v1/embeddings",
        body={"input": "smoke test"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if code != 200 or not isinstance(data, dict):
        return Check("embeddings", "FAIL", f"HTTP {code}: {raw[:200]}")
    emb = (data.get("data") or [{}])[0].get("embedding") or []
    if len(emb) < 8:
        return Check("embeddings", "FAIL", f"unexpected embedding len={len(emb)}")
    return Check("embeddings", "PASS", f"dims={len(emb)}")


def check_models_registry(backend_url: str) -> Check:
    code, data, raw = _http_json("GET", f"{backend_url.rstrip('/')}/v1/models")
    if code != 200 or not isinstance(data, dict):
        return Check("models_registry", "FAIL", f"HTTP {code}: {raw[:200]}")
    items = data.get("data") or []
    if not items:
        return Check("models_registry", "WARN", "no local models listed")
    has_mtp_field = any(
        "mtp_capable" in m or "mtpCapable" in m for m in items if isinstance(m, dict)
    )
    if has_mtp_field:
        mtp_models = [
            m.get("name")
            for m in items
            if isinstance(m, dict) and (m.get("mtp_capable") or m.get("mtpCapable"))
        ]
        return Check(
            "models_registry",
            "PASS",
            f"{len(items)} models, {len(mtp_models)} mtp_capable",
        )
    return Check(
        "models_registry",
        "WARN",
        f"{len(items)} models but no mtp_capable field — restart backend-api with Phase 1 code",
    )


def check_mtp_validate_endpoint(backend_url: str) -> Check:
    """POST invalid MTP config must be rejected when backend has Phase 2 code."""
    payload = {
        "config": {
            "model": {"name": "nonexistent-mtp-test", "variant": "Q4_K_M"},
            "mtp": {"enabled": True, "draft_n_max": 3, "draft_p_min": 0.75},
            "performance": {"parallel_slots": 1},
        }
    }
    code, data, raw = _http_json(
        "POST",
        f"{backend_url.rstrip('/')}/api/v1/service/config/validate",
        body=payload,
    )
    if code == 404:
        return Check("mtp_config_validate", "SKIP", "endpoint not found")
    if code != 200 or not isinstance(data, dict):
        return Check("mtp_config_validate", "WARN", f"HTTP {code}: {raw[:200]}")
    if data.get("valid") is False and data.get("errors"):
        return Check("mtp_config_validate", "PASS", f"errors={data['errors'][:1]}")
    return Check(
        "mtp_config_validate",
        "WARN",
        f"expected validation failure for fake MTP model, got valid={data.get('valid')}",
    )


def check_container_mtp_flags(container: str) -> Check:
    try:
        r = subprocess.run(
            ["docker", "exec", container, "llama-server", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        return Check("container_mtp_flags", "SKIP", str(exc))
    help_text = (r.stdout or "") + (r.stderr or "")
    if "draft-mtp" in help_text:
        return Check("container_mtp_flags", "PASS", "llama-server lists draft-mtp")
    if "spec-type" in help_text and "draft-mtp" not in help_text:
        return Check(
            "container_mtp_flags",
            "FAIL",
            "spec-type present but draft-mtp missing — rebuild llamacpp-api from Dockerfile b9193",
        )
    return Check("container_mtp_flags", "WARN", "could not detect spec-type in --help")


def check_container_build_tag(container: str) -> Check:
    try:
        r = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "cat",
                "/etc/llama-nexus/llamacpp-build-tag",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        return Check("container_build_tag", "SKIP", str(exc))
    tag = (r.stdout or "").strip()
    if not tag:
        return Check(
            "container_build_tag",
            "WARN",
            "/etc/llama-nexus/llamacpp-build-tag missing — image predates Phase 0 labeling",
        )
    m = re.search(r"(\d+)", tag)
    if m and int(m.group(1)) >= MTP_MIN_BUILD:
        return Check("container_build_tag", "PASS", f"tag={tag}")
    return Check(
        "container_build_tag",
        "FAIL",
        f"tag={tag or 'empty'} (need >={MTP_MIN_BUILD})",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MTP / Phase 0 smoke tests")
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--backend-url", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8700"))
    parser.add_argument("--llamacpp-url", default=os.environ.get("LLAMACPP_URL", "http://127.0.0.1:8600"))
    parser.add_argument("--embed-url", default=os.environ.get("EMBED_URL", "http://127.0.0.1:8602"))
    parser.add_argument("--llamacpp-api-key", default=os.environ.get("LLAMACPP_API_KEY", "placeholder-api-key"))
    parser.add_argument("--embed-api-key", default=os.environ.get("EMBED_API_KEY", "llamacpp-embed"))
    parser.add_argument("--container", default=os.environ.get("LLAMACPP_CONTAINER", "llamacpp-api"))
    args = parser.parse_args()

    checks: List[Check] = [run_unit_tests()]
    if not args.unit_only:
        checks.extend(
            [
                check_backend_health(args.backend_url),
                *check_service_status(args.backend_url),
                check_llamacpp_health(args.llamacpp_url),
                check_chat_completions(args.llamacpp_url, args.llamacpp_api_key),
                check_embeddings(args.embed_url, args.embed_api_key),
                check_models_registry(args.backend_url),
                check_mtp_validate_endpoint(args.backend_url),
                check_container_build_tag(args.container),
                check_container_mtp_flags(args.container),
            ]
        )

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           MTP / Phase 0 Smoke Tests                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    icons = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "SKIP": "⏭️ "}
    fails = warns = 0
    for c in checks:
        icon = icons.get(c.status, "?")
        print(f"{icon} {c.name}: {c.status}")
        if c.detail:
            print(f"   {c.detail}")
        if c.status == "FAIL":
            fails += 1
        elif c.status == "WARN":
            warns += 1

    print()
    print(f"Summary: {sum(1 for c in checks if c.status == 'PASS')} pass, {warns} warn, {fails} fail")
    if fails:
        print()
        print("Rebuild hint:")
        print("  docker compose --profile extra build llamacpp-api backend-api")
        print("  docker compose --profile extra up -d llamacpp-api backend-api")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
