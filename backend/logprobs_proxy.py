#!/usr/bin/env python3
"""
LLM Completions Proxy for lm-eval.

Translates between llama.cpp server logprobs format and the OpenAI format
that lm-eval's local-completions model type expects.

Uses aiohttp for async upstream requests to handle high concurrency without
thread overhead or connection bottlenecks.
"""

import json
import os
import asyncio
from aiohttp import web, ClientSession, TCPConnector
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

UPSTREAM_URL = os.environ.get("LLAMACPP_API_URL", "http://llamacpp-api:8080")
API_KEY = os.environ.get("LM_EVAL_API_KEY", "placeholder-api-key")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8888"))

# Shared session with connection pooling
_session: ClientSession = None


async def get_session() -> ClientSession:
    global _session
    if _session is None or _session.closed:
        connector = TCPConnector(
            limit=50,
            limit_per_host=50,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )
        _session = ClientSession(connector=connector)
    return _session


def translate_logprobs(llamacpp_logprobs):
    """Translate llama.cpp logprobs format to OpenAI format expected by lm-eval."""
    if not llamacpp_logprobs:
        return None

    if "token_logprobs" in llamacpp_logprobs:
        return llamacpp_logprobs

    content = llamacpp_logprobs.get("content")
    if not content:
        return llamacpp_logprobs

    tokens = []
    token_logprobs = []
    top_logprobs = []

    for entry in content:
        tokens.append(entry.get("token", ""))
        token_logprobs.append(entry.get("logprob"))

        top_entries = entry.get("top_logprobs", [])
        top_dict = {}
        for te in top_entries:
            top_dict[te.get("token", "")] = te.get("logprob", 0.0)
        top_logprobs.append(top_dict if top_dict else {})

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
    }


def translate_response(resp_data):
    """Translate logprobs in all response choices."""
    if "choices" in resp_data:
        for choice in resp_data["choices"]:
            if "logprobs" in choice and choice["logprobs"]:
                choice["logprobs"] = translate_logprobs(choice["logprobs"])
            elif "logprobs" in choice and choice["logprobs"] is None:
                choice["logprobs"] = {
                    "tokens": [],
                    "token_logprobs": [],
                    "top_logprobs": [],
                }
    return resp_data


async def handle_post(request: web.Request):
    body = await request.read()

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    target_url = f"{UPSTREAM_URL.rstrip('/')}{request.path}"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    session = await get_session()
    try:
        async with session.post(
            target_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            resp_data = await resp.json(content_type=None)
    except Exception as e:
        return web.Response(status=502, text=f"Upstream error: {e}")

    if resp.status != 200:
        return web.Response(
            status=resp.status,
            content_type="application/json",
            text=json.dumps(resp_data),
        )

    resp_data = translate_response(resp_data)
    return web.Response(
        status=200,
        content_type="application/json",
        text=json.dumps(resp_data),
    )


async def handle_get(request: web.Request):
    target_url = f"{UPSTREAM_URL.rstrip('/')}{request.path}"
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    session = await get_session()
    try:
        async with session.get(
            target_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            body = await resp.read()
            return web.Response(status=resp.status, content_type="application/json", body=body)
    except Exception as e:
        return web.Response(status=502, text=f"Upstream error: {e}")


async def on_cleanup(app: web.Application):
    global _session
    if _session and not _session.closed:
        await _session.close()


def main():
    print(f"[proxy] Starting async logprobs translator proxy on port {PROXY_PORT}")
    print(f"[proxy] Upstream: {UPSTREAM_URL}")

    app = web.Application()
    app.on_cleanup.append(on_cleanup)
    app.router.add_post("/{path:.*}", handle_post)
    app.router.add_get("/{path:.*}", handle_get)

    web.run_app(app, host="0.0.0.0", port=PROXY_PORT, print=None)


if __name__ == "__main__":
    import aiohttp
    main()
