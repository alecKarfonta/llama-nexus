#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.HUIHUI_PROXY | rewrite Cursor/OpenAI chat bodies for
ninfer. Strips llama.cpp-only fields, forces enable_thinking=false so
message.content is non-empty, and reads chunked bodies (Cloudflare/Cursor).
"""

import json
import os
import socket
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("HUIHUI_UPSTREAM", "http://192.168.50.78:8603")
LISTEN_PORT = int(os.environ.get("HUIHUI_PORT", "8788"))
LOG_PATH = os.environ.get("HUIHUI_LOG", "/tmp/huihui_proxy.log")
# [2026-09-05 14:45] | INFO | HUIHUI_PROXY | public model id is uncensored
REAL_MODEL = os.environ.get("HUIHUI_REAL_MODEL", "Huihui-Qwen3.8-27B-abliterated")
PUBLIC_MODEL = os.environ.get("HUIHUI_PUBLIC_MODEL", "uncensored")
MODEL_ALIASES = {None, "", PUBLIC_MODEL, "huihui", REAL_MODEL}

# fields ninfer 400s on or that hide content from Cursor
DROP_KEYS = (
    "chat_template_kwargs",
    "response_format",
    "n",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "store",
    "service_tier",
    "prompt_cache_key",
    "safety_identifier",
    "include",
    "reasoning",
    "reasoning_effort",
    "verbosity",
    "modalities",
    "prediction",
    "web_search_options",
    "stream_options",
)


def _log(msg):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    except Exception:
        pass


def _read_body(handler):
    te = (handler.headers.get("Transfer-Encoding") or "").lower()
    if "chunked" in te:
        parts = []
        while True:
            line = handler.rfile.readline()
            if not line:
                break
            try:
                size = int(line.split(b";", 1)[0].strip(), 16)
            except ValueError:
                break
            if size == 0:
                handler.rfile.readline()
                break
            parts.append(handler.rfile.read(size))
            handler.rfile.read(2)
        return b"".join(parts)
    n = int(handler.headers.get("Content-Length") or 0)
    return handler.rfile.read(n) if n else b""


def _rewrite(raw):
    try:
        body = json.loads(raw) if raw else {}
    except Exception as e:
        _log(f"json-parse-fail {e} raw={raw[:200]!r}")
        return raw
    if not isinstance(body, dict):
        return raw
    for k in DROP_KEYS:
        body.pop(k, None)
    # [2026-09-05 14:45] | INFO | HUIHUI_PROXY.REWRITE | map public model id
    if body.get("model") in MODEL_ALIASES or "model" not in body:
        body["model"] = REAL_MODEL
    # [TIME] | INFO | HUIHUI_PROXY.REWRITE | ninfer: top-level flag, not kwargs
    body["enable_thinking"] = False
    if not body.get("messages"):
        if isinstance(body.get("input"), list):
            body["messages"] = body["input"]
        elif isinstance(body.get("prompt"), str):
            body["messages"] = [{"role": "user", "content": body["prompt"]}]
    # [TIME] | INFO | HUIHUI_PROXY.REWRITE | ninfer rejects any message whose
    # content is empty or missing (even assistant/tool). Cursor legitimately
    # emits such messages in long agent traces. Normalize:
    #   - dict content parts -> extract text parts
    #   - assistant with tool_calls but no text -> "(tool call)" placeholder
    #   - tool result with empty content -> "(no output)"
    #   - anything else empty -> single space
    msgs = body.get("messages")
    dropped = 0
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, list):
                texts = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text") or "")
                    elif isinstance(part, str):
                        texts.append(part)
                c = "".join(texts)
                m["content"] = c
            if c is None or (isinstance(c, str) and c == ""):
                role = m.get("role")
                if role == "assistant" and m.get("tool_calls"):
                    m["content"] = "(tool call)"
                elif role == "tool":
                    m["content"] = "(no output)"
                else:
                    m["content"] = " "
                dropped += 1
    out = json.dumps(body).encode()
    _log(
        f"rewrite keys={sorted(body.keys())} n_msg={len(msgs or [])} "
        f"empty_fixed={dropped} in={len(raw)} out={len(out)} model={body.get('model')!r}"
    )
    return out
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        _log("http " + (fmt % args))

    def do_GET(self):
        self._forward("GET", b"")

    def do_POST(self):
        raw = _read_body(self)
        _log(
            f"POST {self.path} cl={self.headers.get('Content-Length')} "
            f"te={self.headers.get('Transfer-Encoding')} n={len(raw)} "
            f"ua={self.headers.get('User-Agent')}"
        )
        path = self.path.split("?")[0]
        if path.endswith("/chat/completions") or path.endswith("/completions"):
            raw = _rewrite(raw)
        self._forward("POST", raw)

    def do_OPTIONS(self):
        self._forward("OPTIONS", b"")

    def _forward(self, method, raw):
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        accept = self.headers.get("Accept")
        if accept:
            headers["Accept"] = accept
        req = urllib.request.Request(
            UPSTREAM + self.path, data=raw if method == "POST" else None,
            headers=headers, method=method,
        )
        try:
            up = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            up = e
        except Exception as e:
            _log(f"upstream-err {e}")
            self.send_error(502, str(e))
            return

        status = up.code if hasattr(up, "code") else 502
        ctype = up.headers.get("Content-Type", "application/json")
        is_sse = "text/event-stream" in ctype
        _log(f"upstream {status} sse={is_sse} ctype={ctype}")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if is_sse:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                try:
                    up.fp.raw._sock.settimeout(15.0)
                except Exception:
                    pass
                while True:
                    try:
                        chunk = up.read(4096)
                    except (TimeoutError, socket.timeout):
                        ping = b": keepalive\n\n"
                        self.wfile.write(b"%x\r\n" % len(ping) + ping + b"\r\n")
                        self.wfile.flush()
                        continue
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                up.close()
            return
        body = up.read()
        up.close()
        if status >= 400:
            _log(f"err-body {body[:300]!r}")
        # [2026-09-05 14:45] | INFO | HUIHUI_PROXY.FORWARD | expose public model id
        if status < 400 and REAL_MODEL.encode() in body:
            body = body.replace(REAL_MODEL.encode(), PUBLIC_MODEL.encode())
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"[huihui-proxy] upstream={UPSTREAM} port={LISTEN_PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
