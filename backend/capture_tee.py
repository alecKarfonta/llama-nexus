#!/usr/bin/env python3
"""
[TIME] | INFO | CAPTURE_TEE.MODULE | streaming reverse proxy in front of
llama-server that tees every completed /v1 exchange to a JSONL capture file
for the Cursor-specific MTP draft-head dataset (self-distillation).

Design constraints:
- SSE pass-through must not buffer (TTFT is the metric this rig was tuned for):
  chunks are forwarded as they arrive; the capture record is assembled after
  the stream ends, so capture never adds latency.
- Token-exactness: after a completion, re-derive the exact server-side token
  sequence via /apply-template + /tokenize so training sequences match what
  the server actually processed. template_ok=false marks unverifiable records.
- Only OpenAI-ish chat/completion endpoints are captured; everything else is
  blind-forwarded.
"""

import json
import os
import select
import socket
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("TEE_UPSTREAM", "http://llamacpp:8080")
LISTEN_PORT = int(os.environ.get("TEE_PORT", "8601"))
CAPTURE_DIR = os.environ.get("TEE_CAPTURE_DIR", "/data/captures")
# bearer keys accepted upstream (forwarded verbatim); logged as key_id label
KEY_LABELS = json.loads(os.environ.get("TEE_KEY_LABELS", "{}"))
# [TIME] | INFO | CAPTURE_TEE.UPSTREAM_AUTH | clients present the public
# nginx-issued key (sk-mlapi-cursor-*), but llama-server enforces its own
# --api-key; substitute it on the upstream leg. Capture records keep the
# ORIGINAL client key id, never the upstream secret.
UPSTREAM_API_KEY = os.environ.get("TEE_UPSTREAM_API_KEY", "")
MAX_CAPTURE_BODY = int(os.environ.get("TEE_MAX_CAPTURE_BODY", str(2 * 1024 * 1024)))
# [TIME] | INFO | CAPTURE_TEE.COMPACT_BUDGET | Cursor cannot be told how much
# context to send (no UI for custom endpoints) and abandons a request after
# ~120s. Cold prefill is quadratic in depth on this rig, so ~120s of patience
# runs out near 40k tokens. Cap the prompt below that instead of letting a
# 128k dump prefill for 14 minutes into a client that already gave up.
MAX_PROMPT_TOKENS = int(os.environ.get("TEE_MAX_PROMPT_TOKENS", "48000"))
COMPACT_ENABLED = os.environ.get("TEE_COMPACT", "1") not in ("0", "false", "no")
ELISION_NOTE = (
    "[context compacted: {n} earlier message(s) from the middle of this "
    "conversation were dropped to fit the model's prompt budget. The system "
    "prompt and the most recent turns are intact.]"
)
# endpoints whose exchanges become training samples
CAPTURE_PATHS = {"/v1/chat/completions", "/v1/completions"}

_stats = {"requests": 0, "captured": 0, "dropped": 0, "dropped_reasons": {}}
_stats_lock = threading.Lock()


def _bump_dropped(reason):
    with _stats_lock:
        _stats["dropped"] += 1
        _stats["dropped_reasons"][reason] = _stats["dropped_reasons"].get(reason, 0) + 1


def _key_id(auth_header):
    if not auth_header:
        return "none"
    token = auth_header.removeprefix("Bearer ").strip()
    return KEY_LABELS.get(token, token[:12] + "..." if token else "none")


def _capture_path():
    day = time.strftime("%Y-%m-%d")
    return os.path.join(CAPTURE_DIR, f"captures-{day}.jsonl")


def _write_record(record):
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    with open(_capture_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _upstream_auth():
    """Authorization header value for server-side helper calls."""
    return f"Bearer {UPSTREAM_API_KEY}" if UPSTREAM_API_KEY else ""


def _upstream_call(path, payload, auth):
    req = urllib.request.Request(
        UPSTREAM + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": auth},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _token_snapshot(handler_auth, body):
    """
    [TIME] | INFO | CAPTURE_TEE.TOKEN_SNAPSHOT | derive the exact token
    sequence the server saw: apply the chat template, then tokenize. Returns
    (prompt_tokens, completion_tokens, template_ok). Falls back to plain
    text tokenize on any failure — records marked template_ok=false so the
    dataset builder can exclude them.
    """
    try:
        if "messages" in body:
            tmpl_payload = {"messages": body["messages"]}
            if body.get("tools"):
                tmpl_payload["tools"] = body["tools"]
            tmpl = _upstream_call("/apply-template", tmpl_payload, _upstream_auth())
            prompt_text = tmpl.get("prompt", "")
        elif "prompt" in body:
            prompt_text = body["prompt"]
        else:
            return None, None, False
        prompt_ids = _upstream_call("/tokenize", {"content": prompt_text}, _upstream_auth()).get("tokens", [])
        return prompt_ids, None, bool(prompt_ids)
    except Exception:
        return None, None, False


def _record_model_size(model, tokens):
    name = model or "(unset)"
    with _stats_lock:
        by_model = _stats.setdefault("by_model", {})
        e = by_model.setdefault(
            name, {"requests": 0, "max_prompt_tokens": 0, "last_prompt_tokens": 0}
        )
        e["requests"] += 1
        e["last_prompt_tokens"] = tokens
        e["max_prompt_tokens"] = max(e["max_prompt_tokens"], tokens)


def _count_prompt_tokens(messages, tools, auth):
    """Exact upstream token count for a candidate message list, or None."""
    try:
        payload = {"messages": messages}
        if tools:
            payload["tools"] = tools
        text = _upstream_call("/apply-template", payload, auth).get("prompt", "")
        toks = _upstream_call("/tokenize", {"content": text}, auth).get("tokens", [])
        return len(toks) or None
    except Exception:
        return None


def _msg_text(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in c
        )
    return "" if c is None else str(c)


def _compact_messages(messages, tools, auth, budget=MAX_PROMPT_TOKENS):
    """
    [TIME] | INFO | CAPTURE_TEE.COMPACT | Shrink an oversized chat history to
    fit ``budget`` prompt tokens. Keeps every leading system message and as
    many of the most recent turns as fit, dropping from the middle outward.

    Tool-call pairing matters: a ``tool`` result whose originating assistant
    message was dropped is an orphan and upstream rejects it, so the kept
    window is advanced past any leading orphan tool results.

    Returns (messages, info, incoming_tokens); info is None when nothing was
    changed. incoming_tokens is the pre-compaction size (None if unmeasurable).
    """
    total = _count_prompt_tokens(messages, tools, auth)
    if total is None or total <= budget:
        return messages, None, total

    head = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        head.append(messages[idx])
        idx += 1
    rest = messages[idx:]
    if not rest:
        return messages, None, total

    # char-based first guess, then verify/shrink exactly (chars per token ~3.5)
    approx_budget_chars = max(0, budget - _count_prompt_tokens(head, tools, auth) or 0) * 3
    keep = []
    used = 0
    for m in reversed(rest):
        c = len(_msg_text(m)) + 64
        if keep and used + c > approx_budget_chars:
            break
        keep.append(m)
        used += c
    keep.reverse()
    if not keep:
        keep = rest[-1:]

    def drop_orphan_tools(seq):
        while seq and seq[0].get("role") == "tool":
            seq = seq[1:]
        return seq

    keep = drop_orphan_tools(keep) or rest[-1:]

    # exact verification; shrink the window until it fits
    for _ in range(8):
        note = dict(head[-1]) if head else None
        candidate_head = head[:-1] + [note] if note else []
        dropped = len(rest) - len(keep)
        if note is not None and dropped > 0:
            note["content"] = _msg_text(note) + "\n\n" + ELISION_NOTE.format(n=dropped)
        candidate = candidate_head + keep
        n = _count_prompt_tokens(candidate, tools, auth)
        if n is None:
            return messages, None, total
        if n <= budget:
            return candidate, {
                "original_tokens": total,
                "final_tokens": n,
                "dropped_messages": dropped,
                "kept_messages": len(candidate),
            }, total
        if len(keep) <= 1:
            break
        keep = drop_orphan_tools(keep[max(1, len(keep) // 4):]) or keep[-1:]

    return messages, {
        "original_tokens": total,
        "final_tokens": None,
        "dropped_messages": len(rest) - len(keep),
        "kept_messages": len(head) + len(keep),
        "note": "could not reach budget",
    }, total


class TeeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------
    def log_message(self, fmt, *args):
        pass  # keep stdout clean; stats live on /tee/stats

    def do_GET(self):
        if self.path == "/tee/stats":
            with _stats_lock:
                body = json.dumps(_stats).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._forward(method="GET")

    def do_POST(self):
        self._forward(method="POST")

    def do_OPTIONS(self):
        self._forward(method="OPTIONS")

    # -- core --------------------------------------------------------------
    def _forward(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        req_body = self.rfile.read(length) if length else b""

        auth = self.headers.get("Authorization") or ""
        want_capture = method == "POST" and self.path.split("?")[0] in CAPTURE_PATHS
        parse_ok = False
        body_json = None
        if want_capture:
            if length and length <= MAX_CAPTURE_BODY:
                try:
                    body_json = json.loads(req_body)
                    parse_ok = isinstance(body_json, dict)
                except Exception:
                    parse_ok = False
            if not parse_ok:
                _bump_dropped("body-parse")
                want_capture = False

        # [TIME] | INFO | CAPTURE_TEE.COMPACT_HOOK | rewrite an oversized
        # history before it reaches llama-server, so a prompt that could not
        # finish prefilling inside the client's timeout never gets started.
        if (
            COMPACT_ENABLED
            and parse_ok
            and self.path.split("?")[0] == "/v1/chat/completions"
            and isinstance(body_json.get("messages"), list)
        ):
            try:
                new_msgs, info, incoming_tokens = _compact_messages(
                    body_json["messages"], body_json.get("tools"), _upstream_auth()
                )
                # [TIME] | INFO | CAPTURE_TEE.MODEL_SIZES | record how big a
                # prompt each client-declared model name arrives with. Cursor
                # sizes its own auto-compaction from the context window it
                # associates with the model NAME, so this is the only way to
                # tell whether a rename actually made Cursor send less.
                if incoming_tokens:
                    _record_model_size(body_json.get("model"), incoming_tokens)
                if info:
                    body_json = dict(body_json, messages=new_msgs)
                    req_body = json.dumps(body_json).encode()
                    with _stats_lock:
                        _stats["compacted"] = _stats.get("compacted", 0) + 1
                        _stats["last_compaction"] = info
                    print(f"[capture-tee] compacted {info}", flush=True)
            except Exception as e:
                print(f"[capture-tee] compact failed: {type(e).__name__}: {e}", flush=True)

        # build upstream request, forwarding headers that matter
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        # swap the public client key for llama-server's --api-key upstream
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}" if UPSTREAM_API_KEY else auth
        accept = self.headers.get("Accept")
        if accept:
            headers["Accept"] = accept
        req = urllib.request.Request(
            UPSTREAM + self.path, data=req_body if method == "POST" else None,
            headers=headers, method=method,
        )

        t0 = time.time()
        try:
            upstream = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            upstream = e
        except Exception as e:
            self.send_error(502, f"upstream unreachable: {e}")
            return
        status = upstream.code if hasattr(upstream, "code") else 502
        with _stats_lock:
            _stats["requests"] += 1

        # stream the response through untouched
        self.send_response(status)
        content_type = upstream.headers.get("Content-Type", "application/json")
        is_sse = "text/event-stream" in content_type
        self.send_header("Content-Type", content_type)
        if is_sse:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
        else:
            body = upstream.read()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            upstream.close()
            if want_capture:
                threading.Thread(
                    target=self._capture_nonstream, daemon=True,
                    args=(body_json, body, status, t0, auth),
                ).start()
            return

        # SSE: forward chunks as they arrive; accumulate for capture.
        # [TIME] | INFO | CAPTURE_TEE.SSE_KEEPALIVE | llama-server emits
        # nothing until prefill finishes (minutes on a Cursor-sized prompt).
        # Idle proxies/clients then cancel. Short socket timeouts + SSE
        # comment pings keep the pipe alive without touching TTFT.
        # [TIME] | INFO | CAPTURE_TEE.DETACH | Cursor abandons a request after
        # ~120s, but a cold Cursor-sized prefill needs minutes (quadratic in
        # depth). Closing the upstream leg here makes llama-server cancel the
        # task and discard the partial KV, so the retry restarts from zero and
        # can never converge. Instead keep draining upstream after the client
        # leaves: the prefill completes, the prompt cache warms, and the retry
        # hits it at ~100% reuse.
        chunks = []
        raw_sock = None
        client_gone = False
        try:
            try:
                raw_sock = upstream.fp.raw._sock
                raw_sock.setblocking(True)
                raw_sock.settimeout(None)
            except Exception:
                raw_sock = None
            KEEPALIVE_S = 15.0
            while True:
                if raw_sock is not None:
                    try:
                        if raw_sock.fileno() < 0:
                            break
                        ready, _, _ = select.select([raw_sock], [], [], KEEPALIVE_S)
                    except (ValueError, OSError):
                        break
                    if not ready:
                        if client_gone:
                            continue
                        ping = b": keepalive\n\n"
                        try:
                            self.wfile.write(b"%x\r\n" % len(ping) + ping + b"\r\n")
                            self.wfile.flush()
                        except Exception:
                            client_gone = True
                        continue
                try:
                    chunk = upstream.read(4096)
                except Exception:
                    # [TIME] | WARN | CAPTURE_TEE.SSE_READ | upstream aborted
                    # mid-chunk (llama-server cancels the task when the client
                    # gives up). IncompleteRead is an HTTPException, not an
                    # OSError, so it must not escape — an escaping exception
                    # kills the handler and the client sees a truncated body.
                    break
                if not chunk:
                    break
                chunks.append(chunk)
                if not client_gone:
                    try:
                        self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                        self.wfile.flush()
                    except Exception:
                        client_gone = True
            if not client_gone:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass
            else:
                with _stats_lock:
                    _stats["detached_completions"] = _stats.get("detached_completions", 0) + 1
        except Exception:
            pass
        finally:
            try:
                upstream.close()
            except Exception:
                pass
        if want_capture:
            threading.Thread(
                target=self._capture_stream, daemon=True,
                args=(body_json, chunks, status, t0, auth),
            ).start()

    # -- capture assembly ----------------------------------------------------
    def _capture_stream(self, body_json, chunks, status, t0, auth):
        try:
            text = b"".join(chunks).decode("utf-8", errors="replace")
            content, tool_calls, finish_reason, usage, timings = _parse_sse(text)
            self._finish_record(body_json, content, tool_calls, finish_reason,
                                usage, timings, status, t0, auth, stream=True)
        except Exception as e:
            _bump_dropped(f"sse-parse:{type(e).__name__}")

    def _capture_nonstream(self, body_json, body, status, t0, auth):
        try:
            r = json.loads(body)
            choice = (r.get("choices") or [{}])[0]
            msg = choice.get("message", {})
            self._finish_record(
                body_json, msg.get("content"), msg.get("tool_calls") or [],
                choice.get("finish_reason"), r.get("usage"), r.get("timings"),
                status, t0, auth, stream=False, raw_response=r,
            )
        except Exception as e:
            _bump_dropped(f"json-parse:{type(e).__name__}")

    def _finish_record(self, body_json, content, tool_calls, finish_reason,
                       usage, timings, status, t0, auth, stream, raw_response=None):
        # [2026-09-01] | INFO | CAPTURE_TEE._FINISH_RECORD | keep tool-only
        # Cursor turns; empty-content used to drop finish_reason=tool_calls
        if status != 200:
            _bump_dropped("bad-status")
            return
        if not content and not tool_calls:
            _bump_dropped("empty-content")
            return
        prompt_ids, _, template_ok = _token_snapshot(auth, body_json)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "req_id": uuid.uuid4().hex[:12],
            "key_id": _key_id(auth),
            "model": body_json.get("model"),
            "stream": stream,
            "request": {
                "messages": body_json.get("messages"),
                "tools": body_json.get("tools"),
                "tool_choice": body_json.get("tool_choice"),
                "sampling": {k: body_json.get(k) for k in
                             ("temperature", "top_p", "top_k", "max_tokens",
                              "stop", "presence_penalty", "frequency_penalty")},
                "chat_template_kwargs": body_json.get("chat_template_kwargs"),
            },
            "response": {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
            },
            "prompt_token_ids": prompt_ids,
            "template_ok": template_ok,
            "usage": usage,
            "timings": timings,
            "wall_ms": int((time.time() - t0) * 1000),
        }
        if raw_response is not None and len(json.dumps(raw_response)) < 2048:
            record["raw_response"] = raw_response
        _write_record(record)
        with _stats_lock:
            _stats["captured"] += 1


def _parse_sse(text):
    """
    [TIME] | INFO | CAPTURE_TEE.PARSE_SSE | fold an SSE stream back into the
    pieces the dataset needs: content deltas, tool-call deltas, finish_reason,
    usage and timings from the final chunk.
    """
    content_parts, tool_calls, finish_reason = [], {}, None
    usage = timings = None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        if obj.get("timings"):
            timings = obj["timings"]
        for choice in obj.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_calls.setdefault(idx, {"id": "", "type": "function",
                                                   "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    return "".join(content_parts), list(tool_calls.values()), finish_reason, usage, timings


def main():
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), TeeHandler)
    print(f"[capture-tee] upstream={UPSTREAM} port={LISTEN_PORT} dir={CAPTURE_DIR}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
