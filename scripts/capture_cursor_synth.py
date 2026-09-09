#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.CAPTURE_CURSOR_SYNTH | send Cursor-shaped
chat/completions through capture-tee so they land in the MTP draft-head
dataset. Hits :8601 (not :8600 or :8700 — those skip the tee).

Modes:
  --source tasks        fixed synthetic tasks
  --source transcripts  replay local agent transcripts turn-by-turn

Transcript corpora (--corpus):
  cursor      main Cursor agent transcripts   (tool_use only, results stubbed)
  subagents   Cursor subagent transcripts     (tool_use only, results stubbed)
  claude      Claude Code transcripts         (real tool_result blocks)
  all         every corpus above

Usage:
  python3 scripts/capture_cursor_synth.py --source transcripts --sessions 1
  python3 scripts/capture_cursor_synth.py --source transcripts --corpus all \
      --sessions 200 --max-turns 40 --deep-frac 0.4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

TEE_URL = "http://127.0.0.1:8601/v1/chat/completions"
MODEL = "qwen38-flash"
CLIENT_KEY = "sk-mtp-synth-local"
REPLAY_KEY = "sk-mtp-replay-local"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENVELOPE_PATH = os.path.join(SCRIPT_DIR, "fixtures", "cursor_envelope.json")

CORPUS_GLOBS = {
    "cursor": "/home/alec/.cursor/projects/*/agent-transcripts/*/*.jsonl",
    "subagents": "/home/alec/.cursor/projects/*/agent-transcripts/*/subagents/*.jsonl",
    "claude": "/home/alec/.claude/projects/**/*.jsonl",
}

MAX_TOOL_RESULT_CHARS = 2600
MAX_HISTORY_MESSAGES = 60
# Cursor stores redacted assistant text in transcripts; replaying it teaches the
# model to emit the placeholder itself, so it never enters the prompt history.
REDACTION_RE = re.compile(r"\[REDACTED\]")

SECRET_RE = re.compile(r"\b(?:sk|hf|ghp|gho|xoxb|AKIA)[-_][A-Za-z0-9_-]{16,}\b")
BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._-]{16,}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

READONLY_TOOLS = frozenset({
    "Read", "ReadFile", "Grep", "rg", "Glob", "ReadLints",
    "SemanticSearch", "SemSearch", "WebFetch", "WebSearch", "LS", "ListDir",
})
MUTATING_TOOLS = frozenset({
    "Shell", "Bash", "AwaitShell", "Await", "StrReplace", "Write", "ApplyPatch",
    "TodoWrite", "Task", "AskQuestion", "CallMcpTool", "CallDynamicTool",
    "EditNotebook", "Delete", "SwitchMode", "Edit", "MultiEdit", "NotebookEdit",
})

# [TIME] | INFO | CAPTURE_CURSOR_SYNTH.SAMPLING | spread of decode settings so the
# draft head sees more than one operating point
TEMPERATURES = [0.2, 0.3, 0.4, 0.6, 0.7, 1.0]
# Below ~512 the model gets truncated mid-answer and dataset_build drops the
# sample as finish-length, so the floor stays high enough to be useful.
MAX_TOKEN_CHOICES = [512, 768, 1024, 1536]
THINKING_PROB = 0.10

CURSOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search file contents with a regex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Shell",
            "description": "Run a command in the workspace shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "working_directory": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "StrReplace",
            "description": "Replace a unique string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
]


def scrub_secrets(text: str) -> str:
    """[TIME] | INFO | CAPTURE_CURSOR_SYNTH.scrub_secrets | keep keys out of prompt ids."""
    text = SECRET_RE.sub("<SECRET_KEY>", text)
    text = BEARER_RE.sub(r"\1<SECRET_TOKEN>", text)
    return EMAIL_RE.sub("<EMAIL>", text)


def _clip_text(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    text = scrub_secrets(text)
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.25):]
    return f"{head}\n...[{len(text)} chars truncated]...\n{tail}"


def _trim_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """[TIME] | INFO | CAPTURE_CURSOR_SYNTH._trim_messages | cap replay history.

    Trims from the front but never splits an assistant tool_calls message from
    the tool messages answering it, which the server rejects.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    budget = MAX_HISTORY_MESSAGES - len(system_msgs)
    keep = rest[-budget:]
    while keep and keep[0].get("role") == "tool":
        keep.pop(0)
    return system_msgs + keep


# --------------------------------------------------------------------------
# envelopes
# --------------------------------------------------------------------------

def load_envelopes(include_variants: bool = True) -> List[Dict[str, Any]]:
    """[TIME] | INFO | CAPTURE_CURSOR_SYNTH.load_envelopes | real Cursor envelopes."""
    with open(ENVELOPE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "envelopes" in data:
        base = list(data["envelopes"])
    else:
        base = [{
            "id": "legacy",
            "system": data["system"],
            "tools": data["tools"],
        }]
    envs: List[Dict[str, Any]] = []
    for e in base:
        envs.append({"id": e["id"], "system": e["system"], "tools": e["tools"]})
    if not include_variants:
        return envs

    # Mode variants: Cursor also runs tool-free (Ask) and reduced tool surfaces.
    primary = max(base, key=lambda e: len(e["system"]))
    envs.append({
        "id": primary["id"] + "-notools",
        "system": primary["system"],
        "tools": [],
    })
    core = {"Read", "Grep", "Glob", "Shell", "StrReplace", "Write", "TodoWrite"}
    subset = [t for t in primary["tools"] if (t.get("function") or {}).get("name") in core]
    if subset:
        envs.append({
            "id": primary["id"] + "-core",
            "system": primary["system"],
            "tools": subset,
        })
    envs.append({
        "id": "slim",
        "system": primary["system"][:3000],
        "tools": CURSOR_TOOLS,
    })
    return envs


# --------------------------------------------------------------------------
# tool-result reconstruction (only needed for corpora without tool_result)
# --------------------------------------------------------------------------

_STUB_POOL: Optional[List[str]] = None


def _load_stub_pool() -> List[str]:
    global _STUB_POOL
    if _STUB_POOL is not None:
        return _STUB_POOL
    snippets: List[str] = []
    for root in ("/home/alec/git/llama-nexus/backend", "/home/alec/git/llama-nexus/scripts"):
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, "**/*.py"), recursive=True))[:40]:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read(3000)
                if text.strip():
                    snippets.append(text)
            except OSError:
                continue
    _STUB_POOL = snippets or ["# no sources available\n"]
    return _STUB_POOL


def _length_matched_stub(target_len: int = 800) -> str:
    pool = _load_stub_pool()
    base = pool[target_len % len(pool)]
    if len(base) >= target_len:
        return base[:target_len]
    return (base * ((target_len // max(len(base), 1)) + 1))[:target_len]


def _read_file_tool(args: Dict[str, Any]) -> str:
    path = args.get("path") or args.get("target_file") or args.get("file_path") or ""
    if not path or not os.path.isfile(path):
        return f"Error: file not found: {path}"
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            if args.get("offset") is not None or args.get("limit") is not None:
                lines = f.readlines()
                start = max(0, int(args.get("offset") or 1) - 1)
                end = start + int(args["limit"]) if args.get("limit") else len(lines)
                return "".join(lines[start:end])
            return f.read(4000)
    except OSError as exc:
        return f"Error reading {path}: {exc}"


def _grep_tool(args: Dict[str, Any]) -> str:
    pattern = args.get("pattern") or "."
    path = args.get("path") or "."
    if not os.path.exists(path):
        return "No matches found."
    cmd = ["rg", "--no-heading", "-n", "-m", str(int(args.get("head_limit") or 40)),
           pattern, path]
    if args.get("glob"):
        cmd.extend(["--glob", str(args["glob"])])
    try:
        # errors="replace": rg walks real repos, so a hit inside a latin-1 source
        # file or a binary blob emits bytes that are not valid UTF-8. text=True
        # alone decodes strictly and raises inside communicate(), which killed a
        # replay worker 33 sessions into a 40-session shard. Every other read in
        # this file already replaces; this call was the one that did not.
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                              check=False, errors="replace")
        out = (proc.stdout or "").strip()
        return out[:3000] if out else "No matches found."
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError):
        return "No matches found."


def _glob_tool(args: Dict[str, Any]) -> str:
    pattern = args.get("glob_pattern") or args.get("pattern") or "*"
    root = args.get("target_directory") or args.get("path") or "."
    try:
        matches = sorted(glob.glob(os.path.join(root, pattern), recursive=True))[:150]
    except OSError:
        matches = []
    return "\n".join(matches) if matches else "No files matched."


_SHELL_TEMPLATES = {
    "git": (
        "On branch master\nYour branch is up to date with 'origin/master'.\n\n"
        "Changes not staged for commit:\n  modified:   backend/capture_tee.py\n"
        "  modified:   scripts/dataset_build.py\n\n"
        "no changes added to commit\n"
    ),
    "du": (
        "12G\t./models\n4.3G\t./data\n2.1G\t./node_modules\n880M\t./.git\n"
        "412M\t./backend\n96M\t./frontend\n18M\t./scripts\n"
    ),
    "df": (
        "Filesystem      Size  Used Avail Use% Mounted on\n"
        "/dev/nvme0n1p2  915G  790G   79G  91% /\n"
        "/dev/sda1       469G  201G  268G  43% /mnt/comfyui-models\n"
    ),
    "docker": (
        "NAMES                   STATUS\n"
        "llamacpp-api            Up 2 days (healthy)\n"
        "llamacpp-capture-tee    Up 3 hours (healthy)\n"
        "llamacpp-backend        Up 2 days (healthy)\n"
    ),
    "ls": (
        "total 584\ndrwxrwxr-x 22 alec alec  4096 Aug 13 14:30 .\n"
        "drwxrwxr-x 20 alec alec  4096 Jul 29 19:01 ..\n"
        "-rw-rw-r--  1 alec alec  8213 Sep  1 16:06 docker-compose.yml\n"
        "drwxrwxr-x  8 alec alec  4096 Sep  4 14:38 backend\n"
        "drwxrwxr-x  3 alec alec  4096 Sep  4 14:38 scripts\n"
    ),
    "pytest": (
        "============================= test session starts ==============================\n"
        "collected 42 items\n\ntests/test_capture.py ..........                        [ 23%]\n"
        "tests/test_dataset.py ................                                   [ 61%]\n"
        "tests/test_mtp.py ................                                       [100%]\n\n"
        "============================== 42 passed in 6.13s ==============================\n"
    ),
    "curl": '{"status":"ok","model":"qwen38-flash","slots_idle":1,"slots_processing":0}\n',
    "nvidia-smi": (
        "0, 21344 MiB, 24564 MiB, 74 %\n1, 21980 MiB, 24564 MiB, 71 %\n"
        "2, 21344 MiB, 24564 MiB, 69 %\n3, 22012 MiB, 24564 MiB, 73 %\n"
    ),
}


def _shell_stub(args: Dict[str, Any]) -> str:
    """[TIME] | INFO | CAPTURE_CURSOR_SYNTH._shell_stub | command-aware output."""
    cmd = str(args.get("command") or "").strip()
    if not cmd:
        return "Exit code: 0\n"
    head = os.path.basename(cmd.split()[0]) if cmd.split() else ""
    for key, body in _SHELL_TEMPLATES.items():
        if head == key or cmd.startswith(key + " "):
            return f"Exit code: 0\n{body}"
    if head in ("cat", "head", "tail", "less"):
        parts = [p for p in cmd.split()[1:] if not p.startswith("-")]
        if parts and os.path.isfile(parts[0]):
            return "Exit code: 0\n" + _read_file_tool({"path": parts[0]})[:2000]
    if head in ("echo", "printf"):
        return "Exit code: 0\n" + cmd.split(" ", 1)[-1].strip("'\"") + "\n"
    if head in ("mkdir", "cp", "mv", "rm", "chmod", "chown", "touch", "ln"):
        return "Exit code: 0\n"
    if head in ("wc", "find", "which", "whoami", "date", "pwd"):
        return "Exit code: 0\n42\n"
    return "Exit code: 0\n" + _length_matched_stub(min(1400, max(200, len(cmd) * 6)))


def _mutate_stub(name: str, args: Dict[str, Any]) -> str:
    path = args.get("path") or args.get("file_path") or args.get("target_notebook") or "<path>"
    if name in ("TodoWrite",):
        return "Todos updated successfully."
    if name in ("Write", "ApplyPatch", "Edit", "MultiEdit", "StrReplace"):
        return f"The file {path} has been updated."
    return f"{name} completed."


def reconstruct_tool_result(name: str, args: Any) -> str:
    """[TIME] | INFO | CAPTURE_CURSOR_SYNTH.reconstruct_tool_result | tool replay."""
    if not isinstance(args, dict):
        args = {}
    if name in ("Read", "ReadFile"):
        return _clip_text(_read_file_tool(args))
    if name in ("Grep", "rg"):
        return _clip_text(_grep_tool(args))
    if name == "Glob":
        return _clip_text(_glob_tool(args))
    if name == "ReadLints":
        return "No linter errors found."
    if name in ("SemanticSearch", "SemSearch", "WebSearch"):
        return _clip_text(_length_matched_stub(1200))
    if name == "WebFetch":
        return _clip_text(_length_matched_stub(1800))
    if name in ("Shell", "Bash", "AwaitShell", "Await"):
        return _clip_text(_shell_stub(args))
    if name in MUTATING_TOOLS:
        return _mutate_stub(name, args)
    return _clip_text(_length_matched_stub(700))


# --------------------------------------------------------------------------
# transcript parsing -> normalized event stream
# --------------------------------------------------------------------------

def _normalize_blocks(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return []


def _blocks_text(blocks: List[Dict[str, Any]]) -> str:
    parts = []
    for b in blocks:
        if b.get("type") != "text":
            continue
        t = REDACTION_RE.sub("", str(b.get("text") or "")).strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


def _blocks_tool_calls(blocks: List[Dict[str, Any]], seq: int) -> List[Dict[str, Any]]:
    calls = []
    for i, b in enumerate(blocks):
        if b.get("type") != "tool_use":
            continue
        inp = b.get("input")
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except json.JSONDecodeError:
                inp = {"raw": inp}
        if not isinstance(inp, dict):
            inp = {}
        calls.append({
            "id": str(b.get("id") or f"call_{seq}_{i}")[:40],
            "type": "function",
            "function": {
                "name": b.get("name") or "unknown",
                "arguments": json.dumps(inp, ensure_ascii=False),
            },
        })
    return calls


def _result_text(block: Dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
            elif isinstance(b, dict) and b.get("type") == "image":
                parts.append("[image]")
        return "\n".join(parts)
    return json.dumps(content)[:2000] if content is not None else ""


def parse_transcript(path: str) -> List[Dict[str, Any]]:
    """[TIME] | INFO | CAPTURE_CURSOR_SYNTH.parse_transcript | Cursor + Claude formats.

    Emits a normalized stream of {kind: user|assistant|tool_results} events.
    Claude transcripts carry real tool_result blocks; Cursor ones do not.
    """
    events: List[Dict[str, Any]] = []
    seq = 0
    try:
        raw_lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return events

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Cursor puts role at the top level; Claude nests it under message.role
        # and mirrors it in the row's type field.
        rtype = row.get("type")
        msg = row.get("message") or {}
        role = row.get("role") or msg.get("role")
        if role not in ("user", "assistant"):
            role = rtype if rtype in ("user", "assistant") else None
        if role not in ("user", "assistant"):
            continue
        blocks = _normalize_blocks(msg.get("content"))
        if not blocks:
            continue

        results = {
            str(b.get("tool_use_id")): _result_text(b)
            for b in blocks if b.get("type") == "tool_result" and b.get("tool_use_id")
        }
        if results:
            events.append({"kind": "tool_results", "results": results})

        text = _blocks_text(blocks)
        calls = _blocks_tool_calls(blocks, seq)
        seq += 1

        if role == "user":
            if text:
                events.append({"kind": "user", "text": text})
        elif text or calls:
            events.append({"kind": "assistant", "text": text, "tool_calls": calls})

    return events


def _load_stack_profile():
    """[TIME] | INFO | CAPTURE_CURSOR_SYNTH._load_stack_profile | optional stack targeting."""
    import importlib.util
    path = os.path.join(SCRIPT_DIR, "stack_profile.py")
    spec = importlib.util.spec_from_file_location("stack_profile", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_transcript_paths(corpora: Iterable[str], min_bytes: int = 2000) -> List[str]:
    paths: List[str] = []
    for name in corpora:
        pattern = CORPUS_GLOBS.get(name)
        if not pattern:
            continue
        for p in glob.glob(pattern, recursive=True):
            if name == "cursor" and "/subagents/" in p:
                continue
            try:
                if os.path.getsize(p) >= min_bytes:
                    paths.append(p)
            except OSError:
                continue
    return sorted(set(paths))


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _post(url: str, key: str, body: Dict[str, Any], timeout: int) -> Tuple[int, bytes]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001 - network errors must not kill a long run
        return 0, str(e).encode()


def _drain_sse(raw: bytes) -> Tuple[str, List[Any], Optional[str]]:
    text = raw.decode("utf-8", errors="replace")
    content, tool_calls, finish = [], [], None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content.append(delta["content"])
        if delta.get("tool_calls"):
            tool_calls.extend(delta["tool_calls"])
        finish = choice.get("finish_reason") or finish
    return "".join(content), tool_calls, finish


def _parse_response(raw: bytes, stream: bool) -> Tuple[str, List[Any], Optional[str]]:
    if stream:
        return _drain_sse(raw)
    try:
        parsed = json.loads(raw)
        choice = (parsed.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return msg.get("content") or "", msg.get("tool_calls") or [], choice.get("finish_reason")
    except json.JSONDecodeError:
        return raw[:200].decode("utf-8", "replace"), [], "parse-error"


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def replay_session(path: str, args: argparse.Namespace,
                   envelopes: List[Dict[str, Any]], rng: random.Random) -> Tuple[int, int]:
    """[TIME] | INFO | CAPTURE_CURSOR_SYNTH.replay_session | one transcript file."""
    events = parse_transcript(path)
    assistant_idx = [i for i, e in enumerate(events) if e["kind"] == "assistant"]
    if not assistant_idx:
        return 0, 0

    env = envelopes[rng.randrange(len(envelopes))]
    tools = env["tools"]
    messages: List[Dict[str, Any]] = [{"role": "system", "content": env["system"]}]
    pending: Dict[str, Dict[str, Any]] = {}
    session_id = os.path.basename(path).replace(".jsonl", "")[:12]

    # Deep sampling: build history silently up to a random point so some prompts
    # start from mid-conversation state instead of always from turn one.
    skip_until = 0
    if args.deep_frac and rng.random() < args.deep_frac and len(assistant_idx) > 6:
        skip_until = assistant_idx[rng.randrange(len(assistant_idx) // 2)]

    ok = sent = 0
    for idx, ev in enumerate(events):
        kind = ev["kind"]

        if kind == "user":
            messages.append({"role": "user", "content": _clip_text(ev["text"], 6000)})
            continue

        if kind == "tool_results":
            for call_id, result in ev["results"].items():
                if call_id in pending:
                    pending[call_id]["content"] = _clip_text(result)
            continue

        if args.max_turns and sent >= args.max_turns:
            break

        assistant = {"role": "assistant"}
        if ev["text"]:
            assistant["content"] = _clip_text(ev["text"], 4000)
        elif not ev["tool_calls"]:
            assistant["content"] = ""
        if ev["tool_calls"]:
            assistant["tool_calls"] = ev["tool_calls"]

        if idx >= skip_until and messages[-1].get("role") in ("user", "tool"):
            temperature = rng.choice(TEMPERATURES)
            max_tokens = rng.choice(MAX_TOKEN_CHOICES)
            thinking = rng.random() < THINKING_PROB
            if thinking:
                max_tokens = max(max_tokens, 1024)
            body = {
                "model": args.model,
                "messages": _trim_messages(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": rng.choice([0.8, 0.9, 0.95, 1.0]),
                "stream": args.stream,
                "chat_template_kwargs": {"enable_thinking": thinking},
            }
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"

            t0 = time.time()
            status, raw = _post(args.url, args.key, body, args.timeout)
            wall = time.time() - t0
            content, tcalls, finish = _parse_response(raw, args.stream)
            sent += 1
            if status == 200:
                ok += 1
            if args.verbose or sent % 10 == 0 or status != 200:
                print(
                    f"[replay:{session_id}] env={env['id'][:8]} step={sent} status={status} "
                    f"finish={finish} msgs={len(messages)} T={temperature} mt={max_tokens} "
                    f"think={int(thinking)} chars={len(content)} tools={len(tcalls)} wall={wall:.1f}s",
                    flush=True,
                )
            if args.sleep:
                time.sleep(args.sleep)

        messages.append(assistant)
        pending = {}
        for tc in ev["tool_calls"]:
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": reconstruct_tool_result(
                    (tc.get("function") or {}).get("name"),
                    json.loads((tc.get("function") or {}).get("arguments") or "{}"),
                ),
            }
            pending[tc["id"]] = tool_msg
            messages.append(tool_msg)

    return ok, sent


def run_transcripts(args: argparse.Namespace) -> int:
    """[TIME] | INFO | CAPTURE_CURSOR_SYNTH.run_transcripts | replay local sessions."""
    corpora = ["cursor", "subagents", "claude"] if args.corpus == "all" else [args.corpus]
    envelopes = load_envelopes(include_variants=not args.no_variants)
    paths = list_transcript_paths(corpora)
    if not paths:
        print(f"[capture_cursor_synth] no transcripts for corpus={args.corpus}", flush=True)
        return 1

    rng = random.Random(args.seed)
    if args.stack:
        sp = _load_stack_profile()
        profs = sp.profile_corpus(paths, parse_transcript)
        picked = sp.select_sessions(profs, args.sessions + args.offset,
                                    args.stack, seed=args.seed)
        selected = [p["path"] for p in picked][args.offset:]
        print(f"[capture_cursor_synth] stack={args.stack}\n{sp.summarize(picked)}", flush=True)
    else:
        rng.shuffle(paths)
        if args.offset:
            paths = paths[args.offset:]
        selected = paths[: args.sessions]

    # Deterministic split so several workers (e.g. one against the local rig and
    # one against a rented pod) cover disjoint sessions from the same selection.
    if args.shards > 1:
        before = len(selected)
        selected = selected[args.shard :: args.shards]
        print(
            f"[capture_cursor_synth] shard {args.shard}/{args.shards}: "
            f"{len(selected)} of {before} sessions",
            flush=True,
        )

    print(
        f"[capture_cursor_synth] corpus={args.corpus} sessions={len(selected)}/{len(paths)} "
        f"envelopes={len(envelopes)} max_turns={args.max_turns} deep_frac={args.deep_frac}",
        flush=True,
    )

    total_ok = total_sent = 0
    t_start = time.time()
    for i, path in enumerate(selected):
        ok, sent = replay_session(path, args, envelopes, rng)
        total_ok += ok
        total_sent += sent
        print(
            f"[capture_cursor_synth] session {i+1}/{len(selected)} ok={ok}/{sent} "
            f"total={total_ok}/{total_sent} elapsed={time.time()-t_start:.0f}s",
            flush=True,
        )
    print(f"[capture_cursor_synth] replay ok={total_ok}/{total_sent}", flush=True)
    return 0


def run_task(task: Dict[str, Any], args: argparse.Namespace) -> bool:
    messages = [
        {"role": "system", "content": task["system"]},
        {"role": "user", "content": task["user"]},
    ]
    body = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0.4,
        "stream": args.stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if task["tools"]:
        body["tools"] = CURSOR_TOOLS
        body["tool_choice"] = "auto"
    t0 = time.time()
    status, raw = _post(args.url, args.key, body, args.timeout)
    wall = time.time() - t0
    content, tool_calls, finish = _parse_response(raw, args.stream)
    print(
        f"[{task['id']}] status={status} finish={finish} "
        f"chars={len(content)} tool_calls={len(tool_calls)} wall={wall:.1f}s",
        flush=True,
    )
    return status == 200


def _repo_ctx(name: str, n_funcs: int = 24) -> str:
    lines = [f"# workspace excerpt: {name}", ""]
    for i in range(n_funcs):
        lines.append(f"def handler_{name}_{i}(req):")
        lines.append(f"    return {{'ok': True, 'n': {i}, 'route': '/{name}/{i}'}}")
        lines.append("")
    return "\n".join(lines)


TASKS = [
    {
        "id": "code-review",
        "tools": False,
        "system": "You are a coding assistant in llama-nexus. Be concise.",
        "user": _repo_ctx("backend", 20) + "\nReview handler_backend_3. Name one bug risk in 4 sentences.",
    },
    {
        "id": "json-config",
        "tools": False,
        "system": "You emit only valid JSON. No markdown fences.",
        "user": (
            "Build a llama.cpp MTP deploy JSON with keys enabled, draft_n_max, "
            "draft_n_min, draft_p_min, workload=agent."
        ),
    },
    {
        "id": "tool-read",
        "tools": True,
        "system": "You are Cursor. Prefer tools over guessing file contents.",
        "user": (
            "Open /home/alec/git/llama-nexus/backend/capture_tee.py and tell "
            "me how capture records are written. Use the Read tool."
        ),
    },
    {
        "id": "tool-grep",
        "tools": True,
        "system": "You are Cursor in /home/alec/git/llama-nexus. Use Grep first.",
        "user": "Find every reference to spec-type draft-mtp. Use Grep, then say which files matter.",
    },
]


def main() -> int:
    p = argparse.ArgumentParser(description="Cursor-shaped MTP capture traffic")
    p.add_argument("--source", choices=("tasks", "transcripts"), default="tasks")
    p.add_argument("--corpus", choices=("cursor", "subagents", "claude", "all"), default="cursor")
    p.add_argument("--url", default=TEE_URL)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--key", default=None)
    p.add_argument("--limit", type=int, default=len(TASKS))
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--sessions", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=0, help="0 = no cap per session")
    p.add_argument("--deep-frac", type=float, default=0.0,
                   help="fraction of sessions that start mid-conversation")
    p.add_argument("--no-variants", action="store_true", help="only real captured envelopes")
    p.add_argument("--stack", default=None,
                   help="target this environment's stack mix: natural|balanced|python|frontend|backend-api")
    p.add_argument("--shards", type=int, default=1,
                   help="split the selected sessions across this many workers")
    p.add_argument("--shard", type=int, default=0,
                   help="which shard this worker handles (0-based)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.source == "transcripts":
        if args.key is None:
            args.key = REPLAY_KEY
        if args.timeout == 180:
            args.timeout = 600
        return run_transcripts(args)

    if args.key is None:
        args.key = CLIENT_KEY
    selected = TASKS[: args.limit]
    print(f"[capture_cursor_synth] n={len(selected)} url={args.url}", flush=True)
    ok = sum(1 for t in selected if run_task(t, args))
    print(f"[capture_cursor_synth] ok={ok}/{len(selected)}", flush=True)
    return 0 if ok == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
