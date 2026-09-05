#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.OPENROUTER_SFT_BUILD | build a Cursor-agent SFT dataset
by replaying real agent conversation prefixes against frontier models on
OpenRouter.

This is NOT the MTP draft-head pipeline. The MTP head needs self-distillation
from the local target model, so teacher outputs would lower acceptance. This
dataset is knowledge distillation instead: teach agent BEHAVIOUR in Cursor's
exact request shape. It talks straight to OpenRouter and never touches
capture-tee, so MTP captures stay clean.

Prompt construction (envelopes, transcript parsing, tool-result
reconstruction) is reused from capture_cursor_synth.py rather than duplicated.

Usage:
  export OPENROUTER_API_KEY=sk-or-v1-...
  python3 scripts/openrouter_sft_build.py --sessions 5 --budget-usd 1.00
  python3 scripts/openrouter_sft_build.py --corpus all --sessions 400 \
      --max-turns 12 --budget-usd 40 --models deepseek/deepseek-v4-flash,qwen/qwen3-coder-next
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OUT = "/mnt/comfyui-models/llama-nexus/datasets/sft-cursor-agent"

# Strong tool-calling models at a range of price points; sampled per request so
# the set is not one teacher's idiosyncrasies.
DEFAULT_MODELS = [
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-coder-next",
    "openai/gpt-5.6-luna",
    "z-ai/glm-4.7",
]

REFUSAL_MARKERS = (
    "i cannot assist", "i can't assist", "i'm sorry, but i can",
    "i am unable to", "as an ai language model",
)


def _load_shared():
    """[TIME] | INFO | OPENROUTER_SFT_BUILD._load_shared | reuse replay machinery."""
    path = os.path.join(SCRIPT_DIR, "capture_cursor_synth.py")
    spec = importlib.util.spec_from_file_location("capture_cursor_synth", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CCS = _load_shared()


def _key_from_dotenv(path: str) -> Optional[str]:
    """[TIME] | INFO | OPENROUTER_SFT_BUILD._key_from_dotenv | parse KEY=VALUE file."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == "OPENROUTER_API_KEY":
                    return value.strip().strip("'\"") or None
    except OSError:
        return None
    return None


def resolve_api_key(explicit: Optional[str]) -> Optional[str]:
    """[TIME] | INFO | OPENROUTER_SFT_BUILD.resolve_api_key | flag, env, .env, key file."""
    if explicit:
        return explicit.strip()
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env.strip()
    repo_env = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".env"))
    from_dotenv = _key_from_dotenv(repo_env)
    if from_dotenv:
        return from_dotenv
    for candidate in (
        os.path.expanduser("~/.config/openrouter/key"),
        os.path.expanduser("~/.openrouter_key"),
    ):
        try:
            with open(candidate, encoding="utf-8") as f:
                val = f.read().strip()
            if val:
                return val
        except OSError:
            continue
    return None


def post_openrouter(key: str, body: Dict[str, Any], timeout: int) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/llama-nexus",
            "X-Title": "llama-nexus cursor-agent SFT build",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:  # noqa: BLE001
            return e.code, {}
    except Exception as e:  # noqa: BLE001 - long runs must survive transient errors
        return 0, {"error": {"message": str(e)}}


def is_good_sample(content: str, tool_calls: List[Any], finish: Optional[str]) -> Tuple[bool, str]:
    """[TIME] | INFO | OPENROUTER_SFT_BUILD.is_good_sample | curate teacher output.

    Unlike the MTP set we WANT to filter here: the goal is good agent behaviour,
    not a faithful sample of the model's natural distribution.
    """
    if finish == "length":
        return False, "truncated"
    if not content.strip() and not tool_calls:
        return False, "empty"
    low = content.lower()
    if any(m in low for m in REFUSAL_MARKERS):
        return False, "refusal"
    for tc in tool_calls or []:
        args = ((tc or {}).get("function") or {}).get("arguments")
        if args:
            try:
                json.loads(args)
            except (TypeError, json.JSONDecodeError):
                return False, "bad-tool-json"
    if not tool_calls and len(content.strip()) < 24:
        return False, "too-short"
    return True, "ok"


def build_session_records(path: str, args, envelopes, rng, key, budget,
                          stack_meta: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict], float, Dict[str, int]]:
    """[TIME] | INFO | OPENROUTER_SFT_BUILD.build_session_records | one transcript."""
    events = CCS.parse_transcript(path)
    if not any(e["kind"] == "assistant" for e in events):
        return [], 0.0, {}

    env = envelopes[rng.randrange(len(envelopes))]
    messages: List[Dict[str, Any]] = [{"role": "system", "content": env["system"]}]
    records: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {}
    spent = 0.0
    sent = 0
    session_id = os.path.basename(path).replace(".jsonl", "")[:12]
    pending: Dict[str, Dict[str, Any]] = {}

    for ev in events:
        if ev["kind"] == "user":
            messages.append({"role": "user", "content": CCS._clip_text(ev["text"], 6000)})
            continue
        if ev["kind"] == "tool_results":
            for cid, result in ev["results"].items():
                if cid in pending:
                    pending[cid]["content"] = CCS._clip_text(result)
            continue

        if args.max_turns and sent >= args.max_turns:
            break

        if messages[-1].get("role") in ("user", "tool"):
            model = rng.choice(args.models)
            body = {
                "model": model,
                "messages": CCS._trim_messages(messages),
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "usage": {"include": True},
            }
            if env["tools"]:
                body["tools"] = env["tools"]
                body["tool_choice"] = "auto"

            t0 = time.time()
            status, resp = post_openrouter(key, body, args.timeout)
            wall = time.time() - t0
            sent += 1

            if status != 200:
                msg = ((resp.get("error") or {}).get("message") or "")[:120]
                stats[f"http-{status}"] = stats.get(f"http-{status}", 0) + 1
                print(f"[sft:{session_id}] step={sent} HTTP {status} {msg}", flush=True)
                if status in (401, 403):
                    raise SystemExit("[sft] auth failed - check OPENROUTER_API_KEY")
                if status == 402:
                    raise SystemExit("[sft] out of credits")
                continue

            choice = (resp.get("choices") or [{}])[0]
            out = choice.get("message") or {}
            content = out.get("content") or ""
            tool_calls = out.get("tool_calls") or []
            finish = choice.get("finish_reason")
            usage = resp.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            spent += cost

            good, reason = is_good_sample(content, tool_calls, finish)
            stats[reason] = stats.get(reason, 0) + 1

            if good:
                target = {"role": "assistant"}
                if content:
                    target["content"] = CCS.scrub_secrets(content)
                if tool_calls:
                    target["tool_calls"] = tool_calls
                convo = CCS._trim_messages(messages) + [target]
                records.append({
                    "messages": convo,
                    "tools": env["tools"],
                    "target_index": len(convo) - 1,
                    "meta": {
                        "teacher": model,
                        "envelope": env["id"],
                        "session": session_id,
                        "source": path,
                        "finish_reason": finish,
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                        },
                        "cost_usd": cost,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "stack": stack_meta or {},
                    },
                })

            print(
                f"[sft:{session_id}] step={sent} {model.split('/')[-1][:22]:22s} "
                f"finish={finish} chars={len(content)} tools={len(tool_calls)} "
                f"{reason} ${cost:.5f} wall={wall:.1f}s",
                flush=True,
            )

            if spent >= budget:
                break
            if args.sleep:
                time.sleep(args.sleep)

        assistant = {"role": "assistant"}
        if ev["text"]:
            assistant["content"] = CCS._clip_text(ev["text"], 4000)
        elif not ev["tool_calls"]:
            assistant["content"] = ""
        if ev["tool_calls"]:
            assistant["tool_calls"] = ev["tool_calls"]
        messages.append(assistant)

        pending = {}
        for tc in ev["tool_calls"]:
            fn = tc.get("function") or {}
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": CCS.reconstruct_tool_result(
                    fn.get("name"), json.loads(fn.get("arguments") or "{}")
                ),
            }
            pending[tc["id"]] = tool_msg
            messages.append(tool_msg)

    return records, spent, stats


def main() -> int:
    p = argparse.ArgumentParser(description="Cursor-agent SFT dataset via OpenRouter")
    p.add_argument("--corpus", choices=("cursor", "subagents", "claude", "all"), default="all")
    p.add_argument("--sessions", type=int, default=5)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--budget-usd", type=float, default=5.0)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--api-key", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dry-run", action="store_true", help="build prompts, call nothing")
    p.add_argument("--stack", default=None,
                   help="target this environment's stack mix: natural|balanced|python|frontend|backend-api")
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]

    key = resolve_api_key(args.api_key)
    if not key and not args.dry_run:
        print(
            "[sft] no API key. Set OPENROUTER_API_KEY, pass --api-key, or write it to\n"
            "      ~/.config/openrouter/key",
            file=sys.stderr,
        )
        return 2

    corpora = ["cursor", "subagents", "claude"] if args.corpus == "all" else [args.corpus]
    envelopes = CCS.load_envelopes(include_variants=True)
    paths = CCS.list_transcript_paths(corpora)
    rng = random.Random(args.seed)
    stack_by_path: Dict[str, Dict[str, Any]] = {}

    if args.stack:
        sp = CCS._load_stack_profile()
        profs = sp.profile_corpus(paths, CCS.parse_transcript)
        picked = sp.select_sessions(profs, args.sessions + args.offset,
                                    args.stack, seed=args.seed)
        picked = picked[args.offset:]
        selected = [p["path"] for p in picked]
        stack_by_path = {
            p["path"]: {
                "dominant": p["dominant"],
                "project": p["project"],
                "frameworks": p["frameworks"],
                "categories": p["categories"],
            }
            for p in picked
        }
        print(f"[sft] stack={args.stack}\n{sp.summarize(picked)}", flush=True)
    else:
        rng.shuffle(paths)
        if args.offset:
            paths = paths[args.offset:]
        selected = paths[: args.sessions]

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"sft-{time.strftime('%Y-%m-%d')}.jsonl")

    print(
        f"[sft] corpus={args.corpus} sessions={len(selected)} envelopes={len(envelopes)} "
        f"models={len(args.models)} budget=${args.budget_usd:.2f} out={out_path}",
        flush=True,
    )

    if args.dry_run:
        total = 0
        for path in selected:
            ev = CCS.parse_transcript(path)
            n = sum(1 for e in ev if e["kind"] == "assistant")
            total += min(n, args.max_turns or n)
        print(f"[sft] dry-run: {total} prompts would be sent across {len(selected)} sessions")
        return 0

    spent = 0.0
    kept = 0
    agg: Dict[str, int] = {}
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, path in enumerate(selected):
            if spent >= args.budget_usd:
                print(f"[sft] budget ${args.budget_usd:.2f} reached, stopping", flush=True)
                break
            recs, cost, stats = build_session_records(
                path, args, envelopes, rng, key, args.budget_usd - spent,
                stack_meta=stack_by_path.get(path),
            )
            spent += cost
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            kept += len(recs)
            for k, v in stats.items():
                agg[k] = agg.get(k, 0) + v
            print(
                f"[sft] session {i+1}/{len(selected)} kept={len(recs)} total={kept} "
                f"spent=${spent:.4f} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    per = (spent / kept) if kept else 0.0
    print(f"\n[sft] kept={kept} spent=${spent:.4f} (${per:.5f}/sample)", flush=True)
    print(f"[sft] filter histogram: {agg}", flush=True)
    print(f"[sft] out: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
