#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.SWEEP_DRAFT_KNOBS | sweep --spec-draft-p-min / -n-min for a
draft head and report accepted tokens per verify round, which is what actually
buys throughput.

Acceptance ratio on its own is misleading. The draft-mtp loop stops at the first
token whose top-1 probability falls under p_min (common/speculative.cpp:1636), so
a better-calibrated head can post a higher acceptance ratio while drafting shorter
runs and delivering *fewer* accepted tokens per round. Since p_min is an absolute
threshold on the draft head's own softmax, it has to be retuned whenever the head
changes -- a value tuned for an overconfident head truncates a calibrated one early.

Per-request speculative overrides are compiled out upstream (`#if 0` in
tools/server/server-schema.cpp), so every sweep point costs a server restart.

Usage:
  python3 scripts/sweep_draft_knobs.py --head stock   --p-min 0.75 0.5 0.3
  python3 scripts/sweep_draft_knobs.py --head trained --p-min 0.75 0.5 0.3 0.1
  python3 scripts/sweep_draft_knobs.py --head trained --p-min 0.5 --n-min 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acceptance_log import LINE_RE  # noqa: E402  single source of truth for the log format

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8700")
API_KEY = os.getenv("API_KEY", "placeholder-api-key")
LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8600")

HEADS = {
    "stock": "Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf",
    "trained": "Qwen3.8-Flash-Next-MTP-lora-r32-Q4_K_M.gguf",
    "kstep": "Qwen3.8-Flash-Next-MTP-kstep-Q4_K_M.gguf",
    "kstep_e2": "Qwen3.8-Flash-Next-MTP-kstep-e2-Q4_K_M.gguf",
    "kstep_ss": "Qwen3.8-Flash-Next-MTP-kstep-ss-Q4_K_M.gguf",
    "kstep_pmin": "Qwen3.8-Flash-Next-MTP-kstep-pmin-Q4_K_M.gguf",
}

# Same three prompts as scripts/bench_tps.sh so numbers stay comparable.
WORKLOADS = {
    "prose": "Write a detailed 300-word story about a lighthouse keeper.",
    "code": "Write a Python module with three classes Sensor, Actuator, Device. "
            "Each has __init__, read, calibrate, __repr__. Consistent style.",
    "json": "Output a JSON array of 30 sensor objects, each with fields id, name, "
            "unit, min, max, sampling_hz. No prose, JSON only.",
}


def api(method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BACKEND}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def llama_key():
    return api("GET", "/api/v1/service/config")["config"]["server"]["api_key"]


def container_state():
    """(id, args string, health) for the container currently holding the name.

    Parsed from the inspect JSON rather than a --format template: `join .Args`
    errors out with "expected []string; got []interface {}" on this container,
    and a template failure is indistinguishable from a missing container.
    """
    p = subprocess.run(["docker", "inspect", "llamacpp-api"],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        return "", "", "missing"
    try:
        c = json.loads(p.stdout)[0]
    except (ValueError, IndexError, KeyError):
        return "", "", "missing"
    args = " ".join(str(a) for a in (c.get("Args") or []))
    health = (c.get("State", {}).get("Health") or {}).get("Status", "none")
    return c.get("Id", ""), args, health


def container_id():
    return container_state()[0]


def accept_lines(since=None):
    """Parsed (acc, accepted, generated, mean_len) tuples from the container log.

    Reads the whole log on purpose. `docker logs` only ever shows the container
    currently holding the name, and a restart replaces it, so a time window buys
    nothing while a stale window silently reads someone else's requests. A freshly
    restarted container's log contains exactly this sweep point.
    """
    cmd = ["docker", "logs", "llamacpp-api"]
    if since:
        cmd += ["--since", since]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = []
    for line in (p.stderr + p.stdout).splitlines():
        m = LINE_RE.search(line)
        if m:
            out.append((float(m.group("acc")), int(m.group("accepted")),
                        int(m.group("generated")), float(m.group("meanlen"))))
    return out


def flag(args, name):
    """Value of `--name X` inside a docker Args string, or None."""
    parts = args.split()
    for i, tok in enumerate(parts):
        if tok == f"--{name}" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def wait_for_point(head, p_min, n_min, prev_id=None, timeout=1200):
    """Block until the container actually serving is the one we asked for.

    POST /service/restart is asynchronous: it returns while the outgoing server
    is still answering requests. Probing for health alone therefore succeeds
    against the *previous* container, whose logs are then discarded when the
    name is rebound -- which reads as a point that ran fine and produced no
    acceptance lines at all. Each point has a distinct p_min, so the command
    line is an exact discriminator.
    """
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        cid, args, health = container_state()
        got_p = flag(args, "spec-draft-p-min")
        got_n = flag(args, "spec-draft-n-min")
        got_d = (flag(args, "model-draft") or "").rsplit("/", 1)[-1]
        ok = (cid and cid != prev_id and got_d == HEADS[head]
              and health == "healthy"
              and got_p is not None and abs(float(got_p) - p_min) < 1e-9
              and got_n is not None and int(got_n) == n_min)
        if ok:
            probe = _wait_serving(timeout=timeout - (time.time() - t0))
            if container_id() == cid:  # not swapped out from under the probe
                return cid, time.time() - t0, probe
            continue
        msg = f"draft={got_d} p_min={got_p} n_min={got_n} health={health}"
        if msg != last:
            print(f"    waiting ({time.time() - t0:.0f}s): {msg}")
            last = msg
        time.sleep(10)
    raise TimeoutError(f"container never reached head={head} p_min={p_min} n_min={n_min}")


def _wait_serving(timeout=900):
    """Block until the server answers a trivial completion, not just /status.

    A 104 GiB model reports 'running' before the draft head is resident, and a
    request issued in that window returns without any speculative log line at
    all, which would silently drop a sweep point.
    """
    t0 = time.time()
    key = None
    last = None
    while time.time() - t0 < timeout:
        try:
            if key is None:
                key = llama_key()
            body = json.dumps({"model": "qwen38-flash", "max_tokens": 1,
                               "temperature": 0.0,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(
                f"{LLAMA_URL}/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json",
                         "User-Agent": "llama-nexus-sweep/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                json.load(r)
            return time.time() - t0
        except Exception as e:
            # A swallowed error here looks identical to a slow model load, which
            # burns the whole timeout before saying anything useful.
            msg = f"{type(e).__name__}: {e}"
            if msg != last:
                print(f"    waiting ({time.time() - t0:.0f}s): {msg}")
                last = msg
            time.sleep(10)
    raise TimeoutError(f"server not serving after {timeout}s")


def one_request(prompt, key, max_tokens):
    """Fire one request, return (tok_per_s, acceptance tuple or None).

    Slots are serialised (parallel_slots=1), so the acceptance line that appears
    after this call belongs to this call.
    """
    n0 = len(accept_lines())
    body = json.dumps({
        "model": "qwen38-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "llama-nexus-sweep/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.load(resp)
    tps = r.get("timings", {}).get("predicted_per_second", 0.0)

    lines = accept_lines()
    for _ in range(10):  # the log line is flushed just after the response
        if len(lines) > n0:
            break
        time.sleep(1)
        lines = accept_lines()
    return tps, (lines[-1] if len(lines) > n0 else None)


def measure(key, runs, max_tokens, cid):
    """Run every workload and aggregate. Warm run is discarded."""
    per = {}
    for name, prompt in WORKLOADS.items():
        one_request(prompt, key, max_tokens)  # warm
        rows, tpss = [], []
        for _ in range(runs):
            tps, acc = one_request(prompt, key, max_tokens)
            tpss.append(tps)
            if acc:
                rows.append(acc)
        if container_id() != cid:
            raise RuntimeError("container was replaced mid-measurement; "
                               "results for this point are not trustworthy")
        if not rows:
            print(f"    WARN {name}: no acceptance lines -- speculation off?")
            continue
        accepted = sum(r[1] for r in rows)
        generated = sum(r[2] for r in rows)
        acc = accepted / generated if generated else 0.0
        mlen = sum(r[3] for r in rows) / len(rows)
        per[name] = {
            "acceptance": acc,
            "draft_len": mlen,
            "accepted_per_round": acc * mlen,
            "tok_s": sum(tpss) / len(tpss),
            "accepted": accepted,
            "generated": generated,
            "n": len(rows),
        }
        print(f"    {name:5s} acc={acc:.3f}  len={mlen:5.2f}  "
              f"acc/round={acc * mlen:5.2f}  {per[name]['tok_s']:6.1f} tok/s")
    return per


def apply_point(head, p_min, n_min):
    api("PUT", "/api/v1/service/config", {
        "speculative": {"model_draft": HEADS[head], "gpu_layers_draft": 999},
        "mtp": {"draft_p_min": p_min, "draft_n_min": n_min},
    })
    api("POST", "/api/v1/service/restart", timeout=600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=sorted(HEADS), default="trained")
    ap.add_argument("--p-min", type=float, nargs="+", default=[0.75, 0.5, 0.3])
    ap.add_argument("--n-min", type=int, default=0)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--out", default="/tmp/draft_knob_sweep.json")
    ap.add_argument("--no-restore", action="store_true",
                    help="leave the last point live; for chaining sweeps back to "
                         "back without paying a restart to restore in between")
    args = ap.parse_args()

    original = api("GET", "/api/v1/service/config")["config"]
    orig_draft = original.get("speculative", {}).get("model_draft")
    orig_mtp = original.get("mtp", {})
    print(f"[TIME] | INFO | SWEEP | restoring on exit: draft={orig_draft} "
          f"p_min={orig_mtp.get('draft_p_min')} n_min={orig_mtp.get('draft_n_min')}")

    results, key = [], None
    try:
        for p_min in args.p_min:
            print(f"\n=== {args.head} p_min={p_min} n_min={args.n_min} ===")
            t0 = time.time()
            prev_id = container_id()
            apply_point(args.head, p_min, args.n_min)
            cid, up, probe = wait_for_point(args.head, p_min, args.n_min, prev_id)
            key = llama_key()
            print(f"    up in {up:.0f}s (serving probe {probe:.0f}s), "
                  f"container {cid[:12]}")
            per = measure(key, args.runs, args.max_tokens, cid)
            if per:
                struct = sum(per[w]["accepted_per_round"] for w in ("code", "json")
                             if w in per) / 2
                print(f"    STRUCT acc/round = {struct:.2f}   "
                      f"(point took {time.time() - t0:.0f}s)")
            results.append({"head": args.head, "p_min": p_min,
                            "n_min": args.n_min, "workloads": per})
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    finally:
        if args.no_restore:
            # No `return` here: that would swallow an in-flight exception.
            print(f"\n[TIME] | WARN | SWEEP | --no-restore: leaving "
                  f"{args.head} p_min={results[-1]['p_min'] if results else '?'} live")
        else:
            print(f"\n[TIME] | INFO | SWEEP | restoring original config")
            try:
                api("PUT", "/api/v1/service/config", {
                    "speculative": {"model_draft": orig_draft,
                                    "gpu_layers_draft": 999},
                    "mtp": {"draft_p_min": orig_mtp.get("draft_p_min", 0.75),
                            "draft_n_min": orig_mtp.get("draft_n_min", 0)},
                })
                api("POST", "/api/v1/service/restart", timeout=600)
                _wait_serving()
                print(f"    restored: {orig_draft} "
                      f"p_min={orig_mtp.get('draft_p_min')}")
            except Exception as e:
                print(f"    RESTORE FAILED: {e} -- fix the draft head manually")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
