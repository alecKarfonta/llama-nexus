#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.STACK_PROFILE | classify agent transcript sessions by the
stack they exercise, so dataset builds can target the work actually done in this
environment instead of sampling 937 sessions uniformly.

Measured working distribution (160-session sample of tool_use arguments):
  .py 58%   .tsx 15%   .ts 10%   .md/.js/.json ~4% each   .yml/.sh minor
  docker is the dominant shell verb; then python3, curl, git, uv
  projects: stocker 55%, browser 20%, graphrag 10%, Walker 10%

Used by capture_cursor_synth.py (MTP self-distillation) and
openrouter_sft_build.py (teacher SFT) via --stack.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

EXT_RE = re.compile(r"\.([A-Za-z0-9]{1,5})$")
PROJECT_RE = re.compile(r"/home/alec/git/([^/\s]+)")

# Framework fingerprints drawn from this environment's manifests.
FRAMEWORK_SIGNALS: Dict[str, Tuple[str, ...]] = {
    "fastapi": ("fastapi", "uvicorn", "pydantic", "starlette", "APIRouter"),
    "pytest": ("pytest", "conftest", "test_", "assert "),
    "ml": ("torch", "transformers", "sentence_transformers", "sklearn",
           "numpy", "scipy", "pandas", "peft", "lora"),
    "vectordb": ("qdrant", "redis", "neo4j", "embedding", "graphrag"),
    "react": ("react", "useState", "useEffect", "tsx", "jsx"),
    "mui": ("@mui", "@emotion", "MuiButton", "ThemeProvider"),
    "viz": ("recharts", "three", "d3", "chart", "force-graph"),
    "docker": ("docker", "compose", "Dockerfile", "kubectl", "container"),
    "llm": ("llama", "gguf", "vllm", "openai", "chat/completions", "tokeniz"),
}

# Category -> file extensions that identify it.
CATEGORY_EXTS: Dict[str, Tuple[str, ...]] = {
    "python": ("py",),
    "frontend": ("tsx", "ts", "jsx", "css", "html"),
    "infra": ("yml", "yaml", "sh", "conf", "toml", "dockerfile"),
    "docs": ("md", "txt", "rst"),
    "data": ("json", "jsonl", "csv", "sql"),
    "other": (),
}
_EXT_TO_CATEGORY = {
    ext: cat for cat, exts in CATEGORY_EXTS.items() for ext in exts
}

# Named target mixes over CATEGORY_EXTS keys.
PROFILES: Dict[str, Dict[str, float]] = {
    # mirrors the measured distribution above
    "natural": {"python": 0.58, "frontend": 0.25, "infra": 0.07,
                "data": 0.05, "docs": 0.05},
    # keeps python dominant but guarantees minority categories are not starved
    "balanced": {"python": 0.45, "frontend": 0.30, "infra": 0.15,
                 "data": 0.05, "docs": 0.05},
    "python": {"python": 1.0},
    "frontend": {"frontend": 1.0},
    "backend-api": {"python": 0.75, "infra": 0.15, "data": 0.10},
}
DEFAULT_PROFILE = "natural"


def categorize_ext(ext: str) -> str:
    return _EXT_TO_CATEGORY.get(ext.lower(), "other")


def profile_session(path: str, parse_transcript) -> Dict[str, Any]:
    """[TIME] | INFO | STACK_PROFILE.profile_session | stack fingerprint for one file.

    ``parse_transcript`` is injected so this module stays independent of the
    replay script that owns transcript parsing.
    """
    exts: Counter = Counter()
    projects: Counter = Counter()
    commands: Counter = Counter()
    frameworks: set = set()
    turns = 0

    try:
        events = parse_transcript(path)
    except Exception:  # noqa: BLE001 - a bad transcript must not kill a build
        events = []

    for ev in events:
        if ev.get("kind") != "assistant":
            continue
        turns += 1
        blob_parts: List[str] = [ev.get("text") or ""]
        for tc in ev.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            blob_parts.append(raw_args)
            try:
                args = json.loads(raw_args)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(args, dict):
                continue
            for field in ("path", "file_path", "target_file", "target_directory"):
                val = args.get(field)
                if not isinstance(val, str) or not val:
                    continue
                m = EXT_RE.search(val)
                if m:
                    exts[m.group(1).lower()] += 1
                pm = PROJECT_RE.search(val)
                if pm:
                    projects[pm.group(1)] += 1
            cmd = args.get("command")
            if isinstance(cmd, str) and cmd.strip():
                commands[cmd.strip().split()[0].split("/")[-1]] += 1
        blob = " ".join(blob_parts).lower()
        for name, needles in FRAMEWORK_SIGNALS.items():
            if name in frameworks:
                continue
            if any(n.lower() in blob for n in needles):
                frameworks.add(name)

    cats: Counter = Counter()
    for ext, n in exts.items():
        cats[categorize_ext(ext)] += n
    known = {c: n for c, n in cats.items() if c != "other"}
    dominant = max(known, key=known.get) if known else "other"

    return {
        "path": path,
        "turns": turns,
        "exts": dict(exts),
        "categories": dict(cats),
        "dominant": dominant,
        "project": (projects.most_common(1)[0][0] if projects else "unknown"),
        "frameworks": sorted(frameworks),
        "commands": dict(commands.most_common(8)),
        "signal": sum(known.values()),
    }


def profile_corpus(paths: Iterable[str], parse_transcript,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out = []
    for i, p in enumerate(paths):
        if limit and i >= limit:
            break
        prof = profile_session(p, parse_transcript)
        if prof["turns"]:
            out.append(prof)
    return out


def select_sessions(profiles: List[Dict[str, Any]], n: int,
                    profile_name: str = DEFAULT_PROFILE,
                    seed: int = 1234,
                    min_signal: int = 3) -> List[Dict[str, Any]]:
    """[TIME] | INFO | STACK_PROFILE.select_sessions | quota-sample to a target mix.

    Buckets sessions by dominant category and draws each bucket's quota. Buckets
    that cannot fill their quota hand the remainder back to the largest bucket,
    so the result is always ``n`` sessions when the corpus is big enough.
    """
    target = PROFILES.get(profile_name) or PROFILES[DEFAULT_PROFILE]
    rng = random.Random(seed)

    usable = [p for p in profiles if p["signal"] >= min_signal]
    if not usable:
        usable = list(profiles)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for p in usable:
        buckets.setdefault(p["dominant"], []).append(p)
    for b in buckets.values():
        rng.shuffle(b)
        # richer sessions first inside a bucket
        b.sort(key=lambda x: -x["turns"])

    picked: List[Dict[str, Any]] = []
    shortfall = 0
    for cat, frac in sorted(target.items(), key=lambda kv: -kv[1]):
        quota = int(round(n * frac))
        avail = buckets.get(cat, [])
        take = avail[:quota]
        picked.extend(take)
        shortfall += quota - len(take)

    if shortfall > 0:
        used = {p["path"] for p in picked}
        pool = [p for p in usable if p["path"] not in used]
        pool.sort(key=lambda x: -x["turns"])
        picked.extend(pool[:shortfall])

    rng.shuffle(picked)
    return picked[:n]


def summarize(profiles: List[Dict[str, Any]]) -> str:
    cats = Counter(p["dominant"] for p in profiles)
    projects = Counter(p["project"] for p in profiles)
    fw = Counter(f for p in profiles for f in p["frameworks"])
    turns = sum(p["turns"] for p in profiles)
    lines = [
        f"sessions={len(profiles)} assistant_turns={turns}",
        f"  dominant category: {dict(cats.most_common())}",
        f"  projects:          {dict(projects.most_common(8))}",
        f"  frameworks:        {dict(fw.most_common(10))}",
    ]
    return "\n".join(lines)


def _cli() -> int:
    import argparse
    import importlib.util

    ap = argparse.ArgumentParser(description="profile agent transcripts by stack")
    ap.add_argument("--corpus", default="all",
                    choices=("cursor", "subagents", "claude", "all"))
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES))
    ap.add_argument("--sessions", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="cap sessions profiled")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "capture_cursor_synth", os.path.join(here, "capture_cursor_synth.py"))
    ccs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ccs)

    corpora = (["cursor", "subagents", "claude"]
               if args.corpus == "all" else [args.corpus])
    paths = ccs.list_transcript_paths(corpora)
    print(f"profiling {len(paths)} transcripts ...", flush=True)
    profs = profile_corpus(paths, ccs.parse_transcript,
                           limit=args.limit or None)
    print("\n=== whole corpus ===")
    print(summarize(profs))
    picked = select_sessions(profs, args.sessions, args.profile)
    print(f"\n=== selected {len(picked)} with profile '{args.profile}' ===")
    print(summarize(picked))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
