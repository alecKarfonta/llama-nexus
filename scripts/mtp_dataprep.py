#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.MTP_DATAPREP | turn dataset_build.py shards into
token-exact training sequences for the qwen4exp NextN (MTP) draft head.

Input : <dataset>/train.jsonl, val.jsonl   (from scripts/dataset_build.py)
Output: <out>/{split}.tokens.bin  flat int32 token stream
        <out>/{split}.index.jsonl one row per sample (offsets + supervised span)
        <out>/prep_report.json    stats, alignment failures, token budgets

Why this stage exists
--------------------
dataset_build.py stores exact server-side `prompt_token_ids` but only *text*
for the completion, and that text is not what the model emitted:

  1. Tool calls are stored as `<tool_call>name(args)</tool_call>`, a shape
     invented by render_completion(). The real template emits an XML form:
         <tool_call>\\n<function=read_file>\\n<parameter=path>\\na.py\\n...
     ~80% of the corpus carries tool calls, so training on the stored render
     would teach the head a token sequence that never occurs in production.
  2. The prompt ends inside an open reasoning block (`<|im_start|>assistant\\n
     <think>\\n`), so the true continuation opens with `\\n</think>\\n\\n`
     before any content. The stored completion starts after that.

Both are fixed by asking the server's own chat template to render the
assistant turn, then joining in *token* space and proving the join did not
disturb the captured prompt ids.

The assistant turn cannot be rendered in trailing position — llama.cpp
refuses with "Cannot continue an assistant message that contains tool calls".
So the turn is rendered as *history*, parking a sentinel user message behind
it, and sliced back out. Verified identical with and without a `tools`
declaration, which matters because dataset_build.py does not keep the tools.

Supervision layout (nextn_predict_layers=1)
------------------------------------------
The head at position i consumes the trunk hidden state h_i plus the embedding
of t_{i+1} and predicts t_{i+1+nextn}. For nextn=1 that is t_{i+2}, so a
sample of N tokens supports positions i in [0, N-3]. Restricting targets to
completion tokens gives i in [n_prompt-2, N-3] — exactly n_completion
positions, one per token the model actually generated.

The emitted index doubles as the extraction manifest for the trunk
hidden-state pass: `sup_start`/`sup_end` are the only positions whose hidden
states a trainer ever needs, which is what keeps that pass affordable.

Usage:
  python3 scripts/mtp_dataprep.py --dataset /mnt/.../dataset-v2 --limit 32
  python3 scripts/mtp_dataprep.py --dataset /mnt/.../dataset-v2 \\
      --supervise completion --workers 4
"""

import argparse
import array
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

DEFAULT_DATASET = "/mnt/comfyui-models/llama-nexus/datasets/dataset-v2"
DEFAULT_BASE = "http://127.0.0.1:8600"

# [TIME] | INFO | SCRIPTS.MTP_DATAPREP | sentinel must not occur in any real
# assistant turn; it only exists to push that turn out of trailing position.
SENTINEL = "ZZ_MTP_DATAPREP_SENTINEL_ZZ"
SENTINEL_MARK = f"<|im_start|>user\n{SENTINEL}"
STUB = [{"role": "user", "content": "PROBE"}]


class Server:
    """Thin client over llama-server's tokenizer + template endpoints."""

    def __init__(self, base, key, timeout=120):
        self.base = base.rstrip("/")
        self.key = key
        self.timeout = timeout
        self._local = threading.local()

    def _post(self, path, payload):
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base + path, data=data, headers=headers)
        for attempt in (1, 2, 3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf8", "replace")
                raise RuntimeError(f"{path} HTTP {e.code}: {body[:200]}") from None
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(0.5 * attempt)

    def tokenize(self, text):
        return self._post("/tokenize", {"content": text})["tokens"]

    def detokenize(self, tokens):
        return self._post("/detokenize", {"tokens": list(tokens)})["content"]

    def apply_template(self, messages):
        return self._post("/apply-template", {"messages": messages})["prompt"]


def assistant_message(row):
    """Rebuild the assistant turn as the API originally returned it."""
    msg = {"role": "assistant", "content": row.get("completion_text") or ""}
    calls = []
    for i, tc in enumerate(row.get("tool_calls") or []):
        fn = (tc or {}).get("function") or {}
        # [TIME] | WARN | SCRIPTS.MTP_DATAPREP.ASSISTANT_MESSAGE | the template
        # hard-errors with "Missing tool call type" when type/id are absent, so
        # backfill them rather than dropping the sample.
        calls.append({
            "id": (tc or {}).get("id") or f"call_{i}",
            "type": (tc or {}).get("type") or "function",
            "function": {
                "name": fn.get("name") or "unknown",
                "arguments": fn.get("arguments") if fn.get("arguments") is not None else "{}",
            },
        })
    if calls:
        msg["tool_calls"] = calls
    return msg


class Renderer:
    """Renders an assistant turn to its native text via the server template."""

    def __init__(self, server):
        self.server = server
        base = server.apply_template(STUB)
        cut = base.rindex("<|im_start|>assistant")
        # Everything before the generation prompt is shared by stub renders and
        # gets sliced off; the generation prompt itself is already the tail of
        # every captured prompt, so the continuation starts after it.
        self.head = base[:cut]
        self.gen_prompt = base[cut:]

    def continuation_text(self, row):
        """Native text the model emitted after the captured prompt."""
        msgs = STUB + [assistant_message(row), {"role": "user", "content": SENTINEL}]
        full = self.server.apply_template(msgs)
        if not full.startswith(self.head) or SENTINEL_MARK not in full:
            raise RuntimeError("template render did not match expected layout")
        turn = full[len(self.head):full.index(SENTINEL_MARK)]
        if not turn.startswith(self.gen_prompt):
            raise RuntimeError("assistant turn missing the generation prompt")
        return turn[len(self.gen_prompt):]


def join_tokens(server, prompt_ids, cont_text, window):
    """Tokenize the continuation so it splices onto prompt_ids exactly.

    The prompt ends on a newline and the continuation opens with one, so the
    naive concat can merge them into a single token and silently shift every
    downstream position. Re-tokenize a short prompt tail together with the
    continuation and require the tail to come back unchanged; escalate to the
    whole prompt when a small window is itself ambiguous.

    Returns (completion_ids, n_prompt_effective, flag).
    """
    for w in (min(window, len(prompt_ids)), len(prompt_ids)):
        tail = list(prompt_ids[-w:])
        joined = server.tokenize(server.detokenize(tail) + cont_text)
        if joined[:w] == tail:
            return joined[w:], len(prompt_ids), "clean"
        # Locate the divergence. Late divergence is a real merge at the seam;
        # early divergence means the window sliced mid-token and the tail text
        # simply retokenizes differently, so widen and retry.
        k = 0
        while k < w and k < len(joined) and joined[k] == tail[k]:
            k += 1
        if k >= w - 2 and w > 2:
            # The seam consumed the last (w - k) prompt tokens; hand the trainer
            # a shorter prompt so prompt_ids + completion_ids stays exact.
            return joined[k:], len(prompt_ids) - (w - k), "seam-merge"
        if w == len(prompt_ids):
            return joined[k:], k, "boundary-drift"
    raise RuntimeError("unreachable")


def prepare_row(row, server, renderer, args):
    prompt_ids = row.get("prompt_token_ids") or []
    if len(prompt_ids) < 8:
        return None, "drop:short-prompt"
    try:
        cont_text = renderer.continuation_text(row)
    except RuntimeError as e:
        return None, f"drop:render-{str(e).split()[0].lower()}"
    if not cont_text.strip():
        return None, "drop:empty-continuation"
    completion_ids, n_prompt, flag = join_tokens(
        server, prompt_ids, cont_text, args.boundary_window
    )
    if not completion_ids:
        return None, "drop:empty-completion-ids"

    n_total = n_prompt + len(completion_ids)
    # Head at i predicts t_{i+1+nextn}; the last usable i leaves that in range.
    last = n_total - 1 - args.nextn - 1
    first = 0 if args.supervise == "all" else max(0, n_prompt - 1 - args.nextn)
    if last < first:
        return None, "drop:no-supervised-positions"

    return {
        "tokens": list(prompt_ids[:n_prompt]) + list(completion_ids),
        "n_prompt": n_prompt,
        "n_completion": len(completion_ids),
        "sup_start": first,
        "sup_end": last,
        "n_supervised": last - first + 1,
        "join": flag,
        "sample_key": row.get("sample_key"),
        "prefix_key": row.get("prefix_key"),
        "completion_kind": row.get("completion_kind"),
        "tools_present": row.get("tools_present"),
    }, f"kept:{flag}"


def run_split(name, src_path, n_rows, server, renderer, args, out_dir, stats):
    bin_path = os.path.join(out_dir, f"{name}.tokens.bin")
    idx_path = os.path.join(out_dir, f"{name}.index.jsonl")
    lock = threading.Lock()
    offset = 0
    done = 0
    totals = Counter()

    with open(bin_path, "wb") as fbin, open(idx_path, "w", encoding="utf-8") as fidx:
        def work(row):
            nonlocal offset, done
            try:
                rec, tag = prepare_row(row, server, renderer, args)
            except Exception as e:
                rec, tag = None, f"drop:error-{type(e).__name__}"
                if args.verbose:
                    print(f"  ! {e}", file=sys.stderr)
            with lock:
                stats[tag] += 1
                done += 1
                if rec is not None:
                    buf = array.array("i", rec["tokens"])
                    if sys.byteorder != "little":
                        buf.byteswap()
                    fbin.write(buf.tobytes())
                    entry = {k: v for k, v in rec.items() if k != "tokens"}
                    entry["offset"] = offset
                    entry["n_tokens"] = len(rec["tokens"])
                    fidx.write(json.dumps(entry) + "\n")
                    offset += len(rec["tokens"])
                    totals["tokens"] += len(rec["tokens"])
                    totals["supervised"] += rec["n_supervised"]
                    totals["samples"] += 1
                if done % 500 == 0 or done == n_rows:
                    print(f"  {name}: {done}/{n_rows} kept={totals['samples']}",
                          file=sys.stderr, flush=True)

        # [TIME] | INFO | SCRIPTS.MTP_DATAPREP.RUN_SPLIT | stream in chunks. A
        # full read is ~8 GB resident here: 212M prompt ids become boxed Python
        # ints, 36 bytes apiece, for a 1.2 GB file on disk.
        pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
        try:
            for chunk in iter_chunks(src_path, args.chunk, args.limit):
                if pool is not None:
                    list(pool.map(work, chunk))
                else:
                    for row in chunk:
                        work(row)
        finally:
            if pool is not None:
                pool.shutdown()

    return {
        "samples": totals["samples"],
        "tokens": totals["tokens"],
        "supervised_tokens": totals["supervised"],
        "tokens_bin": os.path.basename(bin_path),
        "index": os.path.basename(idx_path),
        "bytes": offset * 4,
    }


def verify(name, dataset, out_dir, server, args):
    """Re-check written artifacts against the captured ids, independently.

    The prep pass proves the seam locally; this proves the *files* — offsets
    land where the index says, the captured prompt still prefixes the stored
    sequence, the sequence is its own retokenization, and every supervised
    position has a target inside the sequence.
    """
    idx_path = os.path.join(out_dir, f"{name}.index.jsonl")
    bin_path = os.path.join(out_dir, f"{name}.tokens.bin")
    if not (os.path.exists(idx_path) and os.path.exists(bin_path)):
        return {"error": f"{name}: artifacts missing"}
    index = [json.loads(l) for l in open(idx_path, encoding="utf-8") if l.strip()]
    n_words = os.path.getsize(bin_path) // 4
    captured = {}
    want = {e["sample_key"] for e in random_sample(index, args.verify)}
    for chunk in iter_chunks(os.path.join(dataset, f"{name}.jsonl"), 256, 0):
        for row in chunk:
            if row.get("sample_key") in want:
                captured[row["sample_key"]] = row["prompt_token_ids"]
        if len(captured) == len(want):
            break

    fails = Counter()
    # Contiguity is a whole-file property, so check every row for it.
    expect = 0
    for e in index:
        if e["offset"] != expect:
            fails["offset-gap"] += 1
        expect = e["offset"] + e["n_tokens"]
        if e["n_prompt"] + e["n_completion"] != e["n_tokens"]:
            fails["length-mismatch"] += 1
        if e["sup_end"] > e["n_tokens"] - 2 - args.nextn or e["sup_start"] < 0:
            fails["supervised-out-of-range"] += 1
    if expect != n_words:
        fails["bin-size-mismatch"] += 1

    checked = 0
    with open(bin_path, "rb") as fbin:
        for e in index:
            if e["sample_key"] not in captured:
                continue
            fbin.seek(e["offset"] * 4)
            buf = array.array("i")
            buf.frombytes(fbin.read(e["n_tokens"] * 4))
            if sys.byteorder != "little":
                buf.byteswap()
            seq = list(buf)
            cap = captured[e["sample_key"]]
            if seq[:e["n_prompt"]] != cap[:e["n_prompt"]]:
                fails["prompt-prefix-diverged"] += 1
            if server.tokenize(server.detokenize(seq)) != seq:
                fails["retokenize-diverged"] += 1
            checked += 1

    return {
        "rows": len(index),
        "deep_checked": checked,
        "bin_tokens": n_words,
        "failures": dict(fails) or "none",
    }


def plan_extraction(name, out_dir, nextn):
    """Order samples so the trunk prefill can be reused, and emit the plan.

    Extracting h means running the 125B trunk over every sample, and a naive
    pass pays full prefill each time: 213M tokens. But samples are clustered —
    813 sessions, median 8 turns — and turn N+1's prompt begins with turn N's
    whole sequence. Walking a session in length order lets llama.cpp keep the
    KV cache and decode only the delta.

    All the planning lives here rather than in the C++ tool: the token data is
    already at hand, and a plan is far easier to verify in Python than inside
    a binary that needs a CUDA rebuild to run. The tool just executes it.
    """
    import numpy as np

    idx_path = os.path.join(out_dir, f"{name}.index.jsonl")
    bin_path = os.path.join(out_dir, f"{name}.tokens.bin")
    index = [json.loads(l) for l in open(idx_path, encoding="utf-8") if l.strip()]
    toks = np.memmap(bin_path, dtype="<i4", mode="r")

    # [TIME] | INFO | SCRIPTS.MTP_DATAPREP.PLAN_EXTRACTION | order the whole
    # split lexicographically by token id. Sorting strings lexicographically
    # maximises the shared prefix of adjacent pairs, which is exactly what the
    # KV cache can keep. Measured on dataset-v2: naive 213M tokens, grouping by
    # session and sorting by length 57.7M, this 42.8M. Costs ~850 MB of sort
    # keys, which is cheaper than the hours it saves.
    ordered = sorted(index, key=lambda e: toks[e["offset"]:e["offset"] + e["n_tokens"]].tobytes())

    plan_path = os.path.join(out_dir, f"{name}.extract_plan.tsv")
    naive = decoded = 0
    reused_samples = 0
    prev = None
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write("# offset\tn_tokens\tn_prompt\tsup_start\tsup_end\tn_reuse\n")
        for e in ordered:
            seq = toks[e["offset"]:e["offset"] + e["n_tokens"]]
            n_reuse = 0
            if prev is not None:
                n = min(len(prev), len(seq))
                if n:
                    # first mismatch, or the whole overlap if identical
                    diff = np.flatnonzero(prev[:n] != seq[:n])
                    n_reuse = int(diff[0]) if diff.size else n
            # Never reuse across a supervised position: those rows must be
            # freshly computed or they never get written.
            n_reuse = min(n_reuse, e["sup_start"])
            naive += e["n_tokens"]
            decoded += e["n_tokens"] - n_reuse
            reused_samples += 1 if n_reuse > 0 else 0
            f.write(f"{e['offset']}\t{e['n_tokens']}\t{e['n_prompt']}\t"
                    f"{e['sup_start']}\t{e['sup_end']}\t{n_reuse}\n")
            prev = np.asarray(seq)

    sup = sum(e["n_supervised"] for e in index)
    return {
        "plan": os.path.basename(plan_path),
        "order": "lexicographic-by-token",
        "samples": len(index),
        "sessions": len({e["prefix_key"] for e in index}),
        "naive_decode_tokens": naive,
        "planned_decode_tokens": decoded,
        "saving": f"{naive / decoded:.1f}x" if decoded else "n/a",
        "samples_with_reuse": reused_samples,
        "max_seq_tokens": max(e["n_tokens"] for e in index),
        "supervised_rows": sup,
        "h_bytes_fp16": sup * 10240 * 2,
    }


def random_sample(seq, k):
    import random as _r
    if k <= 0 or k >= len(seq):
        return list(seq)
    return _r.Random(0).sample(list(seq), k)


def iter_chunks(path, chunk, limit):
    """Yield lists of parsed rows, never holding more than `chunk` at once."""
    rows, seen = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            seen += 1
            if len(rows) >= chunk:
                yield rows
                rows = []
            if limit and seen >= limit:
                break
    if rows:
        yield rows


def count_rows(path, limit):
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
                if limit and n >= limit:
                    break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out", default=None, help="default: <dataset>/mtp-prep")
    ap.add_argument("--base", default=DEFAULT_BASE, help="llama-server base URL")
    ap.add_argument("--api-key", default=os.environ.get("LLAMACPP_API_KEY", ""))
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--supervise", choices=["completion", "all"], default="completion",
                    help="completion: loss only on generated tokens (default). "
                         "all: every position, ~250x more signal but teaches "
                         "prediction of user-authored context too")
    ap.add_argument("--nextn", type=int, default=1,
                    help="nextn_predict_layers of the head (qwen4exp MTP = 1)")
    ap.add_argument("--boundary-window", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=256, help="rows held in memory")
    ap.add_argument("--limit", type=int, default=0, help="first N rows per split")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="verify existing output instead of building: deep-check "
                         "N random samples plus whole-file structure")
    ap.add_argument("--plan", action="store_true",
                    help="emit the trunk hidden-state extraction plan "
                         "(prefix-reuse ordering) instead of building")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.dataset, "mtp-prep")
    os.makedirs(out_dir, exist_ok=True)
    splits = ["train", "val"] if args.split == "both" else [args.split]

    # Planning reads only the artifacts already on disk, so it must not require
    # the inference server — extraction runs with llama-server stopped.
    if args.plan:
        t0 = time.time()
        result = {name: plan_extraction(name, out_dir, args.nextn) for name in splits}
        print(json.dumps({"extract_plan": result,
                          "elapsed_s": round(time.time() - t0, 1)}, indent=2))
        return 0

    server = Server(args.base, args.api_key)
    try:
        renderer = Renderer(server)
    except Exception as e:
        print(f"cannot reach the template endpoint on {args.base}: {e}", file=sys.stderr)
        print("the harness needs a live llama-server with the target model "
              "loaded (its tokenizer and chat template are the ground truth).",
              file=sys.stderr)
        return 2
    print(f"template ok; generation prompt = {renderer.gen_prompt!r}", file=sys.stderr)

    stats = Counter()
    per_split = {}
    t0 = time.time()

    if args.verify:
        result = {name: verify(name, args.dataset, out_dir, server, args)
                  for name in splits}
        print(json.dumps({"verify": result,
                          "elapsed_s": round(time.time() - t0, 1)}, indent=2))
        return 0

    for name in splits:
        path = os.path.join(args.dataset, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"  skip {name}: {path} missing", file=sys.stderr)
            continue
        n_rows = count_rows(path, args.limit)
        print(f"  {name}: {n_rows} rows", file=sys.stderr, flush=True)
        per_split[name] = run_split(name, path, n_rows, server, renderer,
                                    args, out_dir, stats)

    report = {
        "dataset": args.dataset,
        "out": out_dir,
        "supervise": args.supervise,
        "nextn": args.nextn,
        "boundary_window": args.boundary_window,
        "generation_prompt": renderer.gen_prompt,
        "splits": per_split,
        "stats": dict(sorted(stats.items())),
        "elapsed_s": round(time.time() - t0, 1),
        "token_dtype": "int32-le",
        "note": ("index rows give sup_start/sup_end — the only trunk positions "
                 "whose hidden states the head consumes, i.e. the manifest for "
                 "the hidden-state extraction pass"),
    }
    with open(os.path.join(out_dir, "prep_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
