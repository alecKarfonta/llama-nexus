#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.MTP_EXTRACT_H | run the trunk over the prepared token
sequences and write the NextN hidden state h for every supervised position.

This is the input side of draft-head training. The shipped NextN block consumes
two things per position: the embedding of the next token (cheap, the trainer
looks it up) and h, the trunk's wide hyper-connection residual
(n_embd_out = hc_mult * n_embd = 10240) captured before the final mixer
collapses it. h can only come from a real forward pass of the 125B trunk, so it
gets computed once here and cached on disk rather than recomputed every epoch.

Pipeline position:
    dataset_build.py   -> train/val rows (prompt ids + completion text)
    mtp_dataprep.py    -> token-exact sequences + supervised spans (tokens.bin)
    mtp_dataprep.py --plan -> execution order that maximises KV reuse
    THIS SCRIPT        -> h.bf16, one 10240-wide bf16 row per supervised token
    (next)             -> LoRA trainer over (h, next-token id) -> target token

Why the plan matters: extracting h naively costs a full prefill per sample,
213M tokens for dataset-v2. Samples are near-duplicates of each other (813
sessions, each turn's prompt containing the previous turn), so visiting them in
lexicographic token order lets llama.cpp keep the shared prefix in KV cache and
decode only the divergent tail: 42.8M tokens, a 5.0x saving. Correctness of
that trick is not assumed -- run --check-reuse to prove cached and cold h agree.

Requires libmtpextract.so (see scripts/mtp_extract_shim.cpp for why a shim
exists and how to build it) and exclusive use of the GPUs, so llama-server has
to be stopped first. Reads the model straight from the Docker volume, which is
root-owned, hence sudo.

Output layout:
    {split}.h.bf16        raw little-endian bf16, row-major, 10240 per row
    {split}.h.index.jsonl {offset, n_tokens, sup_start, sup_end, n_rows, h_row}
                          `offset` is the token-stream offset, the join key back
                          to {split}.index.jsonl; `h_row` is this sample's first
                          row in h.bf16.

The run is resumable: the index is flushed per sample and --resume truncates
h.bf16 back to an exact row boundary before continuing.
"""

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time

import numpy as np

DEFAULT_PREP = "/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-prep"
DEFAULT_LIB = "/mnt/comfyui-models/llama-nexus/build/llama.cpp/build/bin/libmtpextract.so"
DEFAULT_MODEL = ("/var/lib/docker/volumes/llama-nexus_gpt_oss_models/_data/UD-Q4_K_XL/"
                 "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")


class Shim:
    """ctypes binding for scripts/mtp_extract_shim.cpp."""

    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)
        c = self.lib
        c.mtpx_open.restype = ctypes.c_void_p
        c.mtpx_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_char_p]
        c.mtpx_n_embd_out.restype = ctypes.c_int
        c.mtpx_n_embd_out.argtypes = [ctypes.c_void_p]
        c.mtpx_n_ctx_train.restype = ctypes.c_int
        c.mtpx_n_ctx_train.argtypes = [ctypes.c_void_p]
        c.mtpx_decode.restype = ctypes.c_int
        c.mtpx_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                  ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        c.mtpx_read_h_bf16.restype = ctypes.c_int
        c.mtpx_read_h_bf16.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                       ctypes.c_int32, ctypes.POINTER(ctypes.c_uint16)]
        c.mtpx_trim.restype = ctypes.c_int
        c.mtpx_trim.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        c.mtpx_close.argtypes = [ctypes.c_void_p]
        c.mtpx_last_error.restype = ctypes.c_char_p
        self.h = None

    def err(self):
        return self.lib.mtpx_last_error().decode("utf-8", "replace")

    def open(self, args):
        self.h = self.lib.mtpx_open(
            args.model.encode(), args.n_gpu_layers, args.tensor_split.encode(),
            args.main_gpu, args.n_ctx, args.n_batch, args.n_ubatch,
            args.threads, args.kv_type.encode(), 1 if args.flash_attn else 0,
            args.cpu_tensors.encode())
        if not self.h:
            raise RuntimeError(f"mtpx_open failed: {self.err()}")
        self.n_embd_out = self.lib.mtpx_n_embd_out(self.h)
        self.n_ctx_train = self.lib.mtpx_n_ctx_train(self.h)
        return self

    def decode(self, chunk, pos0, last_logit):
        p = chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        rc = self.lib.mtpx_decode(self.h, p, len(chunk), pos0, 1 if last_logit else 0)
        if rc != 0:
            raise RuntimeError(f"decode failed at pos {pos0}: {self.err()}")

    def read_h(self, rows, out):
        rp = rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        op = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        if self.lib.mtpx_read_h_bf16(self.h, rp, len(rows), op) != 0:
            raise RuntimeError(f"read_h failed: {self.err()}")

    def trim(self, keep):
        """Drop KV from `keep` onward. Returns False if the arch refuses.

        Not an exception: on Qwen3.8-Flash-Next this is expected to fail for any
        keep > 0. The model is a hybrid whose Gated Delta Net layers hold a
        rolling recurrent state with no position to truncate to, so
        llama_memory_recurrent::seq_rm rejects partial removal outright (and
        llama_memory_hybrid::seq_rm leaves the cache untouched when it does).
        keep == 0 clears everything and always succeeds.
        """
        return self.lib.mtpx_trim(self.h, keep) == 0

    def close(self):
        if self.h:
            self.lib.mtpx_close(self.h)
            self.h = None


def load_plan(prep_dir, split):
    path = os.path.join(prep_dir, f"{split}.extract_plan.tsv")
    if not os.path.exists(path):
        raise SystemExit(f"no plan at {path} -- run: mtp_dataprep.py --plan")
    plan = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            offset, n_tokens, n_prompt, sup_start, sup_end, n_reuse = map(int, line.split())
            plan.append({"offset": offset, "n_tokens": n_tokens, "n_prompt": n_prompt,
                         "sup_start": sup_start, "sup_end": sup_end, "n_reuse": n_reuse})
    return plan


def select(plan, prep_dir, split, per_session, top):
    """Trim the plan to the samples that buy the most supervision per GPU-hour.

    Cost is the whole sequence (prefill) but yield is only the completion, so
    rows-per-token varies ~100x across samples: the median turn returns 4.7 rows
    per 1k tokens, the best 644. Left unordered, dataset-v2 costs 213M tokens of
    prefill for 1.25M rows; taking the best-yield turn from each of the 793
    sessions costs 16.1M for 220k rows -- 18% of the supervision for 8% of the
    compute, with every session still represented.

    --per-session keeps sessions balanced, which matters because these are
    Cursor conversations and context diversity is the point. --top ignores
    session balance and is purely greedy.
    """
    if not per_session and not top:
        return plan

    sess = {}
    with open(os.path.join(prep_dir, f"{split}.index.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                sess[e["offset"]] = e["prefix_key"]

    def rows_per_token(s):
        return (s["sup_end"] - s["sup_start"] + 1) / s["n_tokens"]

    if per_session:
        groups = {}
        for s in plan:
            groups.setdefault(sess.get(s["offset"], s["offset"]), []).append(s)
        keep = set()
        for g in groups.values():
            for s in sorted(g, key=rows_per_token, reverse=True)[:per_session]:
                keep.add(s["offset"])
    else:
        keep = {s["offset"] for s in sorted(plan, key=rows_per_token, reverse=True)[:top]}

    # keep the plan's own order so --resume stays meaningful
    return [s for s in plan if s["offset"] in keep]


def extract_one(shim, toks, s, keep, n_batch, W, scratch):
    """Decode one sample from position `keep` and return its h rows as bf16.

    Returns (rows, decoded_tokens). Rows come back in ascending position order.
    """
    base, n = s["offset"], s["n_tokens"]
    lo_sup, hi_sup = s["sup_start"], s["sup_end"] + 1
    want = hi_sup - lo_sup
    if want > scratch.shape[0]:
        raise RuntimeError(f"scratch holds {scratch.shape[0]} rows, sample needs {want}")
    out = scratch[:want]
    filled = 0
    decoded = 0

    p = keep
    while p < n:
        take = min(n_batch, n - p)
        chunk = np.ascontiguousarray(toks[base + p: base + p + take], dtype=np.int32)
        shim.decode(chunk, p, last_logit=(p + take >= n))
        decoded += take

        lo = max(lo_sup, p)
        hi = min(hi_sup, p + take)
        if hi > lo:
            rows = np.arange(lo - p, hi - p, dtype=np.int32)
            shim.read_h(rows, out[filled:filled + len(rows)])
            filled += len(rows)
        p += take

    if filled != want:
        raise RuntimeError(f"sample at {base}: got {filled} of {want} supervised rows")
    return out, decoded


def check_reuse(shim, toks, plan, n_batch, W, n_check):
    """Prove prefix reuse does not change h.

    The 5x saving rests entirely on cached-prefix h matching cold-prefill h. A
    quantised KV cache makes that a claim to test, not to assume: attention over
    reused q8_0 keys is not bit-identical to attention over freshly written
    ones. Compare a reused pass against a cold pass on the same sample.
    """
    widest = max(s["sup_end"] - s["sup_start"] + 1 for s in plan)
    scratch = np.zeros((widest, W), dtype=np.uint16)
    scratch2 = np.zeros((widest, W), dtype=np.uint16)
    picked = [(i, s) for i, s in enumerate(plan) if s["n_reuse"] > 0 and i > 0][:n_check]
    if not picked:
        return {"checked": 0, "note": "no sample in the plan reuses a prefix"}

    results = []
    for idx, s in picked:
        # warm path: replay the predecessor so the cache holds the shared prefix
        shim.trim(0)
        extract_one(shim, toks, plan[idx - 1], 0, n_batch, W, scratch)
        if not shim.trim(s["n_reuse"]):
            return {"checked": 0, "reuse_supported": False,
                    "note": "partial KV rollback refused by the arch (hybrid "
                            "recurrent layers) -- prefix reuse is unavailable, "
                            "every sample needs a full prefill"}
        warm, _ = extract_one(shim, toks, s, s["n_reuse"], n_batch, W, scratch)
        warm = warm.copy()

        # cold path: same sample, empty cache
        shim.trim(0)
        cold, _ = extract_one(shim, toks, s, 0, n_batch, W, scratch2)

        a = warm.view(np.uint16).astype(np.uint32) << 16
        b = cold.view(np.uint16).astype(np.uint32) << 16
        fa = a.view(np.float32).reshape(-1)
        fb = b.view(np.float32).reshape(-1)
        denom = np.linalg.norm(fa) * np.linalg.norm(fb)
        results.append({
            "offset": s["offset"],
            "n_reuse": s["n_reuse"],
            "rows": int(warm.shape[0]),
            "identical_rows_pct": round(float((warm == cold).all(axis=1).mean()) * 100, 2),
            "max_abs_diff": round(float(np.abs(fa - fb).max()), 6),
            "rel_l2": round(float(np.linalg.norm(fa - fb) / (np.linalg.norm(fb) + 1e-9)), 8),
            "cosine": round(float(fa @ fb / (denom + 1e-9)), 8),
        })
    return {"checked": len(results), "samples": results}


def emit_subset(args, prep_dir, split, out_dir):
    """Write a self-contained prep dir holding only the selected samples.

    dataset-v2's token stream is 853 MB because it carries all 9,825 turns, but
    a --per-session run touches a fraction of them. Shipping the whole blob to a
    rented pod wastes the slowest link in the pipeline, so the selection runs
    here and only the chosen samples travel, with offsets rewritten to the new
    packed layout. Selection uses the same select() the extractor would apply,
    so the pod reproduces this set exactly without needing the flags.
    """
    plan = select(load_plan(prep_dir, split), prep_dir, split, args.per_session, args.top)
    if args.limit:
        plan = plan[:args.limit]
    toks = np.memmap(os.path.join(prep_dir, f"{split}.tokens.bin"), dtype="<i4", mode="r")

    index = {}
    with open(os.path.join(prep_dir, f"{split}.index.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                index[e["offset"]] = e

    os.makedirs(out_dir, exist_ok=True)
    new_off = 0
    with open(os.path.join(out_dir, f"{split}.tokens.bin"), "wb") as bf, \
         open(os.path.join(out_dir, f"{split}.index.jsonl"), "w", encoding="utf-8") as xf, \
         open(os.path.join(out_dir, f"{split}.extract_plan.tsv"), "w", encoding="utf-8") as pf:
        pf.write("# offset\tn_tokens\tn_prompt\tsup_start\tsup_end\tn_reuse\n")
        for s in plan:
            n = s["n_tokens"]
            bf.write(np.asarray(toks[s["offset"]: s["offset"] + n]).tobytes())
            e = dict(index.get(s["offset"], {}))
            e["offset"] = new_off
            e["n_tokens"] = n
            xf.write(json.dumps(e) + "\n")
            # n_reuse is 0: the packed order carries no prefix-sharing claim, and
            # this arch refuses partial KV rollback anyway
            pf.write(f"{new_off}\t{n}\t{s['n_prompt']}\t{s['sup_start']}\t{s['sup_end']}\t0\n")
            new_off += n

    # Checksums travel with the payload. A prep dir that is the right size but
    # the wrong bytes trains a head on states paired to the wrong tokens, which
    # shows up only as a mediocre model many GPU-hours later -- so the receiver
    # is given the means to reject it up front.
    names = [f"{split}.tokens.bin", f"{split}.index.jsonl", f"{split}.extract_plan.tsv"]
    sums = {}
    for n in names:
        h = hashlib.md5()
        with open(os.path.join(out_dir, n), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sums[n] = h.hexdigest()

    mpath = os.path.join(out_dir, "manifest.md5")
    keep = {}
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d, n = line.split(None, 1)
                    keep[n.strip()] = d
    keep.update(sums)
    with open(mpath, "w", encoding="utf-8") as f:
        for n in sorted(keep):
            f.write(f"{keep[n]}  {n}\n")

    rows = sum(s["sup_end"] - s["sup_start"] + 1 for s in plan)
    return {"split": split, "samples": len(plan), "tokens": new_off,
            "supervised_rows": rows, "bytes": new_off * 4, "md5": sums}


def run_split(shim, args, prep_dir, split, out_dir):
    plan = load_plan(prep_dir, split)
    plan = select(plan, prep_dir, split, args.per_session, args.top)
    if args.limit:
        plan = plan[:args.limit]
    toks = np.memmap(os.path.join(prep_dir, f"{split}.tokens.bin"), dtype="<i4", mode="r")
    W = shim.n_embd_out

    longest = max(s["n_tokens"] for s in plan)
    if longest > args.n_ctx:
        raise SystemExit(f"--n-ctx {args.n_ctx} is below the longest sample ({longest})")

    h_path = os.path.join(out_dir, f"{split}.h.bf16")
    idx_path = os.path.join(out_dir, f"{split}.h.index.jsonl")

    if args.check_reuse:
        return check_reuse(shim, toks, plan, args.n_batch, W, args.check_reuse)

    # resume: trust the index, cut the blob back to a row boundary
    done = 0
    h_rows = 0
    if args.resume and os.path.exists(idx_path):
        # Reconcile rather than trust the index: keep only the samples whose
        # rows are actually backed by bytes in h.bf16, then cut both files to
        # that boundary. Truncating h to the index's expectation would
        # zero-extend it if the index had run ahead, quietly poisoning training
        # data with rows of zeros.
        have_rows = os.path.getsize(h_path) // (W * 2) if os.path.exists(h_path) else 0
        entries = []
        with open(idx_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn final line
                if h_rows + e["n_rows"] > have_rows:
                    break
                h_rows += e["n_rows"]
                entries.append(line if line.endswith("\n") else line + "\n")
        done = len(entries)
        with open(idx_path, "w", encoding="utf-8") as f:
            f.writelines(entries)
        with open(h_path, "r+b") as f:
            f.truncate(h_rows * W * 2)
        print(f"  resume: {done} samples, {h_rows} rows kept "
              f"({have_rows - h_rows} unindexed rows dropped)", file=sys.stderr)
    elif os.path.exists(h_path):
        raise SystemExit(f"{h_path} exists -- pass --resume, or move it aside")

    widest = max(s["sup_end"] - s["sup_start"] + 1 for s in plan)
    scratch = np.zeros((widest, W), dtype=np.uint16)

    t0 = time.time()
    tokens_done = 0
    partial_trim = args.reuse
    tokens_todo = sum(s["n_tokens"] - (s["n_reuse"] if args.reuse else 0)
                      for s in plan[done:])
    cached = 0  # tokens currently in the KV cache; 0 after a fresh open

    with open(h_path, "ab") as hf, open(idx_path, "a", encoding="utf-8") as xf:
        for i, s in enumerate(plan[done:], start=done):
            # the plan's n_reuse assumes the predecessor just ran; that only
            # holds if we are walking the plan in order with a warm cache
            warm = (i > done and args.reuse and partial_trim
                    and cached == plan[i - 1]["n_tokens"])
            keep = min(s["n_reuse"], cached, s["sup_start"]) if warm else 0
            if keep > 0 and not shim.trim(keep):
                # this arch cannot roll back a partial prefix; stop asking and
                # prefill every sample from scratch for the rest of the run
                partial_trim = False
                keep = 0
            if keep == 0:
                shim.trim(0)

            rows, decoded = extract_one(shim, toks, s, keep, args.n_batch, W, scratch)
            # Data before the pointer to it: the index is the recovery record, so
            # h must be on disk before a row claims to exist. The reverse order
            # lets a crash leave the index ahead of h.
            hf.write(rows.tobytes())
            hf.flush()
            xf.write(json.dumps({"offset": s["offset"], "n_tokens": s["n_tokens"],
                                 "sup_start": s["sup_start"], "sup_end": s["sup_end"],
                                 "n_rows": int(rows.shape[0]), "h_row": h_rows}) + "\n")
            xf.flush()
            h_rows += int(rows.shape[0])
            tokens_done += decoded
            cached = s["n_tokens"]

            if (i + 1) % args.report_every == 0 or i + 1 == len(plan):
                el = time.time() - t0
                rate = tokens_done / max(el, 1e-6)
                eta = (tokens_todo - tokens_done) / max(rate, 1e-6)
                hf.flush()
                print(f"  [{i + 1}/{len(plan)}] {h_rows} rows  "
                      f"{tokens_done / 1e6:.1f}M/{tokens_todo / 1e6:.1f}M tok  "
                      f"{rate:.0f} tok/s  eta {eta / 3600:.1f}h", file=sys.stderr, flush=True)

    return {"samples": len(plan), "h_rows": h_rows, "row_width": W,
            "bytes": h_rows * W * 2, "decoded_tokens": tokens_done,
            "elapsed_s": round(time.time() - t0, 1),
            "h": os.path.basename(h_path), "index": os.path.basename(idx_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-dir", default=DEFAULT_PREP)
    ap.add_argument("--out", default=None, help="default: <prep-dir>")
    ap.add_argument("--lib", default=DEFAULT_LIB)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--split", choices=["train", "val", "both"], default="val")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N samples (applied after selection)")
    ap.add_argument("--emit-subset", metavar="DIR",
                    help="write a prep dir containing only the selected samples "
                         "and exit; shrinks what has to be shipped to a remote pod")
    ap.add_argument("--per-session", type=int, default=0, metavar="K",
                    help="keep only the K best rows-per-token turns of each "
                         "session; covers all sessions at a fraction of the cost")
    ap.add_argument("--top", type=int, default=0, metavar="N",
                    help="keep the N best rows-per-token samples overall "
                         "(ignores session balance)")
    ap.add_argument("--resume", action="store_true")
    # Off by default: measured on this model, partial KV rollback is refused by
    # the Gated Delta Net layers, so the plan's n_reuse cannot be honoured. Kept
    # as a flag because it is correct (and a 5x saving) on a pure-attention arch.
    ap.add_argument("--reuse", action="store_true",
                    help="try the plan's KV prefix reuse (unsupported on hybrid "
                         "recurrent models; falls back automatically)")
    ap.add_argument("--check-reuse", type=int, default=0, metavar="N",
                    help="validate prefix reuse on N samples and exit")
    ap.add_argument("--report-every", type=int, default=25)
    # served config, mirrored so h matches the trunk the head will be paired with
    ap.add_argument("--n-ctx", type=int, default=88064)
    ap.add_argument("--n-batch", type=int, default=2048)
    # Measured on 4x3090Ti with this model: 569 tok/s at ubatch 256, 707 at 512,
    # 832 at 1024, 898 at 2048. Prefill is the whole cost here, so default high
    # rather than mirroring the server's 256 (which is tuned for latency).
    # Not 2048: that peaked a 2.3 GB compute buffer on a long sample and died of
    # cudaMalloc mid-run, with only ~2 GB spare per card after weights and KV.
    # 1024 halves the buffers and gives up 7% throughput for a run that finishes.
    ap.add_argument("--n-ubatch", type=int, default=1024)
    ap.add_argument("--n-gpu-layers", type=int, default=999)
    ap.add_argument("--tensor-split", default="0.92,1.08,1.08,0.92")
    ap.add_argument("--main-gpu", type=int, default=0)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--kv-type", default="q8_0")
    ap.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    ap.add_argument("--cpu-tensors", default="per_layer_token_embd",
                    help="tensor patterns pinned to host memory (served default)")
    args = ap.parse_args()

    # [TIME] | INFO | SCRIPTS.MTP_EXTRACT_H.MAIN | mandatory for qwen4exp, and
    # part of matching the served numerics. A quantized KV cache switches on
    # Hadamard key/value rotation, but the patch's build_attn overload asserts
    # self_k_rot == nullptr, so a q8_0 cache aborts the graph without this. The
    # serving container sets it too (backend/modules/managers/llamacpp_manager.py),
    # so h comes out of the same attention path the head will be paired with.
    os.environ.setdefault("LLAMA_ATTN_ROT_DISABLE", "1")

    out_dir = args.out or args.prep_dir
    os.makedirs(out_dir, exist_ok=True)

    # Subsetting is pure data movement: no shim, no model, no GPU, no sudo.
    if args.emit_subset:
        sel = ["train", "val"] if args.split == "both" else [args.split]
        rep = {s: emit_subset(args, args.prep_dir, s, args.emit_subset) for s in sel}
        print(json.dumps({"emit_subset": rep, "out": args.emit_subset}, indent=2))
        return 0

    if not os.path.exists(args.lib):
        raise SystemExit(f"no shim at {args.lib}; build it per mtp_extract_shim.cpp")
    if not os.access(args.model, os.R_OK):
        raise SystemExit(f"cannot read {args.model} (the model volume is root-owned; use sudo)")

    splits = ["train", "val"] if args.split == "both" else [args.split]

    # Size the context to the work actually queued. The served default of 88064
    # reserves KV for a prompt length no val sample comes near, and that VRAM is
    # exactly what the compute buffers ran out of. Padded to 1024 and never
    # raised above the request, so an explicit --n-ctx still wins. Mirrors
    # run_split's selection, or --per-session would still pay for the long
    # samples it is about to discard.
    def _queued(sp):
        p = select(load_plan(args.prep_dir, sp), args.prep_dir, sp,
                   args.per_session, args.top)
        return p[:args.limit] if args.limit else p

    longest = max(s["n_tokens"] for sp in splits for s in _queued(sp))
    fitted = min(args.n_ctx, (longest + 1023) // 1024 * 1024 + 1024)
    if fitted < args.n_ctx:
        print(f"n_ctx {args.n_ctx} -> {fitted} (longest sample {longest} tokens)",
              file=sys.stderr)
        args.n_ctx = fitted

    t0 = time.time()
    shim = Shim(args.lib).open(args)
    print(f"model loaded: n_embd_out={shim.n_embd_out} n_ctx_train={shim.n_ctx_train} "
          f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    if shim.n_embd_out != 10240:
        print(f"  warning: expected n_embd_out 10240 for qwen4exp, got {shim.n_embd_out}",
              file=sys.stderr)

    try:
        report = {s: run_split(shim, args, args.prep_dir, s, out_dir) for s in splits}
    finally:
        shim.close()

    print(json.dumps({"extract_h": report, "dtype": "bfloat16-le",
                      "kv_type": args.kv_type,
                      "reuse": args.reuse,
                      "elapsed_s": round(time.time() - t0, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
