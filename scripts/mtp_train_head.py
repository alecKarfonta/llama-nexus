#!/usr/bin/env python3
"""[TIME] | INFO | MTP_TRAIN_HEAD | train the qwen4exp NextN draft head on this
server's own trunk states.

Consumes the two halves of the prep pipeline:
  <split>.tokens.bin / .index.jsonl   mtp_dataprep.py -- the token stream
  <split>.h.bf16     / .h.index.jsonl mtp_extract_h.py -- trunk hidden states

The supervision convention is fixed by mtp_dataprep.py: at supervised position
i the head reads h_i together with the embedding of tokens[i+1] and predicts
tokens[i+2]. sup_start is n_prompt-2 so the first scored prediction is the
first generated token, and sup_end is n_total-3 so the last one stays in range.

--eval-only is the gate that has to pass before any training is worth doing. It
scores the *shipped* head, unmodified, against the extracted states. That one
number exercises four things that are otherwise only checkable in combination:
the hidden states really are the tensor the head consumes, the PyTorch block
matches the ggml graph, GGUF_MAP lines the weights up correctly, and the i/i+1/
i+2 indexing is not off by one. A plausible accuracy means all four hold; a
near-zero one means the training data would be garbage and no amount of
optimisation would reveal why.

One approximation is deliberate. graph_mtp gives the draft block a KV cache and
absolute positions, so in production it attends over the whole conversation,
but h was only extracted for the supervised span. Each span is therefore scored
as its own sequence at its true positions, with attention reaching back only to
the start of the span. --context-profile measures what that costs by reporting
accuracy against the number of in-span positions a row can attend to.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen4exp_nextn import (  # noqa: E402
    NextNConfig, Qwen4ExpNextNHead, load_gguf_weights, write_gguf_weights,
)


class SpanData:
    """Supervised spans, paired with the trunk states extracted for them.

    mtp_extract_h.py may cover only a subset of the prepared samples (it selects
    by yield), so the h index is the authority on what exists and the token
    stream is joined onto it by byte offset.
    """

    def __init__(self, prep_dir, split, n_embd_out=10240):
        self.tokens = np.memmap(os.path.join(prep_dir, f"{split}.tokens.bin"),
                                dtype="<i4", mode="r")
        h_path = os.path.join(prep_dir, f"{split}.h.bf16")
        self.h = np.memmap(h_path, dtype=np.uint16, mode="r").reshape(-1, n_embd_out)
        self.spans = []
        with open(os.path.join(prep_dir, f"{split}.h.index.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.spans.append(json.loads(line))
        # a torn tail from an interrupted run would index past the blob
        backed = self.h.shape[0]
        self.spans = [s for s in self.spans if s["h_row"] + s["n_rows"] <= backed]
        self.n_rows = sum(s["n_rows"] for s in self.spans)

    def __len__(self):
        return len(self.spans)

    def get(self, k, device):
        """One span as (token_in, target, h, positions), all batch-size 1."""
        s = self.spans[k]
        off, i0, n = s["offset"], s["sup_start"], s["n_rows"]
        pos = np.arange(i0, i0 + n, dtype=np.int64)
        tok_in = np.asarray(self.tokens[off + pos + 1], dtype=np.int64)
        target = np.asarray(self.tokens[off + pos + 2], dtype=np.int64)
        # copied off the read-only memmap so torch owns writable storage;
        # bf16 arrives as raw uint16, so reinterpret rather than convert
        raw = np.array(self.h[s["h_row"]: s["h_row"] + n])
        h = torch.from_numpy(raw).view(torch.bfloat16)
        return (torch.from_numpy(tok_in).unsqueeze(0).to(device),
                torch.from_numpy(target).unsqueeze(0).to(device),
                h.unsqueeze(0).to(device),
                torch.from_numpy(pos).unsqueeze(0).to(device))


@torch.no_grad()
def evaluate(model, data, device, max_samples=0, context_profile=False):
    """Top-1/top-5 accuracy and loss of the head over supervised rows."""
    n = len(data) if not max_samples else min(max_samples, len(data))
    tot = hit1 = hit5 = 0
    loss_sum = 0.0
    # accuracy against how many in-span positions precede a row, to expose how
    # much the missing pre-span context costs
    buckets = [(0, 0), (1, 4), (5, 16), (17, 64), (65, 1 << 30)]
    bucket_hit = [0] * len(buckets)
    bucket_tot = [0] * len(buckets)

    for k in range(n):
        tok_in, target, h, pos = data.get(k, device)
        logits = model(tok_in, h, positions=pos).float()
        loss_sum += torch.nn.functional.cross_entropy(
            logits[0], target[0], reduction="sum").item()
        top5 = logits[0].topk(5, dim=-1).indices
        ok1 = (top5[:, 0] == target[0])
        ok5 = (top5 == target[0].unsqueeze(-1)).any(-1)
        hit1 += ok1.sum().item()
        hit5 += ok5.sum().item()
        tot += target.numel()
        if context_profile:
            depth = torch.arange(target.shape[1], device=device)
            for b, (lo, hi) in enumerate(buckets):
                m = (depth >= lo) & (depth <= hi)
                if m.any():
                    bucket_hit[b] += ok1[m].sum().item()
                    bucket_tot[b] += int(m.sum().item())
        if (k + 1) % 25 == 0:
            print(f"    [{k + 1}/{n}] {tot} rows  top1={hit1 / max(tot, 1):.4f}",
                  file=sys.stderr, flush=True)

    out = {"samples": n, "rows": tot,
           "top1": hit1 / max(tot, 1), "top5": hit5 / max(tot, 1),
           "loss": loss_sum / max(tot, 1)}
    if context_profile:
        out["by_context_depth"] = {
            (f"{lo}-{hi}" if hi < (1 << 30) else f"{lo}+"):
                {"rows": bucket_tot[b], "top1": bucket_hit[b] / bucket_tot[b]}
            for b, (lo, hi) in enumerate(buckets) if bucket_tot[b]}
    return out


def plan_rollout_anchors(data, span_indices, k_max, window, per_span, rng,
                         min_depth=0):
    """Pick anchor positions for k-step rollout / rollout_loss.

    Each anchor needs `window` true h rows before it and k_max targets after.
    """
    anchors = []
    for si in span_indices:
        s = data.spans[si]
        i0, rows = s["sup_start"], s["n_rows"]
        lo = i0 + max(min_depth, window - 1)
        hi = i0 + rows - k_max - 1
        if hi < lo:
            continue
        anchors += [(si, p) for p in
                    sorted(rng.sample(range(lo, hi + 1),
                                      min(per_span, hi - lo + 1)))]
    return anchors


def rollout_batch_tensors(data, grp, window, device, dtype):
    """Materialise one rectangular batch of rollout anchors."""
    hs, ts, ps, offs, pj = [], [], [], [], []
    for si, p in grp:
        s = data.spans[si]
        idx = np.arange(p - window + 1, p + 1)
        r0 = s["h_row"] + (p - window + 1 - s["sup_start"])
        hs.append(np.array(data.h[r0: r0 + window]))
        ts.append(np.asarray(data.tokens[s["offset"] + idx + 1], dtype=np.int64))
        ps.append(idx)
        offs.append(s["offset"])
        pj.append(p)
    h_seq = torch.from_numpy(np.stack(hs)).view(torch.bfloat16)
    h_seq = h_seq.to(device=device, dtype=dtype)
    tok_seq = torch.from_numpy(np.stack(ts)).to(device)
    pos_seq = torch.from_numpy(np.stack(ps)).to(device)
    return h_seq, tok_seq, pos_seq, np.array(offs), np.array(pj)


def rollout_loss(model, data, anchors, device, k_max=8, window=64, batch=8,
                 teacher_p=1.0, p_min=0.0, conf_margin=0.0):
    """Mean CE over k unrolled steps while feeding back the head's own h.

    Mirrors rollout() but keeps the graph: state[:, -1:] is concatenated
    without detach so gradients teach the head to emit usable next-h.

    teacher_p is the probability of feeding the gold previous token at steps
    k>=2. Step 1 always uses gold (serving conditions on the last accepted
    trunk token). At teacher_p=1.0 this is pure teacher-forced k-step; at
    0.0 it matches serving's own-token feedback. Scheduled sampling mixes
    per-row with a Bernoulli coin flip so batches stay rectangular.

    When p_min > 0, also returns a confidence-margin hinge on p(gold):
      mean(relu(p_min + conf_margin - p_gold))
    so later draft steps stay above the serving cutoff.
    """
    if not anchors:
        return None, None
    dtype = next(model.parameters()).dtype
    ce_total, conf_total, n_steps = 0.0, 0.0, 0

    for b0 in range(0, len(anchors), batch):
        grp = anchors[b0: b0 + batch]
        h_seq, tok_seq, pos_seq, oj, pj = rollout_batch_tensors(
            data, grp, window, device, dtype)
        B = len(grp)

        for k in range(1, k_max + 1):
            logits, state = model(tok_seq, h_seq, positions=pos_seq,
                                  return_state=True, last_only=True)
            tgt = torch.from_numpy(
                np.asarray(data.tokens[oj + pj + k + 1], dtype=np.int64)
            ).to(device)
            ce_total += F.cross_entropy(logits[:, -1], tgt, reduction="sum")
            if p_min > 0.0:
                probs = F.softmax(logits[:, -1].float(), dim=-1)
                p_gold = probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
                conf_total += F.relu(p_min + conf_margin - p_gold).sum()
            n_steps += B

            if k == k_max:
                break
            h_seq = torch.cat([h_seq, state[:, -1:]], dim=1)
            if k == 1 or teacher_p >= 1.0:
                nxt = tgt
            elif teacher_p <= 0.0:
                nxt = logits[:, -1].argmax(-1)
            else:
                pred = logits[:, -1].argmax(-1)
                coin = torch.rand(B, device=device) < teacher_p
                nxt = torch.where(coin, tgt, pred)
            tok_seq = torch.cat([tok_seq, nxt.unsqueeze(1)], dim=1)
            pos_seq = torch.cat(
                [pos_seq,
                 torch.from_numpy(pj + k).unsqueeze(1).to(device)], dim=1)

    ce = ce_total / max(n_steps, 1)
    conf = (conf_total / max(n_steps, 1)) if p_min > 0.0 else None
    return ce, conf


@torch.no_grad()
def rollout(model, data, device, k_max=16, window=32, per_span=2, max_samples=0,
            min_depth=0, teacher_token=False, seed=0, batch=8, p_min=0.0):
    """Top-1 at draft step 1..k_max while the head consumes its own state.

    Mirrors common_speculative_impl_draft_mtp::draft. Step 1 reads the trunk's
    h and the last accepted token; step k reads back the block's own t_h_nextn
    and its own step-(k-1) token, at position p+k-1. graph_mtp publishes exactly
    the post-FFN hc_combine stack that Qwen4ExpNextNBlock.forward returns, so
    the feedback tensor is the same one serving uses.

    Step 1 is precisely what --eval-only scores. Everything past it is the one
    regime training never exercised: the head was fit single-step and
    teacher-forced from *true* trunk states, but serving makes it walk up to 16
    steps on its own output. A decay steeper than the stock head's is that
    exposure bias, measured.

    Per step:
      acc   top-1 over every rollout, including ones already derailed
      cond  top-1 among rollouts still correct at every earlier step
      surv  fraction of all rollouts correct at every step through k
    Verification is sequential, so `cond` and `surv` are the serving-relevant
    curves; sum(surv) is the expected accepted draft length, the offline
    analogue of the accepted-tokens-per-round the live p_min sweep measured.

    teacher_token feeds the true previous token while still feeding the head's
    own h, separating hidden-state drift from token drift.

    Every anchor takes a full `window` of true context, so all rollouts in a
    batch share a length and can run as one forward. That matters more than it
    looks: SparseMoE walks the routed experts in a Python loop, so its cost is
    per-batch rather than per-row and batching amortises it away.

    Attention reaches back exactly `window` true rows: the same span-only
    approximation --eval-only makes, since h was extracted per span. Production
    has the whole conversation in the draft's KV.
    """
    rng = random.Random(seed)
    n = len(data) if not max_samples else min(max_samples, len(data))
    dtype = next(model.parameters()).dtype
    anchors = plan_rollout_anchors(
        data, range(n), k_max, window, per_span, rng, min_depth)
    if not anchors:
        raise SystemExit(f"no span has {window + k_max} rows; lower --window")

    hit = [0] * (k_max + 1)
    tot = [0] * (k_max + 1)
    cond_hit = [0] * (k_max + 1)
    cond_tot = [0] * (k_max + 1)
    pmin_len_sum = 0.0
    t0 = time.time()

    for b0 in range(0, len(anchors), batch):
        grp = anchors[b0: b0 + batch]
        h_seq, tok_seq, pos_seq, oj, pj = rollout_batch_tensors(
            data, grp, window, device, dtype)
        alive = np.ones(len(grp), dtype=bool)
        row_len = np.zeros(len(grp), dtype=np.int32)

        for k in range(1, k_max + 1):
            logits, state = model(tok_seq, h_seq, positions=pos_seq,
                                  return_state=True, last_only=True)
            probs = F.softmax(logits[:, -1].float(), dim=-1)
            p_top = probs.max(dim=-1).values.cpu().numpy()
            pred = logits[:, -1].argmax(-1).cpu().numpy()
            tgt = np.asarray(data.tokens[oj + pj + k + 1], dtype=np.int64)
            ok = pred == tgt

            tot[k] += len(grp)
            hit[k] += int(ok.sum())
            cond_tot[k] += int(alive.sum())
            cond_hit[k] += int((alive & ok).sum())

            if p_min > 0.0:
                for i in range(len(grp)):
                    if not alive[i]:
                        continue
                    if p_top[i] < p_min or not ok[i]:
                        alive[i] = False
                    else:
                        row_len[i] += 1
            else:
                alive &= ok

            if k == k_max:
                break
            if p_min > 0.0 and not alive.any():
                break
            # the head's own hyper-connection state becomes the next h
            h_seq = torch.cat([h_seq, state[:, -1:]], dim=1)
            nxt = tgt if teacher_token else pred
            tok_seq = torch.cat(
                [tok_seq, torch.from_numpy(nxt).unsqueeze(1).to(device)], dim=1)
            pos_seq = torch.cat(
                [pos_seq, torch.from_numpy(pj + k).unsqueeze(1).to(device)], dim=1)

        pmin_len_sum += row_len.sum() if p_min > 0.0 else 0.0
        done = b0 + len(grp)
        elen_raw = sum(cond_hit[1:]) / done if done else 0.0
        elen_msg = f"E[len]={elen_raw:.2f}"
        if p_min > 0.0:
            elen_msg += f" E[len_pmin]={pmin_len_sum / done:.2f}"
        step1_acc = hit[1] / tot[1] if tot[1] else 0.0
        stepk_acc = hit[k_max] / tot[k_max] if tot[k_max] else 0.0
        print(f"    [{done}/{len(anchors)}] step1={step1_acc:.4f} "
              f"step{k_max}={stepk_acc:.4f} "
              f"{elen_msg} {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)

    n_roll = len(anchors)
    steps = [{"k": k,
              "acc": hit[k] / tot[k],
              "cond": cond_hit[k] / cond_tot[k] if cond_tot[k] else None,
              "surv": cond_hit[k] / n_roll,
              "n": tot[k], "n_cond": cond_tot[k]}
             for k in range(1, k_max + 1) if tot[k]]
    elen_raw = sum(s["surv"] for s in steps)
    elen_pmin = (pmin_len_sum / n_roll) if p_min > 0.0 else elen_raw
    gate_elen = elen_pmin if p_min > 0.0 else elen_raw
    return {"rollouts": n_roll, "spans": n, "k_max": k_max, "window": window,
            "per_span": per_span, "min_depth": max(min_depth, window - 1),
            "teacher_token": teacher_token, "batch": batch, "p_min": p_min,
            "expected_accepted_len": elen_raw,
            "expected_accepted_len_pmin": elen_pmin,
            "gate_elen": gate_elen,
            "steps": steps}


class LoRALinear(nn.Module):
    """Frozen base projection plus a trainable low-rank update.

    Adapters are kept in fp32 while the base stays bf16: they are ~1.6M
    parameters against 3.88B, so the memory is free, and bf16's 8-bit mantissa
    loses small updates outright once the optimiser starts accumulating.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # B stays zero: no-op at step 0
        self.scale = alpha / r

    def forward(self, x):
        out = self.base(x)
        delta = (x.float() @ self.A.t() @ self.B.t()) * self.scale
        return out + delta.to(out.dtype)


# Attention, the embedding/hidden join, the router and the shared expert. The
# 512 routed experts are deliberately excluded: at r=32 they would add ~150M
# trainable parameters and a grouped-weight LoRA implementation, for capacity
# this task may not need. Revisit only if the cheap scope underperforms.
LORA_TARGETS = (
    "block.attn.q_proj", "block.attn.k_proj", "block.attn.v_proj", "block.attn.o_proj",
    "block.eh_proj",
    "block.mlp.gate",
    "block.mlp.shared_gate_proj", "block.mlp.shared_up_proj", "block.mlp.shared_down_proj",
)


def inject_lora(model, r, alpha, targets=LORA_TARGETS):
    """Wrap each target Linear in place and freeze everything else."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = {}
    for name in targets:
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path)
        base = getattr(parent, leaf)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is {type(base).__name__}, not nn.Linear")
        setattr(parent, leaf, LoRALinear(base, r, alpha))
        wrapped[name] = (base.in_features, base.out_features)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return wrapped, trainable


def adapter_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if p.requires_grad and (n.endswith(".A") or n.endswith(".B"))}


def load_adapter(model, path):
    """Restore saved A/B tensors after inject_lora."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    adapter = ckpt["adapter"]
    missing = []
    for name, p in model.named_parameters():
        if not (name.endswith(".A") or name.endswith(".B")):
            continue
        if name not in adapter:
            missing.append(name)
            continue
        p.data.copy_(adapter[name].to(device=p.device, dtype=p.dtype))
    if missing:
        raise KeyError(f"adapter missing keys for {len(missing)} parameters: "
                       f"{missing[:3]}...")
    return ckpt


def merge_lora(model, ckpt, targets=LORA_TARGETS):
    """Fold W <- W + (B @ A) * (alpha/r) into each target Linear.

    The serving path has no LoRA runtime, so the update has to live in the
    base weights before they are written back to GGUF. Matches LoRALinear:
    delta = (x @ A.T @ B.T) * scale, i.e. dW = (B @ A) * (alpha/r).
    """
    if isinstance(ckpt, str):
        ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
    adapter = ckpt["adapter"]
    scale = ckpt["alpha"] / ckpt["r"]
    ckpt_targets = tuple(ckpt.get("targets") or targets)
    merged = []
    with torch.no_grad():
        for name in ckpt_targets:
            linear = model.get_submodule(name)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"{name} is {type(linear).__name__}, not nn.Linear")
            A = adapter[f"{name}.A"].to(torch.float32)
            B = adapter[f"{name}.B"].to(torch.float32)
            delta = (B @ A) * scale
            if tuple(delta.shape) != tuple(linear.weight.shape):
                raise ValueError(f"{name}: delta {tuple(delta.shape)} != "
                                 f"weight {tuple(linear.weight.shape)}")
            linear.weight.add_(delta.to(linear.weight.dtype))
            merged.append(name)
    return {"merged": merged, "r": ckpt["r"], "alpha": ckpt["alpha"],
            "step": ckpt.get("step"), "eval": ckpt.get("eval")}


def train(model, train_data, val_data, args, device):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    order = list(range(len(train_data)))
    rng = random.Random(args.seed)
    rng.shuffle(order)

    total = args.max_steps or (len(order) * args.epochs // args.accum)
    if args.lr_schedule == "constant":
        # A continue epoch must not anneal to zero: the last run's OneCycle
        # hit 1.6e-7 at the checkpoint that was still climbing.
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=max(total, 1), pct_start=0.05)

    best = {"gate_elen": getattr(args, "resume_elen", -1.0)}
    step = 0
    seen = 0
    t0 = time.time()
    loss_run = {"tf": 0.0, "roll": 0.0, "conf": 0.0}
    use_roll = args.rollout_lambda > 0
    use_conf = args.rollout_conf_lambda > 0 and args.rollout_p_min > 0
    abort_checks = []
    model.train()

    def _teacher_p(cur_step):
        """Linear schedule for gold-token probability in rollout_loss."""
        p0 = args.rollout_teacher_p
        p1 = args.rollout_teacher_p_end
        if total <= 1:
            return p1
        t = (cur_step - 1) / (total - 1)
        return p0 + (p1 - p0) * t

    for epoch in range(args.epochs):
        for k in order:
            tok_in, target, h, pos = train_data.get(k, device)
            loss_tf = model.loss(tok_in, h, target, positions=pos)
            loss = loss_tf
            loss_roll = None
            loss_conf = None
            if use_roll:
                anchors = plan_rollout_anchors(
                    train_data, [k], args.rollout_k, args.rollout_window,
                    args.rollout_per_span, rng)
                tp = _teacher_p(max(step + 1, 1))
                loss_roll, loss_conf = rollout_loss(
                    model, train_data, anchors, device,
                    k_max=args.rollout_k, window=args.rollout_window,
                    batch=args.rollout_batch, teacher_p=tp,
                    p_min=args.rollout_p_min if use_conf else 0.0,
                    conf_margin=args.rollout_conf_margin)
                if loss_roll is not None:
                    loss = loss + args.rollout_lambda * loss_roll
                if loss_conf is not None and use_conf:
                    loss = loss + args.rollout_conf_lambda * loss_conf

            (loss / args.accum).backward()
            loss_run["tf"] += loss_tf.item()
            if loss_roll is not None:
                loss_run["roll"] += loss_roll.item()
            if loss_conf is not None:
                loss_run["conf"] += loss_conf.item()
            seen += 1
            if seen % args.accum:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                rate = step / max(time.time() - t0, 1e-6)
                nlog = args.log_every * args.accum
                msg = (f"  step {step}/{total} "
                       f"loss_tf={loss_run['tf'] / nlog:.4f}")
                if use_roll:
                    msg += f" loss_roll={loss_run['roll'] / nlog:.4f}"
                    if use_conf:
                        msg += f" loss_conf={loss_run['conf'] / nlog:.4f}"
                    msg += f" p_teacher={_teacher_p(step):.2f}"
                msg += (f" lr={sched.get_last_lr()[0]:.2e} {rate:.2f} step/s")
                print(msg, file=sys.stderr, flush=True)
                loss_run = {"tf": 0.0, "roll": 0.0, "conf": 0.0}

            if step % args.eval_every == 0 or step == total:
                model.eval()
                rep_tf = evaluate(model, val_data, device, args.eval_samples)
                rep = rollout(
                    model, val_data, device,
                    k_max=args.eval_rollout_k,
                    window=args.rollout_window,
                    per_span=args.rollout_per_span,
                    max_samples=args.eval_samples,
                    seed=args.seed + step,
                    batch=args.rollout_batch,
                    p_min=args.rollout_p_min)
                model.train()
                step1 = rep["steps"][0]["acc"] if rep["steps"] else 0.0
                elen = rep["gate_elen"]
                elen_raw = rep["expected_accepted_len"]
                elen_pmin = rep.get("expected_accepted_len_pmin", elen_raw)
                print(f"  eval @ {step}: top1={rep_tf['top1']:.4f} "
                      f"rollout_step1={step1:.4f} E[len]={elen_raw:.4f} "
                      f"E[len_pmin]={elen_pmin:.4f}",
                      file=sys.stderr, flush=True)
                if args.abort_elen > 0 and step in args.abort_at:
                    abort_checks.append(elen < args.abort_elen)
                    if len(abort_checks) >= len(args.abort_at) and all(abort_checks):
                        print(f"  ABORT: gate E[len] < {args.abort_elen} at "
                              f"steps {args.abort_at}", file=sys.stderr, flush=True)
                        return best
                if elen > best["gate_elen"]:
                    best = {**rep_tf, **rep, "step": step, "gate_elen": elen}
                    torch.save({"adapter": adapter_state(model), "r": args.rank,
                                "alpha": args.alpha, "targets": list(LORA_TARGETS),
                                "eval": rep_tf, "rollout": rep, "step": step},
                               args.out)
                    print(f"    saved {args.out} (gate E[len]={elen:.4f})",
                          file=sys.stderr, flush=True)
            if step >= total:
                return best
    return best


def build_model(args, device):
    cfg = NextNConfig()
    torch.set_default_dtype(torch.bfloat16)  # 3.88B params: 7.8 GB, not 15.5
    model = Qwen4ExpNextNHead(cfg)
    load_gguf_weights(model, args.head_gguf, args.block_index)
    torch.set_default_dtype(torch.float32)
    if getattr(args, "fp32", False):
        # Widening bf16 to fp32 is exact, so the weights are unchanged. This
        # box is AVX2 only -- no AVX-512 or AMX -- so torch has no hardware
        # bf16 matmul and emulates it about 25x slower than fp32 oneDNN.
        model = model.float()
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-dir",
                    default="/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-prep")
    ap.add_argument("--head-gguf",
                    default="/mnt/comfyui-models/llama-nexus/build/mtp-bf16.gguf")
    ap.add_argument("--block-index", type=int, default=48)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-only", action="store_true",
                    help="score the shipped head and stop: the gate")
    ap.add_argument("--merge", metavar="ADAPTER.pt",
                    help="fold a saved LoRA into the GGUF weights and exit "
                         "(pair with --export-gguf)")
    ap.add_argument("--export-gguf", metavar="OUT.gguf",
                    help="write a BF16 GGUF with merged weights; copies the "
                         "source header so tokenizer and unused tensors survive")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--context-profile", action="store_true",
                    help="break accuracy down by in-span attention depth")
    ap.add_argument("--rollout", action="store_true",
                    help="score top-1 at draft steps 1..k_max while the head "
                         "consumes its own hidden state, as serving does")
    ap.add_argument("--k-max", type=int, default=16,
                    help="draft steps to roll out; matches --spec-draft-n-max")
    ap.add_argument("--window", type=int, default=32,
                    help="true rows of in-span attention context per rollout; "
                         "also the shallowest usable anchor, and a span needs "
                         "window+k_max rows to contribute at all")
    ap.add_argument("--per-span", type=int, default=2,
                    help="rollout anchors sampled per span")
    ap.add_argument("--min-depth", type=int, default=0,
                    help="extra anchor depth beyond --window")
    ap.add_argument("--batch", type=int, default=8,
                    help="rollouts per forward; amortises the MoE expert loop")
    ap.add_argument("--teacher-token", action="store_true",
                    help="feed the true previous token but still the head's own "
                         "h, to separate state drift from token drift")
    ap.add_argument("--p-min", type=float, default=0.0,
                    help="rollout: stop a row when top-1 p < p_min; report "
                         "E[len_pmin] alongside raw E[len]")
    ap.add_argument("--fp32", action="store_true",
                    help="run fp32 instead of bf16; exact widening, and ~25x "
                         "faster on an AVX2-only CPU")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--out", default="mtp_lora.pt")
    ap.add_argument("--resume", metavar="ADAPTER.pt",
                    help="continue from a saved LoRA; auto-lowers lr to 3e-5 "
                         "unless --lr is explicitly set")
    ap.add_argument("--adapter", metavar="ADAPTER.pt",
                    help="load a LoRA for --rollout or --eval-only without merging")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None,
                    help="AdamW peak lr; default 1e-4 fresh, 3e-5 with --resume")
    ap.add_argument("--lr-schedule", choices=("onecycle", "constant"),
                    default="onecycle",
                    help="onecycle anneals to ~0; constant holds --lr "
                         "(use for a continue epoch)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8, help="spans per optimiser step")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-samples", type=int, default=60,
                    help="val spans per mid-training eval; 0 for all")
    ap.add_argument("--eval-rollout-k", type=int, default=8,
                    help="k for mid-training rollout gate")
    ap.add_argument("--rollout-lambda", type=float, default=1.0,
                    help="weight on k-step rollout loss; 0 = teacher-forced only")
    ap.add_argument("--rollout-k", type=int, default=8,
                    help="unroll depth for rollout_loss during training")
    ap.add_argument("--rollout-window", type=int, default=64,
                    help="true context rows per rollout anchor in training")
    ap.add_argument("--rollout-per-span", type=int, default=2,
                    help="rollout anchors sampled per span during training")
    ap.add_argument("--rollout-batch", type=int, default=8,
                    help="anchors per rollout forward during training")
    ap.add_argument("--rollout-teacher-p", type=float, default=1.0,
                    help="rollout_loss: prob of gold prev token at k>=2 (start)")
    ap.add_argument("--rollout-teacher-p-end", type=float, default=1.0,
                    help="rollout_loss: prob of gold prev token at k>=2 (end)")
    ap.add_argument("--rollout-p-min", type=float, default=0.0,
                    help="confidence hinge target and eval rollout p_min cutoff")
    ap.add_argument("--rollout-conf-margin", type=float, default=0.05,
                    help="margin above p_min for p(gold) hinge loss")
    ap.add_argument("--rollout-conf-lambda", type=float, default=0.0,
                    help="weight on confidence-margin loss in rollout_loss")
    ap.add_argument("--abort-elen", type=float, default=0.0,
                    help="if >0 and gate E[len] stays below this at --abort-at "
                         "steps, stop early")
    ap.add_argument("--abort-at", type=int, nargs="*", default=[],
                    help="optimizer steps to check abort_elen (e.g. 100 200)")
    ap.add_argument("--resume-elen", type=float, default=None,
                    help="override incumbent gate E[len] for save bar (p_min metric)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 3e-5 if args.resume else 1e-4

    if args.merge:
        if not args.export_gguf:
            raise SystemExit("--merge needs --export-gguf (the serving path has no LoRA runtime)")
        # Merge is a weight copy, not a forward: CPU avoids fighting the live
        # server for the 8 GB the BF16 head occupies on a card.
        device = torch.device("cpu")
        model = build_model(args, device)
        info = merge_lora(model, args.merge)
        # Linear targets are stored as "<module>.weight" in GGUF_MAP. Only those
        # nine tensors are rewritten; F32 norms stay as the copied source bytes
        # so they are not rounded through the model's bf16 default dtype.
        only = [f"{n}.weight" for n in info["merged"]]
        write_gguf_weights(model, args.head_gguf, args.export_gguf, args.block_index,
                           only=only)
        print(json.dumps({"merge": info, "export": args.export_gguf}, indent=2,
                         default=str))
        return 0

    device = torch.device(args.device)
    data = SpanData(args.prep_dir, args.split)
    print(f"{args.split}: {len(data)} spans, {data.n_rows} supervised rows",
          file=sys.stderr)
    if not len(data):
        raise SystemExit("no spans -- has mtp_extract_h.py written anything yet?")

    model = build_model(args, device)

    if args.adapter:
        inject_lora(model, args.rank, args.alpha)
        ckpt = load_adapter(model, args.adapter)
        print(f"loaded adapter {args.adapter} @ step {ckpt.get('step')}",
              file=sys.stderr)

    if args.rollout:
        rep = rollout(model, data, device, args.k_max, args.window,
                      args.per_span, args.max_samples, args.min_depth,
                      args.teacher_token, args.seed, args.batch, args.p_min)
        print(json.dumps({"rollout": rep, "head": os.path.basename(args.head_gguf),
                          "split": args.split}, indent=2))
        return 0

    if args.eval_only:
        rep = evaluate(model, data, device, args.max_samples, args.context_profile)
        print(json.dumps({"gate": rep, "head": os.path.basename(args.head_gguf),
                          "split": args.split}, indent=2))
        return 0

    val_data = SpanData(args.prep_dir, args.val_split)
    print(f"{args.val_split}: {len(val_data)} spans, {val_data.n_rows} rows",
          file=sys.stderr)
    wrapped, trainable = inject_lora(model, args.rank, args.alpha)
    args.resume_elen = -1.0
    if args.resume:
        resume_ckpt = load_adapter(model, args.resume)
        rollout_rep = resume_ckpt.get("rollout") or {}
        if args.rollout_p_min > 0.0:
            args.resume_elen = float(
                rollout_rep.get("expected_accepted_len_pmin")
                or rollout_rep.get("gate_elen")
                or -1.0)
            if args.resume_elen < 0:
                # Incumbent lacks p_min metric; use diagnostic e1 bar if present.
                diag = "/tmp/pmin_diagnostic/e1_p0.85.json"
                if os.path.isfile(diag):
                    t = open(diag).read()
                    if "{" in t:
                        dr = json.loads(t[t.index("{"):])["rollout"]
                        args.resume_elen = float(dr["expected_accepted_len_pmin"])
        else:
            args.resume_elen = float(
                rollout_rep.get("gate_elen")
                or rollout_rep.get("expected_accepted_len")
                or -1.0)
        print(f"resumed {args.resume} @ step {resume_ckpt.get('step')} "
              f"(prior gate E[len]={args.resume_elen})",
              file=sys.stderr)
        if args.resume_elen is not None:
            args.resume_elen = float(args.resume_elen)
            print(f"  resume gate bar overridden to {args.resume_elen}",
                  file=sys.stderr)
        # Keep the previous checkpoint as the incumbent so a worse epoch
        # does not overwrite it. train() only writes on a new high.
        if not os.path.exists(args.out):
            import shutil
            shutil.copy2(args.resume, args.out)
    model.to(device)
    print(f"lora r={args.rank} alpha={args.alpha} on {len(wrapped)} projections, "
          f"{trainable / 1e6:.2f}M trainable", file=sys.stderr)

    base = evaluate(model, val_data, device, args.eval_samples)
    print(f"baseline (adapters are no-ops at init): top1={base['top1']:.4f}",
          file=sys.stderr)

    best = train(model, data, val_data, args, device)
    print(json.dumps({"baseline": base, "best": best, "out": args.out,
                      "trainable": trainable}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
