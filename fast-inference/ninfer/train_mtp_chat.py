#!/usr/bin/env python3
# [2026-08-28 | INFO | train_mtp_chat | Fine-tunes the Qwen3-Next (qwen3_5) MTP
# draft head on a chat-domain corpus to lift MTP acceptance for
# chat/casual + chat/technical prompts. The main text backbone is loaded frozen
# in NF4 via bitsandbytes; only the ~425M-param MTP block + its fusion projection
# and norms are trainable in BF16. Output: a safetensors file containing the
# 15 mtp.* tensors ready to splice into the ninfer requant pipeline.
#
# Usage inside the ninfer-convert image:
#   python3 fast-inference/ninfer/train_mtp_chat.py \
#     --source /models/huihui-bf16 \
#     --out /models/mtp-chat/mtp.chat.safetensors \
#     --tokens 5_000_000 --seq-len 2048 --batch 2 --lr 1e-4
#
# Design notes:
# * The MTP block is a single full-attention Qwen3NextDecoderLayer (attn_output_gate
#   + 24 heads @ 256 head_dim + 4 KV heads) wrapped with an fc fusion projection
#   and two pre-norms. Tensor names match Huihui's mtp.* HF keys 1:1.
# * MTP forward: z_t = fc(cat(norm_h(h_t), norm_e(embed(x_{t+1}))))
#                h'_t = decoder_layer(z_t)         (with full causal attention)
#                logits_t = frozen_lm_head(mtp_norm(h'_t))
#                CE(logits_t, x_{t+2})            (teacher-forced)
# * Training positions use the standard "shift by 2" convention: we can only
#   compute loss on t in [0, L-3] since we need x_{t+2} as target.
# * Backbone hidden states are extracted from Qwen3NextModel (base .model), NOT
#   the causal-LM wrapper, so we can skip the LM-head matmul during the forward
#   over the backbone (still needed for MTP output projection though).

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

# --- Environment: reduce memory pressure ------------------------------------

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# --- MTP module -------------------------------------------------------------

class MTPBlock(nn.Module):
    """Trainable MTP draft block for Qwen3-Next.

    Wires:
      pre_fc_norm_hidden(h)  ─┐
                              cat → fc → decoder_layer → norm → (external lm_head)
      pre_fc_norm_embedding(e)┘
    """

    def __init__(self, backbone_config):
        super().__init__()
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextDecoderLayer,
            Qwen3NextRMSNorm,
        )

        # Build a shallow-copy config that pins the single MTP block to full attention
        # regardless of what index it lives at in the backbone stack.
        mtp_cfg = _clone_config(backbone_config)
        mtp_cfg.layer_types = ["full_attention"]
        mtp_cfg.num_hidden_layers = 1
        # DecoderLayer references some Qwen3-MoE fields that Qwen3_5TextConfig
        # doesn't declare by default; give them innocuous scalar defaults.
        for attr, default in [
            ("mlp_only_layers", []),
            ("decoder_sparse_step", 1),
            ("num_experts", 0),
        ]:
            if not hasattr(mtp_cfg, attr):
                setattr(mtp_cfg, attr, default)
        self._cfg = mtp_cfg

        hidden = mtp_cfg.hidden_size
        eps = getattr(mtp_cfg, "rms_norm_eps", 1e-6)

        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(hidden, eps=eps)
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(hidden, eps=eps)
        # fc: cat(h,e) [2*hidden] -> hidden
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.layers = nn.ModuleList([Qwen3NextDecoderLayer(mtp_cfg, layer_idx=0)])
        self.norm = Qwen3NextRMSNorm(hidden, eps=eps)

    def forward(
        self,
        h_prev: torch.Tensor,        # [B, S, H] hidden state at t
        e_next: torch.Tensor,        # [B, S, H] embedding of x_{t+1}
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # cos, sin
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
    ) -> torch.Tensor:
        # Order must match Qwen's fc.weight: [embedding, hidden] along last dim.
        # (Confirmed against vllm/model_executor/models/qwen3_next_mtp.py.)
        z = torch.cat([self.pre_fc_norm_embedding(e_next),
                       self.pre_fc_norm_hidden(h_prev)], dim=-1)
        z = self.fc(z)
        h = self.layers[0](
            hidden_states=z,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
        )
        # DecoderLayer returns a single tensor in transformers>=5
        return self.norm(h)


def _clone_config(cfg):
    # cfg here is the *text* config (Qwen3NextConfig); we build a fresh shallow copy.
    from copy import deepcopy
    return deepcopy(cfg)


# --- Weight loading ---------------------------------------------------------

def load_mtp_state_dict(hf_dir: Path) -> dict[str, torch.Tensor]:
    """Read mtp.* tensors from a Huihui HF checkpoint and remap to MTPBlock keys."""
    idx_path = hf_dir / "model.safetensors.index.json"
    with idx_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    mtp_keys = {k: shard for k, shard in weight_map.items() if k.startswith("mtp.")}
    tensors: dict[str, torch.Tensor] = {}
    shards: dict[str, list[str]] = {}
    for k, sh in mtp_keys.items():
        shards.setdefault(sh, []).append(k)
    for shard, keys in shards.items():
        with safe_open(str(hf_dir / shard), framework="pt") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)

    # Strip the "mtp." prefix; "layers.0.*" -> "layers.0.*" (matches nn.ModuleList[0])
    return {k[len("mtp."):]: v for k, v in tensors.items()}


# --- Data pipeline ----------------------------------------------------------

@dataclass
class PackedBatch:
    input_ids: torch.LongTensor   # [B, S]
    attention_mask: torch.LongTensor  # [B, S]


def build_selfgen_stream(
    tokenizer,
    seq_len: int,
    max_tokens: int,
    jsonl: Path,
    loop: bool = False,
) -> Iterator[list[int]]:
    """Pack Huihui self-generated {prompt, completion} JSONL into seq_len chunks.

    [2026-08-31 | INFO | train_mtp_chat.build_selfgen_stream | Default is one pass.
    Looping a 111k corpus to hit a 200k budget was the Phase-4b overfit.]
    """
    buffer: list[int] = []
    yielded = 0
    passes = 0
    while yielded < max_tokens:
        passes += 1
        with jsonl.open() as f:
            for line in f:
                if yielded >= max_tokens:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = rec.get("prompt") or ""
                completion = rec.get("completion") or ""
                if not completion:
                    continue
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                ids = tokenizer(text, add_special_tokens=False).input_ids
                buffer.extend(ids)
                buffer.append(tokenizer.eos_token_id)
                while len(buffer) >= seq_len:
                    chunk = buffer[:seq_len]
                    buffer = buffer[seq_len:]
                    yielded += seq_len
                    yield chunk
        if yielded == 0:
            raise RuntimeError(f"self-gen corpus {jsonl} produced no token chunks")
        if not loop:
            break
        if passes >= 64:
            break


def build_chat_stream(tokenizer, seq_len: int, max_tokens: int) -> Iterator[list[int]]:
    """Yield lists of exactly ``seq_len`` token ids drawn from chat corpora.

    Sources are streamed (never fully downloaded). We concatenate turns with the
    Qwen chat template so the MTP block sees realistic conversation structure.
    """
    from datasets import load_dataset

    def _turns_to_text(sample):
        # ultrachat_200k schema: {'messages': [{'role', 'content'}, ...]}
        # Fall back to raw 'text' if present.
        if "messages" in sample and sample["messages"]:
            return tokenizer.apply_chat_template(
                sample["messages"], tokenize=False, add_generation_prompt=False
            )
        if "text" in sample and sample["text"]:
            return sample["text"]
        return None

    # Two chat sources; iterate them in a simple round-robin.
    sources = [
        ("HuggingFaceH4/ultrachat_200k", "train_sft"),
        ("allenai/tulu-3-sft-mixture", "train"),
    ]
    streams = []
    for repo, split in sources:
        try:
            streams.append(iter(load_dataset(repo, split=split, streaming=True)))
        except Exception as e:  # noqa: BLE001
            print(f"[data] source {repo} unavailable: {e}", file=sys.stderr)

    if not streams:
        raise RuntimeError("No chat data sources available")

    buffer: list[int] = []
    yielded = 0
    idx = 0
    while yielded < max_tokens:
        stream = streams[idx % len(streams)]
        idx += 1
        try:
            sample = next(stream)
        except StopIteration:
            continue
        text = _turns_to_text(sample)
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        # Bookend with EOS to prevent conversations from bleeding together.
        buffer.extend(ids)
        buffer.append(tokenizer.eos_token_id)
        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            buffer = buffer[seq_len:]
            yielded += seq_len
            yield chunk


def batch_stream(stream: Iterator[list[int]], batch_size: int, device: str) -> Iterator[PackedBatch]:
    while True:
        rows = []
        for _ in range(batch_size):
            try:
                rows.append(next(stream))
            except StopIteration:
                if not rows:
                    return
                break
        if not rows:
            return
        ids = torch.tensor(rows, dtype=torch.long, device=device)
        yield PackedBatch(input_ids=ids, attention_mask=torch.ones_like(ids))


# --- Training ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="Huihui BF16 HF checkpoint directory")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output safetensors path for trained mtp.* tensors")
    ap.add_argument("--tokens", type=int, default=5_000_000,
                    help="Training token budget (approx)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--origin-mix", type=float, default=0.3,
                    help="Lerp trained MTP weights back toward the stock head at save (0=off).")
    ap.add_argument("--loop-corpus", action="store_true",
                    help="Replay the JSONL to fill --tokens. Off by default to avoid overfit.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 20 steps only, no data pull past first batches")
    ap.add_argument("--corpus-jsonl", type=Path, default=None,
                    help="Huihui self-generated corpus.jsonl from generate_huihui_corpus.py")
    args = ap.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "MTP training requires a CUDA GPU"

    print(f"[env] torch {torch.__version__} | device {torch.cuda.get_device_name(0)}", flush=True)

    from transformers import (
        AutoTokenizer,
        AutoConfig,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    print(f"[load] tokenizer + config from {args.source}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    cfg = AutoConfig.from_pretrained(args.source)
    text_cfg = cfg.text_config

    # NF4 quantization for the frozen backbone. bf16 compute + double-quant for
    # smaller VRAM footprint. MTP compute stays bf16.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("[load] backbone (NF4 frozen)... this takes ~1-2 min", flush=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.source,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Grab handles to inner submodules. The Qwen3-Next causal-LM wraps a base
    # model; we need embed_tokens (for e_next), the language backbone (for h_prev),
    # and lm_head (frozen projection to logits).
    base_model = backbone.model.language_model if hasattr(backbone.model, "language_model") else backbone.model
    embed_tokens = base_model.embed_tokens
    lm_head = backbone.lm_head
    rotary = base_model.rotary_emb

    print(f"[load] MTP block (BF16 trainable) ...", flush=True)
    mtp = MTPBlock(text_cfg).to(device=device, dtype=torch.bfloat16)
    missing, unexpected = mtp.load_state_dict(load_mtp_state_dict(args.source), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"MTP load mismatch: missing={missing} unexpected={unexpected}")
    mtp.train()

    n_train = sum(p.numel() for p in mtp.parameters())
    print(f"[model] MTP trainable params: {n_train/1e6:.1f}M", flush=True)

    # Optimizer: 8-bit AdamW to keep momentum states cheap.
    try:
        from bitsandbytes.optim import AdamW8bit
        opt = AdamW8bit(mtp.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        print("[opt] using bnb.AdamW8bit", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[opt] bnb AdamW8bit unavailable ({e}); using torch AdamW", flush=True)
        opt = torch.optim.AdamW(mtp.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # [2026-08-31 | INFO | train_mtp_chat.main | Keep a frozen copy of the stock head for --origin-mix
    origin_sd = {k: v.detach().clone() for k, v in mtp.state_dict().items()}

    # --- data ---
    if args.corpus_jsonl:
        print(f"[data] self-gen corpus {args.corpus_jsonl} loop={args.loop_corpus}", flush=True)
        stream = build_selfgen_stream(
            tokenizer, args.seq_len, args.tokens, args.corpus_jsonl, loop=args.loop_corpus,
        )
    else:
        stream = build_chat_stream(tokenizer, args.seq_len, args.tokens)
    batches = batch_stream(stream, args.batch, device)

    # --- schedule ---
    total_steps = max(1, args.tokens // (args.batch * args.seq_len))
    if args.smoke:
        total_steps = min(total_steps, 20)
    print(f"[train] total_steps={total_steps} batch={args.batch} seq={args.seq_len} "
          f"tokens/step={args.batch*args.seq_len}", flush=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        frac = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac)))

    # Per-run stats
    tstart = time.perf_counter()
    smoothed = None
    step = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for batch in batches:
        if step >= total_steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step)

        input_ids = batch.input_ids       # [B, S]
        B, S = input_ids.shape
        if S < 3:
            continue

        # --- 1. Frozen backbone forward, extract last hidden state ---
        with torch.no_grad():
            out = base_model(input_ids=input_ids, use_cache=False)
            h_last = out.last_hidden_state  # [B, S, H]

        # --- 2. Build MTP inputs for positions t in [0, S-3] ---
        # h_prev = h_last[:, :-2]     -> hidden at t                (predict x_{t+2})
        # e_next = embed(x_{t+1})     -> embedding at t+1
        # target = x_{t+2}
        h_prev = h_last[:, :-2, :]                         # [B, S-2, H]
        with torch.no_grad():
            e_next = embed_tokens(input_ids[:, 1:-1])      # [B, S-2, H]
        target = input_ids[:, 2:]                          # [B, S-2]

        # Position ids for the MTP layer: use the anchor positions (t+1) so that
        # rotary phase aligns with what a real streaming inference call would see
        # (h_prev came from the base model at t; the MTP layer conditions on e_{t+1}
        # so rotary at position t+1 is the semantically correct choice for it).
        seq_positions = torch.arange(1, S - 1, device=device, dtype=torch.long)
        position_ids = seq_positions.unsqueeze(0).expand(B, -1)   # [B, S-2]
        cos, sin = rotary(h_prev, position_ids)

        # Causal mask (bfloat16 additive form). transformers' decoder layer accepts
        # a 4D mask [B, 1, T, T] with additive -inf on masked positions.
        T = h_prev.shape[1]
        causal = torch.zeros(T, T, dtype=torch.bfloat16, device=device)
        causal.masked_fill_(torch.ones(T, T, dtype=torch.bool, device=device).triu_(1), float("-inf"))
        attn_mask = causal[None, None, :, :].expand(B, 1, T, T)

        # --- 3. MTP forward ---
        h_out = mtp(
            h_prev=h_prev,
            e_next=e_next,
            position_embeddings=(cos, sin),
            attention_mask=attn_mask,
            position_ids=position_ids,
        )                                                  # [B, S-2, H]

        # --- 4. Logits via frozen lm_head; loss ---
        logits = lm_head(h_out).float()                    # [B, S-2, V]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            reduction="mean",
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Grad clip helps because we init from a well-trained head; big spikes are noise.
        torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
        opt.step()

        smoothed = loss.item() if smoothed is None else 0.9 * smoothed + 0.1 * loss.item()
        step += 1
        if step % args.log_every == 0 or step <= 5:
            elapsed = time.perf_counter() - tstart
            tok_per_s = step * B * S / max(1e-6, elapsed)
            print(f"[step {step:5d}/{total_steps}] loss={loss.item():.4f} "
                  f"ema={smoothed:.4f} lr={lr_at(step):.2e} "
                  f"tok/s={tok_per_s:,.0f} elapsed={elapsed:.1f}s", flush=True)

        if step % args.save_every == 0 and step < total_steps:
            _save_mtp(
                mtp, args.out, step=step, origin=origin_sd, origin_mix=args.origin_mix,
                extra={"partial": True},
            )

    # Final save
    _save_mtp(
        mtp, args.out, step=step, origin=origin_sd, origin_mix=args.origin_mix,
        extra={"partial": False, "final_loss_ema": smoothed, "origin_mix": args.origin_mix},
    )
    print(f"[done] wrote {args.out}  final ema loss={smoothed} origin_mix={args.origin_mix}", flush=True)


def _save_mtp(
    mtp: MTPBlock,
    out_path: Path,
    *,
    step: int,
    extra: dict,
    origin: dict[str, torch.Tensor] | None = None,
    origin_mix: float = 0.0,
):
    """Save MTP weights with the exact HF ``mtp.*`` naming so downstream tools work."""
    sd = {}
    trained = mtp.state_dict()
    mix = max(0.0, min(1.0, float(origin_mix)))
    for k, v in trained.items():
        t = v.detach().to(torch.bfloat16)
        if origin is not None and mix > 0.0 and k in origin:
            t = ((1.0 - mix) * t + mix * origin[k].detach().to(dtype=t.dtype, device=t.device)).to(torch.bfloat16)
        sd["mtp." + k] = t.cpu().contiguous().clone()
    metadata = {"step": str(step), **{k: str(v) for k, v in extra.items()}}
    save_file(sd, str(out_path), metadata=metadata)


if __name__ == "__main__":
    main()
