#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.QWEN4EXP_NEXTN | PyTorch implementation of the
qwen4exp NextN (MTP) draft block, for training the Cursor-specific draft head.

Transcribed from the reference graph, not guessed:
  - llama.cpp `src/models/qwen4exp.cpp` @ unslothai/llama.cpp bea3b12
    (build_hc_mix / build_hc_combine / build_layer_attn / build_layer_ffn)
  - `patches/qwen4exp-mtp-draft-head.patch` in this repo, which adds
    `graph_mtp` — the MTP block's own forward
  - `Qwen/Qwen3.8-Flash-Next` config.json (hc_count=4, hc_lowrank=320,
    head_dim=256, 512 experts top-10, mtp.num_hidden_layers=1)

The block is NOT a standalone language model. It consumes the trunk's
hyper-connection state h ([hc_count * hidden_size] = 10240 wide) alongside the
embedding of the *next* token, and predicts the token after that. See
scripts/mtp_dataprep.py for how the supervised positions are laid out.

Architecture notes that are easy to get wrong
--------------------------------------------
1. Two different RMSNorm reductions coexist. `hnorm` normalizes across the
   whole 10240-wide hyper-connection row; the per-block `hc_*_norm` normalizes
   within one 2560 stream and then scales all streams with a 10240 gamma.
2. `eh_proj` is the concatenation of the checkpoint's `fc_embedding` and
   `fc_hidden`, embedding first: A*e + B*h == [A|B] @ concat(e, h).
3. `wq` emits query and output gate *interleaved per head*, so the layout per
   token is [h0_q, h0_gate, h1_q, h1_gate, ...], not [all_q, all_gate].
4. Norm gammas are stored zero-centred; the effective scale is (1 + w). The
   GGUF converter bakes the +1 in, so weights loaded from GGUF must set
   zero_centered_norms=False or every norm in the block is wrong by one.
5. RoPE is interleaved mRoPE over a 0.25 partial rotary factor (64 of 256 head
   dims). For text-only batches the three mRoPE sections all carry the same
   position, which makes it identical to plain partial RoPE — that equivalence
   is why this file implements the simple form. It does NOT hold if image or
   video tokens are ever fed in.

Usage:
  python3 scripts/qwen4exp_nextn.py --self-test
  python3 scripts/qwen4exp_nextn.py --self-test --tiny   # fast, small dims
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class NextNConfig:
    hidden_size: int = 2560
    hc_count: int = 4                       # parallel residual streams
    hc_lowrank: int = 320                   # hyper-connection bottleneck
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    attention_bias: bool = False
    # HF checkpoints store zero-centred gammas; the GGUF converter pre-bakes +1
    zero_centered_norms: bool = True
    # mtp_use_dedicated_embeddings=False upstream: the head reuses the trunk
    # embedding table. The detached GGUF sidecar ships its own copy of it.
    tie_embedding_to_output: bool = False

    @property
    def hc_dim(self):
        return self.hc_count * self.hidden_size

    @property
    def rotary_dim(self):
        # even count required for the half-rotation pairing
        return int(self.head_dim * self.partial_rotary_factor) // 2 * 2

    @classmethod
    def from_hf_config(cls, path):
        """Build from an official Qwen4Exp config.json (text_config section)."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        tc = raw.get("text_config", raw)
        rope = tc.get("rope_parameters") or {}
        mtp = tc.get("mtp") or {}
        return cls(
            hidden_size=tc["hidden_size"],
            hc_count=tc["hc_count"],
            hc_lowrank=tc["hc_lowrank"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            num_experts=tc["num_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
            moe_intermediate_size=tc["moe_intermediate_size"],
            shared_expert_intermediate_size=tc["shared_expert_intermediate_size"],
            vocab_size=tc["vocab_size"],
            rms_norm_eps=tc["rms_norm_eps"],
            # the MTP block carries its own rope_theta
            rope_theta=float(mtp.get("rope_theta") or rope.get("rope_theta") or 1e7),
            partial_rotary_factor=float(
                rope.get("partial_rotary_factor") or tc.get("partial_rotary_factor") or 0.25
            ),
            attention_bias=bool(tc.get("attention_bias", False)),
        )


def rms_norm(x, weight, eps, zero_centered):
    """RMSNorm over the last axis, with the zero-centred gamma convention."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    scale = weight.float() + 1.0 if zero_centered else weight.float()
    return (x * scale).to(dtype)


class HyperConnectionMix(nn.Module):
    """Read one mixed [.., hidden] view out of the hc parallel streams.

    Mirrors build_hc_mix: a grouped RMSNorm, a low-rank sigmoid gate over the
    full stream row, the per-stream mean, and the injection weights that
    build_hc_combine later uses to scatter the block output back.
    """

    def __init__(self, cfg: NextNConfig, with_inject=True):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.Parameter(torch.zeros(cfg.hc_dim))
        self.down = nn.Linear(cfg.hc_dim, cfg.hc_lowrank, bias=False)
        self.up = nn.Linear(cfg.hc_lowrank, cfg.hc_dim, bias=False)
        self.inject = nn.Linear(cfg.hc_dim, cfg.hc_count, bias=False) if with_inject else None

    def forward(self, x):
        """x: [B, T, C, H] -> (mixed [B, T, H], inject [B, T, C] or None)"""
        cfg = self.cfg
        B, T, C, H = x.shape
        # [TIME] | INFO | QWEN4EXP_NEXTN.HYPERCONNECTIONMIX.FORWARD | the norm
        # reduces inside a single stream (H), then one hc_dim-wide gamma scales
        # every stream. Normalising across the flattened row instead would mix
        # the streams' scales and is what `hnorm` does — a different tensor.
        xn = rms_norm(x, self.norm.view(C, H), cfg.rms_norm_eps, cfg.zero_centered_norms)
        xn = xn.reshape(B, T, cfg.hc_dim)

        lo = F.silu(self.down(xn) / C)
        gate = torch.sigmoid(self.up(lo))
        gated = xn * gate

        mixed = gated.view(B, T, C, H).mean(dim=2)
        inject = self.inject(xn) if self.inject is not None else None
        return mixed, inject


def hc_combine(residual, block_out, inject, hc_count):
    """Scatter a block's output back across the streams.

    w = 2*sigmoid(inject/hc) is centred on 1, so a zeroed injection matrix
    degenerates to the plain residual add — the invariant the self-test pins.
    """
    w = 2.0 * torch.sigmoid(inject / hc_count)
    return residual + block_out.unsqueeze(2) * w.unsqueeze(-1)


def apply_rope(x, cos, sin, rotary_dim):
    """Rotate the leading `rotary_dim` channels, pass the rest through."""
    if rotary_dim <= 0:
        return x
    rot, keep = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    a, b = rot[..., :half], rot[..., half:]
    rotated = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
    return torch.cat((rotated, keep), dim=-1)


class GatedAttention(nn.Module):
    """Dense causal GQA with a sigmoid output gate carried inside wq.

    The draft block attends dense: upstream marks its layer_types as
    full_attention and skips the QSA indexer, whose weights ship in the GGUF
    but are never read.
    """

    def __init__(self, cfg: NextNConfig):
        super().__init__()
        self.cfg = cfg
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        # one projection for query *and* gate, interleaved per head
        self.q_proj = nn.Linear(cfg.hidden_size, nh * hd * 2, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, nkv * hd, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, nkv * hd, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(nh * hd, cfg.hidden_size, bias=False)
        self.q_norm = nn.Parameter(torch.zeros(hd))
        self.k_norm = nn.Parameter(torch.zeros(hd))

    def _rope_tables(self, T, device, dtype, positions=None):
        """RoPE tables for absolute positions, not 0..T-1.

        graph_mtp feeds the draft block build_inp_pos(), so the head rotates by
        each token's true position in the sequence. Training rows come from the
        middle of ~20k-token conversations, and rotating them as if they sat at
        position 0 would put the model in a phase regime it never sees at
        inference. `positions` is [B, T] absolute; None keeps the 0..T-1
        behaviour that the shape self-tests rely on.
        """
        cfg = self.cfg
        half = cfg.rotary_dim // 2
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, half, device=device).float() / half))
        pos = (torch.arange(T, device=device).float().unsqueeze(0)
               if positions is None else positions.to(device).float())
        ang = pos.unsqueeze(-1) * inv  # [B, T, half]
        return ang.cos().to(dtype), ang.sin().to(dtype)

    def forward(self, x, attn_mask=None, positions=None):
        cfg = self.cfg
        B, T, _ = x.shape
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

        qg = self.q_proj(x).view(B, T, nh, 2, hd)
        q = qg[:, :, :, 0, :]
        gate = qg[:, :, :, 1, :].reshape(B, T, nh * hd)

        q = rms_norm(q, self.q_norm, cfg.rms_norm_eps, cfg.zero_centered_norms)
        k = self.k_proj(x).view(B, T, nkv, hd)
        k = rms_norm(k, self.k_norm, cfg.rms_norm_eps, cfg.zero_centered_norms)
        v = self.v_proj(x).view(B, T, nkv, hd)

        cos, sin = self._rope_tables(T, x.device, q.dtype, positions)
        cos = cos[:, :, None, :]  # [B, T, 1, half], broadcasts over heads
        sin = sin[:, :, None, :]
        q = apply_rope(q, cos, sin, cfg.rotary_dim)
        k = apply_rope(k, cos, sin, cfg.rotary_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=attn_mask is None,
            scale=1.0 / math.sqrt(hd),
            enable_gqa=nkv != nh,
        )
        o = o.transpose(1, 2).reshape(B, T, nh * hd)
        return self.o_proj(o * torch.sigmoid(gate))


class SparseMoE(nn.Module):
    """512-expert SwiGLU MoE, top-10 renormalised, plus a gated shared expert.

    Experts are held as stacked 3-D parameters and dispatched with a scatter so
    the module stays a single nn.Module rather than 512 submodules.
    """

    def __init__(self, cfg: NextNConfig):
        super().__init__()
        self.cfg = cfg
        E, H, I = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.gate = nn.Linear(H, E, bias=False)
        self.w_gate = nn.Parameter(torch.empty(E, I, H))
        self.w_up = nn.Parameter(torch.empty(E, I, H))
        self.w_down = nn.Parameter(torch.empty(E, H, I))
        S = cfg.shared_expert_intermediate_size
        self.shared_gate_proj = nn.Linear(H, S, bias=False)
        self.shared_up_proj = nn.Linear(H, S, bias=False)
        self.shared_down_proj = nn.Linear(S, H, bias=False)
        # [TIME] | INFO | QWEN4EXP_NEXTN.SPARSEMOE | one scalar gate per token.
        # Kept 1-D to match the checkpoint, which stores it as a bare [hidden]
        # vector rather than a [1, hidden] matrix.
        self.shared_gate = nn.Parameter(torch.empty(H))
        for p in (self.w_gate, self.w_up, self.w_down, self.shared_gate):
            nn.init.normal_(p, std=0.02)

    def forward(self, x):
        cfg = self.cfg
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        N = flat.shape[0]

        probs = F.softmax(self.gate(flat).float(), dim=-1)
        weight, index = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
        weight = weight.to(flat.dtype)

        out = torch.zeros_like(flat)
        # [TIME] | INFO | QWEN4EXP_NEXTN.SPARSEMOE.FORWARD | walk only the
        # experts this batch actually routed to. With 512 experts and top-10 a
        # dense pass over every expert wastes ~50x the compute.
        for e in index.unique():
            rows, slot = (index == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            xe = flat[rows]
            h = F.silu(xe @ self.w_gate[e].t()) * (xe @ self.w_up[e].t())
            out.index_add_(0, rows, (h @ self.w_down[e].t()) * weight[rows, slot, None])

        shared = self.shared_down_proj(
            F.silu(self.shared_gate_proj(flat)) * self.shared_up_proj(flat)
        )
        out = out + shared * torch.sigmoid(flat @ self.shared_gate).unsqueeze(-1)
        return out.view(B, T, H)


class Qwen4ExpNextNBlock(nn.Module):
    """The single NextN block: joins (embedding, trunk state) then one layer."""

    def __init__(self, cfg: NextNConfig):
        super().__init__()
        self.cfg = cfg
        self.enorm = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.hnorm = nn.Parameter(torch.zeros(cfg.hc_dim))
        self.eh_proj = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hc_attn = HyperConnectionMix(cfg)
        self.attn = GatedAttention(cfg)
        self.hc_ffn = HyperConnectionMix(cfg)
        self.mlp = SparseMoE(cfg)

    def forward(self, embeds, h, attn_mask=None, positions=None):
        """embeds: [B,T,H] embedding of the next token. h: [B,T,hc_dim] trunk.

        positions: [B,T] absolute sequence positions for RoPE (None = 0..T-1).
        Returns the updated stream stack [B, T, C, H].
        """
        cfg = self.cfg
        B, T, _ = embeds.shape
        C, H = cfg.hc_count, cfg.hidden_size

        e = rms_norm(embeds, self.enorm, cfg.rms_norm_eps, cfg.zero_centered_norms)
        # hnorm spans the whole stream row, unlike the per-stream hc_*_norm
        hn = rms_norm(h, self.hnorm, cfg.rms_norm_eps, cfg.zero_centered_norms)
        hn = hn.view(B, T, C, H)
        # every stream is joined against the same embedding term
        e = e.unsqueeze(2).expand(B, T, C, H)
        x = self.eh_proj(torch.cat((e, hn), dim=-1))

        mixed, inject = self.hc_attn(x)
        x = hc_combine(x, self.attn(mixed, attn_mask, positions), inject, C)

        mixed, inject = self.hc_ffn(x)
        x = hc_combine(x, self.mlp(mixed), inject, C)
        return x


class Qwen4ExpNextNHead(nn.Module):
    """Draft head end to end: embedding, NextN block, output mixer, logits."""

    def __init__(self, cfg: NextNConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.block = Qwen4ExpNextNBlock(cfg)
        # model-level mixer, `output_hc_*` in the GGUF sidecar; it doubles as
        # the final norm, there is no separate one
        self.hc_head = HyperConnectionMix(cfg, with_inject=False)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embedding_to_output:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, next_token_ids, h, attn_mask=None, positions=None,
                return_state=False, last_only=False):
        """next_token_ids: [B,T] = t_{i+1}. h: [B,T,hc_dim] trunk state at i.

        positions: [B,T] absolute sequence positions for RoPE (None = 0..T-1).
        Returns logits [B,T,V] predicting t_{i+2}.

        return_state also returns the updated [B,T,hc_dim] stream stack. That is
        the tensor graph_mtp publishes as t_h_nextn -- the post-FFN hc_combine
        output, reshaped hc*n_embd wide -- and it is what the speculative draft
        loop feeds back in as the next step's h. Returning it lets a caller run
        the head autoregressively the way serving does.

        last_only applies lm_head to the final position only. A rollout needs
        one row of logits per step but the block output for all of them, and
        lm_head is the single most expensive projection here.
        """
        x = self.block(self.embed_tokens(next_token_ids), h, attn_mask, positions)
        mixed, _ = self.hc_head(x)
        logits = self.lm_head(mixed[:, -1:] if last_only else mixed)
        if return_state:
            B, T = next_token_ids.shape
            return logits, x.reshape(B, T, self.cfg.hc_dim)
        return logits

    def loss(self, next_token_ids, h, targets, attn_mask=None, ignore_index=-100,
             positions=None):
        logits = self(next_token_ids, h, attn_mask, positions)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )


# --- structural cross-check against the shipped GGUF ------------------------

# [TIME] | INFO | SCRIPTS.QWEN4EXP_NEXTN | parameter -> GGUF tensor. ggml
# stores dims fastest-axis-first, so a torch [out, in] weight appears as
# [in, out]; the check reverses one side before comparing. The four
# blk.N.indexer.* tensors are deliberately absent: the draft attends dense, so
# upstream loads them with TENSOR_SKIP and never reads them.
GGUF_MAP = {
    "embed_tokens.weight":               "token_embd.weight",
    "lm_head.weight":                    "output.weight",
    "hc_head.norm":                      "output_hc_norm.weight",
    "hc_head.down.weight":               "output_hc_down.weight",
    "hc_head.up.weight":                 "output_hc_up.weight",
    "block.enorm":                       "blk.{n}.nextn.enorm.weight",
    "block.hnorm":                       "blk.{n}.nextn.hnorm.weight",
    "block.eh_proj.weight":              "blk.{n}.nextn.eh_proj.weight",
    "block.hc_attn.norm":                "blk.{n}.hc_attn_norm.weight",
    "block.hc_attn.down.weight":         "blk.{n}.hc_attn_down.weight",
    "block.hc_attn.up.weight":           "blk.{n}.hc_attn_up.weight",
    "block.hc_attn.inject.weight":       "blk.{n}.hc_attn_inject.weight",
    "block.hc_ffn.norm":                 "blk.{n}.hc_ffn_norm.weight",
    "block.hc_ffn.down.weight":          "blk.{n}.hc_ffn_down.weight",
    "block.hc_ffn.up.weight":            "blk.{n}.hc_ffn_up.weight",
    "block.hc_ffn.inject.weight":        "blk.{n}.hc_ffn_inject.weight",
    "block.attn.q_proj.weight":          "blk.{n}.attn_q.weight",
    "block.attn.k_proj.weight":          "blk.{n}.attn_k.weight",
    "block.attn.v_proj.weight":          "blk.{n}.attn_v.weight",
    "block.attn.o_proj.weight":          "blk.{n}.attn_output.weight",
    "block.attn.q_norm":                 "blk.{n}.attn_q_norm.weight",
    "block.attn.k_norm":                 "blk.{n}.attn_k_norm.weight",
    "block.mlp.gate.weight":             "blk.{n}.ffn_gate_inp.weight",
    "block.mlp.w_gate":                  "blk.{n}.ffn_gate_exps.weight",
    "block.mlp.w_up":                    "blk.{n}.ffn_up_exps.weight",
    "block.mlp.w_down":                  "blk.{n}.ffn_down_exps.weight",
    "block.mlp.shared_gate_proj.weight": "blk.{n}.ffn_gate_shexp.weight",
    "block.mlp.shared_up_proj.weight":   "blk.{n}.ffn_up_shexp.weight",
    "block.mlp.shared_down_proj.weight": "blk.{n}.ffn_down_shexp.weight",
    "block.mlp.shared_gate":             "blk.{n}.ffn_gate_inp_shexp.weight",
}

_GGUF_TYPES = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
               5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
               12: ("<d", 8)}


def read_gguf_header(path):
    """Parse a GGUF header.

    Returns (kv, tensors, data_start) where tensors maps name ->
    (dims, ggml_type, offset). `offset` is relative to data_start, which is the
    end of the header rounded up to general.alignment (32 unless overridden).
    Dims stay in ggml order, which is the reverse of the torch shape.
    """
    import struct
    with open(path, "rb") as f:
        def num(t):
            fmt, size = _GGUF_TYPES[t]
            return struct.unpack(fmt, f.read(size))[0]

        def string():
            return f.read(num(10)).decode("utf8", "replace")

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = num(4), num(10)
                if et == 8:
                    return [string() for _ in range(n)]
                fmt, size = _GGUF_TYPES[et]
                f.read(size * n)
                return None
            return num(t)

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        num(4)
        n_tensors, n_kv = num(10), num(10)
        kv = {}
        for _ in range(n_kv):
            key = string()
            kv[key] = value(num(4))
        tensors = {}
        for _ in range(n_tensors):
            name = string()
            dims = [num(10) for _ in range(num(4))]
            ttype = num(4)
            offset = num(10)
            tensors[name] = (dims, ttype, offset)
        align = kv.get("general.alignment") or 32
        pos = f.tell()
        data_start = pos + (-pos % align)
    return kv, tensors, data_start


def read_gguf_tensor_shapes(path):
    """Return {tensor_name: [dims...]} from a GGUF header, ggml dim order."""
    _, tensors, _ = read_gguf_header(path)
    return {name: dims for name, (dims, _t, _o) in tensors.items()}


# only the float types a BF16 head export can contain; a quantized file needs
# dequantization this loader deliberately does not implement, so it says so
# rather than silently reading garbage
_TORCH_OF_GGML = {0: "float32", 1: "float16", 30: "bfloat16"}


def load_gguf_weights(model, path, block_index, verbose=True):
    """Copy the shipped head's weights out of a GGUF into `model`.

    Every parameter must be matched: an unmatched one would keep its random
    init and quietly poison training, so this raises instead. The four
    blk.N.indexer.* tensors are expected to go unused -- the patched trunk
    notes that "the draft attends dense, so the indexer weights are present but
    never read" -- but they still have to survive the round trip on export.
    """
    _, tensors, data_start = read_gguf_header(path)

    # The +1 of the zero-centred gamma convention is already baked into a GGUF
    # (llama.cpp's build_norm is a bare ggml_rms_norm followed by ggml_mul, with
    # no add), so adding it again would scale every norm in the block wrong.
    # Enforced here rather than left to callers: the failure is silent, it looks
    # like a merely mediocre model, and every submodule shares this one config.
    if model.cfg.zero_centered_norms:
        model.cfg.zero_centered_norms = False

    params = dict(model.named_parameters())
    loaded, total_bytes = {}, 0

    with open(path, "rb") as f:
        for pname, template in GGUF_MAP.items():
            gname = template.format(n=block_index)
            if gname not in tensors:
                raise KeyError(f"{gname} missing from {path}")
            if pname not in params:
                raise KeyError(f"{pname} is not a parameter of the model")
            dims, ttype, offset = tensors[gname]
            if ttype not in _TORCH_OF_GGML:
                raise NotImplementedError(
                    f"{gname} is ggml type {ttype}; this loader reads only "
                    f"F32/F16/BF16 (use the BF16 export, not a quantized one)")

            want = tuple(reversed(dims))  # ggml dim order is reversed vs torch
            p = params[pname]
            # Writers disagree about whether a vector is 1-D or 2-D with a
            # singleton: ffn_gate_inp_shexp is [2560] in the Q4_K_M export and
            # [2560,1] in the BF16 one. Same values either way, so match on
            # element count and let the reshape settle the layout.
            n_want = 1
            for d in want:
                n_want *= d
            if p.numel() != n_want:
                raise ValueError(f"{pname} is {tuple(p.shape)} ({p.numel()} elems) "
                                 f"but {gname} is {want} ({n_want} elems)")

            dtype = getattr(torch, _TORCH_OF_GGML[ttype])
            nbytes = p.numel() * dtype.itemsize
            f.seek(data_start + offset)
            buf = f.read(nbytes)
            if len(buf) != nbytes:
                raise IOError(f"{gname}: wanted {nbytes} bytes, got {len(buf)} "
                              f"(truncated download?)")
            t = torch.frombuffer(bytearray(buf), dtype=dtype).reshape(p.shape)
            with torch.no_grad():
                p.copy_(t)
            loaded[pname] = gname
            total_bytes += nbytes

    missing = sorted(set(params) - set(loaded))
    if missing:
        raise RuntimeError(f"parameters left at random init: {missing}")

    unused = sorted(n for n in tensors
                    if n not in {t.format(n=block_index) for t in GGUF_MAP.values()})
    if verbose:
        print(f"  loaded      : {len(loaded)} tensors, {total_bytes / 1e9:.2f} GB")
        print(f"  unused      : {unused or 'none'}")
    return {"loaded": len(loaded), "bytes": total_bytes, "unused": unused}


def write_gguf_weights(model, src_path, dst_path, block_index, verbose=True, only=None):
    """Copy `src_path` and overwrite mapped tensors from `model`.

    convert_hf_to_gguf.py --mtp wants a Hugging Face tree. We already have a
    working BF16 GGUF plus a 1.6M LoRA, so the cheap correct path is to patch
    the file we loaded from: header, tokenizer, metadata, and the four unused
    indexer tensors stay bit-identical.

    `only` is a set of parameter names to write. The model is constructed under
    bf16 default dtype, so F32 GGUF tensors (every RMSNorm gamma) would be
    rounded through bf16 if rewritten. Leaving them in the copied bytes keeps
    the serving norms exact; an unmatched name in `only` still raises.
    """
    import shutil
    shutil.copy2(src_path, dst_path)
    _, tensors, data_start = read_gguf_header(src_path)
    params = dict(model.named_parameters())
    written, total_bytes = {}, 0
    want = set(only) if only is not None else set(GGUF_MAP)

    with open(dst_path, "r+b") as f:
        for pname, template in GGUF_MAP.items():
            if pname not in want:
                continue
            gname = template.format(n=block_index)
            if gname not in tensors:
                raise KeyError(f"{gname} missing from {src_path}")
            if pname not in params:
                raise KeyError(f"{pname} is not a parameter of the model")
            dims, ttype, offset = tensors[gname]
            if ttype not in _TORCH_OF_GGML:
                raise NotImplementedError(
                    f"{gname} is ggml type {ttype}; writer emits only F32/F16/BF16")
            p = params[pname]
            n_want = 1
            for d in dims:
                n_want *= d
            if p.numel() != n_want:
                raise ValueError(f"{pname} is {tuple(p.shape)} ({p.numel()} elems) "
                                 f"but {gname} is {dims} ({n_want} elems)")
            dtype = getattr(torch, _TORCH_OF_GGML[ttype])
            t = p.detach().contiguous().cpu().to(dtype)
            # bf16 has no numpy dtype; view as uint16 so the bytes are exact
            raw = t.view(torch.uint16).numpy() if t.dtype == torch.bfloat16 \
                else t.numpy()
            buf = raw.tobytes()
            nbytes = p.numel() * dtype.itemsize
            if len(buf) != nbytes:
                raise ValueError(f"{gname}: packed {len(buf)} bytes, expected {nbytes}")
            f.seek(data_start + offset)
            f.write(buf)
            written[pname] = gname
            total_bytes += nbytes

    missing = sorted(want - set(written))
    if missing:
        raise RuntimeError(f"parameters not written: {missing}")
    if verbose:
        print(f"  wrote       : {len(written)} tensors, {total_bytes / 1e9:.2f} GB -> {dst_path}")
    return {"written": len(written), "bytes": total_bytes, "path": dst_path}


def check_gguf(cfg, gguf_path, block_index):
    """Prove every parameter lines up with a real tensor in the shipped head."""
    shapes = read_gguf_tensor_shapes(gguf_path)
    model = Qwen4ExpNextNHead(cfg)
    params = dict(model.named_parameters())

    print(f"  gguf        : {gguf_path.split('/')[-1]} ({len(shapes)} tensors)")
    bad, matched = [], 0
    for pname, template in GGUF_MAP.items():
        gname = template.format(n=block_index)
        if pname not in params:
            bad.append(f"{pname}: not a parameter of the module")
            continue
        if gname not in shapes:
            bad.append(f"{gname}: absent from the gguf")
            continue
        want = list(params[pname].shape)
        got = list(reversed(shapes[gname]))
        if want != got:
            bad.append(f"{gname}: gguf {got} != module {pname} {want}")
        else:
            matched += 1

    unmapped = set(params) - set(GGUF_MAP)
    extra = set(shapes) - {t.format(n=block_index) for t in GGUF_MAP.values()}
    print(f"  mapped      : {matched}/{len(GGUF_MAP)} parameters match a gguf "
          f"tensor by exact shape")
    print(f"  unmapped    : {sorted(unmapped) if unmapped else 'none'}")
    print(f"  gguf-only   : {sorted(extra) if extra else 'none'}")
    for line in bad:
        print(f"  MISMATCH    : {line}")
    return not bad and not unmapped


# --- checks -----------------------------------------------------------------

TINY = NextNConfig(
    hidden_size=64, hc_count=4, hc_lowrank=16, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
    moe_intermediate_size=32, shared_expert_intermediate_size=32,
    vocab_size=256,
)


def self_test(cfg, device):
    torch.manual_seed(0)
    B, T = 2, 6
    model = Qwen4ExpNextNHead(cfg).to(device)
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    h = torch.randn(B, T, cfg.hc_dim, device=device)

    print(f"  config      : hidden={cfg.hidden_size} hc={cfg.hc_count}x "
          f"lowrank={cfg.hc_lowrank} heads={cfg.num_attention_heads}/"
          f"{cfg.num_key_value_heads}x{cfg.head_dim} "
          f"experts={cfg.num_experts}/{cfg.num_experts_per_tok} "
          f"vocab={cfg.vocab_size} rot={cfg.rotary_dim}/{cfg.head_dim}")
    total = sum(p.numel() for p in model.parameters())
    block = sum(p.numel() for p in model.block.parameters())
    print(f"  parameters  : {total/1e9:.3f}B total, {block/1e9:.3f}B in the block, "
          f"{(total-block)/1e9:.3f}B in embed+lm_head+mixer")

    logits = model(ids, h)
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape
    print(f"  forward     : logits {tuple(logits.shape)} finite={torch.isfinite(logits).all().item()}")

    # the stream stack keeps its shape through the block
    stack = model.block(model.embed_tokens(ids), h)
    assert stack.shape == (B, T, cfg.hc_count, cfg.hidden_size), stack.shape
    print(f"  stream stack: {tuple(stack.shape)}")

    # zeroed injection must degenerate to a plain residual add
    resid = torch.randn(B, T, cfg.hc_count, cfg.hidden_size, device=device)
    out = torch.randn(B, T, cfg.hidden_size, device=device)
    zero = torch.zeros(B, T, cfg.hc_count, device=device)
    got = hc_combine(resid, out, zero, cfg.hc_count)
    want = resid + out.unsqueeze(2)
    err = (got - want).abs().max().item()
    assert err < 1e-5, err
    print(f"  hc_combine  : zero-inject == residual add (max err {err:.2e})")

    # causality: a later token must not change an earlier one's logits
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % cfg.vocab_size
    h2 = h.clone()
    h2[:, -1] = torch.randn_like(h2[:, -1])
    with torch.no_grad():
        a = model(ids, h)[:, :-1]
        b = model(ids2, h2)[:, :-1]
    drift = (a - b).abs().max().item()
    print(f"  causality   : prefix logits unchanged by the last position "
          f"(max drift {drift:.2e}){'  OK' if drift < 1e-3 else '  FAIL'}")
    assert drift < 1e-3, drift

    # zero-centred gammas: an all-zero norm weight must act as scale 1
    x = torch.randn(4, cfg.hidden_size, device=device)
    w = torch.zeros(cfg.hidden_size, device=device)
    unit = rms_norm(x, w, cfg.rms_norm_eps, True)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    err = (unit - ref).abs().max().item()
    assert err < 1e-5, err
    print(f"  norm gamma  : zero-centred w=0 is scale 1 (max err {err:.2e})")

    # gradients must reach every parameter, routed experts included
    tgt = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    model.loss(ids, h, tgt).backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    dead = [n for n in missing if "w_gate" not in n and "w_up" not in n and "w_down" not in n]
    print(f"  gradients   : {len(list(model.parameters())) - len(missing)}"
          f"/{len(list(model.parameters()))} tensors got finite grads")
    assert not dead, f"no gradient into {dead[:4]}"
    print(f"  expert grads: sparse by design; "
          f"{len([n for n in missing if n not in dead])} stacked expert tensors "
          f"unrouted in this batch" if missing else "  expert grads: all routed")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="small dims, runs on CPU in seconds")
    ap.add_argument("--hf-config", default=None, help="build config from an official config.json")
    ap.add_argument("--check-gguf", default=None, metavar="PATH",
                    help="cross-check parameter shapes against a NextN gguf")
    ap.add_argument("--block-index", type=int, default=48,
                    help="NextN block index inside the gguf (trunk n_layer)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.hf_config:
        cfg = NextNConfig.from_hf_config(args.hf_config)
    elif args.tiny:
        cfg = TINY
    else:
        cfg = NextNConfig()

    if args.check_gguf:
        print(f"structural check against {args.check_gguf}")
        ok = check_gguf(cfg, args.check_gguf, args.block_index)
        print("PASS" if ok else "FAIL")
        if not args.self_test:
            return 0 if ok else 1

    if not args.self_test:
        print(f"config: {cfg}")
        return 0

    print(f"self-test on {args.device} ({'tiny' if args.tiny else 'full'} config)")
    ok = self_test(cfg, torch.device(args.device))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
