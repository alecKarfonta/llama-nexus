# Throughput Tuning — Qwen3.8-Flash-Next on 4x3090 Ti

Started as "what else can we speed up". Turned into two findings: **the live
config had drifted off the tuned optimum**, and **the prompt cache is thrashing**.
Almost everything else I proposed was already tested and settled in `notes.md`.

## Finding 1 — config drift (the big one)

`notes.md` 2026-08-30 round 2 records a tuned config. The server on 2026-09-05
was running something else on five axes, every one of them the worse value:

| Setting | Tuned (notes.md) | Live 2026-09-05 | Cost of the drift |
|---|---|---|---|
| `ubatch_size` | **1024** | 256 | cold prefill 996 → 905 tok/s (-10%) |
| `draft_p_min` | **0.9** | 0.75 | STRUCT 95.9 → 90.9 |
| `parallel_slots` | **2** | 1 | 2-stream aggregate 68 → 59.7 (-10-15%), wall +20% |
| `context_size` | **131072** | 196608 | 64k/slot at parallel 2; 196608 forces parallel 1 |
| `tensor_split` | **0.85,1.1,1.0,1.05** | 0.92,1.08,1.08,0.92 | code -4%, prose -2%, and GPU 0 loses the ~2.5 GB that lets ubatch 1024 fit |
| `mmproj` | **off** | off (live), but the deploy script forced it back on | its 0.9 GB is what blocks ubatch 1024 |

Restoring these is worth more than any new tuning in this document, and it needs
only a restart. `scripts/deploy_qwen38_flash.sh` now defaults to the tuned values
(see step 1) so a redeploy restores rather than perpetuates the drift.

## Finding 2 — the prompt cache is thrashing (new, not in notes.md)

Covered in step 4. `--cache-ram` sits at its 8192 MiB default while mean cache
entries are 980 MiB, so the cache holds about 8 conversations and evicts
constantly.

## Measured baseline (live `/metrics`, ~3 days uptime)

| Counter | Value |
|---|---|
| `prompt_seconds_total` | 17,636 s (**68%** of compute time) |
| `tokens_predicted_seconds_total` | 8,196 s (32%) |
| prompt throughput | 513 tok/s |
| generation throughput | 51 tok/s |
| prompt tokens processed / cached | 9.05M / 53.83M (85.6% hit rate) |
| `n_tokens_max` | 112,394 |
| draft tokens / accepted | 414,402 / 308,210 (**74.4%**) |
| drafts | 64,127 → mean draft 6.46, mean accepted 4.81 |

Decode is the minority of wall time. Note this baseline was collected under the
*drifted* config, so it is a floor, not a characterisation of the tuned setup.

Hardware state: VRAM free 2170/1277/1044/1842 MiB. Host RAM 125 GB, 13 GB in
swap, 4 GB free, PLE 83% resident (28.6 GiB) — healthy but no margin.

## Constraint

Two `capture_cursor_synth.py` replay jobs were in flight (400 sessions each,
shard 0 local + shard 1 to the pod). **Anything needing a restart is blocked
until they finish.** Steps are ordered so all no-restart work lands first.

## Steps

### 1. `deploy_qwen38_flash.sh` defaults — DONE 2026-09-05, no restart

First pass wrongly aligned the script to the *live* config. Corrected: defaults
are now the notes.md round-2 optimum — `ctx 131072`, `parallel 2`, `ubatch 1024`
(`batch 2048`), `tensor_split 0.85,1.1,1.0,1.05`, `mmproj null`, `p_min 0.9`,
`n_max 16`, all 4 GPUs, `n_cpu_moe 0`, `override_tensor per_layer_token_embd=CPU`.
`MMPROJ`, `PARALLEL`, `BATCH`, `UBATCH`, `TENSOR_SPLIT`, `MTP_DRAFT_P_MIN` are
now env-overridable. Verified by dry-run: renders valid JSON, nothing deployed.

Also removed two stale claims: the old default pushed 24 layers of experts
(39.0 GB) to CPU on 3 GPUs, and the header claimed `batch 1024 / ubatch 512`
which never matched anything.

### 2. PCIe lane mapping — DONE 2026-09-05, read-only

MSI TRX40 PRO 10G (MS-7C60):

| GPU | BDF | Root port | Slot | Slot is | Trained |
|---|---|---|---|---|---|
| 0 | 02:00.0 | 00:01.3 | PCIE5 | x8 | **x4** |
| 1 | 21:00.0 | 20:03.1 | PCIE7 | x16 | x16 |
| 2 | 49:00.0 | 40:01.3 | PCIE1 | x8 | x8 |
| 3 | 4a:00.0 | 40:03.1 | PCIE3 | x16 | x16 |

Root port 00:01.3 reports **LnkCap x8, LnkSta x4** — the port is configured for
x8 and the card advertises x16, so this is not bifurcation or a lane budget
problem. The link failed to train. Physical cause: reseat GPU 0, inspect slot
contacts, check any riser. No BIOS change needed. There is no free x16 slot; all
four GPUs occupy PCIE1/3/5/7, and GPU 2 at x8 is by design.

**Not a throughput lever.** Decode moves ~5-10 KB per inter-GPU hop (~20-40 µs
per token against 19.6 ms, under 0.2%); prefill moves ~1.3 MB per hop plus GPU
0's share of the PLE gather, so x4 → x8 saves ~250 µs of a 499 ms ubatch, about
0.05%. Fix it because it is free, not for tok/s.

### 3. Prefill benchmark — ALREADY EXISTS

`scripts/bench_prefill.sh` (untracked, newer) already does this properly:
nonce-prefixed cache-defeating prompts, size sweep, `cache_n` in output,
best-of-N to reject contended samples.

```bash
LLAMA_KEY=$(...) SIZES="8000 28000 48000" ./scripts/bench_prefill.sh some-label
```

Cleanup owed: `scripts/bench_prefill.py` is the older *tracked* version — bash
despite the `.py` extension, constant `"word " * n` prompt with no nonce, and a
header claiming `cache_reuse=0` forces reprocessing, false now that the server
runs `--cache-reuse 128`. Superseded; should be deleted.

Validation measurements taken along the way:

- **The nonce is load-bearing.** An identical prompt sent twice prefilled 3137
  tokens the first time and **4** the second.
- Prefill at ~3.1K depth is **~990 tok/s** vs the 513 tok/s production average —
  the gap is depth. Sweep, do not trust one number.
- Under replay contention, 3 calls totalling 6.4 s of server prompt time took
  126 s wall. Read `prompt_ms`, ignore wall time.

### 4. Prompt cache sizing — DONE (diagnosis); fix needs backend work

`--slot-save-path` exists but is the wrong tool; the build already has an
automatic prompt cache.

| Flag | Default | Running |
|---|---|---|
| `-cram, --cache-ram N` | **8192 MiB** | not passed → 8192 |
| `--cache-idle-slots` | enabled (requires cache-ram) | default |
| `-ctxcp, --ctx-checkpoints` | 32 | not passed → 32 |
| `--cache-reuse` | 0 | 128 |
| `-sps, --slot-prompt-similarity` | 0.10 | default |

Server log over this uptime: **141 evictions, 135.0 GiB evicted, mean entry
980 MiB, max 1597 MiB** (`W srv alloc: - making room for prompt cache entry`).
At ~980 MiB per entry an 8 GiB cache holds about **8 conversations** while the
replay cycles through hundreds — this is the mechanism behind the 9.05M
recomputed prompt tokens and 4.9 h of prefill.

This is consistent with round 2's finding that `cache-reuse` 128 vs 512 made no
difference because "the exact-prefix prompt cache does the work" — that cache is
the bottleneck, and it is undersized.

Blocker: `cache_ram` is **not plumbed through the backend**.
`backend/modules/managers/llamacpp_manager.py` emits `--cache-reuse` (~line 409)
and `--ctx-checkpoints` (~line 378) via `add_param_if_set` but has no
`--cache-ram`. Needs the same three touch points as `cache_reuse`: the
`add_param_if_set` call, the schema entry (~line 1592), and `editable_fields` in
`backend/routes/service.py` (~line 287).

Sizing caution: the prompt cache is host RAM and competes with the 28.8 GB PLE
mmap. Go 8192 → 16384 MiB first, watch eviction count and PLE residency.

### 5. DEFERRED — restore tuned config (needs restart)

Push the step-1 defaults and re-bench with `bench_tps.sh` + `bench_prefill.sh`.
Expected targets from notes.md: prose ~37 / code ~77 / JSON ~104-107, 2-stream
aggregate ~68, cold prefill ~996 tok/s, warm agent TTFT ~1.25 s.

### 6. DEFERRED — plumb and raise `--cache-ram` (needs restart)

Backend change from step 4, then 8192 → 16384 MiB, measure eviction rate.

## Settled — do not re-litigate

From `notes.md` 2026-08-30 rounds 1-2, all measured:

- **Power limit**: 50-129 W of 350 W under load, clocks pinned 1860-1935 MHz, no
  throttling. Launch/latency-bound. Raising the cap is pointless. (I suggested
  this before reading the log — it was already disproved.)
- **`draft_p_min` lower than 0.75**: the direction is *up*. 0.9 beat 0.75
  (95.9 vs 90.9). The per-position acceptance curve looks like it argues for
  longer drafts, but those deep-position rates are conditioned on drafts
  confident enough to reach that depth, so the curve is misleading and the A/B
  wins.
- **`draft_n_max`**: 16 is the winner; depth was the dominant knob (JSON +42%
  from n3 → n16).
- **`threads` 46 vs 40**: a wash — PLE lookups do not parallelize further.
- **`cache-reuse` 128 vs 512**: no TTFT difference.
- **`--parallel > 1`**: unblocked. The PR #27742 indexer-cache assert is gone on
  the patched build. The "parallel stays 1, not a tuning choice" comment in
  `scripts/runpod_serve.sh` is stale.

## Not in this pass

- RAM 125 → 256 GB. Now has a concrete justification: it is what would let
  `--cache-ram` grow enough to stop the eviction churn.
- Single 96 GB card (`scripts/runpod_bootstrap.sh`) — removes the layer-split
  pipeline, but not a "simple" change.

## Results

| Step | Date | Outcome |
|---|---|---|
| 1 deploy defaults | 2026-09-05 | Done, dry-run verified, not deployed |
| 2 PCIe map | 2026-09-05 | GPU 0 link-trains x4 in an x8 slot; ~0.05-0.2% impact |
| 3 prefill bench | 2026-09-05 | Tooling already existed; `.py` duplicate to delete |
| 4 prompt cache | 2026-09-05 | 141 evictions / 135 GiB; `cache_ram` not plumbed |
| 5 restore config | | blocked on replay |
| 6 cache-ram | | blocked on step 4 backend work |
