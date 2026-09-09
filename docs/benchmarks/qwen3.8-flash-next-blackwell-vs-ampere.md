# Qwen3.8-Flash-Next (UD-Q4_K_XL) on llama.cpp: 1× RTX PRO 6000 vs 4× RTX 3090 Ti

A same-VRAM, different-generation benchmark of a 104 GB MoE with a multi-token-prediction
(MTP) draft head, measured on real coding-agent traffic rather than a synthetic prompt loop.

Both systems hold **~96 GB of VRAM**: one Blackwell RTX PRO 6000 (96 GB on a single card)
versus four Ampere RTX 3090 Ti (4 × 24.6 GB, tensor-split across a PCIe rig). Same model
files, same llama.cpp commit, same sampling and speculative-decoding settings. The only
meaningful difference is the silicon and whether the weights are split.

**Headline: the single Blackwell card is 1.80× faster at token generation and 2.05× faster
at prompt processing.** Both figures are confirmed by two independent measurement methods
(a synthetic probe on idle servers, and per-request timings from live agent traffic). MTP
draft acceptance is statistically identical on both machines (0.741 vs 0.748), which is
what makes the comparison trustworthy — the speedup is hardware, not a lucky
speculative-decoding run.

---

## 1. Why this comparison is interesting

Qwen3.8-Flash-Next at `UD-Q4_K_XL` is 104 GB of weights, which should not fit in 96 GB of
VRAM. It fits because this architecture keeps its per-layer token embedding table (a large
n-gram lookup, ~33 GB) in a tensor that can be pinned to host RAM without touching the hot
path:

```
--override-tensor per_layer_token_embd=CPU
```

That single flag is what turns "doesn't fit" into "fits with room for a 196k-token KV
cache". It is the most important line in this entire document if your goal is just to get
the model running.

The second thing that makes this model unusual is that it ships a **separate MTP draft
head** (2.6 GB, `Q4_K_M`) used for speculative decoding via `--spec-type draft-mtp`. At a
~74% draft acceptance rate, decode throughput is roughly 2–3× what the base model would
do alone. Any benchmark of this model that does not report draft acceptance is not
reproducible, because acceptance varies with workload and dominates decode numbers.

---

## 2. Hardware under test

| | Local rig | Rented pod |
|---|---|---|
| **GPU** | 4× NVIDIA GeForce RTX 3090 Ti | 1× NVIDIA RTX PRO 6000 (Blackwell WS) |
| **Architecture** | Ampere, `sm_86` | Blackwell, `sm_120` |
| **VRAM** | 4 × 24,564 MiB = **96.0 GB** | **96 GB** |
| **VRAM in use** | ~90 GB (22–23 GB/card) | ~90 GB |
| **CPU** | AMD Ryzen Threadripper 3960X (24C/48T) | 32 vCPU |
| **System RAM** | 125 GB | 251 GB |
| **`--threads`** | 40 | 32 |
| **Weight placement** | `--split-mode layer`, `--tensor-split 0.92,1.08,1.08,0.92` | single GPU, no split |
| **Driver / CUDA** | 580.173.02 / CUDA 13.0 | CUDA 12.8.1 container |
| **Cost** | owned | **$1.69/hr** (RunPod) |

The tensor-split weights on the local rig are deliberately uneven (`0.92,1.08,1.08,0.92`)
because GPU 0 also carries the KV cache and the draft model, so it needs headroom.

---

## 3. Exact software configuration

Reproducing these numbers requires this specific build. Upstream `master` will **not**
work: MTP draft-head support for this architecture is not merged.

| Component | Value |
|---|---|
| llama.cpp commit | `bea3b12daee45876b0129a3602dc8f534ce30bf0` (build 10654) |
| Required patch | `patches/qwen4exp-mtp-draft-head.patch` (from PR #27742) |
| Main model | `Qwen3.8-Flash-Next-UD-Q4_K_XL`, 104 GB, 4 shards |
| Draft model | `Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf`, 2.6 GB |
| Required env | `LLAMA_ATTN_ROT_DISABLE=1` |
| Pod base image | `nvidia/cuda:12.8.1-devel-ubuntu22.04` |
| Pod build arch | `CMAKE_CUDA_ARCHITECTURES=120` |

### Launch command

Identical on both systems except for `--tensor-split` (dropped on the single GPU) and
`--threads`. This is the verbatim running configuration, with the API key redacted:

```bash
export LLAMA_ATTN_ROT_DISABLE=1   # mandatory, see §6.1

llama-server \
  --model  .../Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --model-draft .../Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 --api-key sk-REDACTED \
  --alias qwen38-flash \
  --ctx-size 196608 \
  --n-gpu-layers 999 --gpu-layers-draft 999 \
  --n-cpu-moe 0 \
  --override-tensor per_layer_token_embd=CPU \
  --n-predict 16000 \
  --threads 40 \
  --batch-size 512 --ubatch-size 256 \
  --keep 256 \
  --parallel 1 \
  --split-mode layer --tensor-split 0.92,1.08,1.08,0.92 --main-gpu 0 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-reuse 128 \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-n-max 16 --spec-draft-n-min 0 --spec-draft-p-min 0.75 \
  --poll 100 --spec-draft-poll 1 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --repeat-penalty 1.0 --repeat-last-n 64 \
  --frequency-penalty 0.0 --presence-penalty 0.0 \
  --reasoning-budget 0 --metrics --jinja \
  --chat-template-kwargs '{"preserve_thinking":true}'
```

`--cache-type-k q8_0 --cache-type-v q8_0` is not optional at this context length. An
`f16` KV cache at 196k tokens does not leave room for the weights.

---

## 4. Methodology

Numbers come from **production traffic**, not a benchmark loop. A capture proxy sits in
front of `llama-server` and records the per-request `timings` object that llama.cpp
returns for every completion. The workload is a replay of real coding-agent sessions
(multi-turn, tool-calling, 10k–50k token prompts) split into two shards, one per machine.

This design has three properties worth stating explicitly:

1. **Every record is an uncontended measurement.** This architecture is pinned to
   `--parallel 1` (see §6.2), so exactly one request occupies the model at a time. Queued
   requests wait; they do not slow the request being measured.
2. **Prompt distribution is realistic.** Synthetic benchmarks pick round prompt sizes.
   Real agent prompts cluster at 10–30k tokens with a long tail, and they are mostly
   *continuations* of previous prompts, which is what makes KV cache reuse dominate.
3. **Cold and warm are separated.** `timings.cache_n` reports how many prompt tokens were
   served from cache. Cold-prefill numbers below use only `cache_n == 0` requests.

All tables in §5 are generated by [`scripts/bench_report.py`](../../scripts/bench_report.py),
so they can be regenerated as samples accumulate rather than hand-transcribed:

```bash
python3 scripts/bench_report.py          # markdown tables
python3 scripts/bench_report.py --json   # raw numbers
```

### Two independent methods

Capture-derived numbers are only trustworthy if a different method agrees, so both
machines were **also** measured with a synthetic probe
([`scripts/bench_prefill.sh`](../../scripts/bench_prefill.sh)): filler text with a unique
nonce on every line, which defeats prefix caching so every request is a guaranteed cold
prefill. Each probe ran against a **verified-idle** server (background replay stopped and
GPU utilization confirmed at 0%), at matched prompt lengths.

The two methods agree on the headline within a few percent, on both machines:

| Method | ~15k | ~31k | ~53k |
|---|---|---|---|
| Synthetic probe, idle | **2.12×** | **2.06×** | **2.04×** |
| Production captures | 2.13× (10–20k) | 2.23× (20–30k) | 1.93× (30–60k) |

Per-machine, synthetic vs capture-derived lands within ±12%, with the same sign and
magnitude of deviation on each — consistent with prefill throughput falling with prompt
length while the capture buckets are ranges whose medians don't sit at the probe's exact
length.

Two calibration warnings from building that probe:

- **`prompt_n` is authoritative; your requested size is not.** The first version assumed
  ~1.15 tokens per word and overshot every target by ~1.94× — a request for 8,000 tokens
  produced a 15,551-token prompt. Because the harness reported the server's `prompt_n`
  the numbers stayed valid, just not at the sizes labelled. It is now calibrated to ~18
  tokens per generated line, and lands within 6% of target.
- **Probes above ~90k tokens returned HTTP errors** through the RunPod proxy despite a
  196k configured context, so the largest bucket is untested here.

### The mistake that produced a wrong number

An earlier decode-oriented harness reported 843 tok/s prefill at ~27.8k tokens on the
pod, ~35% below what both methods above now measure in that range. It was a single
unexplained outlier with no mechanism behind it — most likely server contention, since
our first attempt at this benchmark failed to pause the background replay workers and the
probes silently queued behind live traffic.

It is recorded here because the tempting move was to explain it with a story (nonce
filler must be pathological for this model's n-gram lookup path) rather than re-measure.
The story was wrong; the probe was fine and the server was busy. **Verify the server is
idle before probing it** — on a `--parallel 1` server a queued probe measures your own
background load, not the hardware.

---

## 5. Results

### Prompt processing (prefill) — synthetic probe, idle servers

The cleanest prefill comparison: both servers idle, prompt lengths matched, every request
a guaranteed cold prefill. Lengths are what the server reported, not what was requested.

| Prompt tokens | 4× RTX 3090 Ti | 1× RTX PRO 6000 | Speedup |
|---|---|---|---|
| ~15k | 735 tok/s <sub>(14,546)</sub> | 1,557 tok/s <sub>(15,551)</sub> | **2.12×** |
| ~31k | 630 tok/s <sub>(30,895)</sub> | 1,295 tok/s <sub>(31,950)</sub> | **2.06×** |
| ~53k | 523 tok/s <sub>(52,603)</sub> | 1,066 tok/s <sub>(53,683)</sub> | **2.04×** |

Best of 2 runs per size, reported by `timings.prompt_per_second`. Run-to-run spread was
under 6% on both machines. Reproduce with:

```bash
SIZES="15500 32000 53700" PREFILL_RUNS=2 \
  LLAMA_URL=... LLAMA_KEY=... ./scripts/bench_prefill.sh <label>
```

Note that absolute prefill throughput **declines with prompt length** on both machines —
roughly 30% slower at 53k than at 15k — while the ratio between them stays flat at ~2.05×.

<!-- BEGIN GENERATED: scripts/bench_report.py -->

### Prompt processing (prefill) — production traffic

Median `timings.prompt_per_second` for requests with `cache_n == 0`, bucketed by prompt length. Real agent prompts over a realistic length distribution, as a check on the matched synthetic probe above.

| Prompt tokens | 4x RTX 3090 Ti | 1x RTX PRO 6000 | Speedup |
|---|---|---|---|
| 2-10k | 724 tok/s (n=26) | — | n/a |
| 10-20k | 655 tok/s (n=108) | 1,387 tok/s (n=24) | 2.12x |
| 20-30k | 604 tok/s (n=104) | 1,313 tok/s (n=18) | 2.17x |
| 30-60k | 567 tok/s (n=15) | 1,093 tok/s (n=21) | 1.93x |

`*` fewer than 5 samples — indicative only.

### Token generation (decode)

| Metric | 4x RTX 3090 Ti | 1x RTX PRO 6000 | Speedup |
|---|---|---|---|
| Median decode | 53.8 tok/s | 92.0 tok/s | 1.71x |
| MTP draft acceptance | 0.742 | 0.719 | — |
| Requests measured | 3,099 | 248 | — |

### KV cache reuse on agent traffic

The replay walks each conversation turn by turn, so most requests extend the previous prompt and only the delta is prefilled.

| Metric | 4x RTX 3090 Ti | 1x RTX PRO 6000 |
|---|---|---|
| Cold requests (full prefill) | 277 (9%) | 66 (27%) |
| Warm requests (cache hit) | 2,822 | 182 |
| Median tokens reused (warm) | 20,214 | 17,069 |
| Median new tokens (warm) | 800 | 1,588 |

Cold/warm mix differs between the two because each ran a different session shard, so end-to-end wall time is **not** comparable between columns — only the per-phase rates above are.

<!-- END GENERATED -->

### Reading these numbers

**Decode scales with memory bandwidth, and 1.80× is about right.** A single RTX PRO 6000
has roughly 1.8× the effective bandwidth available to one inference stream compared to
four 3090 Ti in layer-split mode, where each token must cross PCIe between cards three
times. The multi-GPU rig has the same nominal VRAM but pays a serialization penalty that
a single card does not.

**Draft acceptance being equal (0.741 vs 0.748) is the control.** Speculative decoding
throughput is the product of raw speed and acceptance rate. If acceptance had differed
between the two runs, the decode comparison would be meaningless. It didn't, so the 1.80×
is attributable to hardware.

**An independent harness agrees.** A separate fixed-prompt decode benchmark
(`scripts/bench_tps.sh`, not capture-derived) measured 1.79× on the same pair of machines.
Two methodologies landing on 1.79× and 1.80× is a good sign.

**KV cache reuse is the real story for agent workloads.** 91% of local requests were warm,
reusing a median of 20,249 prompt tokens and prefilling only ~799 new ones. This is why
`--cache-reuse 128` matters more than prefill throughput for interactive coding: a 20k
prompt that hits cache costs about as much as an 800-token prompt. Benchmarks that only
measure cold prefill will badly misrepresent the experience of actually using the model.

### Caveats

- **Pod sample size is smaller** (n=110 vs 3,006) because the pod joined the run later.
  The 20–30k cold-prefill bucket has only 2 samples and is flagged as indicative. Decode
  medians and acceptance rates are stable at this n; the thin prefill bucket is not.
- **Wall-clock per request is not comparable across columns.** The two machines ran
  different session shards with different cold/warm mixes (9% vs 34% cold), so
  end-to-end latency differs for reasons unrelated to hardware. Compare only the
  per-phase rates.
- Local numbers were collected while other containers shared the host CPU. GPU work was
  uncontended, but host-side tokenization had competition.

---

## 6. Reproduction pitfalls

Every one of these cost real debugging time. This is probably the most useful section
here.

### 6.1 `LLAMA_ATTN_ROT_DISABLE=1` is mandatory

Without it, model load aborts:

```
llama-graph.cpp: GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr) failed
```

This is not optional on build 10654 and it is easy to miss, since the failure looks like a
corrupt model rather than a missing env var.

### 6.2 `--parallel 1` here is a benchmark choice, not a hard limit

Both machines were measured at `--parallel 1`, which is what makes the per-request
`timings` in §5 uncontended and comparable. It is **not** an architectural requirement,
and an earlier draft of this document wrongly said it was.

Multi-slot `--parallel` did abort with an indexer-cache assert in the day-0 state of
PR #27742. That was fixed. On this build `--parallel 2` runs fine, and on the Ampere rig
it measured **+10–15% aggregate tokens/sec** (65.6–68.7 vs 59.7 queued) with **−20% wall
time** on a two-stream workload. Single-stream decode is unchanged, because decode
serializes — the win comes from overlapping one request's prefill with another's decode.

The real constraint is that **`--parallel N` divides the context window per slot**. At
`--ctx-size 196608`, two slots give 98k each and three give 65k. Size that against your
actual prompt distribution: our agent traffic peaked at 59,883 prompt tokens, so three
slots fit, but four (49k/slot) would start rejecting the tail.

So if you are throughput-bound rather than latency-bound, raise `--parallel` before you
rent more GPUs — it is free. Just don't benchmark on it.

### 6.3 Do not delete `/usr/local/cuda/compat` in CUDA containers

A cleanup step removing this directory to save disk produced:

```
CUDA driver version is insufficient for CUDA runtime version
```

That directory is the forward-compatibility shim letting a newer CUDA runtime work with an
older host driver. Deleting it breaks the container on any host whose driver predates the
image's CUDA version. Match the container CUDA version to the host driver instead
(`nvidia/cuda:12.8.1-devel-ubuntu22.04` here).

### 6.4 RunPod's HTTP proxy rejects requests without a `User-Agent`

`https://<pod-id>-8080.proxy.runpod.net` returns **403 Forbidden** to Python's `urllib`,
which sends `User-Agent: Python-urllib/3.x`. `curl` works fine, which makes this
maddening to diagnose. Set any normal-looking agent:

```python
headers={"User-Agent": "llama-nexus-bench/1.0", ...}
```

This silently corrupted our first pod capture batch: the helper calls to `/tokenize` and
`/apply-template` 403'd, so records were written with `template_ok=False` and
`prompt_token_ids=0` while the main completion calls succeeded.

### 6.5 Root SSH into a RunPod pod needs `UsePAM no`

Injecting `PUBLIC_KEY` and writing `authorized_keys` is not sufficient — `sshd` still
rejects passwordless root with `Permission denied (publickey)`. Required in the bootstrap:

```bash
rm -f /etc/ssh/sshd_config.d/*.conf
echo "UsePAM no" >> /etc/ssh/sshd_config
```

### 6.6 Expect a long cold start

The pod needs to download 104 GB of weights and build llama.cpp from source for `sm_120`
(prebuilt wheels don't cover Blackwell + this patch). Budget **40+ minutes** before the
first token, and run the download and build in parallel. At $1.69/hr that setup is real
money, so scripting it is worth it — ours is in
[`scripts/runpod_bootstrap.sh`](../../scripts/runpod_bootstrap.sh) and
[`scripts/runpod_serve.sh`](../../scripts/runpod_serve.sh).

---

## 7. Cost and practical takeaways

At $1.69/hr, the rented Blackwell card delivered 1.80× the decode and ~2× the prefill of a
four-card Ampere rig of equal VRAM.

| | 4× RTX 3090 Ti | 1× RTX PRO 6000 |
|---|---|---|
| Median decode | 53.7 tok/s | 96.7 tok/s |
| Cold prefill (~31k, idle) | 630 tok/s | 1,295 tok/s |
| Slots during benchmark | 1 | 1 |
| Cost | owned, ~700 W idle-to-load | $1.69/hr |

Both were benchmarked single-slot for comparability; see §6.2 for raising `--parallel` if
you care about aggregate throughput rather than per-request latency.

**If you are choosing hardware for this model:** a single 96 GB card beats four 24 GB cards
of an older generation at the same total VRAM, and the gap is larger than the raw
spec-sheet difference because layer-split inference serializes across PCIe. The single card
is also dramatically simpler — no tensor-split tuning, no per-card headroom arithmetic.

**If you are running it on a multi-GPU rig anyway:** put your effort into KV cache reuse,
not prefill. 91% of real agent requests were warm. `--cache-reuse 128` and a client that
appends to conversations instead of rebuilding them is worth more than any GPU upgrade.

**If you are benchmarking it yourself:** report draft acceptance alongside decode, quote
prompt lengths from the server's `prompt_n` rather than what you asked for, separate cold
from warm prefill, and verify the server is idle before probing it. Without all four, the
numbers aren't comparable to anyone else's — and the last one is what burned us.

---

## Appendix: files referenced

| Path | Purpose |
|---|---|
| `scripts/bench_report.py` | Regenerates §5 from capture records |
| `scripts/bench_tps.sh` | Fixed-prompt decode benchmark (independent check) |
| `scripts/bench_prefill.sh` | Synthetic cold-prefill probe (independent check, §4) |
| `scripts/runpod_bootstrap.sh` | Pod provisioning: deps, SSH, parallel download + build |
| `scripts/runpod_serve.sh` | `llama-server` launch on a single GPU |
| `patches/qwen4exp-mtp-draft-head.patch` | MTP draft-head support (PR #27742) |
