# TurboQuant benchmarking: a technical blueprint

**TurboQuant is a data-oblivious vector quantization method from Google Research (ICLR 2026) that compresses KV caches to 2–4 bits with near-zero accuracy loss, and the llama.cpp community has already produced multiple working implementations — though none are merged upstream yet.** Building a benchmarking framework requires integrating the new `TBQ3_0`/`TBQ4_0` GGML types (PR #21089), leveraging llama.cpp's built-in `llama-bench` and `llama-perplexity` tools, and tracking quality via KL divergence rather than perplexity alone. The `alecKarfonta/llama-nexus` repository does not exist publicly, but alternative infrastructure patterns exist for orchestrating benchmark runs across specific llama.cpp commits.

---

## How TurboQuant achieves near-optimal compression

TurboQuant (arXiv: **2504.19874**, published April 28, 2025; presented at ICLR 2026) was developed by Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni. It targets **KV cache compression during inference** — not weight quantization — making it complementary to methods like GPTQ and AWQ.

The algorithm operates in two stages. **Stage 1 (PolarQuant)** applies a random orthogonal rotation (implemented as a randomized Hadamard transform) to the input vector, which transforms outlier-heavy coordinate distributions into a concentrated **Beta((d-1)/2, (d-1)/2) distribution** on [-1, 1]. Because this distribution is known analytically and independent of the data, a pre-computed **Lloyd-Max codebook** can be applied per-coordinate with no calibration data needed. This eliminates the 1–2 bits of overhead that traditional methods spend storing per-block normalization constants. **Stage 2 (QJL)** applies a 1-bit Quantized Johnson-Lindenstrauss transform to the quantization residual, producing an unbiased inner product estimator critical for attention score accuracy.

Two variants exist: **TurboQuant_mse** allocates all bits to Lloyd-Max quantization (best for reconstruction, simplest to implement), and **TurboQuant_prod** splits (b-1) bits to Lloyd-Max plus 1 bit to QJL (best for inner product estimation but requires a custom attention kernel). The community has converged on implementing TurboQuant_mse exclusively — multiple independent implementers found QJL residual correction injected noise when naively added back to reconstructed vectors.

The key theoretical guarantee: TurboQuant's MSE distortion is within a factor of **≈2.7×** of the information-theoretic lower bound (proven via Yao's minimax principle and Shannon's lower bound). Supported quantization levels with measured performance on unit-sphere vectors:

- **2-bit**: MSE = 0.118, cosine similarity = 0.940, **15.5× compression**
- **3-bit**: MSE = 0.035, cosine similarity = 0.983, **10.4× compression**
- **3.5-bit**: Zero measurable accuracy loss vs. full precision
- **4-bit**: MSE = 0.010, cosine similarity = 0.995, **7.9× compression**

Google claims **≥6× KV cache memory reduction** and up to **8× speedup** in attention logit computation on NVIDIA H100 GPUs — though that speedup figure is for attention computation only, not end-to-end inference. On LongBench with Llama-3.1-8B-Instruct, TurboQuant achieved **100% retrieval accuracy** on Needle-In-A-Haystack up to 104,000 tokens and matched full-cache LongBench scores at 3.5-bit. Perplexity change at 3-bit was reported as **<0.5%** for Llama 3 and Mistral.

---

## llama.cpp implementation status: PR #21089 and four community forks

Activity exploded immediately after the Google Research blog post on March 24, 2026. Here is every upstream reference and community fork, with specific identifiers:

**Upstream (ggml-org/llama.cpp):**

- **Issue #20977** — "Feature Request: TurboQuant support" — opened Mar 25, 2026 by **mudler** (Ettore Di Giacinto, LocalAI maintainer). Status: Open, labeled `enhancement`.
- **Issue #20979** — Duplicate, closed immediately in favor of #20977.
- **Discussion #20969** — "TurboQuant - Extreme KV Cache Quantization" — the main community coordination thread (15 upvotes). Started Mar 25, 2026 by **kth8** in the Ideas category.
- **PR #21089** — "ggml : add CPU TurboQuant KV cache types (TBQ3_0 / TBQ4_0)" — opened Mar 27, 2026 by **elusznik**. This is the first formal upstream PR. Status: **Open, not merged** as of March 28, 2026. Labels: `ggml`, `examples`, `server`, `testing`.

**New GGML types introduced by PR #21089:**

| Type Enum | Value | Block Structure | Bits/Value | Compression vs FP16 |
|-----------|-------|-----------------|------------|---------------------|
| `GGML_TYPE_TBQ3_0` | 41 | fp16 scale + QK_K × 3/8 packed indices | 3.25 | 4.9× |
| `GGML_TYPE_TBQ4_0` | 42 | fp16 scale + QK_K/2 packed indices | 4.25 | 3.8× |

The "TBQ" prefix (TurboBit Quant) was chosen deliberately to avoid collision with pre-existing `TQ1_0` and `TQ2_0` ternary types already in the codebase. CLI usage: `--cache-type-k tbq3_0 --cache-type-v tbq3_0`. Dot product functions `ggml_vec_dot_tbq3_0_q8_K` and `ggml_vec_dot_tbq4_0_q8_K` are implemented for flash attention compatibility.

**Community forks with working implementations:**

- **mudler/llama.cpp** (branch `feat/turbo-quant`) — Commit **`dee102db1bfd723c91f67138b8018ce35a6be477`**. Changes 16 files (+759/-1 lines). Adds `ggml-turboq-tables.h` (pre-computed Lloyd-Max codebooks), `ggml-turboq.c`/`.h` (core quantize/dequantize), and modifies `ggml.h`, `ggml-common.h`, `ggml-cpu.c`, `quants.c/h`, `ggml-quants.c/h`, `llama.h`, `llama-quant.cpp`, `quantize.cpp`.
- **TheTom/turboquant_plus** — Most mature implementation. 511+ Python tests, 100% code coverage, C port with **Metal GPU kernels** for Apple Silicon. Successfully ran Qwen 3.5 35B-A3B MoE with turbo3 KV cache on M5 Max, achieving q8_0 speed parity at **2747 vs 2694 tok/s prefill**. NIAH retrieval: **9/9 (100%)** with sparse V, beating q8_0 (7/9). Apache License 2.0.
- **spiritbuun/llama-cpp-turboquant-cuda** — CUDA fork tested on RTX 3090. Achieved 98.8% of q8_0 prefill speed. Norm correction makes turbo3 PPL beat q8_0 on CUDA.
- **Aaryan-Kapoor/llama.cpp** (branch `turboquant-tq3_0`) — Uses `GGML_TYPE_TQ3_0` naming. Block size: 32 values → 14 bytes (3.5 bpw). CPU-only, flash attention compatible.

**Practical VRAM impact** (from Discussion #20969, 70B Q4_K_M model with 34GB free VRAM): FP16 KV cache supports ~109K tokens, Q8_0 supports ~218K tokens, and **TBQ3_0 supports ~536K tokens** — nearly 5× the context length.

---

## The llama-nexus repository does not exist publicly

The repository at `https://github.com/alecKarfonta/llama-nexus` **is not publicly accessible**. The GitHub user `alecKarfonta` exists and has several public repositories (Walker, Gridworld, Libgdx-Projects), but no `llama-nexus` repo is visible. It may be private, deleted, or not yet created.

A different project, **LlamaEdge/llama-nexus**, exists but is unrelated — it's a Rust/WASM gateway service for orchestrating OpenAI-compatible API servers, not a benchmarking framework. For benchmarking TurboQuant across specific llama.cpp commits, the infrastructure would need to be built from scratch or adapted from existing tools like Phoronix Test Suite (`pts/llama-cpp`), the community's standardized benchmark discussions (#4167 for Apple Silicon, #15013 for NVIDIA CUDA), or `llama-benchy` (PyPI package for endpoint benchmarking).

---

## Benchmarking framework: tools, metrics, and quant types

### Built-in llama.cpp tools

**`llama-bench`** (location: `tools/llama-bench/`) measures pure inference throughput using randomly generated tokens. Default configuration: pp512 + tg128 with 5 repetitions. Key flags for TurboQuant KV cache testing: `-ctk tbq3_0 -ctv tbq3_0` (or `tbq4_0`), `-fa 0,1` for flash attention comparison, `-ngl 99` for full GPU offload. Outputs CSV, JSON, JSONL, Markdown, or SQL. The SQL output schema includes: `build_commit`, `build_number`, `cpu_info`, `gpu_info`, `type_k`, `type_v`, `flash_attn`, `avg_ts`, `stddev_ts`.

```bash
# TurboQuant KV cache benchmark example
llama-bench -m model-Q4_K_M.gguf -ngl 99 -fa 1 \
  -ctk f16,q8_0,q4_0,tbq3_0,tbq4_0 \
  -ctv f16,q8_0,q4_0,tbq3_0,tbq4_0 \
  -p 512,1024,2048,4096 -n 128 -o csv > turboquant_bench.csv
```

**`llama-perplexity`** (location: `tools/perplexity/`) measures model quality on WikiText-2. For KV cache quantization evaluation, the critical workflow is KL divergence analysis against an FP16 baseline:

```bash
# Record FP16 baseline logits (WARNING: 11-37 GiB output file)
./llama-perplexity -m model-f16.gguf -f wiki.test.raw --kl-divergence-base logits_f16.kld -ngl 80

# Measure KL divergence with TurboQuant KV cache
./llama-perplexity -m model-f16.gguf -f wiki.test.raw \
  --kl-divergence-base logits_f16.kld --kl-divergence \
  -ctk tbq3_0 -ctv tbq3_0 -fa 1 -ngl 80
```

**`llama-batched-bench`** tests multi-user scenarios with concurrent prompt processing and generation streams — essential for measuring TurboQuant's context-length advantage under load.

### Critical metrics for TurboQuant evaluation

Since TurboQuant targets KV cache compression (not weight quantization), the metrics differ from standard GGUF quant benchmarking:

- **KL divergence** (not just perplexity) is the primary quality metric. The **99.9th percentile KLD** and **max KLD** are more informative than mean KLD, as perplexity averages can mask significant per-token degradation. Unsloth's research demonstrates that models with similar perplexity can diverge meaningfully on downstream tasks.
- **Tokens/sec at various context depths** (-d flag) reveals how TurboQuant's memory savings translate to sustained throughput at long contexts where standard KV caches cause OOM or cache thrashing.
- **Maximum achievable context length** before OOM — the primary value proposition of KV cache compression.
- **NIAH (Needle-In-A-Haystack) retrieval accuracy** at extended contexts is critical because TurboQuant claims 100% retrieval at 104K tokens.
- **ΔPPL relative to FP16 KV** (not relative to other KV cache quant types) establishes the quality baseline.
- **Memory usage** at fixed context lengths, measurable via `nvidia-smi` or system memory monitoring.

### All current GGUF quant types for comparison baselines

Standard KV cache types available in llama.cpp: **f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1**. The new TurboQuant types (TBQ3_0, TBQ4_0) would slot in below q4_0 in bits-per-value, making q4_0 (4.5 bpw) and q8_0 (8.5 bpw) the most relevant comparison points.

For weight quantization baselines (to hold constant while varying KV cache type): **Q4_K_M** (~4.58 bpw, recommended default) and **Q8_0** (~8 bpw, near-lossless) are the community standards.

### Best practices for fair TurboQuant comparisons

**Record exact build information.** Performance changes significantly between llama.cpp commits. Since TurboQuant requires a specific fork/PR, the exact commit hash must be tracked. PR #21089 or mudler's commit `dee102db` are the current reference implementations.

**Test at multiple context depths.** TurboQuant's advantage grows with context length. Use `-d 0,4096,8192,16384,32768,65536` to show the performance crossover point where reduced KV cache size overcomes the Hadamard rotation overhead.

**Account for the Hadamard rotation cost.** TheTom's benchmarks show turbo3 is **3–8× slower** than q8_0 on per-token generation (due to O(d log d) rotation per KV block). This overhead is amortized at long contexts and batched inference but dominates at short contexts. Any fair benchmark must report both regimes.

**Use 5+ repetitions** (the llama-bench default) and monitor GPU thermals. First runs may be slower due to mmap page faults. The `--delay` flag between tests helps with thermal consistency.

---

## Conclusion

The technical ingredients for a TurboQuant benchmarking framework are now available but fragmented. **PR #21089** (types `TBQ3_0`/`TBQ4_0`, enum values 41/42) is the upstream integration target, while **mudler's commit `dee102db`** and **TheTom/turboquant_plus** provide the most complete working implementations today. The framework should orchestrate builds from specific commits (since nothing is merged yet), run `llama-bench` with `-ctk`/`-ctv` flags across all KV cache types, compute KL divergence via `llama-perplexity`, and crucially test at extended context lengths where TurboQuant's 4.9× KV compression translates to **~5× more usable context**. The `alecKarfonta/llama-nexus` repository is inaccessible, so the benchmark orchestration layer needs a different foundation — the community's standardized benchmark methodology from GitHub discussions #4167/#15013 and the SQL output format from `llama-bench` provide a solid starting schema. The most important open question is whether TheTom's observation of 3–8× generation slowdown at short contexts persists after upstream optimization, making context-depth-aware benchmarking essential rather than optional.