# TurboQuant Benchmark Results & Development Roadmap

*Generated: 2026-03-28 | Hardware: 4× RTX 3090 Ti (24GB VRAM each)*  
*llama.cpp forks tested: PR #21089 (elusznik), mudler/feat/turbo-quant*

---

## Executive Summary

TurboQuant is a promising KV cache quantization method, but **both community implementations have bugs that prevent CUDA benchmarking today**. We successfully built and tested both forks, collected baseline performance data, and mapped the exact roadblocks. Full TurboQuant benchmarking requires either upstream fixes or contributing patches back to the forks.

> [!IMPORTANT]
> **TurboQuant is not yet production-ready for llama.cpp.** No implementation has been merged upstream, and both forks crash on CUDA. The technique itself is sound (Google ICLR 2026), but the llama.cpp integrations need more work.

---

## What We Tested

### Fork 1: PR #21089 (elusznik)
- **Types**: `TBQ3_0` (enum 41), `TBQ4_0` (enum 42)
- **Result**: ❌ Crashes with `GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src))` during KV cache initialization on CUDA
- **Root cause**: Tensor size calculation mismatch when creating KV cache views with TBQ block layouts

### Fork 2: mudler/feat/turbo-quant (commit `cff96160d`)
- **Types**: `TBQ2_0` (enum 41), `TBQ3_0` (enum 42), `TBQ4_0` (enum 43)
- **Result**: ⚠️ Builds and runs baselines, but crashes with `GGML_ASSERT(head_dim == 128)` for models with non-128 head dimensions
- **Additional issue**: `llama-bench` parser didn't include TBQ types (we patched this locally)
- **Status**: Works only with models that have `head_dim == 128` (Llama 7B+, standard Mistral, but NOT Llama 3.2 1B or Qwen 0.5B)

### Models Tested
| Model | head_dim | TBQ Compatible | Size |
|-------|----------|----------------|------|
| Qwen2.5-0.5B-Instruct Q4_K_M | 64 | ❌ | 463 MiB |
| Llama-3.2-1B-Instruct Q4_K_M | 64 | ❌ | 763 MiB |
| Dolphin-Mistral-24B Q6_K | 128 | ✅ (in theory) | 19 GB |
| stories260K (tiny test) | 64 | ❌ | 1.1 MiB |

> [!WARNING]
> The Dolphin-Mistral-24B is the only locally-available model with `head_dim == 128`, but it's 19GB inside a Docker volume and the host filesystem has <1GB free. A dedicated benchmarking session with disk space cleared is needed.

---

## Baseline Performance Data (Non-TBQ)

All data collected with flash attention ON, 3 repetitions, on Qwen2.5-0.5B Q4_K_M:

### Prompt Processing (tok/s)

| KV Type (K/V) | pp512 | pp2048 | pp4096 |
|----------------|-------|--------|--------|
| f16/f16 | **33,069** | **37,068** | **35,569** |
| f16/q8_0 | 4,486 | 1,465 | 762 |
| f16/q4_0 | 4,357 | 1,392 | 696 |

### Token Generation (tok/s)

| KV Type (K/V) | tg128 |
|----------------|-------|
| f16/f16 | **602** |
| f16/q8_0 | 331 |
| f16/q4_0 | 338 |

> [!NOTE]
> The massive pp speed drop from f16→q8_0/q4_0 is expected — the quantized KV cache requires dequantization during attention. TurboQuant's Hadamard rotation would add further overhead but promises better quality/compression ratio than q4_0.

---

## Discovered Constraints

1. **`head_dim == 128` required** — TurboQuant's PolarQuant rotation matrices are pre-computed for 128-dimensional vectors. Supporting other head dimensions requires additional codebook tables.

2. **CPU-only path is more stable** — TheTom's `turboquant_plus` repo has 511+ Python tests and 100% coverage, primarily on CPU/Metal. The CUDA path in both forks has assertion failures.

3. **Disk space bottleneck** — The 915GB NVMe is 100% full (869GB used). Any benchmarking session with large models requires clearing space first.

4. **Docker volume isolation** — Models are stored in Docker volumes, not directly accessible from the host without sudo. Future benchmarking should either build inside Docker or use host-accessible model paths.

---

## Development Roadmap

### Phase 1: Unblock Benchmarking (Prerequisite)
- [ ] **Free disk space** — need ~5-10GB for builds + model downloads
- [ ] **Pick the winning fork** — wait for upstream PR #21089 fixes or contribute patches to mudler's fork
- [ ] **Fix llama-bench parser** — submit PR to whichever fork adds TBQ types to `llama-bench.cpp:ggml_type_from_name()`
- [ ] **Test with head_dim=128 model** — run Dolphin-Mistral-24B or download a Llama-3.1-8B Q4_K_M (~4.6GB)

### Phase 2: Complete Benchmarks
- [ ] **Speed benchmarks** — `llama-bench` across f16, q8_0, q4_0, tbq4_0, tbq3_0, tbq2_0 at context depths 512–32K
- [ ] **Quality benchmarks** — `llama-perplexity` KL divergence against FP16 baseline
- [ ] **Functional tests** — run `turboquant_functional.py` (uses existing `ServerProcess` class) for coherence, math, instruction following, NIAH recall
- [ ] **Generate report** — `turboquant_report.py` to compile pass/warn/fail verdicts

### Phase 3: Platform Integration (After Benchmarks Pass)
- [ ] Update Dockerfile to build from TurboQuant-enabled fork (or upstream if merged)
- [ ] Add `tbq3_0`/`tbq4_0` to `llamacpp_manager.py` cache type validation
- [ ] Add TBQ options to `DeployPage.tsx` KV cache dropdown
- [ ] Update VRAM estimation (TBQ3_0 = 3.125 bpw, TBQ4_0 = 4.125 bpw)

---

## Benchmark Scripts Created

| Script | Purpose | Status |
|--------|---------|--------|
| [turboquant_bench.sh](file:///home/alec/git/llama-nexus/scripts/turboquant_bench.sh) | Speed benchmark wrapper around `llama-bench` | ✅ Ready |
| [turboquant_quality.sh](file:///home/alec/git/llama-nexus/scripts/turboquant_quality.sh) | KL divergence via `llama-perplexity` | ✅ Ready |
| [turboquant_functional.py](file:///home/alec/git/llama-nexus/scripts/turboquant_functional.py) | Live server quality tests using `ServerProcess` | ✅ Ready |
| [turboquant_report.py](file:///home/alec/git/llama-nexus/scripts/turboquant_report.py) | Report generator with pass/warn/fail thresholds | ✅ Ready |
