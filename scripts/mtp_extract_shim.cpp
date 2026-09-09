// [TIME] | INFO | SCRIPTS.MTP_EXTRACT_SHIM | minimal C ABI over llama.cpp for
// pulling the trunk's NextN hidden state (h) out of the target model.
//
// Why a shim instead of driving libllama.so from ctypes directly: the two calls
// that matter live in src/llama-ext.h, a staging header with no `extern "C"`,
// so their symbols are C++-mangled; and llama_context_params has ~35 fields
// whose layout would have to be mirrored byte-for-byte in Python and rebroken
// on every llama.cpp bump. Compiling ~200 lines against the real headers costs
// seconds and moves both risks to the compiler. Everything above this line --
// plan execution, prefix reuse, output layout -- stays in Python.
//
// What h is: the patched qwen4exp trunk publishes res_hc (the wide 4-branch
// hyper-connection residual, n_embd_out = hc_mult * n_embd = 10240) as
// res->t_h_nextn, taken before the final mixer collapses it to n_embd. That is
// exactly the tensor the shipped NextN block consumes via its hnorm, so it is
// the input the draft head has to be trained against.
//
// Extraction runs in *unmasked* nextn mode, matching what speculative.cpp does
// for the target model: rows land densely, one per token of the batch just
// decoded, indexed by position within that batch. No per-token logits flags are
// needed, so a plain prefill yields h for every position it covered.

#include "llama.h"
#include "llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string g_err;

struct mtpx_ctx {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    int32_t n_embd_out = 0;
    int32_t n_batch    = 0;

    // batch scratch, allocated once at n_batch and refilled per decode
    std::vector<llama_token>  tok;
    std::vector<llama_pos>    pos;
    std::vector<int32_t>      n_seq_id;
    std::vector<llama_seq_id> seq_id0;
    std::vector<llama_seq_id*> seq_id;
    std::vector<int8_t>       logits;

    // keeps the pattern strings alive for as long as the model params reference them
    std::vector<llama_model_tensor_buft_override> bufts;
    std::vector<std::string> buft_patterns;
    std::vector<float> tsplit;
};

ggml_type parse_ggml_type(const char * s) {
    const std::string t = s ? s : "f16";
    if (t == "f32")     return GGML_TYPE_F32;
    if (t == "f16")     return GGML_TYPE_F16;
    if (t == "bf16")    return GGML_TYPE_BF16;
    if (t == "q8_0")    return GGML_TYPE_Q8_0;
    if (t == "q5_1")    return GGML_TYPE_Q5_1;
    if (t == "q5_0")    return GGML_TYPE_Q5_0;
    if (t == "q4_1")    return GGML_TYPE_Q4_1;
    if (t == "q4_0")    return GGML_TYPE_Q4_0;
    g_err = "unknown cache type: " + t;
    return GGML_TYPE_COUNT;
}

// bf16, not fp16: h is a pre-norm residual and its magnitudes are not bounded by
// fp16's 65504, so fp16 would risk silent infinities. bf16 keeps f32's exponent
// range, halves the file, and is the dtype the head trains in on Ampere anyway.
inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t) ((u >> 16) | 0x0040u); // keep NaN a NaN
    }
    const uint32_t r = u + 0x7fffu + ((u >> 16) & 1u); // round to nearest even
    return (uint16_t) (r >> 16);
}

} // namespace

extern "C" {

const char * mtpx_last_error(void) {
    return g_err.c_str();
}

// n_gpu_layers<0 offloads everything. tensor_split_csv may be null (even split).
// cpu_tensor_patterns is a comma-separated list of tensor-name patterns forced to
// host memory -- the served config needs "per_layer_token_embd" there, without
// which the per-layer embedding table alone will not fit in 96 GB of VRAM.
void * mtpx_open(const char * model_path,
                 int n_gpu_layers,
                 const char * tensor_split_csv,
                 int main_gpu,
                 int n_ctx,
                 int n_batch,
                 int n_ubatch,
                 int n_threads,
                 const char * kv_type,
                 int flash_attn,
                 const char * cpu_tensor_patterns) {
    g_err.clear();

    static bool backend_ready = false;
    if (!backend_ready) {
        llama_backend_init();
        backend_ready = true;
    }

    auto * h = new mtpx_ctx();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.main_gpu     = main_gpu;

    if (tensor_split_csv && *tensor_split_csv) {
        h->tsplit.assign(llama_max_devices(), 0.0f);
        size_t i = 0;
        const char * p = tensor_split_csv;
        while (*p && i < h->tsplit.size()) {
            h->tsplit[i++] = std::strtof(p, (char **) &p);
            if (*p == ',') p++;
        }
        mparams.tensor_split = h->tsplit.data();
    }

    if (cpu_tensor_patterns && *cpu_tensor_patterns) {
        std::string all(cpu_tensor_patterns);
        size_t start = 0;
        while (start <= all.size()) {
            const size_t comma = all.find(',', start);
            std::string one = all.substr(start, comma == std::string::npos ? comma : comma - start);
            if (!one.empty()) h->buft_patterns.push_back(one);
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        for (const auto & pat : h->buft_patterns) {
            h->bufts.push_back({ pat.c_str(), ggml_backend_cpu_buffer_type() });
        }
        h->bufts.push_back({ nullptr, nullptr }); // list is NULL-terminated
        mparams.tensor_buft_overrides = h->bufts.data();
    }

    h->model = llama_model_load_from_file(model_path, mparams);
    if (!h->model) {
        g_err = "failed to load model";
        delete h;
        return nullptr;
    }

    const ggml_type kv = parse_ggml_type(kv_type);
    if (kv == GGML_TYPE_COUNT) {
        llama_model_free(h->model);
        delete h;
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) n_ctx;
    cparams.n_batch         = (uint32_t) n_batch;
    cparams.n_ubatch        = (uint32_t) n_ubatch;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.type_k          = kv;
    cparams.type_v          = kv;
    cparams.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                         : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // llama_context::decode only copies h out when pooling is NONE; anything else
    // silently yields no rows.
    cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    cparams.embeddings      = false; // the normal embedding path is not what we want
    cparams.kv_unified      = true;  // single sequence, and prefix reuse wants one buffer

    h->ctx = llama_init_from_model(h->model, cparams);
    if (!h->ctx) {
        g_err = "failed to create context";
        llama_model_free(h->model);
        delete h;
        return nullptr;
    }

    // Dense rows, one per decoded token. Must be set before the first decode:
    // the buffer is sized inside output_reserve(), which every decode calls.
    llama_set_embeddings_nextn(h->ctx, true, /*masked*/ false);

    h->n_embd_out = llama_model_n_embd_out(h->model);
    h->n_batch    = n_batch;

    h->tok.resize(n_batch);
    h->pos.resize(n_batch);
    h->n_seq_id.assign(n_batch, 1);
    h->seq_id0.assign(n_batch, 0);
    h->seq_id.resize(n_batch);
    for (int i = 0; i < n_batch; i++) h->seq_id[i] = &h->seq_id0[i];
    h->logits.assign(n_batch, 0);

    return h;
}

int mtpx_n_embd_out(void * handle) {
    auto * h = (mtpx_ctx *) handle;
    return h ? h->n_embd_out : -1;
}

int mtpx_n_ctx_train(void * handle) {
    auto * h = (mtpx_ctx *) handle;
    return h ? llama_model_n_ctx_train(h->model) : -1;
}

// Decode n tokens at absolute positions [pos0, pos0+n). n must be <= n_batch:
// in unmasked mode the h buffer is exactly n_embd_out*n_batch wide, and decode
// asserts against that. Returns 0 on success.
int mtpx_decode(void * handle, const int32_t * tokens, int32_t n, int32_t pos0, int32_t want_last_logit) {
    auto * h = (mtpx_ctx *) handle;
    if (!h)            { g_err = "null handle"; return -1; }
    if (n <= 0)        { g_err = "empty decode"; return -1; }
    if (n > h->n_batch) { g_err = "decode larger than n_batch"; return -1; }

    for (int32_t i = 0; i < n; i++) {
        h->tok[i]    = tokens[i];
        h->pos[i]    = pos0 + i;
        h->logits[i] = 0;
    }
    // One output row keeps n_outputs > 0; h itself does not depend on it, but a
    // zero-output decode is a needless edge case to rely on.
    if (want_last_logit) h->logits[n - 1] = 1;

    llama_batch batch = {};
    batch.n_tokens = n;
    batch.token    = h->tok.data();
    batch.embd     = nullptr;
    batch.pos      = h->pos.data();
    batch.n_seq_id = h->n_seq_id.data();
    batch.seq_id   = h->seq_id.data();
    batch.logits   = h->logits.data();

    const int rc = llama_decode(h->ctx, batch);
    if (rc != 0) {
        g_err = "llama_decode returned " + std::to_string(rc);
    }
    return rc;
}

// Copy h for selected rows of the batch that was just decoded. `rows` are
// indices into that batch (0-based), not absolute sequence positions. `out`
// receives n_rows * n_embd_out bf16 values.
int mtpx_read_h_bf16(void * handle, const int32_t * rows, int32_t n_rows, uint16_t * out) {
    auto * h = (mtpx_ctx *) handle;
    if (!h) { g_err = "null handle"; return -1; }

    const float * base = llama_get_embeddings_nextn(h->ctx);
    if (!base) { g_err = "no nextn embeddings (wrong pooling type, or flag unset)"; return -1; }

    const int32_t W = h->n_embd_out;
    for (int32_t r = 0; r < n_rows; r++) {
        const int32_t src = rows[r];
        if (src < 0 || src >= h->n_batch) { g_err = "row out of batch range"; return -1; }
        const float * in = base + (size_t) src * W;
        uint16_t * dst = out + (size_t) r * W;
        for (int32_t j = 0; j < W; j++) dst[j] = f32_to_bf16(in[j]);
    }
    return 0;
}

// Drop KV entries from absolute position `keep` onward, so the next sample can
// reuse the shared prefix instead of re-prefilling it. keep<=0 clears the cache.
int mtpx_trim(void * handle, int32_t keep) {
    auto * h = (mtpx_ctx *) handle;
    if (!h) { g_err = "null handle"; return -1; }
    llama_memory_t mem = llama_get_memory(h->ctx);
    if (keep <= 0) {
        llama_memory_clear(mem, true);
        return 0;
    }
    return llama_memory_seq_rm(mem, 0, keep, -1) ? 0 : -1;
}

void mtpx_close(void * handle) {
    auto * h = (mtpx_ctx *) handle;
    if (!h) return;
    if (h->ctx)   llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

} // extern "C"
