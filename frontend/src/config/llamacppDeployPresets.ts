/**
 * One-click llama.cpp deployment presets tuned for 4×3090 Ti conversation workloads.
 * Shared principles: thinking off, GGUF jinja template, f16 KV, moderate batches, short n-predict.
 */

export type DeployPresetCategory = 'conversation' | 'assistant' | 'rp' | 'fast' | 'multi';

export interface LlamaCppDeployPresetConfig {
  model?: {
    name?: string;
    variant?: string;
    context_size?: number;
    gpu_layers?: number;
    n_cpu_moe?: number;
    flash_attn?: 'on' | 'off' | 'auto';
  };
  template?: { selected: string };
  sampling?: Record<string, number | undefined>;
  performance?: Record<string, number | string | boolean | undefined>;
  server?: Record<string, number | boolean | undefined>;
  /** Empty object clears speculative decoding fields */
  speculative?: Record<string, unknown>;
}

export interface LlamaCppDeployPreset {
  id: string;
  name: string;
  description: string;
  category: DeployPresetCategory;
  config: LlamaCppDeployPresetConfig;
}

/** Qwen3.6 non-thinking conversation sampling (Tier 1 / 2) */
const QWEN36_CHAT_SAMPLING = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  min_p: 0.0,
  repeat_penalty: 1.0,
  presence_penalty: 1.5,
  frequency_penalty: 0.0,
  dry_multiplier: 0,
  dry_base: 2.0,
  dry_allowed_length: 2,
  dry_penalty_last_n: 0,
};

/** Modern min-p + DRY stack for long character / RP sessions (Tier 3) */
const QWEN36_RP_SAMPLING = {
  temperature: 0.85,
  top_p: 1.0,
  top_k: 0,
  min_p: 0.05,
  repeat_penalty: 1.0,
  presence_penalty: 0.0,
  frequency_penalty: 0.0,
  dry_multiplier: 0.8,
  dry_base: 1.75,
  dry_allowed_length: 2,
  dry_penalty_last_n: -1,
};

/** Shared runtime settings for 35B-A3B conversation tiers */
function conversationRuntime(overrides: LlamaCppDeployPresetConfig = {}): LlamaCppDeployPresetConfig {
  return {
    model: {
      context_size: 32768,
      gpu_layers: 999,
      n_cpu_moe: 0,
      flash_attn: 'auto',
      ...overrides.model,
    },
    template: { selected: '' },
    performance: {
      threads: 22,
      batch_size: 1024,
      ubatch_size: 512,
      num_keep: 1024,
      num_predict: 1024,
      parallel_slots: 1,
      split_mode: 'layer',
      main_gpu: 2,
      cache_type_k: 'f16',
      cache_type_v: 'f16',
      ...overrides.performance,
    },
    server: {
      reasoning_budget: 0,
      cache_reuse: 256,
      jinja: true,
      metrics: true,
      ...overrides.server,
    },
    speculative: {},
    sampling: overrides.sampling,
  };
}

export const LLAMACPP_DEPLOY_PRESETS: LlamaCppDeployPreset[] = [
  {
    id: 'chat-vtuber',
    name: 'Chat · VTuber',
    description:
      'Qwen3.6-35B-A3B uncensored heretic Q8_0 — personality-driven chat, low latency, presence penalty anti-repeat. Best for Twitch/YouTube/Kick bot replies.',
    category: 'conversation',
    config: conversationRuntime({
      model: {
        name: 'Qwen3.6-35B-A3B-uncensored-heretic',
        variant: 'Q8_0',
        context_size: 32768,
        gpu_layers: 999,
        n_cpu_moe: 0,
        flash_attn: 'auto',
      },
      sampling: QWEN36_CHAT_SAMPLING,
    }),
  },
  {
    id: 'chat-assistant',
    name: 'Chat · Assistant',
    description:
      'Qwen3.6-35B-A3B official Q8_0 — same speed profile as VTuber tier but official weights for accurate, instruction-following assistant chat.',
    category: 'assistant',
    config: conversationRuntime({
      model: {
        name: 'Qwen3.6-35B-A3B',
        variant: 'Q8_0',
        context_size: 32768,
        gpu_layers: 999,
        n_cpu_moe: 0,
        flash_attn: 'auto',
      },
      sampling: QWEN36_CHAT_SAMPLING,
    }),
  },
  {
    id: 'chat-rp',
    name: 'Chat · RP / Character',
    description:
      'Uncensored heretic Q8_0 with min-p + DRY sampling — warmer output and better long-session anti-repetition for persona-driven / roleplay chat.',
    category: 'rp',
    config: conversationRuntime({
      model: {
        name: 'Qwen3.6-35B-A3B-uncensored-heretic',
        variant: 'Q8_0',
        context_size: 32768,
        gpu_layers: 999,
        n_cpu_moe: 0,
        flash_attn: 'auto',
      },
      sampling: QWEN36_RP_SAMPLING,
    }),
  },
  {
    id: 'chat-multi-platform',
    name: 'Chat · Multi-Platform',
    description:
      'One model, 4 parallel slots (~16K ctx each) — run per-platform bot personalities on Twitch, YouTube, Kick, etc. without separate servers.',
    category: 'multi',
    config: conversationRuntime({
      model: {
        name: 'Qwen3.6-35B-A3B-uncensored-heretic',
        variant: 'Q8_0',
        context_size: 65536,
        gpu_layers: 999,
        n_cpu_moe: 0,
        flash_attn: 'auto',
      },
      performance: {
        parallel_slots: 4,
      },
      sampling: QWEN36_CHAT_SAMPLING,
    }),
  },
  {
    id: 'chat-fast',
    name: 'Chat · Fast / Side Bot',
    description:
      'Qwen3-4B-Instruct Q8_0 on a single GPU — minimal VRAM and latency for a low-priority side endpoint (change port if running alongside main server).',
    category: 'fast',
    config: conversationRuntime({
      model: {
        name: 'Qwen3-4B-Instruct',
        variant: 'Q8_0',
        context_size: 16384,
        gpu_layers: 999,
        n_cpu_moe: 0,
        flash_attn: 'auto',
      },
      performance: {
        num_predict: 512,
        num_keep: 512,
        parallel_slots: 1,
        split_mode: 'none',
        main_gpu: 0,
      },
      sampling: QWEN36_CHAT_SAMPLING,
    }),
  },
];

/** Sampling-only quick presets for the Sampling tab (model-family specific) */
export interface SamplingPreset {
  name: string;
  description: string;
  category: 'conversation' | 'coding' | 'rp' | 'assistant';
  sampling: Record<string, number>;
}

export const SAMPLING_PRESETS: SamplingPreset[] = [
  {
    name: 'Chat (Qwen3.6)',
    description: 'Qwen3.6 non-thinking conversation — temp 0.7, presence penalty 1.5',
    category: 'conversation',
    sampling: QWEN36_CHAT_SAMPLING,
  },
  {
    name: 'Assistant (Qwen3.6)',
    description: 'Official Qwen3.6 assistant chat — same sampler as VTuber tier',
    category: 'assistant',
    sampling: QWEN36_CHAT_SAMPLING,
  },
  {
    name: 'RP / Character (DRY)',
    description: 'Long character sessions — min-p 0.05 + DRY instead of presence penalty',
    category: 'rp',
    sampling: QWEN36_RP_SAMPLING,
  },
  {
    name: 'Coding (Qwen3-Coder)',
    description: 'Qwen3-Coder — temp 0.7, top-k 20, no presence/frequency penalties',
    category: 'coding',
    sampling: {
      temperature: 0.7,
      top_p: 0.8,
      top_k: 20,
      min_p: 0.0,
      repeat_penalty: 1.0,
      presence_penalty: 0.0,
      frequency_penalty: 0.0,
      dry_multiplier: 0,
    },
  },
];

export const DEPLOY_PRESET_CHIP_COLOR: Record<
  DeployPresetCategory,
  'primary' | 'secondary' | 'info' | 'success' | 'warning'
> = {
  conversation: 'primary',
  assistant: 'success',
  rp: 'secondary',
  fast: 'info',
  multi: 'warning',
};

export const SAMPLING_PRESET_CHIP_COLOR: Record<
  SamplingPreset['category'],
  'primary' | 'secondary' | 'info' | 'success'
> = {
  conversation: 'primary',
  assistant: 'success',
  rp: 'secondary',
  coding: 'info',
};
