import React, { useEffect, useLayoutEffect, useState, useMemo, useCallback, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Box,
  Grid,
  Typography,
  TextField,
  Select,
  MenuItem,
  Button,
  Alert,
  Paper,
  CircularProgress,
  Snackbar,
  IconButton,
  Tooltip,
  Menu,
  FormHelperText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  ToggleButtonGroup,
  ToggleButton,
  useMediaQuery,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  RestartAlt as RestartIcon,
  CheckCircle as ValidateIcon,
  Save as SaveIcon,
  CloudDownload as DownloadIcon,
  Refresh as ResetIcon,
  ExpandMore as ExpandMoreIcon,
  Visibility as VisionIcon,
  VisibilityOff as VisionOffIcon,
  Sync as RefreshIcon,
  FileDownload as FileDownloadIcon,
  RocketLaunch as DeployIcon,
  Storage as LibraryIcon,
  Dns as EndpointIcon,
  Memory as MemoryIcon,
  Tune as TuneIcon,
  Terminal as TerminalIcon,
  Chat as ChatIcon,
  Speed as SpeedIcon,
  Insights as MonitoringIcon,
  MoreHoriz as MoreIcon,
} from '@mui/icons-material'
import { apiService } from '@/services/api'
import { TEMPLATE_EMBEDDED_JINJA } from '@/constants/chatTemplate'
import type { ModelInfo } from '@/types/api'
import type { DeployRouteState, DeploymentLaunchContext } from '@/types/runtimeHandoff'
import { modelInfoFromDeployment } from '@/types/runtimeHandoff'
import axios, { AxiosError } from 'axios'
import LlamaCppCommitSelector from '@/components/LlamaCppCommitSelector'
import LogViewer, { LogViewerRef } from '@/components/LogViewer'
import { AgentToolsConfigSection } from '@/components/deploy/AgentToolsConfigSection'
import { MtpConfigSection } from '@/components/deploy/MtpConfigSection'
import { MtpStatsPanel } from '@/components/monitoring/MtpStatsPanel'
import {
  MetricTile,
  PageShell,
  SectionCard,
  StatusChip,
  type StatusChipConfigMap,
} from '@/components/common'
import { useMtpStats } from '@/hooks/useMtpStats'
import {
  defaultMtpSettings,
  modelMtpKey,
  resolveMtpForModel,
  saveMtpForModel,
} from '@/utils/mtpModelSettings'
import {
  applyMtpWorkloadProfile,
  clampMtpForModelCapability,
  type MtpWorkloadProfile,
} from '@/utils/mtpWorkloadProfiles'
import { debugLog } from '@/utils/debug'
import { settingsManager } from '@/utils/settings'
import { accentColors } from '@/utils/theme'
import {
  LLAMACPP_DEPLOY_PRESETS,
  SAMPLING_PRESETS,
  type LlamaCppDeployPreset,
  type SamplingPreset,
} from '@/config/llamacppDeployPresets'
import { getRoutePath } from '@/config/navigation'

function formatServiceActionError(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const ax = err as AxiosError<{ detail?: unknown }>
    const det = ax.response?.data?.detail
    if (typeof det === 'string') return det
    if (Array.isArray(det)) {
      return det.map((x: { msg?: string }) => (typeof x?.msg === 'string' ? x.msg : JSON.stringify(x))).join('; ')
    }
    if (det != null) return String(det)
    return ax.message || 'Request failed'
  }
  return err instanceof Error ? err.message : String(err)
}

// Local Config type mirrors backend snake_case fields to avoid camelCase/casing drift
interface Config {
  model: {
    name: string;
    variant: string;
    context_size: number;
    gpu_layers: number;
    lora?: string;
    lora_base?: string;
    mmproj?: string;
    rope_scaling?: 'none' | 'linear' | 'yarn';
    rope_freq_base?: number;
    rope_freq_scale?: number;
    n_cpu_moe?: number;
    flash_attn?: 'on' | 'off' | 'auto';
    kv_offload?: boolean;
    defrag_thold?: number;
  };
  sampling: {
    temperature: number;
    top_p: number;
    top_k: number;
    min_p: number;
    repeat_penalty: number;
    repeat_last_n: number;
    frequency_penalty: number;
    presence_penalty: number;
    dry_multiplier: number;
    dry_base: number;
    dry_allowed_length: number;
    dry_penalty_last_n: number;
    // Advanced sampling
    top_n_sigma?: number;
    typical_p?: number;
    xtc_probability?: number;
    xtc_threshold?: number;
    mirostat?: number;
    mirostat_lr?: number;
    mirostat_ent?: number;
    dynatemp_range?: number;
    dynatemp_exp?: number;
  };
  performance: {
    threads: number;
    threads_batch?: number;
    batch_size: number;
    ubatch_size: number;
    num_keep: number;
    num_predict: number;
    memory_f32?: boolean;
    mlock?: boolean;
    no_mmap?: boolean;
    numa?: 'distribute' | 'isolate' | 'numactl';
    split_mode?: 'none' | 'layer' | 'row';
    tensor_split?: string;
    main_gpu?: number;
    continuous_batching?: boolean;
    parallel_slots?: number;
    ctx_checkpoints?: number;
    cache_type_k?: 'f16' | 'q8_0' | 'q4_0';
    cache_type_v?: 'f16' | 'q8_0' | 'q4_0';
  };
  speculative?: {
    model_draft?: string;
    gpu_layers_draft?: number;
    ctx_size_draft?: number;
    draft_max?: number;
    draft_min?: number;
    draft_p_min?: number;
  };
  mtp?: {
    enabled?: boolean;
    workload_profile?: MtpWorkloadProfile;
    draft_n_max?: number;
    draft_n_min?: number;
    draft_p_min?: number;
  };
  execution?: {
    mode: 'gpu' | 'cpu';
    cuda_devices: string;
  };
  context_extension?: {
    yarn_ext_factor?: number;
    yarn_attn_factor?: number;
    yarn_beta_slow?: number;
    yarn_beta_fast?: number;
    group_attn_n?: number;
    group_attn_w?: number;
  };
  server: {
    host: string;
    port: number;
    api_key: string;
    timeout?: number;
    embedding?: boolean;
    system_prompt_file?: string;
    log_format?: 'json' | 'text';
    log_disable?: boolean;
    slots_endpoint_disable?: boolean;
    metrics?: boolean;
    cache_prompt?: boolean;
    cache_reuse?: number | null;
    reasoning_format?: 'deepseek' | 'none';
    reasoning_budget?: number;
    jinja?: boolean;
    chat_template_kwargs?: Record<string, unknown> | null;
    sleep_idle_seconds?: number;
  };
  template?: {
    selected?: string;
  };
}

// vLLM-specific config type
interface VllmConfig {
  backend_type: string;
  model: {
    name: string;
    served_name: string;
    dtype: string;
    quantization: string;
  };
  sampling: {
    temperature: number;
    top_p: number;
    top_k: number;
    repetition_penalty: number;
    frequency_penalty: number;
    presence_penalty: number;
  };
  performance: {
    max_model_len: number;
    gpu_memory_utilization: number;
    tensor_parallel_size: number;
    pipeline_parallel_size: number;
    data_parallel_size: number;
    max_num_seqs: number;
    max_num_batched_tokens: number;
    kv_cache_dtype: string;
    enforce_eager: boolean;
    enable_chunked_prefill: boolean;
    async_scheduling: boolean;
  };
  moe: {
    moe_backend: string;
    mamba_ssm_cache_dtype: string;
  };
  reasoning: {
    reasoning_parser: string;
    reasoning_parser_plugin: string;
  };
  speculative: {
    method: string;
    num_speculative_tokens: number;
    speculative_moe_backend: string;
  };
  tools: {
    enable_auto_tool_choice: boolean;
    tool_call_parser: string;
  };
  media: {
    video_pruning_rate: number;
    video_fps: number;
    video_num_frames: number;
  };
  environment: {
    vllm_nvfp4_gemm_backend: string;
    vllm_allow_long_max_model_len: string;
    vllm_flashinfer_allreduce_backend: string;
    vllm_use_flashinfer_moe_fp4: string;
    hf_token: string;
  };
  server: {
    host: string;
    port: number;
    api_key: string;
    trust_remote_code: boolean;
  };
}

// Field metadata from backend
interface FieldMeta {
  scope: 'shared' | 'vllm' | 'llamacpp';
  type: 'number' | 'text' | 'select' | 'boolean';
  description: string;
  vllm_flag?: string;
  llamacpp_flag?: string;
  llamacpp_equivalent?: string | null;
  vllm_equivalent?: string | null;
  mapping_note?: string;
  note?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
}
type FieldMetadata = Record<string, Record<string, FieldMeta>>;

const FRAMEWORK_BADGE_STATUS_CONFIG: StatusChipConfigMap = {
  shared: { color: accentColors.success, label: 'Shared' },
  vllm: { color: accentColors.info, label: 'vLLM' },
  llamacpp: { color: accentColors.warning, label: 'llama.cpp' },
}

const FrameworkBadge: React.FC<{ scope: string }> = ({ scope }) => {
  const normalizedScope = scope === 'vllm' || scope === 'llamacpp' ? scope : 'shared'

  return (
    <StatusChip
      status={normalizedScope}
      statusConfig={FRAMEWORK_BADGE_STATUS_CONFIG}
      height={18}
      sx={{
        fontSize: '0.625rem',
        '& .MuiChip-label': { px: 0.75 },
      }}
    />
  )
}

// Default values matching backend's load_default_config() method
const DEFAULT_VALUES = {
  model: {
    context_size: 128000,
    gpu_layers: 999,
    lora: '',
    lora_base: '',
    mmproj: '',
    rope_scaling: 'linear' as const,
    rope_freq_base: 0,
    rope_freq_scale: 0,
    n_cpu_moe: 0,
    flash_attn: 'auto' as const,
    kv_offload: undefined,
    defrag_thold: undefined,
  },
  sampling: {
    temperature: 0.7,
    top_p: 0.8,
    top_k: 20,
    min_p: 0.03,
    repeat_penalty: 1.05,
    repeat_last_n: 256,
    frequency_penalty: 0.3,
    presence_penalty: 0.2,
    dry_multiplier: 0.6,
    dry_base: 2.0,
    dry_allowed_length: 1,
    dry_penalty_last_n: 1024,
    top_n_sigma: undefined,
    typical_p: undefined,
    xtc_probability: undefined,
    xtc_threshold: undefined,
    mirostat: undefined,
    mirostat_lr: undefined,
    mirostat_ent: undefined,
    dynatemp_range: undefined,
    dynatemp_exp: undefined,
  },
  performance: {
    threads: -1,
    threads_batch: -1,
    batch_size: 2048,
    ubatch_size: 512,
    num_keep: 1024,
    num_predict: 64768,
    memory_f32: false,
    mlock: false,
    no_mmap: false,
    numa: '' as const,
    split_mode: 'layer' as const,
    tensor_split: '2,1',
    main_gpu: 0,
    continuous_batching: false,
    parallel_slots: 1,
    ctx_checkpoints: '' as unknown as number,
    cache_type_k: 'q4_0',
    cache_type_v: 'q4_0',
  },
  speculative: {
    model_draft: '',
    gpu_layers_draft: undefined,
    ctx_size_draft: undefined,
    draft_max: undefined,
    draft_min: undefined,
    draft_p_min: undefined,
  },
  mtp: {
    enabled: false,
    workload_profile: 'chat' as MtpWorkloadProfile,
    draft_n_max: 3,
    draft_n_min: 0,
    draft_p_min: 0.75,
  },
  execution: {
    mode: 'gpu' as const,
    cuda_devices: 'all',
  },
  context_extension: {
    yarn_ext_factor: 1.0,
    yarn_attn_factor: 1.0,
    yarn_beta_slow: 1.0,
    yarn_beta_fast: 32.0,
    group_attn_n: 1,
    group_attn_w: 512,
  },
  server: {
    host: '0.0.0.0',
    port: 8080,
    api_key: 'placeholder-api-key',
    timeout: 600,
    embedding: false,
    system_prompt_file: '',
    log_format: 'json' as const,
    log_disable: false,
    slots_endpoint_disable: false,
    metrics: false,
    cache_prompt: undefined,
    cache_reuse: undefined,
    reasoning_format: undefined,
    reasoning_budget: undefined,
    jinja: undefined,
    sleep_idle_seconds: undefined,
  },
};

// vLLM default values matching backend VLLMManager.load_default_config()
const VLLM_DEFAULT_VALUES: VllmConfig = {
  backend_type: 'vllm',
  model: {
    name: 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4',
    served_name: 'Nemotron-3-Nano-Omni-30B-A3B-Reasoning',
    dtype: 'auto',
    quantization: 'fp4',
  },
  sampling: {
    temperature: 0.7,
    top_p: 0.8,
    top_k: -1,
    repetition_penalty: 1.0,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
  },
  performance: {
    max_model_len: 16384,
    gpu_memory_utilization: 0.95,
    tensor_parallel_size: 1,
    pipeline_parallel_size: 1,
    data_parallel_size: 1,
    max_num_seqs: 16,
    max_num_batched_tokens: 8192,
    kv_cache_dtype: 'fp8',
    enforce_eager: true,
    enable_chunked_prefill: true,
    async_scheduling: true,
  },
  moe: {
    moe_backend: 'marlin',
    mamba_ssm_cache_dtype: 'float16',
  },
  reasoning: {
    reasoning_parser: 'nemotron_v3',
    reasoning_parser_plugin: '',
  },
  speculative: {
    method: '',
    num_speculative_tokens: 0,
    speculative_moe_backend: '',
  },
  tools: {
    enable_auto_tool_choice: true,
    tool_call_parser: 'qwen3_coder',
  },
  media: {
    video_pruning_rate: 0.5,
    video_fps: 2,
    video_num_frames: 256,
  },
  environment: {
    vllm_nvfp4_gemm_backend: 'marlin',
    vllm_allow_long_max_model_len: '1',
    vllm_flashinfer_allreduce_backend: 'trtllm',
    vllm_use_flashinfer_moe_fp4: '0',
    hf_token: '',
  },
  server: {
    host: '0.0.0.0',
    port: 8080,
    api_key: 'placeholder-api-key',
    trust_remote_code: true,
  },
};

/** Last tab index for each backend (Model = 0). */
const DEPLOY_TAB_MAX_LLAMACPP = 6
const DEPLOY_TAB_MAX_VLLM = 8
const LLAMACPP_CONTEXT_SIZE_MAX = 1000000

const DEPLOY_STATUS_CONFIG: StatusChipConfigMap = {
  control: { color: accentColors.primary, label: 'Deployment control' },
  running: { color: accentColors.success, label: 'Running' },
  stopped: { color: accentColors.warning, label: 'Stopped' },
  changed: { color: accentColors.info, label: 'Unsaved changes' },
  warning: { color: accentColors.warning, label: 'Deploy warning' },
  vllm: { color: accentColors.info, label: 'vLLM (HF weights)' },
  llamacpp: { color: accentColors.warning, label: 'llama.cpp' },
  vision: { color: accentColors.success, label: 'Vision ready' },
  text: { color: accentColors.slate, label: 'Text only' },
  'vram-fit': { color: accentColors.success, label: 'Fits in VRAM' },
  'vram-exceeds': { color: accentColors.rose, label: 'Exceeds VRAM' },
  sampling: { color: accentColors.purple, label: 'Sampling' },
  reference: { color: accentColors.info, label: 'Request defaults' },
  custom: { color: accentColors.slate, label: 'Custom' },
  performance: { color: accentColors.success, label: 'Runtime tuning' },
  memory: { color: accentColors.warning, label: 'Memory tuning' },
  reasoning: { color: accentColors.purple, label: 'Reasoning' },
  context: { color: accentColors.info, label: 'Context extension' },
  tools: { color: accentColors.warning, label: 'Tools & speculative' },
  server: { color: accentColors.info, label: 'Server' },
  secured: { color: accentColors.success, label: 'API key set' },
  open: { color: accentColors.warning, label: 'Open access' },
  environment: { color: accentColors.primary, label: 'Environment' },
  command: { color: accentColors.warning, label: 'Preview' },
  logs: { color: accentColors.info, label: 'Live logs' },
  conversation: { color: accentColors.primary, label: 'Conversation' },
  assistant: { color: accentColors.success, label: 'Assistant' },
  rp: { color: accentColors.purple, label: 'Roleplay' },
  fast: { color: accentColors.info, label: 'Fast' },
  multi: { color: accentColors.warning, label: 'Multi-user' },
  coding: { color: accentColors.info, label: 'Coding' },
  template: { color: accentColors.primary, label: 'Templates' },
  models: { color: accentColors.purple, label: 'Models' },
  key: { color: accentColors.slate, label: 'API key' },
}

const getDeployStatusColor = (status: string): string =>
  DEPLOY_STATUS_CONFIG[status]?.color ?? accentColors.slate

const deployPresetChipSx = (status: string, selected: boolean) => {
  const color = getDeployStatusColor(status)

  return {
    cursor: 'pointer',
    fontWeight: selected ? 700 : 600,
    bgcolor: selected ? alpha(color, 0.18) : alpha(color, 0.07),
    borderColor: selected ? alpha(color, 0.48) : alpha(color, 0.26),
    '&:hover': {
      bgcolor: alpha(color, selected ? 0.24 : 0.12),
    },
  }
}

const deployActionChipSx = (status: string) => {
  const color = getDeployStatusColor(status)

  return {
    cursor: 'pointer',
    borderRadius: '4px',
    fontWeight: 600,
    fontSize: '0.6875rem',
    borderColor: alpha(color, 0.32),
    '& .MuiChip-icon': { color },
    '&:hover': {
      bgcolor: alpha(color, 0.1),
      borderColor: alpha(color, 0.46),
    },
  }
}

const deployFieldSx = {
  '& .MuiOutlinedInput-root': {
    fontSize: '0.875rem',
    borderRadius: '8px',
    backgroundColor: 'background.default',
    '&.Mui-focused': {
      borderColor: 'primary.main',
    },
  },
  '& .MuiInputLabel-root': {
    fontSize: '0.875rem',
  },
}

const deploySelectSx = {
  fontSize: '0.875rem',
  '& .MuiOutlinedInput-root': {
    borderRadius: '8px',
    backgroundColor: 'background.default',
  },
}

const deployMenuProps = {
  disableScrollLock: true,
  PaperProps: {
    sx: {
      maxHeight: 'min(420px, calc(100dvh - 96px))',
      overflowY: 'auto',
    },
  },
} as const

const deploySectionSelectSx = {
  minWidth: { xs: 104, sm: 118 },
  height: 30,
  fontSize: '0.75rem',
  fontWeight: 800,
  borderRadius: '8px',
  bgcolor: 'action.hover',
  '& .MuiSelect-select': {
    py: 0.45,
    pl: 1,
    pr: '28px !important',
  },
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: 'divider',
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: alpha(accentColors.primary, 0.45),
  },
  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
    borderColor: accentColors.primary,
  },
} as const

const deployCommandPreviewSx = {
  p: 2,
  backgroundColor: 'rgba(2, 6, 23, 0.72)',
  color: 'grey.100',
  fontFamily: '"Fira Code", "SFMono-Regular", Consolas, monospace',
  fontSize: '0.875rem',
  lineHeight: 1.6,
  overflowX: 'auto',
  borderRadius: '8px',
  border: '1px solid',
  borderColor: 'divider',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
}

/** Merge API payload into defaults so nested sections always exist (avoids blank panels / crashes). */
function mergeVllmApiWithDefaults(raw: unknown): VllmConfig | null {
  if (raw == null || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const d = VLLM_DEFAULT_VALUES
  const obj = (x: unknown) =>
    x && typeof x === 'object' && !Array.isArray(x) ? (x as Record<string, unknown>) : {}

  return {
    backend_type: typeof r.backend_type === 'string' ? r.backend_type : d.backend_type,
    model: { ...d.model, ...obj(r.model) } as VllmConfig['model'],
    sampling: { ...d.sampling, ...obj(r.sampling) } as VllmConfig['sampling'],
    performance: { ...d.performance, ...obj(r.performance) } as VllmConfig['performance'],
    moe: { ...d.moe, ...obj(r.moe) } as VllmConfig['moe'],
    reasoning: { ...d.reasoning, ...obj(r.reasoning) } as VllmConfig['reasoning'],
    speculative: { ...d.speculative, ...obj(r.speculative) } as VllmConfig['speculative'],
    tools: { ...d.tools, ...obj(r.tools) } as VllmConfig['tools'],
    media: { ...d.media, ...obj(r.media) } as VllmConfig['media'],
    environment: { ...d.environment, ...obj(r.environment) } as VllmConfig['environment'],
    server: { ...d.server, ...obj(r.server) } as VllmConfig['server'],
  }
}

/** Deep-merge preset partials into an existing deploy config */
function mergeDeployPresetConfig(base: Config, preset: LlamaCppDeployPreset['config']): Config {
  const next: Config = {
    ...base,
    model: { ...base.model, ...(preset.model ?? {}) },
    sampling: { ...base.sampling, ...(preset.sampling ?? {}) },
    performance: { ...base.performance, ...(preset.performance ?? {}) },
    server: { ...base.server, ...(preset.server ?? {}) },
  }
  if (preset.speculative !== undefined) {
    next.speculative = {
      model_draft: '',
      gpu_layers_draft: undefined,
      ctx_size_draft: undefined,
      draft_max: undefined,
      draft_min: undefined,
      draft_p_min: undefined,
      ...preset.speculative,
    }
  }
  if (preset.server?.reasoning_budget === 0) {
    next.server = { ...next.server, reasoning_format: undefined }
  }
  return next
}

// LocalStorage key for persisting deployment settings (llama.cpp config, API keys, backend toggle, vLLM snapshot)
const DEPLOY_SETTINGS_KEY = 'llama-nexus-deploy-settings'

function readDeployStorage(): Record<string, unknown> {
  try {
    const raw = localStorage.getItem(DEPLOY_SETTINGS_KEY)
    if (!raw) return {}
    const o = JSON.parse(raw)
    return typeof o === 'object' && o !== null && !Array.isArray(o) ? o : {}
  } catch {
    return {}
  }
}

/** Merge-write so we never drop unrelated fields (e.g. backend when saving llama config). */
function writeDeployStorage(patch: Record<string, unknown>) {
  try {
    const prev = readDeployStorage()
    localStorage.setItem(DEPLOY_SETTINGS_KEY, JSON.stringify({ ...prev, ...patch, timestamp: Date.now() }))
  } catch (error) {
    debugLog('deploy', 'Failed to write deploy settings', error)
  }
}

const saveDeploySettings = (config: Config, selectedApiKey: string) => {
  writeDeployStorage({ config, selectedApiKey })
}

const loadDeploySettings = (): {
  config: Config | null
  selectedApiKey: string
  deployBackend: 'llamacpp' | 'vllm'
  vllmConfig: VllmConfig | undefined
} => {
  const p = readDeployStorage()
  const deployBackend = p.deployBackend === 'vllm' ? 'vllm' : 'llamacpp'
  return {
    config: (p.config as Config) ?? null,
    selectedApiKey: typeof p.selectedApiKey === 'string' ? p.selectedApiKey : '',
    deployBackend,
    vllmConfig:
      p.vllmConfig && typeof p.vllmConfig === 'object' && !Array.isArray(p.vllmConfig)
        ? (p.vllmConfig as VllmConfig)
        : undefined,
  }
}

// API Key management helpers
const API_KEYS_STORAGE_KEY = 'llama-nexus-api-keys';

const getStoredApiKeys = (): string[] => {
  try {
    const stored = localStorage.getItem(API_KEYS_STORAGE_KEY);
    if (stored) {
      const keys = JSON.parse(stored);
      return Array.isArray(keys) ? keys : [];
    }
  } catch (error) {
    debugLog('deploy', 'Failed to load API keys from localStorage', error);
  }
  return [];
};

const saveApiKeys = (keys: string[]) => {
  try {
    localStorage.setItem(API_KEYS_STORAGE_KEY, JSON.stringify(keys));
  } catch (error) {
    debugLog('deploy', 'Failed to save API keys to localStorage', error);
  }
};

/** Paths for optional llama.cpp params — cleared fields must be sent as null so merge drops them. */
const OPTIONAL_LLAMACPP_PATHS: string[] = (() => {
  const paths: string[] = []
  for (const [section, fields] of Object.entries(DEFAULT_VALUES)) {
    if (fields && typeof fields === 'object' && !Array.isArray(fields)) {
      for (const key of Object.keys(fields as object)) {
        paths.push(`${section}.${key}`)
      }
    }
  }
  return paths
})()

function setAtPath(obj: Record<string, unknown>, parts: string[], value: unknown) {
  let ref: Record<string, unknown> = obj
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i]
    if (typeof ref[p] !== 'object' || ref[p] === null || Array.isArray(ref[p])) {
      ref[p] = {}
    }
    ref = ref[p] as Record<string, unknown>
  }
  ref[parts[parts.length - 1]] = value
}

/** Serialize config for backend preview/save — explicit null clears optional params on merge. */
function prepareConfigForBackend(cfg: Config, modelMtpCapable = true): Record<string, unknown> {
  const clamped = clampMtpForModelCapability(cfg, modelMtpCapable)
  const out = JSON.parse(
    JSON.stringify(clamped, (_key, value) => (value === undefined ? null : value))
  ) as Record<string, unknown>
  for (const path of OPTIONAL_LLAMACPP_PATHS) {
    const parts = path.split('.')
    let src: unknown = clamped
    for (const p of parts) {
      if (src == null || typeof src !== 'object') {
        src = undefined
        break
      }
      src = (src as Record<string, unknown>)[p]
    }
    if (src === undefined || src === null) {
      setAtPath(out, parts, null)
    }
  }
  return out
}

// Reusable component for parameter inputs with descriptions and reset functionality
interface ParameterFieldProps {
  label: string;
  description: string;
  path: string;
  value: any;
  defaultValue: any;
  type?: 'text' | 'number' | 'select';
  options?: { value: string; label: string }[];
  min?: number;
  max?: number;
  step?: number;
  compact?: boolean;
  onChange: (path: string, value: any) => void;
  onReset: (path: string) => void;
}

const ParameterField: React.FC<ParameterFieldProps> = ({
  label,
  description,
  path,
  value,
  defaultValue,
  type = 'text',
  options,
  min,
  max,
  step,
  compact = false,
  onChange,
  onReset,
}) => {
  const isEmpty = value === '' || value === null || value === undefined;
  const isDefault = isEmpty;

  return (
      <Box sx={{ position: 'relative', minWidth: 0, pt: compact ? 0.35 : 0 }}>
      {type === 'select' ? (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Typography gutterBottom sx={{ fontSize: '0.875rem', mb: 0 }}>{label}</Typography>
            <Tooltip title={isEmpty ? 'Using llama-server default (not in command)' : 'Clear — omit from command (llama-server default)'}>
              <IconButton
                size="small"
                onClick={() => onReset(path)}
                disabled={isDefault}
                sx={{
                  opacity: isDefault ? 0.3 : 1,
                  transition: 'opacity 0.2s',
                  '&:hover': { opacity: isDefault ? 0.3 : 0.7 }
                }}
              >
                <ResetIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
            <Select
              size="small"
              fullWidth
              value={value || ''}
              MenuProps={deployMenuProps}
              onChange={(e) => onChange(path, e.target.value === '' ? null : e.target.value)}
              sx={{
              fontSize: '0.875rem',
              '& .MuiOutlinedInput-root': {
                borderRadius: '8px',
                backgroundColor: 'background.default',
              }
            }}
          >
            <MenuItem value="">
              <em>Use llama-server default ({typeof defaultValue === 'boolean' ? (defaultValue ? 'Enabled' : 'Disabled') : defaultValue})</em>
            </MenuItem>
            {options?.map((option) => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}
          </Select>
        </>
      ) : (
        <>
            {compact && (
              <Typography
                sx={{
                  fontSize: '0.75rem',
                  fontWeight: 800,
                  lineHeight: 1.25,
                  color: 'text.secondary',
                  mb: 0.55,
                }}
              >
                {label}
              </Typography>
            )}
          <TextField
            size="small"
            label={compact ? undefined : label}
            aria-label={compact ? label : undefined}
            type={type}
            fullWidth
            value={value ?? ''}
            placeholder={`Default: ${typeof defaultValue === 'boolean' ? (defaultValue ? 'Enabled' : 'Disabled') : defaultValue}`}
            onChange={(e) => {
              const inputValue = e.target.value;
              if (inputValue === '') {
                onChange(path, null);
              } else if (type === 'number') {
                const numValue = step && step < 1 ? parseFloat(inputValue) : parseInt(inputValue);
                onChange(path, isNaN(numValue) ? null : numValue);
              } else {
                onChange(path, inputValue);
              }
            }}
            inputProps={{ min, max, step }}
            InputProps={{
              endAdornment: (
                <Tooltip title={isEmpty ? 'Using llama-server default (not in command)' : 'Clear — omit from command (llama-server default)'}>
                  <IconButton
                    size="small"
                    onClick={() => onChange(path, null)}
                    sx={{
                      opacity: 1,
                      transition: 'opacity 0.2s',
                      '&:hover': { opacity: 0.7 }
                    }}
                  >
                    <ResetIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              ),
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                fontSize: '0.875rem',
                borderRadius: '8px',
                backgroundColor: isEmpty ? 'action.hover' : 'background.default',
                '&.Mui-focused': {
                  borderColor: 'primary.main'
                }
              },
              '& .MuiInputLabel-root': {
                fontSize: '0.875rem'
              }
            }}
          />
        </>
      )}
      <FormHelperText sx={{ mt: 0.35, fontSize: '0.72rem', lineHeight: 1.25, color: 'text.secondary' }}>
        {description}
      </FormHelperText>
    </Box>
  );
};

// Logging utility for DeployPage debugging
const deployLog = (context: string, message: string, data?: any) => {
  const timestamp = new Date().toISOString().split('T')[1].slice(0, 12)
  const prefix = `[Deploy ${timestamp}] [${context}]`
  if (data !== undefined) {
    debugLog('deploy', prefix, message, data)
  } else {
    debugLog('deploy', prefix, message)
  }
}

export const DeployPage: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const isNarrow = useMediaQuery('(max-width:599.95px)')
  const benchmarkPath = getRoutePath('benchmark')
  const chatPath = getRoutePath('chat')
  const modelsPath = getRoutePath('models')
  const monitoringPath = getRoutePath('monitoring')
  const routeState = useMemo(() => location.state as DeployRouteState | null, [location.state])
  const incomingModel = useMemo(() => {
    return routeState?.model && typeof routeState.model.name === 'string' ? routeState.model : null
  }, [routeState])
  const incomingBackend = routeState?.backend === 'vllm' || routeState?.backend === 'llamacpp'
    ? routeState.backend
    : null
  const [models, setModels] = useState<ModelInfo[]>([])
  const [config, setConfig] = useState<Config | null>(null)
  const [originalConfig, setOriginalConfig] = useState<Config | null>(null)
  const [commandLine, setCommandLine] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [validating, setValidating] = useState(false)
  const [actionLoading, setActionLoading] = useState<'start' | 'stop' | 'restart' | null>(null)
  const [runtimeMenuAnchorEl, setRuntimeMenuAnchorEl] = useState<HTMLElement | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [lastLaunchContext, setLastLaunchContext] = useState<DeploymentLaunchContext | null>(null)
  const [tab, setTab] = useState(0)
  const [showAdvancedModelSettings, setShowAdvancedModelSettings] = useState(false)
  const [validateErrors, setValidateErrors] = useState<string[] | null>(null)
  const [validateWarnings, setValidateWarnings] = useState<string[] | null>(null)
  const [currentModel, setCurrentModel] = useState<any | null>(null)
  const [templates, setTemplates] = useState<string[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState<string>('')
  const [templatesDir, setTemplatesDir] = useState<string>('')
  const [templatesLoading, setTemplatesLoading] = useState<boolean>(false)
  const [backend, setBackend] = useState<'llamacpp' | 'vllm'>(() => loadDeploySettings().deployBackend)
  const [vllmStatus, setVllmStatus] = useState<any | null>(null)
  const [vllmConfig, setVllmConfig] = useState<VllmConfig | null>(null)
  const [originalVllmConfig, setOriginalVllmConfig] = useState<VllmConfig | null>(null)
  const [vllmReloading, setVllmReloading] = useState(false)
  const [vllmReloadError, setVllmReloadError] = useState<string | null>(null)
  const [fieldMetadata, setFieldMetadata] = useState<FieldMetadata>({})

  // API Key management
  const [selectedApiKey, setSelectedApiKey] = useState<string>('')
  const [availableApiKeys, setAvailableApiKeys] = useState<string[]>([])
  const [newApiKey, setNewApiKey] = useState<string>('')

  // Ref for LogViewer to control logs
  const logViewerRef = useRef<LogViewerRef>(null)
  const deployTabsRef = useRef<HTMLDivElement | null>(null)

  const [llamacppMtpSupported, setLlamacppMtpSupported] = useState<boolean | null>(null)
  const [llamacppLlguidanceEnabled, setLlamacppLlguidanceEnabled] = useState<boolean | null>(null)
  const [llamacppBuildTag, setLlamacppBuildTag] = useState<string | undefined>()
  const prevModelMtpKey = useRef('')

  // Preset selection (full deploy + sampling-only)
  const [selectedDeployPreset, setSelectedDeployPreset] = useState<string | null>(null)
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)

  const buildLaunchContext = useCallback((action: 'start' | 'restart'): DeploymentLaunchContext => {
    const host = backend === 'vllm' ? vllmConfig?.server.host : config?.server.host
    const port = backend === 'vllm' ? vllmConfig?.server.port : config?.server.port
    const modelName = backend === 'vllm'
      ? vllmConfig?.model.served_name || vllmConfig?.model.name || 'Local vLLM model'
      : [config?.model.name, config?.model.variant].filter(Boolean).join(' / ') || 'Local llama.cpp model'

    return {
      action,
      backend,
      modelName,
      endpoint: host && port ? `${host === '0.0.0.0' ? 'localhost' : host}:${port}` : 'Backend proxy',
    }
  }, [backend, config, vllmConfig])

  // Log component mount
  useEffect(() => {
    deployLog('mount', 'DeployPage component mounted')
    return () => {
      deployLog('mount', 'DeployPage component unmounted')
    }
  }, [])

  useEffect(() => {
    if (backend !== 'vllm') {
      setVllmReloading(false)
    }
  }, [backend])

  useLayoutEffect(() => {
    setTab(0)
  }, [backend])

  useEffect(() => {
    writeDeployStorage({ deployBackend: backend })
  }, [backend])

  useEffect(() => {
    if (!incomingBackend || incomingModel || incomingBackend === backend) {
      return
    }

    setBackend(incomingBackend)
    writeDeployStorage({ deployBackend: incomingBackend })
    setSuccess(`Opened ${incomingBackend === 'vllm' ? 'vLLM' : 'llama.cpp'} deployment workspace`)
  }, [incomingBackend, incomingModel, backend])

  const refreshVllmStatus = useCallback(async () => {
    try {
      const r = await fetch('/api/v1/service/vllm/status')
      if (r.ok) setVllmStatus(await r.json())
    } catch {
      /* ignore */
    }
  }, [])

  useEffect(() => {
    if (backend !== 'vllm') return undefined
    refreshVllmStatus()
    const id = window.setInterval(() => refreshVllmStatus(), 5000)
    return () => window.clearInterval(id)
  }, [backend, refreshVllmStatus])

  // Reload config when backend selection changes
  useEffect(() => {
    if (loading) return

    const activeBackend = backend
    if (activeBackend === 'vllm') {
      setVllmReloading(true)
      setVllmReloadError(null)
    }

    let cancelled = false
    ;(async () => {
      deployLog('backend', `Backend changed to ${activeBackend}, reloading config`)
      try {
        const cfgRes = await fetch(`/api/v1/service/config?backend=${activeBackend}`)
        if (!cfgRes.ok) {
          const errText = await cfgRes.text()
          const msg = errText || `HTTP ${cfgRes.status}`
          deployLog('backend', 'Failed to load config:', msg)
          if (!cancelled && activeBackend === 'vllm') {
            setVllmReloadError(msg)
            setVllmConfig(null)
            setOriginalVllmConfig(null)
          }
        } else if (!cancelled) {
          const cfgJson = await cfgRes.json()
          deployLog('backend', 'Loaded config for backend:', activeBackend)
          if (activeBackend === 'vllm') {
            const persisted = loadDeploySettings()
            // Server (/data/vllm_deploy_config.json + in-memory) is authoritative so Save/Start survive refresh.
            const raw =
              cfgJson.config != null && typeof cfgJson.config === 'object'
                ? cfgJson.config
                : persisted.vllmConfig
            const merged = mergeVllmApiWithDefaults(raw)
            if (merged == null) {
              setVllmReloadError('Server returned an invalid or empty vLLM configuration')
              setVllmConfig(null)
              setOriginalVllmConfig(null)
            } else {
              setVllmConfig(merged)
              setOriginalVllmConfig(JSON.parse(JSON.stringify(merged)))
              setVllmReloadError(null)
              writeDeployStorage({ vllmConfig: merged })
            }
          } else {
            setConfig(cfgJson.config)
            setOriginalConfig(JSON.parse(JSON.stringify(cfgJson.config)))
          }
          setCommandLine(cfgJson.command || '')
        }

        const fieldsRes = await fetch(`/api/v1/service/config/fields?backend=${activeBackend}`)
        if (fieldsRes.ok && !cancelled) {
          const fieldsJson = await fieldsRes.json()
          setFieldMetadata(fieldsJson.fields || {})
        }
      } catch (e) {
        deployLog('backend', 'Failed to load config for backend:', e)
        if (!cancelled && activeBackend === 'vllm') {
          setVllmReloadError(e instanceof Error ? e.message : 'Failed to load vLLM config')
          setVllmConfig(null)
        }
      } finally {
        if (!cancelled && activeBackend === 'vllm') {
          setVllmReloading(false)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [backend, loading])

  // VRAM estimation state
  const [vramEstimate, setVramEstimate] = useState<{
    model_weights_mb: number
    kv_cache_mb: number
    compute_buffer_mb: number
    overhead_mb: number
    total_mb: number
    total_gb: number
    fits_in_vram: boolean
    available_vram_gb: number
    utilization_percent: number
    warnings: string[]
  } | null>(null)
  const [vramLoading, setVramLoading] = useState(false)

  // GPU list state
  const [gpuList, setGpuList] = useState<Array<{
    index: number
    name: string
    vram_total_mb: number
    vram_used_mb: number
    vram_free_mb: number
    utilization_percent: number
    temperature_c: number
  }>>([])
  const [gpusAvailable, setGpusAvailable] = useState(false)

  // mmproj files for VL models
  const [mmprojFiles, setMmprojFiles] = useState<Array<{
    name: string
    path: string
    size_mb: number
  }>>([])
  const [mmprojLoading, setMmprojLoading] = useState(false)
  const [downloadMmprojDialogOpen, setDownloadMmprojDialogOpen] = useState(false)
  const [mmprojAutoMatched, setMmprojAutoMatched] = useState(false)

  // Refresh mmproj file list from backend
  const refreshMmprojFiles = async () => {
    setMmprojLoading(true)
    try {
      const mmRes = await fetch('/v1/models/mmproj-files')
      if (mmRes.ok) {
        const mmData = await mmRes.json()
        if (mmData.success && mmData.data?.files) {
          setMmprojFiles(mmData.data.files)
        }
      }
    } catch (e) {
      deployLog('mmproj', 'Failed to refresh mmproj files:', e)
    } finally {
      setMmprojLoading(false)
    }
  }

  // Auto-match mmproj file to selected model name
  const autoMatchMmproj = (modelName: string, currentMmprojFiles: typeof mmprojFiles) => {
    if (!modelName || currentMmprojFiles.length === 0) return
    // Normalize for comparison: lowercase, remove hyphens/underscores
    const normalize = (s: string) => s.toLowerCase().replace(/[-_]/g, '')
    const normalizedModel = normalize(modelName)
    const match = currentMmprojFiles.find(f => {
      const normalizedFile = normalize(f.name)
      return normalizedFile.includes(normalizedModel)
    })
    if (match) {
      deployLog('mmproj', `Auto-matched mmproj for ${modelName}: ${match.name}`)
      setMmprojAutoMatched(true)
      return match.name
    } else {
      setMmprojAutoMatched(false)
      return null
    }
  }

  const applyDeployPreset = (preset: LlamaCppDeployPreset) => {
    deployLog('deployPreset', `Applying deploy preset: ${preset.id}`, preset.config)
    if (!config) {
      deployLog('deployPreset', 'ABORT: config is null')
      return
    }
    const nextConfig = mergeDeployPresetConfig(config, preset.config)
    setConfig(nextConfig)
    if (preset.config.template?.selected !== undefined) {
      setSelectedTemplate(preset.config.template.selected)
    }
    setSelectedDeployPreset(preset.id)
    setSelectedPreset(null)
    saveDeploySettings(nextConfig, selectedApiKey)
    updateCommandPreview(nextConfig)
    deployLog('deployPreset', 'Deploy preset applied successfully')
  }

  const applySamplingPreset = (preset: SamplingPreset) => {
    deployLog('preset', `Applying sampling preset: ${preset.name}`, { sampling: preset.sampling })
    if (!config) {
      deployLog('preset', 'ABORT: config is null')
      return
    }
    const nextConfig = {
      ...config,
      sampling: {
        ...config.sampling,
        ...preset.sampling,
      },
    }
    setConfig(nextConfig)
    setSelectedPreset(preset.name)
    setSelectedDeployPreset(null)
    saveDeploySettings(nextConfig, selectedApiKey)
    updateCommandPreview(nextConfig)
    deployLog('preset', 'Sampling preset applied successfully')
  }

  // Fetch VRAM estimation when config changes
  const fetchVramEstimate = async () => {
    if (!config?.model?.name) return;

    setVramLoading(true);
    try {
      const modelName = config.model.variant
        ? `${config.model.name}-${config.model.variant}`
        : config.model.name;

      const selectedModel = models.find(
        (m) => m.name === config.model?.name && m.variant === config.model?.variant,
      );

      const cudaDevices = config.execution?.cuda_devices || 'all';
      let availableVramGb = 24;
      if (gpuList.length > 0) {
        const selectedGpus =
          cudaDevices === 'all'
            ? gpuList
            : cudaDevices
                .split(',')
                .map((s) => parseInt(s.trim(), 10))
                .filter((n) => !Number.isNaN(n))
                .map((idx) => gpuList.find((g) => g.index === idx))
                .filter((g): g is NonNullable<typeof g> => g != null);
        const vramMb = selectedGpus.reduce((sum, g) => sum + g.vram_total_mb, 0);
        if (vramMb > 0) {
          availableVramGb = vramMb / 1024;
        }
      }

      const flashAttn = config.model?.flash_attn;
      const flashAttention = flashAttn !== 'off';

      const response = await fetch('/api/v1/vram/estimate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          context_size: config.model?.context_size || 4096,
          batch_size: config.performance?.batch_size || 1,
          gpu_layers: config.model?.gpu_layers ?? -1,
          kv_cache_type: config.performance?.cache_type_k || 'f16',
          flash_attention: flashAttention,
          parallel_slots: config.performance?.parallel_slots ?? 1,
          mtp_enabled: config.mtp?.enabled === true,
          mtp_nextn_layers: selectedModel?.mtpNextnLayers ?? 1,
          available_vram_gb: availableVramGb,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setVramEstimate(data);
      }
    } catch (err) {
      deployLog('vram', 'Failed to fetch VRAM estimate', err);
    } finally {
      setVramLoading(false);
    }
  };

  // Debounced VRAM estimation update
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchVramEstimate();
    }, 500); // Debounce 500ms

    return () => clearTimeout(timer);
  }, [
    config?.model?.name,
    config?.model?.variant,
    config?.model?.context_size,
    config?.performance?.batch_size,
    config?.model?.gpu_layers,
    config?.performance?.parallel_slots,
    config?.performance?.cache_type_k,
    config?.model?.flash_attn,
    config?.execution?.cuda_devices,
    config?.mtp?.enabled,
    gpuList,
  ]);

  useEffect(() => {
    const init = async () => {
      deployLog('init', 'Starting DeployPage initialization')
      try {
        setLoading(true)

        // Load persisted settings first
        const persistedSettings = loadDeploySettings()
        deployLog('init', 'Loaded persisted settings:', persistedSettings)

        // Load available local models (non-fatal - page should still work without model list)
        let modelList: ModelInfo[] = []
        try {
          deployLog('init', 'Fetching available models...')
          modelList = await apiService.getModels()
          deployLog('init', 'Loaded models:', { count: modelList.length, models: modelList.map(m => `${m.name}/${m.variant}`) })
          setModels(modelList)
        } catch (e) {
          deployLog('init', 'Failed to fetch models (non-fatal):', e)
        }

        // Load available GPUs
        try {
          deployLog('init', 'Fetching available GPUs...')
          const gpuRes = await fetch('/api/v1/system/gpus')
          const gpuData = await gpuRes.json()
          deployLog('init', 'GPU data:', gpuData)
          if (gpuData.available && gpuData.gpus) {
            setGpuList(gpuData.gpus)
            setGpusAvailable(true)
          }
        } catch (e) {
          deployLog('init', 'Failed to fetch GPU list (non-fatal):', e)
        }

        // Load available mmproj files
        let loadedMmprojFiles: typeof mmprojFiles = []
        try {
          deployLog('init', 'Fetching mmproj files...')
          setMmprojLoading(true)
          const mmRes = await fetch('/v1/models/mmproj-files')
          if (mmRes.ok) {
            const mmData = await mmRes.json()
            if (mmData.success && mmData.data?.files) {
              loadedMmprojFiles = mmData.data.files
              setMmprojFiles(loadedMmprojFiles)
            }
          }
        } catch (e) {
          deployLog('init', 'Failed to fetch mmproj files (non-fatal):', e)
        } finally {
          setMmprojLoading(false)
        }

        // Load current service config + generated command
        deployLog('init', 'Fetching service config...')
        const cfgRes = await fetch(`/api/v1/service/config`)
        if (!cfgRes.ok) {
          deployLog('init', 'Failed to fetch config:', { status: cfgRes.status, statusText: cfgRes.statusText })
          throw new Error('Failed to fetch configuration')
        }
        const cfgJson = await cfgRes.json()
        deployLog('init', 'Loaded service config:', cfgJson)

        // Use persisted config if available, otherwise use server config. Clone before applying
        // route handoffs and inferred defaults so the loaded source object remains untouched.
        const configToUse: Config = JSON.parse(JSON.stringify(persistedSettings.config || cfgJson.config))
        deployLog('init', 'Using config:', { source: persistedSettings.config ? 'persisted' : 'server', config: configToUse })

        if (incomingModel) {
          const nextVariant = incomingModel.variant || incomingModel.quantization || configToUse.model?.variant || ''
          configToUse.model = {
            ...configToUse.model,
            name: incomingModel.name,
            variant: nextVariant,
          }
          if (incomingModel.contextLength) {
            configToUse.model.context_size = incomingModel.contextLength
          }
          configToUse.model.mmproj = autoMatchMmproj(incomingModel.name, loadedMmprojFiles) || ''
          setBackend('llamacpp')
          writeDeployStorage({ deployBackend: 'llamacpp' })
          deployLog('init', 'Applied model handoff from library:', {
            name: incomingModel.name,
            variant: nextVariant,
            contextLength: incomingModel.contextLength,
            localPath: incomingModel.localPath,
          })
        }

        // Ensure execution settings exist with defaults
        if (!configToUse.execution) {
          configToUse.execution = { mode: 'gpu', cuda_devices: 'all' }
        }
        if (!configToUse.mtp) {
          configToUse.mtp = { ...defaultMtpSettings() }
        }
        if (configToUse.model?.name) {
          const initModel =
            modelList.find(
              (m) =>
                m.name === configToUse.model.name && m.variant === configToUse.model.variant
            ) ?? modelList.find((m) => m.name === configToUse.model.name)
          const initMtpCapable = initModel?.mtpCapable === true
          const resolvedMtp = resolveMtpForModel(
            configToUse.model.name,
            configToUse.model.variant || '',
            { mtpCapable: initMtpCapable }
          )
          configToUse.mtp = { ...defaultMtpSettings(), ...configToUse.mtp, ...resolvedMtp }
          const profile = (configToUse.mtp.workload_profile ?? 'chat') as MtpWorkloadProfile
          const applied = applyMtpWorkloadProfile(profile, configToUse)
          configToUse.mtp = applied.mtp
          configToUse.performance = { ...configToUse.performance, ...applied.performance }
          configToUse.server = { ...configToUse.server, ...applied.server }
          Object.assign(
            configToUse,
            clampMtpForModelCapability(configToUse, initMtpCapable)
          )
        }
        if (configToUse.model?.name) {
          prevModelMtpKey.current = modelMtpKey(configToUse.model.name, configToUse.model.variant || '')
        }

        setConfig(configToUse)
        setOriginalConfig(JSON.parse(JSON.stringify(cfgJson.config)))
        // Sync template dropdown with the config we are editing (not stale server-only state)
        if (configToUse.template?.selected !== undefined) {
          setSelectedTemplate(configToUse.template.selected || '')
        }
        // Command preview must match the config shown in the form (localStorage may differ from server)
        void updateCommandPreview(configToUse as Config)
        if (incomingModel) {
          saveDeploySettings(configToUse, persistedSettings.selectedApiKey || '')
          setSuccess(`Prepared ${incomingModel.name}${configToUse.model.variant ? ` / ${configToUse.model.variant}` : ''} for deployment`)
        }

        // Fetch field metadata for the current backend
        try {
          deployLog('init', 'Fetching field metadata...')
          const fieldsRes = await fetch(`/api/v1/service/config/fields?backend=llamacpp`)
          if (fieldsRes.ok) {
            const fieldsJson = await fieldsRes.json()
            setFieldMetadata(fieldsJson.fields || {})
          }
        } catch (e) {
          deployLog('init', 'Failed to fetch field metadata (non-fatal):', e)
        }

        // Load currently deployed model info (best-effort)
        try {
          deployLog('init', 'Fetching current model...')
          const cm = await apiService.getCurrentModel()
          deployLog('init', 'Current model:', cm)
          setCurrentModel(cm)
        } catch (e) {
          deployLog('init', 'Failed to get current model (non-fatal):', e)
        }

        // Load templates
        try {
          setTemplatesLoading(true)
          deployLog('init', 'Fetching templates...')
          const data = await apiService.listTemplates()
          deployLog('init', 'Loaded templates:', data)
          setTemplates(data.files)
          setSelectedTemplate(data.selected)
          setTemplatesDir(data.directory)
        } finally {
          setTemplatesLoading(false)
        }

        // Initialize API key management
        const currentApiKey = settingsManager.getApiKey()
        const storedKeys = getStoredApiKeys()
        setAvailableApiKeys(storedKeys)
        setSelectedApiKey(persistedSettings.selectedApiKey || currentApiKey || '')

        // Fetch vLLM backend status (best-effort)
        try {
          deployLog('init', 'Fetching vLLM status...')
          const vllmRes = await fetch('/api/v1/service/vllm/status')
          if (vllmRes.ok) {
            const vllmData = await vllmRes.json()
            deployLog('init', 'vLLM status:', vllmData)
            setVllmStatus(vllmData)
          }
        } catch (e) {
          deployLog('init', 'Failed to fetch vLLM status (non-fatal):', e)
        }

        try {
          const statusRes = await fetch('/api/v1/service/status')
          if (statusRes.ok) {
            const sd = await statusRes.json()
            setLlamacppMtpSupported(sd.mtp_supported === true)
            const llg =
              sd.llguidance_enabled ?? sd.llamacpp_build?.llguidance_enabled
            setLlamacppLlguidanceEnabled(
              llg === true ? true : llg === false ? false : null
            )
            setLlamacppBuildTag(sd.llamacpp_build?.tag ?? sd.llamacpp_build?.cli_version)
          }
        } catch (e) {
          deployLog('init', 'Failed to fetch llamacpp build status (non-fatal):', e)
        }

        deployLog('init', 'Initialization complete')

      } catch (e) {
        deployLog('init', 'Initialization error:', e)
        setError(e instanceof Error ? e.message : 'Failed to initialize deploy page')
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [])

  const availableModelNames = useMemo(() => {
    const names = Array.from(new Set(models.map((m) => m.name))).sort()
    deployLog('models', 'Computed available model names:', { count: names.length, names })
    return names
  }, [models])

  const availableVariantsForSelected = useMemo(() => {
    if (!config) {
      deployLog('variants', 'No config, returning empty variants')
      return [] as string[]
    }
    const variants = models.filter((m) => m.name === config.model.name).map((m) => m.variant)
    const uniqueVariants = Array.from(new Set(variants.filter(Boolean))).sort()
    deployLog('variants', `Computed variants for ${config.model.name}:`, { count: uniqueVariants.length, variants: uniqueVariants })
    return uniqueVariants
  }, [models, config?.model?.name]) // Only depend on the model name, not the entire config

  const selectedDeployModel = useMemo(() => {
    if (!config?.model?.name) return null
    return (
      models.find((m) => m.name === config.model.name && m.variant === config.model.variant) ??
      models.find((m) => m.name === config.model.name) ??
      null
    )
  }, [models, config?.model?.name, config?.model?.variant])

  const selectedModelMtpCapable = selectedDeployModel?.mtpCapable === true

  // Restore per-model MTP settings when the selected GGUF changes
  useEffect(() => {
    if (!config?.model?.name) return
    const key = modelMtpKey(config.model.name, config.model.variant || '')
    if (prevModelMtpKey.current === key) return
    prevModelMtpKey.current = key
    const resolved = resolveMtpForModel(config.model.name, config.model.variant || '', {
      mtpCapable: selectedModelMtpCapable,
    })
    setConfig((prev) => {
      if (!prev) return prev
      const base: Config = {
        ...prev,
        mtp: { ...defaultMtpSettings(), ...resolved },
      }
      const applied = applyMtpWorkloadProfile(resolved.workload_profile, base)
      return clampMtpForModelCapability(
        {
          ...base,
          mtp: applied.mtp,
          performance: { ...base.performance, ...applied.performance },
          server: { ...base.server, ...applied.server },
        },
        selectedModelMtpCapable
      )
    })
  }, [config?.model?.name, config?.model?.variant, selectedModelMtpCapable])

  // Family presets can leave mtp.enabled=true while the switch looks off (non-MTP GGUF)
  useEffect(() => {
    if (!config || selectedModelMtpCapable || !config.mtp?.enabled) return
    const next = clampMtpForModelCapability(config, false)
    setConfig(next)
    saveDeploySettings(next, selectedApiKey)
    void updateCommandPreview(next)
  }, [selectedModelMtpCapable, config?.mtp?.enabled, config?.model?.name, config?.model?.variant])

  const mtpStatsWatch =
    backend === 'llamacpp' &&
    config?.mtp?.enabled === true &&
    currentModel?.status === 'loaded'
  const { stats: mtpStats, connected: mtpStatsConnected } = useMtpStats({
    enabled: mtpStatsWatch,
    backend,
  })

  const availableVllmVariantsForSelected = useMemo(() => {
    const served = (vllmConfig?.model?.served_name || '').trim()
    if (!served) return [] as string[]
    const variants = models.filter((m) => m.name === served).map((m) => m.variant)
    return Array.from(new Set(variants.filter(Boolean))).sort()
  }, [models, vllmConfig?.model?.served_name])

  const vllmSelectModelNameValue = useMemo(() => {
    const served = (vllmConfig?.model?.served_name || '').trim()
    if (served && availableModelNames.includes(served)) return served
    return ''
  }, [availableModelNames, vllmConfig?.model?.served_name])

  const vllmSelectVariantValue = useMemo(() => {
    if (!vllmConfig) return ''
    const served = (vllmConfig.model.served_name || '').trim()
    const repo = (vllmConfig.model.name || '').trim()
    const candidates = models.filter((m) => m.name === served)
    if (candidates.length === 0) return ''
    if (repo) {
      const hit = candidates.find((m) => (m.repositoryId || '').trim() === repo)
      if (hit) return hit.variant
    }
    if (availableVllmVariantsForSelected.length === 1) return availableVllmVariantsForSelected[0]
    return candidates[0].variant
  }, [models, vllmConfig, availableVllmVariantsForSelected])

  const hasChanges = useMemo(() => {
    if (backend === 'vllm') {
      if (!vllmConfig || !originalVllmConfig) return false
      return JSON.stringify(vllmConfig) !== JSON.stringify(originalVllmConfig)
    }
    return JSON.stringify(config) !== JSON.stringify(originalConfig)
  }, [backend, config, originalConfig, vllmConfig, originalVllmConfig])

  // Function to update command line preview in real-time
  const updateCommandPreview = async (configToPreview: Config) => {
    deployLog('commandPreview', 'Updating command preview', { modelName: configToPreview?.model?.name, backend })
    try {
      // Convert undefined values to null so they get properly serialized and handled by backend
      const configForPreview = prepareConfigForBackend(configToPreview, selectedModelMtpCapable)

      // Send the config to backend to get command preview without saving
      deployLog('commandPreview', 'Sending preview request to backend')
      const response = await fetch('/api/v1/service/config/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: configForPreview, backend: 'llamacpp' })
      })
      deployLog('commandPreview', 'Preview response:', { ok: response.ok, status: response.status })
      if (response.ok) {
        const data = await response.json()
        deployLog('commandPreview', 'Command preview updated:', { command: data.command?.substring(0, 100) + '...' })
        setCommandLine(data.command || '')
      } else {
        const errorText = await response.text()
        deployLog('commandPreview', 'Preview request failed:', { status: response.status, error: errorText })
      }
    } catch (error) {
      // If preview fails, keep the old command line
      deployLog('commandPreview', 'Preview request error:', error)
    }
  }

  // vLLM command preview updater
  const updateCommandPreviewVllm = async (configToPreview: VllmConfig) => {
    deployLog('commandPreview', 'Updating vLLM command preview')
    try {
      const configForPreview = JSON.parse(JSON.stringify(configToPreview))
      const response = await fetch('/api/v1/service/config/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: configForPreview, backend: 'vllm' })
      })
      if (response.ok) {
        const data = await response.json()
        setCommandLine(data.command || '')
      }
    } catch (error) {
      deployLog('commandPreview', 'vLLM preview error:', error)
    }
  }

  const updateConfig = (path: string, value: any) => {
    debugLog('deploy', 'updateConfig called', { path, value })
    deployLog('updateConfig', `Updating config path: ${path}`, { value, currentConfig: config })
    if (!config) {
      deployLog('updateConfig', 'ABORT: config is null')
      return
    }
    const next: Config = JSON.parse(JSON.stringify(config))
    const keys = path.split('.')
    let ref: any = next
    for (let i = 0; i < keys.length - 1; i++) ref = ref[keys[i]]
    const oldValue = ref[keys[keys.length - 1]]
    ref[keys[keys.length - 1]] = value
    debugLog('deploy', 'Config value changed', { path, oldValue, newValue: value })
    deployLog('updateConfig', `Config updated: ${path}`, { oldValue, newValue: value })
    debugLog('deploy', 'Calling setConfig with new config', next)
    setConfig(next)

    if (path.startsWith('mtp.') && next.model?.name) {
      saveMtpForModel(next.model.name, next.model.variant || '', next.mtp ?? {})
    }

    // Save to localStorage for persistence
    deployLog('updateConfig', 'Saving to localStorage')
    saveDeploySettings(next, selectedApiKey)

    // Update command line preview in real-time
    deployLog('updateConfig', 'Updating command preview')
    updateCommandPreview(next)
  }

  const switchMtpToCustomProfile = () => {
    if (config?.mtp?.workload_profile === 'custom') return
    updateConfig('mtp.workload_profile', 'custom')
  }

  const handleMtpWorkloadProfileChange = (profile: MtpWorkloadProfile) => {
    if (!config) return
    const next: Config = clampMtpForModelCapability(
      (() => {
        const draft: Config = JSON.parse(JSON.stringify(config))
        const applied = applyMtpWorkloadProfile(profile, draft)
        draft.mtp = applied.mtp
        draft.performance = { ...draft.performance, ...applied.performance }
        draft.server = { ...draft.server, ...applied.server }
        return draft
      })(),
      selectedModelMtpCapable
    )
    setConfig(next)
    if (next.model?.name) {
      saveMtpForModel(next.model.name, next.model.variant || '', next.mtp ?? {})
    }
    saveDeploySettings(next, selectedApiKey)
    updateCommandPreview(next)
  }

  const handleMtpEnabledChange = (enabled: boolean) => {
    if (!config) return
    const next: Config = JSON.parse(JSON.stringify(config))
    next.mtp = {
      ...defaultMtpSettings(),
      ...next.mtp,
      workload_profile: 'custom',
      enabled,
    }
    setConfig(next)
    if (next.model?.name) {
      saveMtpForModel(next.model.name, next.model.variant || '', next.mtp ?? {})
    }
    saveDeploySettings(next, selectedApiKey)
    updateCommandPreview(next)
  }

  // vLLM config updater - works on VllmConfig instead of Config
  const updateVllmConfig = (path: string, value: any) => {
    deployLog('updateVllmConfig', `Updating vLLM config path: ${path}`, { value })
    if (!vllmConfig) {
      deployLog('updateVllmConfig', 'ABORT: vllmConfig is null')
      return
    }
    const next: VllmConfig = JSON.parse(JSON.stringify(vllmConfig))
    const keys = path.split('.')
    let ref: any = next
    for (let i = 0; i < keys.length - 1; i++) ref = ref[keys[i]]
    ref[keys[keys.length - 1]] = value
    setVllmConfig(next)
    writeDeployStorage({ vllmConfig: next })

    // Update command line preview
    updateCommandPreviewVllm(next)
  }

  const applyVllmModelFromCatalog = useCallback(
    (displayName: string, variant: string) => {
      deployLog('vllmCatalog', 'Apply catalog model', { displayName, variant })
      if (!vllmConfig) return
      const m = models.find((x) => x.name === displayName && x.variant === variant)
      const repoId = (m?.repositoryId && String(m.repositoryId).trim()) || ''
      const next = JSON.parse(JSON.stringify(vllmConfig)) as VllmConfig
      next.model.name = repoId
      next.model.served_name = displayName
      setVllmConfig(next)
      writeDeployStorage({ vllmConfig: next })
      updateCommandPreviewVllm(next)
    },
    [vllmConfig, models]
  )

  const resetToDefault = (path: string) => {
    // null → backend merge clears the key; param omitted from llama-server command
    updateConfig(path, null)
  }

  const getDefaultValue = (path: string) => {
    const keys = path.split('.')
    let defaultValue: any = DEFAULT_VALUES
    for (const key of keys) {
      defaultValue = defaultValue[key]
    }
    return defaultValue
  }

  // API Key management functions
  const handleApiKeyChange = (apiKey: string) => {
    setSelectedApiKey(apiKey)
    if (config) {
      saveDeploySettings(config, apiKey)
    }
    // Update the global settings manager
    settingsManager.setApiKey(apiKey)
  }

  const handleAddApiKey = () => {
    if (!newApiKey.trim()) return
    const trimmedKey = newApiKey.trim()
    if (availableApiKeys.includes(trimmedKey)) {
      setError('API key already exists')
      return
    }
    const updatedKeys = [...availableApiKeys, trimmedKey]
    setAvailableApiKeys(updatedKeys)
    saveApiKeys(updatedKeys)
    setSelectedApiKey(trimmedKey)
    setNewApiKey('')
    handleApiKeyChange(trimmedKey)
    setSuccess('API key added successfully')
  }

  const handleRemoveApiKey = (keyToRemove: string) => {
    const updatedKeys = availableApiKeys.filter(key => key !== keyToRemove)
    setAvailableApiKeys(updatedKeys)
    saveApiKeys(updatedKeys)
    if (selectedApiKey === keyToRemove) {
      const newSelected = updatedKeys.length > 0 ? updatedKeys[0] : ''
      setSelectedApiKey(newSelected)
      handleApiKeyChange(newSelected)
    }
    setSuccess('API key removed successfully')
  }

  const handleValidate = async () => {
    deployLog('validate', 'Starting validation', { modelName: config?.model?.name })
    if (backend === 'vllm') {
      setError(null)
      setValidateErrors(null)
      setValidateWarnings(null)
      setSuccess('Validation only applies to llama.cpp. Switch the backend toggle to llama.cpp to run it.')
      return
    }
    if (!config) {
      deployLog('validate', 'ABORT: config is null')
      return
    }
    try {
      setValidating(true)
      setValidateErrors(null)
      setValidateWarnings(null)

      // Convert undefined values to null for proper backend handling
      const configForValidation = JSON.parse(JSON.stringify(config, (_key, value) => {
        return value === undefined ? null : value
      }))

      deployLog('validate', 'Sending validation request')
      const data = await apiService.validateServiceConfig(configForValidation as any)
      deployLog('validate', 'Validation result:', data)
      setValidateErrors(data.errors || null)
      setValidateWarnings(data.warnings || null)
      if (data.valid) setSuccess('Configuration is valid')
    } catch (e) {
      deployLog('validate', 'Validation error:', e)
      setError(e instanceof Error ? e.message : 'Validation failed')
    } finally {
      setValidating(false)
    }
  }

  const handleSave = async () => {
    deployLog('save', 'Starting save', { backend })
    if (backend === 'vllm') {
      if (!vllmConfig) {
        deployLog('save', 'ABORT: vllmConfig is null')
        return
      }
      try {
        setSaving(true)
        const body = JSON.parse(JSON.stringify(vllmConfig))
        const res = await fetch('/api/v1/service/config?backend=vllm', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!res.ok) {
          const t = await res.text()
          throw new Error(t || res.statusText)
        }
        const cfgRes = await fetch('/api/v1/service/config?backend=vllm')
        const cfgJson = await cfgRes.json()
        setCommandLine(cfgJson.command || '')
        setOriginalVllmConfig(JSON.parse(JSON.stringify(vllmConfig)))
        setSuccess('vLLM configuration saved')
      } catch (e) {
        deployLog('save', 'Save error:', e)
        setError(e instanceof Error ? e.message : 'Failed to save configuration')
      } finally {
        setSaving(false)
      }
      return
    }

    deployLog('save', 'Starting save', { modelName: config?.model?.name })
    if (!config) {
      deployLog('save', 'ABORT: config is null')
      return
    }
    try {
      setSaving(true)

      // Convert undefined values to null for proper backend handling
      const configForSave = prepareConfigForBackend(config, selectedModelMtpCapable)

      deployLog('save', 'Sending save request', { model: configForSave.model })
      const data = await (async () => {
        const updated = await apiService.updateServiceConfig({ config: configForSave as any })
        deployLog('save', 'Config saved, fetching updated command')
        // Re-query command line preview from backend since apiService doesn't return it
        const cfgRes = await fetch(`/api/v1/service/config?backend=${backend}`)
        const cfgJson = await cfgRes.json()
        return { command: cfgJson.command, updated }
      })()
      deployLog('save', 'Save complete', { commandLength: data.command?.length })
      setOriginalConfig(JSON.parse(JSON.stringify(config)))
      setCommandLine(data.command || '')
      setSuccess('Configuration saved')
    } catch (e) {
      deployLog('save', 'Save error:', e)
      setError(e instanceof Error ? e.message : 'Failed to save configuration')
    } finally {
      setSaving(false)
    }
  }

  const handleResetConfig = () => {
    deployLog('reset', 'Clear All button clicked')
    if (backend === 'vllm') {
      const next = JSON.parse(JSON.stringify(VLLM_DEFAULT_VALUES)) as VllmConfig
      setVllmConfig(next)
      writeDeployStorage({ vllmConfig: next })
      updateCommandPreviewVllm(next)
      return
    }
    if (!config) {
      deployLog('reset', 'ABORT: config is null')
      return
    }

    const sections = ['model', 'sampling', 'performance', 'speculative', 'context_extension', 'server']
    const resetConfig = JSON.parse(JSON.stringify(config))

    sections.forEach((section) => {
      const sectionDefaults = DEFAULT_VALUES[section as keyof typeof DEFAULT_VALUES]
      if (sectionDefaults && resetConfig[section as keyof Config]) {
        Object.keys(sectionDefaults).forEach((key) => {
          // null means the backend merge clears optional params, so omitted fields use server defaults.
          ;(resetConfig[section as keyof Config] as any)[key] = null
        })
      }
    })

    deployLog('reset', 'Config reset to defaults', { modelName: resetConfig.model?.name })
    setConfig(resetConfig)
    setSelectedDeployPreset(null)
    setSelectedPreset(null)
    saveDeploySettings(resetConfig, selectedApiKey)
    updateCommandPreview(resetConfig)
  }

  const runAction = async (action: 'start' | 'stop' | 'restart') => {
    deployLog('action', `Running action: ${action}`, { backend, hasConfig: !!config, hasVllm: !!vllmConfig })
    if (backend === 'vllm') {
      if (!vllmConfig && action !== 'stop') {
        deployLog('action', 'ABORT: vllmConfig is null and action is not stop')
        return
      }
    } else if (!config && action !== 'stop') {
      deployLog('action', 'ABORT: config is null and action is not stop')
      return
    }
    try {
      setActionLoading(action)

      // Clear logs when restarting
      if (action === 'restart' && logViewerRef.current) {
        deployLog('action', 'Clearing logs for restart')
        logViewerRef.current.clearLogs()
      }

      let configForAction: any = null
      if (backend === 'vllm' && vllmConfig) {
        configForAction = JSON.parse(JSON.stringify(vllmConfig))
      } else if (config) {
        configForAction = prepareConfigForBackend(config, selectedModelMtpCapable)
      }

      // Ensure the current template selection is included in the config
      // The template dropdown updates selectedTemplate state separately from config,
      // so we must inject it here to avoid the stale default overwriting the user's choice
      if (configForAction && backend === 'llamacpp') {
        if (!configForAction.template) {
          configForAction.template = { directory: templatesDir || '/home/llamacpp/templates', selected: selectedTemplate }
        } else {
          configForAction.template.selected = selectedTemplate
        }

        // Explicitly null out optional model fields when not set, so the backend
        // deep_merge clears any stale persisted values (e.g. mmproj from a previous VL model)
        if (configForAction.model) {
          if (!configForAction.model.mmproj) configForAction.model.mmproj = null
          if (!configForAction.model.lora) configForAction.model.lora = null
          if (!configForAction.model.lora_base) configForAction.model.lora_base = null
        }

        if (selectedTemplate === TEMPLATE_EMBEDDED_JINJA) {
          configForAction.server = { ...configForAction.server, jinja: true }
        }

        deployLog('action', 'Injected template selection into config', { selectedTemplate })
      }

      deployLog('action', 'Sending action to backend', { action, backend, config: configForAction?.model })
      await apiService.performServiceAction(action === 'stop' ? { action, backend } : { action, backend, config: configForAction as any })
      deployLog('action', 'Action completed successfully')

      if (backend === 'vllm') await refreshVllmStatus()
      try {
        deployLog('action', 'Refreshing current model info')
        const res = await fetch(`/api/v1/models/current`)
        if (res.ok) {
          const current = await res.json()
          deployLog('action', 'Current model updated:', current)
          setCurrentModel(current)
        }
      } catch (e) {
        deployLog('action', 'Failed to refresh current model (non-fatal):', e)
      }
      if (action === 'start' || action === 'restart') {
        setLastLaunchContext(buildLaunchContext(action))
      } else {
        setLastLaunchContext(null)
      }
      setSuccess(`Service ${action}ed successfully`)
    } catch (e) {
      deployLog('action', 'Action failed:', e)
      setError(formatServiceActionError(e))
    } finally {
      setActionLoading(null)
    }
  }

  const vllmFieldMeta = (section: string, field: string): FieldMeta | undefined =>
    fieldMetadata[section]?.[field]

  const vllmHintLines = (section: string, field: string): string[] => {
    const m = vllmFieldMeta(section, field)
    if (!m) return []
    const lines: string[] = []
    if (m.description) lines.push(m.description)
    if (m.mapping_note) lines.push(`Mapping: ${m.mapping_note}`)
    if (m.note) lines.push(m.note)
    if (m.vllm_flag) lines.push(`CLI / env: ${m.vllm_flag}`)
    if (m.llamacpp_equivalent) lines.push(`llama.cpp equivalent: ${m.llamacpp_equivalent}`)
    return lines
  }

  const vllmCatalogModelPickers = () => {
    if (!vllmConfig) return null
    const selectStyles = {
      fontSize: '0.875rem',
      '& .MuiOutlinedInput-root': {
        borderRadius: '8px',
        backgroundColor: 'background.default',
      },
    }
    const anchorForVariant = vllmSelectModelNameValue || (vllmConfig.model.served_name || '').trim()
    return (
      <>
        <Grid item xs={12} md={6}>
          <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Model Name</Typography>
            <Select
              fullWidth
              value={vllmSelectModelNameValue}
              displayEmpty
              MenuProps={deployMenuProps}
              renderValue={(selected) => {
              if (!selected) {
                return <em>Select a model...</em>
              }
              return selected
            }}
            onChange={(e) => {
              const newName = e.target.value as string
              const newModelVariants = models.filter((m) => m.name === newName).map((m) => m.variant)
              const uniqueNewVariants = Array.from(new Set(newModelVariants.filter(Boolean)))
              const nextVariant = uniqueNewVariants[0] || 'unknown'
              deployLog('vllmCatalog', `Model name -> ${newName}, variant ${nextVariant}`, {
                availableVariants: uniqueNewVariants,
              })
              applyVllmModelFromCatalog(newName, nextVariant)
            }}
            sx={selectStyles}
          >
            {availableModelNames.map((name) => (
              <MenuItem key={name} value={name}>{name}</MenuItem>
            ))}
          </Select>
        </Grid>
        {availableVllmVariantsForSelected.length > 1 && (
          <Grid item xs={12} md={6}>
            <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Model Variant</Typography>
              <Select
                fullWidth
                value={vllmSelectVariantValue}
                MenuProps={deployMenuProps}
                onChange={(e) => {
                const v = e.target.value as string
                const anchor = anchorForVariant
                if (anchor) applyVllmModelFromCatalog(anchor, v)
              }}
              sx={selectStyles}
            >
              {availableVllmVariantsForSelected.map((variant) => (
                <MenuItem key={variant} value={variant}>{variant}</MenuItem>
              ))}
            </Select>
          </Grid>
        )}
        {availableVllmVariantsForSelected.length <= 1 && vllmSelectModelNameValue && (
          <Grid item xs={12} md={6}>
            <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Model Variant</Typography>
            <Typography
              sx={{
                fontSize: '0.875rem',
                color: 'text.secondary',
                fontStyle: 'italic',
                py: 1.5,
                px: 1,
                backgroundColor: 'action.hover',
                borderRadius: '8px',
                border: '1px solid',
                borderColor: 'divider',
              }}
            >
              {vllmSelectVariantValue || '—'} (only variant available)
            </Typography>
          </Grid>
        )}
        <Grid item xs={12} md={6}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
            <Typography sx={{ fontSize: '0.875rem' }}>Hugging Face repo id</Typography>
            {vllmFieldMeta('model', 'name') && <FrameworkBadge scope={vllmFieldMeta('model', 'name')!.scope} />}
          </Box>
          <TextField
            fullWidth
            size="small"
            value={vllmConfig.model.name}
            placeholder="org/model"
            onChange={(e) => updateVllmConfig('model.name', e.target.value)}
          />
          <FormHelperText sx={{ fontSize: '0.75rem' }}>
            {[vllmHintLines('model', 'name').join(' — '), 'Auto-filled when the catalog entry has repository metadata.'].filter(Boolean).join(' ')}
          </FormHelperText>
        </Grid>
        <Grid item xs={12} md={6}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
            <Typography sx={{ fontSize: '0.875rem' }}>Served model name</Typography>
            {vllmFieldMeta('model', 'served_name') && <FrameworkBadge scope={vllmFieldMeta('model', 'served_name')!.scope} />}
          </Box>
          <TextField
            fullWidth
            size="small"
            value={vllmConfig.model.served_name}
            onChange={(e) => updateVllmConfig('model.served_name', e.target.value)}
          />
          <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('model', 'served_name').join(' — ')}</FormHelperText>
        </Grid>
      </>
    )
  }

  const deployTabMax = backend === 'llamacpp' ? DEPLOY_TAB_MAX_LLAMACPP : DEPLOY_TAB_MAX_VLLM
  const activeDeployTab = Math.min(Math.max(0, tab), deployTabMax)
  const deployTabOptions = backend === 'llamacpp'
    ? ['Model', 'Sampling', 'Performance', 'Context', 'Server', 'Version', 'Command']
    : ['Model', 'Sampling', 'Performance', 'MoE', 'Tools', 'Env', 'Version', 'Server', 'Command']

  const selectDeployTab = useCallback((idx: number) => {
    const max = backend === 'llamacpp' ? DEPLOY_TAB_MAX_LLAMACPP : DEPLOY_TAB_MAX_VLLM
    setTab(Math.min(Math.max(0, idx), max))
  }, [backend])

  if (loading) {
    deployLog('render', 'Rendering loading state')
    return (
      <Box
        sx={{
          p: { xs: 2, sm: 3, md: 4 },
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          minHeight: 400
        }}
      >
        <CircularProgress />
      </Box>
    )
  }

  if (!config) {
    deployLog('render', 'Rendering error state: config is null')
    return (
      <Box sx={{ p: { xs: 2, sm: 3, md: 4 } }}>
        <Alert
          severity="error"
          sx={{
            borderRadius: '8px',
            border: '1px solid',
            borderColor: 'error.main'
          }}
        >
          Failed to load configuration.
        </Alert>
      </Box>
    )
  }

  const backendLabel = backend === 'vllm' ? 'vLLM' : 'llama.cpp'
  const endpointHost = backend === 'vllm' ? vllmConfig?.server.host : config.server.host
  const endpointPort = backend === 'vllm' ? vllmConfig?.server.port : config.server.port
  const endpointDisplay = endpointHost && endpointPort
    ? `${endpointHost === '0.0.0.0' ? 'localhost' : endpointHost}:${endpointPort}`
    : 'Not configured'
  const deploymentRunning = backend === 'vllm'
    ? vllmStatus?.running === true
    : ['active', 'loaded', 'running'].includes(String(currentModel?.status ?? '').toLowerCase())
  const deploymentStatus = backend === 'vllm'
    ? vllmStatus
      ? vllmStatus.running ? 'Running' : 'Stopped'
      : 'Status unknown'
    : currentModel?.status
      ? String(currentModel.status)
      : 'Status unknown'
  const deploymentStatusKey = deploymentRunning ? 'running' : 'stopped'
  const modelCapabilityStatus = backend === 'vllm' ? 'vllm' : config?.model?.mmproj ? 'vision' : 'text'
  const modelCapabilityLabel = backend === 'vllm' ? 'vLLM (HF weights)' : config?.model?.mmproj ? 'Vision ready' : 'Text only'
  const modelCapabilityIcon = backend === 'vllm'
    ? <DeployIcon sx={{ fontSize: 14 }} />
    : config?.model?.mmproj
      ? <VisionIcon sx={{ fontSize: 14 }} />
      : <VisionOffIcon sx={{ fontSize: 14 }} />
  const selectedRuntimeModel = backend === 'vllm'
    ? vllmConfig?.model.served_name || vllmConfig?.model.name || 'No model selected'
    : [config.model.name, config.model.variant].filter(Boolean).join(' / ') || 'No model selected'
  const vramSummary = vramLoading
    ? 'Estimating'
    : vramEstimate
      ? `${vramEstimate.total_gb.toFixed(1)} GB ${vramEstimate.fits_in_vram ? 'fits' : 'needed'}`
      : 'Not estimated'
  const vramStatusKey = vramEstimate?.fits_in_vram ? 'vram-fit' : 'vram-exceeds'
  const vramAccentColor = vramEstimate?.fits_in_vram ? accentColors.success : accentColors.rose
  const vramUtilizationPercent = vramEstimate
    ? Math.min(Math.max(vramEstimate.utilization_percent, 0), 100)
    : 0
  const vramUtilizationColor = vramEstimate
    ? vramEstimate.utilization_percent > 95
      ? accentColors.rose
      : vramEstimate.utilization_percent > 85
        ? accentColors.warning
        : accentColors.success
    : accentColors.slate
  const commandIconButtonSx = {
    width: 30,
    height: 30,
    borderRadius: '8px',
    border: '1px solid',
    borderColor: 'divider',
    bgcolor: 'action.hover',
    color: 'text.secondary',
    '&:hover': { bgcolor: 'action.selected' },
  } as const
  const activeLaunchContext: DeploymentLaunchContext | null = lastLaunchContext ?? (deploymentRunning ? {
    action: 'start',
    backend,
    modelName: selectedRuntimeModel,
    endpoint: endpointDisplay,
  } : null)
  const deploymentNextActions = activeLaunchContext
    ? [
        {
          label: 'Chat',
          icon: <ChatIcon />,
          onClick: () => navigate(chatPath, { state: { deployment: activeLaunchContext } }),
        },
        {
          label: 'Benchmark',
          icon: <SpeedIcon />,
          onClick: () => navigate(benchmarkPath, {
            state: {
              deployment: activeLaunchContext,
              model: modelInfoFromDeployment(activeLaunchContext),
            },
          }),
        },
        {
          label: 'Monitor',
          icon: <MonitoringIcon />,
          onClick: () => navigate(monitoringPath, { state: { deployment: activeLaunchContext } }),
        },
      ]
    : []

  deployLog('render', 'Rendering main content', {
    modelName: config.model?.name,
    modelCount: models.length,
    hasCurrentModel: !!currentModel
  })

  return (
    <PageShell sx={{ py: { xs: 0.75, md: 1 } }}>
      <Box
        ref={deployTabsRef}
        sx={{
          mb: 1,
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1fr) auto' },
          columnGap: 0.75,
          rowGap: 0.45,
          alignItems: 'center',
        }}
      >
        <Box sx={{ minWidth: 0, display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
          <Typography
            variant="body2"
            title={selectedRuntimeModel}
            sx={{
              flex: { xs: '1 1 150px', md: '0 1 420px' },
              minWidth: 0,
              maxWidth: { xs: 210, lg: 420 },
              fontWeight: 800,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {selectedRuntimeModel}
          </Typography>
          <StatusChip
            status={deploymentStatusKey}
            label={deploymentStatus}
            height={22}
            statusConfig={DEPLOY_STATUS_CONFIG}
          />
          <StatusChip
            status={modelCapabilityStatus}
            label={modelCapabilityLabel}
            icon={modelCapabilityIcon}
            height={22}
            statusConfig={DEPLOY_STATUS_CONFIG}
            sx={{ display: { xs: 'none', sm: 'inline-flex' } }}
          />
          {hasChanges && (
            <StatusChip
              status="changed"
              label="Unsaved"
              height={22}
              statusConfig={DEPLOY_STATUS_CONFIG}
            />
          )}
          {backend === 'vllm' && vllmStatus?.last_action_error ? (
            <Tooltip title={vllmStatus.last_action_error}>
              <StatusChip
                status="warning"
                label="Warning"
                height={22}
                statusConfig={DEPLOY_STATUS_CONFIG}
              />
            </Tooltip>
          ) : null}
          <Typography
            variant="caption"
            color="text.secondary"
            title={endpointDisplay}
            sx={{
              display: { xs: 'none', md: 'block' },
              overflowWrap: 'anywhere',
            }}
          >
            {backendLabel} / {endpointDisplay}
          </Typography>
          {vramEstimate && (
            <StatusChip
              status={vramStatusKey}
              label={vramSummary}
              icon={<MemoryIcon sx={{ fontSize: 14 }} />}
              height={22}
              statusConfig={DEPLOY_STATUS_CONFIG}
            />
          )}
        </Box>

        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 0.5,
            justifyContent: { xs: 'flex-start', lg: 'flex-end' },
            alignItems: 'center',
          }}
        >
            <Tooltip title="Configuration section">
              <Select
                aria-label="Configuration section"
                size="small"
                value={activeDeployTab}
                MenuProps={deployMenuProps}
                onChange={(event) => selectDeployTab(Number(event.target.value))}
              sx={deploySectionSelectSx}
            >
              {deployTabOptions.map((label, index) => (
                <MenuItem key={label} value={index}>
                  {label}
                </MenuItem>
              ))}
            </Select>
          </Tooltip>
          <ToggleButtonGroup
            value={backend}
            exclusive
            size="small"
            onChange={(_, value) => {
              if (value) {
                setBackend(value)
                deployLog('backend', `Switched backend to ${value}`)
              }
            }}
            sx={{
              minHeight: 30,
              flexShrink: 0,
              '& .MuiToggleButtonGroup-grouped': {
                minHeight: 30,
                px: 0.9,
                borderColor: 'divider',
              },
              '& .MuiToggleButton-root': {
                textTransform: 'none',
                fontSize: '0.75rem',
                fontWeight: 700,
                color: 'text.secondary',
                '&.Mui-selected': {
                  bgcolor: alpha(accentColors.primary, 0.18),
                  color: accentColors.primary,
                },
              },
            }}
          >
            <ToggleButton value="llamacpp">llama.cpp</ToggleButton>
            <ToggleButton value="vllm">vLLM</ToggleButton>
          </ToggleButtonGroup>
          <Tooltip title="Start runtime">
            <span>
              <IconButton
                aria-label="Start runtime"
                size="small"
                onClick={() => runAction('start')}
                disabled={actionLoading !== null}
                sx={{
                  ...commandIconButtonSx,
                  bgcolor: alpha(accentColors.success, 0.16),
                  color: accentColors.success,
                  borderColor: alpha(accentColors.success, 0.34),
                }}
              >
                {actionLoading === 'start' ? <CircularProgress size={15} color="inherit" /> : <StartIcon sx={{ fontSize: 18 }} />}
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Stop runtime">
            <span>
              <IconButton
                aria-label="Stop runtime"
                size="small"
                onClick={() => runAction('stop')}
                disabled={actionLoading !== null}
                sx={{
                  ...commandIconButtonSx,
                  display: { xs: 'none', sm: 'inline-flex' },
                  color: accentColors.rose,
                  borderColor: alpha(accentColors.rose, 0.3),
                }}
              >
                {actionLoading === 'stop' ? <CircularProgress size={15} color="inherit" /> : <StopIcon sx={{ fontSize: 18 }} />}
              </IconButton>
            </span>
          </Tooltip>
          {(hasChanges || saving) && (
            <Tooltip title={saving ? 'Saving configuration' : 'Save configuration'}>
              <span>
                <IconButton
                  aria-label="Save configuration"
                  size="small"
                  onClick={handleSave}
                  disabled={saving}
                  sx={{
                    ...commandIconButtonSx,
                    bgcolor: accentColors.primary,
                    color: '#fff',
                    borderColor: accentColors.primary,
                  }}
                >
                  {saving ? <CircularProgress size={15} color="inherit" /> : <SaveIcon sx={{ fontSize: 18 }} />}
                </IconButton>
              </span>
            </Tooltip>
          )}
          <Tooltip title="More runtime actions">
            <IconButton
              aria-label="More runtime actions"
              size="small"
              onClick={(event) => setRuntimeMenuAnchorEl(event.currentTarget)}
              sx={commandIconButtonSx}
            >
              <MoreIcon sx={{ fontSize: 18 }} />
            </IconButton>
          </Tooltip>
          <Menu
            anchorEl={runtimeMenuAnchorEl}
            open={Boolean(runtimeMenuAnchorEl)}
            onClose={() => setRuntimeMenuAnchorEl(null)}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
            slotProps={{
              paper: {
                sx: {
                  mt: 0.5,
                  minWidth: 190,
                  border: '1px solid',
                  borderColor: 'divider',
                  bgcolor: 'background.paper',
                },
              },
            }}
          >
            {deploymentNextActions.map((action) => (
              <MenuItem
                key={action.label}
                onClick={() => {
                  setRuntimeMenuAnchorEl(null)
                  action.onClick()
                }}
                sx={{ gap: 1, fontSize: '0.8125rem' }}
              >
                {React.isValidElement(action.icon)
                  ? React.cloneElement(action.icon as React.ReactElement<{ sx?: any }>, { sx: { fontSize: 18, color: accentColors.info } })
                  : action.icon}
                {action.label}
              </MenuItem>
            ))}
            {isNarrow && (
              <MenuItem
                disabled={actionLoading !== null}
                onClick={() => {
                  setRuntimeMenuAnchorEl(null)
                  void runAction('stop')
                }}
                sx={{ gap: 1, fontSize: '0.8125rem' }}
              >
                {actionLoading === 'stop' ? <CircularProgress size={15} color="inherit" /> : <StopIcon sx={{ fontSize: 18, color: accentColors.rose }} />}
                Stop runtime
              </MenuItem>
            )}
            <MenuItem
              disabled={actionLoading !== null}
              onClick={() => {
                setRuntimeMenuAnchorEl(null)
                void runAction('restart')
              }}
              sx={{ gap: 1, fontSize: '0.8125rem' }}
            >
              {actionLoading === 'restart' ? <CircularProgress size={15} color="inherit" /> : <RestartIcon sx={{ fontSize: 18, color: accentColors.warning }} />}
              Restart runtime
            </MenuItem>
            <MenuItem
              disabled={validating}
              onClick={() => {
                setRuntimeMenuAnchorEl(null)
                void handleValidate()
              }}
              sx={{ gap: 1, fontSize: '0.8125rem' }}
            >
              {validating ? <CircularProgress size={15} color="inherit" /> : <ValidateIcon sx={{ fontSize: 18, color: accentColors.success }} />}
              Validate configuration
            </MenuItem>
            <MenuItem
              onClick={() => {
                setRuntimeMenuAnchorEl(null)
                handleResetConfig()
              }}
              sx={{ gap: 1, fontSize: '0.8125rem' }}
            >
              <ResetIcon sx={{ fontSize: 18 }} />
              Reset configuration
            </MenuItem>
          </Menu>
        </Box>
      </Box>

      {error && (
        <Alert
          severity="error"
          sx={{
            mb: 2,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: 'error.main'
          }}
          onClose={() => setError(null)}
        >
          {error}
        </Alert>
      )}
      {validateWarnings && validateWarnings.length > 0 && (
        <Alert
          severity="warning"
          sx={{
            mb: 2,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: 'warning.main'
          }}
        >
          {validateWarnings.map((w, i) => (
            <Box key={i} sx={{ fontSize: '0.8125rem' }}>{w}</Box>
          ))}
        </Alert>
      )}
      {validateErrors && validateErrors.length > 0 && (
        <Alert
          severity="error"
          sx={{
            mb: 2,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: 'error.main'
          }}
        >
          {validateErrors.map((e, i) => (
            <Box key={i} sx={{ fontSize: '0.8125rem' }}>{e}</Box>
          ))}
        </Alert>
      )}

      {/* Model selection */}
      {backend === 'llamacpp' && activeDeployTab === 0 && (
      <Box
        data-deploy-model-overview
        sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            lg: vramEstimate ? 'minmax(0, 1.05fr) minmax(360px, 0.95fr)' : '1fr',
          },
          gap: 1,
          mb: 1,
          alignItems: 'stretch',
        }}
      >
      <SectionCard
        fullHeight
        title="Model & runtime"
        icon={<LibraryIcon />}
        accentColor={accentColors.purple}
        sx={{ height: '100%' }}
        action={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <StatusChip
              status={modelCapabilityStatus}
              label={modelCapabilityLabel}
              icon={modelCapabilityIcon}
              statusConfig={DEPLOY_STATUS_CONFIG}
            />
            <Tooltip title={showAdvancedModelSettings ? 'Hide advanced model settings' : 'Show advanced model settings'}>
              <IconButton
                aria-label={showAdvancedModelSettings ? 'Hide advanced model settings' : 'Show advanced model settings'}
                size="small"
                onClick={() => setShowAdvancedModelSettings((value) => !value)}
                sx={{
                  ...commandIconButtonSx,
                  width: 28,
                  height: 28,
                  color: showAdvancedModelSettings ? accentColors.primary : 'text.secondary',
                }}
              >
                <TuneIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </Tooltip>
          </Box>
        }
      >
          <Grid container spacing={1.25}>
            <Grid item xs={12} md={3} sx={{ display: { xs: 'none', md: 'block' } }}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Use case</Typography>
              <Select
                fullWidth
                size="small"
                  value={selectedDeployPreset ?? ''}
                  displayEmpty
                  MenuProps={deployMenuProps}
                  renderValue={(selected) => {
                  const preset = LLAMACPP_DEPLOY_PRESETS.find((item) => item.id === selected)
                  return preset?.name ?? 'Custom settings'
                }}
                onChange={(e) => {
                  const preset = LLAMACPP_DEPLOY_PRESETS.find((item) => item.id === e.target.value)
                  if (preset) {
                    applyDeployPreset(preset)
                  } else {
                    setSelectedDeployPreset(null)
                  }
                }}
                sx={deploySelectSx}
              >
                <MenuItem value="">
                  <em>Custom settings</em>
                </MenuItem>
                {LLAMACPP_DEPLOY_PRESETS.map((preset) => (
                  <MenuItem key={preset.id} value={preset.id}>
                    {preset.name}
                  </MenuItem>
                ))}
              </Select>
            </Grid>
            <Grid item xs={12} md={5}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Model Name</Typography>
              <Select
                fullWidth
                size="small"
                  value={config?.model?.name ?? ''}
                  displayEmpty
                  MenuProps={deployMenuProps}
                  renderValue={(selected) => {
                  if (!selected) {
                    return <em>Select a model...</em>
                  }
                  return selected
                }}
                onChange={(e) => {
                  debugLog('deploy', 'Model selection changed', e.target.value)
                  const newName = e.target.value as string

                  // Calculate variants for the NEW model name
                  const newModelVariants = models.filter((m) => m.name === newName).map((m) => m.variant)
                  const uniqueNewVariants = Array.from(new Set(newModelVariants.filter(Boolean)))

                  // Always set the first available variant for the new model
                  const nextVariant = uniqueNewVariants[0] || 'Q4_K_M'

                  deployLog('modelSelect', `Model changed to ${newName} with variant ${nextVariant}`, {
                    availableVariants: uniqueNewVariants
                  })

                  // Update both name and variant atomically to avoid race conditions
                  if (!config) {
                    deployLog('modelSelect', 'ABORT: config is null')
                    return;
                  }
                  // Atomic update: set both name and variant in one setConfig call
                  const nextConfig = { ...config }
                  nextConfig.model = { ...nextConfig.model }
                  nextConfig.model.name = newName
                  nextConfig.model.variant = nextVariant

                  // Auto-match mmproj file for the new model
                  const matchedMmproj = autoMatchMmproj(newName, mmprojFiles)
                  if (matchedMmproj) {
                    nextConfig.model.mmproj = matchedMmproj
                  } else {
                    nextConfig.model.mmproj = ''
                  }

                  debugLog('deploy', 'Calling setConfig with atomic model update', { name: newName, variant: nextVariant, mmproj: nextConfig.model.mmproj })
                  setConfig(nextConfig)

                  // Save to localStorage for persistence
                  saveDeploySettings(nextConfig, selectedApiKey)

                  // Update command line preview (single call)
                  updateCommandPreview(nextConfig)
                }}
	                sx={{
	                  fontSize: '0.875rem',
	                  '& .MuiOutlinedInput-root': {
	                    borderRadius: '8px',
	                    backgroundColor: 'background.default',
	                  }
	                }}
              >
                {availableModelNames.map((name) => (
                  <MenuItem key={name} value={name}>{name}</MenuItem>
                ))}
              </Select>
            </Grid>
            {availableVariantsForSelected.length > 1 && (
              <Grid item xs={12} md={4}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Model Variant</Typography>
                <Select
                    fullWidth
                    size="small"
                    value={config?.model?.variant ?? ''}
                    MenuProps={deployMenuProps}
                    onChange={(e) => {
                    deployLog('variantSelect', 'Variant selection changed', {
                      newVariant: e.target.value,
                      previousVariant: config?.model?.variant,
                      modelName: config?.model?.name
                    })
                    updateConfig('model.variant', e.target.value)
                  }}
	                  sx={{
	                    fontSize: '0.875rem',
	                    '& .MuiOutlinedInput-root': {
	                      borderRadius: '8px',
	                      backgroundColor: 'background.default',
	                    }
	                  }}
                >
                  {availableVariantsForSelected.map((variant) => (
                    <MenuItem key={variant} value={variant}>{variant}</MenuItem>
                  ))}
                </Select>
              </Grid>
            )}
            {availableVariantsForSelected.length <= 1 && (
              <Grid item xs={12} md={4}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Model Variant</Typography>
                <Typography
                  sx={{
                    fontSize: '0.875rem',
                    color: 'text.secondary',
                    fontStyle: 'italic',
                    py: 1,
                    px: 1,
                    backgroundColor: 'action.hover',
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: 'divider'
                  }}
                >
                  {config?.model?.variant ?? '—'} (only variant available)
                </Typography>
              </Grid>
            )}
            {showAdvancedModelSettings && (
              <Grid item xs={12} md={6}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Chat Template</Typography>
                <Select
                  fullWidth
                    size="small"
                    value={selectedTemplate}
                    disabled={templatesLoading}
                    MenuProps={deployMenuProps}
                    onChange={async (e) => {
                    const filename = e.target.value as string
                    deployLog('templateSelect', 'Template selection changed', {
                      newTemplate: filename,
                      previousTemplate: selectedTemplate
                    })
                    setSelectedTemplate(filename)
                    try {
                      deployLog('templateSelect', 'Sending template selection to backend')
                      await apiService.selectTemplate(filename)
                      // Refresh config and command preview to reflect new template selection
                      const cfgRes = await fetch(`/api/v1/service/config`)
                      if (cfgRes.ok) {
                        const cfgJson = await cfgRes.json()
                        deployLog('templateSelect', 'Template applied, config refreshed')
                        // Preserve current model name and variant - only update command preview
                        // Don't overwrite user's model selection with server config
                        setCommandLine(cfgJson.command || '')
                      }
                    } catch (err) {
                      deployLog('templateSelect', 'Template selection failed:', err)
                      // revert on error
                      setSelectedTemplate((prev) => prev)
                      setError(err instanceof Error ? err.message : 'Failed to select template')
                    }
                  }}
	                  sx={{
	                    fontSize: '0.875rem',
	                    '& .MuiOutlinedInput-root': {
	                      borderRadius: '8px'
	                    }
	                  }}
                >
                  <MenuItem value="">
                    <em>GGUF embedded (no --jinja)</em>
                  </MenuItem>
                  <MenuItem value={TEMPLATE_EMBEDDED_JINJA}>
                    GGUF embedded + Jinja (--jinja only)
                  </MenuItem>
                  {templates.length === 0 ? (
                    <MenuItem value="no-templates" disabled>
                      No custom templates found in {templatesDir || '/home/llamacpp/templates'}
                    </MenuItem>
                  ) : (
                    templates.map((f) => (
                      <MenuItem key={f} value={f}>{f}</MenuItem>
                    ))
                  )}
                </Select>
              </Grid>
            )}
          </Grid>
          <Box
            sx={{
                mt: 1.75,
                pt: 1.75,
                borderTop: '1px solid',
                borderColor: 'divider',
              }}
            >
              <Grid container spacing={1.5}>
              <Grid item xs={6} md={3}>
                <ParameterField
                  label="Context Size"
                  description="Prompt context in tokens."
                  path="model.context_size"
                  value={config.model.context_size}
                  defaultValue={getDefaultValue('model.context_size')}
                  type="number"
                  min={512}
                  max={LLAMACPP_CONTEXT_SIZE_MAX}
                  compact
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={6} md={3}>
                <ParameterField
                  label="GPU Layers"
                  description="Layers to offload to GPU. -1 uses all available layers."
                  path="model.gpu_layers"
                  value={config.model.gpu_layers}
                  defaultValue={getDefaultValue('model.gpu_layers')}
                  type="number"
                  min={-1}
                  max={999}
                  compact
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Execution Mode</Typography>
                <Select
                  fullWidth
                    size="small"
                    value={config.execution?.mode || 'gpu'}
                    MenuProps={deployMenuProps}
                    onChange={(e) => updateConfig('execution.mode', e.target.value)}
                  sx={deploySelectSx}
                >
                  <MenuItem value="gpu">GPU Acceleration</MenuItem>
                  <MenuItem value="cpu">CPU Only</MenuItem>
                </Select>
                <FormHelperText sx={{ mt: 0.5, fontSize: '0.75rem', color: 'text.secondary' }}>
                  CPU mode overrides GPU layers.
                </FormHelperText>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>CUDA Devices</Typography>
                {['all', 'custom', ...gpuList.map(g => g.index.toString())].includes(config.execution?.cuda_devices || 'all') ? (
                  <Select
                    fullWidth
                      size="small"
                      value={config.execution?.cuda_devices || 'all'}
                      MenuProps={deployMenuProps}
                      onChange={(e) => {
                      if (e.target.value === 'custom') {
                        updateConfig('execution.cuda_devices', '0,1')
                      } else {
                        updateConfig('execution.cuda_devices', e.target.value)
                      }
                    }}
                    disabled={config.execution?.mode === 'cpu'}
                    sx={deploySelectSx}
                  >
                    <MenuItem value="all">All GPUs</MenuItem>
                    {gpuList.map((gpu) => (
                      <MenuItem key={gpu.index} value={gpu.index.toString()}>
                        GPU {gpu.index}: {gpu.name} ({(gpu.vram_total_mb / 1024).toFixed(1)} GB)
                      </MenuItem>
                    ))}
                    {gpuList.length > 1 && (
                      <MenuItem value="custom">Custom</MenuItem>
                    )}
                  </Select>
                ) : (
                  <TextField
                    fullWidth
                    size="small"
                    value={config.execution?.cuda_devices || 'all'}
                    onChange={(e) => updateConfig('execution.cuda_devices', e.target.value)}
                    disabled={config.execution?.mode === 'cpu'}
                    placeholder="0,1"
                    sx={deployFieldSx}
                  />
                )}
                <FormHelperText sx={{ mt: 0.5, fontSize: '0.75rem', color: 'text.secondary' }}>
                  {config.execution?.mode === 'cpu' ? 'Disabled in CPU mode.' : gpusAvailable ? `${gpuList.length} GPU(s) detected.` : 'No GPU detected.'}
                </FormHelperText>
              </Grid>
            </Grid>
          </Box>
      </SectionCard>

      {/* VRAM Estimation Display */}
      {vramEstimate && (
        <SectionCard
          fullHeight
          title="VRAM estimation"
          subtitle={`${vramEstimate.total_gb.toFixed(1)} GB required / ${vramEstimate.available_vram_gb.toFixed(1)} GB available`}
          icon={<MemoryIcon />}
          accentColor={vramAccentColor}
          collapsible={isNarrow}
          defaultExpanded={!isNarrow}
          sx={{ height: '100%' }}
          action={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {vramLoading && <CircularProgress size={16} />}
              <StatusChip status={vramStatusKey} statusConfig={DEPLOY_STATUS_CONFIG} />
            </Box>
          }
        >
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', lg: 'repeat(4, minmax(0, 1fr))' },
              gap: 1,
            }}
          >
            <MetricTile
              label="Model weights"
              value={`${(vramEstimate.model_weights_mb / 1024).toFixed(1)} GB`}
              detail="weights"
              icon={<LibraryIcon />}
              color={accentColors.purple}
            />
            <MetricTile
              label="KV cache"
              value={`${(vramEstimate.kv_cache_mb / 1024).toFixed(2)} GB`}
              detail="context memory"
              icon={<MemoryIcon />}
              color={accentColors.info}
            />
            <MetricTile
              label="Compute buffer"
              value={`${(vramEstimate.compute_buffer_mb / 1024).toFixed(2)} GB`}
              detail="runtime buffer"
              icon={<TuneIcon />}
              color={accentColors.warning}
            />
            <MetricTile
              label="Total required"
              value={`${vramEstimate.total_gb.toFixed(1)} GB`}
              detail={`${vramEstimate.available_vram_gb.toFixed(1)} GB available`}
              icon={<MemoryIcon />}
              color={vramAccentColor}
            />
          </Box>

          <Box
            sx={{
              mt: 1,
              p: 1,
              borderRadius: '8px',
              border: '1px solid',
              borderColor: 'divider',
              bgcolor: 'action.hover',
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, mb: 0.75 }}>
              <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700 }}>
                VRAM utilization
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700 }}>
                {vramEstimate.utilization_percent}%
              </Typography>
            </Box>
            <Box
              role="progressbar"
              aria-label="VRAM utilization"
              aria-valuenow={vramUtilizationPercent}
              aria-valuemin={0}
              aria-valuemax={100}
              sx={{
                width: '100%',
                height: 10,
                bgcolor: 'background.default',
                borderRadius: '999px',
                border: '1px solid',
                borderColor: 'divider',
                overflow: 'hidden',
              }}
            >
              <Box
                sx={{
                  width: `${vramUtilizationPercent}%`,
                  height: '100%',
                  bgcolor: vramUtilizationColor,
                  borderRadius: '999px',
                  transition: 'width 0.3s ease',
                }}
              />
            </Box>
          </Box>

          {vramEstimate.warnings.length > 0 && (
            <Box sx={{ mt: 2 }}>
              {vramEstimate.warnings.map((warning, idx) => (
                <Alert
                  key={idx}
                  severity="warning"
                  sx={{
                    mb: 1,
                    py: 0.25,
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: 'warning.main',
                    fontSize: '0.75rem',
                  }}
                >
                  {warning}
                </Alert>
              ))}
            </Box>
          )}
        </SectionCard>
      )}
      </Box>
      )}

      {backend === 'vllm' && vllmReloading && (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2, py: 4, mb: 2 }}>
          <CircularProgress size={28} />
          <Typography variant="body2" color="text.secondary">Loading vLLM configuration…</Typography>
        </Box>
      )}
      {backend === 'vllm' && !vllmReloading && vllmReloadError && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setVllmReloadError(null)}>
          {vllmReloadError}. Ensure API and UI images are rebuilt/recreated, then hard-refresh (Ctrl+Shift+R).
        </Alert>
      )}

      {backend === 'vllm' && vllmConfig && activeDeployTab === 0 && (
        <SectionCard
          title="vLLM model"
          subtitle="Pick the catalog model, served name, dtype, and quantization used for the vLLM launch."
          icon={<LibraryIcon />}
          accentColor={accentColors.info}
          sx={{ mb: 3, scrollMarginTop: '96px' }}
          action={<StatusChip status="vllm" statusConfig={DEPLOY_STATUS_CONFIG} />}
        >
          <Grid container spacing={2}>
            {vllmCatalogModelPickers()}
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem' }}>dtype</Typography>
                {vllmFieldMeta('model', 'dtype') && <FrameworkBadge scope={vllmFieldMeta('model', 'dtype')!.scope} />}
              </Box>
              <Select fullWidth size="small" value={vllmConfig.model.dtype}
                onChange={(e) => updateVllmConfig('model.dtype', e.target.value)}>
                {(vllmFieldMeta('model', 'dtype')?.options || ['auto', 'float16', 'bfloat16', 'float32']).map((o) => (
                  <MenuItem key={o} value={o}>{o}</MenuItem>
                ))}
              </Select>
              <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('model', 'dtype').join(' — ')}</FormHelperText>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem' }}>quantization</Typography>
                {vllmFieldMeta('model', 'quantization') && <FrameworkBadge scope={vllmFieldMeta('model', 'quantization')!.scope} />}
              </Box>
              <Select fullWidth size="small" value={vllmConfig.model.quantization || ''}
                onChange={(e) => updateVllmConfig('model.quantization', e.target.value)}>
                <MenuItem value=""><em>None</em></MenuItem>
                {(vllmFieldMeta('model', 'quantization')?.options || ['fp4', 'awq', 'gptq', 'bitsandbytes']).filter((o) => o !== 'None').map((o) => (
                  <MenuItem key={o} value={o}>{o}</MenuItem>
                ))}
              </Select>
              <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('model', 'quantization').join(' — ')}</FormHelperText>
            </Grid>
          </Grid>
        </SectionCard>
      )}

        {backend === 'llamacpp' && activeDeployTab === 0 && showAdvancedModelSettings && (
          <SectionCard
            title="Advanced model settings"
            icon={<LibraryIcon />}
            accentColor={modelCapabilityStatus === 'vision' ? accentColors.success : accentColors.purple}
            sx={{ mb: 1.5, scrollMarginTop: '72px' }}
          >
            <Grid container spacing={1.5}>
                <Grid item xs={12} md={6}>
                <ParameterField
                  label="LoRA Adapter"
                  description="Path to a LoRA (Low-Rank Adaptation) adapter file. LoRA allows fine-tuning specific behaviors without modifying the base model weights."
                  path="model.lora"
                  value={config.model.lora || ''}
                  defaultValue={getDefaultValue('model.lora')}
                  type="text"
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="LoRA Base Model"
                  description="Path to the base model to use with the LoRA adapter. This should be the original model that the LoRA was trained on."
                  path="model.lora_base"
                  value={config.model.lora_base || ''}
                  defaultValue={getDefaultValue('model.lora_base')}
                  type="text"
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
                {/* Vision / Multimodal Projector Section */}
                <Grid item xs={12}>
                  <Box sx={{
                    p: 1.25,
                    bgcolor: config.model.mmproj ? alpha(accentColors.success, 0.08) : 'action.hover',
                    borderRadius: '8px',
                    border: '1px solid',
                  borderColor: config.model.mmproj ? alpha(accentColors.success, 0.25) : 'divider',
                  transition: 'all 0.2s ease'
                }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {config.model.mmproj ? (
                        <VisionIcon sx={{ fontSize: '1.1rem', color: accentColors.success }} />
                      ) : (
                        <VisionOffIcon sx={{ fontSize: '1.1rem', color: 'text.secondary' }} />
                      )}
                      <Typography variant="subtitle2" sx={{ fontSize: '0.875rem', fontWeight: 600 }}>
                        Vision / Multimodal Projector
                      </Typography>
                      <StatusChip
                        status={config.model.mmproj ? 'vision' : 'text'}
                        label={config.model.mmproj ? 'Active' : 'Disabled'}
                        height={20}
                        statusConfig={DEPLOY_STATUS_CONFIG}
                      />
                    </Box>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <Tooltip title="Refresh mmproj file list">
                        <IconButton
                          size="small"
                          onClick={refreshMmprojFiles}
                          disabled={mmprojLoading}
                          sx={{ opacity: mmprojLoading ? 0.5 : 1 }}
                        >
                          {mmprojLoading ? <CircularProgress size={16} /> : <RefreshIcon fontSize="small" />}
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Download mmproj from HuggingFace">
                        <IconButton
                          size="small"
                          onClick={() => setDownloadMmprojDialogOpen(true)}
                        >
                          <FileDownloadIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </Box>

                    {mmprojAutoMatched && config.model.mmproj && (
                    <Alert
                      severity="info"
                      sx={{
                        mb: 1.5,
                        py: 0.25,
                        borderRadius: '8px',
                        border: '1px solid',
                        borderColor: 'info.main',
                        fontSize: '0.75rem',
                      }}
                    >
                      Auto-matched projector file for this model
                    </Alert>
                  )}

                    <Select
                      fullWidth
                      size="small"
                      value={config.model.mmproj || ''}
                      MenuProps={deployMenuProps}
                      onChange={(e) => {
                      const value = e.target.value === '' ? undefined : e.target.value
                      updateConfig('model.mmproj', value)
                      setMmprojAutoMatched(false)
                    }}
                    displayEmpty
                    sx={{
                      fontSize: '0.875rem',
                      '& .MuiOutlinedInput-root': {
                        borderRadius: '8px',
                        backgroundColor: 'background.default',
                      }
                    }}
                  >
                    <MenuItem value="">
                      <em>None — text-only mode</em>
                    </MenuItem>
                    {mmprojFiles.map(f => (
                      <MenuItem key={f.name} value={f.name}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                          <Typography sx={{ fontSize: '0.875rem' }}>{f.name}</Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ ml: 2 }}>
                            {f.size_mb} MB
                          </Typography>
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>

                  {mmprojFiles.length === 0 && !mmprojLoading && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                      No projector files found. Download one using the button above, or place mmproj GGUF files in the models directory.
                    </Typography>
                  )}
                </Box>
              </Grid>

              {/* Download mmproj Dialog */}
	              {downloadMmprojDialogOpen && (
	                <Grid item xs={12}>
	                  <Box sx={{
	                    p: 2,
	                    bgcolor: alpha(accentColors.info, 0.08),
	                    borderRadius: '8px',
	                    border: '1px solid',
	                    borderColor: alpha(accentColors.info, 0.25),
	                  }}>
                    <Typography variant="subtitle2" sx={{ fontSize: '0.875rem', fontWeight: 600, mb: 1 }}>
                      Download Vision Projector
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem', mb: 1.5 }}>
                      Download the mmproj file from the same HuggingFace repository as your model.
                      Look for files starting with "mmproj-" in the repo's file list.
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button
	                        variant="outlined"
	                        size="small"
	                        startIcon={<DownloadIcon />}
	                        onClick={() => {
	                          setDownloadMmprojDialogOpen(false)
	                          navigate(modelsPath)
	                        }}
	                        sx={{ borderRadius: '8px', textTransform: 'none', fontSize: '0.8125rem' }}
	                      >
                        Go to Models Page to Download
                      </Button>
	                      <Button
	                        size="small"
	                        onClick={() => setDownloadMmprojDialogOpen(false)}
	                        sx={{ borderRadius: '8px', textTransform: 'none', fontSize: '0.8125rem' }}
	                      >
                        Cancel
                      </Button>
                    </Box>
                  </Box>
                </Grid>
              )}
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="RoPE Scaling Method"
                  description="RoPE (Rotary Position Embedding) frequency scaling method for extending context length. Leave empty to use model default."
                  path="model.rope_scaling"
                  value={config.model.rope_scaling ?? ''}
                  defaultValue={getDefaultValue('model.rope_scaling')}
                  type="select"
                  options={[
                    { value: 'none', label: 'None' },
                    { value: 'linear', label: 'Linear' },
                    { value: 'yarn', label: 'YaRN' }
                  ]}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="RoPE Base Frequency"
                  description="Base frequency for RoPE calculations. Leave empty to use the model's default value. Higher values may help with longer contexts."
                  path="model.rope_freq_base"
                  value={config.model.rope_freq_base ?? ''}
                  defaultValue={getDefaultValue('model.rope_freq_base')}
                  type="number"
                  min={0}
                  step={0.1}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="RoPE Frequency Scale"
                  description="RoPE frequency scaling factor. Values less than 1.0 expand the context window. Leave empty to use model default."
                  path="model.rope_freq_scale"
                  value={config.model.rope_freq_scale ?? ''}
                  defaultValue={getDefaultValue('model.rope_freq_scale')}
                  type="number"
                  min={0}
                  step={0.1}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="CPU MoE Layers"
                  description="Keep the Mixture of Experts (MoE) weights of the first N layers in the CPU. Set to 0 to disable."
                  path="model.n_cpu_moe"
                  value={config.model.n_cpu_moe ?? ''}
                  defaultValue={getDefaultValue('model.n_cpu_moe')}
                  type="number"
                  min={0}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600, mt: 2 }}>Attention & Memory</Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <ParameterField
                  label="Flash Attention"
                  description="Enable flash attention for faster inference and lower memory usage. 'auto' lets llama-server decide based on model compatibility. (--flash-attn)"
                  path="model.flash_attn"
                  value={config.model.flash_attn ?? ''}
                  defaultValue={getDefaultValue('model.flash_attn')}
                  type="select"
                  options={[
                    { value: 'auto', label: 'Auto (recommended)' },
                    { value: 'on', label: 'On' },
                    { value: 'off', label: 'Off' },
                  ]}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <ParameterField
                  label="KV Offload"
                  description="Offload KV cache to GPU. Disabling keeps KV cache on CPU, freeing GPU VRAM but slowing inference. (--kv-offload / --no-kv-offload)"
                  path="model.kv_offload"
                  value={config.model.kv_offload === undefined ? '' : config.model.kv_offload ? 'true' : 'false'}
                  defaultValue={'true'}
                  type="select"
                  options={[
                    { value: 'true', label: 'Enabled (GPU)' },
                    { value: 'false', label: 'Disabled (CPU)' },
                  ]}
                  onChange={(path, value) => updateConfig(path, value === '' ? undefined : value === 'true')}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <ParameterField
                  label="Defrag Threshold"
                  description="KV cache defragmentation threshold. When fragmentation exceeds this ratio (0.0-1.0), the cache is defragmented. Lower values trigger more frequent defrag. (--defrag-thold)"
                  path="model.defrag_thold"
                  value={config.model.defrag_thold ?? ''}
                  defaultValue={getDefaultValue('model.defrag_thold')}
                  type="number"
                  min={0}
                  max={1}
                  step={0.1}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>

              {/* Multi-Token Prediction (MTP) */}
              <Grid item xs={12}>
                <MtpConfigSection
                  mtp={config.mtp}
                  modelMtpCapable={selectedDeployModel?.mtpCapable === true}
                  backendMtpSupported={llamacppMtpSupported}
                  backendBuildLabel={llamacppBuildTag}
                  parallelSlots={config.performance?.parallel_slots ?? 1}
                  selectedModel={selectedDeployModel}
                  onChange={updateConfig}
                  onWorkloadProfileChange={handleMtpWorkloadProfileChange}
                  onMtpEnabledChange={handleMtpEnabledChange}
                />
              </Grid>

              <Grid item xs={12}>
                <AgentToolsConfigSection
                  server={config.server}
                  workloadProfile={config.mtp?.workload_profile}
                  grammarGbnfSupported
                  llguidanceEnabled={llamacppLlguidanceEnabled}
                  onChange={updateConfig}
                  onSwitchToCustomProfile={switchMtpToCustomProfile}
                />
              </Grid>

              {/* Classic draft-model speculative decoding */}
              <Grid item xs={12}>
                <Accordion
                  disableGutters
                  sx={{
                    bgcolor: 'action.hover',
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: '8px !important',
                    '&:before': { display: 'none' },
                    boxShadow: 'none',
                  }}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Speculative Decoding (draft model)</Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 1, mt: 0.3 }}>(advanced)</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: '0.75rem' }}>
                      Speculative decoding uses a smaller draft model to predict tokens, then verifies with the main model. This can dramatically speed up inference (2-3x) with no quality loss.
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <ParameterField
                          label="Draft Model"
                          description="Path to a smaller draft model for speculative decoding. Should be the same architecture as the main model. (--model-draft)"
                          path="speculative.model_draft"
                          value={config.speculative?.model_draft || ''}
                          defaultValue={getDefaultValue('speculative.model_draft')}
                          type="text"
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <ParameterField
                          label="Draft GPU Layers"
                          description="Number of layers to offload to GPU for the draft model. Use fewer than main model to save VRAM. (--gpu-layers-draft)"
                          path="speculative.gpu_layers_draft"
                          value={config.speculative?.gpu_layers_draft ?? ''}
                          defaultValue={getDefaultValue('speculative.gpu_layers_draft')}
                          type="number"
                          min={-1}
                          max={999}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Draft Context Size"
                          description="Context size for the draft model. Can be smaller than the main model. (--ctx-size-draft)"
                          path="speculative.ctx_size_draft"
                          value={config.speculative?.ctx_size_draft ?? ''}
                          defaultValue={getDefaultValue('speculative.ctx_size_draft')}
                          type="number"
                          min={128}
                          max={131072}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Draft Max Tokens"
                          description="Maximum number of tokens to draft before verification. Higher values are faster but may waste compute if rejected. (--draft-max)"
                          path="speculative.draft_max"
                          value={config.speculative?.draft_max ?? ''}
                          defaultValue={getDefaultValue('speculative.draft_max')}
                          type="number"
                          min={1}
                          max={64}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Draft Min Tokens"
                          description="Minimum number of tokens to draft. Ensures at least this many tokens are drafted each cycle. (--draft-min)"
                          path="speculative.draft_min"
                          value={config.speculative?.draft_min ?? ''}
                          defaultValue={getDefaultValue('speculative.draft_min')}
                          type="number"
                          min={0}
                          max={64}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Grid>
          </Grid>
        </SectionCard>
      )}

      {backend === 'vllm' && vllmConfig && activeDeployTab === 1 && (
        <SectionCard
          title="vLLM sampling"
          subtitle="Stored defaults for request merging; OpenAI-compatible sampling is usually chosen per request."
          icon={<ChatIcon />}
          accentColor={accentColors.info}
          sx={{ mb: 3, scrollMarginTop: '96px' }}
          action={<StatusChip status="reference" statusConfig={DEPLOY_STATUS_CONFIG} />}
        >
          <Grid container spacing={2}>
            {(['temperature', 'top_p', 'top_k', 'repetition_penalty', 'frequency_penalty', 'presence_penalty'] as const).map((key) => (
              <Grid item xs={12} md={6} key={key}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                  <Typography sx={{ fontSize: '0.875rem', textTransform: 'capitalize' }}>{key.replace(/_/g, ' ')}</Typography>
                  {vllmFieldMeta('sampling', key) && <FrameworkBadge scope={vllmFieldMeta('sampling', key)!.scope} />}
                </Box>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  value={vllmConfig.sampling[key]}
                  onChange={(e) => {
                    const raw = e.target.value
                    const n = key === 'top_k' ? parseInt(raw, 10) : parseFloat(raw)
                    updateVllmConfig(`sampling.${key}`, Number.isNaN(n) ? 0 : n)
                  }}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      borderRadius: '8px',
                      backgroundColor: 'background.default',
                    }
                  }}
                />
                <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('sampling', key).join(' — ')}</FormHelperText>
              </Grid>
            ))}
          </Grid>
        </SectionCard>
      )}

      {backend === 'llamacpp' && activeDeployTab === 1 && (
        <SectionCard
          title="Sampling configuration"
          subtitle="Core token selection defaults and presets."
          icon={<ChatIcon />}
          accentColor={accentColors.purple}
          sx={{ mb: 3 }}
          action={
            <StatusChip
              status={selectedPreset ? 'sampling' : 'custom'}
              label={selectedPreset || 'Custom'}
              statusConfig={DEPLOY_STATUS_CONFIG}
            />
          }
        >
          <Box
            sx={{
              mb: 2,
              p: 1.25,
              borderRadius: '8px',
              border: '1px solid',
              borderColor: 'divider',
              bgcolor: 'action.hover',
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              flexWrap: 'wrap',
            }}
          >
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem', fontWeight: 700, whiteSpace: 'nowrap' }}>
              Presets
            </Typography>
            <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', alignItems: 'center' }}>
              {SAMPLING_PRESETS.map((preset) => {
                const selected = selectedPreset === preset.name

                return (
                  <Tooltip key={preset.name} title={preset.description}>
                    <StatusChip
                      status={preset.category}
                      label={preset.name}
                      variant={selected ? 'filled' : 'outlined'}
                      statusConfig={DEPLOY_STATUS_CONFIG}
                      onClick={() => applySamplingPreset(preset)}
                      sx={deployPresetChipSx(preset.category, selected)}
                    />
                  </Tooltip>
                )
              })}
            </Box>
          </Box>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} lg={3}>
                <ParameterField
                  label="Temperature"
                  description="Randomness; lower is steadier."
                  path="sampling.temperature"
                  value={config.sampling.temperature}
                  defaultValue={getDefaultValue('sampling.temperature')}
                  type="number"
                  min={0}
                  max={2}
                  step={0.1}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} sm={6} lg={3}>
                <ParameterField
                  label="Top P"
                  description="Nucleus probability."
                  path="sampling.top_p"
                  value={config.sampling.top_p}
                  defaultValue={getDefaultValue('sampling.top_p')}
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} sm={6} lg={3}>
                <ParameterField
                  label="Top K"
                  description="Candidate token limit; 0 disables."
                  path="sampling.top_k"
                  value={config.sampling.top_k}
                  defaultValue={getDefaultValue('sampling.top_k')}
                  type="number"
                  min={0}
                  max={200}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} sm={6} lg={3}>
                <ParameterField
                  label="Min P"
                  description="Relative probability floor; 0 disables."
                  path="sampling.min_p"
                  value={config.sampling.min_p}
                  defaultValue={getDefaultValue('sampling.min_p')}
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} sm={6} lg={3}>
                <ParameterField
                  label="Repeat Penalty"
                  description="Repetition pressure; 1 disables."
                  path="sampling.repeat_penalty"
                  value={config.sampling.repeat_penalty}
                  defaultValue={getDefaultValue('sampling.repeat_penalty')}
                  type="number"
                  min={0.1}
                  max={2.0}
                  step={0.1}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} sm={6} lg={3}>
                <ParameterField
                  label="Repeat Last N"
                  description="Recent-token window; -1 uses context."
                  path="sampling.repeat_last_n"
                  value={config.sampling.repeat_last_n}
                  defaultValue={getDefaultValue('sampling.repeat_last_n')}
                  type="number"
                  min={-1}
                  max={2048}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} sm={6} lg={3}>
                <ParameterField
                  label="Frequency Penalty"
                  description="Penalty for frequent tokens."
                  path="sampling.frequency_penalty"
                  value={config.sampling.frequency_penalty}
                  defaultValue={getDefaultValue('sampling.frequency_penalty')}
                  type="number"
                  min={-2.0}
                  max={2.0}
                  step={0.1}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} sm={6} lg={3}>
                <ParameterField
                  label="Presence Penalty"
                  description="Penalty for used tokens."
                  path="sampling.presence_penalty"
                  value={config.sampling.presence_penalty}
                  defaultValue={getDefaultValue('sampling.presence_penalty')}
                  type="number"
                  min={-2.0}
                  max={2.0}
                  step={0.1}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12}>
                <Accordion
                  disableGutters
                  sx={{
                    bgcolor: 'action.hover',
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: '8px !important',
                    '&:before': { display: 'none' },
                    boxShadow: 'none',
                  }}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>DRY sampling</Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 1, mt: 0.3 }}>repeat sequence penalty</Typography>
                  </AccordionSummary>
                  <AccordionDetails sx={{ pt: 0 }}>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6} lg={3}>
                        <ParameterField
                          label="DRY Multiplier"
                          description="Penalty strength; 0 disables."
                          path="sampling.dry_multiplier"
                          value={config.sampling.dry_multiplier}
                          defaultValue={getDefaultValue('sampling.dry_multiplier')}
                          type="number"
                          min={0.0}
                          max={5.0}
                          step={0.1}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} lg={3}>
                        <ParameterField
                          label="DRY Base"
                          description="Penalty scaling base."
                          path="sampling.dry_base"
                          value={config.sampling.dry_base}
                          defaultValue={getDefaultValue('sampling.dry_base')}
                          type="number"
                          min={1.0}
                          max={4.0}
                          step={0.1}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} lg={3}>
                        <ParameterField
                          label="DRY Allowed Length"
                          description="Minimum repeat length."
                          path="sampling.dry_allowed_length"
                          value={config.sampling.dry_allowed_length}
                          defaultValue={getDefaultValue('sampling.dry_allowed_length')}
                          type="number"
                          min={1}
                          max={20}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} lg={3}>
                        <ParameterField
                          label="DRY Penalty Last N"
                          description="History window; -1 uses context."
                          path="sampling.dry_penalty_last_n"
                          value={config.sampling.dry_penalty_last_n}
                          defaultValue={getDefaultValue('sampling.dry_penalty_last_n')}
                          type="number"
                          min={-1}
                          max={2048}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Grid>

              {/* Advanced Sampling - Collapsible */}
              <Grid item xs={12}>
                <Accordion
                  disableGutters
                  sx={{
                    bgcolor: 'action.hover',
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: '8px !important',
                    '&:before': { display: 'none' },
                    boxShadow: 'none',
                  }}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Advanced Sampling</Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 1, mt: 0.3 }}>(mirostat, dynamic temperature, XTC)</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={3}>
                      <Grid item xs={12}>
                        <Typography variant="subtitle2" gutterBottom sx={{ fontSize: '0.8125rem', fontWeight: 600 }}>Mirostat Sampling</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontSize: '0.75rem' }}>
                          Mirostat is an adaptive sampling algorithm that maintains a target perplexity (surprise level), dynamically adjusting the sampling distribution.
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Mirostat Mode"
                          description="Mirostat sampling mode. 0 = disabled, 1 = Mirostat v1, 2 = Mirostat v2. When enabled, top_k/top_p/min_p are ignored. (--mirostat)"
                          path="sampling.mirostat"
                          value={config.sampling.mirostat ?? ''}
                          defaultValue={getDefaultValue('sampling.mirostat')}
                          type="select"
                          options={[
                            { value: '0', label: '0 - Disabled' },
                            { value: '1', label: '1 - Mirostat v1' },
                            { value: '2', label: '2 - Mirostat v2' },
                          ]}
                          onChange={(path, value) => updateConfig(path, value === '' ? undefined : parseInt(value))}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Mirostat Learning Rate"
                          description="Mirostat learning rate (eta). Controls how quickly the algorithm adapts. Higher values = faster adaptation. (--mirostat-lr)"
                          path="sampling.mirostat_lr"
                          value={config.sampling.mirostat_lr ?? ''}
                          defaultValue={getDefaultValue('sampling.mirostat_lr')}
                          type="number"
                          min={0}
                          max={1}
                          step={0.01}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Mirostat Entropy"
                          description="Mirostat target entropy (tau). The desired perplexity level. Lower = more focused, Higher = more creative. (--mirostat-ent)"
                          path="sampling.mirostat_ent"
                          value={config.sampling.mirostat_ent ?? ''}
                          defaultValue={getDefaultValue('sampling.mirostat_ent')}
                          type="number"
                          min={0}
                          max={10}
                          step={0.1}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>

                      <Grid item xs={12}>
                        <Typography variant="subtitle2" gutterBottom sx={{ fontSize: '0.8125rem', fontWeight: 600, mt: 1 }}>Dynamic Temperature</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontSize: '0.75rem' }}>
                          Dynamically adjusts temperature within a range based on token entropy. Provides lower temperature for confident predictions and higher for uncertain ones.
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <ParameterField
                          label="Dynamic Temp Range"
                          description="Range of dynamic temperature adjustments. 0 = disabled. When set, temperature varies from (temp - range) to (temp + range). (--dynatemp-range)"
                          path="sampling.dynatemp_range"
                          value={config.sampling.dynatemp_range ?? ''}
                          defaultValue={getDefaultValue('sampling.dynatemp_range')}
                          type="number"
                          min={0}
                          max={5}
                          step={0.1}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <ParameterField
                          label="Dynamic Temp Exponent"
                          description="Exponent for dynamic temperature scaling. Controls the shape of the temperature curve. (--dynatemp-exp)"
                          path="sampling.dynatemp_exp"
                          value={config.sampling.dynatemp_exp ?? ''}
                          defaultValue={getDefaultValue('sampling.dynatemp_exp')}
                          type="number"
                          min={0}
                          max={5}
                          step={0.1}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>

                      <Grid item xs={12}>
                        <Typography variant="subtitle2" gutterBottom sx={{ fontSize: '0.8125rem', fontWeight: 600, mt: 1 }}>Other Advanced Samplers</Typography>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Top N-Sigma"
                          description="Filters tokens based on statistical significance. Only tokens within N standard deviations of the mean logit are kept. 0 = disabled. (--top-n-sigma)"
                          path="sampling.top_n_sigma"
                          value={config.sampling.top_n_sigma ?? ''}
                          defaultValue={getDefaultValue('sampling.top_n_sigma')}
                          type="number"
                          min={0}
                          max={10}
                          step={0.1}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Typical P"
                          description="Locally typical sampling threshold. Filters tokens based on how typical they are given the context. 1.0 = disabled. (--typical-p)"
                          path="sampling.typical_p"
                          value={config.sampling.typical_p ?? ''}
                          defaultValue={getDefaultValue('sampling.typical_p')}
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="XTC Probability"
                          description="XTC (eXtreme Token Control) sampling probability. Probability of removing top tokens. 0 = disabled. (--xtc-probability)"
                          path="sampling.xtc_probability"
                          value={config.sampling.xtc_probability ?? ''}
                          defaultValue={getDefaultValue('sampling.xtc_probability')}
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="XTC Threshold"
                          description="XTC threshold for token removal. Tokens with probability above this threshold may be removed. (--xtc-threshold)"
                          path="sampling.xtc_threshold"
                          value={config.sampling.xtc_threshold ?? ''}
                          defaultValue={getDefaultValue('sampling.xtc_threshold')}
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Grid>
            </Grid>
        </SectionCard>
      )}

      {backend === 'vllm' && vllmConfig && activeDeployTab === 2 && (
        <SectionCard
          title="vLLM performance"
          subtitle="Tune model length, batching, parallelism, KV cache behavior, and scheduler flags."
          icon={<SpeedIcon />}
          accentColor={accentColors.success}
          sx={{ mb: 3, scrollMarginTop: '96px' }}
          action={<StatusChip status="performance" statusConfig={DEPLOY_STATUS_CONFIG} />}
        >
          <Grid container spacing={2}>
            {([
              { path: 'performance.max_model_len', label: 'max_model_len', type: 'int', meta: 'max_model_len' },
              { path: 'performance.gpu_memory_utilization', label: 'gpu_memory_utilization', type: 'float', meta: 'gpu_memory_utilization' },
              { path: 'performance.tensor_parallel_size', label: 'tensor_parallel_size', type: 'int', meta: 'tensor_parallel_size' },
              { path: 'performance.pipeline_parallel_size', label: 'pipeline_parallel_size', type: 'int', meta: 'pipeline_parallel_size' },
              { path: 'performance.data_parallel_size', label: 'data_parallel_size', type: 'int', meta: 'data_parallel_size' },
              { path: 'performance.max_num_seqs', label: 'max_num_seqs', type: 'int', meta: 'max_num_seqs' },
              { path: 'performance.max_num_batched_tokens', label: 'max_num_batched_tokens', type: 'int', meta: 'max_num_batched_tokens' },
            ] as const).map((row) => (
              <Grid item xs={12} md={6} key={row.path}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                  <Typography sx={{ fontSize: '0.875rem' }}>{row.label}</Typography>
                  {vllmFieldMeta('performance', row.meta) && <FrameworkBadge scope={vllmFieldMeta('performance', row.meta)!.scope} />}
                </Box>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  value={(vllmConfig as any).performance[row.meta]}
                  onChange={(e) => {
                    const raw = e.target.value
                    const n = row.type === 'int' ? parseInt(raw, 10) : parseFloat(raw)
                    updateVllmConfig(row.path, Number.isNaN(n) ? 0 : n)
                  }}
                  sx={deployFieldSx}
                />
                <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('performance', row.meta).join(' — ')}</FormHelperText>
              </Grid>
            ))}
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem' }}>kv_cache_dtype</Typography>
                {vllmFieldMeta('performance', 'kv_cache_dtype') && <FrameworkBadge scope={vllmFieldMeta('performance', 'kv_cache_dtype')!.scope} />}
              </Box>
              <Select
                fullWidth
                size="small"
                value={vllmConfig.performance.kv_cache_dtype}
                onChange={(e) => updateVllmConfig('performance.kv_cache_dtype', e.target.value)}
                sx={deploySelectSx}
              >
                {(vllmFieldMeta('performance', 'kv_cache_dtype')?.options || ['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3']).map((o) => (
                  <MenuItem key={o} value={o}>{o}</MenuItem>
                ))}
              </Select>
              <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('performance', 'kv_cache_dtype').join(' — ')}</FormHelperText>
            </Grid>
            {([
              { path: 'performance.enforce_eager', label: 'enforce_eager', meta: 'enforce_eager' },
              { path: 'performance.enable_chunked_prefill', label: 'enable_chunked_prefill', meta: 'enable_chunked_prefill' },
              { path: 'performance.async_scheduling', label: 'async_scheduling', meta: 'async_scheduling' },
            ] as const).map((row) => (
              <Grid item xs={12} md={4} key={row.path}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                  <Typography sx={{ fontSize: '0.875rem' }}>{row.label}</Typography>
                  {vllmFieldMeta('performance', row.meta) && <FrameworkBadge scope={vllmFieldMeta('performance', row.meta)!.scope} />}
                </Box>
                <Select
                  fullWidth
                  size="small"
                  value={vllmConfig.performance[row.meta as keyof typeof vllmConfig.performance] ? 'true' : 'false'}
                  onChange={(e) => updateVllmConfig(row.path, e.target.value === 'true')}
                  sx={deploySelectSx}
                >
                  <MenuItem value="true">true</MenuItem>
                  <MenuItem value="false">false</MenuItem>
                </Select>
                <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('performance', row.meta).join(' — ')}</FormHelperText>
              </Grid>
            ))}
          </Grid>
        </SectionCard>
      )}

      {backend === 'llamacpp' && activeDeployTab === 2 && (
        <SectionCard
          title="Performance configuration"
          subtitle="Tune CPU threads, prompt batching, KV cache formats, GPU split behavior, and memory options."
          icon={<SpeedIcon />}
          accentColor={accentColors.success}
          sx={{ mb: 3 }}
          action={<StatusChip status="performance" statusConfig={DEPLOY_STATUS_CONFIG} />}
        >
          <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Threads"
                  type="number"
                  fullWidth
	                  value={config.performance.threads}
	                  onChange={(e) => updateConfig('performance.threads', parseInt(e.target.value))}
	                  helperText="Number of threads for computation (--threads)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Threads Batch"
                  type="number"
                  fullWidth
	                  value={config.performance.threads_batch || config.performance.threads}
	                  onChange={(e) => updateConfig('performance.threads_batch', parseInt(e.target.value))}
	                  helperText="Threads for batch processing (--threads-batch)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Batch Size"
                  type="number"
                  fullWidth
	                  value={config.performance.batch_size}
	                  onChange={(e) => updateConfig('performance.batch_size', parseInt(e.target.value))}
	                  helperText="Batch size for prompt processing (--batch-size)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="uBatch Size"
                  type="number"
                  fullWidth
	                  value={config.performance.ubatch_size}
	                  onChange={(e) => updateConfig('performance.ubatch_size', parseInt(e.target.value))}
	                  helperText="Micro batch size"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Num Keep"
                  type="number"
                  fullWidth
	                  value={config.performance.num_keep}
	                  onChange={(e) => updateConfig('performance.num_keep', parseInt(e.target.value))}
	                  helperText="Number of tokens to keep from initial prompt"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Max Tokens (n-predict)"
                  type="number"
                  fullWidth
	                  value={config.performance.num_predict}
	                  onChange={(e) => updateConfig('performance.num_predict', parseInt(e.target.value))}
	                  helperText="Maximum tokens to predict (--n-predict)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={3}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography sx={{ fontSize: '0.875rem' }}>Cache Type K</Typography>
	                </Box>
	                <Select
	                  fullWidth
	                  value={config.performance.cache_type_k || 'q4_0'}
	                  onChange={(e) => updateConfig('performance.cache_type_k', e.target.value)}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="f16">F16 (High Quality)</MenuItem>
                  <MenuItem value="q8_0">Q8_0 (8-bit)</MenuItem>
                  <MenuItem value="q4_0">Q4_0 (4-bit, Default)</MenuItem>

                </Select>
                <FormHelperText sx={{ mt: 0.5, fontSize: '0.75rem', color: 'text.secondary' }}>
                  KV cache data type for K (--cache-type-k)
                </FormHelperText>
              </Grid>
              <Grid item xs={12} md={3}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography sx={{ fontSize: '0.875rem' }}>Cache Type V</Typography>
	                </Box>
	                <Select
	                  fullWidth
	                  value={config.performance.cache_type_v || 'q4_0'}
	                  onChange={(e) => updateConfig('performance.cache_type_v', e.target.value)}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="f16">F16 (High Quality)</MenuItem>
                  <MenuItem value="q8_0">Q8_0 (8-bit)</MenuItem>
                  <MenuItem value="q4_0">Q4_0 (4-bit, Default)</MenuItem>

                </Select>
                <FormHelperText sx={{ mt: 0.5, fontSize: '0.75rem', color: 'text.secondary' }}>
                  KV cache data type for V (--cache-type-v)
                </FormHelperText>
              </Grid>

              <Grid item xs={12} md={3}>
                <TextField
                  label="Parallel Slots"
                  type="number"
                  fullWidth
	                  value={config.performance.parallel_slots || 1}
	                  onChange={(e) => updateConfig('performance.parallel_slots', parseInt(e.target.value))}
	                  helperText="Number of slots for processing requests (--parallel)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Context Checkpoints"
                  type="number"
                  fullWidth
                  value={config.performance.ctx_checkpoints ?? ''}
                  onChange={(e) => {
                    const val = e.target.value === '' ? '' : parseInt(e.target.value);
                    updateConfig('performance.ctx_checkpoints', val === '' ? undefined : val);
	                  }}
	                  helperText="Max context checkpoints per slot. 0 to disable (--ctx-checkpoints)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Split Mode</Typography>
                <Select
	                  fullWidth
	                  value={config.performance.split_mode || 'none'}
	                  onChange={(e) => updateConfig('performance.split_mode', e.target.value)}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="none">None (single GPU)</MenuItem>
                  <MenuItem value="layer">Layer (default)</MenuItem>
                  <MenuItem value="row">Row</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Tensor Split"
                  type="text"
                  fullWidth
	                  value={config.performance.tensor_split || ''}
	                  onChange={(e) => updateConfig('performance.tensor_split', e.target.value)}
	                  helperText="Tensor split across GPUs (e.g., '3,1')"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Main GPU"
                  type="number"
                  fullWidth
	                  value={config.performance.main_gpu || 0}
	                  onChange={(e) => updateConfig('performance.main_gpu', parseInt(e.target.value))}
	                  helperText="Main GPU index (--main-gpu)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600, mt: 2 }}>Memory Options</Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Memory F32</Typography>
                <Select
	                  fullWidth
	                  value={config.performance.memory_f32 ? 'true' : 'false'}
	                  onChange={(e) => updateConfig('performance.memory_f32', e.target.value === 'true')}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="false">Disabled (Default)</MenuItem>
                  <MenuItem value="true">Enabled</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Memory Lock (mlock)</Typography>
                <Select
	                  fullWidth
	                  value={config.performance.mlock ? 'true' : 'false'}
	                  onChange={(e) => updateConfig('performance.mlock', e.target.value === 'true')}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="false">Disabled (Default)</MenuItem>
                  <MenuItem value="true">Enabled</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>No Memory Map</Typography>
                <Select
	                  fullWidth
	                  value={config.performance.no_mmap ? 'true' : 'false'}
	                  onChange={(e) => updateConfig('performance.no_mmap', e.target.value === 'true')}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="false">Disabled (Default)</MenuItem>
                  <MenuItem value="true">Enabled</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Continuous Batching</Typography>
                <Select
	                  fullWidth
	                  value={config.performance.continuous_batching ? 'true' : 'false'}
	                  onChange={(e) => updateConfig('performance.continuous_batching', e.target.value === 'true')}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="false">Disabled (Default)</MenuItem>
                  <MenuItem value="true">Enabled</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>NUMA Optimization</Typography>
                <Select
	                  fullWidth
	                  value={config.performance.numa || ''}
	                  onChange={(e) => updateConfig('performance.numa', e.target.value)}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="">None (Default)</MenuItem>
                  <MenuItem value="distribute">Distribute</MenuItem>
                  <MenuItem value="isolate">Isolate</MenuItem>
                  <MenuItem value="numactl">Numactl</MenuItem>
                </Select>
	              </Grid>
	            </Grid>
        </SectionCard>
      )}

      {backend === 'vllm' && vllmConfig && activeDeployTab === 3 && (
        <SectionCard
          title="MoE and reasoning"
          subtitle="Configure vLLM expert kernels, cache dtype, parser selection, and optional parser plugin path."
          icon={<TuneIcon />}
          accentColor={accentColors.purple}
          sx={{ mb: 3, scrollMarginTop: '96px' }}
          action={<StatusChip status="reasoning" statusConfig={DEPLOY_STATUS_CONFIG} />}
        >
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem' }}>moe_backend</Typography>
                {vllmFieldMeta('moe', 'moe_backend') && <FrameworkBadge scope={vllmFieldMeta('moe', 'moe_backend')!.scope} />}
              </Box>
              <Select
                fullWidth
                size="small"
                value={vllmConfig.moe.moe_backend}
                onChange={(e) => updateVllmConfig('moe.moe_backend', e.target.value)}
                sx={deploySelectSx}
              >
                {(vllmFieldMeta('moe', 'moe_backend')?.options || ['marlin', 'triton']).map((o) => (
                  <MenuItem key={o} value={o}>{o}</MenuItem>
                ))}
              </Select>
              <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('moe', 'moe_backend').join(' — ')}</FormHelperText>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem' }}>mamba_ssm_cache_dtype</Typography>
                {vllmFieldMeta('moe', 'mamba_ssm_cache_dtype') && <FrameworkBadge scope={vllmFieldMeta('moe', 'mamba_ssm_cache_dtype')!.scope} />}
              </Box>
              <Select
                fullWidth
                size="small"
                value={vllmConfig.moe.mamba_ssm_cache_dtype}
                onChange={(e) => updateVllmConfig('moe.mamba_ssm_cache_dtype', e.target.value)}
                sx={deploySelectSx}
              >
                {(vllmFieldMeta('moe', 'mamba_ssm_cache_dtype')?.options || ['float16', 'float32', 'bfloat16']).map((o) => (
                  <MenuItem key={o} value={o}>{o}</MenuItem>
                ))}
              </Select>
              <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('moe', 'mamba_ssm_cache_dtype').join(' — ')}</FormHelperText>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem' }}>reasoning_parser</Typography>
                {vllmFieldMeta('reasoning', 'reasoning_parser') && <FrameworkBadge scope={vllmFieldMeta('reasoning', 'reasoning_parser')!.scope} />}
              </Box>
              <Select
                fullWidth
                size="small"
                value={vllmConfig.reasoning.reasoning_parser}
                onChange={(e) => updateVllmConfig('reasoning.reasoning_parser', e.target.value)}
                sx={deploySelectSx}
              >
                {(vllmFieldMeta('reasoning', 'reasoning_parser')?.options || ['nemotron_v3', 'deepseek_r1']).filter(Boolean).map((o) => (
                  <MenuItem key={o} value={o}>{o}</MenuItem>
                ))}
              </Select>
              <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('reasoning', 'reasoning_parser').join(' — ')}</FormHelperText>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem' }}>reasoning_parser_plugin (host path)</Typography>
                {vllmFieldMeta('reasoning', 'reasoning_parser_plugin') && <FrameworkBadge scope={vllmFieldMeta('reasoning', 'reasoning_parser_plugin')!.scope} />}
              </Box>
              <TextField
                fullWidth
                size="small"
                placeholder="/app/plugin.py"
                value={vllmConfig.reasoning.reasoning_parser_plugin}
                onChange={(e) => updateVllmConfig('reasoning.reasoning_parser_plugin', e.target.value)}
                sx={deployFieldSx}
              />
              <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('reasoning', 'reasoning_parser_plugin').join(' — ')}</FormHelperText>
            </Grid>
          </Grid>
        </SectionCard>
      )}

      {backend === 'llamacpp' && activeDeployTab === 3 && (
        <SectionCard
          title="Context extension"
          subtitle="Adjust YaRN scaling and group attention for longer-context llama.cpp runs."
          icon={<MemoryIcon />}
          accentColor={accentColors.info}
          sx={{ mb: 3 }}
          action={<StatusChip status="context" statusConfig={DEPLOY_STATUS_CONFIG} />}
        >
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 600, mt: 2 }}>YaRN Parameters</Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="YaRN Ext Factor"
                    type="number"
                    fullWidth
                    value={config.context_extension?.yarn_ext_factor || 1.0}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value)
                      const next = { ...(config.context_extension || {}), yarn_ext_factor: value }
                      updateConfig('context_extension', next)
                    }}
                    helperText="YaRN extrapolation mix factor (--yarn-ext-factor)"
                    sx={deployFieldSx}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="YaRN Attn Factor"
                    type="number"
                    fullWidth
                    value={config.context_extension?.yarn_attn_factor || 1.0}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value)
                      const next = { ...(config.context_extension || {}), yarn_attn_factor: value }
                      updateConfig('context_extension', next)
                    }}
                    helperText="YaRN attention scaling factor (--yarn-attn-factor)"
                    sx={deployFieldSx}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="YaRN Beta Slow"
                    type="number"
                    fullWidth
                    value={config.context_extension?.yarn_beta_slow || 1.0}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value)
                      const next = { ...(config.context_extension || {}), yarn_beta_slow: value }
                      updateConfig('context_extension', next)
                    }}
                    helperText="YaRN high correction dimension (--yarn-beta-slow)"
                    sx={deployFieldSx}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="YaRN Beta Fast"
                    type="number"
                    fullWidth
                    value={config.context_extension?.yarn_beta_fast || 32.0}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value)
                      const next = { ...(config.context_extension || {}), yarn_beta_fast: value }
                      updateConfig('context_extension', next)
                    }}
                    helperText="YaRN low correction dimension (--yarn-beta-fast)"
                    sx={deployFieldSx}
                  />
                </Grid>
              </Grid>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 600, mt: 2 }}>Group Attention</Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Group Attention Factor"
                    type="number"
                    fullWidth
                    value={config.context_extension?.group_attn_n || 1}
                    onChange={(e) => {
                      const value = parseInt(e.target.value)
                      const next = { ...(config.context_extension || {}), group_attn_n: value }
                      updateConfig('context_extension', next)
                    }}
                    helperText="Group attention factor (--grp-attn-n)"
                    sx={deployFieldSx}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Group Attention Width"
                    type="number"
                    fullWidth
                    value={config.context_extension?.group_attn_w || 512}
                    onChange={(e) => {
                      const value = parseInt(e.target.value)
                      const next = { ...(config.context_extension || {}), group_attn_w: value }
                      updateConfig('context_extension', next)
                    }}
                    helperText="Group attention width (--grp-attn-w)"
                    sx={deployFieldSx}
                  />
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </SectionCard>
      )}

      {backend === 'vllm' && vllmConfig && activeDeployTab === 4 && (
        <SectionCard
          title="Tools and speculative decoding"
          subtitle="Configure tool-call parsing, speculative execution, and multimodal media defaults."
          icon={<ChatIcon />}
          accentColor={accentColors.warning}
          sx={{ mb: 3, scrollMarginTop: '96px' }}
          action={<StatusChip status="tools" statusConfig={DEPLOY_STATUS_CONFIG} />}
        >
          <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                  <Typography sx={{ fontSize: '0.875rem' }}>enable_auto_tool_choice</Typography>
                  {vllmFieldMeta('tools', 'enable_auto_tool_choice') && <FrameworkBadge scope={vllmFieldMeta('tools', 'enable_auto_tool_choice')!.scope} />}
                </Box>
                <Select fullWidth size="small" value={vllmConfig.tools.enable_auto_tool_choice ? 'true' : 'false'}
                  onChange={(e) => updateVllmConfig('tools.enable_auto_tool_choice', e.target.value === 'true')}
                  sx={deploySelectSx}>
                  <MenuItem value="true">true</MenuItem>
                  <MenuItem value="false">false</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                  <Typography sx={{ fontSize: '0.875rem' }}>tool_call_parser</Typography>
                  {vllmFieldMeta('tools', 'tool_call_parser') && <FrameworkBadge scope={vllmFieldMeta('tools', 'tool_call_parser')!.scope} />}
                </Box>
                <Select fullWidth size="small" value={vllmConfig.tools.tool_call_parser}
                  onChange={(e) => updateVllmConfig('tools.tool_call_parser', e.target.value)}
                  sx={deploySelectSx}>
                  {(vllmFieldMeta('tools', 'tool_call_parser')?.options || ['qwen3_coder', 'hermes', 'mistral', 'internlm', 'llama3_json']).filter((x) => x !== '').map((o) => (
                    <MenuItem key={o} value={o}>{o}</MenuItem>
                  ))}
                </Select>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2" sx={{ mt: 1, mb: 1 }}>Speculative config (leave method empty to disable)</Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography sx={{ fontSize: '0.875rem', mb: 0.5 }}>speculative method</Typography>
                <Select fullWidth size="small" value={vllmConfig.speculative.method || ''}
                  onChange={(e) => updateVllmConfig('speculative.method', e.target.value)}
                  sx={deploySelectSx}>
                  <MenuItem value=""><em>Disabled</em></MenuItem>
                  {(vllmFieldMeta('speculative', 'method')?.options || ['mtp', 'ngram', 'eagle', 'medusa']).filter(Boolean).map((o) => (
                    <MenuItem key={o} value={o}>{o}</MenuItem>
                  ))}
                </Select>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography sx={{ fontSize: '0.875rem', mb: 0.5 }}>num_speculative_tokens</Typography>
                <TextField fullWidth size="small" type="number"
                  value={vllmConfig.speculative.num_speculative_tokens}
                  onChange={(e) => updateVllmConfig('speculative.num_speculative_tokens', parseInt(e.target.value, 10) || 0)}
                  sx={deployFieldSx} />
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography sx={{ fontSize: '0.875rem', mb: 0.5 }}>speculative moe_backend (JSON field)</Typography>
                <TextField fullWidth size="small" placeholder="triton"
                  value={vllmConfig.speculative.speculative_moe_backend}
                  onChange={(e) => updateVllmConfig('speculative.speculative_moe_backend', e.target.value)}
                  sx={deployFieldSx} />
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2" sx={{ mt: 1, mb: 1 }}>Media (Omni / video)</Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography sx={{ fontSize: '0.875rem', mb: 0.5 }}>video_pruning_rate</Typography>
                <TextField fullWidth size="small" type="number"
                  value={vllmConfig.media.video_pruning_rate}
                  onChange={(e) => updateVllmConfig('media.video_pruning_rate', parseFloat(e.target.value) || 0)}
                  sx={deployFieldSx} />
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography sx={{ fontSize: '0.875rem', mb: 0.5 }}>video_fps</Typography>
                <TextField fullWidth size="small" type="number"
                  value={vllmConfig.media.video_fps}
                  onChange={(e) => updateVllmConfig('media.video_fps', parseInt(e.target.value, 10) || 0)}
                  sx={deployFieldSx} />
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography sx={{ fontSize: '0.875rem', mb: 0.5 }}>video_num_frames</Typography>
                <TextField fullWidth size="small" type="number"
                  value={vllmConfig.media.video_num_frames}
                  onChange={(e) => updateVllmConfig('media.video_num_frames', parseInt(e.target.value, 10) || 0)}
                  sx={deployFieldSx} />
              </Grid>
          </Grid>
        </SectionCard>
      )}

      {backend === 'llamacpp' && activeDeployTab === 4 && (
        <SectionCard
          title="Server configuration"
          subtitle="Configure listener, API keys, observability, prompt caching, and reasoning extraction."
          icon={<EndpointIcon />}
          accentColor={accentColors.info}
          sx={{ mb: 3 }}
          action={
            <StatusChip
              status={selectedApiKey ? 'secured' : 'open'}
              statusConfig={DEPLOY_STATUS_CONFIG}
            />
          }
        >
          <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Host"
                  type="text"
                  fullWidth
	                  value={config.server.host}
	                  onChange={(e) => updateConfig('server.host', e.target.value)}
	                  helperText="IP address to listen on (--host)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Port"
                  type="number"
                  fullWidth
	                  value={config.server.port}
	                  onChange={(e) => updateConfig('server.port', parseInt(e.target.value))}
	                  helperText="Port to listen on (--port)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600, mt: 2 }}>API Key Management</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Select API Key</Typography>
                    <Select
                      fullWidth
                      value={selectedApiKey}
                      onChange={(e) => {
                        const newKey = e.target.value as string
                        handleApiKeyChange(newKey)
                        updateConfig('server.api_key', newKey)
                      }}
	                      sx={deploySelectSx}
	                    >
                      <MenuItem value="">
                        <em>No API key (open access)</em>
                      </MenuItem>
                      {availableApiKeys.map((key) => (
                        <MenuItem key={key} value={key}>
                          {key.length > 20 ? `${key.substring(0, 20)}...` : key}
                        </MenuItem>
                      ))}
                    </Select>
                    <FormHelperText sx={{ mt: 0.5, fontSize: '0.75rem', color: 'text.secondary' }}>
                      Choose an API key for authentication. Leave empty for open access.
                    </FormHelperText>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Add New API Key</Typography>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <TextField
                        placeholder="Enter new API key"
                        value={newApiKey}
	                        onChange={(e) => setNewApiKey(e.target.value)}
	                        onKeyPress={(e) => e.key === 'Enter' && handleAddApiKey()}
	                        sx={{
	                          flex: 1,
	                          '& .MuiOutlinedInput-root': {
	                            fontSize: '0.875rem',
	                            borderRadius: '8px',
	                            backgroundColor: 'background.default',
	                          }
	                        }}
                      />
                      <Button
                        variant="outlined"
                        onClick={handleAddApiKey}
	                        disabled={!newApiKey.trim()}
	                        sx={{
	                          borderRadius: '8px',
	                          fontSize: '0.8125rem',
	                          textTransform: 'none',
	                          borderColor: 'divider',
	                        }}
                      >
                        Add
                      </Button>
                    </Box>
                    <FormHelperText sx={{ mt: 0.5, fontSize: '0.75rem', color: 'text.secondary' }}>
                      Add a new API key to the list of available keys.
                    </FormHelperText>
                  </Grid>
                  {selectedApiKey && (
                    <Grid item xs={12}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <StatusChip
                          status="key"
                          label={`Current: ${selectedApiKey.length > 30 ? `${selectedApiKey.substring(0, 30)}...` : selectedApiKey}`}
                          variant="outlined"
                          statusConfig={DEPLOY_STATUS_CONFIG}
                          sx={deployActionChipSx('key')}
                        />
                        <Button
                          size="small"
                          color="error"
                          variant="outlined"
	                          onClick={() => handleRemoveApiKey(selectedApiKey)}
	                          sx={{
	                            borderRadius: '8px',
	                            fontSize: '0.75rem',
	                            textTransform: 'none',
	                            minWidth: 'auto',
                            px: 1
                          }}
                        >
                          Remove
                        </Button>
                      </Box>
                    </Grid>
                  )}
                </Grid>
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Timeout"
                  type="number"
                  fullWidth
	                  value={config.server.timeout || 600}
	                  onChange={(e) => updateConfig('server.timeout', parseInt(e.target.value))}
	                  helperText="Server read/write timeout in seconds (--timeout)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="System Prompt File"
                  type="text"
                  fullWidth
	                  value={config.server.system_prompt_file || ''}
	                  onChange={(e) => updateConfig('server.system_prompt_file', e.target.value)}
	                  helperText="File to load system prompt (--system-prompt-file)"
	                  sx={deployFieldSx}
	                />
              </Grid>
              <Grid item xs={6} md={2}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Embedding</Typography>
                <Select
	                  fullWidth
	                  value={config.server.embedding ? 'true' : 'false'}
	                  onChange={(e) => updateConfig('server.embedding', e.target.value === 'true')}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="false">Disabled (Default)</MenuItem>
                  <MenuItem value="true">Enabled</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={6} md={2}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Metrics</Typography>
                <Select
	                  fullWidth
	                  value={config.server.metrics ? 'true' : 'false'}
	                  onChange={(e) => updateConfig('server.metrics', e.target.value === 'true')}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="false">Disabled (Default)</MenuItem>
                  <MenuItem value="true">Enabled</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Log Format</Typography>
                <Select
	                  fullWidth
	                  value={config.server.log_format || 'json'}
	                  onChange={(e) => updateConfig('server.log_format', e.target.value)}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="json">JSON (Default)</MenuItem>
                  <MenuItem value="text">Text</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Disable Logging</Typography>
                <Select
	                  fullWidth
	                  value={config.server.log_disable ? 'true' : 'false'}
	                  onChange={(e) => updateConfig('server.log_disable', e.target.value === 'true')}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="false">Enabled (Default)</MenuItem>
                  <MenuItem value="true">Disabled</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Disable Slots Endpoint</Typography>
                <Select
	                  fullWidth
	                  value={config.server.slots_endpoint_disable ? 'true' : 'false'}
	                  onChange={(e) => updateConfig('server.slots_endpoint_disable', e.target.value === 'true')}
	                  sx={deploySelectSx}
	                >
                  <MenuItem value="false">Enabled (Default)</MenuItem>
                  <MenuItem value="true">Disabled</MenuItem>
                </Select>
              </Grid>

              {/* Advanced Server Options - Collapsible */}
              <Grid item xs={12}>
                <Accordion
	                  disableGutters
	                  sx={{
	                    bgcolor: 'action.hover',
	                    border: '1px solid',
	                    borderColor: 'divider',
	                    borderRadius: '8px !important',
	                    '&:before': { display: 'none' },
	                    boxShadow: 'none',
	                    mt: 2,
                  }}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Prompt Caching & Reasoning</Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 1, mt: 0.3 }}>(advanced)</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Prompt Caching"
                          description="Cache processed prompts to speed up repeated or similar requests. Particularly effective for multi-turn conversations. (--cache-prompt / --no-cache-prompt)"
                          path="server.cache_prompt"
                          value={config.server.cache_prompt === undefined ? '' : config.server.cache_prompt ? 'true' : 'false'}
                          defaultValue={'true'}
                          type="select"
                          options={[
                            { value: 'true', label: 'Enabled' },
                            { value: 'false', label: 'Disabled' },
                          ]}
                          onChange={(path, value) => updateConfig(path, value === '' ? undefined : value === 'true')}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Cache Reuse"
                          description="Number of tokens from the cached prompt to reuse. Allows reusing cached computation for partially matching prompts. (--cache-reuse)"
                          path="server.cache_reuse"
                          value={config.server.cache_reuse ?? ''}
                          defaultValue={getDefaultValue('server.cache_reuse')}
                          type="number"
                          min={0}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Sleep on Idle (seconds)"
                          description="Automatically unload the model from VRAM after N seconds of inactivity. 0 = disabled. Useful for shared GPU environments. (--sleep-idle-seconds)"
                          path="server.sleep_idle_seconds"
                          value={config.server.sleep_idle_seconds ?? ''}
                          defaultValue={getDefaultValue('server.sleep_idle_seconds')}
                          type="number"
                          min={0}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Reasoning Format"
                          description="Format for extracting reasoning/thinking content from model output. 'deepseek' extracts <think> tags into reasoning_content. 'none' treats everything as content. (--reasoning-format)"
                          path="server.reasoning_format"
                          value={config.server.reasoning_format ?? ''}
                          defaultValue={getDefaultValue('server.reasoning_format')}
                          type="select"
                          options={[
                            { value: 'deepseek', label: 'DeepSeek (<think> tags)' },
                            { value: 'none', label: 'None (no extraction)' },
                          ]}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Reasoning Budget"
                          description="Maximum number of thinking/reasoning tokens the model can generate. -1 = unlimited. (--reasoning-budget)"
                          path="server.reasoning_budget"
                          value={config.server.reasoning_budget ?? ''}
                          defaultValue={getDefaultValue('server.reasoning_budget')}
                          type="number"
                          min={-1}
                          onChange={updateConfig}
                          onReset={resetToDefault}
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <ParameterField
                          label="Jinja Templates"
                          description="Enable Jinja template processing. For GGUF embedded templates, use Chat Template → “GGUF embedded + Jinja” instead. Custom .jinja files enable this automatically. (--jinja / --no-jinja)"
                          path="server.jinja"
                          value={config.server.jinja === undefined ? '' : config.server.jinja ? 'true' : 'false'}
                          defaultValue={'true'}
                          type="select"
                          options={[
                            { value: 'true', label: 'Enabled' },
                            { value: 'false', label: 'Disabled' },
                          ]}
                          onChange={(path, value) => updateConfig(path, value === '' ? undefined : value === 'true')}
                          onReset={resetToDefault}
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
	              </Grid>
	            </Grid>
        </SectionCard>
      )}

      {backend === 'vllm' && vllmConfig && activeDeployTab === 5 && (
        <SectionCard
          title="Docker-style environment"
          subtitle="Common container variables for vLLM launch scripts and compose files."
          icon={<DeployIcon />}
          accentColor={accentColors.primary}
          action={<StatusChip status="environment" statusConfig={DEPLOY_STATUS_CONFIG} />}
          sx={{ mb: 3, scrollMarginTop: '96px' }}
        >
          <Grid container spacing={2}>
            {([
              { path: 'environment.vllm_nvfp4_gemm_backend', label: 'VLLM_NVFP4_GEMM_BACKEND', metaKey: 'vllm_nvfp4_gemm_backend' },
              { path: 'environment.vllm_allow_long_max_model_len', label: 'VLLM_ALLOW_LONG_MAX_MODEL_LEN', metaKey: 'vllm_allow_long_max_model_len' },
              { path: 'environment.vllm_flashinfer_allreduce_backend', label: 'VLLM_FLASHINFER_ALLREDUCE_BACKEND', metaKey: 'vllm_flashinfer_allreduce_backend' },
              { path: 'environment.vllm_use_flashinfer_moe_fp4', label: 'VLLM_USE_FLASHINFER_MOE_FP4', metaKey: 'vllm_use_flashinfer_moe_fp4' },
            ] as const).map((row) => {
              const meta = vllmFieldMeta('environment', row.metaKey)
              const opts = meta?.options
              const val = (vllmConfig.environment as any)[row.metaKey] as string
              return (
                <Grid item xs={12} md={6} key={row.path}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Typography sx={{ fontSize: '0.875rem', fontWeight: 600 }}>{row.label}</Typography>
                    {meta && <FrameworkBadge scope={meta.scope} />}
                  </Box>
                  {opts ? (
                    <Select
                      fullWidth
                      size="small"
                      value={val}
                      sx={deploySelectSx}
                      onChange={(e) => updateVllmConfig(row.path, e.target.value)}
                    >
                      {opts.map((o) => (
                        <MenuItem key={o} value={o}>{o}</MenuItem>
                      ))}
                    </Select>
                  ) : (
                    <TextField
                      fullWidth
                      size="small"
                      value={val}
                      sx={deployFieldSx}
                      onChange={(e) => updateVllmConfig(row.path, e.target.value)}
                    />
                  )}
                  <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('environment', row.metaKey).join(' — ')}</FormHelperText>
                </Grid>
              )
            })}
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem', fontWeight: 600 }}>HF_TOKEN</Typography>
                {vllmFieldMeta('environment', 'hf_token') && <FrameworkBadge scope={vllmFieldMeta('environment', 'hf_token')!.scope} />}
              </Box>
              <TextField
                fullWidth
                size="small"
                type="password"
                value={vllmConfig.environment.hf_token}
                sx={deployFieldSx}
                onChange={(e) => updateVllmConfig('environment.hf_token', e.target.value)}
              />
              <FormHelperText sx={{ fontSize: '0.75rem' }}>{vllmHintLines('environment', 'hf_token').join(' — ')}</FormHelperText>
            </Grid>
          </Grid>
        </SectionCard>
      )}

      {backend === 'llamacpp' && activeDeployTab === 5 && (
        <Box sx={{ mb: 3 }}>
          <LlamaCppCommitSelector
            onCommitChanged={(commit) => {
              debugLog('deploy', 'LlamaCPP commit changed', commit);
              // Optionally refresh configuration or show notification
            }}
          />
        </Box>
      )}

      {backend === 'vllm' && vllmConfig && activeDeployTab === 6 && (
        <Box sx={{ mb: 3 }}>
          <LlamaCppCommitSelector
            variant="vllm"
            onCommitChanged={(tag) => {
              debugLog('deploy', 'vLLM base image tag changed', tag)
            }}
          />
        </Box>
      )}

      {backend === 'vllm' && vllmConfig && activeDeployTab === 7 && (
        <SectionCard
          title="vLLM server"
          subtitle="Endpoint binding, API access, and trust boundary settings."
          icon={<EndpointIcon />}
          accentColor={accentColors.info}
          action={<StatusChip status={vllmConfig.server.api_key ? 'secured' : 'open'} statusConfig={DEPLOY_STATUS_CONFIG} />}
          sx={{ mb: 3, scrollMarginTop: '96px' }}
        >
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem', fontWeight: 600 }}>Host</Typography>
                {vllmFieldMeta('server', 'host') && <FrameworkBadge scope={vllmFieldMeta('server', 'host')!.scope} />}
              </Box>
              <TextField
                fullWidth
                size="small"
                value={vllmConfig.server.host}
                sx={deployFieldSx}
                onChange={(e) => updateVllmConfig('server.host', e.target.value)}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem', fontWeight: 600 }}>Port</Typography>
                {vllmFieldMeta('server', 'port') && <FrameworkBadge scope={vllmFieldMeta('server', 'port')!.scope} />}
              </Box>
              <TextField
                fullWidth
                size="small"
                type="number"
                value={vllmConfig.server.port}
                sx={deployFieldSx}
                onChange={(e) => updateVllmConfig('server.port', parseInt(e.target.value, 10) || 0)}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem', fontWeight: 600 }}>API key</Typography>
                {vllmFieldMeta('server', 'api_key') && <FrameworkBadge scope={vllmFieldMeta('server', 'api_key')!.scope} />}
              </Box>
              <TextField
                fullWidth
                size="small"
                value={vllmConfig.server.api_key}
                sx={deployFieldSx}
                onChange={(e) => {
                  updateVllmConfig('server.api_key', e.target.value)
                  handleApiKeyChange(e.target.value)
                }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography sx={{ fontSize: '0.875rem', fontWeight: 600 }}>trust_remote_code</Typography>
                {vllmFieldMeta('server', 'trust_remote_code') && <FrameworkBadge scope={vllmFieldMeta('server', 'trust_remote_code')!.scope} />}
              </Box>
              <Select
                fullWidth
                size="small"
                value={vllmConfig.server.trust_remote_code ? 'true' : 'false'}
                sx={deploySelectSx}
                onChange={(e) => updateVllmConfig('server.trust_remote_code', e.target.value === 'true')}
              >
                <MenuItem value="true">true</MenuItem>
                <MenuItem value="false">false</MenuItem>
              </Select>
            </Grid>
          </Grid>
        </SectionCard>
      )}

      {backend === 'llamacpp' && activeDeployTab === 6 && (
        <SectionCard
          title="Generated command"
          subtitle="Launch arguments generated from the saved llama.cpp configuration."
          icon={<TerminalIcon />}
          accentColor={accentColors.warning}
          action={<StatusChip status="command" statusConfig={DEPLOY_STATUS_CONFIG} />}
        >
          <Paper sx={deployCommandPreviewSx}>
            {commandLine || 'Save configuration to refresh command preview.'}
          </Paper>
        </SectionCard>
      )}

      {backend === 'vllm' && vllmConfig && activeDeployTab === 8 && (
        <SectionCard
          title="vLLM launch command"
          subtitle="CLI flags from the saved vLLM config; container env wiring stays in deployment scripts."
          icon={<TerminalIcon />}
          accentColor={accentColors.warning}
          action={<StatusChip status="command" statusConfig={DEPLOY_STATUS_CONFIG} />}
          sx={{ scrollMarginTop: '96px' }}
        >
          <Paper sx={deployCommandPreviewSx}>
            {commandLine || 'Adjust settings to refresh command preview.'}
          </Paper>
        </SectionCard>
      )}

      {/* Container Logs Section */}
      {currentModel && currentModel.status === 'loaded' && !isNarrow && (
        <SectionCard
          key={actionLoading ? 'runtime-logs-active' : 'runtime-logs-idle'}
          title="Logs"
          icon={<MonitoringIcon />}
          accentColor={accentColors.info}
          action={<StatusChip status="logs" label={backend === 'vllm' ? 'vLLM API' : 'llama.cpp API'} statusConfig={DEPLOY_STATUS_CONFIG} />}
          collapsible
          defaultExpanded={Boolean(actionLoading)}
          sx={{
            mt: 1,
            minWidth: 0,
            '&[data-expanded="false"] .MuiCardContent-root': {
              px: 1.25,
              py: 1,
              '&:last-child': {
                pb: 1,
              },
            },
            '&[data-expanded="false"] .MuiTypography-h6': {
              fontSize: '0.875rem',
            },
          }}
        >
          {backend === 'llamacpp' && config?.mtp?.enabled && (
            <MtpStatsPanel
              stats={mtpStats}
              connected={mtpStatsConnected}
              compact
            />
          )}
          <LogViewer
            ref={logViewerRef}
            containerName={backend === 'vllm' ? 'vllm-api' : 'llamacpp-api'}
            backend={backend}
            maxLines={100}
            height={72}
            compact
            collapseEmpty
          />
        </SectionCard>
      )}

      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess(null)}
        message={success || ''}
        sx={{
          '& .MuiSnackbarContent-root': {
            backgroundColor: 'success.dark',
            color: 'success.contrastText',
            fontSize: '0.8125rem',
            borderRadius: '8px'
          }
        }}
      />
    </PageShell>
  )
}

export default DeployPage
