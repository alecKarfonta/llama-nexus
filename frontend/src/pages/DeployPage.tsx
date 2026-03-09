import React, { useEffect, useState, useMemo, useRef } from 'react'
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  CardHeader,
  TextField,
  Slider,
  Select,
  MenuItem,
  Button,
  Alert,
  Chip,
  Paper,
  Tabs,
  Tab,
  CircularProgress,
  Snackbar,
  IconButton,
  Tooltip,
  FormHelperText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material'
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
} from '@mui/icons-material'
import { apiService } from '@/services/api'
import type { ModelInfo } from '@/types/api'
import LlamaCppCommitSelector from '@/components/LlamaCppCommitSelector'
import LogViewer, { LogViewerRef } from '@/components/LogViewer'
import { settingsManager } from '@/utils/settings'

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
    cache_type_k?: string;
    cache_type_v?: string;
  };
  speculative?: {
    model_draft?: string;
    gpu_layers_draft?: number;
    ctx_size_draft?: number;
    draft_max?: number;
    draft_min?: number;
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
    cache_reuse?: number;
    reasoning_format?: 'deepseek' | 'none';
    reasoning_budget?: number;
    jinja?: boolean;
    sleep_idle_seconds?: number;
  };
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

// Parameter Presets for different use cases
interface ParameterPreset {
  name: string;
  description: string;
  category: 'coding' | 'creative' | 'balanced' | 'precise';
  sampling: Partial<Config['sampling']>;
  performance?: Partial<Config['performance']>;
}

const PARAMETER_PRESETS: ParameterPreset[] = [
  {
    name: 'Balanced',
    description: 'Good general-purpose settings for most tasks',
    category: 'balanced',
    sampling: {
      temperature: 0.7,
      top_p: 0.8,
      top_k: 20,
      min_p: 0.03,
      repeat_penalty: 1.05,
      frequency_penalty: 0.3,
      presence_penalty: 0.2,
    },
  },
  {
    name: 'Coding',
    description: 'Optimized for code generation with higher precision',
    category: 'coding',
    sampling: {
      temperature: 0.2,
      top_p: 0.95,
      top_k: 40,
      min_p: 0.05,
      repeat_penalty: 1.1,
      frequency_penalty: 0.1,
      presence_penalty: 0.0,
    },
  },
  {
    name: 'Creative',
    description: 'Higher creativity for storytelling and brainstorming',
    category: 'creative',
    sampling: {
      temperature: 1.0,
      top_p: 0.9,
      top_k: 50,
      min_p: 0.02,
      repeat_penalty: 1.15,
      frequency_penalty: 0.5,
      presence_penalty: 0.5,
    },
  },
  {
    name: 'Precise',
    description: 'Maximum determinism for factual and analytical tasks',
    category: 'precise',
    sampling: {
      temperature: 0.1,
      top_p: 0.5,
      top_k: 10,
      min_p: 0.1,
      repeat_penalty: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
    },
  },
];

// LocalStorage key for persisting deployment settings
const DEPLOY_SETTINGS_KEY = 'llama-nexus-deploy-settings';

// Helper functions for localStorage persistence
const saveDeploySettings = (config: Config, selectedApiKey: string) => {
  try {
    const settingsToSave = {
      config,
      selectedApiKey,
      timestamp: Date.now()
    };
    localStorage.setItem(DEPLOY_SETTINGS_KEY, JSON.stringify(settingsToSave));
  } catch (error) {
    console.warn('Failed to save deploy settings to localStorage:', error);
  }
};

const loadDeploySettings = (): { config: Config | null; selectedApiKey: string } => {
  try {
    const stored = localStorage.getItem(DEPLOY_SETTINGS_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return {
        config: parsed.config || null,
        selectedApiKey: parsed.selectedApiKey || ''
      };
    }
  } catch (error) {
    console.warn('Failed to load deploy settings from localStorage:', error);
  }
  return { config: null, selectedApiKey: '' };
};

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
    console.warn('Failed to load API keys from localStorage:', error);
  }
  return [];
};

const saveApiKeys = (keys: string[]) => {
  try {
    localStorage.setItem(API_KEYS_STORAGE_KEY, JSON.stringify(keys));
  } catch (error) {
    console.warn('Failed to save API keys to localStorage:', error);
  }
};

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
  onChange,
  onReset,
}) => {
  const isEmpty = value === '' || value === null || value === undefined;
  const isDefault = value === defaultValue || (isEmpty && defaultValue === '');

  return (
    <Box sx={{ position: 'relative' }}>
      {type === 'select' ? (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Typography gutterBottom sx={{ fontSize: '0.875rem', mb: 0 }}>{label}</Typography>
            <Tooltip title={isEmpty ? `Set to default (${defaultValue})` : `Clear (use llama-server default)`}>
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
            fullWidth
            value={value || ''}
            onChange={(e) => onChange(path, e.target.value === '' ? undefined : e.target.value)}
            sx={{
              fontSize: '0.875rem',
              '& .MuiOutlinedInput-root': {
                borderRadius: 1,
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
        <TextField
          label={label}
          type={type}
          fullWidth
          value={value ?? ''}
          placeholder={`Default: ${typeof defaultValue === 'boolean' ? (defaultValue ? 'Enabled' : 'Disabled') : defaultValue}`}
          onChange={(e) => {
            const inputValue = e.target.value;
            if (inputValue === '') {
              onChange(path, undefined);
            } else if (type === 'number') {
              const numValue = step && step < 1 ? parseFloat(inputValue) : parseInt(inputValue);
              onChange(path, isNaN(numValue) ? undefined : numValue);
            } else {
              onChange(path, inputValue);
            }
          }}
          inputProps={{ min, max, step }}
          InputProps={{
            endAdornment: (
              <Tooltip title={isEmpty ? `Set to default (${defaultValue})` : `Clear (use llama-server default)`}>
                <IconButton
                  size="small"
                  onClick={() => onChange(path, undefined)}
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
              borderRadius: 1,
              backgroundColor: isEmpty ? 'rgba(255, 255, 255, 0.02)' : 'background.default',
              '&.Mui-focused': {
                borderColor: 'primary.main'
              }
            },
            '& .MuiInputLabel-root': {
              fontSize: '0.875rem'
            }
          }}
        />
      )}
      <FormHelperText sx={{ mt: 0.5, fontSize: '0.75rem', color: 'text.secondary' }}>
        {description} {isEmpty ? '(Using llama-server default)' : `(llama-server default: ${typeof defaultValue === 'boolean' ? (defaultValue ? 'Enabled' : 'Disabled') : defaultValue})`}
      </FormHelperText>
    </Box>
  );
};

// Logging utility for DeployPage debugging
const deployLog = (context: string, message: string, data?: any) => {
  const timestamp = new Date().toISOString().split('T')[1].slice(0, 12)
  const prefix = `[Deploy ${timestamp}] [${context}]`
  if (data !== undefined) {
    console.log(prefix, message, data)
  } else {
    console.log(prefix, message)
  }
}

export const DeployPage: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [config, setConfig] = useState<Config | null>(null)
  const [originalConfig, setOriginalConfig] = useState<Config | null>(null)
  const [commandLine, setCommandLine] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [validating, setValidating] = useState(false)
  const [actionLoading, setActionLoading] = useState<'start' | 'stop' | 'restart' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [tab, setTab] = useState(0)
  const [validateErrors, setValidateErrors] = useState<string[] | null>(null)
  const [validateWarnings, setValidateWarnings] = useState<string[] | null>(null)
  const [currentModel, setCurrentModel] = useState<any | null>(null)
  const [templates, setTemplates] = useState<string[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState<string>('')
  const [templatesDir, setTemplatesDir] = useState<string>('')
  const [templatesLoading, setTemplatesLoading] = useState<boolean>(false)

  // API Key management
  const [selectedApiKey, setSelectedApiKey] = useState<string>('')
  const [availableApiKeys, setAvailableApiKeys] = useState<string[]>([])
  const [newApiKey, setNewApiKey] = useState<string>('')

  // Ref for LogViewer to control logs
  const logViewerRef = useRef<LogViewerRef>(null)

  // Parameter preset selection
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)

  // Log component mount
  useEffect(() => {
    deployLog('mount', 'DeployPage component mounted')
    return () => {
      deployLog('mount', 'DeployPage component unmounted')
    }
  }, [])

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

  // Apply a parameter preset
  const applyPreset = (preset: ParameterPreset) => {
    deployLog('preset', `Applying preset: ${preset.name}`, { sampling: preset.sampling })
    if (!config) {
      deployLog('preset', 'ABORT: config is null')
      return;
    }

    setConfig({
      ...config,
      sampling: {
        ...config.sampling,
        ...preset.sampling,
      },
      ...(preset.performance && {
        performance: {
          ...config.performance,
          ...preset.performance,
        },
      }),
    });
    setSelectedPreset(preset.name);
    deployLog('preset', 'Preset applied successfully')
  }

  // Fetch VRAM estimation when config changes
  const fetchVramEstimate = async () => {
    if (!config?.model?.name) return;

    setVramLoading(true);
    try {
      const modelName = config.model.variant
        ? `${config.model.name}-${config.model.variant}`
        : config.model.name;

      const response = await fetch('/api/v1/vram/estimate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          context_size: config.model?.context_size || 4096,
          batch_size: config.performance?.batch_size || 1,
          gpu_layers: config.model?.gpu_layers ?? -1,
          available_vram_gb: 24, // Default, could be detected
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setVramEstimate(data);
      }
    } catch (err) {
      console.error('Failed to fetch VRAM estimate:', err);
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
    config?.performance?.contextSize,
    config?.performance?.batchSize,
    config?.performance?.gpuLayers,
    config?.performance?.flashAttention,
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
        try {
          deployLog('init', 'Fetching available models...')
          const list = await apiService.getModels()
          deployLog('init', 'Loaded models:', { count: list.length, models: list.map(m => `${m.name}/${m.variant}`) })
          setModels(list)
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
        try {
          deployLog('init', 'Fetching mmproj files...')
          setMmprojLoading(true)
          const mmRes = await fetch('/v1/models/mmproj-files')
          if (mmRes.ok) {
            const mmData = await mmRes.json()
            if (mmData.success && mmData.data?.files) {
              setMmprojFiles(mmData.data.files)
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

        // Use persisted config if available, otherwise use server config
        const configToUse = persistedSettings.config || cfgJson.config
        deployLog('init', 'Using config:', { source: persistedSettings.config ? 'persisted' : 'server', config: configToUse })

        // Ensure execution settings exist with defaults
        if (!configToUse.execution) {
          configToUse.execution = { mode: 'gpu', cuda_devices: 'all' }
        }

        setConfig(configToUse)
        setOriginalConfig(JSON.parse(JSON.stringify(cfgJson.config)))
        setCommandLine(cfgJson.command || '')

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

  const hasChanges = useMemo(
    () => JSON.stringify(config) !== JSON.stringify(originalConfig),
    [config, originalConfig]
  )

  // Function to update command line preview in real-time
  const updateCommandPreview = async (configToPreview: Config) => {
    deployLog('commandPreview', 'Updating command preview', { modelName: configToPreview?.model?.name })
    try {
      // Convert undefined values to null so they get properly serialized and handled by backend
      const configForPreview = JSON.parse(JSON.stringify(configToPreview, (key, value) => {
        return value === undefined ? null : value
      }))

      // Send the config to backend to get command preview without saving
      deployLog('commandPreview', 'Sending preview request to backend')
      const response = await fetch('/api/v1/service/config/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: configForPreview })
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

  const updateConfig = (path: string, value: any) => {
    console.log('🔄 updateConfig CALLED:', { path, value })
    deployLog('updateConfig', `Updating config path: ${path}`, { value, currentConfig: config })
    if (!config) {
      console.error('❌ ABORT: config is null')
      deployLog('updateConfig', 'ABORT: config is null')
      return
    }
    const next: Config = JSON.parse(JSON.stringify(config))
    const keys = path.split('.')
    let ref: any = next
    for (let i = 0; i < keys.length - 1; i++) ref = ref[keys[i]]
    const oldValue = ref[keys[keys.length - 1]]
    ref[keys[keys.length - 1]] = value
    console.log('✅ Config value changed:', { path, oldValue, newValue: value })
    deployLog('updateConfig', `Config updated: ${path}`, { oldValue, newValue: value })
    console.log('📝 Calling setConfig with new config:', next)
    setConfig(next)

    // Save to localStorage for persistence
    deployLog('updateConfig', 'Saving to localStorage')
    saveDeploySettings(next, selectedApiKey)

    // Update command line preview in real-time
    deployLog('updateConfig', 'Updating command preview')
    updateCommandPreview(next)
  }

  const resetToDefault = (path: string) => {
    // Set to undefined so backend skips the parameter (uses llama-server defaults)
    updateConfig(path, undefined)
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
    if (!config) {
      deployLog('validate', 'ABORT: config is null')
      return
    }
    try {
      setValidating(true)
      setValidateErrors(null)
      setValidateWarnings(null)

      // Convert undefined values to null for proper backend handling
      const configForValidation = JSON.parse(JSON.stringify(config, (key, value) => {
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
    deployLog('save', 'Starting save', { modelName: config?.model?.name })
    if (!config) {
      deployLog('save', 'ABORT: config is null')
      return
    }
    try {
      setSaving(true)

      // Convert undefined values to null for proper backend handling
      const configForSave = JSON.parse(JSON.stringify(config, (key, value) => {
        return value === undefined ? null : value
      }))

      deployLog('save', 'Sending save request', { model: configForSave.model })
      const data = await (async () => {
        const updated = await apiService.updateServiceConfig({ config: configForSave as any })
        deployLog('save', 'Config saved, fetching updated command')
        // Re-query command line preview from backend since apiService doesn't return it
        const cfgRes = await fetch(`/api/v1/service/config`)
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

  const runAction = async (action: 'start' | 'stop' | 'restart') => {
    deployLog('action', `Running action: ${action}`, { hasConfig: !!config, modelName: config?.model?.name })
    if (!config && action !== 'stop') {
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

      // Convert undefined values to null for proper backend handling
      const configForAction = config ? JSON.parse(JSON.stringify(config, (key, value) => {
        return value === undefined ? null : value
      })) : null

      // Ensure the current template selection is included in the config
      // The template dropdown updates selectedTemplate state separately from config,
      // so we must inject it here to avoid the stale default overwriting the user's choice
      if (configForAction) {
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

        deployLog('action', 'Injected template selection into config', { selectedTemplate })
      }

      deployLog('action', 'Sending action to backend', { action, config: configForAction?.model })
      await apiService.performServiceAction(action === 'stop' ? { action } : { action, config: configForAction as any })
      deployLog('action', 'Action completed successfully')

      // Refresh current model info after action
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
      setSuccess(`Service ${action}ed successfully`)
    } catch (e) {
      deployLog('action', 'Action failed:', e)
      setError(e instanceof Error ? e.message : `Failed to ${action}`)
    } finally {
      setActionLoading(null)
    }
  }


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
            borderRadius: 1,
            border: '1px solid',
            borderColor: 'error.main'
          }}
        >
          Failed to load configuration.
        </Alert>
      </Box>
    )
  }

  deployLog('render', 'Rendering main content', {
    modelName: config.model?.name,
    modelCount: models.length,
    hasCurrentModel: !!currentModel
  })

  return (
    <Box sx={{
      p: { xs: 2, sm: 3, md: 4 },
      maxWidth: '100%',
      overflow: 'hidden'
    }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography
            variant="h1"
            sx={{
              fontWeight: 700,
              color: 'text.primary',
              mb: 0.5,
              fontSize: { xs: '1.25rem', sm: '1.5rem' },
              lineHeight: 1
            }}
          >
            Deploy
          </Typography>
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{
              fontSize: '0.8125rem',
              mb: { xs: 1, sm: 2 }
            }}
          >
            Configure and deploy your AI models
          </Typography>
        </Box>
        <Box display="flex" gap={1}>
          <Button
            variant="outlined"
            startIcon={<ResetIcon />}
            onClick={() => {
              deployLog('reset', 'Clear All button clicked')
              if (!config) {
                deployLog('reset', 'ABORT: config is null')
                return;
              }
              const sections = ['model', 'sampling', 'performance', 'speculative', 'context_extension', 'server'];
              const resetConfig = JSON.parse(JSON.stringify(config));

              sections.forEach(section => {
                const sectionDefaults = DEFAULT_VALUES[section as keyof typeof DEFAULT_VALUES];
                if (sectionDefaults && resetConfig[section as keyof Config]) {
                  Object.keys(sectionDefaults).forEach(key => {
                    // Set to undefined so backend uses llama-server defaults
                    (resetConfig[section as keyof Config] as any)[key] = undefined;
                  });
                }
              });

              deployLog('reset', 'Config reset to defaults', { modelName: resetConfig.model?.name })
              setConfig(resetConfig);
              saveDeploySettings(resetConfig, selectedApiKey);

              // Update command line preview
              updateCommandPreview(resetConfig);
            }}
            sx={{
              borderRadius: 1,
              fontWeight: 500,
              fontSize: '0.8125rem',
              textTransform: 'none',
              borderColor: 'rgba(255, 255, 255, 0.2)',
              '&:hover': { borderColor: 'warning.main', bgcolor: 'warning.50' }
            }}
          >
            Clear All (Use Server Defaults)
          </Button>
          <Button
            variant="outlined"
            startIcon={<ValidateIcon />}
            onClick={handleValidate}
            disabled={validating}
            sx={{
              borderRadius: 1,
              fontWeight: 500,
              fontSize: '0.8125rem',
              textTransform: 'none',
              borderColor: 'rgba(255, 255, 255, 0.2)',
              '&:hover': { borderColor: 'primary.main', bgcolor: 'primary.50' }
            }}
          >
            {validating ? 'Validating...' : 'Validate'}
          </Button>
          <Button
            variant="contained"
            startIcon={saving ? <CircularProgress size={20} /> : <SaveIcon />}
            onClick={handleSave}
            disabled={saving || !hasChanges}
            sx={{
              borderRadius: 1,
              fontWeight: 500,
              fontSize: '0.8125rem',
              textTransform: 'none',
              boxShadow: 'none',
              '&:hover': { boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)' }
            }}
          >
            Save
          </Button>
          <Button
            color="success"
            variant="contained"
            startIcon={actionLoading === 'start' ? <CircularProgress size={20} /> : <StartIcon />}
            onClick={() => runAction('start')}
            disabled={actionLoading !== null}
            sx={{
              borderRadius: 1,
              fontWeight: 500,
              fontSize: '0.8125rem',
              textTransform: 'none',
              boxShadow: 'none',
              '&:hover': { boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)' }
            }}
          >
            Start
          </Button>
          <Button
            color="warning"
            variant="contained"
            startIcon={actionLoading === 'restart' ? <CircularProgress size={20} /> : <RestartIcon />}
            onClick={() => runAction('restart')}
            disabled={actionLoading !== null}
            sx={{
              borderRadius: 1,
              fontWeight: 500,
              fontSize: '0.8125rem',
              textTransform: 'none',
              boxShadow: 'none',
              '&:hover': { boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)' }
            }}
          >
            Restart
          </Button>
          <Button
            color="error"
            variant="outlined"
            startIcon={actionLoading === 'stop' ? <CircularProgress size={20} /> : <StopIcon />}
            onClick={() => runAction('stop')}
            disabled={actionLoading !== null}
            sx={{
              borderRadius: 1,
              fontWeight: 500,
              fontSize: '0.8125rem',
              textTransform: 'none',
              borderColor: 'rgba(255, 255, 255, 0.2)',
              '&:hover': { borderColor: 'error.main', bgcolor: 'error.50' }
            }}
          >
            Stop
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert
          severity="error"
          sx={{
            mb: 2,
            borderRadius: 1,
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
            borderRadius: 1,
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
            borderRadius: 1,
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
      <Card sx={{
        mb: 3,
        borderRadius: 1,
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        bgcolor: 'background.paper'
      }}>
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 600 }}>Currently Deployed</Typography>
              {config?.model?.mmproj ? (
                <Chip
                  icon={<VisionIcon sx={{ fontSize: '0.875rem' }} />}
                  label="Vision Ready"
                  size="small"
                  sx={{
                    fontWeight: 600,
                    fontSize: '0.6875rem',
                    height: '22px',
                    bgcolor: 'rgba(76, 175, 80, 0.15)',
                    color: '#66bb6a',
                    border: '1px solid rgba(76, 175, 80, 0.3)',
                    '& .MuiChip-icon': { color: '#66bb6a' }
                  }}
                />
              ) : (
                <Chip
                  icon={<VisionOffIcon sx={{ fontSize: '0.875rem' }} />}
                  label="Text Only"
                  size="small"
                  variant="outlined"
                  sx={{
                    fontWeight: 500,
                    fontSize: '0.6875rem',
                    height: '22px',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    color: 'text.secondary',
                    '& .MuiChip-icon': { color: 'text.secondary' }
                  }}
                />
              )}
            </Box>
          }
          subheader={<Typography variant="body2" sx={{ fontSize: '0.8125rem', color: 'text.secondary' }}>
            {currentModel ? `${currentModel.name} • ${currentModel.variant} • ${currentModel.status}` : 'No model information available'}
          </Typography>}
          action={currentModel?.file_path ? (
            <Chip
              label={currentModel.file_path}
              variant="outlined"
              sx={{
                borderRadius: '4px',
                fontWeight: 500,
                fontSize: '0.6875rem',
                height: '24px',
                borderColor: 'rgba(255, 255, 255, 0.2)'
              }}
            />
          ) : undefined}
          sx={{ pb: 0 }}
        />
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Model</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Model Name</Typography>
              <Select
                fullWidth
                value={config.model.name}
                displayEmpty
                renderValue={(selected) => {
                  if (!selected) {
                    return <em>Select a model...</em>
                  }
                  return selected
                }}
                onChange={(e) => {
                  console.log('==== MODEL ONCHANGE FIRED ====')
                  console.log('New value:', e.target.value)
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

                  console.log('📝 Calling setConfig with atomic update:', { name: newName, variant: nextVariant, mmproj: nextConfig.model.mmproj })
                  setConfig(nextConfig)

                  // Save to localStorage for persistence
                  saveDeploySettings(nextConfig, selectedApiKey)

                  // Update command line preview (single call)
                  updateCommandPreview(nextConfig)
                }}
                sx={{
                  fontSize: '0.875rem',
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 1,
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
              <Grid item xs={12} md={6}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Model Variant</Typography>
                <Select
                  fullWidth
                  value={config.model.variant}
                  onChange={(e) => {
                    deployLog('variantSelect', 'Variant selection changed', {
                      newVariant: e.target.value,
                      previousVariant: config.model.variant,
                      modelName: config.model.name
                    })
                    updateConfig('model.variant', e.target.value)
                  }}
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1,
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
              <Grid item xs={12} md={6}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Model Variant</Typography>
                <Typography
                  sx={{
                    fontSize: '0.875rem',
                    color: 'text.secondary',
                    fontStyle: 'italic',
                    py: 1.5,
                    px: 1,
                    backgroundColor: 'rgba(255, 255, 255, 0.02)',
                    borderRadius: 1,
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                  }}
                >
                  {config.model.variant} (only variant available)
                </Typography>
              </Grid>
            )}
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Chat Template</Typography>
              <Select
                fullWidth
                value={selectedTemplate}
                disabled={templatesLoading}
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
                    borderRadius: 1
                  }
                }}
              >
                <MenuItem value="">
                  <em>Use tokenizer default (no custom template)</em>
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
              <Box mt={1}>
                <Chip
                  label="Manage templates"
                  variant="outlined"
                  onClick={() => window.location.assign('/templates')}
                  sx={{
                    borderRadius: '4px',
                    fontWeight: 500,
                    fontSize: '0.6875rem',
                    height: '24px',
                    borderColor: 'rgba(255, 255, 255, 0.2)'
                  }}
                />
              </Box>
            </Grid>
            <Grid item xs={12}>
              <Button
                variant="contained"
                color="warning"
                startIcon={actionLoading === 'restart' ? <CircularProgress size={20} /> : <RestartIcon />}
                onClick={() => runAction('restart')}
                disabled={actionLoading !== null}
                sx={{
                  borderRadius: 1,
                  fontWeight: 500,
                  fontSize: '0.8125rem',
                  textTransform: 'none',
                  boxShadow: 'none',
                  '&:hover': { boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)' }
                }}
              >
                Deploy this model now (restart)
              </Button>
            </Grid>
          </Grid>
          <Box mt={1}>
            <Chip
              icon={<DownloadIcon />}
              label="Download more models"
              variant="outlined"
              onClick={() => window.location.assign('/models')}
              sx={{
                borderRadius: '4px',
                fontWeight: 500,
                fontSize: '0.6875rem',
                height: '24px',
                borderColor: 'rgba(255, 255, 255, 0.2)'
              }}
            />
          </Box>
        </CardContent>
      </Card>

      {/* VRAM Estimation Display */}
      {vramEstimate && (
        <Card sx={{
          mb: 3,
          borderRadius: 1,
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
          border: vramEstimate.fits_in_vram
            ? '1px solid rgba(76, 175, 80, 0.3)'
            : '1px solid rgba(244, 67, 54, 0.3)',
          bgcolor: 'background.paper'
        }}>
          <CardContent sx={{ py: 2, '&:last-child': { pb: 2 } }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="h6" sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>
                  VRAM Estimation
                </Typography>
                {vramLoading && <CircularProgress size={16} />}
              </Box>
              <Chip
                label={vramEstimate.fits_in_vram ? 'Fits in VRAM' : 'Exceeds VRAM'}
                color={vramEstimate.fits_in_vram ? 'success' : 'error'}
                size="small"
                sx={{ fontWeight: 500 }}
              />
            </Box>

            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">Model Weights</Typography>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {(vramEstimate.model_weights_mb / 1024).toFixed(1)} GB
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">KV Cache</Typography>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {(vramEstimate.kv_cache_mb / 1024).toFixed(2)} GB
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">Compute Buffer</Typography>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {(vramEstimate.compute_buffer_mb / 1024).toFixed(2)} GB
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">Total Required</Typography>
                <Typography variant="body2" sx={{ fontWeight: 600, color: vramEstimate.fits_in_vram ? 'success.main' : 'error.main' }}>
                  {vramEstimate.total_gb.toFixed(1)} GB / {vramEstimate.available_vram_gb} GB
                </Typography>
              </Grid>
            </Grid>

            {/* VRAM Usage Bar */}
            <Box sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="caption" color="text.secondary">VRAM Utilization</Typography>
                <Typography variant="caption" color="text.secondary">{vramEstimate.utilization_percent}%</Typography>
              </Box>
              <Box sx={{
                width: '100%',
                height: 8,
                bgcolor: 'rgba(255,255,255,0.1)',
                borderRadius: 1,
                overflow: 'hidden'
              }}>
                <Box sx={{
                  width: `${Math.min(vramEstimate.utilization_percent, 100)}%`,
                  height: '100%',
                  bgcolor: vramEstimate.utilization_percent > 95 ? 'error.main' :
                    vramEstimate.utilization_percent > 85 ? 'warning.main' : 'success.main',
                  borderRadius: 1,
                  transition: 'width 0.3s ease'
                }} />
              </Box>
            </Box>

            {/* Warnings */}
            {vramEstimate.warnings.length > 0 && (
              <Box sx={{ mt: 2 }}>
                {vramEstimate.warnings.map((warning, idx) => (
                  <Alert key={idx} severity="warning" sx={{ mb: 1, py: 0, fontSize: '0.75rem' }}>
                    {warning}
                  </Alert>
                ))}
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* Tabs for settings */}
      <Tabs
        value={tab}
        onChange={(_, v) => setTab(v)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          mb: 3,
          '& .MuiTabs-indicator': {
            height: 3,
            borderRadius: '3px 3px 0 0'
          },
          '& .MuiTab-root': {
            textTransform: 'none',
            fontWeight: 500,
            fontSize: '0.875rem',
            minWidth: 100
          }
        }}
      >
        <Tab label="Model" />
        <Tab label="Sampling" />
        <Tab label="Performance" />
        <Tab label="Context Extension" />
        <Tab label="Server" />
        <Tab label="LlamaCPP Version" />
        <Tab label="Command Line" />
      </Tabs>

      {tab === 0 && (
        <Card sx={{
          mb: 3,
          borderRadius: 1,
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          bgcolor: 'background.paper'
        }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Model Configuration</Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="Context Size"
                  description="The size of the prompt context in tokens. Determines how much conversation history the model can remember. Higher values use more memory but allow for longer conversations."
                  path="model.context_size"
                  value={config.model.context_size}
                  defaultValue={getDefaultValue('model.context_size')}
                  type="number"
                  min={512}
                  max={131072}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="GPU Layers"
                  description="Number of model layers to offload to GPU memory. Set to -1 to offload all layers, 0 to use CPU only, or a specific number for partial GPU acceleration."
                  path="model.gpu_layers"
                  value={config.model.gpu_layers}
                  defaultValue={getDefaultValue('model.gpu_layers')}
                  type="number"
                  min={-1}
                  max={999}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600, mt: 2 }}>Execution Settings</Typography>
              </Grid>

              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 1 }}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Execution Mode</Typography>
                  <Select
                    fullWidth
                    value={config.execution?.mode || 'gpu'}
                    onChange={(e) => updateConfig('execution.mode', e.target.value)}
                    sx={{
                      fontSize: '0.875rem',
                      '& .MuiOutlinedInput-root': {
                        borderRadius: 1,
                        backgroundColor: 'background.default',
                      }
                    }}
                  >
                    <MenuItem value="gpu">GPU Acceleration</MenuItem>
                    <MenuItem value="cpu">CPU Only</MenuItem>
                  </Select>
                  <FormHelperText sx={{ mt: 0.5, fontSize: '0.75rem', color: 'text.secondary' }}>
                    Choose whether to use GPU acceleration or run on CPU only. CPU mode will override GPU layers to 0.
                  </FormHelperText>
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 1 }}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>CUDA Devices</Typography>
                  {['all', 'custom', ...gpuList.map(g => g.index.toString())].includes(config.execution?.cuda_devices || 'all') ? (
                    <Select
                      fullWidth
                      value={config.execution?.cuda_devices || 'all'}
                      onChange={(e) => {
                        if (e.target.value === 'custom') {
                          updateConfig('execution.cuda_devices', '0,1')
                        } else {
                          updateConfig('execution.cuda_devices', e.target.value)
                        }
                      }}
                      disabled={config.execution?.mode === 'cpu'}
                      sx={{
                        fontSize: '0.875rem',
                        '& .MuiOutlinedInput-root': {
                          borderRadius: 1,
                          backgroundColor: config.execution?.mode === 'cpu' ? 'rgba(255,255,255,0.05)' : 'background.default',
                        }
                      }}
                    >
                      <MenuItem value="all">All GPUs</MenuItem>
                      {gpuList.map((gpu) => (
                        <MenuItem key={gpu.index} value={gpu.index.toString()}>
                          GPU {gpu.index}: {gpu.name} ({(gpu.vram_total_mb / 1024).toFixed(1)} GB)
                        </MenuItem>
                      ))}
                      {gpuList.length > 1 && (
                        <MenuItem value="custom">Custom (e.g., 0,1 or 0,2)</MenuItem>
                      )}
                    </Select>
                  ) : (
                    <TextField
                      fullWidth
                      value={config.execution?.cuda_devices || 'all'}
                      onChange={(e) => updateConfig('execution.cuda_devices', e.target.value)}
                      disabled={config.execution?.mode === 'cpu'}
                      placeholder="e.g., 0,1 or 0,2"
                      InputProps={{
                        endAdornment: (
                          <Tooltip title="Switch back to dropdown">
                            <IconButton
                              size="small"
                              onClick={() => updateConfig('execution.cuda_devices', 'all')}
                            >
                              <ResetIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        ),
                      }}
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          fontSize: '0.875rem',
                          borderRadius: 1,
                          backgroundColor: config.execution?.mode === 'cpu' ? 'rgba(255,255,255,0.05)' : 'background.default',
                        }
                      }}
                    />
                  )}
                  <FormHelperText sx={{ mt: 0.5, fontSize: '0.75rem', color: 'text.secondary' }}>
                    {config.execution?.mode === 'cpu'
                      ? 'GPU selection is disabled in CPU mode'
                      : gpusAvailable
                        ? `Select which GPU(s) to use. Separate multiple GPUs with commas (e.g., 0,1). Found ${gpuList.length} GPU(s).`
                        : 'No GPUs detected. Using CPU mode.'}
                  </FormHelperText>
                </Box>
              </Grid>

              {gpusAvailable && gpuList.length > 0 && config.execution?.mode === 'gpu' && (
                <Grid item xs={12}>
                  <Box sx={{
                    p: 2,
                    bgcolor: 'rgba(255,255,255,0.02)',
                    borderRadius: 1,
                    border: '1px solid rgba(255,255,255,0.1)'
                  }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ fontSize: '0.8125rem', fontWeight: 600 }}>
                      Available GPUs
                    </Typography>
                    <Grid container spacing={1}>
                      {gpuList.map((gpu) => (
                        <Grid item xs={12} md={6} key={gpu.index}>
                          <Box sx={{
                            p: 1.5,
                            bgcolor: 'rgba(255,255,255,0.03)',
                            borderRadius: 1,
                            border: '1px solid rgba(255,255,255,0.05)'
                          }}>
                            <Typography variant="body2" sx={{ fontWeight: 500, mb: 0.5 }}>
                              GPU {gpu.index}: {gpu.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                              VRAM: {(gpu.vram_used_mb / 1024).toFixed(1)} GB / {(gpu.vram_total_mb / 1024).toFixed(1)} GB
                              ({((gpu.vram_used_mb / gpu.vram_total_mb) * 100).toFixed(0)}% used)
                            </Typography>
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                              Utilization: {gpu.utilization_percent.toFixed(0)}% | Temp: {gpu.temperature_c.toFixed(0)}°C
                            </Typography>
                          </Box>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                </Grid>
              )}

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
                  p: 2,
                  bgcolor: config.model.mmproj ? 'rgba(76, 175, 80, 0.04)' : 'rgba(255,255,255,0.02)',
                  borderRadius: 1,
                  border: config.model.mmproj
                    ? '1px solid rgba(76, 175, 80, 0.2)'
                    : '1px solid rgba(255,255,255,0.1)',
                  transition: 'all 0.2s ease'
                }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {config.model.mmproj ? (
                        <VisionIcon sx={{ fontSize: '1.1rem', color: '#66bb6a' }} />
                      ) : (
                        <VisionOffIcon sx={{ fontSize: '1.1rem', color: 'text.secondary' }} />
                      )}
                      <Typography variant="subtitle2" sx={{ fontSize: '0.875rem', fontWeight: 600 }}>
                        Vision / Multimodal Projector
                      </Typography>
                      {config.model.mmproj ? (
                        <Chip
                          label="Active"
                          size="small"
                          sx={{
                            fontWeight: 600,
                            fontSize: '0.625rem',
                            height: '18px',
                            bgcolor: 'rgba(76, 175, 80, 0.15)',
                            color: '#66bb6a',
                            border: '1px solid rgba(76, 175, 80, 0.3)',
                          }}
                        />
                      ) : (
                        <Chip
                          label="Disabled"
                          size="small"
                          variant="outlined"
                          sx={{
                            fontWeight: 500,
                            fontSize: '0.625rem',
                            height: '18px',
                            borderColor: 'rgba(255, 255, 255, 0.15)',
                            color: 'text.secondary',
                          }}
                        />
                      )}
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

                  <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem', mb: 1.5 }}>
                    Multimodal projector files enable vision capabilities, allowing the model to understand images and video alongside text.
                    Select a projector file that matches your model, or download one from the model's HuggingFace repository.
                  </Typography>

                  {mmprojAutoMatched && config.model.mmproj && (
                    <Alert severity="info" sx={{ mb: 1.5, py: 0, fontSize: '0.75rem' }}>
                      Auto-matched projector file for this model
                    </Alert>
                  )}

                  <Select
                    fullWidth
                    value={config.model.mmproj || ''}
                    onChange={(e) => {
                      const value = e.target.value === '' ? undefined : e.target.value
                      updateConfig('model.mmproj', value)
                      setMmprojAutoMatched(false)
                    }}
                    displayEmpty
                    sx={{
                      fontSize: '0.875rem',
                      '& .MuiOutlinedInput-root': {
                        borderRadius: 1,
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
                    bgcolor: 'rgba(33, 150, 243, 0.04)',
                    borderRadius: 1,
                    border: '1px solid rgba(33, 150, 243, 0.2)',
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
                          window.location.assign('/models')
                        }}
                        sx={{ textTransform: 'none', fontSize: '0.8125rem' }}
                      >
                        Go to Models Page to Download
                      </Button>
                      <Button
                        size="small"
                        onClick={() => setDownloadMmprojDialogOpen(false)}
                        sx={{ textTransform: 'none', fontSize: '0.8125rem' }}
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

              {/* Speculative Decoding - Collapsible */}
              <Grid item xs={12}>
                <Accordion
                  disableGutters
                  sx={{
                    bgcolor: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: '4px !important',
                    '&:before': { display: 'none' },
                    boxShadow: 'none',
                  }}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Speculative Decoding</Typography>
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
          </CardContent>
        </Card>
      )}

      {tab === 1 && (
        <Card sx={{
          mb: 3,
          borderRadius: 1,
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          bgcolor: 'background.paper'
        }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
              <Typography variant="h6" sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Sampling Configuration</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontSize: '0.75rem' }}>
                  Quick Presets:
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {PARAMETER_PRESETS.map((preset) => (
                    <Tooltip key={preset.name} title={preset.description}>
                      <Chip
                        label={preset.name}
                        size="small"
                        variant={selectedPreset === preset.name ? 'filled' : 'outlined'}
                        color={
                          preset.category === 'coding' ? 'info' :
                            preset.category === 'creative' ? 'secondary' :
                              preset.category === 'precise' ? 'success' : 'primary'
                        }
                        onClick={() => applyPreset(preset)}
                        sx={{
                          cursor: 'pointer',
                          fontWeight: selectedPreset === preset.name ? 600 : 400,
                          '&:hover': { opacity: 0.8 }
                        }}
                      />
                    </Tooltip>
                  ))}
                </Box>
              </Box>
            </Box>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="Temperature"
                  description="Controls randomness in text generation. Higher values (0.8-1.0) produce more creative output, lower values (0.1-0.3) produce more focused and deterministic text."
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="Top P"
                  description="Nucleus sampling: limits next token selection to a subset with cumulative probability above threshold. Lower values (0.1-0.5) for focused text, higher (0.9-1.0) for diversity."
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="Top K"
                  description="Limits next token selection to the K most probable tokens. Lower values (10-20) for focused output, higher (40-100) for more variety. Set to 0 to disable."
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="Min P"
                  description="Minimum probability threshold relative to the most likely token. Tokens below this threshold are filtered out. Set to 0.0 to disable."
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="Repeat Penalty"
                  description="Penalizes repeated tokens to reduce repetition. Values > 1.0 discourage repetition, 1.0 = no penalty, < 1.0 encourages repetition."
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="Repeat Last N"
                  description="Number of recent tokens to consider for repetition penalty. Higher values check more history for repetition. Set to -1 to use context size."
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="Frequency Penalty"
                  description="Penalizes tokens based on their frequency in the text so far. Positive values discourage frequent tokens, negative values encourage them."
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="Presence Penalty"
                  description="Penalizes tokens that have already appeared in the text, encouraging new topics. Positive values promote diversity, negative values encourage repetition."
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
                <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600, mt: 2 }}>DRY (Don't Repeat Yourself) Sampling</Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="DRY Multiplier"
                  description="Controls the strength of DRY repetition penalty. Set to 0.0 to disable DRY sampling. Higher values apply stronger penalties to repeated sequences."
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="DRY Base"
                  description="Base value for DRY repetition penalty calculation. This affects how the penalty scales with sequence length and repetition."
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="DRY Allowed Length"
                  description="Minimum sequence length before DRY penalty is applied. Shorter sequences than this length are not penalized for repetition."
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="DRY Penalty Last N"
                  description="Number of recent tokens to scan for DRY repetitions. Set to -1 to use the full context size. Higher values check more history but use more computation."
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

              {/* Advanced Sampling - Collapsible */}
              <Grid item xs={12}>
                <Accordion
                  disableGutters
                  sx={{
                    bgcolor: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: '4px !important',
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
          </CardContent>
        </Card>
      )}

      {tab === 2 && (
        <Card sx={{
          mb: 3,
          borderRadius: 1,
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          bgcolor: 'background.paper'
        }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Performance Configuration</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Threads"
                  type="number"
                  fullWidth
                  value={config.performance.threads}
                  onChange={(e) => updateConfig('performance.threads', parseInt(e.target.value))}
                  helperText="Number of threads for computation (--threads)"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
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
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
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
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
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
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
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
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
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
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Parallel Slots"
                  type="number"
                  fullWidth
                  value={config.performance.parallel_slots || 1}
                  onChange={(e) => updateConfig('performance.parallel_slots', parseInt(e.target.value))}
                  helperText="Number of slots for processing requests (--parallel)"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Split Mode</Typography>
                <Select
                  fullWidth
                  value={config.performance.split_mode || 'none'}
                  onChange={(e) => updateConfig('performance.split_mode', e.target.value)}
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
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
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
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
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
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
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
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
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
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
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
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
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
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
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
                >
                  <MenuItem value="">None (Default)</MenuItem>
                  <MenuItem value="distribute">Distribute</MenuItem>
                  <MenuItem value="isolate">Isolate</MenuItem>
                  <MenuItem value="numactl">Numactl</MenuItem>
                </Select>
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Cache Type K"
                  type="text"
                  fullWidth
                  value={config.performance.cache_type_k || 'q4_0'}
                  onChange={(e) => updateConfig('performance.cache_type_k', e.target.value)}
                  helperText="KV cache data type for K (--cache-type-k)"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Cache Type V"
                  type="text"
                  fullWidth
                  value={config.performance.cache_type_v || 'q4_0'}
                  onChange={(e) => updateConfig('performance.cache_type_v', e.target.value)}
                  helperText="KV cache data type for V (--cache-type-v)"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {tab === 3 && (
        <Card sx={{
          mb: 3,
          borderRadius: 1,
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          bgcolor: 'background.paper'
        }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Context Extension Options</Typography>
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
                        const value = parseFloat(e.target.value);
                        const next = { ...(config.context_extension || {}), yarn_ext_factor: value };
                        updateConfig('context_extension', next);
                      }}
                      helperText="YaRN extrapolation mix factor (--yarn-ext-factor)"
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          fontSize: '0.875rem',
                          borderRadius: 1,
                          backgroundColor: 'background.default',
                          '&.Mui-focused': {
                            borderColor: 'primary.main'
                          }
                        },
                        '& .MuiInputLabel-root': {
                          fontSize: '0.875rem'
                        }
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      label="YaRN Attn Factor"
                      type="number"
                      fullWidth
                      value={config.context_extension?.yarn_attn_factor || 1.0}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        const next = { ...(config.context_extension || {}), yarn_attn_factor: value };
                        updateConfig('context_extension', next);
                      }}
                      helperText="YaRN attention scaling factor (--yarn-attn-factor)"
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          fontSize: '0.875rem',
                          borderRadius: 1,
                          backgroundColor: 'background.default',
                          '&.Mui-focused': {
                            borderColor: 'primary.main'
                          }
                        },
                        '& .MuiInputLabel-root': {
                          fontSize: '0.875rem'
                        }
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      label="YaRN Beta Slow"
                      type="number"
                      fullWidth
                      value={config.context_extension?.yarn_beta_slow || 1.0}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        const next = { ...(config.context_extension || {}), yarn_beta_slow: value };
                        updateConfig('context_extension', next);
                      }}
                      helperText="YaRN high correction dimension (--yarn-beta-slow)"
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          fontSize: '0.875rem',
                          borderRadius: 1,
                          backgroundColor: 'background.default',
                          '&.Mui-focused': {
                            borderColor: 'primary.main'
                          }
                        },
                        '& .MuiInputLabel-root': {
                          fontSize: '0.875rem'
                        }
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      label="YaRN Beta Fast"
                      type="number"
                      fullWidth
                      value={config.context_extension?.yarn_beta_fast || 32.0}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        const next = { ...(config.context_extension || {}), yarn_beta_fast: value };
                        updateConfig('context_extension', next);
                      }}
                      helperText="YaRN low correction dimension (--yarn-beta-fast)"
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          fontSize: '0.875rem',
                          borderRadius: 1,
                          backgroundColor: 'background.default',
                          '&.Mui-focused': {
                            borderColor: 'primary.main'
                          }
                        },
                        '& .MuiInputLabel-root': {
                          fontSize: '0.875rem'
                        }
                      }}
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
                        const value = parseInt(e.target.value);
                        const next = { ...(config.context_extension || {}), group_attn_n: value };
                        updateConfig('context_extension', next);
                      }}
                      helperText="Group attention factor (--grp-attn-n)"
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          fontSize: '0.875rem',
                          borderRadius: 1,
                          backgroundColor: 'background.default',
                          '&.Mui-focused': {
                            borderColor: 'primary.main'
                          }
                        },
                        '& .MuiInputLabel-root': {
                          fontSize: '0.875rem'
                        }
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      label="Group Attention Width"
                      type="number"
                      fullWidth
                      value={config.context_extension?.group_attn_w || 512}
                      onChange={(e) => {
                        const value = parseInt(e.target.value);
                        const next = { ...(config.context_extension || {}), group_attn_w: value };
                        updateConfig('context_extension', next);
                      }}
                      helperText="Group attention width (--grp-attn-w)"
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          fontSize: '0.875rem',
                          borderRadius: 1,
                          backgroundColor: 'background.default',
                          '&.Mui-focused': {
                            borderColor: 'primary.main'
                          }
                        },
                        '& .MuiInputLabel-root': {
                          fontSize: '0.875rem'
                        }
                      }}
                    />
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {tab === 4 && (
        <Card sx={{
          mb: 3,
          borderRadius: 1,
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          bgcolor: 'background.paper'
        }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Server Configuration</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Host"
                  type="text"
                  fullWidth
                  value={config.server.host}
                  onChange={(e) => updateConfig('server.host', e.target.value)}
                  helperText="IP address to listen on (--host)"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
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
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
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
                      sx={{
                        fontSize: '0.875rem',
                        '& .MuiOutlinedInput-root': {
                          borderRadius: 1
                        }
                      }}
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
                            borderRadius: 1,
                            backgroundColor: 'background.default',
                          }
                        }}
                      />
                      <Button
                        variant="outlined"
                        onClick={handleAddApiKey}
                        disabled={!newApiKey.trim()}
                        sx={{
                          borderRadius: 1,
                          fontSize: '0.8125rem',
                          textTransform: 'none',
                          borderColor: 'rgba(255, 255, 255, 0.2)',
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
                        <Chip
                          label={`Current: ${selectedApiKey.length > 30 ? `${selectedApiKey.substring(0, 30)}...` : selectedApiKey}`}
                          variant="outlined"
                          sx={{
                            borderRadius: '4px',
                            fontWeight: 500,
                            fontSize: '0.6875rem',
                            height: '24px',
                            borderColor: 'rgba(255, 255, 255, 0.2)'
                          }}
                        />
                        <Button
                          size="small"
                          color="error"
                          variant="outlined"
                          onClick={() => handleRemoveApiKey(selectedApiKey)}
                          sx={{
                            borderRadius: 1,
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
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
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
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    },
                    '& .MuiInputLabel-root': {
                      fontSize: '0.875rem'
                    }
                  }}
                />
              </Grid>
              <Grid item xs={6} md={2}>
                <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Embedding</Typography>
                <Select
                  fullWidth
                  value={config.server.embedding ? 'true' : 'false'}
                  onChange={(e) => updateConfig('server.embedding', e.target.value === 'true')}
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
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
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
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
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
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
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
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
                  sx={{
                    fontSize: '0.875rem',
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 1
                    }
                  }}
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
                    bgcolor: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: '4px !important',
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
                          description="Enable Jinja template processing for chat templates. Required for advanced template features. Automatically enabled when a custom template file is selected. (--jinja / --no-jinja)"
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
          </CardContent>
        </Card>
      )}

      {tab === 5 && (
        <Box sx={{ mb: 3 }}>
          <LlamaCppCommitSelector
            onCommitChanged={(commit) => {
              console.log('LlamaCPP commit changed to:', commit);
              // Optionally refresh configuration or show notification
            }}
          />
        </Box>
      )}

      {tab === 6 && (
        <Card sx={{
          borderRadius: 1,
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          bgcolor: 'background.paper'
        }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Generated Command</Typography>
            <Paper sx={{
              p: 2,
              backgroundColor: '#1e1e1e',
              color: '#d4d4d4',
              fontFamily: 'monospace',
              fontSize: '0.9rem',
              overflowX: 'auto',
              borderRadius: 1,
              border: '1px solid rgba(255, 255, 255, 0.05)'
            }}>
              {commandLine || 'Save configuration to refresh command preview.'}
            </Paper>
          </CardContent>
        </Card>
      )}

      {/* Container Logs Section */}
      {currentModel && currentModel.status === 'loaded' && (
        <Card sx={{
          borderRadius: 1,
          boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
          bgcolor: 'background.paper',
          mt: 3
        }}>
          <CardHeader
            title="Container Logs"
            titleTypographyProps={{
              variant: 'h6',
              sx: { fontSize: '0.9375rem', fontWeight: 600 }
            }}
            subheader="Real-time logs from the deployed container"
            subheaderTypographyProps={{
              variant: 'caption',
              sx: { fontSize: '0.75rem' }
            }}
          />
          <CardContent sx={{ pt: 0 }}>
            <LogViewer
              ref={logViewerRef}
              containerName="llamacpp-api"
              maxLines={500}
              height={500}
            />
          </CardContent>
        </Card>
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
            borderRadius: 1
          }
        }}
      />
    </Box>
  )
}

export default DeployPage


