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
} from '@mui/material'
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  RestartAlt as RestartIcon,
  CheckCircle as ValidateIcon,
  Save as SaveIcon,
  CloudDownload as DownloadIcon,
  Refresh as ResetIcon,
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
    n_cpu_moe: 12,
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

        // Load available local models
        deployLog('init', 'Fetching available models...')
        const list = await apiService.getModels()
        deployLog('init', 'Loaded models:', { count: list.length, models: list.map(m => `${m.name}/${m.variant}`) })
        setModels(list)

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
              const sections = ['model', 'sampling', 'performance', 'context_extension', 'server'];
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
          title={<Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 600 }}>Currently Deployed</Typography>}
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

                  const nextConfig = JSON.parse(JSON.stringify(config))
                  nextConfig.model.name = newName
                  nextConfig.model.variant = nextVariant

                  console.log('📝 Calling setConfig with atomic update:', { name: newName, variant: nextVariant })
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
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="Multimodal Projector"
                  description="Path to multimodal projector file for LLaVA models. Required for vision-language models that can process both text and images."
                  path="model.mmproj"
                  value={config.model.mmproj || ''}
                  defaultValue={getDefaultValue('model.mmproj')}
                  type="select"
                  options={[
                    ...mmprojFiles.map(f => ({
                      value: f.name,
                      label: `${f.name} (${f.size_mb} MB)`
                    }))
                  ]}
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
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


