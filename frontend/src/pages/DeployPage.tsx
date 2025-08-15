import React, { useEffect, useMemo, useState } from 'react'
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

// Default values for all parameters based on llama.cpp documentation
const DEFAULT_VALUES = {
  model: {
    context_size: 512,
    gpu_layers: 0,
    lora: '',
    lora_base: '',
    mmproj: '',
    rope_scaling: 'linear' as const,
    rope_freq_base: 0,
    rope_freq_scale: 0,
    n_cpu_moe: 0,
  },
  sampling: {
    temperature: 0.8,
    top_p: 0.95,
    top_k: 40,
    min_p: 0.05,
    repeat_penalty: 1.1,
    repeat_last_n: 64,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
    dry_multiplier: 0.0,
    dry_base: 1.75,
    dry_allowed_length: 2,
    dry_penalty_last_n: -1,
  },
  performance: {
    threads: 8,
    threads_batch: 8,
    batch_size: 512,
    ubatch_size: 512,
    num_keep: 0,
    num_predict: -1,
    memory_f32: false,
    mlock: false,
    no_mmap: false,
    numa: '' as const,
    split_mode: 'layer' as const,
    tensor_split: '',
    main_gpu: 0,
    continuous_batching: false,
    parallel_slots: 1,
    cache_type_k: 'q4_0',
    cache_type_v: 'q4_0',
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
    host: '127.0.0.1',
    port: 8080,
    api_key: '',
    timeout: 600,
    embedding: false,
    system_prompt_file: '',
    log_format: 'json' as const,
    log_disable: false,
    slots_endpoint_disable: false,
    metrics: false,
  },
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
  const isDefault = value === defaultValue || (value === '' && defaultValue === '');
  
  return (
    <Box sx={{ position: 'relative' }}>
      {type === 'select' ? (
        <>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Typography gutterBottom sx={{ fontSize: '0.875rem', mb: 0 }}>{label}</Typography>
            <Tooltip title={`Reset to default (${defaultValue})`}>
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
            value={value}
            onChange={(e) => onChange(path, e.target.value)}
            sx={{ 
              fontSize: '0.875rem',
              '& .MuiOutlinedInput-root': {
                borderRadius: 1,
                backgroundColor: 'background.default',
              }
            }}
          >
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
          value={value}
          onChange={(e) => onChange(path, type === 'number' ? (step && step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value)) : e.target.value)}
          inputProps={{ min, max, step }}
          InputProps={{
            endAdornment: (
              <Tooltip title={`Reset to default (${defaultValue})`}>
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
            ),
          }}
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
      )}
      <FormHelperText sx={{ mt: 0.5, fontSize: '0.75rem', color: 'text.secondary' }}>
        {description} (Default: {typeof defaultValue === 'boolean' ? (defaultValue ? 'Enabled' : 'Disabled') : defaultValue})
      </FormHelperText>
    </Box>
  );
};

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

  useEffect(() => {
    const init = async () => {
      try {
        setLoading(true)
        // Load available local models
        const list = await apiService.getModels()
        setModels(list)
        // Load current service config + generated command
        const cfgRes = await fetch(`/api/v1/service/config`)
        if (!cfgRes.ok) throw new Error('Failed to fetch configuration')
        const cfgJson = await cfgRes.json()
        setConfig(cfgJson.config)
        setOriginalConfig(JSON.parse(JSON.stringify(cfgJson.config)))
        setCommandLine(cfgJson.command || '')

        // Load currently deployed model info (best-effort)
        try {
          const cm = await apiService.getCurrentModel()
          setCurrentModel(cm)
        } catch {}

        // Load templates
        try {
          setTemplatesLoading(true)
          const data = await apiService.listTemplates()
          setTemplates(data.files)
          setSelectedTemplate(data.selected)
          setTemplatesDir(data.directory)
        } finally {
          setTemplatesLoading(false)
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to initialize deploy page')
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [])

  const availableModelNames = useMemo(
    () => Array.from(new Set(models.map((m) => m.name))).sort(),
    [models]
  )
  const availableVariantsForSelected = useMemo(() => {
    if (!config) return [] as string[]
    const variants = models.filter((m) => m.name === config.model.name).map((m) => m.variant)
    return Array.from(new Set(variants.filter(Boolean))).sort()
  }, [models, config])

  const hasChanges = useMemo(
    () => JSON.stringify(config) !== JSON.stringify(originalConfig),
    [config, originalConfig]
  )

  const updateConfig = (path: string, value: any) => {
    if (!config) return
    const next: Config = JSON.parse(JSON.stringify(config))
    const keys = path.split('.')
    let ref: any = next
    for (let i = 0; i < keys.length - 1; i++) ref = ref[keys[i]]
    ref[keys[keys.length - 1]] = value
    setConfig(next)
  }

  const resetToDefault = (path: string) => {
    const keys = path.split('.')
    let defaultValue: any = DEFAULT_VALUES
    for (const key of keys) {
      defaultValue = defaultValue[key]
    }
    updateConfig(path, defaultValue)
  }

  const getDefaultValue = (path: string) => {
    const keys = path.split('.')
    let defaultValue: any = DEFAULT_VALUES
    for (const key of keys) {
      defaultValue = defaultValue[key]
    }
    return defaultValue
  }

  const handleValidate = async () => {
    if (!config) return
    try {
      setValidating(true)
      setValidateErrors(null)
      setValidateWarnings(null)
      const data = await apiService.validateServiceConfig(config as any)
      setValidateErrors(data.errors || null)
      setValidateWarnings(data.warnings || null)
      if (data.valid) setSuccess('Configuration is valid')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Validation failed')
    } finally {
      setValidating(false)
    }
  }

  const handleSave = async () => {
    if (!config) return
    try {
      setSaving(true)
      const data = await (async () => {
        const updated = await apiService.updateServiceConfig({ config: config as any })
        // Re-query command line preview from backend since apiService doesn't return it
        const cfgRes = await fetch(`/api/v1/service/config`)
        const cfgJson = await cfgRes.json()
        return { command: cfgJson.command, updated }
      })()
      setOriginalConfig(JSON.parse(JSON.stringify(config)))
      setCommandLine(data.command || '')
      setSuccess('Configuration saved')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save configuration')
    } finally {
      setSaving(false)
    }
  }

  const runAction = async (action: 'start' | 'stop' | 'restart') => {
    if (!config && action !== 'stop') return
    try {
      setActionLoading(action)
      await apiService.performServiceAction(action === 'stop' ? { action } : { action, config: config as any })
      // Refresh current model info after action
      try {
        const res = await fetch(`/api/v1/models/current`)
        if (res.ok) {
          const current = await res.json()
          setCurrentModel(current)
        }
      } catch {}
      setSuccess(`Service ${action}ed successfully`)
    } catch (e) {
      setError(e instanceof Error ? e.message : `Failed to ${action}`)
    } finally {
      setActionLoading(null)
    }
  }

  if (loading) {
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
              if (!config) return;
              const sections = ['model', 'sampling', 'performance', 'context_extension', 'server'];
              sections.forEach(section => {
                const sectionDefaults = DEFAULT_VALUES[section as keyof typeof DEFAULT_VALUES];
                if (sectionDefaults) {
                  Object.keys(sectionDefaults).forEach(key => {
                    resetToDefault(`${section}.${key}`);
                  });
                }
              });
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
            Reset All to Defaults
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
              <Select
                fullWidth
                value={config.model.name}
                sx={{ 
                  fontSize: '0.875rem',
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 1
                  }
                }}
                onChange={(e) => {
                  const newName = e.target.value as string
                  updateConfig('model.name', newName)
                  // If current variant not available for new name, reset
                  if (!availableVariantsForSelected.includes(config.model.variant)) {
                    const nextVariant = models.find((m) => m.name === newName)?.variant || 'Q4_K_M'
                    updateConfig('model.variant', nextVariant)
                  }
                }}
              >
                {availableModelNames.length === 0 && (
                  <MenuItem value="" disabled>
                    No local models found. Use Download to fetch one.
                  </MenuItem>
                )}
                {availableModelNames.map((name) => (
                  <MenuItem key={name} value={name}>{name}</MenuItem>
                ))}
              </Select>
            </Grid>
            <Grid item xs={12} md={6}>
              <Select
                fullWidth
                value={config.model.variant}
                onChange={(e) => updateConfig('model.variant', e.target.value)}
                sx={{ 
                  fontSize: '0.875rem',
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 1
                  }
                }}
              >
                {availableVariantsForSelected.length === 0 && (
                  <MenuItem value={config.model.variant}>{config.model.variant}</MenuItem>
                )}
                {availableVariantsForSelected.map((variant) => (
                  <MenuItem key={variant} value={variant}>{variant}</MenuItem>
                ))}
              </Select>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>Chat Template</Typography>
              <Select
                fullWidth
                value={selectedTemplate}
                disabled={templatesLoading}
                onChange={async (e) => {
                  const filename = e.target.value as string
                  setSelectedTemplate(filename)
                  try {
                    await apiService.selectTemplate(filename)
                    // Refresh config and command preview to reflect new template selection
                    const cfgRes = await fetch(`/api/v1/service/config`)
                    if (cfgRes.ok) {
                      const cfgJson = await cfgRes.json()
                      setConfig(cfgJson.config)
                      setCommandLine(cfgJson.command || '')
                    }
                  } catch (err) {
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
                {templates.length === 0 && (
                  <MenuItem value="" disabled>
                    No templates found in {templatesDir || '/home/llamacpp/templates'}
                  </MenuItem>
                )}
                {templates.map((f) => (
                  <MenuItem key={f} value={f}>{f}</MenuItem>
                ))}
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
                  type="text"
                  onChange={updateConfig}
                  onReset={resetToDefault}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <ParameterField
                  label="RoPE Scaling Method"
                  description="RoPE (Rotary Position Embedding) frequency scaling method for extending context length. Linear scaling works for most cases, YaRN is more sophisticated."
                  path="model.rope_scaling"
                  value={config.model.rope_scaling || 'linear'}
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
                  description="Base frequency for RoPE calculations. Set to 0 to use the model's default value. Higher values may help with longer contexts."
                  path="model.rope_freq_base"
                  value={config.model.rope_freq_base || 0}
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
                  description="RoPE frequency scaling factor. Values less than 1.0 expand the context window. For example, 0.5 doubles the effective context length."
                  path="model.rope_freq_scale"
                  value={config.model.rope_freq_scale || 0}
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
                  value={config.model.n_cpu_moe || 21}
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
            <Typography variant="h6" gutterBottom sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Sampling Configuration</Typography>
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
                        const next = {...(config.context_extension || {}), yarn_ext_factor: value};
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
                        const next = {...(config.context_extension || {}), yarn_attn_factor: value};
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
                        const next = {...(config.context_extension || {}), yarn_beta_slow: value};
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
                        const next = {...(config.context_extension || {}), yarn_beta_fast: value};
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
                        const next = {...(config.context_extension || {}), group_attn_n: value};
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
                        const next = {...(config.context_extension || {}), group_attn_w: value};
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
              <Grid item xs={12} md={4}>
                <TextField
                  label="API Key"
                  type="text"
                  fullWidth
                  value={config.server.api_key}
                  onChange={(e) => updateConfig('server.api_key', e.target.value)}
                  helperText="API key for authentication (--api-key)"
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


