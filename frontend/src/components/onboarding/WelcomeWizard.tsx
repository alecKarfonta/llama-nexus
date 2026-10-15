/**
 * WelcomeWizard - First-run onboarding wizard for llama-nexus.
 *
 * 4 steps: (1) welcome + detected hardware, (2) use-case pick,
 * (3) curated model download, (4) done with links to Deploy / Chat.
 *
 * Visibility is driven by the localStorage flag `ln-onboarding-complete`.
 * Call `resetOnboarding()` from a "Restart setup" affordance to show it again.
 */
import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Alert,
  alpha,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  LinearProgress,
  Paper,
  Step,
  StepLabel,
  Stepper,
  Typography,
} from '@mui/material'
import {
  Chat as ChatIcon,
  CheckCircle as CheckIcon,
  CloudDownload as DownloadIcon,
  Memory as MemoryIcon,
  Mic as MicIcon,
  RecordVoiceOver as VoiceIcon,
  RocketLaunch as RocketIcon,
  Storage as StorageIcon,
  TextFields as EmbeddingIcon,
  TravelExplore as RagIcon,
} from '@mui/icons-material'
import { apiService } from '@/services/api'
import { useToast } from '@/contexts/ToastContext'
import type { ModelDownload } from '@/types/api'

// [2026-10-15] | INFO | WELCOME_WIZARD.MODULE | First-run flag helpers.
// The flag lives in plain localStorage (not useLocalPreference) because it is
// a one-shot marker, not a reusable UI preference.
const ONBOARDING_COMPLETE_FLAG = 'ln-onboarding-complete'
const ONBOARDING_RESET_EVENT = 'ln-onboarding-reset'

export const isOnboardingComplete = (): boolean => {
  if (typeof window === 'undefined') return true
  return window.localStorage.getItem(ONBOARDING_COMPLETE_FLAG) === 'true'
}

export const completeOnboarding = (): void => {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(ONBOARDING_COMPLETE_FLAG, 'true')
}

/**
 * [2026-10-15] | INFO | WELCOME_WIZARD.RESET_ONBOARDING | Clear the first-run
 * flag and notify any mounted wizard so it reopens. Wire this to a
 * "Restart setup" affordance wherever convenient.
 */
export const resetOnboarding = (): void => {
  if (typeof window === 'undefined') return
  window.localStorage.removeItem(ONBOARDING_COMPLETE_FLAG)
  window.dispatchEvent(new Event(ONBOARDING_RESET_EVENT))
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type UseCase = 'chat' | 'voice' | 'rag'

interface GpuInfo {
  index: number
  name: string
  vramTotalMb: number
}

interface HardwareSummary {
  gpuBacked: boolean
  gpus: GpuInfo[]
  cpuCount?: number
  ramTotalGb?: number
  source: 'gpus' | 'service-status' | 'default'
}

interface RecommendedVariant {
  filename: string
  quantization: string
  sizeBytes?: number
  minVramGb?: number
  recommendedVramGb?: number
  qualityScore?: number
}

interface RecommendedModel {
  repoId: string
  name: string
  description?: string
  modelType?: string
  variants: RecommendedVariant[]
}

interface SystemGpusResponse {
  available?: boolean
  gpus?: Array<{ index?: number; name?: string; vram_total_mb?: number }>
}

interface SystemResourcesResponse {
  cpu?: { count?: number }
  memory?: { total_mb?: number }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const GB_BYTES = 1024 ** 3

const WIZARD_STEPS = ['Welcome', 'Use Case', 'Models', 'Done']

const DEFAULT_VRAM_BUDGET_GB = 8

const formatGb = (bytes?: number): string => (
  typeof bytes === 'number' && bytes > 0 ? `${(bytes / GB_BYTES).toFixed(1)} GB` : '—'
)

const formatSpeed = (bytesPerSecond: number): string => (
  `${(bytesPerSecond / (1024 * 1024)).toFixed(1)} MB/s`
)

const formatEta = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds <= 0) return ''
  if (seconds >= 3600) {
    const hours = Math.floor(seconds / 3600)
    return `${hours}h ${Math.floor((seconds % 3600) / 60)}m left`
  }
  if (seconds >= 60) {
    return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s left`
  }
  return `${Math.round(seconds)}s left`
}

// [2026-10-15] | INFO | WELCOME_WIZARD.MODEL_ID_FOR_FILENAME | Mirror the
// backend download manager: model_id is the filename without extension
// (GGUF keeps its inner dots, e.g. "Model-Q4_K_M.gguf" -> "Model-Q4_K_M").
const modelIdForFilename = (filename: string): string => (
  filename.toLowerCase().endsWith('.gguf') ? filename.slice(0, -5) : filename.replace(/\.[^./]+$/, '')
)

// Mirrors the extension allowlist enforced by POST /v1/models/download.
const DIRECT_DOWNLOAD_RE = /(\.gguf|\.safetensors|\.bin|\.pth|\.pt|\.ninfer)(\.part\d+of\d+)?$/i
const isDirectDownloadable = (filename: string): boolean => DIRECT_DOWNLOAD_RE.test(filename)

interface UseCaseOption {
  id: UseCase
  label: string
  blurb: string
  icon: React.ReactNode
  color: string
  modelTypes: string[]
}

const USE_CASES: UseCaseOption[] = [
  {
    id: 'chat',
    label: 'Chat',
    blurb: 'Conversations, reasoning, and coding with a local LLM.',
    icon: <ChatIcon />,
    color: '#06b6d4',
    modelTypes: ['llm'],
  },
  {
    id: 'voice',
    label: 'Voice Assistant',
    blurb: 'Speech-to-text transcription for spoken interaction.',
    icon: <VoiceIcon />,
    color: '#10b981',
    modelTypes: ['stt'],
  },
  {
    id: 'rag',
    label: 'Knowledge RAG',
    blurb: 'Embed and chat over your own document collection.',
    icon: <RagIcon />,
    color: '#6366f1',
    modelTypes: ['embedding', 'llm'],
  },
]

const MODEL_TYPE_META: Record<string, { label: string; icon: React.ReactNode; color: string }> = {
  llm: { label: 'LLM', icon: <ChatIcon />, color: '#06b6d4' },
  embedding: { label: 'Embedding', icon: <EmbeddingIcon />, color: '#8b5cf6' },
  stt: { label: 'Speech', icon: <MicIcon />, color: '#10b981' },
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const WelcomeWizard: React.FC = () => {
  const navigate = useNavigate()
  const toast = useToast()

  const [open, setOpen] = useState(() => !isOnboardingComplete())
  const [activeStep, setActiveStep] = useState(0)
  const [hardware, setHardware] = useState<HardwareSummary | null>(null)
  const [hardwareLoading, setHardwareLoading] = useState(true)
  const [hardwareError, setHardwareError] = useState<string | null>(null)
  const [useCase, setUseCase] = useState<UseCase | null>(null)
  const [models, setModels] = useState<RecommendedModel[]>([])
  const [modelsLoading, setModelsLoading] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [showAllModels, setShowAllModels] = useState(false)
  const [variantSelection, setVariantSelection] = useState<Record<string, number>>({})
  const [downloads, setDownloads] = useState<Record<string, ModelDownload>>({})

  // Guards: one hardware fetch per epoch, one recommendations fetch per epoch.
  const fetchEpochRef = useRef(0)
  const hardwareFetchedEpochRef = useRef<number | null>(null)
  const modelsFetchedEpochRef = useRef<number | null>(null)

  // [2026-10-15] | INFO | WELCOME_WIZARD.USE_EFFECT_RESET_LISTENER | Reopen the
  // wizard when resetOnboarding() fires so "Restart setup" works live.
  useEffect(() => {
    const handleReset = () => {
      fetchEpochRef.current += 1
      setActiveStep(0)
      setUseCase(null)
      setModels([])
      setModelsError(null)
      setShowAllModels(false)
      setVariantSelection({})
      setDownloads({})
      setHardware(null)
      setHardwareError(null)
      setHardwareLoading(true)
      setOpen(true)
    }
    window.addEventListener(ONBOARDING_RESET_EVENT, handleReset)
    return () => window.removeEventListener(ONBOARDING_RESET_EVENT, handleReset)
  }, [])

  // [2026-10-15] | INFO | WELCOME_WIZARD.USE_EFFECT_HARDWARE | Step 1 data:
  // primary GET /api/v1/system/gpus (+ /api/v1/resources for CPU/RAM),
  // fallback apiService.getServiceStatus() to confirm the backend is reachable.
  useEffect(() => {
    if (!open) return
    const epoch = fetchEpochRef.current
    if (hardwareFetchedEpochRef.current === epoch) return
    hardwareFetchedEpochRef.current = epoch
    let cancelled = false

    const loadHardware = async () => {
      setHardwareLoading(true)
      setHardwareError(null)
      try {
        const [gpuSettled, resourcesSettled] = await Promise.allSettled([
          fetch('/api/v1/system/gpus'),
          fetch('/api/v1/resources'),
        ])
        if (cancelled || epoch !== fetchEpochRef.current) return

        let gpus: GpuInfo[] = []
        if (gpuSettled.status === 'fulfilled' && gpuSettled.value.ok) {
          const data = await gpuSettled.value.json() as SystemGpusResponse
          if (data.available && Array.isArray(data.gpus)) {
            gpus = data.gpus
              .filter((g): g is { index: number; name: string; vram_total_mb: number } => (
                typeof g.name === 'string' && typeof g.vram_total_mb === 'number'
              ))
              .map((g) => ({ index: g.index ?? 0, name: g.name, vramTotalMb: g.vram_total_mb }))
          }
        }

        let cpuCount: number | undefined
        let ramTotalGb: number | undefined
        if (resourcesSettled.status === 'fulfilled' && resourcesSettled.value.ok) {
          const data = await resourcesSettled.value.json() as SystemResourcesResponse
          if (typeof data.cpu?.count === 'number') cpuCount = data.cpu.count
          if (typeof data.memory?.total_mb === 'number') {
            ramTotalGb = Math.round((data.memory.total_mb / 1024) * 10) / 10
          }
        }

        if (cancelled || epoch !== fetchEpochRef.current) return

        if (gpus.length > 0 || cpuCount !== undefined || ramTotalGb !== undefined) {
          setHardware({
            gpuBacked: gpus.length > 0,
            gpus,
            cpuCount,
            ramTotalGb,
            source: 'gpus',
          })
        } else {
          // Fallback: prove the backend is reachable via service status.
          try {
            await apiService.getServiceStatus()
            if (cancelled || epoch !== fetchEpochRef.current) return
            setHardware({ gpuBacked: false, gpus: [], source: 'service-status' })
          } catch {
            if (cancelled || epoch !== fetchEpochRef.current) return
            setHardware({ gpuBacked: false, gpus: [], source: 'default' })
            setHardwareError('Could not detect hardware — using default recommendations.')
          }
        }
      } catch {
        if (!cancelled && epoch === fetchEpochRef.current) {
          setHardware({ gpuBacked: false, gpus: [], source: 'default' })
          setHardwareError('Could not detect hardware — using default recommendations.')
        }
      } finally {
        if (!cancelled && epoch === fetchEpochRef.current) {
          setHardwareLoading(false)
        }
      }
    }

    loadHardware()
    return () => { cancelled = true }
  }, [open])

  // [2026-10-15] | INFO | WELCOME_WIZARD.DETECTED_VRAM_GB | Memory budget used
  // for recommendations: largest GPU VRAM, or ~85% of system RAM on CPU-only.
  const detectedVramGb = useMemo(() => {
    if (!hardware) return null
    if (hardware.gpus.length > 0) {
      const maxVramMb = Math.max(...hardware.gpus.map((g) => g.vramTotalMb))
      return Math.round((maxVramMb / 1024) * 10) / 10
    }
    if (hardware.ramTotalGb && hardware.ramTotalGb > 2) {
      return Math.max(2, Math.floor(hardware.ramTotalGb * 0.85))
    }
    return null
  }, [hardware])

  const knownModelIds = useMemo(() => {
    const ids = new Set<string>()
    models.forEach((m) => m.variants.forEach((v) => ids.add(modelIdForFilename(v.filename))))
    return ids
  }, [models])

  const mergeDownloads = useCallback((list: ModelDownload[]) => {
    setDownloads((prev) => {
      const next = { ...prev }
      list.forEach((record) => {
        if (knownModelIds.has(record.modelId) || next[record.modelId]) {
          next[record.modelId] = record
        }
      })
      return next
    })
  }, [knownModelIds])

  // [2026-10-15] | INFO | WELCOME_WIZARD.USE_EFFECT_MODELS | Step 3 data:
  // curated registry recommendations + any pre-existing download records.
  useEffect(() => {
    if (!open || activeStep !== 2 || !hardware) return
    const epoch = fetchEpochRef.current
    if (modelsFetchedEpochRef.current === epoch) return
    modelsFetchedEpochRef.current = epoch
    let cancelled = false

    const loadModels = async () => {
      setModelsLoading(true)
      setModelsError(null)
      try {
        const vramGb = detectedVramGb ?? DEFAULT_VRAM_BUDGET_GB
        const { recommendations } = await apiService.getHardwareRecommendations(
          vramGb,
          hardware.ramTotalGb,
        )
        if (cancelled || epoch !== fetchEpochRef.current) return

        // Group flat variant rows into one card per repository.
        const grouped: RecommendedModel[] = []
        const byRepo = new Map<string, RecommendedModel>()
        for (const row of recommendations) {
          let entry = byRepo.get(row.repo_id)
          if (!entry) {
            entry = {
              repoId: row.repo_id,
              name: row.name,
              description: row.description,
              modelType: row.model_type,
              variants: [],
            }
            byRepo.set(row.repo_id, entry)
            grouped.push(entry)
          }
          if (!entry.variants.some((v) => v.filename === row.filename)) {
            entry.variants.push({
              filename: row.filename,
              quantization: row.quantization,
              sizeBytes: row.size_bytes,
              minVramGb: row.min_vram_gb,
              recommendedVramGb: row.recommended_vram_gb,
              qualityScore: row.quality_score,
            })
          }
        }
        setModels(grouped)
        setVariantSelection(Object.fromEntries(grouped.map((m) => [m.repoId, 0])))

        // Pick up downloads that already exist (e.g. started before the wizard).
        try {
          const existing = await apiService.getModelDownloads()
          if (!cancelled && epoch === fetchEpochRef.current) {
            setDownloads((prev) => {
              const next = { ...prev }
              const ids = new Set<string>()
              grouped.forEach((m) => m.variants.forEach((v) => ids.add(modelIdForFilename(v.filename))))
              existing.forEach((record) => {
                if (ids.has(record.modelId) || next[record.modelId]) {
                  next[record.modelId] = record
                }
              })
              return next
            })
          }
        } catch {
          // Non-fatal: progress polling will retry.
        }
      } catch (err) {
        if (!cancelled && epoch === fetchEpochRef.current) {
          setModelsError(err instanceof Error ? err.message : 'Failed to load recommendations')
        }
      } finally {
        if (!cancelled && epoch === fetchEpochRef.current) {
          setModelsLoading(false)
        }
      }
    }

    loadModels()
    return () => { cancelled = true }
  }, [open, activeStep, hardware, detectedVramGb])

  const hasActiveDownloads = useMemo(
    () => Object.values(downloads).some((d) => d.status === 'downloading' || d.status === 'queued'),
    [downloads],
  )

  // [2026-10-15] | INFO | WELCOME_WIZARD.USE_EFFECT_POLL | Poll download
  // progress while any tracked download is queued or active.
  useEffect(() => {
    if (!open || !hasActiveDownloads) return
    const interval = window.setInterval(async () => {
      try {
        mergeDownloads(await apiService.getModelDownloads())
      } catch {
        // Transient poll failure — the next tick retries.
      }
    }, 2000)
    return () => window.clearInterval(interval)
  }, [open, hasActiveDownloads, mergeDownloads])

  const closeWizard = useCallback((nextPath?: string) => {
    completeOnboarding()
    setOpen(false)
    if (nextPath) navigate(nextPath)
  }, [navigate])

  const handleNext = useCallback(() => {
    setActiveStep((prev) => Math.min(prev + 1, WIZARD_STEPS.length - 1))
  }, [])

  const handleBack = useCallback(() => {
    setActiveStep((prev) => Math.max(prev - 1, 0))
  }, [])

  // [2026-10-15] | INFO | WELCOME_WIZARD.HANDLE_DOWNLOAD | One-click download
  // via the existing model download API; progress comes from polling.
  const handleDownload = useCallback(async (model: RecommendedModel) => {
    const variant = model.variants[variantSelection[model.repoId] ?? 0]
    if (!variant || !isDirectDownloadable(variant.filename)) return
    try {
      const record = await apiService.downloadModel({
        repositoryId: model.repoId,
        filename: variant.filename,
        priority: 'normal',
      })
      setDownloads((prev) => ({ ...prev, [record.modelId]: record }))
      toast.success(`Download started: ${model.name}`, 'Model Download')
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : 'Failed to start download',
        'Model Download',
      )
    }
  }, [variantSelection, toast])

  const activeUseCase = USE_CASES.find((c) => c.id === useCase) ?? null

  const visibleModels = useMemo(() => {
    if (showAllModels || !activeUseCase) return models
    return models.filter((m) => activeUseCase.modelTypes.includes(m.modelType ?? 'llm'))
  }, [models, showAllModels, activeUseCase])

  const startedDownloadCount = Object.keys(downloads).length
  const completedDownloadCount = Object.values(downloads).filter((d) => d.status === 'completed').length

  const canAdvance = activeStep !== 1 || useCase !== null

  // -------------------------------------------------------------------------
  // Step renderers
  // -------------------------------------------------------------------------

  const renderWelcomeStep = () => (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <RocketIcon sx={{ fontSize: 44, color: 'primary.main' }} />
        <Box>
          <Typography variant="h5" fontWeight={600}>
            Welcome to llama-nexus
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Your local model workspace: deploy, chat, and search — all on your own hardware.
          </Typography>
        </Box>
      </Box>

      <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
        Detected hardware
      </Typography>

      {hardwareLoading ? (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 3 }}>
          <CircularProgress size={22} />
          <Typography variant="body2" color="text.secondary">
            Detecting GPUs and system memory...
          </Typography>
        </Box>
      ) : (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          {hardwareError && <Alert severity="warning">{hardwareError}</Alert>}

          {hardware?.gpuBacked && hardware.gpus.length > 0 ? (
            hardware.gpus.map((gpu) => (
              <Paper
                key={`${gpu.index}-${gpu.name}`}
                elevation={0}
                sx={{
                  p: 1.5,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.5,
                  bgcolor: alpha('#10b981', 0.08),
                  border: '1px solid',
                  borderColor: alpha('#10b981', 0.35),
                }}
              >
                <MemoryIcon sx={{ color: '#10b981' }} />
                <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                  <Typography variant="body2" fontWeight={600} noWrap>
                    {gpu.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {(gpu.vramTotalMb / 1024).toFixed(1)} GB VRAM
                  </Typography>
                </Box>
                <Chip size="small" label="GPU ready" sx={{ bgcolor: alpha('#10b981', 0.15) }} />
              </Paper>
            ))
          ) : (
            <Paper
              elevation={0}
              sx={{
                p: 1.5,
                display: 'flex',
                alignItems: 'center',
                gap: 1.5,
                bgcolor: alpha('#f59e0b', 0.08),
                border: '1px solid',
                borderColor: alpha('#f59e0b', 0.35),
              }}
            >
              <StorageIcon sx={{ color: '#f59e0b' }} />
              <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                <Typography variant="body2" fontWeight={600}>
                  CPU-only mode
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {hardware?.cpuCount ? `${hardware.cpuCount} CPU cores · ` : ''}
                  {hardware?.ramTotalGb ? `${hardware.ramTotalGb} GB RAM · ` : ''}
                  smaller quantized models work best
                </Typography>
              </Box>
              <Chip size="small" label="No GPU" sx={{ bgcolor: alpha('#f59e0b', 0.15) }} />
            </Paper>
          )}

          {detectedVramGb !== null && (
            <Typography variant="caption" color="text.secondary">
              Model recommendations in the next steps are sized for a{' '}
              <strong>{detectedVramGb} GB</strong> memory budget.
            </Typography>
          )}
        </Box>
      )}
    </Box>
  )

  const renderUseCaseStep = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        What do you want to do first?
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Pick a starting point — you can use every feature later.
      </Typography>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr 1fr' }, gap: 1.5 }}>
        {USE_CASES.map((option) => {
          const selected = useCase === option.id
          return (
            <Paper
              key={option.id}
              elevation={0}
              onClick={() => setUseCase(option.id)}
              sx={{
                p: 2,
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                gap: 1,
                height: '100%',
                border: '2px solid',
                borderColor: selected ? option.color : alpha('#94a3b8', 0.3),
                bgcolor: selected ? alpha(option.color, 0.08) : 'transparent',
                transition: 'border-color 0.15s ease, background-color 0.15s ease',
                '&:hover': { borderColor: option.color },
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box sx={{ color: option.color }}>{option.icon}</Box>
                {selected && <CheckIcon sx={{ color: option.color, fontSize: 20 }} />}
              </Box>
              <Typography variant="subtitle1" fontWeight={600}>
                {option.label}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {option.blurb}
              </Typography>
            </Paper>
          )
        })}
      </Box>
    </Box>
  )

  const renderDownloadState = (model: RecommendedModel) => {
    const variant = model.variants[variantSelection[model.repoId] ?? 0] ?? model.variants[0]
    if (!variant) return null

    // [2026-10-15] | INFO | WELCOME_WIZARD.RENDER_DOWNLOAD_STATE | STT ct2
    // bundles are fetched by the STT service, not /v1/models/download — route
    // the user to the Speech deploy page instead of a failing download call.
    if (!isDirectDownloadable(variant.filename)) {
      return (
        <Button
          size="small"
          variant="outlined"
          onClick={() => closeWizard('/stt-deploy')}
        >
          Set up in Speech
        </Button>
      )
    }

    const record = downloads[modelIdForFilename(variant.filename)]

    if (!record) {
      return (
        <Button
          size="small"
          variant="contained"
          startIcon={<DownloadIcon />}
          onClick={() => { void handleDownload(model) }}
        >
          Download
        </Button>
      )
    }

    if (record.status === 'completed') {
      return (
        <Chip
          size="small"
          color="success"
          icon={<CheckIcon />}
          label="Downloaded"
          variant="outlined"
        />
      )
    }

    if (record.status === 'failed' || record.status === 'cancelled') {
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, minWidth: 160 }}>
          <Alert severity="error" sx={{ py: 0 }}>
            {record.status === 'failed' ? 'Download failed' : 'Download cancelled'}
          </Alert>
          <Button size="small" startIcon={<DownloadIcon />} onClick={() => { void handleDownload(model) }}>
            Retry
          </Button>
        </Box>
      )
    }

    // queued or downloading
    return (
      <Box sx={{ minWidth: 200, maxWidth: 260 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="caption" color="text.secondary">
            {record.status === 'queued' ? 'Queued…' : `${Math.round(record.progress)}%`}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {record.status === 'downloading'
              ? `${formatSpeed(record.speed)}${record.eta > 0 ? ` · ${formatEta(record.eta)}` : ''}`
              : ''}
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={record.status === 'queued' ? 0 : record.progress}
          sx={{ height: 6, borderRadius: 3 }}
        />
      </Box>
    )
  }

  const renderModelStep = () => (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mb: 1.5 }}>
        <Typography variant="h6">
          Recommended models
          {activeUseCase && (
            <Typography component="span" variant="body2" color="text.secondary" sx={{ ml: 1 }}>
              for {activeUseCase.label}
            </Typography>
          )}
        </Typography>
        {models.length > 0 && activeUseCase && (
          <Chip
            size="small"
            label={showAllModels ? 'Curated only' : 'Show all models'}
            variant="outlined"
            onClick={() => setShowAllModels((prev) => !prev)}
          />
        )}
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Curated list from the local registry, filtered to your{' '}
        <strong>{detectedVramGb ?? DEFAULT_VRAM_BUDGET_GB} GB</strong> budget. Downloads run in the
        background — you can continue and pick more later.
      </Typography>

      {modelsLoading ? (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 4 }}>
          <CircularProgress size={22} />
          <Typography variant="body2" color="text.secondary">
            Fetching registry recommendations...
          </Typography>
        </Box>
      ) : modelsError ? (
        <Alert severity="error">{modelsError}</Alert>
      ) : visibleModels.length === 0 ? (
        <Alert severity="info">
          {models.length === 0
            ? 'No curated models matched your hardware budget. You can search HuggingFace from the Models page.'
            : 'No curated matches for this use case — toggle "Show all models" to see the full list.'}
        </Alert>
      ) : (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, maxHeight: 380, overflowY: 'auto', pr: 0.5 }}>
          {visibleModels.map((model) => {
            const typeMeta = MODEL_TYPE_META[model.modelType ?? 'llm'] ?? MODEL_TYPE_META.llm
            const selectedVariant = model.variants[variantSelection[model.repoId] ?? 0] ?? model.variants[0]
            const exceedsBudget = Boolean(
              selectedVariant?.minVramGb !== undefined
              && detectedVramGb !== null
              && selectedVariant.minVramGb > detectedVramGb,
            )
            return (
              <Paper
                key={model.repoId}
                elevation={0}
                sx={{
                  p: 1.5,
                  border: '1px solid',
                  borderColor: alpha('#94a3b8', 0.3),
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 1,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
                  <Box sx={{ color: typeMeta.color, mt: 0.5 }}>{typeMeta.icon}</Box>
                  <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                    <Typography variant="subtitle2" fontWeight={600} noWrap>
                      {model.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.75 }}>
                      {model.description ?? model.repoId}
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, alignItems: 'center' }}>
                      <Chip size="small" label={typeMeta.label} sx={{ bgcolor: alpha(typeMeta.color, 0.12) }} />
                      {selectedVariant && (
                        <>
                          <Chip size="small" variant="outlined" label={formatGb(selectedVariant.sizeBytes)} />
                          {selectedVariant.minVramGb !== undefined && (
                            <Chip
                              size="small"
                              variant="outlined"
                              label={`≥ ${selectedVariant.minVramGb} GB VRAM`}
                              sx={exceedsBudget ? { color: '#f59e0b', borderColor: alpha('#f59e0b', 0.5) } : undefined}
                            />
                          )}
                          {selectedVariant.qualityScore !== undefined && (
                            <Chip size="small" variant="outlined" label={`Quality ${selectedVariant.qualityScore}/5`} />
                          )}
                        </>
                      )}
                    </Box>
                  </Box>
                  <Box sx={{ flexShrink: 0, alignSelf: 'center' }}>
                    {renderDownloadState(model)}
                  </Box>
                </Box>

                {model.variants.length > 1 && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mr: 0.5 }}>
                      Variant:
                    </Typography>
                    {model.variants.map((variant, index) => {
                      const selected = (variantSelection[model.repoId] ?? 0) === index
                      return (
                        <Chip
                          key={variant.filename}
                          size="small"
                          label={variant.quantization || variant.filename}
                          color={selected ? 'primary' : 'default'}
                          variant={selected ? 'filled' : 'outlined'}
                          onClick={() => setVariantSelection((prev) => ({ ...prev, [model.repoId]: index }))}
                        />
                      )
                    })}
                  </Box>
                )}
              </Paper>
            )
          })}
        </Box>
      )}
    </Box>
  )

  const renderDoneStep = () => (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', py: 2, gap: 2 }}>
      <CheckIcon sx={{ fontSize: 56, color: '#10b981' }} />
      <Box>
        <Typography variant="h5" fontWeight={600} gutterBottom>
          You&apos;re all set
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 460 }}>
          {startedDownloadCount > 0
            ? `${startedDownloadCount} download${startedDownloadCount > 1 ? 's' : ''} started (${completedDownloadCount} finished). Progress continues in the background — track it on the Models page.`
            : 'No downloads were started — you can add models anytime from the Models or Registry pages.'}
          {' '}Launch a runtime on the Deploy page, then talk to it in Chat.
        </Typography>
      </Box>
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', justifyContent: 'center' }}>
        <Button variant="contained" startIcon={<RocketIcon />} onClick={() => closeWizard('/deploy')}>
          Open Deploy
        </Button>
        <Button variant="outlined" startIcon={<ChatIcon />} onClick={() => closeWizard('/chat')}>
          Start Chatting
        </Button>
      </Box>
    </Box>
  )

  const renderStepContent = () => {
    switch (activeStep) {
      case 0:
        return renderWelcomeStep()
      case 1:
        return renderUseCaseStep()
      case 2:
        return renderModelStep()
      case 3:
        return renderDoneStep()
      default:
        return null
    }
  }

  if (!open) return null

  return (
    <Dialog open={open} onClose={() => closeWizard()} maxWidth="md" fullWidth>
      <DialogTitle sx={{ pb: 1 }}>First-run setup</DialogTitle>
      <DialogContent>
        <Stepper activeStep={activeStep} sx={{ mb: 3, mt: 1 }}>
          {WIZARD_STEPS.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        {renderStepContent()}
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2, justifyContent: 'space-between' }}>
        <Button color="inherit" onClick={() => closeWizard()}>
          Skip setup
        </Button>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button disabled={activeStep === 0} onClick={handleBack}>
            Back
          </Button>
          {activeStep < WIZARD_STEPS.length - 1 ? (
            <Button variant="contained" disabled={!canAdvance} onClick={handleNext}>
              {activeStep === 2 ? 'Continue' : 'Next'}
            </Button>
          ) : (
            <Button variant="contained" onClick={() => closeWizard()}>
              Get Started
            </Button>
          )}
        </Box>
      </DialogActions>
    </Dialog>
  )
}

export default WelcomeWizard
