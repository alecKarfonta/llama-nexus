import React, { useState, useEffect, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Typography,
  Box,
  Grid,
  Button,
  TextField,
  InputAdornment,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Stepper,
  Step,
  StepLabel,
  FormControl,
  FormLabel,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Alert,
  Snackbar,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Autocomplete,
  alpha,
} from '@mui/material'
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Search as SearchIcon,
  Delete as DeleteIcon,
  Info as InfoIcon,
  Cancel as CancelIcon,
  Tune as TuneIcon,
  RocketLaunch as DeployIcon,
} from '@mui/icons-material'
import type { QuantizationRouteState } from '@/types/quantizationHandoff'
import {
  deployableOutputForJob,
  modelInfoFromQuantizedOutput,
  quantizationJobNameFromModel,
  quantizationJobHandoffKey,
  quantizationModelHandoffKey,
} from '@/types/quantizationHandoff'
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from '@/components/common'
import { getRoutePath } from '@/config/navigation'
import { accentColors } from '@/utils/theme'

// Type definitions
interface QuantizationJob {
  id: string
  name: string
  description?: string
  source_model: string
  source_type: string
  output_formats: string[]
  gguf_quant_types: string[]
  gptq_bits: number[]
  awq_bits: number[]
  status: string
  progress: number
  current_step: string
  total_outputs: number
  completed_outputs: number
  outputs: QuantizedOutput[]
  created_at: string
  started_at?: string
  completed_at?: string
  error?: string
  estimated_disk_gb?: number
  estimated_time_minutes?: number
}

interface QuantizedOutput {
  id: string
  format: string
  quant_type: string
  file_path: string
  file_size: number
  status: string
  error?: string
  created_at: string
}

interface GGUFType {
  value: string
  name: string
  description: string
  size_factor: number
}

interface LocalModel {
  name: string
  localPath?: string
  repositoryId?: string
  framework?: string
  source?: string
  parameters?: string
}

interface ModelOption {
  label: string
  value: string
  type: 'local' | 'huggingface'
  parameters?: string
  source?: string
}

const QUANTIZATION_STATUS_CONFIG: StatusChipConfigMap = {
  active: { color: accentColors.warning, label: 'Active' },
  artifact: { color: accentColors.success, label: 'Artifacts' },
  cancelled: { color: accentColors.slate, label: 'Cancelled' },
  completed: { color: accentColors.success, label: 'Completed' },
  downloading: { color: accentColors.info, label: 'Downloading' },
  failed: { color: accentColors.rose, label: 'Failed' },
  lab: { color: accentColors.primary, label: 'Quantization lab' },
  outputs: { color: accentColors.purple, label: 'Outputs' },
  pipeline: { color: accentColors.info, label: 'Pipeline' },
  preparing: { color: accentColors.info, label: 'Preparing' },
  quantizing: { color: accentColors.warning, label: 'Quantizing' },
  queued: { color: accentColors.purple, label: 'Queued' },
  ready: { color: accentColors.success, label: 'Ready' },
}

export const QuantizationPage: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const deployPath = getRoutePath('deploy')
  const routeState = location.state as QuantizationRouteState | null
  const incomingJob = routeState?.job || routeState?.quantizationJob || null
  const incomingModel = routeState?.model || null
  const appliedHandoffRef = useRef('')
  const appliedModelHandoffRef = useRef('')
  const [jobs, setJobs] = useState<QuantizationJob[]>([])
  const [loading, setLoading] = useState(true)
  const [creatingJob, setCreatingJob] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [wizardOpen, setWizardOpen] = useState(false)
  const [infoDialogOpen, setInfoDialogOpen] = useState(false)
  const [selectedJob, setSelectedJob] = useState<QuantizationJob | null>(null)
  const [handoffNotice, setHandoffNotice] = useState<string | null>(null)
  const [snackbar, setSnackbar] = useState<{
    open: boolean
    message: string
    severity: 'success' | 'error' | 'info' | 'warning'
  }>({
    open: false,
    message: '',
    severity: 'info',
  })

  // Wizard state
  const [activeStep, setActiveStep] = useState(0)
  const [jobName, setJobName] = useState('')
  const [sourceModel, setSourceModel] = useState('')
  const [selectedFormats, setSelectedFormats] = useState<string[]>([])
  const [selectedGGUFTypes, setSelectedGGUFTypes] = useState<string[]>([])
  const [ggufTypes, setGGUFTypes] = useState<GGUFType[]>([])
  const [estimatedDisk, setEstimatedDisk] = useState<number>(0)
  const [estimatedTime, setEstimatedTime] = useState<number>(0)
  const [localModels, setLocalModels] = useState<LocalModel[]>([])
  const [modelInputValue, setModelInputValue] = useState('')

  const wizardSteps = ['Select Model', 'Choose Formats', 'Configure', 'Review']

  useEffect(() => {
    fetchJobs()
    fetchGGUFTypes()
    fetchLocalModels()
    const interval = setInterval(fetchJobs, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!incomingJob?.id || jobs.length === 0) return

    const handoffKey = quantizationJobHandoffKey(incomingJob)
    const matchedJob = jobs.find((job) => job.id === incomingJob.id)

    if (!matchedJob) {
      if (appliedHandoffRef.current !== `${handoffKey}:missing`) {
        setSearchTerm(incomingJob.name || incomingJob.id)
        setHandoffNotice(`${incomingJob.name || incomingJob.id} has been queued; refresh will surface it when the worker records the job.`)
        appliedHandoffRef.current = `${handoffKey}:missing`
      }
      return
    }

    if (appliedHandoffRef.current === `${handoffKey}:matched`) return

    setSelectedJob(matchedJob)
    setInfoDialogOpen(true)
    setSearchTerm('')
    setHandoffNotice(
      incomingJob.source === 'adapter-export'
        ? `${incomingJob.adapterName || matchedJob.name} export is open in the quantization pipeline.`
        : `${matchedJob.name} is open in the quantization pipeline.`
    )
    appliedHandoffRef.current = `${handoffKey}:matched`
  }, [
    incomingJob?.adapterName,
    incomingJob?.id,
    incomingJob?.name,
    incomingJob?.quantType,
    incomingJob?.source,
    jobs,
  ])

  useEffect(() => {
    if (!incomingModel?.sourceModel) return

    const handoffKey = quantizationModelHandoffKey(incomingModel)
    if (appliedModelHandoffRef.current === handoffKey) return
    appliedModelHandoffRef.current = handoffKey

    setWizardOpen(true)
    setActiveStep(0)
    setJobName(quantizationJobNameFromModel(incomingModel.name))
    setSourceModel(incomingModel.sourceModel)
    setModelInputValue(incomingModel.sourceModel)
    setSelectedFormats(['gguf'])
    setSelectedGGUFTypes(['Q4_K_M'])
    setHandoffNotice(`${incomingModel.name} is ready for GGUF quantization. Review the defaults and create a job when ready.`)
  }, [
    incomingModel?.localPath,
    incomingModel?.name,
    incomingModel?.repositoryId,
    incomingModel?.sourceModel,
    incomingModel?.sourceType,
  ])

  useEffect(() => {
    if (!selectedJob) return
    const refreshedJob = jobs.find((job) => job.id === selectedJob.id)
    if (refreshedJob) {
      setSelectedJob(refreshedJob)
    }
  }, [jobs, selectedJob?.id])

  const fetchJobs = async () => {
    try {
      const response = await fetch('/api/v1/quantize/jobs')
      const data = await response.json()
      setJobs(data.jobs || [])
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchGGUFTypes = async () => {
    try {
      const response = await fetch('/api/v1/quantize/formats/gguf/types')
      const data = await response.json()
      setGGUFTypes(data.types || [])
    } catch (error) {
      console.error('Failed to fetch GGUF types:', error)
    }
  }

  const fetchLocalModels = async () => {
    try {
      const response = await fetch('/v1/models')
      const data = await response.json()
      const models = Array.isArray(data) ? data : (data.data || [])
      // Filter to Transformers models (safetensors) - these can be quantized
      const transformersModels = models.filter((m: LocalModel) =>
        m.framework === 'transformers' || m.localPath?.includes('safetensors') || m.source === 'trained'
      )
      setLocalModels(transformersModels)
    } catch (error) {
      console.error('Failed to fetch local models:', error)
    }
  }

  // Build model options for Autocomplete
  const modelOptions: ModelOption[] = localModels.map((m) => ({
    label: `${m.name}${m.parameters ? ` (${m.parameters})` : ''}${m.source === 'trained' ? ' 🎓' : ''}`,
    value: m.localPath || m.repositoryId || m.name,
    type: 'local' as const,
    parameters: m.parameters,
    source: m.source,
  }))
  const selectedModelOption = modelOptions.find((option) => option.value === sourceModel)
  const selectedSourceType = selectedModelOption?.type || (incomingModel?.sourceModel === sourceModel ? incomingModel.sourceType : undefined) || 'huggingface'

  const handleWizardNext = () => {
    if (activeStep === wizardSteps.length - 2) {
      // Before final step, get estimate
      getEstimate()
    }
    setActiveStep((prev) => prev + 1)
  }

  const handleWizardBack = () => {
    setActiveStep((prev) => prev - 1)
  }

  const getEstimate = async () => {
    try {
      const response = await fetch('/api/v1/quantize/estimate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_model: sourceModel,
          output_formats: selectedFormats,
          gguf_quant_types: selectedGGUFTypes,
        }),
      })
      const data = await response.json()
      setEstimatedDisk(data.disk_space_gb || 0)
      setEstimatedTime(data.estimated_time_minutes || 0)
    } catch (error) {
      console.error('Failed to get estimate:', error)
    }
  }

  const handleSubmitJob = async () => {
    setCreatingJob(true)
    try {
      const response = await fetch('/api/v1/quantize/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: jobName,
          source_model: sourceModel,
          source_type: selectedSourceType,
          output_formats: selectedFormats,
          gguf_quant_types: selectedGGUFTypes,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to create job')
      }

      setSnackbar({
        open: true,
        message: 'Quantization job created successfully!',
        severity: 'success',
      })
      setWizardOpen(false)
      fetchJobs()
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to create job: ' + (error as Error).message,
        severity: 'error',
      })
    } finally {
      setCreatingJob(false)
    }
  }

  const defaultGGUFType = ggufTypes.some((type) => type.value === 'Q4_K_M')
    ? 'Q4_K_M'
    : ggufTypes[0]?.value || 'Q4_K_M'
  const quickCreateReady = Boolean(jobName.trim() && sourceModel.trim())

  useEffect(() => {
    if (incomingModel?.sourceModel || sourceModel || localModels.length === 0) return

    const firstModel = localModels[0]
    const modelValue = firstModel.localPath || firstModel.repositoryId || firstModel.name
    if (!modelValue) return

    setSourceModel(modelValue)
    setModelInputValue(modelOptions[0]?.label || modelValue)
    setJobName((current) => current || quantizationJobNameFromModel(firstModel.name || modelValue))
    setSelectedFormats((current) => current.length > 0 ? current : ['gguf'])
    setSelectedGGUFTypes((current) => current.length > 0 ? current : [defaultGGUFType])
  }, [defaultGGUFType, incomingModel?.sourceModel, localModels, modelOptions, sourceModel])

  const handleQuickCreateJob = async () => {
    if (!jobName.trim() || !sourceModel.trim()) return

    setCreatingJob(true)
    try {
      const response = await fetch('/api/v1/quantize/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: jobName.trim(),
          source_model: sourceModel.trim(),
          source_type: selectedSourceType,
          output_formats: ['gguf'],
          gguf_quant_types: [defaultGGUFType],
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to create job')
      }

      setSnackbar({
        open: true,
        message: 'Quantization job created successfully!',
        severity: 'success',
      })
      fetchJobs()
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to create job: ' + (error as Error).message,
        severity: 'error',
      })
    } finally {
      setCreatingJob(false)
    }
  }

  const handleOpenDraftInWizard = () => {
    setWizardOpen(true)
    setActiveStep(0)
    if (selectedFormats.length === 0) setSelectedFormats(['gguf'])
    if (selectedGGUFTypes.length === 0) setSelectedGGUFTypes([defaultGGUFType])
  }

  const handleCancelJob = async (jobId: string) => {
    try {
      await fetch(`/api/v1/quantize/jobs/${jobId}/cancel`, { method: 'POST' })
      setSnackbar({
        open: true,
        message: 'Job cancelled',
        severity: 'info',
      })
      fetchJobs()
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to cancel job',
        severity: 'error',
      })
    }
  }

  const handleDeleteJob = async (jobId: string) => {
    try {
      await fetch(`/api/v1/quantize/jobs/${jobId}`, { method: 'DELETE' })
      setSnackbar({
        open: true,
        message: 'Job deleted',
        severity: 'success',
      })
      fetchJobs()
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to delete job',
        severity: 'error',
      })
    }
  }

  const handleShowInfo = (job: QuantizationJob) => {
    setSelectedJob(job)
    setInfoDialogOpen(true)
  }

  const handleDeployOutput = (job: QuantizationJob, output: QuantizedOutput) => {
    navigate(deployPath, {
      state: {
        backend: 'llamacpp',
        model: modelInfoFromQuantizedOutput(job, output),
      },
    })
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
  }

  const filteredJobs = jobs.filter((job) =>
    job.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    job.source_model.toLowerCase().includes(searchTerm.toLowerCase())
  )
  const activeJobs = jobs.filter((job) => !['completed', 'failed', 'cancelled'].includes(job.status)).length
  const completedJobs = jobs.filter((job) => job.status === 'completed').length
  const failedJobs = jobs.filter((job) => job.status === 'failed').length
  const totalOutputs = jobs.reduce((total, job) => total + (job.total_outputs || 0), 0)
  const completedOutputs = jobs.reduce((total, job) => total + (job.completed_outputs || 0), 0)
  const latestJob = jobs[0]
  const visibleJobCountLabel = `${filteredJobs.length} visible`
  const hasJobSearch = searchTerm.trim().length > 0
  const showJobsPanel = loading || jobs.length > 0 || hasJobSearch
  const shouldShowJobSearch = jobs.length > 0 || hasJobSearch
  const prioritizeCreatePanel = !showJobsPanel
  const quantizationHeaderActions = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      {shouldShowJobSearch && (
        <TextField
          placeholder="Search jobs"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
          size="small"
          sx={{
            minWidth: { xs: '100%', sm: 220 },
            width: { xs: '100%', sm: 250 },
            '& .MuiInputBase-root': { height: 34, borderRadius: '8px' },
          }}
        />
      )}

      <Button
        size="small"
        variant="outlined"
        startIcon={<RefreshIcon />}
        onClick={fetchJobs}
        disabled={loading}
        sx={{ height: 34, borderRadius: '8px', borderColor: 'divider', fontWeight: 800, px: 1.25 }}
      >
        Refresh
      </Button>
    </Box>
  )
  const quantizationSummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      <StatusChip
        status={activeJobs > 0 ? 'active' : 'ready'}
        label={activeJobs > 0 ? `${activeJobs} active` : 'Ready'}
        statusConfig={QUANTIZATION_STATUS_CONFIG}
        height={22}
      />
      {jobs.length > 0 && (
        <StatusChip status="pipeline" label={`${jobs.length} jobs`} statusConfig={QUANTIZATION_STATUS_CONFIG} height={22} />
      )}
      {jobs.length > 0 && (
        <StatusChip status="outputs" label={visibleJobCountLabel} statusConfig={QUANTIZATION_STATUS_CONFIG} height={22} />
      )}
      {completedJobs > 0 && (
        <StatusChip status="completed" label={`${completedJobs} done`} statusConfig={QUANTIZATION_STATUS_CONFIG} height={22} />
      )}
      {totalOutputs > 0 && (
        <StatusChip
          status={failedJobs > 0 ? 'failed' : 'outputs'}
          label={`${completedOutputs}/${totalOutputs} outputs`}
          statusConfig={QUANTIZATION_STATUS_CONFIG}
          height={22}
        />
      )}
      {localModels.length > 0 && (
        <StatusChip status="lab" label={`${localModels.length} models`} statusConfig={QUANTIZATION_STATUS_CONFIG} height={22} />
      )}
      <Typography variant="caption" color="text.secondary" sx={{ minWidth: 0 }}>
        {latestJob ? `Latest: ${latestJob.name}` : `Default GGUF: ${defaultGGUFType}`}
      </Typography>
    </Box>
  )

  return (
    <PageShell
      sx={{
        py: 1.25,
        display: 'flex',
        flexDirection: 'column',
        gap: 1.25,
        overflow: 'visible',
      }}
    >
      {handoffNotice && (
        <Alert severity="info" onClose={() => setHandoffNotice(null)}>
          {handoffNotice}
        </Alert>
      )}

      <Grid
        container
        spacing={1.5}
        data-quantization-workspace
        sx={{ alignItems: 'flex-start', overflow: 'visible' }}
      >
        {showJobsPanel && (
          <Grid item xs={12} lg={prioritizeCreatePanel ? 5 : 8} sx={{ order: { xs: 2, lg: prioritizeCreatePanel ? 2 : 1 } }}>
            <SectionCard
              title="Jobs"
              icon={<TuneIcon />}
              accentColor={accentColors.primary}
              action={quantizationHeaderActions}
            >
              {quantizationSummaryRow}
              {loading && jobs.length === 0 ? (
                <Box sx={{ minHeight: 120, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <CircularProgress />
                </Box>
              ) : filteredJobs.length === 0 ? (
                <Box
                  data-quantization-empty-state
                  sx={{
                    p: 1.25,
                    display: 'flex',
                    alignItems: { xs: 'flex-start', sm: 'center' },
                    justifyContent: 'space-between',
                    gap: 1,
                    flexWrap: 'wrap',
                    bgcolor: alpha(accentColors.primary, 0.04),
                    borderRadius: '8px',
                    border: '1px dashed',
                    borderColor: 'divider',
                  }}
                >
                  <Box sx={{ minWidth: 0 }}>
                    <Typography variant="body2" sx={{ fontWeight: 800 }}>
                      {jobs.length === 0 ? 'No jobs' : 'No matching jobs'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {jobs.length === 0
                        ? 'Create a GGUF job.'
                        : 'Adjust the search term or refresh the job list.'}
                    </Typography>
                  </Box>
                  {jobs.length > 0 && (
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={() => setSearchTerm('')}
                      sx={{ height: 34, borderRadius: '8px', borderColor: 'divider', fontWeight: 800 }}
                    >
                      Clear search
                    </Button>
                  )}
                </Box>
              ) : (
                <Grid
                  container
                  spacing={1.25}
                  sx={{ alignContent: 'flex-start' }}
                >
                  {filteredJobs.map((job) => {
                    const deployableOutput = deployableOutputForJob(job)
                    const isHighlighted = incomingJob?.id === job.id || selectedJob?.id === job.id
                    const isActive = !['completed', 'failed', 'cancelled'].includes(job.status)

                    return (
                      <Grid item key={job.id} xs={12} sm={6} md={4}>
                        <Box
                          sx={{
                            height: '100%',
                            minHeight: 210,
                            p: 1.5,
                            display: 'flex',
                            flexDirection: 'column',
                            gap: 1.1,
                            borderRadius: '8px',
                            border: '1px solid',
                            borderColor: isHighlighted ? alpha(accentColors.primary, 0.65) : 'divider',
                            bgcolor: isHighlighted ? alpha(accentColors.primary, 0.08) : 'background.paper',
                            boxShadow: isHighlighted ? `0 0 0 1px ${alpha(accentColors.primary, 0.25)}` : undefined,
                            transition: 'border-color 0.2s ease, background-color 0.2s ease, transform 0.2s ease',
                            '&:hover': {
                              borderColor: alpha(accentColors.primary, 0.42),
                              transform: 'translateY(-1px)',
                            },
                          }}
                        >
                          <Box sx={{ minWidth: 0 }}>
                            <Typography variant="h6" title={job.name} sx={{ fontSize: '0.95rem', fontWeight: 700, mb: 0.5, wordBreak: 'break-word' }}>
                              {job.name}
                            </Typography>
                            <Typography color="text.secondary" sx={{ fontSize: '0.75rem', wordBreak: 'break-all' }}>
                              {job.source_model}
                            </Typography>
                          </Box>

                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                            <StatusChip status={job.status} statusConfig={QUANTIZATION_STATUS_CONFIG} height={22} />
                            <StatusChip
                              status="outputs"
                              label={`${job.completed_outputs}/${job.total_outputs}`}
                              statusConfig={QUANTIZATION_STATUS_CONFIG}
                              height={22}
                            />
                          </Box>

                          {isActive && (
                            <Box>
                              <Box display="flex" justifyContent="space-between" alignItems="center" gap={1} mb={0.5}>
                                <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.68rem', overflowWrap: 'anywhere' }}>
                                  {job.current_step}
                                </Typography>
                                <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.68rem', flexShrink: 0 }}>
                                  {Math.round(job.progress)}%
                                </Typography>
                              </Box>
                              <LinearProgress
                                variant="determinate"
                                value={job.progress}
                                sx={{
                                  height: 5,
                                  borderRadius: '3px',
                                  bgcolor: alpha(accentColors.primary, 0.12),
                                  '& .MuiLinearProgress-bar': {
                                    borderRadius: '3px',
                                    bgcolor: accentColors.primary,
                                  },
                                }}
                              />
                            </Box>
                          )}

                          {job.error && (
                            <Alert severity="error" sx={{ fontSize: '0.7rem' }}>
                              {job.error}
                            </Alert>
                          )}

                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 'auto' }}>
                            <Button size="small" onClick={() => handleShowInfo(job)} startIcon={<InfoIcon />} sx={{ borderRadius: '8px', fontSize: '0.72rem' }}>
                              Info
                            </Button>
                            {deployableOutput && (
                              <Button
                                size="small"
                                onClick={() => handleDeployOutput(job, deployableOutput)}
                                startIcon={<DeployIcon />}
                                sx={{ borderRadius: '8px', fontSize: '0.72rem' }}
                              >
                                Deploy
                              </Button>
                            )}
                            {isActive && (
                              <Button
                                size="small"
                                onClick={() => handleCancelJob(job.id)}
                                color="warning"
                                startIcon={<CancelIcon />}
                                sx={{ borderRadius: '8px', fontSize: '0.72rem' }}
                              >
                                Cancel
                              </Button>
                            )}
                            {!isActive && (
                              <Button
                                size="small"
                                onClick={() => handleDeleteJob(job.id)}
                                color="error"
                                startIcon={<DeleteIcon />}
                                sx={{ borderRadius: '8px', fontSize: '0.72rem' }}
                              >
                                Delete
                              </Button>
                            )}
                          </Box>
                        </Box>
                      </Grid>
                    )
                  })}
                </Grid>
              )}
            </SectionCard>
          </Grid>
        )}

        <Grid
          item
          xs={12}
          md={showJobsPanel ? 12 : 10}
          lg={showJobsPanel ? 4 : 8}
          sx={{ minHeight: 0, order: { xs: 1, lg: prioritizeCreatePanel ? 1 : 2 } }}
        >
          <SectionCard
            title="Create"
            icon={<AddIcon />}
            accentColor={accentColors.info}
            action={!showJobsPanel ? quantizationHeaderActions : undefined}
          >
            <Box
              sx={{
                display: 'grid',
                gap: 1.25,
                gridTemplateColumns: {
                  xs: '1fr',
                  sm: prioritizeCreatePanel || !showJobsPanel ? 'repeat(2, minmax(0, 1fr))' : '1fr',
                },
              }}
            >
              <TextField
                fullWidth
                size="small"
                label="Job name"
                value={jobName}
                onChange={(event) => setJobName(event.target.value)}
                placeholder="my-model-q4"
              />
              <Autocomplete
                freeSolo
                size="small"
                options={modelOptions}
                value={modelOptions.find((option) => option.value === sourceModel) || null}
                inputValue={modelInputValue}
                onInputChange={(_, newInputValue) => {
                  setModelInputValue(newInputValue)
                  setSourceModel(newInputValue)
                }}
                onChange={(_, newValue) => {
                  if (typeof newValue === 'string') {
                    setSourceModel(newValue)
                    setModelInputValue(newValue)
                  } else if (newValue) {
                    setSourceModel(newValue.value)
                    setModelInputValue(newValue.label)
                    if (!jobName && newValue.label) {
                      const baseName = newValue.label.split(' (')[0]
                      setJobName(`${baseName}-q4`)
                    }
                  } else {
                    setSourceModel('')
                    setModelInputValue('')
                  }
                }}
                getOptionLabel={(option) => (typeof option === 'string' ? option : option.label)}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Source model"
                    placeholder="Local model or HuggingFace ID"
                  />
                )}
              />
              <Box
                sx={{
                  display: 'flex',
                  gap: 0.75,
                  flexWrap: 'wrap',
                  gridColumn: { sm: prioritizeCreatePanel || !showJobsPanel ? '1 / -1' : 'auto' },
                }}
              >
                <StatusChip status="outputs" label="GGUF" statusConfig={QUANTIZATION_STATUS_CONFIG} height={22} />
                <StatusChip status="quantizing" label={defaultGGUFType} statusConfig={QUANTIZATION_STATUS_CONFIG} height={22} />
                <StatusChip status="pipeline" label={selectedSourceType} statusConfig={QUANTIZATION_STATUS_CONFIG} height={22} />
              </Box>
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  gap: 1,
                  flexWrap: 'wrap',
                  gridColumn: { sm: prioritizeCreatePanel || !showJobsPanel ? '1 / -1' : 'auto' },
                }}
              >
                <Button
                  variant="outlined"
                  size="small"
                  onClick={handleOpenDraftInWizard}
                  sx={{ height: 34, borderRadius: '8px', borderColor: 'divider', fontWeight: 800 }}
                >
                  Wizard
                </Button>
                {quickCreateReady ? (
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={creatingJob ? <CircularProgress size={14} color="inherit" /> : <AddIcon />}
                    onClick={handleQuickCreateJob}
                    disabled={creatingJob}
                    sx={{ height: 34, borderRadius: '8px', fontWeight: 800 }}
                  >
                    Create
                  </Button>
                ) : (
                  <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center', ml: 'auto' }}>
                    Select a source model
                  </Typography>
                )}
              </Box>
            </Box>
          </SectionCard>
        </Grid>
      </Grid>

      {/* Create Job Wizard Dialog */}
      <Dialog open={wizardOpen} onClose={() => setWizardOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create Quantization Job</DialogTitle>
        <DialogContent>
          <Stepper activeStep={activeStep} sx={{ mb: 3, mt: 1 }}>
            {wizardSteps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>

          {/* Step 0: Select Model */}
          {activeStep === 0 && (
            <Box>
              <TextField
                fullWidth
                label="Job Name"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                margin="normal"
                required
              />
              <Autocomplete
                freeSolo
                options={modelOptions}
                value={modelOptions.find(o => o.value === sourceModel) || null}
                inputValue={modelInputValue}
                onInputChange={(_, newInputValue) => {
                  setModelInputValue(newInputValue)
                  setSourceModel(newInputValue)
                }}
                onChange={(_, newValue) => {
                  if (typeof newValue === 'string') {
                    setSourceModel(newValue)
                    setModelInputValue(newValue)
                  } else if (newValue) {
                    setSourceModel(newValue.value)
                    setModelInputValue(newValue.label)
                    // Auto-set job name if empty
                    if (!jobName && newValue.label) {
                      const baseName = newValue.label.split(' (')[0]
                      setJobName(`${baseName}-quantized`)
                    }
                  } else {
                    setSourceModel('')
                  }
                }}
                getOptionLabel={(option) => {
                  if (typeof option === 'string') return option
                  return option.label
                }}
                renderOption={(props, option) => (
                  <li {...props} key={option.value}>
                    <Box>
                      <Typography variant="body2">{option.label}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {option.type === 'local' ? '📁 Local Model' : '🌐 HuggingFace'}
                        {option.value !== option.label && ` • ${option.value}`}
                      </Typography>
                    </Box>
                  </li>
                )}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    fullWidth
                    label="Source Model"
                    margin="normal"
                    required
                    placeholder="Select a local model or enter HuggingFace ID"
                    helperText={
                      localModels.length > 0
                        ? `${localModels.length} local Transformers models available, or enter a HuggingFace repo ID`
                        : "Enter a HuggingFace model repository ID (e.g., meta-llama/Llama-2-7b-hf)"
                    }
                  />
                )}
                sx={{ mt: 1 }}
              />
              {sourceModel && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  Selected: <strong>{sourceModel}</strong>
                </Alert>
              )}
            </Box>
          )}

          {/* Step 1: Choose Formats */}
          {activeStep === 1 && (
            <Box>
              <FormControl component="fieldset">
                <FormLabel component="legend">Select Output Formats</FormLabel>
                <FormGroup>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={selectedFormats.includes('gguf')}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedFormats([...selectedFormats, 'gguf'])
                          } else {
                            setSelectedFormats(selectedFormats.filter((f) => f !== 'gguf'))
                            setSelectedGGUFTypes([])
                          }
                        }}
                      />
                    }
                    label="GGUF (llama.cpp) - Recommended"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={selectedFormats.includes('gptq')} onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedFormats([...selectedFormats, 'gptq'])
                      } else {
                        setSelectedFormats(selectedFormats.filter((f) => f !== 'gptq'))
                      }
                    }} />}
                    label="GPTQ (Coming Soon)"
                    disabled
                  />
                  <FormControlLabel
                    control={<Checkbox checked={selectedFormats.includes('awq')} onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedFormats([...selectedFormats, 'awq'])
                      } else {
                        setSelectedFormats(selectedFormats.filter((f) => f !== 'awq'))
                      }
                    }} />}
                    label="AWQ (Coming Soon)"
                    disabled
                  />
                </FormGroup>
              </FormControl>
            </Box>
          )}

          {/* Step 2: Configure */}
          {activeStep === 2 && (
            <Box>
              {selectedFormats.includes('gguf') && (
                <Box>
                  <FormLabel component="legend" sx={{ mb: 1 }}>
                    Select GGUF Quantization Types
                  </FormLabel>
                  <FormGroup>
                    {ggufTypes.map((type) => (
                      <FormControlLabel
                        key={type.value}
                        control={
                          <Checkbox
                            checked={selectedGGUFTypes.includes(type.value)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedGGUFTypes([...selectedGGUFTypes, type.value])
                              } else {
                                setSelectedGGUFTypes(selectedGGUFTypes.filter((t) => t !== type.value))
                              }
                            }}
                          />
                        }
                        label={
                          <Box>
                            <Typography variant="body2">{type.name}</Typography>
                            <Typography variant="caption" color="text.secondary">
                              {type.description}
                            </Typography>
                          </Box>
                        }
                      />
                    ))}
                  </FormGroup>
                </Box>
              )}
              {selectedFormats.length === 0 && (
                <Alert severity="warning">Please select at least one output format</Alert>
              )}
            </Box>
          )}

          {/* Step 3: Review */}
          {activeStep === 3 && (
            <Box>
              <Alert severity="info" sx={{ mb: 2 }}>
                Review your configuration before submitting
              </Alert>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell><strong>Job Name</strong></TableCell>
                      <TableCell>{jobName}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell><strong>Source Model</strong></TableCell>
                      <TableCell>{sourceModel}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell><strong>Formats</strong></TableCell>
                      <TableCell>{selectedFormats.join(', ').toUpperCase()}</TableCell>
                    </TableRow>
                    {selectedFormats.includes('gguf') && (
                      <TableRow>
                        <TableCell><strong>GGUF Types</strong></TableCell>
                        <TableCell>{selectedGGUFTypes.join(', ')}</TableCell>
                      </TableRow>
                    )}
                    <TableRow>
                      <TableCell><strong>Estimated Disk Space</strong></TableCell>
                      <TableCell>{estimatedDisk.toFixed(1)} GB</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell><strong>Estimated Time</strong></TableCell>
                      <TableCell>{estimatedTime} minutes</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}
        </DialogContent>

        <DialogActions>
          <Button onClick={() => setWizardOpen(false)}>Cancel</Button>
          {activeStep > 0 && <Button onClick={handleWizardBack}>Back</Button>}
          {activeStep < wizardSteps.length - 1 && (
            <Button
              onClick={handleWizardNext}
              variant="contained"
              disabled={
                (activeStep === 0 && (!jobName || !sourceModel)) ||
                (activeStep === 1 && selectedFormats.length === 0) ||
                (activeStep === 2 && selectedFormats.includes('gguf') && selectedGGUFTypes.length === 0)
              }
            >
              Next
            </Button>
          )}
          {activeStep === wizardSteps.length - 1 && (
            <Button onClick={handleSubmitJob} variant="contained" color="primary" disabled={creatingJob}>
              {creatingJob ? 'Creating...' : 'Create Job'}
            </Button>
          )}
        </DialogActions>
      </Dialog>

      {/* Job Info Dialog */}
      <Dialog open={infoDialogOpen} onClose={() => setInfoDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Job Details</DialogTitle>
        <DialogContent>
          {selectedJob && (
            <Box>
              <Typography variant="h6" sx={{ mb: 2 }}>
                {selectedJob.name}
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Status
                  </Typography>
                  <StatusChip status={selectedJob.status} statusConfig={QUANTIZATION_STATUS_CONFIG} sx={{ mt: 0.5 }} />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Progress
                  </Typography>
                  <Typography variant="body1">{Math.round(selectedJob.progress)}%</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Source Model
                  </Typography>
                  <Typography variant="body1">{selectedJob.source_model}</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                    Outputs ({selectedJob.completed_outputs}/{selectedJob.total_outputs})
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Format</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>Size</TableCell>
                          <TableCell>Status</TableCell>
                          <TableCell align="right">Action</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {selectedJob.outputs.map((output) => (
                          <TableRow key={output.id}>
                            <TableCell>{output.format.toUpperCase()}</TableCell>
                            <TableCell>{output.quant_type}</TableCell>
                            <TableCell>{output.file_size > 0 ? formatBytes(output.file_size) : '-'}</TableCell>
                            <TableCell>
                              <StatusChip status={output.status} statusConfig={QUANTIZATION_STATUS_CONFIG} />
                            </TableCell>
                            <TableCell align="right">
                              {output.format === 'gguf' && output.status === 'completed' && output.file_path ? (
                                <Button
                                  size="small"
                                  startIcon={<DeployIcon />}
                                  onClick={() => handleDeployOutput(selectedJob, output)}
                                >
                                  Deploy
                                </Button>
                              ) : (
                                <Typography variant="caption" color="text.secondary">
                                  -
                                </Typography>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setInfoDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert onClose={() => setSnackbar({ ...snackbar, open: false })} severity={snackbar.severity} variant="filled">
          {snackbar.message}
        </Alert>
      </Snackbar>
    </PageShell>
  )
}
