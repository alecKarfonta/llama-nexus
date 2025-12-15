import React, { useState, useEffect } from 'react'
import {
  Typography,
  Box,
  Grid,
  Button,
  Card,
  CardContent,
  CardActions,
  Chip,
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
  Tooltip,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material'
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Search as SearchIcon,
  Delete as DeleteIcon,
  Info as InfoIcon,
  Cancel as CancelIcon,
  CloudDownload as DownloadIcon,
  FolderOpen as FolderIcon,
} from '@mui/icons-material'
import { apiService } from '@/services/api'

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

export const QuantizationPage: React.FC = () => {
  const [jobs, setJobs] = useState<QuantizationJob[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [wizardOpen, setWizardOpen] = useState(false)
  const [infoDialogOpen, setInfoDialogOpen] = useState(false)
  const [selectedJob, setSelectedJob] = useState<QuantizationJob | null>(null)
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

  const wizardSteps = ['Select Model', 'Choose Formats', 'Configure', 'Review']

  useEffect(() => {
    fetchJobs()
    fetchGGUFTypes()
    const interval = setInterval(fetchJobs, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

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

  const handleCreateJob = () => {
    setWizardOpen(true)
    setActiveStep(0)
    setJobName('')
    setSourceModel('')
    setSelectedFormats([])
    setSelectedGGUFTypes([])
  }

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
    try {
      const response = await fetch('/api/v1/quantize/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: jobName,
          source_model: sourceModel,
          source_type: 'huggingface',
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
    }
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

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
  }

  const getStatusColor = (status: string) => {
    const colors: Record<string, 'default' | 'info' | 'warning' | 'success' | 'error'> = {
      queued: 'default',
      downloading: 'info',
      preparing: 'info',
      quantizing: 'warning',
      completed: 'success',
      failed: 'error',
      cancelled: 'default',
    }
    return colors[status] || 'default'
  }

  const filteredJobs = jobs.filter((job) =>
    job.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    job.source_model.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <Box sx={{ width: '100%', px: 3, py: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography variant="h1" sx={{ fontWeight: 700, fontSize: { xs: '1.25rem', sm: '1.5rem' }, mb: 0.5 }}>
            Model Quantization
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', fontSize: '0.8125rem' }}>
            Convert models to optimized formats (GGUF, GPTQ, AWQ)
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchJobs}
            disabled={loading}
            size="small"
            sx={{ fontSize: '0.75rem' }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleCreateJob}
            size="small"
            sx={{ fontSize: '0.75rem' }}
          >
            New Job
          </Button>
        </Box>
      </Box>

      {/* Search */}
      <Box sx={{ mb: 3 }}>
        <TextField
          fullWidth
          placeholder="Search quantization jobs..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
          sx={{ maxWidth: 500 }}
          size="small"
        />
      </Box>

      {/* Jobs Grid */}
      {loading && jobs.length === 0 ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={2}>
          {filteredJobs.map((job) => (
            <Grid item key={job.id} xs={12} sm={6} md={4} lg={3}>
              <Card variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" noWrap title={job.name} sx={{ fontSize: '0.95rem', fontWeight: 600, mb: 0.5 }}>
                    {job.name}
                  </Typography>
                  <Typography color="textSecondary" gutterBottom sx={{ fontSize: '0.75rem' }} noWrap>
                    {job.source_model}
                  </Typography>

                  <Box display="flex" alignItems="center" gap={0.5} mb={1} flexWrap="wrap">
                    <Chip label={job.status} size="small" color={getStatusColor(job.status)} sx={{ fontSize: '0.7rem', height: 20 }} />
                    <Chip
                      label={`${job.completed_outputs}/${job.total_outputs}`}
                      size="small"
                      variant="outlined"
                      sx={{ fontSize: '0.7rem', height: 20 }}
                    />
                  </Box>

                  {job.status !== 'completed' && job.status !== 'failed' && job.status !== 'cancelled' && (
                    <Box sx={{ mt: 1 }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                          {job.current_step}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                          {Math.round(job.progress)}%
                        </Typography>
                      </Box>
                      <LinearProgress variant="determinate" value={job.progress} sx={{ height: 4, borderRadius: 2 }} />
                    </Box>
                  )}

                  {job.error && (
                    <Alert severity="error" sx={{ mt: 1, fontSize: '0.7rem' }}>
                      {job.error}
                    </Alert>
                  )}
                </CardContent>

                <CardActions sx={{ pt: 0, pb: 1, px: 1, gap: 0.5 }}>
                  <Button size="small" onClick={() => handleShowInfo(job)} startIcon={<InfoIcon />} sx={{ fontSize: '0.7rem' }}>
                    Info
                  </Button>
                  {job.status !== 'completed' && job.status !== 'failed' && job.status !== 'cancelled' && (
                    <Button
                      size="small"
                      onClick={() => handleCancelJob(job.id)}
                      color="warning"
                      startIcon={<CancelIcon />}
                      sx={{ fontSize: '0.7rem' }}
                    >
                      Cancel
                    </Button>
                  )}
                  {(job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') && (
                    <Button
                      size="small"
                      onClick={() => handleDeleteJob(job.id)}
                      color="error"
                      startIcon={<DeleteIcon />}
                      sx={{ fontSize: '0.7rem' }}
                    >
                      Delete
                    </Button>
                  )}
                </CardActions>
              </Card>
            </Grid>
          ))}

          {filteredJobs.length === 0 && !loading && (
            <Grid item xs={12}>
              <Box sx={{ textAlign: 'center', py: 6, bgcolor: 'background.paper', borderRadius: 1, border: '1px dashed', borderColor: 'grey.300' }}>
                <Typography variant="h6" color="text.secondary" sx={{ mb: 1, fontWeight: 600, fontSize: '1rem' }}>
                  No quantization jobs found
                </Typography>
                <Button variant="contained" startIcon={<AddIcon />} onClick={handleCreateJob} size="small" sx={{ fontSize: '0.75rem', mt: 2 }}>
                  Create Your First Job
                </Button>
              </Box>
            </Grid>
          )}
        </Grid>
      )}

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
              <TextField
                fullWidth
                label="Source Model"
                value={sourceModel}
                onChange={(e) => setSourceModel(e.target.value)}
                margin="normal"
                required
                placeholder="e.g., meta-llama/Llama-2-7b-hf"
                helperText="Enter a HuggingFace model repository ID"
              />
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
            <Button onClick={handleSubmitJob} variant="contained" color="primary">
              Create Job
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
                  <Chip label={selectedJob.status} color={getStatusColor(selectedJob.status)} size="small" sx={{ mt: 0.5 }} />
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
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {selectedJob.outputs.map((output) => (
                          <TableRow key={output.id}>
                            <TableCell>{output.format.toUpperCase()}</TableCell>
                            <TableCell>{output.quant_type}</TableCell>
                            <TableCell>{output.file_size > 0 ? formatBytes(output.file_size) : '-'}</TableCell>
                            <TableCell>
                              <Chip label={output.status} size="small" color={output.status === 'completed' ? 'success' : 'default'} />
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
    </Box>
  )
}
