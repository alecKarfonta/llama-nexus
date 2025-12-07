import React, { useState, useEffect } from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Alert,
  CircularProgress,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Link,
  Autocomplete
} from '@mui/material'
import { CloudDownload as DownloadIcon, OpenInNew as ExternalIcon } from '@mui/icons-material'
import { apiService } from '@/services/api'
import type { ModelDownloadRequest } from '@/types/api'

interface DownloadModelDialogProps {
  open: boolean
  onClose: () => void
  onDownloadStart: () => void
}

export const DownloadModelDialog: React.FC<DownloadModelDialogProps> = ({
  open,
  onClose,
  onDownloadStart
}) => {
  const [repositoryId, setRepositoryId] = useState('')
  const [availableFiles, setAvailableFiles] = useState<string[]>([])
  const [selectedFile, setSelectedFile] = useState('')
  const [priority, setPriority] = useState<'low' | 'normal' | 'high'>('normal')
  const [loading, setLoading] = useState(false)
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Debounced file list fetching
  useEffect(() => {
    if (!repositoryId || repositoryId.split('/').length < 2) {
      setAvailableFiles([])
      setSelectedFile('')
      return
    }

    const timer = setTimeout(async () => {
      setLoadingFiles(true)
      setError(null)
      try {
        const files = await apiService.listRepoFiles(repositoryId)
        // Filter for model files (GGUF, safetensors, etc.)
        const modelFiles = files.filter(f => 
          f.endsWith('.gguf') || 
          f.endsWith('.safetensors') || 
          f.endsWith('.bin') ||
          f.endsWith('.pth')
        )
        setAvailableFiles(modelFiles)
        if (modelFiles.length === 1) {
          setSelectedFile(modelFiles[0])
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch repository files')
        setAvailableFiles([])
      } finally {
        setLoadingFiles(false)
      }
    }, 800)

    return () => clearTimeout(timer)
  }, [repositoryId])

  const handleClose = () => {
    if (loading) return
    setRepositoryId('')
    setSelectedFile('')
    setAvailableFiles([])
    setPriority('normal')
    setError(null)
    onClose()
  }

  const handleDownload = async () => {
    if (!repositoryId || !selectedFile) {
      setError('Please provide both repository ID and filename')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const request: ModelDownloadRequest = {
        repositoryId,
        filename: selectedFile,
        priority
      }
      
      await apiService.downloadModel(request)
      
      handleClose()
      onDownloadStart()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start download')
    } finally {
      setLoading(false)
    }
  }

  const isValid = repositoryId.trim() !== '' && selectedFile.trim() !== ''

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={1}>
          <DownloadIcon />
          <Typography variant="h6">Download HuggingFace Model</Typography>
        </Box>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5, mt: 1 }}>
          {error && <Alert severity="error">{error}</Alert>}
          
          <Alert severity="info" sx={{ fontSize: '0.875rem' }}>
            Download GGUF models from HuggingFace. Enter the repository ID (e.g., "TheBloke/Llama-2-7B-GGUF") 
            and select the model file you want to download.
          </Alert>

          <TextField
            label="Repository ID"
            value={repositoryId}
            onChange={(e) => setRepositoryId(e.target.value)}
            placeholder="e.g., TheBloke/Llama-2-7B-GGUF"
            helperText={
              <Box component="span">
                Find models at{' '}
                <Link 
                  href="https://huggingface.co/models?library=gguf" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}
                >
                  HuggingFace <ExternalIcon sx={{ fontSize: '0.875rem' }} />
                </Link>
              </Box>
            }
            fullWidth
            disabled={loading}
            required
          />

          {loadingFiles ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 2 }}>
              <CircularProgress size={20} />
              <Typography variant="body2" color="text.secondary">
                Loading available files...
              </Typography>
            </Box>
          ) : availableFiles.length > 0 ? (
            <Autocomplete
              options={availableFiles}
              value={selectedFile}
              onChange={(_, newValue) => setSelectedFile(newValue || '')}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Model File"
                  placeholder="Select a model file"
                  helperText={`${availableFiles.length} model file(s) available`}
                  required
                />
              )}
              disabled={loading}
              fullWidth
            />
          ) : repositoryId && !loadingFiles ? (
            <TextField
              label="Model File"
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              placeholder="e.g., llama-2-7b.Q4_K_M.gguf"
              helperText="Enter the filename manually or check if the repository exists"
              fullWidth
              disabled={loading}
              required
            />
          ) : (
            <TextField
              label="Model File"
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              placeholder="e.g., llama-2-7b.Q4_K_M.gguf"
              helperText="Enter repository ID first to see available files"
              fullWidth
              disabled={true}
              required
            />
          )}

          <FormControl fullWidth disabled={loading}>
            <InputLabel>Download Priority</InputLabel>
            <Select
              value={priority}
              label="Download Priority"
              onChange={(e) => setPriority(e.target.value as 'low' | 'normal' | 'high')}
            >
              <MenuItem value="low">Low</MenuItem>
              <MenuItem value="normal">Normal</MenuItem>
              <MenuItem value="high">High</MenuItem>
            </Select>
          </FormControl>

          <Box sx={{ 
            bgcolor: 'grey.50', 
            p: 2, 
            borderRadius: 1,
            border: '1px solid',
            borderColor: 'grey.200'
          }}>
            <Typography variant="caption" color="text.secondary" component="div">
              <strong>Examples:</strong>
            </Typography>
            <Typography variant="caption" color="text.secondary" component="div" sx={{ mt: 0.5 }}>
              • Repository: <code>TheBloke/Llama-2-7B-GGUF</code>
            </Typography>
            <Typography variant="caption" color="text.secondary" component="div">
              • Repository: <code>unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF</code>
            </Typography>
          </Box>
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={handleClose} disabled={loading}>
          Cancel
        </Button>
        <Button 
          onClick={handleDownload} 
          variant="contained" 
          disabled={loading || !isValid}
          startIcon={loading ? <CircularProgress size={16} /> : <DownloadIcon />}
        >
          {loading ? 'Starting Download...' : 'Download'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default DownloadModelDialog

