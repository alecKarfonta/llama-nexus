import React, { useState } from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Alert
} from '@mui/material'

interface AddTemplateDialogProps {
  open: boolean
  onClose: () => void
  onAdd: (filename: string, content: string) => Promise<void>
}

export const AddTemplateDialog: React.FC<AddTemplateDialogProps> = ({
  open,
  onClose,
  onAdd
}) => {
  const [filename, setFilename] = useState('')
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleClose = () => {
    if (!loading) {
      setFilename('')
      setContent('')
      setError('')
      onClose()
    }
  }

  const handleAdd = async () => {
    if (!filename.trim()) {
      setError('Filename is required')
      return
    }

    setLoading(true)
    setError('')

    try {
      await onAdd(filename.trim(), content)
      handleClose()
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || 'Failed to create template')
    } finally {
      setLoading(false)
    }
  }

  const getDefaultContent = () => {
    return `<|start|>system<|message|>You are a helpful AI assistant.
<|end|>
{{- range $i, $msg := .Messages }}
  {{- if eq $msg.Role "user" }}
<|start|>{{ $msg.Role }}<|message|>{{ $msg.Content }}<|end|>
  {{- else if eq $msg.Role "assistant" }}
<|start|>{{ $msg.Role }}<|message|>{{ $msg.Content }}<|end|>
  {{- end }}
{{- end }}
{{- if not .IsLastMessage }}
<|start|>assistant<|message|>
{{- end }}`
  }

  const handleUseDefault = () => {
    setContent(getDefaultContent())
  }

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle>Add New Template</DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
          {error && <Alert severity="error">{error}</Alert>}
          
          <TextField
            label="Template Name"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            placeholder="my-template"
            helperText="Will automatically add .jinja extension if not provided"
            fullWidth
            disabled={loading}
          />
          
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Button
              variant="outlined"
              size="small"
              onClick={handleUseDefault}
              disabled={loading}
            >
              Use Default Template
            </Button>
          </Box>
          
          <TextField
            label="Template Content"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            multiline
            minRows={12}
            placeholder="Enter your Jinja2 template content here..."
            fullWidth
            disabled={loading}
            sx={{ '& .MuiInputBase-root': { fontFamily: 'monospace' } }}
          />
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} disabled={loading}>
          Cancel
        </Button>
        <Button 
          onClick={handleAdd} 
          variant="contained" 
          disabled={loading || !filename.trim()}
        >
          {loading ? 'Creating...' : 'Create Template'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default AddTemplateDialog
