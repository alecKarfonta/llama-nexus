import React, { useEffect, useState } from 'react'
import { Box, Typography, Select, MenuItem, Button, TextField, Stack, Paper } from '@mui/material'
import { Add as AddIcon } from '@mui/icons-material'
import apiService from '@/services/api'
import AddTemplateDialog from '@/components/AddTemplateDialog'

export const TemplatesPage: React.FC = () => {
  const [files, setFiles] = useState<string[]>([])
  const [directory, setDirectory] = useState<string>('')
  const [selected, setSelected] = useState<string>('')
  const [content, setContent] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [addDialogOpen, setAddDialogOpen] = useState<boolean>(false)

  const loadList = async () => {
    setLoading(true)
    try {
      const data = await apiService.listTemplates()
      setFiles(data.files)
      setDirectory(data.directory)
      setSelected(data.selected)
      if (data.selected) {
        const tpl = await apiService.getTemplate(data.selected)
        setContent(tpl.content)
      } else if (data.files[0]) {
        const tpl = await apiService.getTemplate(data.files[0])
        setSelected(data.files[0])
        setContent(tpl.content)
      } else {
        setContent('')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadList()
  }, [])

  const handleSelect = async (filename: string) => {
    setSelected(filename)
    const tpl = await apiService.getTemplate(filename)
    setContent(tpl.content)
  }

  const handleSave = async () => {
    if (!selected) return
    await apiService.updateTemplate(selected, content)
  }

  const handleActivate = async () => {
    if (!selected) return
    await apiService.selectTemplate(selected)
    await loadList()
  }

  const handleAddTemplate = async (filename: string, content: string) => {
    await apiService.createTemplate(filename, content)
    await loadList()
    // Select the newly created template
    const createdFilename = filename.endsWith('.jinja') ? filename : `${filename}.jinja`
    setSelected(createdFilename)
    setContent(content)
  }

  return (
    <Box sx={{ p: 2, display: 'flex', gap: 2, alignItems: 'stretch' }}>
      <Paper sx={{ p: 2, width: 300 }}>
        <Typography variant="h6" gutterBottom>Templates</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>Dir: {directory}</Typography>
        <Select fullWidth size="small" value={selected} onChange={(e) => handleSelect(e.target.value as string)} disabled={loading}>
          {files.map((f) => (
            <MenuItem key={f} value={f}>{f}</MenuItem>
          ))}
        </Select>
        <Stack direction="row" spacing={1} sx={{ mt: 2 }}>
          <Button variant="outlined" onClick={handleActivate} disabled={!selected || loading}>Activate</Button>
          <Button variant="contained" onClick={handleSave} disabled={!selected || loading}>Save</Button>
        </Stack>
        <Button 
          variant="outlined" 
          startIcon={<AddIcon />} 
          onClick={() => setAddDialogOpen(true)} 
          disabled={loading}
          sx={{ mt: 1 }}
          fullWidth
        >
          Add Template
        </Button>
      </Paper>
      <Paper sx={{ p: 2, flex: 1, minHeight: '70vh' }}>
        <Typography variant="h6" gutterBottom>Editor</Typography>
        <TextField
          value={content}
          onChange={(e) => setContent(e.target.value)}
          multiline
          minRows={24}
          fullWidth
          placeholder="Select a template to view/edit"
        />
      </Paper>
      
      <AddTemplateDialog
        open={addDialogOpen}
        onClose={() => setAddDialogOpen(false)}
        onAdd={handleAddTemplate}
      />
    </Box>
  )
}

export default TemplatesPage


