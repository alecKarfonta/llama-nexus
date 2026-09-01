import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Grid,
  IconButton,
  List,
  Tooltip,
  Typography,
  useMediaQuery,
} from '@mui/material'
import { useTheme } from '@mui/material/styles'
import { alpha } from '@mui/material/styles'
import {
  Add as AddIcon,
  Article as TemplateIcon,
  CheckCircle as ActiveIcon,
  Code as CodeIcon,
  FolderOpen as FolderIcon,
  PlayArrow as ActivateIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon,
} from '@mui/icons-material'
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from '@/components/common'
import apiService from '@/services/api'
import AddTemplateDialog from '@/components/AddTemplateDialog'
import { accentColors } from '@/utils/theme'

const TEMPLATE_STATUS_CONFIG: StatusChipConfigMap = {
  active: { color: accentColors.success, label: 'Active template' },
  chars: { color: accentColors.info, label: 'Characters' },
  draft: { color: accentColors.warning, label: 'Draft selection' },
  empty: { color: accentColors.warning, label: 'No template' },
  files: { color: accentColors.success, label: 'Template files' },
  lines: { color: accentColors.primary, label: 'Lines' },
}

const formatNumber = (value: number): string => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`
  return value.toString()
}

const templateIconButtonSx = {
  width: 32,
  height: 32,
  borderRadius: '8px',
  border: '1px solid',
  borderColor: 'divider',
  bgcolor: 'action.hover',
  color: 'text.secondary',
  '&:hover': {
    bgcolor: 'action.selected',
  },
} as const

export function TemplatesPage() {
  const theme = useTheme()
  const isNarrow = useMediaQuery(theme.breakpoints.down('sm'))
  const [files, setFiles] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [activeTemplate, setActiveTemplate] = useState('')
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [addDialogOpen, setAddDialogOpen] = useState(false)

  const loadList = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiService.listTemplates()
      const nextSelected = data.selected || data.files[0] || ''

      setFiles(data.files)
      setActiveTemplate(data.selected || '')
      setSelected(nextSelected)

      if (nextSelected) {
        const template = await apiService.getTemplate(nextSelected)
        setContent(template.content)
      } else {
        setContent('')
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load chat templates')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadList()
  }, [loadList])

  const handleSelect = async (filename: string) => {
    setSelected(filename)
    setLoading(true)
    setError(null)
    try {
      const template = await apiService.getTemplate(filename)
      setContent(template.content)
    } catch (err: any) {
      setError(err.message || 'Failed to load template')
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (!selected) return
    setError(null)
    try {
      await apiService.updateTemplate(selected, content)
    } catch (err: any) {
      setError(err.message || 'Failed to save template')
    }
  }

  const handleActivate = async () => {
    if (!selected) return
    setError(null)
    try {
      await apiService.selectTemplate(selected)
      setActiveTemplate(selected)
      await loadList()
    } catch (err: any) {
      setError(err.message || 'Failed to activate template')
    }
  }

  const handleAddTemplate = async (filename: string, templateContent: string) => {
    setError(null)
    try {
      await apiService.createTemplate(filename, templateContent)
      await loadList()

      const createdFilename = filename.endsWith('.jinja') ? filename : `${filename}.jinja`
      setSelected(createdFilename)
      setContent(templateContent)
    } catch (err: any) {
      setError(err.message || 'Failed to create template')
      throw err
    }
  }

  const selectedIsActive = Boolean(selected && selected === activeTemplate)
  const lineCount = content ? content.split('\n').length : 0
  const characterCount = content.length
  const selectedStatus = selectedIsActive ? 'active' : selected ? 'draft' : 'empty'
  const editorSubtitle = selected
    ? selected
    : activeTemplate
      ? `Active: ${activeTemplate}`
      : 'Select a template to edit'

  return (
    <PageShell
      sx={{
        py: 1.25,
        height: { xs: 'auto', md: 'calc(100dvh - 56px)' },
        minHeight: { xs: 'calc(100dvh - 56px)', md: 0 },
        overflow: { xs: 'visible', md: 'hidden' },
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
      }}
    >
      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={1.5} sx={{ flex: { md: 1 }, minHeight: { xs: 'auto', md: 0 }, overflow: { xs: 'visible', md: 'hidden' } }}>
        <Grid item xs={12} md={3} lg={2.4}>
          <SectionCard
            title="Template files"
            subtitle={`${formatNumber(files.length)} available`}
            icon={<FolderIcon />}
            accentColor={accentColors.success}
            action={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Tooltip title="Reload templates">
                  <span>
                    <IconButton
                      aria-label="Reload templates"
                      size="small"
                      onClick={loadList}
                      disabled={loading}
                      sx={templateIconButtonSx}
                    >
                      <RefreshIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                  </span>
                </Tooltip>
                <Tooltip title="New template">
                  <span>
                    <IconButton
                      aria-label="New template"
                      size="small"
                      onClick={() => setAddDialogOpen(true)}
                      disabled={loading}
                      sx={templateIconButtonSx}
                    >
                      <AddIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                  </span>
                </Tooltip>
              </Box>
            }
          >
            <Box sx={{ maxHeight: { md: 360 }, overflowY: files.length > 8 ? 'auto' : 'visible' }}>
              {loading && files.length === 0 ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                  <CircularProgress size={24} />
                </Box>
              ) : files.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No templates found.
                </Typography>
              ) : (
                <List dense disablePadding sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 0.35 }}>
                  {files.map((file) => {
                    const isActive = file === activeTemplate
                    const isSelected = file === selected
                    return (
                      <Box
                        component="button"
                        key={file}
                        onClick={() => handleSelect(file)}
                        title={file}
                        aria-pressed={isSelected}
                        sx={{
                          appearance: 'none',
                          maxWidth: '100%',
                          minWidth: 0,
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 0.65,
                          borderRadius: '8px',
                          minHeight: 28,
                          px: 0.75,
                          py: 0.25,
                          border: '1px solid',
                          borderColor: isSelected ? alpha(accentColors.success, 0.36) : 'transparent',
                          bgcolor: isSelected ? alpha(accentColors.success, 0.12) : 'transparent',
                          color: 'text.primary',
                          font: 'inherit',
                          cursor: 'pointer',
                          textAlign: 'left',
                          '&:hover': { bgcolor: alpha(accentColors.success, 0.1) },
                          '&:focus-visible': {
                            outline: `2px solid ${alpha(accentColors.primary, 0.5)}`,
                            outlineOffset: 2,
                          },
                        }}
                      >
                        <Box sx={{ display: 'flex', color: isActive ? accentColors.success : 'text.secondary', '& .MuiSvgIcon-root': { fontSize: 16 } }}>
                          {isActive ? <ActiveIcon fontSize="small" /> : <TemplateIcon fontSize="small" />}
                        </Box>
                        <Typography
                          component="span"
                          sx={{
                            minWidth: 0,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            fontSize: '0.78rem',
                            fontWeight: 800,
                          }}
                        >
                          {file}
                        </Typography>
                      </Box>
                    )
                  })}
                </List>
              )}
            </Box>
          </SectionCard>
        </Grid>

        <Grid item xs={12} md={9} lg={9.6} sx={{ minHeight: 0 }}>
          <SectionCard
            title="Editor"
            subtitle={editorSubtitle}
            icon={<CodeIcon />}
            accentColor={accentColors.primary}
            collapsible={isNarrow}
            defaultExpanded={!isNarrow}
            sx={{
              height: { md: '100%' },
              minHeight: 0,
              '& .MuiCardContent-root': {
                height: { md: '100%' },
                minHeight: 0,
                display: 'flex',
                flexDirection: 'column',
                overflow: { xs: 'visible', md: 'hidden' },
              },
              '& .MuiCardContent-root > .MuiBox-root:last-child': {
                flex: { md: 1 },
                minHeight: { xs: 'auto', md: 0 },
                display: 'flex',
                flexDirection: 'column',
                overflow: { xs: 'visible', md: 'hidden' },
              },
            }}
            action={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                <StatusChip status={selectedStatus} statusConfig={TEMPLATE_STATUS_CONFIG} height={22} />
                <StatusChip status="lines" label={`${formatNumber(lineCount)} lines`} statusConfig={TEMPLATE_STATUS_CONFIG} />
                {!isNarrow && <StatusChip status="chars" label={`${formatNumber(characterCount)} chars`} statusConfig={TEMPLATE_STATUS_CONFIG} />}
                <Tooltip title={selectedIsActive ? 'Template is active' : 'Activate template'}>
                  <span>
                    {isNarrow ? (
                      <IconButton
                        aria-label={selectedIsActive ? 'Template is active' : 'Activate template'}
                        size="small"
                        onClick={handleActivate}
                        disabled={!selected || loading || selectedIsActive}
                        sx={{
                          ...templateIconButtonSx,
                          bgcolor: selectedIsActive ? 'action.hover' : 'primary.main',
                          color: selectedIsActive ? 'text.secondary' : 'primary.contrastText',
                          '&:hover': { bgcolor: selectedIsActive ? 'action.selected' : 'primary.dark' },
                        }}
                      >
                        <ActivateIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    ) : (
                      <Button
                        size="small"
                        variant={selectedIsActive ? 'outlined' : 'contained'}
                        onClick={handleActivate}
                        disabled={!selected || loading || selectedIsActive}
                        startIcon={<ActivateIcon sx={{ fontSize: 16 }} />}
                        sx={{
                          height: 30,
                          minWidth: 0,
                          borderRadius: '8px',
                          fontWeight: 800,
                          px: 1,
                          textTransform: 'none',
                          '& .MuiButton-startIcon': { mr: 0.4 },
                        }}
                      >
                        Activate
                      </Button>
                    )}
                  </span>
                </Tooltip>
                <Tooltip title="Save template">
                  <span>
                    <IconButton
                      aria-label="Save template"
                      size="small"
                      onClick={handleSave}
                      disabled={!selected || loading}
                      sx={{
                        ...templateIconButtonSx,
                        bgcolor: 'primary.main',
                        color: 'primary.contrastText',
                        '&:hover': { bgcolor: 'primary.dark' },
                        '&.Mui-disabled': {
                          bgcolor: 'action.disabledBackground',
                          color: 'text.disabled',
                        },
                      }}
                    >
                      <SaveIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                  </span>
                </Tooltip>
              </Box>
            }
          >
            <Box
              component="textarea"
              value={content}
              onChange={(event) => setContent(event.target.value)}
              disabled={loading || !selected}
              placeholder="Select a template to view/edit"
              sx={{
                width: '100%',
                flex: { md: 1 },
                minHeight: { xs: 320, md: 0 },
                resize: 'none',
                boxSizing: 'border-box',
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: '8px',
                bgcolor: 'background.default',
                color: 'text.primary',
                p: 1.5,
                fontFamily: 'monospace',
                fontSize: '0.86rem',
                lineHeight: 1.6,
                outline: 'none',
                overflow: 'auto',
                '&:focus': {
                  borderColor: accentColors.primary,
                  boxShadow: `0 0 0 1px ${alpha(accentColors.primary, 0.35)}`,
                },
                '&:disabled': {
                  color: 'text.disabled',
                },
              }}
            />
          </SectionCard>
        </Grid>
      </Grid>

      <AddTemplateDialog
        open={addDialogOpen}
        onClose={() => setAddDialogOpen(false)}
        onAdd={handleAddTemplate}
      />
    </PageShell>
  )
}

export default TemplatesPage
