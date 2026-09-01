import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Grid,
  Button,
  Alert,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  LinearProgress,
  Tooltip,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  useMediaQuery,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  CloudUpload as UploadIcon,
  Code as CodeIcon,
  Description as DocumentIcon,
  Check as CheckIcon,
  Close as CloseIcon,
  Warning as WarningIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  ExpandMore as ExpandIcon,
  AutoAwesome as ProcessingIcon,
  Analytics as AnalyticsIcon,
  AccountTree as GraphIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import {
  PageShell,
  SectionCard,
  StatusChip,
  type StatusChipConfigMap,
} from '@/components/common';
import { accentColors } from '@/utils/theme';
import { apiService as api } from '../services/api';

interface FileItem {
  file: File;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  result?: any;
  error?: string;
  fileType?: 'code' | 'document';
  language?: string | null;
}

interface SystemStatus {
  graphrag_healthy: boolean;
  code_rag_healthy: boolean;
  hybrid_available: boolean;
  code_rag_status?: {
    available: boolean;
    status: string;
    response: string;
  };
}

interface ExtractionStats {
  spanbert_available: boolean;
  dependency_available: boolean;
  entity_linking_available: boolean;
  available_methods: string[];
}

interface SupportedFormats {
  supported_formats: string[];
  features: {
    semantic_chunking: boolean;
    metadata_extraction: boolean;
    content_type_classification: boolean;
    structure_preservation: boolean;
  };
}

const HYBRID_STATUS_CONFIG: StatusChipConfigMap = {
  ingestion: { color: accentColors.info, label: 'Hybrid ingestion' },
  checking: { color: accentColors.info, label: 'Checking pipelines' },
  completed: { color: accentColors.success, label: 'Complete' },
  code: { color: accentColors.warning, label: 'Code' },
  domain: { color: accentColors.warning, label: 'Domain' },
  document: { color: accentColors.info, label: 'Document' },
  error: { color: accentColors.rose, label: 'Failed' },
  format: { color: accentColors.slate, label: 'Format' },
  graph: { color: accentColors.info, label: 'GraphRAG Only' },
  hybrid: { color: accentColors.success, label: 'Code RAG + GraphRAG' },
  language: { color: accentColors.warning, label: 'Language' },
  method: { color: accentColors.primary, label: 'Method' },
  offline: { color: accentColors.rose, label: 'Offline' },
  partial: { color: accentColors.warning, label: 'Partial routing' },
  pending: { color: accentColors.slate, label: 'Pending' },
  processing: { color: accentColors.info, label: 'Processing' },
  ready: { color: accentColors.success, label: 'Hybrid ready' },
};

const hybridPanelSx = {
  borderRadius: '8px',
  bgcolor: 'background.paper',
  border: '1px solid',
  borderColor: 'divider',
} as const;

export const HybridProcessingPage: React.FC = () => {
  const isShortViewport = useMediaQuery('(max-height:799.95px)');

  // System status
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [extractionStats, setExtractionStats] = useState<ExtractionStats | null>(null);
  const [supportedFormats, setSupportedFormats] = useState<SupportedFormats | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(true);
  
  // File processing
  const [files, setFiles] = useState<FileItem[]>([]);
  const [processing, setProcessing] = useState(false);
  const [domain, setDomain] = useState('general');
  
  // UI state
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // Load system status on mount
  useEffect(() => {
    loadSystemStatus();
  }, []);

  const loadSystemStatus = async () => {
    setLoadingStatus(true);
    try {
      const [status, stats, formats] = await Promise.all([
        api.getHybridStatus().catch(() => ({
          graphrag_healthy: false,
          code_rag_healthy: false,
          hybrid_available: false,
        })),
        api.getExtractionStats().catch(() => null),
        api.getSupportedFormats().catch(() => null),
      ]);
      
      setSystemStatus(status);
      setExtractionStats(stats);
      setSupportedFormats(formats);
    } catch (err) {
      setError('Failed to load system status');
    } finally {
      setLoadingStatus(false);
    }
  };

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    addFiles(selectedFiles);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    addFiles(droppedFiles);
  }, []);

  const addFiles = (newFiles: File[]) => {
    const fileItems: FileItem[] = newFiles.map(file => ({
      file,
      status: 'pending',
      progress: 0,
    }));
    setFiles(prev => [...prev, ...fileItems]);
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const clearCompleted = () => {
    setFiles(prev => prev.filter(f => f.status !== 'completed' && f.status !== 'error'));
  };

  const processFiles = async () => {
    if (files.length === 0) return;
    
    setProcessing(true);
    setError(null);
    
    for (let i = 0; i < files.length; i++) {
      const fileItem = files[i];
      if (fileItem.status !== 'pending') continue;
      
      // Update status to processing
      setFiles(prev => prev.map((f, idx) => 
        idx === i ? { ...f, status: 'processing', progress: 30 } : f
      ));
      
      try {
        const result = await api.hybridProcess(fileItem.file, domain);
        
        // Update with result
        setFiles(prev => prev.map((f, idx) => 
          idx === i ? { 
            ...f, 
            status: 'completed', 
            progress: 100,
            result,
            fileType: result.file_type,
            language: result.language,
          } : f
        ));
      } catch (err) {
        // Update with error
        setFiles(prev => prev.map((f, idx) => 
          idx === i ? { 
            ...f, 
            status: 'error', 
            progress: 100,
            error: err instanceof Error ? err.message : 'Processing failed',
          } : f
        ));
      }
    }
    
    setProcessing(false);
  };

  const getStatusIcon = (healthy: boolean) => {
    return healthy ? (
      <CheckIcon sx={{ color: 'success.main' }} />
    ) : (
      <CloseIcon sx={{ color: 'error.main' }} />
    );
  };

  const getFileIcon = (fileItem: FileItem) => {
    if (fileItem.fileType === 'code') {
      return <CodeIcon sx={{ color: accentColors.warning }} />;
    }
    return <DocumentIcon sx={{ color: accentColors.info }} />;
  };

  const getFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const isCodeFile = (filename: string) => {
    const codeExtensions = ['.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb', '.cs', '.swift', '.kt', '.scala'];
    return codeExtensions.some(ext => filename.toLowerCase().endsWith(ext));
  };

  const completedFiles = files.filter(f => f.status === 'completed');
  const pendingFiles = files.filter(f => f.status === 'pending');
  const processingFiles = files.filter(f => f.status === 'processing');
  const errorFiles = files.filter(f => f.status === 'error');
  const codeFileCount = files.filter(f => f.fileType === 'code' || isCodeFile(f.file.name)).length;
  const documentFileCount = Math.max(files.length - codeFileCount, 0);
  const showClearDoneAction = completedFiles.length > 0 || errorFiles.length > 0;
  const serviceStatus = loadingStatus
    ? 'checking'
    : systemStatus?.hybrid_available
      ? 'ready'
      : systemStatus?.graphrag_healthy
        ? 'partial'
        : 'offline';
  const serviceLabel = loadingStatus
    ? 'Checking pipelines'
    : systemStatus?.hybrid_available
      ? 'Hybrid ready'
      : systemStatus?.graphrag_healthy
        ? 'Partial routing'
        : 'Offline';
  const availableMethodCount = extractionStats?.available_methods?.length
    ?? [
      extractionStats?.spanbert_available,
      extractionStats?.dependency_available,
      extractionStats?.entity_linking_available,
    ].filter(Boolean).length;
  const showProcessAction = pendingFiles.length > 0 || processing;
  const pipelinesUnavailable = !loadingStatus
    && !systemStatus?.graphrag_healthy
    && !systemStatus?.code_rag_healthy
    && !systemStatus?.hybrid_available;
  const showPipelineTools = files.length > 0 || (!isShortViewport && !pipelinesUnavailable);
  const hybridHeaderActions = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      <FormControl size="small" sx={{ width: { xs: '100%', sm: 170 } }}>
        <InputLabel>Domain</InputLabel>
        <Select
          value={domain}
          onChange={(e) => setDomain(e.target.value)}
          label="Domain"
          sx={{ height: 34 }}
        >
          <MenuItem value="general">General</MenuItem>
          <MenuItem value="technical">Technical</MenuItem>
          <MenuItem value="legal">Legal</MenuItem>
          <MenuItem value="medical">Medical</MenuItem>
          <MenuItem value="financial">Financial</MenuItem>
        </Select>
      </FormControl>
      <Tooltip title="Refresh pipeline status">
        <span>
          <IconButton
            aria-label="Refresh pipeline status"
            size="small"
            onClick={loadSystemStatus}
            disabled={loadingStatus}
            sx={{
              width: 34,
              height: 34,
              borderRadius: '8px',
              border: '1px solid',
              borderColor: 'divider',
              bgcolor: 'action.hover',
              '&:hover': { bgcolor: alpha(accentColors.info, 0.12) },
            }}
          >
            {loadingStatus ? <CircularProgress size={16} /> : <RefreshIcon sx={{ fontSize: 18 }} />}
          </IconButton>
        </span>
      </Tooltip>
      {showProcessAction && (
        <Button
          variant="contained"
          size="small"
          onClick={processFiles}
          disabled={processing}
          startIcon={processing ? <CircularProgress size={16} /> : <ProcessingIcon />}
          sx={{ height: 34, borderRadius: '8px', fontWeight: 800, px: 1.25 }}
        >
          {processing ? 'Processing' : 'Process Queue'}
        </Button>
      )}
    </Box>
  );
  const hybridSummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      <StatusChip
        status={serviceStatus}
        label={serviceLabel}
        statusConfig={HYBRID_STATUS_CONFIG}
        height={22}
      />
      {files.length > 0 && (
        <StatusChip
          status="pending"
          label={`${files.length.toLocaleString()} queued`}
          statusConfig={HYBRID_STATUS_CONFIG}
          height={22}
        />
      )}
      {files.length > 0 && (
        <StatusChip
          status="code"
          label={`${codeFileCount} code / ${documentFileCount} docs`}
          statusConfig={HYBRID_STATUS_CONFIG}
          height={22}
        />
      )}
    </Box>
  );
  return (
    <PageShell
      sx={{
        py: 1.25,
        display: 'flex',
        flexDirection: 'column',
        gap: 1.25,
      }}
    >
      {error && (
        <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid
        container
        spacing={1.25}
        data-hybrid-processing-workspace
        sx={{ alignItems: 'flex-start' }}
      >
        <Grid item xs={12} lg={showPipelineTools ? 7.5 : 12} sx={{ display: files.length > 0 ? 'flex' : undefined }}>
          <SectionCard
            title="Ingestion queue"
            subtitle={
              files.length
                ? `${pendingFiles.length} pending, ${processingFiles.length} processing, ${completedFiles.length} complete, ${errorFiles.length} failed`
                : 'Add mixed code and document batches'
            }
            icon={<UploadIcon />}
            accentColor={accentColors.info}
            action={hybridHeaderActions}
            fullHeight={files.length > 0}
            sx={files.length > 0 ? {
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              '& .MuiCardContent-root': {
                flex: 1,
                minHeight: 0,
                display: 'flex',
                flexDirection: 'column',
              },
              '& .MuiCardContent-root > .MuiBox-root:last-child': {
                flex: 1,
                minHeight: 0,
                display: 'flex',
                flexDirection: 'column',
              },
            } : undefined}
          >
            {hybridSummaryRow}
            {showClearDoneAction && (
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  mb: 1,
                }}
              >
                <Button
                  size="small"
                  onClick={clearCompleted}
                  disabled={processing}
                  sx={{ height: 30, borderRadius: '8px', fontWeight: 800 }}
                >
                  Clear Done
                </Button>
              </Box>
            )}

            <Box
              component="label"
              htmlFor="hybrid-file-input"
              data-hybrid-processing-drop-zone
              sx={{
                flex: '0 0 auto',
                minHeight: files.length > 0 ? undefined : 132,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                p: { xs: 1.5, md: 1.75 },
                mb: files.length > 0 ? 2 : 0,
                border: '1px dashed',
                borderColor: dragOver ? accentColors.info : 'divider',
                borderRadius: '8px',
                textAlign: 'center',
                cursor: 'pointer',
                bgcolor: dragOver ? alpha(accentColors.info, 0.08) : 'background.paper',
                transition: 'all 0.2s ease',
              }}
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
            >
              <input
                type="file"
                multiple
                onChange={handleFileSelect}
                style={{ display: 'none' }}
                id="hybrid-file-input"
              />
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, mb: 1 }}>
                <CodeIcon sx={{ fontSize: 28, color: accentColors.warning }} />
                <DocumentIcon sx={{ fontSize: 28, color: accentColors.info }} />
              </Box>
              <Typography variant="body2" sx={{ fontWeight: 800 }}>
                Drop files or browse
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Code routes to Code RAG; documents route to GraphRAG.
              </Typography>
            </Box>

            {files.length > 0 && (
              <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: { xs: 'flex-start', sm: 'center' },
                    justifyContent: 'space-between',
                    flexDirection: { xs: 'column', sm: 'row' },
                    gap: 1,
                    mb: 1,
                  }}
                >
                  <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
                    Files ({files.length})
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                    <StatusChip status="pending" label={`${pendingFiles.length} pending`} statusConfig={HYBRID_STATUS_CONFIG} />
                    <StatusChip status="processing" label={`${processingFiles.length} processing`} statusConfig={HYBRID_STATUS_CONFIG} />
                    <StatusChip status="completed" label={`${completedFiles.length} complete`} statusConfig={HYBRID_STATUS_CONFIG} />
                    <StatusChip status="error" label={`${errorFiles.length} failed`} statusConfig={HYBRID_STATUS_CONFIG} />
                  </Box>
                </Box>
                <List sx={{ flex: 1, minHeight: 0, display: 'grid', gap: 1, overflowY: 'auto', alignContent: 'start', pr: 0.5 }}>
                  {files.map((fileItem, index) => (
                    <ListItem
                      key={index}
                      sx={{
                        bgcolor:
                          fileItem.status === 'completed'
                            ? alpha(accentColors.success, 0.08)
                            : fileItem.status === 'error'
                              ? alpha(accentColors.rose, 0.08)
                              : fileItem.status === 'processing'
                                ? alpha(accentColors.info, 0.08)
                                : 'background.paper',
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: '8px',
                      }}
                      secondaryAction={
                        <Tooltip title="Remove file">
                          <span>
                            <IconButton
                              edge="end"
                              size="small"
                              onClick={() => removeFile(index)}
                              disabled={fileItem.status === 'processing'}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </span>
                        </Tooltip>
                      }
                    >
                      <ListItemIcon>
                        {fileItem.status === 'processing' ? (
                          <CircularProgress size={24} />
                        ) : fileItem.status === 'completed' ? (
                          getFileIcon(fileItem)
                        ) : fileItem.status === 'error' ? (
                          <CloseIcon color="error" />
                        ) : (
                          isCodeFile(fileItem.file.name) ? (
                            <CodeIcon sx={{ color: accentColors.warning }} />
                          ) : (
                            <DocumentIcon sx={{ color: accentColors.info }} />
                          )
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', minWidth: 0 }}>
                            <Typography component="span" variant="body2" sx={{ fontWeight: 700, minWidth: 0 }} noWrap title={fileItem.file.name}>
                              {fileItem.file.name}
                            </Typography>
                            <StatusChip
                              status={fileItem.fileType || (isCodeFile(fileItem.file.name) ? 'code' : 'document')}
                              label={fileItem.fileType || (isCodeFile(fileItem.file.name) ? 'Code' : 'Document')}
                              statusConfig={HYBRID_STATUS_CONFIG}
                            />
                            <StatusChip
                              status={fileItem.status}
                              label={fileItem.status}
                              statusConfig={HYBRID_STATUS_CONFIG}
                            />
                            {fileItem.language && (
                              <StatusChip
                                status="language"
                                label={fileItem.language}
                                variant="outlined"
                                statusConfig={HYBRID_STATUS_CONFIG}
                              />
                            )}
                          </Box>
                        }
                        secondary={
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              {getFileSize(fileItem.file.size)}
                              {fileItem.error && (
                                <Typography component="span" variant="caption" color="error" sx={{ ml: 1 }}>
                                  {fileItem.error}
                                </Typography>
                              )}
                            </Typography>
                            {fileItem.status === 'processing' && (
                              <LinearProgress 
                                variant="indeterminate" 
                                sx={{ mt: 0.5, height: 2 }}
                              />
                            )}
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </SectionCard>
        </Grid>

        {showPipelineTools && (
        <Grid item xs={12} lg={4.5} sx={{ minHeight: { md: 0 } }}>
          <SectionCard
            title="Pipeline tools"
            subtitle="Routing health, extraction methods, and accepted formats."
            icon={<SettingsIcon />}
            accentColor={accentColors.warning}
            collapsible
            defaultExpanded={files.length > 0 || Boolean(systemStatus?.hybrid_available)}
          >
            {loadingStatus ? (
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', py: 5 }}>
                <CircularProgress />
              </Box>
            ) : (
              <Box>
                <List dense sx={{ py: 0 }}>
                  {[
                    {
                      label: 'GraphRAG',
                      healthy: systemStatus?.graphrag_healthy || false,
                      detail: systemStatus?.graphrag_healthy ? 'Connected' : 'Unavailable',
                    },
                    {
                      label: 'Code RAG',
                      healthy: systemStatus?.code_rag_healthy || false,
                      detail: systemStatus?.code_rag_healthy
                        ? 'Connected'
                        : systemStatus?.code_rag_status?.response || 'Unavailable',
                    },
                    {
                      label: 'Hybrid Mode',
                      healthy: systemStatus?.hybrid_available || false,
                      detail: systemStatus?.hybrid_available ? 'Full routing' : 'GraphRAG fallback',
                    },
                  ].map((service) => (
                    <ListItem
                      key={service.label}
                      sx={{
                        px: 0,
                        borderBottom: '1px solid',
                        borderColor: 'divider',
                        '&:last-of-type': { borderBottom: 0 },
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 38 }}>
                        {service.label === 'Hybrid Mode' && !service.healthy ? (
                          <WarningIcon sx={{ color: 'warning.main' }} />
                        ) : (
                          getStatusIcon(service.healthy)
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={service.label}
                        secondary={service.detail}
                        primaryTypographyProps={{ variant: 'body2', fontWeight: 700 }}
                        secondaryTypographyProps={{ variant: 'caption', noWrap: true, title: service.detail }}
                      />
                    </ListItem>
                  ))}
                </List>

                <Divider sx={{ my: 2 }} />

                <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                  Extraction methods
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 2 }}>
                  {extractionStats?.spanbert_available && (
                    <StatusChip status="method" label="SpanBERT" variant="outlined" statusConfig={HYBRID_STATUS_CONFIG} />
                  )}
                  {extractionStats?.dependency_available && (
                    <StatusChip status="method" label="Dependency Parsing" variant="outlined" statusConfig={HYBRID_STATUS_CONFIG} />
                  )}
                  {extractionStats?.entity_linking_available && (
                    <StatusChip status="method" label="Entity Linking" variant="outlined" statusConfig={HYBRID_STATUS_CONFIG} />
                  )}
                  {!availableMethodCount && (
                    <StatusChip status="method" label="Basic extraction" variant="outlined" statusConfig={HYBRID_STATUS_CONFIG} />
                  )}
                </Box>

                {supportedFormats && (
                  <>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                      Supported formats
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 2 }}>
                      {supportedFormats.supported_formats.map(fmt => (
                        <StatusChip key={fmt} status="format" label={fmt} variant="outlined" statusConfig={HYBRID_STATUS_CONFIG} />
                      ))}
                    </Box>

                    <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                      Format capabilities
                    </Typography>
                    <List dense sx={{ py: 0 }}>
                      {Object.entries(supportedFormats.features).map(([key, value]) => (
                        <ListItem key={key} sx={{ px: 0, py: 0.25 }}>
                          <ListItemIcon sx={{ minWidth: 32 }}>
                            {value ? <CheckIcon fontSize="small" color="success" /> : <CloseIcon fontSize="small" color="disabled" />}
                          </ListItemIcon>
                          <ListItemText
                            primary={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            primaryTypographyProps={{ variant: 'body2' }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </>
                )}
              </Box>
            )}
          </SectionCard>
        </Grid>
        )}

        {completedFiles.length > 0 && (
          <Grid item xs={12}>
            <SectionCard
              title="Processing results"
              subtitle={`${completedFiles.length} completed file${completedFiles.length === 1 ? '' : 's'} ready for review.`}
              icon={<AnalyticsIcon />}
              accentColor={accentColors.info}
            >
              {completedFiles.map((fileItem, index) => (
                <Accordion key={index} defaultExpanded={index === 0}>
                  <AccordionSummary expandIcon={<ExpandIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', minWidth: 0 }}>
                      {getFileIcon(fileItem)}
                      <Typography sx={{ fontWeight: 700 }} noWrap title={fileItem.file.name}>
                        {fileItem.file.name}
                      </Typography>
                      <StatusChip
                        status={fileItem.fileType === 'code' ? 'hybrid' : 'graph'}
                        label={fileItem.fileType === 'code' ? 'Code RAG + GraphRAG' : 'GraphRAG Only'}
                        statusConfig={HYBRID_STATUS_CONFIG}
                      />
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={fileItem.result?.code_rag_processing ? 6 : 12}>
                        <Box
                          sx={{
                            ...hybridPanelSx,
                            height: '100%',
                            p: 2,
                            bgcolor: alpha(accentColors.success, 0.05),
                          }}
                        >
                          <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                            <GraphIcon sx={{ mr: 0.5, fontSize: 18, verticalAlign: 'middle' }} />
                            GraphRAG Processing
                          </Typography>
                          {fileItem.result?.graphrag_processing?.success ? (
                            <Box>
                              <Alert severity="success" sx={{ mb: 1 }}>
                                Indexed in knowledge graph
                              </Alert>
                              {fileItem.result?.graphrag_processing?.response && (
                                <Typography variant="body2" color="text.secondary">
                                  {JSON.stringify(fileItem.result.graphrag_processing.response, null, 2).substring(0, 200)}...
                                </Typography>
                              )}
                            </Box>
                          ) : (
                            <Alert severity="error">
                              {fileItem.result?.graphrag_processing?.error || 'Processing failed'}
                            </Alert>
                          )}
                        </Box>
                      </Grid>

                      {fileItem.result?.code_rag_processing && (
                        <Grid item xs={12} md={6}>
                          <Box
                            sx={{
                              ...hybridPanelSx,
                              height: '100%',
                              p: 2,
                              bgcolor: alpha(accentColors.warning, 0.05),
                            }}
                          >
                            <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                              <CodeIcon sx={{ mr: 0.5, fontSize: 18, verticalAlign: 'middle' }} />
                              Code RAG Processing
                            </Typography>
                            {fileItem.result?.code_rag_processing?.success ? (
                              <Box>
                                <Alert severity="success" sx={{ mb: 1 }}>
                                  Indexed for code search
                                </Alert>
                                {fileItem.language && (
                                  <StatusChip
                                    status="language"
                                    label={`Language: ${fileItem.language}`}
                                    statusConfig={HYBRID_STATUS_CONFIG}
                                    sx={{ mt: 1 }}
                                  />
                                )}
                              </Box>
                            ) : (
                              <Alert severity="warning">
                                {fileItem.result?.code_rag_processing?.error || 'Code RAG not available'}
                              </Alert>
                            )}
                          </Box>
                        </Grid>
                      )}
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              ))}
            </SectionCard>
          </Grid>
        )}
      </Grid>
    </PageShell>
  );
};
