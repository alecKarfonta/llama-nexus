import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Alert,
  CircularProgress,
  Chip,
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
  alpha,
  useTheme,
} from '@mui/material';
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
  Hub as HubIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  AutoAwesome as ProcessingIcon,
  Analytics as AnalyticsIcon,
  AccountTree as GraphIcon,
  Functions as FunctionIcon,
  Link as LinkIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
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

export const HybridProcessingPage: React.FC = () => {
  const theme = useTheme();
  
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
      return <CodeIcon sx={{ color: '#f97316' }} />;
    }
    return <DocumentIcon sx={{ color: '#3b82f6' }} />;
  };

  const getFileStatusColor = (status: FileItem['status']) => {
    switch (status) {
      case 'completed': return 'success';
      case 'error': return 'error';
      case 'processing': return 'primary';
      default: return 'default';
    }
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

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <HubIcon sx={{ mr: 1, fontSize: 32 }} />
        <Typography variant="h4">
          Hybrid Processing
        </Typography>
        <IconButton onClick={loadSystemStatus} sx={{ ml: 2 }} disabled={loadingStatus}>
          <RefreshIcon />
        </IconButton>
      </Box>

      <Typography variant="body1" color="text.secondary" paragraph>
        Intelligently route files to the appropriate processing pipeline - code files to Code RAG for specialized indexing, documents to GraphRAG for knowledge graph building.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* System Status */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              <SettingsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              System Status
            </Typography>
            
            {loadingStatus ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}>
                <CircularProgress />
              </Box>
            ) : (
              <Box>
                {/* Service Status */}
                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      {getStatusIcon(systemStatus?.graphrag_healthy || false)}
                    </ListItemIcon>
                    <ListItemText 
                      primary="GraphRAG" 
                      secondary={systemStatus?.graphrag_healthy ? 'Connected' : 'Unavailable'}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      {getStatusIcon(systemStatus?.code_rag_healthy || false)}
                    </ListItemIcon>
                    <ListItemText 
                      primary="Code RAG" 
                      secondary={systemStatus?.code_rag_healthy ? 'Connected' : (systemStatus?.code_rag_status?.response || 'Unavailable')}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      {systemStatus?.hybrid_available ? (
                        <CheckIcon sx={{ color: 'success.main' }} />
                      ) : (
                        <WarningIcon sx={{ color: 'warning.main' }} />
                      )}
                    </ListItemIcon>
                    <ListItemText 
                      primary="Hybrid Mode" 
                      secondary={systemStatus?.hybrid_available ? 'Full' : 'Partial (GraphRAG only)'}
                    />
                  </ListItem>
                </List>

                <Divider sx={{ my: 2 }} />

                {/* Extraction Methods */}
                {extractionStats && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Extraction Methods
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {extractionStats.spanbert_available && (
                        <Chip label="SpanBERT" size="small" color="primary" variant="outlined" />
                      )}
                      {extractionStats.dependency_available && (
                        <Chip label="Dependency Parsing" size="small" color="primary" variant="outlined" />
                      )}
                      {extractionStats.entity_linking_available && (
                        <Chip label="Entity Linking" size="small" color="primary" variant="outlined" />
                      )}
                    </Box>
                  </Box>
                )}

                <Divider sx={{ my: 2 }} />

                {/* Supported Formats */}
                {supportedFormats && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Supported Formats
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {supportedFormats.supported_formats.map(fmt => (
                        <Chip key={fmt} label={fmt} size="small" variant="outlined" />
                      ))}
                    </Box>
                    
                    <Typography variant="subtitle2" sx={{ mt: 2 }} gutterBottom>
                      Features
                    </Typography>
                    <List dense>
                      {Object.entries(supportedFormats.features).map(([key, value]) => (
                        <ListItem key={key} sx={{ py: 0 }}>
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
                  </Box>
                )}
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Upload Area */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              <UploadIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              Upload Files
            </Typography>

            {/* Domain Selection */}
            <FormControl size="small" sx={{ mb: 2, minWidth: 200 }}>
              <InputLabel>Domain</InputLabel>
              <Select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                label="Domain"
              >
                <MenuItem value="general">General</MenuItem>
                <MenuItem value="technical">Technical</MenuItem>
                <MenuItem value="legal">Legal</MenuItem>
                <MenuItem value="medical">Medical</MenuItem>
                <MenuItem value="financial">Financial</MenuItem>
              </Select>
            </FormControl>

            {/* Drop Zone */}
            <Paper
              sx={{
                p: 4,
                mb: 2,
                border: '2px dashed',
                borderColor: dragOver ? 'primary.main' : 'divider',
                borderRadius: 2,
                textAlign: 'center',
                cursor: 'pointer',
                bgcolor: dragOver ? alpha(theme.palette.primary.main, 0.05) : 'transparent',
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
              <label htmlFor="hybrid-file-input" style={{ cursor: 'pointer', display: 'block' }}>
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mb: 2 }}>
                  <CodeIcon sx={{ fontSize: 40, color: '#f97316' }} />
                  <DocumentIcon sx={{ fontSize: 40, color: '#3b82f6' }} />
                </Box>
                <Typography variant="h6" gutterBottom>
                  Drop files here or click to browse
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Code files (.py, .js, .ts, etc.) will be indexed in Code RAG
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Documents (.pdf, .txt, .docx, etc.) will be processed by GraphRAG
                </Typography>
              </label>
            </Paper>

            {/* File Queue */}
            {files.length > 0 && (
              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2">
                    Processing Queue ({files.length} files)
                  </Typography>
                  <Box>
                    <Button size="small" onClick={clearCompleted} disabled={processing}>
                      Clear Completed
                    </Button>
                    <Button 
                      variant="contained" 
                      size="small" 
                      onClick={processFiles}
                      disabled={processing || files.every(f => f.status !== 'pending')}
                      startIcon={processing ? <CircularProgress size={16} /> : <ProcessingIcon />}
                      sx={{ ml: 1 }}
                    >
                      {processing ? 'Processing...' : 'Process All'}
                    </Button>
                  </Box>
                </Box>

                <List>
                  {files.map((fileItem, index) => (
                    <ListItem
                      key={index}
                      sx={{
                        bgcolor: alpha(
                          fileItem.status === 'completed' ? theme.palette.success.main :
                          fileItem.status === 'error' ? theme.palette.error.main :
                          theme.palette.background.paper, 
                          0.05
                        ),
                        borderRadius: 1,
                        mb: 1,
                      }}
                      secondaryAction={
                        <IconButton 
                          edge="end" 
                          size="small"
                          onClick={() => removeFile(index)}
                          disabled={fileItem.status === 'processing'}
                        >
                          <DeleteIcon />
                        </IconButton>
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
                            <CodeIcon sx={{ color: '#f97316' }} />
                          ) : (
                            <DocumentIcon sx={{ color: '#3b82f6' }} />
                          )
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {fileItem.file.name}
                            <Chip 
                              label={fileItem.fileType || (isCodeFile(fileItem.file.name) ? 'Code' : 'Document')}
                              size="small"
                              color={fileItem.fileType === 'code' ? 'warning' : 'info'}
                              variant="outlined"
                            />
                            {fileItem.language && (
                              <Chip label={fileItem.language} size="small" variant="outlined" />
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
          </Paper>
        </Grid>

        {/* Processing Results */}
        {files.some(f => f.status === 'completed') && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                <AnalyticsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Processing Results
              </Typography>

              {files.filter(f => f.status === 'completed').map((fileItem, index) => (
                <Accordion key={index} defaultExpanded={index === 0}>
                  <AccordionSummary expandIcon={<ExpandIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getFileIcon(fileItem)}
                      <Typography>{fileItem.file.name}</Typography>
                      <Chip 
                        label={fileItem.fileType === 'code' ? 'Code RAG + GraphRAG' : 'GraphRAG Only'}
                        size="small"
                        color={fileItem.result?.hybrid_processing ? 'success' : 'info'}
                      />
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      {/* GraphRAG Results */}
                      <Grid item xs={12} md={fileItem.result?.code_rag_processing ? 6 : 12}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                              <GraphIcon sx={{ mr: 0.5, fontSize: 18, verticalAlign: 'middle' }} />
                              GraphRAG Processing
                            </Typography>
                            {fileItem.result?.graphrag_processing?.success ? (
                              <Box>
                                <Alert severity="success" sx={{ mb: 1 }}>
                                  Successfully indexed in knowledge graph
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
                          </CardContent>
                        </Card>
                      </Grid>

                      {/* Code RAG Results */}
                      {fileItem.result?.code_rag_processing && (
                        <Grid item xs={12} md={6}>
                          <Card variant="outlined">
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                <CodeIcon sx={{ mr: 0.5, fontSize: 18, verticalAlign: 'middle' }} />
                                Code RAG Processing
                              </Typography>
                              {fileItem.result?.code_rag_processing?.success ? (
                                <Box>
                                  <Alert severity="success" sx={{ mb: 1 }}>
                                    Successfully indexed for code search
                                  </Alert>
                                  {fileItem.language && (
                                    <Chip 
                                      label={`Language: ${fileItem.language}`}
                                      size="small"
                                      sx={{ mt: 1 }}
                                    />
                                  )}
                                </Box>
                              ) : (
                                <Alert severity="warning">
                                  {fileItem.result?.code_rag_processing?.error || 'Code RAG not available'}
                                </Alert>
                              )}
                            </CardContent>
                          </Card>
                        </Grid>
                      )}
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              ))}
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};
