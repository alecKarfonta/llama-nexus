import React, { useState, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  LinearProgress,
  Alert,
  FormControl,
  FormControlLabel,
  Checkbox,
  Select,
  MenuItem,
  InputLabel,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Paper,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  CircularProgress,
  alpha,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Description as DocumentIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Analytics as AnalyticsIcon,
  Memory as ProcessorIcon,
  Storage as StorageIcon,
  Language as LanguageIcon,
  Code as CodeIcon,
} from '@mui/icons-material';
import * as api from '../../services/api';

interface BatchIngestionDialogProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  destination?: 'local' | 'graphrag';
}

interface FileItem {
  file: File;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  message?: string;
  result?: any;
}

interface ProcessingOptions {
  useSemanticChunking: boolean;
  chunkSize: number;
  chunkOverlap: number;
  extractEntities: boolean;
  extractRelationships: boolean;
  buildKnowledgeGraph: boolean;
  detectCode: boolean;
  enableHybridProcessing: boolean;
  processingMethod: 'standard' | 'enhanced' | 'fast';
  entityLabels: string[];
  customDomain?: string;
}

const defaultEntityLabels = [
  'person', 'organization', 'location', 'date', 'product', 
  'technology', 'concept', 'event', 'metric', 'requirement'
];

export const BatchIngestionDialog: React.FC<BatchIngestionDialogProps> = ({
  open,
  onClose,
  onSuccess,
  destination = 'graphrag'
}) => {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [processing, setProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentFile, setCurrentFile] = useState<string>('');
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const [options, setOptions] = useState<ProcessingOptions>({
    useSemanticChunking: true,
    chunkSize: 1000,
    chunkOverlap: 200,
    extractEntities: true,
    extractRelationships: true,
    buildKnowledgeGraph: true,
    detectCode: false,
    enableHybridProcessing: false,
    processingMethod: 'standard',
    entityLabels: defaultEntityLabels,
  });

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    const newFiles: FileItem[] = selectedFiles.map(file => ({
      file,
      status: 'pending',
      progress: 0
    }));
    setFiles(prev => [...prev, ...newFiles]);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    const newFiles: FileItem[] = droppedFiles.map(file => ({
      file,
      status: 'pending',
      progress: 0
    }));
    setFiles(prev => [...prev, ...newFiles]);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const processFiles = async () => {
    if (files.length === 0) return;

    setProcessing(true);
    setError(null);
    setResults([]);

    try {
      if (destination === 'graphrag') {
        // Process with GraphRAG batch endpoint
        const formData = new FormData();
        files.forEach(item => {
          formData.append('files', item.file);
        });
        
        // Add processing options
        if (options.useSemanticChunking) {
          formData.append('use_semantic_chunking', 'true');
        }
        
        const response = await fetch('/api/v1/graphrag/ingest/batch', {
          method: 'POST',
          body: formData,
          // Note: Don't set Content-Type header, let browser set it with boundary
        });

        if (!response.ok) {
          throw new Error(`Upload failed: ${response.statusText}`);
        }

        const result = await response.json();
        
        // Update file statuses based on results
        setFiles(prev => prev.map((file, index) => ({
          ...file,
          status: result.results?.[index]?.success ? 'completed' : 'error',
          progress: 100,
          message: result.results?.[index]?.message,
          result: result.results?.[index]
        })));

        setResults(result.results || []);
        
        if (result.knowledge_graph_stats) {
          console.log('Knowledge graph updated:', result.knowledge_graph_stats);
        }
        
        if (onSuccess) {
          onSuccess();
        }
      } else {
        // Process locally (existing RAG system)
        for (let i = 0; i < files.length; i++) {
          const fileItem = files[i];
          setCurrentFile(fileItem.file.name);
          
          // Update status to processing
          setFiles(prev => prev.map((f, idx) => 
            idx === i ? { ...f, status: 'processing', progress: 50 } : f
          ));

          try {
            const formData = new FormData();
            formData.append('file', fileItem.file);
            formData.append('chunk_size', options.chunkSize.toString());
            formData.append('chunk_overlap', options.chunkOverlap.toString());
            
            const response = await fetch('/api/v1/rag/documents/upload', {
              method: 'POST',
              body: formData
            });

            if (!response.ok) {
              throw new Error(`Failed to upload ${fileItem.file.name}`);
            }

            const result = await response.json();
            
            // Update status to completed
            setFiles(prev => prev.map((f, idx) => 
              idx === i ? { ...f, status: 'completed', progress: 100, result } : f
            ));
            
            setResults(prev => [...prev, result]);
          } catch (err) {
            // Update status to error
            setFiles(prev => prev.map((f, idx) => 
              idx === i ? { 
                ...f, 
                status: 'error', 
                progress: 100, 
                message: err instanceof Error ? err.message : 'Upload failed' 
              } : f
            ));
          }
          
          setUploadProgress(((i + 1) / files.length) * 100);
        }
        
        if (onSuccess) {
          onSuccess();
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Processing failed');
    } finally {
      setProcessing(false);
      setCurrentFile('');
      setUploadProgress(0);
    }
  };

  const getFileStatusIcon = (status: FileItem['status']) => {
    switch (status) {
      case 'completed':
        return <CheckIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'processing':
        return <CircularProgress size={20} />;
      default:
        return <DocumentIcon />;
    }
  };

  const getFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <UploadIcon />
          Batch Document Ingestion
          <Chip 
            label={destination === 'graphrag' ? 'GraphRAG' : 'Local RAG'} 
            size="small" 
            color={destination === 'graphrag' ? 'primary' : 'default'}
            sx={{ ml: 'auto' }}
          />
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* File Upload Area */}
        <Paper
          sx={{
            p: 3,
            mb: 2,
            border: '2px dashed',
            borderColor: 'divider',
            borderRadius: 2,
            textAlign: 'center',
            cursor: 'pointer',
            bgcolor: alpha('#1976d2', 0.02),
            '&:hover': {
              bgcolor: alpha('#1976d2', 0.05),
              borderColor: 'primary.main'
            }
          }}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <input
            type="file"
            multiple
            accept=".txt,.md,.pdf,.json,.csv,.html,.xml,.doc,.docx"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            id="batch-file-input"
          />
          <label htmlFor="batch-file-input" style={{ cursor: 'pointer' }}>
            <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
            <Typography variant="h6" gutterBottom>
              Drop files here or click to browse
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Supports: TXT, MD, PDF, JSON, CSV, HTML, XML, DOC, DOCX
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Max 100 files per batch
            </Typography>
          </label>
        </Paper>

        {/* File List */}
        {files.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Selected Files ({files.length})
            </Typography>
            <List dense sx={{ maxHeight: 200, overflow: 'auto' }}>
              {files.map((item, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    {getFileStatusIcon(item.status)}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.file.name}
                    secondary={
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          {getFileSize(item.file.size)}
                        </Typography>
                        {item.message && (
                          <Typography 
                            variant="caption" 
                            color={item.status === 'error' ? 'error' : 'success.main'}
                            sx={{ ml: 1 }}
                          >
                            {item.message}
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                  {item.status === 'processing' && (
                    <LinearProgress 
                      variant="determinate" 
                      value={item.progress} 
                      sx={{ width: 100, mr: 2 }}
                    />
                  )}
                  <ListItemSecondaryAction>
                    <IconButton 
                      edge="end" 
                      size="small"
                      onClick={() => removeFile(index)}
                      disabled={processing}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </Box>
        )}

        {/* Processing Options */}
        {destination === 'graphrag' && (
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <SettingsIcon sx={{ mr: 1 }} />
              <Typography>Advanced Processing Options</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={options.useSemanticChunking}
                        onChange={(e) => setOptions({...options, useSemanticChunking: e.target.checked})}
                      />
                    }
                    label="Use Semantic Chunking"
                  />
                  <Tooltip title="Intelligently splits text at natural boundaries">
                    <InfoIcon sx={{ fontSize: 16, ml: 1, color: 'text.secondary' }} />
                  </Tooltip>
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={options.extractEntities}
                        onChange={(e) => setOptions({...options, extractEntities: e.target.checked})}
                      />
                    }
                    label="Extract Entities (GLiNER)"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={options.extractRelationships}
                        onChange={(e) => setOptions({...options, extractRelationships: e.target.checked})}
                      />
                    }
                    label="Extract Relationships"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={options.buildKnowledgeGraph}
                        onChange={(e) => setOptions({...options, buildKnowledgeGraph: e.target.checked})}
                      />
                    }
                    label="Build Knowledge Graph"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={options.detectCode}
                        onChange={(e) => setOptions({...options, detectCode: e.target.checked})}
                      />
                    }
                    label="Detect Code Blocks"
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={options.enableHybridProcessing}
                        onChange={(e) => setOptions({...options, enableHybridProcessing: e.target.checked})}
                      />
                    }
                    label="Enable Hybrid Processing"
                  />
                  <Tooltip title="Combines vector and graph-based processing">
                    <InfoIcon sx={{ fontSize: 16, ml: 1, color: 'text.secondary' }} />
                  </Tooltip>
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Processing Method</InputLabel>
                    <Select
                      value={options.processingMethod}
                      onChange={(e) => setOptions({...options, processingMethod: e.target.value as any})}
                      label="Processing Method"
                    >
                      <MenuItem value="fast">Fast (Basic extraction)</MenuItem>
                      <MenuItem value="standard">Standard (Balanced)</MenuItem>
                      <MenuItem value="enhanced">Enhanced (Thorough analysis)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                {options.extractEntities && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" gutterBottom>
                      Entity Types to Extract
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {defaultEntityLabels.map(label => (
                        <Chip
                          key={label}
                          label={label}
                          size="small"
                          color={options.entityLabels.includes(label) ? 'primary' : 'default'}
                          onClick={() => {
                            setOptions({
                              ...options,
                              entityLabels: options.entityLabels.includes(label)
                                ? options.entityLabels.filter(l => l !== label)
                                : [...options.entityLabels, label]
                            });
                          }}
                          sx={{ cursor: 'pointer' }}
                        />
                      ))}
                    </Box>
                  </Grid>
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Processing Progress */}
        {processing && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Processing: {currentFile || 'Initializing...'}
            </Typography>
            <LinearProgress variant="determinate" value={uploadProgress} sx={{ mb: 1 }} />
            <Typography variant="caption" color="text.secondary">
              {Math.round(uploadProgress)}% complete
            </Typography>
          </Box>
        )}

        {/* Results Summary */}
        {results.length > 0 && !processing && (
          <Alert severity="success" sx={{ mt: 2 }}>
            <Typography variant="subtitle2">Processing Complete</Typography>
            <Typography variant="body2">
              Successfully processed {results.filter(r => r.success).length} of {results.length} files
            </Typography>
            {destination === 'graphrag' && results[0]?.knowledge_graph_stats && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Knowledge Graph Stats:
                </Typography>
                <Typography variant="caption" display="block">
                  • Entities: {results[0].knowledge_graph_stats.total_entities}
                </Typography>
                <Typography variant="caption" display="block">
                  • Relationships: {results[0].knowledge_graph_stats.total_relationships}
                </Typography>
              </Box>
            )}
          </Alert>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} disabled={processing}>
          Cancel
        </Button>
        <Button
          variant="contained"
          startIcon={processing ? <CircularProgress size={20} /> : <UploadIcon />}
          onClick={processFiles}
          disabled={files.length === 0 || processing}
        >
          {processing ? 'Processing...' : `Process ${files.length} File${files.length !== 1 ? 's' : ''}`}
        </Button>
      </DialogActions>
    </Dialog>
  );
};
