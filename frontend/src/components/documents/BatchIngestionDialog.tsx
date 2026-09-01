import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
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
  Info as InfoIcon,
  QuestionAnswer as ChatIcon,
} from '@mui/icons-material';
import { apiService as api } from '@/services/api';
import { GraphRAGPipelineTrace } from '@/components/graphrag/GraphRAGPipelineTrace';
import { getRoutePath } from '@/config/navigation';
import { buildGraphRAGBatchPrompt, buildGraphRAGBatchQuestions } from '@/utils/graphragBatchPrompts';

interface BatchIngestionDialogProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  destination?: 'local' | 'graphrag';
}

interface FileItem {
  file: File;
  status: 'pending' | 'processing' | 'completed' | 'warning' | 'error';
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

const graphRAGFileStatus = (result: Record<string, unknown>): FileItem['status'] => {
  if (result.status === 'upstream_failed') return 'error';
  if (result.status === 'preview_only') return 'warning';
  return 'completed';
};

const graphRAGFileMessage = (result: Record<string, unknown>): string => {
  if (result.status === 'submitted') return 'Submitted to GraphRAG';
  if (result.status === 'preview_only') return 'Pipeline preview ready';
  if (result.status === 'upstream_failed') return 'Preview ready, upstream failed';
  return 'Pipeline run complete';
};

const resultSucceeded = (result: any): boolean =>
  Boolean(result) && result.success !== false && result.status !== 'upstream_failed' && !result.error;

const asRecord = (value: unknown): Record<string, unknown> => (
  value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
);

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asNumber = (value: unknown, fallback = 0): number => {
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : fallback;
};

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim() ? value : fallback;

const asStringArray = (value: unknown): string[] =>
  asArray(value).map(item => asString(item)).filter(Boolean);

const formatNumber = (value: unknown) => asNumber(value).toLocaleString();

const formatDuration = (durationMs: number): string => {
  if (durationMs < 1000) return `${durationMs}ms`;
  if (durationMs < 60_000) return `${(durationMs / 1000).toFixed(1)}s`;
  const minutes = Math.floor(durationMs / 60_000);
  const seconds = Math.round((durationMs % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
};

const formatThroughput = (value: unknown): string => {
  const parsed = asNumber(value);
  if (parsed <= 0) return '0 files/s';
  return `${parsed >= 10 ? parsed.toFixed(1) : parsed.toFixed(2)} files/s`;
};

export const BatchIngestionDialog: React.FC<BatchIngestionDialogProps> = ({
  open,
  onClose,
  onSuccess,
  destination = 'graphrag'
}) => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<FileItem[]>([]);
  const [processing, setProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentFile, setCurrentFile] = useState<string>('');
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [batchResult, setBatchResult] = useState<Record<string, unknown> | null>(null);

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
    setBatchResult(null);

    try {
      if (destination === 'graphrag') {
        setCurrentFile(`${files.length} GraphRAG file${files.length === 1 ? '' : 's'}`);
        setUploadProgress(35);
        setFiles(prev => prev.map(file => ({
          ...file,
          status: 'processing',
          progress: 45,
          message: 'Tracing pipeline...',
        })));

        const batchResult = await api.uploadGraphRAGDocumentsWithPipeline(files.map(item => item.file), {
          domain: options.customDomain || 'general',
          useSemanticChunking: options.useSemanticChunking,
          buildKnowledgeGraph: options.buildKnowledgeGraph,
          sampleLimit: 6,
          maxConcurrency: Math.min(4, Math.max(1, files.length)),
        });
        setBatchResult(batchResult);
        const batchRuns = Array.isArray(batchResult.runs) ? batchResult.runs : [];
        const batchItems = Array.isArray(batchResult.results) ? batchResult.results : [];
        const runById = new Map(batchRuns.map((run: any) => [run?.run_id, run]));
        const completedRuns = files.map((fileItem, index) => {
          const itemResult = batchItems.find((item: any) => item?.index === index);
          const run = runById.get(itemResult?.run_id);
          if (run) {
            const status = graphRAGFileStatus(run as Record<string, unknown>);
            return {
              ...(run as Record<string, unknown>),
              success: status !== 'error' && itemResult?.success !== false,
            };
          }
          return {
            success: false,
            filename: fileItem.file.name,
            status: 'error',
            error: itemResult?.error || 'Upload failed',
          };
        });

        setResults(completedRuns);
        setFiles(prev => prev.map((f, idx) => {
          const result = completedRuns[idx];
          const status = graphRAGFileStatus(result as Record<string, unknown>);
          return {
            ...f,
            status: result.success ? status : 'error',
            progress: 100,
            message: result.success ? graphRAGFileMessage(result as Record<string, unknown>) : (result.error || 'Upload failed'),
            result,
          };
        }));
        setUploadProgress(100);

        if (completedRuns.some(resultSucceeded) && onSuccess) {
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
            formData.append('build_knowledge_graph', options.buildKnowledgeGraph.toString());

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
      case 'warning':
        return <InfoIcon color="warning" />;
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

  const batchRecord = asRecord(batchResult);
  const batchSummary = asRecord(batchRecord.summary);
  const batchUploadManifest = asRecord(batchRecord.upload_manifest || batchSummary.upload_manifest);
  const batchOperationalHandoff = asRecord(batchSummary.operational_handoff);
  const batchResumeHandoff = asRecord(
    batchRecord.batch_resume_handoff
    || batchSummary.batch_resume_handoff
    || batchUploadManifest.batch_resume_handoff
  );
  const batchResumeRouteState = asRecord(batchResumeHandoff.route_state);
  const batchResumeRouteHandoff = asRecord(batchResumeRouteState.graphragHandoff);
  const batchOperationalRouteHandoff = asRecord(asRecord(batchOperationalHandoff.route_state).graphragHandoff);
  const batchManifestRouteHandoff = asRecord(asRecord(batchUploadManifest.route_state).graphragHandoff);
  const batchPreparedRouteHandoff = Object.keys(batchResumeRouteHandoff).length > 0
    ? batchResumeRouteHandoff
    : Object.keys(batchOperationalRouteHandoff).length > 0
      ? batchOperationalRouteHandoff
      : batchManifestRouteHandoff;
  const batchResumeChatContextRequest = asRecord(
    batchResumeHandoff.chat_context_request
    || batchUploadManifest.chat_context_request
    || batchOperationalHandoff.chat_context_request
    || batchPreparedRouteHandoff.chatContextRequest
  );
  const batchResumeSuggestedQuestions = (
    asArray(batchPreparedRouteHandoff.suggestedQuestions).length > 0
      ? asArray(batchPreparedRouteHandoff.suggestedQuestions)
      : asArray(batchResumeHandoff.suggested_questions)
  ).filter((question): question is Record<string, unknown> => (
    Boolean(question) && typeof question === 'object' && !Array.isArray(question)
  ));
  const batchExecution = asRecord(batchSummary.execution);
  const batchDurationMs = asNumber(batchRecord.duration_ms ?? batchExecution.duration_ms);
  const batchThroughput = batchRecord.throughput_files_per_second ?? batchExecution.throughput_files_per_second;
  const batchMaxConcurrency = asNumber(batchRecord.max_concurrency ?? batchRecord.concurrency ?? batchExecution.max_concurrency);
  const batchPeakConcurrency = asNumber(batchRecord.peak_concurrency ?? batchExecution.peak_concurrency);
  const batchRemoteSubmitted = asNumber(batchSummary.remote_submitted);
  const batchLocalPreviewOnly = asNumber(batchSummary.local_preview_only);
  const batchGraphOverview = asRecord(batchSummary.graph_overview);
  const batchGraphRelationshipCount = asNumber(batchGraphOverview.relationship_count);
  const batchGraphSharedEntityCount = asNumber(batchGraphOverview.shared_entity_count);
  const batchGraphBridgeEntityCount = asNumber(batchGraphOverview.bridge_entity_count);
  const batchGraphPathCount = asNumber(batchGraphOverview.cross_document_path_count);
  const hasBatchExecution = destination === 'graphrag' && Boolean(batchResult);
  const graphRAGCompletedRuns = destination === 'graphrag'
    ? results
        .map(result => asRecord(result))
        .filter(result => Boolean(asString(result.run_id)))
    : [];
  const graphRAGBatchRunIds = graphRAGCompletedRuns.map(result => asString(result.run_id)).filter(Boolean);
  const graphRAGBatchFilenames = graphRAGCompletedRuns
    .map(result => asString(asRecord(result.document).filename, asString(result.filename)))
    .filter(Boolean);
  const graphRAGBatchQuestions = buildGraphRAGBatchQuestions({
    filenames: graphRAGBatchFilenames,
    overview: batchGraphOverview,
  }, 4);
  const handleAskChatWithGraphRAGBatch = (promptOverride?: string) => {
    const preparedRunIds = asStringArray(batchResumeChatContextRequest.pipeline_run_ids);
    const routeRunIds = asStringArray(batchPreparedRouteHandoff.runIds);
    const batchRunIds = preparedRunIds.length > 0
      ? preparedRunIds
      : routeRunIds.length > 0
        ? routeRunIds
        : graphRAGBatchRunIds;
    if (batchRunIds.length === 0) return;

    const fileText = graphRAGBatchFilenames.length > 0
      ? graphRAGBatchFilenames.slice(0, 6).join(', ')
      : `${batchRunIds.length} uploaded files`;
    const isBatch = batchRunIds.length > 1;
    const fallbackPrompt = isBatch
      ? buildGraphRAGBatchPrompt({
        filenames: graphRAGBatchFilenames.length > 0 ? graphRAGBatchFilenames : [fileText],
        overview: batchGraphOverview,
      })
      : (
        `Use the GraphRAG pipeline trace for ${fileText} to answer with graph relationships, paths, and provenance. `
        + 'Cite filename plus slide/page/table/chart/connector provenance whenever available.'
      );
    const prompt = promptOverride || asString(batchPreparedRouteHandoff.prompt) || asString(batchResumeChatContextRequest.query) || fallbackPrompt;
    const domain = asString(
      batchResumeChatContextRequest.domain,
      asString(batchPreparedRouteHandoff.domain, options.customDomain?.trim() || 'general'),
    );

    navigate(getRoutePath('chat'), {
      state: {
        graphragHandoff: {
          ...batchPreparedRouteHandoff,
          runIds: batchRunIds,
          filename: asString(
            batchPreparedRouteHandoff.filename,
            isBatch ? `${batchRunIds.length} file GraphRAG batch` : (graphRAGBatchFilenames[0] || batchRunIds[0]),
          ),
          domain,
          prompt,
          chatContextEndpoint: asString(
            batchResumeHandoff.chat_context_endpoint,
            asString(batchPreparedRouteHandoff.chatContextEndpoint, '/api/v1/graphrag/chat-context-v2'),
          ),
          chatContextRequest: {
            ...batchResumeChatContextRequest,
            query: prompt,
            domain,
            pipeline_run_ids: batchRunIds,
            include_graph: batchResumeChatContextRequest.include_graph === false ? false : true,
            use_local_pipeline_context: batchResumeChatContextRequest.use_local_pipeline_context === false ? false : true,
            include_local_pipeline: batchResumeChatContextRequest.include_local_pipeline === false ? false : true,
          },
          suggestedQuestions: batchResumeSuggestedQuestions.length > 0
            ? batchResumeSuggestedQuestions
            : graphRAGBatchQuestions,
        },
      },
    });
    onClose();
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
            accept=".txt,.md,.pdf,.json,.csv,.html,.xml,.eml,.doc,.docx,.pptx,.xlsx,.rtf,.odt,.ods,.odp,.epub,message/rfc822"
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
              Supports: TXT, MD, PDF, JSON, CSV, HTML, XML, DOC, DOCX, PPTX, XLSX, ODT, ODS, ODP, EPUB
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
                            color={item.status === 'error' ? 'error' : item.status === 'warning' ? 'warning.main' : 'success.main'}
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
        {(
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
                        onChange={(e) => setOptions({ ...options, useSemanticChunking: e.target.checked })}
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
                        onChange={(e) => setOptions({ ...options, extractEntities: e.target.checked })}
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
                        onChange={(e) => setOptions({ ...options, extractRelationships: e.target.checked })}
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
                        onChange={(e) => setOptions({ ...options, buildKnowledgeGraph: e.target.checked })}
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
                        onChange={(e) => setOptions({ ...options, detectCode: e.target.checked })}
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
                        onChange={(e) => setOptions({ ...options, enableHybridProcessing: e.target.checked })}
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
                      onChange={(e) => setOptions({ ...options, processingMethod: e.target.value as any })}
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
              Successfully processed {results.filter(resultSucceeded).length} of {results.length} files
            </Typography>
            {hasBatchExecution && (
              <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                {batchDurationMs > 0 && (
                  <Chip label={`Elapsed ${formatDuration(batchDurationMs)}`} size="small" variant="outlined" sx={{ height: 22 }} />
                )}
                {asNumber(batchThroughput) > 0 && (
                  <Chip label={formatThroughput(batchThroughput)} size="small" variant="outlined" sx={{ height: 22 }} />
                )}
                {batchMaxConcurrency > 0 && (
                  <Chip label={`${formatNumber(batchMaxConcurrency)} workers`} size="small" variant="outlined" sx={{ height: 22 }} />
                )}
                {batchPeakConcurrency > 0 && batchPeakConcurrency !== batchMaxConcurrency && (
                  <Chip label={`Peak ${formatNumber(batchPeakConcurrency)}`} size="small" variant="outlined" sx={{ height: 22 }} />
                )}
                {batchRemoteSubmitted > 0 && (
                  <Chip label={`${formatNumber(batchRemoteSubmitted)} remote`} size="small" variant="outlined" sx={{ height: 22 }} />
                )}
                {batchLocalPreviewOnly > 0 && (
                  <Chip label={`${formatNumber(batchLocalPreviewOnly)} local traces`} size="small" variant="outlined" sx={{ height: 22 }} />
                )}
                {batchGraphRelationshipCount > 0 && (
                  <Chip label={`${formatNumber(batchGraphRelationshipCount)} graph relationships`} size="small" variant="outlined" sx={{ height: 22 }} />
                )}
                {batchGraphSharedEntityCount > 0 && (
                  <Chip label={`${formatNumber(batchGraphSharedEntityCount)} shared entities`} size="small" color="info" variant="outlined" sx={{ height: 22 }} />
                )}
                {batchGraphBridgeEntityCount > 0 && (
                  <Chip label={`${formatNumber(batchGraphBridgeEntityCount)} bridge entities`} size="small" color="info" variant="outlined" sx={{ height: 22 }} />
                )}
                {batchGraphPathCount > 0 && (
                  <Chip label={`${formatNumber(batchGraphPathCount)} cross-doc paths`} size="small" color="info" variant="outlined" sx={{ height: 22 }} />
                )}
                {graphRAGBatchRunIds.length > 0 && (
                  <Button
                    size="small"
                    variant="contained"
                    startIcon={<ChatIcon fontSize="small" />}
                    onClick={() => handleAskChatWithGraphRAGBatch()}
                    sx={{
                      minHeight: 24,
                      px: 1,
                      py: 0.15,
                      borderRadius: '8px',
                      textTransform: 'none',
                      boxShadow: 'none',
                    }}
                  >
                    {graphRAGBatchRunIds.length > 1 ? 'Ask Chat Across Batch' : 'Ask Chat'}
                  </Button>
                )}
              </Box>
            )}
            {hasBatchExecution && graphRAGBatchQuestions.length > 0 && graphRAGBatchRunIds.length > 0 && (
              <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                {graphRAGBatchQuestions.slice(0, 4).map(question => (
                  <Tooltip key={question.id} title={question.prompt}>
                    <Button
                      size="small"
                      variant={question.kind === 'summary' ? 'contained' : 'outlined'}
                      color={question.kind === 'summary' ? 'primary' : 'inherit'}
                      startIcon={<ChatIcon fontSize="small" />}
                      onClick={() => handleAskChatWithGraphRAGBatch(question.prompt)}
                      sx={{
                        minHeight: 24,
                        maxWidth: { xs: '100%', sm: 240 },
                        px: 1,
                        py: 0.15,
                        borderRadius: '8px',
                        textTransform: 'none',
                        boxShadow: 'none',
                        justifyContent: 'flex-start',
                        '& .MuiButton-startIcon': { flexShrink: 0 },
                      }}
                    >
                      <Box component="span" sx={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {question.label}
                      </Box>
                    </Button>
                  </Tooltip>
                ))}
              </Box>
            )}
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

        {destination === 'graphrag' && results.length > 0 && !processing && (
          <Box sx={{ mt: 2 }}>
            {results.map((result, index) => (
              <GraphRAGPipelineTrace
                key={result?.run_id || result?.filename || index}
                result={result}
                dense
                title={`Pipeline run ${index + 1}`}
              />
            ))}
          </Box>
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
