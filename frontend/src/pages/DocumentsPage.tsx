import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Menu,
  Chip,
  List,
  ListItemText,
  ListItemIcon,
  Tooltip,
  Alert,
  CircularProgress,
  Divider,
  Tabs,
  Tab,
  TablePagination,
  Slider,
  useMediaQuery,
  alpha,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Search as SearchIcon,
  Refresh as RefreshIcon,
  Folder as FolderIcon,
  FolderOpen as FolderOpenIcon,
  Description as DocIcon,
  Upload as UploadIcon,
  CloudUpload as CloudUploadIcon,
  Link as LinkIcon,
  TextFields as TextIcon,
  ExpandMore as ExpandIcon,
  ChevronRight as ChevronRightIcon,
  Visibility as ViewIcon,
  PlayArrow as ProcessIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  Storage as StorageIcon,
  Code as CodeIcon,
  PictureAsPdf as PdfIcon,
  Article as ArticleIcon,
  DataObject as JsonIcon,
  TableChart as CsvIcon,
  Sync as SyncIcon,
  SelectAll as SelectAllIcon,
  Deselect as DeselectIcon,
  CheckBox as CheckBoxIcon,
  CheckBoxOutlineBlank as CheckBoxOutlineBlankIcon,
  AccountTree as KnowledgeGraphIcon,
  MoreHoriz as MoreIcon,
} from '@mui/icons-material';
import { apiService as api } from '../services/api';
import { BatchIngestionDialog } from '../components/documents/BatchIngestionDialog';
import { ConfirmationDialog } from '../components/common';
import VectorSearchPanel from '../components/documents/VectorSearchPanel';
import { useConfirmableAction } from '@/hooks/useConfirmableAction';
import { GraphRAGPipelineTrace } from '@/components/graphrag/GraphRAGPipelineTrace';

// Types
interface Domain {
  id: string;
  name: string;
  description: string;
  parent_id: string | null;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  document_count: number;
  total_chunks: number;
  created_at: string;
  children?: Domain[];
}

interface Document {
  id: string;
  domain_id: string;
  name: string;
  doc_type: string;
  status: string;
  content: string;
  source_url: string | null;
  chunk_count: number;
  token_count: number;
  metadata: Record<string, unknown>;
  created_at: string;
  processed_at: string | null;
}

interface Chunk {
  id: string;
  document_id: string;
  content: string;
  chunk_index: number;
  total_chunks: number;
  section_header: string | null;
  metadata?: Record<string, unknown>;
}

// Document type icons
const DOC_TYPE_ICONS: Record<string, React.ReactNode> = {
  pdf: <PdfIcon />,
  docx: <ArticleIcon />,
  pptx: <ArticleIcon />,
  xlsx: <CsvIcon />,
  rtf: <ArticleIcon />,
  odt: <ArticleIcon />,
  ods: <CsvIcon />,
  odp: <ArticleIcon />,
  txt: <TextIcon />,
  md: <CodeIcon />,
  html: <CodeIcon />,
  json: <JsonIcon />,
  csv: <CsvIcon />,
  url: <LinkIcon />,
  epub: <ArticleIcon />,
};

// Status colors and icons
const STATUS_CONFIG: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
  pending: { color: '#F59E0B', icon: <PendingIcon />, label: 'Pending' },
  processing: { color: '#3B82F6', icon: <CircularProgress size={16} />, label: 'Processing' },
  ready: { color: '#10B981', icon: <CheckIcon />, label: 'Ready' },
  error: { color: '#EF4444', icon: <ErrorIcon />, label: 'Error' },
  archived: { color: '#6B7280', icon: <StorageIcon />, label: 'Archived' },
};

const flattenDomains = (items: Domain[]): Domain[] =>
  items.flatMap(domain => [domain, ...flattenDomains(domain.children || [])]);

const documentToolbarButtonSx = {
  height: 34,
  borderRadius: '8px',
  fontWeight: 800,
  px: 1.25,
  textTransform: 'none',
  minWidth: 0,
};

const compactDocumentToolbarButtonSx = {
  ...documentToolbarButtonSx,
  width: { xs: 34, sm: 'auto' },
  minWidth: { xs: 34, sm: 0 },
  px: { xs: 0, sm: 1.25 },
  '& .MuiButton-startIcon': { mr: { xs: 0, sm: 0.75 }, ml: 0 },
};

const documentToolbarLabelSx = {
  display: { xs: 'none', sm: 'inline' },
};

const documentIconButtonSx = {
  width: 34,
  height: 34,
  borderRadius: '8px',
  border: '1px solid',
  borderColor: 'divider',
  bgcolor: 'action.hover',
};

const metadataNumber = (value: unknown): number => {
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
};

const metadataString = (value: unknown, fallback: string): string =>
  typeof value === 'string' && value.trim() ? value : fallback;

const formatCount = (count: number, singular: string, plural = `${singular}s`): string =>
  `${count.toLocaleString()} ${count === 1 ? singular : plural}`;

const formatGraphRAGLabel = (value: string): string =>
  value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, letter => letter.toUpperCase());

const LOCAL_UPLOAD_ACCEPT = '.txt,.pdf,.docx,.doc,.md,.html,.json,.csv';
const GRAPHRAG_UPLOAD_ACCEPT = '.txt,.md,.pdf,.json,.csv,.html,.xml,.eml,.doc,.docx,.pptx,.xlsx,.rtf,.odt,.ods,.odp,.epub,.zip,.png,.jpg,.jpeg,.gif,.bmp,.tif,.tiff,.webp,.svg,message/rfc822,image/*';
const BINARY_DOCUMENT_TYPES = new Set(['pdf', 'docx', 'epub', 'pptx', 'xlsx', 'odt', 'ods', 'odp']);

const detectDocumentType = (filename: string): string => {
  const ext = filename.split('.').pop()?.toLowerCase();
  if (ext === 'pdf') return 'pdf';
  if (ext === 'docx' || ext === 'doc') return 'docx';
  if (ext === 'pptx') return 'pptx';
  if (ext === 'xlsx') return 'xlsx';
  if (ext === 'rtf') return 'rtf';
  if (ext === 'odt') return 'odt';
  if (ext === 'ods') return 'ods';
  if (ext === 'odp') return 'odp';
  if (ext === 'md') return 'md';
  if (ext === 'html' || ext === 'htm') return 'html';
  if (ext === 'json') return 'json';
  if (ext === 'csv') return 'csv';
  if (ext === 'epub') return 'epub';
  return 'txt';
};

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const pipelineSummaryNumber = (result: any, key: string): number => {
  const value = result?.summary?.[key] ?? result?.[key];
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
};

const graphRAGResultSeverity = (result: any): 'success' | 'warning' | 'error' | 'info' => {
  if (result?.status === 'submitted') return 'success';
  if (result?.status === 'complete') return 'success';
  if (result?.status === 'partial') return 'warning';
  if (result?.status === 'preview_only') return 'warning';
  if (result?.status === 'upstream_failed' || result?.status === 'error') return 'error';
  return 'info';
};

const graphRAGRuns = (result: any): any[] => {
  if (Array.isArray(result?.runs)) return result.runs.filter(Boolean);
  return result ? [result] : [];
};

const graphRAGBatchEvidenceChips = (result: any): Array<{ label: string; value: number }> => {
  const evidence = result?.summary?.evidence_totals || {};
  return [
    { label: 'archives', value: metadataNumber(evidence.archive_entries) },
    { label: 'attachments', value: metadataNumber(evidence.attachments) },
    { label: 'OCR', value: metadataNumber(evidence.ocr_blocks) },
    { label: 'layout regions', value: metadataNumber(evidence.layout_elements) },
    { label: 'tables', value: metadataNumber(evidence.distinct_tables) },
    { label: 'table cells', value: metadataNumber(evidence.table_elements) },
    { label: 'charts', value: metadataNumber(evidence.charts) },
    { label: 'diagrams', value: metadataNumber(evidence.diagrams) },
    { label: 'shapes', value: metadataNumber(evidence.shapes) },
    { label: 'connectors', value: metadataNumber(evidence.connectors) },
    { label: 'forms', value: metadataNumber(evidence.forms) },
    { label: 'links', value: metadataNumber(evidence.links) },
    { label: 'code blocks', value: metadataNumber(evidence.code_blocks) },
    { label: 'comments', value: metadataNumber(evidence.comments) },
    { label: 'revisions', value: metadataNumber(evidence.revisions) },
    { label: 'speaker notes', value: metadataNumber(evidence.speaker_notes) },
    { label: 'styled text', value: metadataNumber(evidence.styled_text_blocks) },
    { label: 'messages', value: metadataNumber(evidence.message_body_parts) },
    { label: 'media', value: metadataNumber(evidence.media) },
    { label: 'images', value: metadataNumber(evidence.images) },
    { label: 'pivot tables', value: metadataNumber(evidence.pivot_tables) },
    { label: 'hidden slides', value: metadataNumber(evidence.hidden_slides) },
    { label: 'hidden sheets', value: metadataNumber(evidence.hidden_sheets) },
    { label: 'legacy streams', value: metadataNumber(evidence.legacy_office_streams) },
    { label: 'legacy text', value: metadataNumber(evidence.legacy_office_text_blocks) },
    { label: 'graph edges', value: metadataNumber(evidence.graph_relationships) },
    { label: 'semantic edges', value: metadataNumber(evidence.semantic_relationships) },
    { label: 'table facts', value: metadataNumber(evidence.table_facts) },
    { label: 'chart edges', value: metadataNumber(evidence.chart_relationships) },
    { label: 'layout edges', value: metadataNumber(evidence.layout_relationships) },
    { label: 'link edges', value: metadataNumber(evidence.link_relationships) },
    { label: 'code edges', value: metadataNumber(evidence.code_block_relationships) },
    { label: 'diagram edges', value: metadataNumber(evidence.diagram_relationships) },
    { label: 'style edges', value: metadataNumber(evidence.styled_text_relationships) },
    { label: 'attachment edges', value: metadataNumber(evidence.attachment_relationships) },
    { label: 'archive edges', value: metadataNumber(evidence.archive_relationships) },
    { label: 'OCR edges', value: metadataNumber(evidence.ocr_relationships) },
    { label: 'notes edges', value: metadataNumber(evidence.speaker_notes_relationships) },
  ].filter(chip => chip.value > 0);
};

const graphRAGBatchPayloadChips = (result: any): Array<{ label: string; value: number }> =>
  Object.entries(result?.summary?.payload_kinds || {})
    .map(([kind, value]) => ({ label: formatGraphRAGLabel(kind), value: metadataNumber(value) }))
    .filter(chip => chip.value > 0);

const graphRAGResultMessage = (result: any): string => {
  const runs = graphRAGRuns(result);
  const chunks = runs.reduce((total, run) => total + (pipelineSummaryNumber(run, 'chunk_count') || pipelineSummaryNumber(run, 'total_chunks')), 0);
  const rawEntities = runs.reduce(
    (total, run) => total + (pipelineSummaryNumber(run, 'entity_candidate_count') || Number(run?.knowledge_graph?.entities || 0)),
    0,
  );
  const rawRelationships = runs.reduce(
    (total, run) => total + (pipelineSummaryNumber(run, 'relationship_candidate_count') || Number(run?.knowledge_graph?.relationships || 0)),
    0,
  );
  const entities = Number.isFinite(Number(rawEntities)) ? Number(rawEntities) : 0;
  const relationships = Number.isFinite(Number(rawRelationships)) ? Number(rawRelationships) : 0;
  const totalFiles = metadataNumber(result?.total_files);
  const processed = metadataNumber(result?.processed);
  const failed = metadataNumber(result?.failed);
  const prefix = totalFiles > 0
    ? `GraphRAG batch finished: ${processed.toLocaleString()} of ${totalFiles.toLocaleString()} processed${failed ? `, ${failed.toLocaleString()} failed` : ''}`
    : result?.status === 'submitted'
    ? 'Document submitted to GraphRAG'
    : result?.status === 'preview_only'
      ? 'Pipeline preview generated; upstream GraphRAG ingest was skipped'
      : result?.status === 'upstream_failed'
        ? 'Pipeline preview generated, but upstream GraphRAG ingest failed'
        : 'GraphRAG pipeline run finished';

  return `${prefix}: ${formatCount(chunks, 'chunk')}, ${formatCount(entities, 'entity candidate')}, ${formatCount(relationships, 'relationship candidate')}`;
};

const GraphRAGUploadSummary: React.FC<{ result: any }> = ({ result }) => {
  const results = Array.isArray(result?.results) ? result.results.filter(Boolean) : [];
  const totalFiles = metadataNumber(result?.total_files);
  const processed = metadataNumber(result?.processed);
  const failed = metadataNumber(result?.failed);
  const concurrency = metadataNumber(result?.concurrency);
  const remoteSubmitted = metadataNumber(result?.summary?.remote_submitted);
  const localPreviewOnly = metadataNumber(result?.summary?.local_preview_only);
  const nextSteps: string[] = Array.isArray(result?.summary?.next_steps)
    ? result.summary.next_steps.filter((step: unknown): step is string => typeof step === 'string' && Boolean(step.trim()))
    : [];
  const payloadChips = graphRAGBatchPayloadChips(result);
  const evidenceChips = graphRAGBatchEvidenceChips(result);

  if (!totalFiles && results.length === 0 && payloadChips.length === 0 && evidenceChips.length === 0) {
    return null;
  }

  return (
    <Paper
      variant="outlined"
      sx={{
        mb: 2,
        p: 1.25,
        borderRadius: '8px',
        bgcolor: theme => alpha(theme.palette.success.main, 0.035),
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap', mb: payloadChips.length || evidenceChips.length ? 0.75 : 0 }}>
        {totalFiles > 0 && (
          <Chip label={`${processed.toLocaleString()}/${totalFiles.toLocaleString()} processed`} color={failed > 0 ? 'warning' : 'success'} size="small" />
        )}
        {failed > 0 && <Chip label={`${failed.toLocaleString()} failed`} color="warning" size="small" />}
        {concurrency > 0 && <Chip label={`${concurrency.toLocaleString()} workers`} variant="outlined" size="small" />}
        {remoteSubmitted > 0 && <Chip label={`${remoteSubmitted.toLocaleString()} remote`} variant="outlined" size="small" />}
        {localPreviewOnly > 0 && <Chip label={`${localPreviewOnly.toLocaleString()} local traces`} variant="outlined" size="small" />}
      </Box>

      {(payloadChips.length > 0 || evidenceChips.length > 0) && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: results.length || nextSteps.length ? 0.75 : 0 }}>
          {payloadChips.map(chip => (
            <Chip key={chip.label} label={`${chip.value.toLocaleString()} ${chip.label}`} variant="outlined" size="small" sx={{ height: 22 }} />
          ))}
          {evidenceChips.map(chip => (
            <Chip key={chip.label} label={`${chip.value.toLocaleString()} ${chip.label}`} variant="outlined" size="small" sx={{ height: 22 }} />
          ))}
        </Box>
      )}

      {results.length > 0 && (
        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: nextSteps.length ? 0.75 : 0 }}>
          {results.slice(0, 10).map((item: any, index: number) => {
            const filename = metadataString(item?.filename, `File ${index + 1}`);
            const status = metadataString(item?.status, item?.success === false ? 'error' : 'complete');
            return (
              <Chip
                key={`${filename}-${index}`}
                label={`${filename}: ${formatGraphRAGLabel(status)}`}
                title={filename}
                size="small"
                variant="outlined"
                color={item?.success === false ? 'warning' : 'default'}
                sx={{
                  height: 22,
                  maxWidth: 240,
                  '& .MuiChip-label': {
                    minWidth: 0,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  },
                }}
              />
            );
          })}
          {results.length > 10 && <Chip label={`+${results.length - 10} more`} size="small" variant="outlined" sx={{ height: 22 }} />}
        </Box>
      )}

      {nextSteps.slice(0, 2).map(step => (
        <Typography key={step} variant="caption" color="text.secondary" sx={{ display: 'block', fontFamily: 'monospace', lineHeight: 1.35 }}>
          {step}
        </Typography>
      ))}
    </Paper>
  );
};

// Domain Tree Component with Drag & Drop Support
const DomainTree: React.FC<{
  domains: Domain[];
  selectedDomain: string | null;
  expandedDomains: Set<string>;
  dragOverDomain: string | null;
  onSelect: (domain: Domain) => void;
  onToggle: (domainId: string) => void;
  onDelete: (domain: Domain) => void;
  onDragOver: (domainId: string | null) => void;
  onFileDrop: (domain: Domain, files: FileList) => void;
}> = ({ domains, selectedDomain, expandedDomains, dragOverDomain, onSelect, onToggle, onDelete, onDragOver, onFileDrop }) => {
  const handleDragOver = (e: React.DragEvent, domainId: string) => {
    e.preventDefault();
    e.stopPropagation();
    onDragOver(domainId);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onDragOver(null);
  };

  const handleDrop = (e: React.DragEvent, domain: Domain) => {
    e.preventDefault();
    e.stopPropagation();
    onDragOver(null);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileDrop(domain, e.dataTransfer.files);
    }
  };

  const handleDomainKeyDown = (event: React.KeyboardEvent, domain: Domain) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onSelect(domain);
    }
  };

  const renderDomain = (domain: Domain, level: number = 0) => {
    const isExpanded = expandedDomains.has(domain.id);
    const isSelected = selectedDomain === domain.id;
    const hasChildren = domain.children && domain.children.length > 0;
    const isDragOver = dragOverDomain === domain.id;

    return (
      <Box key={domain.id}>
        <Box
          role="option"
          aria-selected={isSelected}
          tabIndex={0}
          onClick={() => onSelect(domain)}
          onKeyDown={(event) => handleDomainKeyDown(event, domain)}
          onDragOver={(e) => handleDragOver(e, domain.id)}
          onDragLeave={handleDragLeave}
          onDrop={(e) => handleDrop(e, domain)}
          sx={{
            pl: 2 + level * 2,
            pr: 0.5,
            py: 0.25,
            minHeight: 34,
            display: 'flex',
            alignItems: 'center',
            gap: 0,
            borderRadius: '8px',
            mb: 0.25,
            bgcolor: isDragOver
              ? alpha('#10B981', 0.3)
              : isSelected
                ? alpha('#6366F1', 0.15)
                : 'transparent',
            border: isDragOver ? '1px dashed #10B981' : '1px solid transparent',
            transition: 'all 0.2s ease',
            '&:hover': { bgcolor: alpha('#6366F1', 0.1) },
            '&:focus-visible': {
              outline: '2px solid',
              outlineColor: alpha('#8B5CF6', 0.7),
              outlineOffset: 1,
            },
          }}
        >
          {hasChildren && (
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                onToggle(domain.id);
              }}
              sx={{ mr: 0.5, width: 24, height: 24 }}
            >
              {isExpanded ? <ExpandIcon /> : <ChevronRightIcon />}
            </IconButton>
          )}
          <ListItemIcon sx={{ minWidth: 28, '& .MuiSvgIcon-root': { fontSize: 18 } }}>
            {isDragOver ? (
              <CloudUploadIcon sx={{ color: '#10B981' }} />
            ) : isExpanded ? (
              <FolderOpenIcon sx={{ color: '#F59E0B' }} />
            ) : (
              <FolderIcon sx={{ color: '#F59E0B' }} />
            )}
          </ListItemIcon>
          <ListItemText
            primary={isDragOver ? `Drop files in ${domain.name}` : domain.name}
            secondary={isDragOver ? 'Release to upload' : undefined}
            primaryTypographyProps={{
              noWrap: true,
              fontSize: '0.84rem',
              sx: { color: isDragOver ? '#10B981' : 'inherit', fontWeight: isDragOver ? 700 : 600 }
            }}
            secondaryTypographyProps={{
              noWrap: true,
              fontSize: '0.72rem',
              lineHeight: 1.2,
            }}
            sx={{ my: 0, minWidth: 0 }}
          />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {!isDragOver && domain.document_count > 0 && (
              <Chip
                label={domain.document_count}
                size="small"
                sx={{
                  height: 20,
                  bgcolor: alpha('#6366F1', 0.2),
                  '& .MuiChip-label': { px: 1, fontSize: 11 },
                }}
              />
            )}
            <Tooltip title="Delete domain">
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(domain);
                }}
                sx={{ width: 26, height: 26, color: 'text.secondary', '&:hover': { color: '#EF4444' } }}
              >
                <DeleteIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        {hasChildren && isExpanded && (
          <Box>
            {domain.children!.map(child => renderDomain(child, level + 1))}
          </Box>
        )}
      </Box>
    );
  };

  return (
    <List component="div" dense disablePadding sx={{ py: 0 }}>
      {domains.map(domain => renderDomain(domain))}
    </List>
  );
};

// Chunk Viewer Component
const ChunkViewer: React.FC<{
  chunks: Chunk[];
  loading: boolean;
}> = ({ chunks, loading }) => {
  const [selectedChunk, setSelectedChunk] = useState<number>(0);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1 }}>
        <CircularProgress size={16} />
        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700 }}>
          Loading chunks
        </Typography>
      </Box>
    );
  }

  if (chunks.length === 0) {
    return (
      <Box
        sx={{
          px: 1,
          py: 0.75,
          borderRadius: '8px',
          border: '1px dashed',
          borderColor: 'divider',
          bgcolor: 'action.hover',
        }}
      >
        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700 }}>
          No chunks available
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <Typography variant="subtitle2">
          Chunk {selectedChunk + 1} of {chunks.length}
        </Typography>
        <Slider
          value={selectedChunk}
          onChange={(_, v) => setSelectedChunk(v as number)}
          min={0}
          max={chunks.length - 1}
          sx={{ flex: 1 }}
        />
      </Box>
      <Paper
        sx={{
          p: 2,
          bgcolor: alpha('#000', 0.3),
          maxHeight: 300,
          overflow: 'auto',
          fontFamily: 'monospace',
          fontSize: 13,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {chunks[selectedChunk]?.section_header && (
          <Chip
            label={chunks[selectedChunk].section_header}
            size="small"
            sx={{ mb: 1, bgcolor: alpha('#6366F1', 0.2) }}
          />
        )}
        {chunks[selectedChunk]?.content}
        {chunks[selectedChunk]?.metadata && Object.keys(chunks[selectedChunk].metadata!).length > 0 && (
          <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block' }}>
              Metadata
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {Object.entries(chunks[selectedChunk].metadata!).map(([key, value]) => (
                <Chip
                  key={key}
                  label={`${key}: ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`}
                  size="small"
                  sx={{ bgcolor: alpha('#10B981', 0.2), fontSize: 11 }}
                />
              ))}
            </Box>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

// Main Component
const DocumentsPage: React.FC = () => {
  const isNarrow = useMediaQuery('(max-width:599.95px)');
  const [domains, setDomains] = useState<Domain[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDomain, setSelectedDomain] = useState<Domain | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [mobileDetailsExpanded, setMobileDetailsExpanded] = useState(false);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(false);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [expandedDomains, setExpandedDomains] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [totalDocuments, setTotalDocuments] = useState(0);

  // Batch operations
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());
  const [batchProcessing, setBatchProcessing] = useState(false);

  // Drag and drop
  const [dragOverDomain, setDragOverDomain] = useState<string | null>(null);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  // Processing queue
  const [processingQueue, setProcessingQueue] = useState<any>(null);

  // Dialogs
  const [domainDialogOpen, setDomainDialogOpen] = useState(false);
  const [documentDialogOpen, setDocumentDialogOpen] = useState(false);
  const [processDialogOpen, setProcessDialogOpen] = useState(false);
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [batchIngestionDialogOpen, setBatchIngestionDialogOpen] = useState(false);
  const [documentActionsAnchorEl, setDocumentActionsAnchorEl] = useState<HTMLElement | null>(null);
  const [selectedDocumentActionsAnchorEl, setSelectedDocumentActionsAnchorEl] = useState<HTMLElement | null>(null);
  const {
    confirmationAction: confirmAction,
    confirmationLoading: confirmingAction,
    requestConfirmation,
    closeConfirmation: closeConfirmDialog,
    confirmAction: handleConfirmAction,
  } = useConfirmableAction();

  // Forms
  const [domainForm, setDomainForm] = useState({
    name: '',
    description: '',
    parent_id: '',
    chunk_size: 512,
    chunk_overlap: 50,
    embedding_model: 'all-MiniLM-L6-v2',
    chunking_strategy: 'semantic',
  });
  const [documentForm, setDocumentForm] = useState({
    name: '',
    doc_type: 'txt',
    content: '',
    source_url: '',
  });
  const [processForm, setProcessForm] = useState({
    chunking_strategy: 'semantic',
    chunk_size: 512,
    chunk_overlap: 50,
    embedding_model: 'all-MiniLM-L6-v2',
    async: true,
  });
  const [uploadMode, setUploadMode] = useState<'paste' | 'file' | 'url'>('paste');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedGraphRAGFiles, setSelectedGraphRAGFiles] = useState<File[]>([]);
  const [uploadDestination, setUploadDestination] = useState<'local' | 'graphrag'>('local');
  const [graphragUploading, setGraphragUploading] = useState(false);
  const [graphragResult, setGraphragResult] = useState<any>(null);
  const graphRAGUploadFiles = selectedGraphRAGFiles.length > 0
    ? selectedGraphRAGFiles
    : selectedFile
      ? [selectedFile]
      : [];

  // Load domains
  const loadDomains = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/rag/domains');
      const flatDomains = res.domains || [];

      // Build tree structure
      const domainMap = new Map<string, Domain>();
      flatDomains.forEach((d: Domain) => {
        domainMap.set(d.id, { ...d, children: [] });
      });

      const rootDomains: Domain[] = [];
      domainMap.forEach(domain => {
        if (domain.parent_id && domainMap.has(domain.parent_id)) {
          domainMap.get(domain.parent_id)!.children!.push(domain);
        } else {
          rootDomains.push(domain);
        }
      });

      setDomains(rootDomains);
    } catch (err) {
      console.error('Failed to load domains:', err);
    }
  }, []);

  // Load documents
  const loadDocuments = useCallback(async () => {
    setLoading(true);
    try {
      const params: Record<string, unknown> = {
        limit: rowsPerPage,
        offset: page * rowsPerPage,
      };
      if (selectedDomain) params.domain_id = selectedDomain.id;
      if (searchTerm) params.search = searchTerm;
      if (statusFilter) params.status = statusFilter;
      if (typeFilter) params.doc_type = typeFilter;

      const res = await api.get('/api/v1/rag/documents', { params });
      setDocuments(res.documents || []);
      setTotalDocuments(res.total || 0);
    } catch (err) {
      console.error('Failed to load documents:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedDomain, searchTerm, statusFilter, typeFilter, page, rowsPerPage]);

  // Load chunks for selected document
  const loadChunks = useCallback(async (documentId: string) => {
    setChunksLoading(true);
    setChunks([]);
    try {
      const res = await api.get(`/api/v1/rag/documents/${documentId}/chunks`);
      setChunks(res.chunks || []);
    } catch (err) {
      console.error('Failed to load chunks:', err);
    } finally {
      setChunksLoading(false);
    }
  }, []);

  const activeDocument = selectedDocument && documents.some(doc => doc.id === selectedDocument.id)
    ? selectedDocument
    : documents[0] || null;

  useEffect(() => {
    loadDomains();
  }, [loadDomains]);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  useEffect(() => {
    if (activeDocument) {
      loadChunks(activeDocument.id);
    } else {
      setChunks([]);
    }
  }, [activeDocument?.id, loadChunks]);

  useEffect(() => {
    if (isNarrow) {
      setMobileDetailsExpanded(false);
    }
  }, [activeDocument?.id, isNarrow]);

  // Handlers
  const handleCreateDomain = async () => {
    try {
      await api.post('/api/v1/rag/domains', {
        ...domainForm,
        parent_id: domainForm.parent_id || undefined,
      });
      setDomainDialogOpen(false);
      setDomainForm({
        name: '',
        description: '',
        parent_id: '',
        chunk_size: 512,
        chunk_overlap: 50,
        embedding_model: 'all-MiniLM-L6-v2',
        chunking_strategy: 'semantic',
      });
      loadDomains();
    } catch (err) {
      setError('Failed to create domain');
    }
  };

  const handleDeleteDomain = (domain: Domain) => {
    requestConfirmation({
      title: 'Delete domain?',
      message: `Delete domain "${domain.name}"? This will also delete all documents in it.`,
      confirmLabel: 'Delete domain',
      destructive: true,
      onConfirm: async () => {
        try {
          await api.delete(`/api/v1/rag/domains/${domain.id}`, { params: { cascade: true } });
          setSelectedDomain(null);
          loadDomains();
        } catch (err) {
          setError('Failed to delete domain');
        }
      },
    });
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    const file = selectedFiles[0];
    if (!file) return;

    setSelectedFile(file);
    setSelectedGraphRAGFiles(uploadDestination === 'graphrag' ? selectedFiles : []);
    setGraphragResult(null);

    // Auto-detect document type from file extension
    const docType = detectDocumentType(file.name);

    if (uploadDestination === 'graphrag') {
      setDocumentForm(prev => ({
        ...prev,
        name: selectedFiles.length > 1 ? `${selectedFiles.length} GraphRAG documents` : prev.name || file.name,
        doc_type: docType,
        content: '',
      }));
      return;
    }

    // Read file content - use base64 for binary files (PDF, DOCX)
    try {
      let content: string;
      if (BINARY_DOCUMENT_TYPES.has(docType)) {
        // Read binary files as base64
        const arrayBuffer = await file.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        content = btoa(binary);
      } else {
        // Read text files as text
        content = await file.text();
      }
      setDocumentForm(prev => ({
        ...prev,
        name: prev.name || file.name,
        doc_type: docType,
        content: content,
      }));
    } catch (err) {
      setError('Failed to read file');
    }
  };

  // Handle drag and drop files onto a domain
  const handleFileDrop = async (domain: Domain, files: FileList) => {
    if (files.length === 0) return;

    setUploadingFiles(true);
    setSuccess(`Uploading ${files.length} file(s) to "${domain.name}"...`);

    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];

      try {
	        // Auto-detect document type from file extension
	        const docType = detectDocumentType(file.name);

	        // Read file content
	        let content: string;
	        if (BINARY_DOCUMENT_TYPES.has(docType)) {
          // Read binary files as base64
          const arrayBuffer = await file.arrayBuffer();
          const bytes = new Uint8Array(arrayBuffer);
          let binary = '';
          for (let j = 0; j < bytes.byteLength; j++) {
            binary += String.fromCharCode(bytes[j]);
          }
          content = btoa(binary);
        } else {
          // Read text files as text
          content = await file.text();
        }

        // Upload document
        await api.post('/api/v1/rag/documents', {
          name: file.name,
          doc_type: docType,
          content: content,
          domain_id: domain.id,
        });

        successCount++;
      } catch (err) {
        console.error(`Failed to upload ${file.name}:`, err);
        errorCount++;
      }
    }

    setUploadingFiles(false);

    if (errorCount === 0) {
      setSuccess(`Successfully uploaded ${successCount} file(s) to "${domain.name}"`);
    } else {
      setError(`Uploaded ${successCount} file(s), ${errorCount} failed`);
    }

    setTimeout(() => setSuccess(null), 3000);

    // Refresh data
    loadDocuments();
    loadDomains();
  };

  const handleCreateDocument = async () => {
    if (!selectedDomain) {
      setError('Please select a domain first');
      return;
    }

    // Handle GraphRAG upload
    if (uploadDestination === 'graphrag' && graphRAGUploadFiles.length > 0) {
      await handleGraphRAGUpload();
      return;
    }

    // Handle local RAG upload
    try {
      await api.post('/api/v1/rag/documents', {
        ...documentForm,
        domain_id: selectedDomain.id,
        source_url: documentForm.source_url || undefined,
      });
      setDocumentDialogOpen(false);
      setDocumentForm({ name: '', doc_type: 'txt', content: '', source_url: '' });
      setSelectedFile(null);
      setSelectedGraphRAGFiles([]);
      setUploadMode('paste');
      loadDocuments();
      loadDomains();
    } catch (err) {
      setError('Failed to create document');
    }
  };

  const handleGraphRAGUpload = async () => {
    if (graphRAGUploadFiles.length === 0) {
      setError('Please select one or more files to upload');
      return;
    }

    try {
      setGraphragUploading(true);
      setGraphragResult(null);

      const pipelineOptions = {
        domain: selectedDomain?.name || 'general',
        useSemanticChunking: true,
        buildKnowledgeGraph: true,
        sampleLimit: 6,
        submitStructuredText: true,
      };
      const result = graphRAGUploadFiles.length > 1
        ? await api.uploadGraphRAGDocumentsWithPipeline(graphRAGUploadFiles, {
          ...pipelineOptions,
          maxConcurrency: 3,
        })
        : await api.uploadGraphRAGDocumentWithPipeline(graphRAGUploadFiles[0], pipelineOptions);

      setGraphragResult(result);
      setSuccess(graphRAGResultMessage(result));
      setTimeout(() => setSuccess(null), 5000);

      // Don't close dialog immediately - show results
    } catch (err: any) {
      setError(err.message || 'Failed to upload to GraphRAG');
    } finally {
      setGraphragUploading(false);
    }
  };

  const handleDeleteDocument = (doc: Document) => {
    requestConfirmation({
      title: 'Delete document?',
      message: `Delete document "${doc.name}"? This removes it from the database and vector store.`,
      confirmLabel: 'Delete document',
      destructive: true,
      onConfirm: async () => {
        try {
          await api.delete(`/api/v1/rag/documents/${doc.id}`);
          setSelectedDocument(null);
          loadDocuments();
          loadDomains();
          setSuccess('Document deleted successfully');
          setTimeout(() => setSuccess(null), 3000);
        } catch (err) {
          setError('Failed to delete document');
        }
      },
    });
  };

  // Batch operations handlers
  const toggleDocumentSelection = (docId: string) => {
    const newSelection = new Set(selectedDocuments);
    if (newSelection.has(docId)) {
      newSelection.delete(docId);
    } else {
      newSelection.add(docId);
    }
    setSelectedDocuments(newSelection);
  };

  const selectAllDocuments = () => {
    const allIds = new Set(documents.map(d => d.id));
    setSelectedDocuments(allIds);
  };

  const clearSelection = () => {
    setSelectedDocuments(new Set());
  };

  const handleBatchProcess = async () => {
    if (selectedDocuments.size === 0) return;

    try {
      setBatchProcessing(true);
      const response = await fetch('/api/v1/rag/documents/batch-process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_ids: Array.from(selectedDocuments),
          chunking_strategy: 'semantic',
          embedding_model: selectedDomain?.embedding_model || 'all-MiniLM-L6-v2'
        })
      });

      if (!response.ok) throw new Error('Batch processing failed');

      const result = await response.json();
      setSuccess(result.message);
      setTimeout(() => setSuccess(null), 3000);
      clearSelection();
      loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to batch process documents');
    } finally {
      setBatchProcessing(false);
    }
  };

  const handleProcessAllPending = async () => {
    try {
      setBatchProcessing(true);
      const response = await fetch('/api/v1/rag/documents/process-all-pending', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          domain_id: selectedDomain?.id,
          chunking_strategy: 'semantic',
          embedding_model: selectedDomain?.embedding_model || 'all-MiniLM-L6-v2'
        })
      });

      if (!response.ok) throw new Error('Failed to process pending documents');

      const result = await response.json();
      setSuccess(result.message);
      setTimeout(() => setSuccess(null), 3000);
      loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process pending documents');
    } finally {
      setBatchProcessing(false);
    }
  };

  const handleReindexDomain = () => {
    if (!selectedDomain) return;
    const domain = selectedDomain;
    requestConfirmation({
      title: 'Reindex domain?',
      message: `Reindex entire domain "${domain.name}"? This will reprocess all documents.`,
      confirmLabel: 'Reindex domain',
      onConfirm: async () => {
        try {
          setBatchProcessing(true);
          const response = await fetch(`/api/v1/rag/domains/${domain.id}/reindex`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              chunking_strategy: 'semantic',
              embedding_model: domain.embedding_model,
              recreate_collection: false
            })
          });

          if (!response.ok) throw new Error('Domain reindex failed');

          const result = await response.json();
          setSuccess(result.message);
          setTimeout(() => setSuccess(null), 3000);
          loadDocuments();
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to reindex domain');
        } finally {
          setBatchProcessing(false);
        }
      },
    });
  };

  const handleClearCollection = () => {
    if (!selectedDomain) return;
    const domain = selectedDomain;
    requestConfirmation({
      title: 'Clear vector collection?',
      message: `Clear all vectors from "${domain.name}"? This deletes all embeddings but keeps documents. You'll need to reindex afterwards.`,
      confirmLabel: 'Clear vectors',
      destructive: true,
      onConfirm: async () => {
        try {
          setBatchProcessing(true);
          const response = await fetch(`/api/v1/rag/domains/${domain.id}/collection`, {
            method: 'DELETE'
          });

          if (!response.ok) throw new Error('Failed to clear collection');

          const result = await response.json();
          setSuccess(`Cleared ${result.vectors_deleted} vectors from collection`);
          setTimeout(() => setSuccess(null), 3000);
          loadDocuments();
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to clear collection');
        } finally {
          setBatchProcessing(false);
        }
      },
    });
  };


  // Knowledge Graph extraction handlers
  const handleExtractKnowledge = async (docId: string) => {
    try {
      setLoading(true);
      const result = await api.extractDocumentKnowledge(docId);
      setSuccess(`Extracted ${result.entity_count} entities and ${result.relationship_count} relationships from "${result.document_name}"`);
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to extract knowledge from document');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchExtractKnowledge = async () => {
    if (selectedDocuments.size === 0) return;

    try {
      setBatchProcessing(true);
      const result = await api.batchExtractDocumentKnowledge(Array.from(selectedDocuments));
      setSuccess(`Queued ${result.documents_queued} documents for knowledge extraction`);
      setTimeout(() => setSuccess(null), 5000);
      clearSelection();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to batch extract knowledge');
    } finally {
      setBatchProcessing(false);
    }
  };

  // Load processing queue
  const loadProcessingQueue = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/rag/processing/queue');
      if (response.ok) {
        const data = await response.json();
        setProcessingQueue(data);
      }
    } catch (err) {
      console.error('Failed to load processing queue:', err);
    }
  }, []);

  useEffect(() => {
    const interval = setInterval(loadProcessingQueue, 30000); // Reduced polling frequency to prevent connection saturation
    loadProcessingQueue();
    return () => clearInterval(interval);
  }, [loadProcessingQueue]);

  const handleProcessDocument = async () => {
    const documentToProcess = selectedDocument || activeDocument;
    if (!documentToProcess) return;
    try {
      setLoading(true);
      await api.post(`/api/v1/rag/documents/${documentToProcess.id}/process`, processForm);
      setProcessDialogOpen(false);
      loadDocuments();
      loadChunks(documentToProcess.id);
    } catch (err) {
      setError('Failed to process document');
    } finally {
      setLoading(false);
    }
  };

  const toggleDomain = (domainId: string) => {
    setExpandedDomains(prev => {
      const next = new Set(prev);
      if (next.has(domainId)) {
        next.delete(domainId);
      } else {
        next.add(domainId);
      }
      return next;
    });
  };

  const flatDomains = flattenDomains(domains);
  const knownDomainDocuments = flatDomains.reduce((sum, domain) => sum + domain.document_count, 0);
  const knownDomainChunks = flatDomains.reduce((sum, domain) => sum + domain.total_chunks, 0);
  const visibleDocumentChunks = documents.reduce((sum, doc) => sum + doc.chunk_count, 0);
  const selectedDomainName = selectedDomain?.name || 'All domains';
  const scopeDocumentCount = selectedDomain?.document_count ?? (knownDomainDocuments || totalDocuments);
  const scopeChunkCount = selectedDomain?.total_chunks ?? (knownDomainChunks || visibleDocumentChunks);
  const handleAllDocumentsSelect = () => {
    setSelectedDomain(null);
    setPage(0);
  };
  const shouldOpenVectorSearch = !isNarrow && scopeChunkCount > 0;
  const processingCount = processingQueue?.processing?.count || 0;
  const pendingQueueCount = processingQueue?.pending?.count || 0;
  const queueErrorCount = processingQueue?.errors?.count || 0;
  const errorDocuments = documents.filter(doc => doc.status === 'error').length;
  const issueDocumentCount = Math.max(errorDocuments, queueErrorCount);
  const queueSummaryParts = [
    processingCount > 0 ? `${processingCount.toLocaleString()} processing` : '',
    pendingQueueCount > 0 ? formatCount(pendingQueueCount, 'queued item') : '',
  ].filter(Boolean);
  const allVisibleDocumentsSelected = documents.length > 0 && selectedDocuments.size === documents.length;
  const activeDocumentKgEntities = metadataNumber(activeDocument?.metadata?.kg_entities);
  const activeDocumentKgRelationships = metadataNumber(activeDocument?.metadata?.kg_relationships);
  const showChunkPanel = chunksLoading || chunks.length > 0;
  const closeDocumentActionsMenu = () => setDocumentActionsAnchorEl(null);
  const closeSelectedDocumentActionsMenu = () => setSelectedDocumentActionsAnchorEl(null);
  const showDocumentDetailsBody = !isNarrow || mobileDetailsExpanded;

  return (
    <Box
      sx={{
        width: '100%',
        maxWidth: '100vw',
        overflow: 'hidden',
        px: { xs: 2, sm: 3, md: 4 },
        py: 1.25,
        boxSizing: 'border-box',
        minHeight: '100%',
      }}
    >
      {error && (
        <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 1.25 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      {uploadingFiles && (
        <Alert
          severity="info"
          sx={{ mb: 1.25 }}
          icon={<CircularProgress size={20} />}
        >
          Uploading files... Please wait.
        </Alert>
      )}

      <Grid container spacing={1.5}>
        {/* Domain Tree Sidebar */}
        <Grid item xs={12} lg={2.4}>
          <Paper sx={{
            p: 1.25,
            height: 'auto',
            maxHeight: { xs: 360, lg: 'calc(100vh - 154px)' },
            overflow: 'auto',
            bgcolor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: '8px',
          }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1, gap: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 800, display: 'flex', alignItems: 'center', gap: 0.75 }}>
                <FolderIcon sx={{ fontSize: 18, color: '#F59E0B' }} />
                Domains
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.35 }}>
                <Tooltip title="Create domain">
                  <IconButton
                    size="small"
                    aria-label="Create domain"
                    onClick={() => setDomainDialogOpen(true)}
                    sx={{ width: 28, height: 28, borderRadius: '8px', border: '1px solid', borderColor: 'divider' }}
                  >
                    <AddIcon sx={{ fontSize: 17 }} />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Refresh domains">
                  <IconButton size="small" aria-label="Refresh domains" onClick={loadDomains} sx={{ width: 28, height: 28 }}>
                    <RefreshIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>

            <Box
              role="listbox"
              aria-label="Document domains"
              data-document-domains
              sx={{ display: 'grid', gap: 0.35 }}
            >
              <Box
                role="option"
                aria-selected={!selectedDomain}
                tabIndex={0}
                onClick={handleAllDocumentsSelect}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    handleAllDocumentsSelect();
                  }
                }}
                sx={{
                  minHeight: 32,
                  px: 1,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.75,
                  borderRadius: '8px',
                  border: '1px dashed',
                  borderColor: !selectedDomain ? alpha('#6366F1', 0.52) : 'divider',
                  bgcolor: !selectedDomain ? alpha('#6366F1', 0.12) : 'transparent',
                  color: !selectedDomain ? '#818CF8' : 'text.primary',
                  fontSize: '0.82rem',
                  fontWeight: 800,
                  cursor: 'pointer',
                  '&:hover': { bgcolor: alpha('#6366F1', 0.1) },
                  '&:focus-visible': {
                    outline: '2px solid',
                    outlineColor: alpha('#8B5CF6', 0.7),
                    outlineOffset: 1,
                  },
                }}
              >
                <DocIcon sx={{ fontSize: 16, color: 'inherit' }} />
                <Box component="span" sx={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  All Documents
                </Box>
              </Box>

              {domains.length > 0 && (
                <Typography variant="caption" color="text.secondary" sx={{ display: { xs: 'none', sm: 'block' }, mt: 0.25 }}>
                  Drop files on a domain to upload.
                </Typography>
              )}

              {domains.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <FolderIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    No domains yet
                  </Typography>
                  <Button
                    size="small"
                    startIcon={<AddIcon />}
                    onClick={() => setDomainDialogOpen(true)}
                    sx={{ mt: 1 }}
                  >
                    Create Domain
                  </Button>
                </Box>
              ) : (
                <DomainTree
                  domains={domains}
                  selectedDomain={selectedDomain?.id || null}
                  expandedDomains={expandedDomains}
                  dragOverDomain={dragOverDomain}
                  onSelect={(domain) => {
                    setSelectedDomain(domain);
                    setPage(0);
                  }}
                  onToggle={toggleDomain}
                  onDelete={handleDeleteDomain}
                  onDragOver={setDragOverDomain}
                  onFileDrop={handleFileDrop}
                />
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Document List */}
        <Grid item xs={12} lg={5.4}>
          <Paper sx={{
            p: 1.25,
            bgcolor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: '8px',
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, mb: 1.25 }}>
              <Box sx={{ minWidth: 0, display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 800, display: 'flex', alignItems: 'center', gap: 0.75 }}>
                  <DocIcon sx={{ fontSize: 18, color: '#10B981' }} />
                  Documents
                </Typography>
                <Chip
                  size="small"
                  icon={<DocIcon sx={{ fontSize: 15 }} />}
                  label={selectedDomainName}
                  sx={{
                    maxWidth: 180,
                    height: 22,
                    bgcolor: alpha('#10B981', 0.14),
                    border: `1px solid ${alpha('#10B981', 0.24)}`,
                    color: '#6EE7B7',
                    fontWeight: 700,
                    '& .MuiChip-icon': { color: '#6EE7B7' },
                    '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis' },
                    display: { xs: 'none', md: 'inline-flex' },
                  }}
                />
                <Chip
                  size="small"
                  label={formatCount(scopeDocumentCount, 'document')}
                  sx={{
                    height: 22,
                    bgcolor: alpha('#6366F1', 0.12),
                    border: `1px solid ${alpha('#6366F1', 0.24)}`,
                    color: '#A5B4FC',
                    fontWeight: 700,
                    display: { xs: 'none', md: 'inline-flex' },
                  }}
                />
                {issueDocumentCount > 0 && (
                  <Chip
                    size="small"
                    label={formatCount(issueDocumentCount, 'error')}
                    sx={{
                      height: 22,
                      bgcolor: alpha('#EF4444', 0.12),
                      border: `1px solid ${alpha('#EF4444', 0.24)}`,
                      color: '#FCA5A5',
                      fontWeight: 700,
                      display: { xs: 'none', md: 'inline-flex' },
                    }}
                  />
                )}
                {queueSummaryParts.length > 0 && (
                  <Chip
                    size="small"
                    label={queueSummaryParts.join(' / ')}
                    sx={{
                      height: 22,
                      bgcolor: alpha('#F59E0B', 0.12),
                      border: `1px solid ${alpha('#F59E0B', 0.24)}`,
                      color: '#FCD34D',
                      fontWeight: 700,
                      display: { xs: 'none', md: 'inline-flex' },
                    }}
                  />
                )}
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexShrink: 0 }}>
                <Tooltip title="Refresh documents">
                  <IconButton
                    size="small"
                    aria-label="Refresh documents"
                    onClick={() => {
                      loadDocuments();
                      loadProcessingQueue();
                    }}
                    sx={{ ...documentIconButtonSx, width: 32, height: 32, '&:hover': { bgcolor: alpha('#10B981', 0.12) } }}
                  >
                    <RefreshIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
                {selectedDomain && (
                  <Tooltip title="Add document">
                    <Button
                      variant="contained"
                      size="small"
                      startIcon={<UploadIcon />}
                      aria-label="Add document"
                      onClick={() => setDocumentDialogOpen(true)}
                      sx={compactDocumentToolbarButtonSx}
                    >
                      <Box component="span" sx={documentToolbarLabelSx}>Add</Box>
                    </Button>
                  </Tooltip>
                )}
                <Tooltip title="More document actions">
                  <IconButton
                    size="small"
                    aria-label="More document actions"
                    onClick={(event) => setDocumentActionsAnchorEl(event.currentTarget)}
                    sx={{ ...documentIconButtonSx, width: 32, height: 32 }}
                  >
                    <MoreIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
                <Menu
                  anchorEl={documentActionsAnchorEl}
                  open={Boolean(documentActionsAnchorEl)}
                  onClose={closeDocumentActionsMenu}
                  anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                  transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                >
                  <MenuItem
                    onClick={() => {
                      closeDocumentActionsMenu();
                      setBatchIngestionDialogOpen(true);
                    }}
                  >
                    <ListItemIcon><UploadIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Batch upload" />
                  </MenuItem>
                  <MenuItem
                    disabled={batchProcessing}
                    onClick={() => {
                      closeDocumentActionsMenu();
                      handleProcessAllPending();
                    }}
                  >
                    <ListItemIcon><ProcessIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Process pending" />
                  </MenuItem>
                  <MenuItem
                    disabled={!selectedDomain || batchProcessing}
                    onClick={() => {
                      closeDocumentActionsMenu();
                      handleReindexDomain();
                    }}
                  >
                    <ListItemIcon><SyncIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Reindex domain" />
                  </MenuItem>
                  <MenuItem
                    disabled={!selectedDomain || batchProcessing}
                    onClick={() => {
                      closeDocumentActionsMenu();
                      handleClearCollection();
                    }}
                  >
                    <ListItemIcon><StorageIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Clear vectors" />
                  </MenuItem>
                  <MenuItem
                    disabled={documents.length === 0}
                    onClick={() => {
                      closeDocumentActionsMenu();
                      allVisibleDocumentsSelected ? clearSelection() : selectAllDocuments();
                    }}
                  >
                    <ListItemIcon>{allVisibleDocumentsSelected ? <DeselectIcon fontSize="small" /> : <SelectAllIcon fontSize="small" />}</ListItemIcon>
                    <ListItemText primary={allVisibleDocumentsSelected ? 'Deselect visible' : 'Select visible'} />
                  </MenuItem>
                </Menu>
              </Box>
            </Box>

            {/* Filters */}
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: 'minmax(0, 1fr) minmax(0, 1fr)', sm: 'minmax(0, 1fr) 120px 120px' },
                gap: 1,
                mb: 1.25,
              }}
            >
              <TextField
                size="small"
                placeholder="Search documents..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.disabled' }} />,
                }}
                sx={{ gridColumn: { xs: '1 / -1', sm: 'auto' }, minWidth: 0 }}
              />
              <FormControl size="small" sx={{ minWidth: 0 }}>
                <InputLabel>Status</InputLabel>
                <Select
                  value={statusFilter}
                  label="Status"
                  onChange={(e) => setStatusFilter(e.target.value)}
                >
                  <MenuItem value="">All</MenuItem>
                  {Object.entries(STATUS_CONFIG).map(([key, config]) => (
                    <MenuItem key={key} value={key}>{config.label}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl size="small" sx={{ minWidth: 0 }}>
                <InputLabel>Type</InputLabel>
                <Select
                  value={typeFilter}
                  label="Type"
                  onChange={(e) => setTypeFilter(e.target.value)}
                >
                  <MenuItem value="">All</MenuItem>
                  {Object.keys(DOC_TYPE_ICONS).map(type => (
                    <MenuItem key={type} value={type}>{type.toUpperCase()}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            {/* Batch Actions Bar */}
            {selectedDocuments.size > 0 && (
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'primary.dark', color: 'white' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {selectedDocuments.size} document(s) selected
                  </Typography>
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={<ProcessIcon />}
                    onClick={handleBatchProcess}
                    disabled={batchProcessing}
                    sx={{ bgcolor: 'white', color: 'primary.main', '&:hover': { bgcolor: 'grey.100' } }}
                  >
                    Process Selected
                  </Button>
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={<KnowledgeGraphIcon />}
                    onClick={handleBatchExtractKnowledge}
                    disabled={batchProcessing}
                    sx={{ bgcolor: '#6366F1', color: 'white', '&:hover': { bgcolor: '#4F46E5' } }}
                  >
                    Extract to Knowledge Graph
                  </Button>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={clearSelection}
                    sx={{ borderColor: 'white', color: 'white' }}
                  >
                    Clear Selection
                  </Button>
                </Box>
              </Paper>
            )}

            {/* Processing Queue Status */}
            {processingQueue && (processingQueue.processing.count > 0 || processingQueue.pending.count > 0) && (
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>Processing Queue</Typography>
                <Typography variant="caption">
                  {processingQueue.processing.count} processing, {processingQueue.pending.count} pending
                  {processingQueue.errors.count > 0 && `, ${processingQueue.errors.count} errors`}
                </Typography>
              </Alert>
            )}

            {/* Document List */}
              <Box
                data-document-list
                role="listbox"
                aria-label="Documents"
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                borderRadius: '8px',
                overflow: 'hidden',
              }}
            >
              <Box
                sx={{
                  display: { xs: 'none', sm: 'grid' },
                  gridTemplateColumns: '34px minmax(0, 1fr) 92px 70px',
                  gap: 1,
                  alignItems: 'center',
                  px: 1,
                  py: 0.5,
                  bgcolor: 'action.hover',
                  borderBottom: '1px solid',
                  borderColor: 'divider',
                }}
              >
                <IconButton
                  size="small"
                  aria-label={allVisibleDocumentsSelected ? 'Deselect visible documents' : 'Select visible documents'}
                  onClick={allVisibleDocumentsSelected ? clearSelection : selectAllDocuments}
                  disabled={documents.length === 0}
                  sx={{ width: 28, height: 28 }}
                >
                  {allVisibleDocumentsSelected ? <CheckBoxIcon sx={{ fontSize: 18 }} /> : <CheckBoxOutlineBlankIcon sx={{ fontSize: 18 }} />}
                </IconButton>
                <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 800, textTransform: 'uppercase' }}>
                  Document
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 800, textTransform: 'uppercase' }}>
                  Status
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 800, textTransform: 'uppercase', textAlign: 'right' }}>
                  Index
                </Typography>
              </Box>

              {loading ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : documents.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <DocIcon sx={{ fontSize: 40, color: 'text.disabled', mb: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    No documents found
                  </Typography>
                </Box>
              ) : (
                documents.map(doc => {
                  const statusConfig = STATUS_CONFIG[doc.status] || STATUS_CONFIG.pending;
                  const isSelected = selectedDocuments.has(doc.id);
                  const isActive = activeDocument?.id === doc.id;
                  const kgStatus = doc.metadata?.kg_status;
                  const kgEntities = metadataNumber(doc.metadata?.kg_entities);
                  const kgRels = metadataNumber(doc.metadata?.kg_relationships);
                  const kgError = metadataString(doc.metadata?.kg_error, 'Extraction failed');

                  const kgBadge = (() => {
                    if (kgStatus === 'extracted') {
                      return (
                        <Tooltip title={`${kgEntities} entities, ${kgRels} relationships`}>
                          <Chip
                            label={`KG ${kgEntities}`}
                            size="small"
                            sx={{ height: 20, fontSize: 10, bgcolor: alpha('#6366F1', 0.15), color: '#A5B4FC', fontWeight: 800 }}
                          />
                        </Tooltip>
                      );
                    }
                    if (kgStatus === 'extracting') {
                      return <Chip label="KG..." size="small" sx={{ height: 20, fontSize: 10, fontWeight: 800 }} />;
                    }
                    if (kgStatus === 'error') {
                      return (
                        <Tooltip title={kgError}>
                          <Chip label="KG error" size="small" color="error" sx={{ height: 20, fontSize: 10, fontWeight: 800 }} />
                        </Tooltip>
                      );
                    }
                    return null;
                  })();

                  return (
                    <Box
                      key={doc.id}
                      role="option"
                      aria-selected={isActive}
                      tabIndex={0}
                      onClick={() => setSelectedDocument(doc)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault();
                          setSelectedDocument(doc);
                        }
                      }}
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: { xs: '30px minmax(0, 1fr) auto', sm: '34px minmax(0, 1fr) 92px 70px' },
                        gap: { xs: 0.75, sm: 1 },
                        alignItems: 'center',
                        px: { xs: 0.75, sm: 1 },
                        py: { xs: 0.45, sm: 0.55 },
                        minWidth: 0,
                        cursor: 'pointer',
                        bgcolor: isActive ? alpha('#6366F1', 0.14) : 'transparent',
                        borderLeft: '3px solid',
                        borderLeftColor: isActive ? '#8B5CF6' : 'transparent',
                        borderBottom: '1px solid',
                        borderColor: 'divider',
                        '&:last-child': { borderBottom: 0 },
                        '&:hover': {
                          bgcolor: isActive ? alpha('#6366F1', 0.18) : 'action.hover',
                        },
                      }}
                    >
                      <IconButton
                        size="small"
                        aria-label={isSelected ? `Deselect ${doc.name}` : `Select ${doc.name}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          toggleDocumentSelection(doc.id);
                        }}
                        sx={{ width: 28, height: 28 }}
                      >
                        {isSelected ? <CheckBoxIcon color="primary" sx={{ fontSize: 18 }} /> : <CheckBoxOutlineBlankIcon sx={{ fontSize: 18 }} />}
                      </IconButton>

                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
                        <Box sx={{ color: 'text.secondary', display: { xs: 'none', sm: 'flex' }, flexShrink: 0, '& .MuiSvgIcon-root': { fontSize: 18 } }}>
                          {DOC_TYPE_ICONS[doc.doc_type] || <DocIcon />}
                        </Box>
                        <Box sx={{ minWidth: 0 }}>
                          <Typography variant="body2" noWrap title={doc.name} sx={{ fontWeight: 800, lineHeight: 1.2 }}>
                            {doc.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block', lineHeight: 1.15 }}>
                            {doc.doc_type.toUpperCase()} / {new Date(doc.created_at).toLocaleDateString()} / {doc.chunk_count.toLocaleString()} chunks
                          </Typography>
                        </Box>
                      </Box>

                      <Chip
                        icon={isNarrow ? undefined : statusConfig.icon as React.ReactElement}
                        label={statusConfig.label}
                        size="small"
                        sx={{
                          justifySelf: 'start',
                          maxWidth: { xs: 78, sm: 90 },
                          height: 22,
                          bgcolor: alpha(statusConfig.color, 0.15),
                          color: statusConfig.color,
                          fontWeight: 800,
                          '& .MuiChip-icon': { color: statusConfig.color },
                          '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis' },
                        }}
                      />

                      <Box sx={{ minWidth: 0, textAlign: 'right', display: { xs: 'none', sm: 'block' } }}>
                        <Typography variant="body2" sx={{ fontWeight: 800, lineHeight: 1.15 }}>
                          {doc.chunk_count}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.15 }}>
                          chunks
                        </Typography>
                        {kgBadge && <Box sx={{ mt: 0.35, display: 'flex', justifyContent: 'flex-end' }}>{kgBadge}</Box>}
                      </Box>
                    </Box>
                  );
                })
              )}
            </Box>

            {totalDocuments > rowsPerPage && (
              <TablePagination
                component="div"
                count={totalDocuments}
                page={page}
                onPageChange={(_, newPage) => setPage(newPage)}
                rowsPerPage={rowsPerPage}
                labelDisplayedRows={({ from, to, count }) => (
                  `${from}-${to} / ${count === -1 ? `>${to}` : count}`
                )}
                onRowsPerPageChange={(e) => {
                  setRowsPerPage(parseInt(e.target.value, 10));
                  setPage(0);
                }}
                sx={{
                  overflow: 'hidden',
                  borderTop: '1px solid',
                  borderColor: 'divider',
                  '& .MuiTablePagination-toolbar': {
                    minHeight: { xs: 40, sm: 52 },
                    px: { xs: 0.75, sm: 2 },
                    py: { xs: 0.5, sm: 0 },
                    display: { xs: 'grid', sm: 'flex' },
                    gridTemplateColumns: { xs: 'minmax(0, 1fr) auto', sm: 'initial' },
                    gap: { xs: 0.5, sm: 1 },
                  },
                  '& .MuiTablePagination-spacer': {
                    display: { xs: 'none', sm: 'block' },
                  },
                  '& .MuiTablePagination-selectLabel, & .MuiTablePagination-input': {
                    display: { xs: 'none', sm: 'flex' },
                  },
                  '& .MuiTablePagination-displayedRows': {
                    m: 0,
                    minWidth: 0,
                    fontSize: { xs: '0.8rem', sm: '0.875rem' },
                    color: 'text.primary',
                  },
                  '& .MuiTablePagination-actions': {
                    ml: { xs: 0, sm: 2 },
                    display: 'flex',
                    alignItems: 'center',
                    '& .MuiIconButton-root': {
                      width: { xs: 30, sm: 36 },
                      height: { xs: 30, sm: 36 },
                    },
                  },
                }}
              />
            )}
          </Paper>
        </Grid>

        {/* Document Details Panel */}
        <Grid item xs={12} lg={4.2} sx={{ display: { xs: activeDocument ? 'block' : 'none', lg: 'block' } }}>
          <Paper sx={{
            p: 1.25,
            bgcolor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: '8px',
          }}>
            {activeDocument ? (
              <>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: { xs: 'center', sm: 'flex-start' }, gap: 1, mb: showDocumentDetailsBody ? 1 : 0 }}>
                  <Box sx={{ minWidth: 0 }}>
                    <Typography
                      variant="h6"
                      title={activeDocument.name}
                      sx={{ fontWeight: 800, mb: { xs: 0.15, sm: 0.5 }, fontSize: { xs: '0.9rem', sm: '1rem' }, lineHeight: 1.15 }}
                      noWrap
                    >
                      {activeDocument.name}
                    </Typography>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ display: { xs: 'block', sm: 'none' }, lineHeight: 1.1 }}
                      noWrap
                    >
                      {STATUS_CONFIG[activeDocument.status]?.label || activeDocument.status} / {activeDocument.chunk_count.toLocaleString()} chunks
                    </Typography>
                    <Box sx={{ display: { xs: 'none', sm: 'flex' }, gap: 0.75, flexWrap: 'wrap' }}>
                      <Chip
                        label={activeDocument.doc_type.toUpperCase()}
                        size="small"
                        sx={{ height: 22, bgcolor: alpha('#6366F1', 0.2), fontWeight: 700 }}
                      />
                      <Chip
                        label={STATUS_CONFIG[activeDocument.status]?.label || activeDocument.status}
                        size="small"
                        sx={{
                          height: 22,
                          bgcolor: alpha(STATUS_CONFIG[activeDocument.status]?.color || '#6B7280', 0.2),
                          color: STATUS_CONFIG[activeDocument.status]?.color || '#6B7280',
                          fontWeight: 700,
                        }}
                      />
                    </Box>
                  </Box>
                  <Box sx={{ display: { xs: 'none', sm: 'flex' }, gap: 0.5, flexShrink: 0, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                    <Tooltip title="Process document">
                      <span>
                        <IconButton
                          size="small"
                          aria-label="Process document"
                          onClick={() => {
                            setSelectedDocument(activeDocument);
                            setProcessDialogOpen(true);
                          }}
                          disabled={activeDocument.status === 'processing'}
                          sx={documentIconButtonSx}
                        >
                          <ProcessIcon sx={{ fontSize: 17 }} />
                        </IconButton>
                      </span>
                    </Tooltip>
                    <Tooltip title="Document actions">
                      <IconButton
                        size="small"
                        aria-label="Document actions"
                        onClick={(event) => setSelectedDocumentActionsAnchorEl(event.currentTarget)}
                        sx={documentIconButtonSx}
                      >
                        <MoreIcon sx={{ fontSize: 17 }} />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Box sx={{ display: { xs: 'flex', sm: 'none' }, alignItems: 'center', gap: 0.5, flexShrink: 0 }}>
                    <Tooltip title="Document actions">
                      <IconButton
                        size="small"
                        aria-label="Document actions"
                        onClick={(event) => setSelectedDocumentActionsAnchorEl(event.currentTarget)}
                        sx={{ ...documentIconButtonSx, width: 30, height: 30 }}
                      >
                        <MoreIcon sx={{ fontSize: 17 }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title={mobileDetailsExpanded ? 'Hide document details' : 'Show document details'}>
                      <IconButton
                        size="small"
                        aria-label={mobileDetailsExpanded ? 'Hide document details' : 'Show document details'}
                        onClick={() => setMobileDetailsExpanded(prev => !prev)}
                        sx={{ ...documentIconButtonSx, width: 30, height: 30 }}
                      >
                        <ExpandIcon sx={{ fontSize: 18, transform: mobileDetailsExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }} />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
                <Menu
                  anchorEl={selectedDocumentActionsAnchorEl}
                  open={Boolean(selectedDocumentActionsAnchorEl)}
                  onClose={closeSelectedDocumentActionsMenu}
                  anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                  transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                >
                  <MenuItem
                    disabled={activeDocument.status === 'processing'}
                    sx={{ display: { xs: 'flex', sm: 'none' } }}
                    onClick={() => {
                      closeSelectedDocumentActionsMenu();
                      setSelectedDocument(activeDocument);
                      setProcessDialogOpen(true);
                    }}
                  >
                    <ListItemIcon><ProcessIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Process" />
                  </MenuItem>
                  <MenuItem
                    onClick={() => {
                      closeSelectedDocumentActionsMenu();
                      setSelectedDocument(activeDocument);
                      setViewDialogOpen(true);
                    }}
                  >
                    <ListItemIcon><ViewIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="View content" />
                  </MenuItem>
                  <MenuItem
                    disabled={activeDocument.status !== 'ready'}
                    onClick={() => {
                      closeSelectedDocumentActionsMenu();
                      handleExtractKnowledge(activeDocument.id);
                    }}
                  >
                    <ListItemIcon><KnowledgeGraphIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Extract KG" />
                  </MenuItem>
                  <MenuItem
                    onClick={() => {
                      closeSelectedDocumentActionsMenu();
                      handleDeleteDocument(activeDocument);
                    }}
                  >
                    <ListItemIcon><DeleteIcon fontSize="small" color="error" /></ListItemIcon>
                    <ListItemText primary="Delete" />
                  </MenuItem>
                </Menu>

                {showDocumentDetailsBody && (
                  <>
                    <Box
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                        gap: 0.25,
                        mb: 1,
                        p: 0.75,
                        borderRadius: '8px',
                        border: '1px solid',
                        borderColor: 'divider',
                        bgcolor: 'action.hover',
                      }}
                    >
                      {[
                        ['Chunks', activeDocument.chunk_count > 0 ? activeDocument.chunk_count.toLocaleString() : '-'],
                        ['Tokens', activeDocument.token_count > 0 ? activeDocument.token_count.toLocaleString() : '-'],
                        ['Created', new Date(activeDocument.created_at).toLocaleDateString()],
                        [
                          'KG',
                          activeDocument.metadata?.kg_status === 'extracted'
                            ? `${activeDocumentKgEntities} / ${activeDocumentKgRelationships}`
                            : metadataString(activeDocument.metadata?.kg_status, '-'),
                        ],
                      ].map(([label, value]) => {
                        const displayValue = String(value);
                        return (
                          <Box
                            key={label}
                            sx={{
                              display: 'flex',
                              alignItems: 'baseline',
                              justifyContent: 'space-between',
                              gap: 1,
                              minWidth: 0,
                              px: 0.25,
                              py: 0.2,
                            }}
                          >
                            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.68rem', fontWeight: 800, textTransform: 'uppercase' }} noWrap>
                              {label}
                            </Typography>
                            <Typography variant="caption" title={displayValue} sx={{ fontWeight: 800, minWidth: 0 }} noWrap>
                              {displayValue}
                            </Typography>
                          </Box>
                        );
                      })}
                    </Box>

                    {showChunkPanel && (
                      <>
                        <Divider sx={{ my: 1 }} />
                        <Typography variant="subtitle2" sx={{ mb: 0.75, fontWeight: 800 }}>
                          Chunks ({chunks.length})
                        </Typography>
                        <ChunkViewer chunks={chunks} loading={chunksLoading} />
                      </>
                    )}
                  </>
                )}
              </>
            ) : (
              <Box
                sx={{
                  py: 3,
                  px: 2,
                  textAlign: 'center',
                  borderRadius: '8px',
                  border: '1px dashed',
                  borderColor: 'divider',
                  bgcolor: 'action.hover',
                }}
              >
                <DocIcon sx={{ fontSize: 34, color: 'text.disabled', mb: 1 }} />
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  No document selected.
                </Typography>
                <Button
                  size="small"
                  variant="contained"
                  startIcon={<UploadIcon sx={{ fontSize: 16 }} />}
                  onClick={() => setDocumentDialogOpen(true)}
                  disabled={!selectedDomain}
                  sx={{ height: 32, borderRadius: '8px', fontWeight: 800 }}
                >
                  Add document
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      <Box sx={{ mt: 1.5 }}>
        <VectorSearchPanel selectedDomain={selectedDomain} defaultExpanded={shouldOpenVectorSearch} />
      </Box>

      {/* Create Domain Dialog */}
      <Dialog open={domainDialogOpen} onClose={() => setDomainDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Domain</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Domain Name"
            value={domainForm.name}
            onChange={(e) => setDomainForm(prev => ({ ...prev, name: e.target.value }))}
            sx={{ mt: 2, mb: 2 }}
          />
          <TextField
            fullWidth
            label="Description"
            value={domainForm.description}
            onChange={(e) => setDomainForm(prev => ({ ...prev, description: e.target.value }))}
            multiline
            rows={2}
            sx={{ mb: 2 }}
          />
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Chunk Size"
                type="number"
                value={domainForm.chunk_size}
                onChange={(e) => setDomainForm(prev => ({ ...prev, chunk_size: parseInt(e.target.value) }))}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Chunk Overlap"
                type="number"
                value={domainForm.chunk_overlap}
                onChange={(e) => setDomainForm(prev => ({ ...prev, chunk_overlap: parseInt(e.target.value) }))}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDomainDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateDomain} variant="contained" disabled={!domainForm.name}>
            Create
          </Button>
        </DialogActions>
      </Dialog>

      {/* Create Document Dialog */}
      <Dialog open={documentDialogOpen} onClose={() => setDocumentDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Add Document to {selectedDomain?.name}</DialogTitle>
        <DialogContent>
          {/* Destination Toggle */}
          <Box sx={{ mb: 2, mt: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Select Destination:
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip
                label="Local RAG"
	                onClick={() => {
	                  setUploadDestination('local');
	                  setSelectedGraphRAGFiles([]);
	                  setGraphragResult(null);
	                }}
                color={uploadDestination === 'local' ? 'primary' : 'default'}
                variant={uploadDestination === 'local' ? 'filled' : 'outlined'}
                icon={<StorageIcon />}
                clickable
              />
              <Chip
                label="GraphRAG (Neo4j + Knowledge Graph)"
                onClick={() => {
	                  setUploadDestination('graphrag');
	                  setUploadMode('file');
	                  setSelectedGraphRAGFiles(selectedFile ? [selectedFile] : []);
	                  setGraphragResult(null);
	                }}
                color={uploadDestination === 'graphrag' ? 'success' : 'default'}
                variant={uploadDestination === 'graphrag' ? 'filled' : 'outlined'}
                icon={<KnowledgeGraphIcon />}
                clickable
              />
            </Box>
          </Box>

          {uploadDestination === 'graphrag' && (
            <Alert severity="info" sx={{ mb: 2 }}>
              Documents will be traced through structure extraction, chunking, graph candidate generation, embedding planning, and upstream GraphRAG ingest.
            </Alert>
          )}

          {graphragResult && (
            <>
              <Alert severity={graphRAGResultSeverity(graphragResult)} sx={{ mb: 2 }}>
                <Typography variant="body2" fontWeight={600}>
                  GraphRAG Pipeline Run
                </Typography>
                <Typography variant="caption" component="div">
                  {graphRAGResultMessage(graphragResult)}
                </Typography>
              </Alert>
              <GraphRAGUploadSummary result={graphragResult} />
              {graphRAGRuns(graphragResult).map((run, index) => (
                <GraphRAGPipelineTrace
                  key={run?.run_id || `${run?.document?.filename || 'run'}-${index}`}
                  result={run}
                  title={graphRAGRuns(graphragResult).length > 1 ? `GraphRAG Pipeline Run ${index + 1}` : 'GraphRAG Pipeline Run'}
                  dense={graphRAGRuns(graphragResult).length > 1}
                />
              ))}
            </>
          )}

          <TextField
            fullWidth
            label="Document Name"
            value={documentForm.name}
            onChange={(e) => setDocumentForm(prev => ({ ...prev, name: e.target.value }))}
            sx={{ mb: 2 }}
          />

          {uploadDestination === 'local' && (
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={6}>
                <FormControl fullWidth>
                  <InputLabel>Document Type</InputLabel>
                  <Select
                    value={documentForm.doc_type}
                    label="Document Type"
                    onChange={(e) => setDocumentForm(prev => ({ ...prev, doc_type: e.target.value }))}
                  >
                    {Object.keys(DOC_TYPE_ICONS).map(type => (
                      <MenuItem key={type} value={type}>{type.toUpperCase()}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Source URL (optional)"
                  value={documentForm.source_url}
                  onChange={(e) => setDocumentForm(prev => ({ ...prev, source_url: e.target.value }))}
                />
              </Grid>
            </Grid>
          )}

          {/* Input Method Tabs */}
          <Tabs
            value={uploadMode}
            onChange={(_, newValue) => setUploadMode(newValue)}
            sx={{ mb: 2 }}
          >
            <Tab
              icon={<CloudUploadIcon />}
              label="Upload File"
              value="file"
            />
            {uploadDestination === 'local' && (
              <>
                <Tab icon={<TextIcon />} label="Paste Content" value="paste" />
                <Tab icon={<LinkIcon />} label="From URL" value="url" />
              </>
            )}
          </Tabs>

          {/* File Upload Mode */}
          {uploadMode === 'file' && (
            <Box>
              <Button
                variant="outlined"
                component="label"
                fullWidth
                startIcon={<CloudUploadIcon />}
                sx={{
                  height: 120,
                  borderStyle: 'dashed',
                  borderWidth: 2,
                  '&:hover': {
                    borderStyle: 'dashed',
                    borderWidth: 2,
                  },
                }}
              >
                {uploadDestination === 'graphrag' && graphRAGUploadFiles.length > 1 ? (
                  <Box sx={{ textAlign: 'center', width: '100%', px: 1 }}>
                    <DocIcon sx={{ fontSize: 36, mb: 0.75 }} />
                    <Typography fontWeight={700}>
                      {graphRAGUploadFiles.length} files selected
                    </Typography>
                    <Typography variant="caption" color="text.secondary" component="div" noWrap>
                      {graphRAGUploadFiles.slice(0, 3).map(file => file.name).join(', ')}
                      {graphRAGUploadFiles.length > 3 ? `, +${graphRAGUploadFiles.length - 3} more` : ''}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatFileSize(graphRAGUploadFiles.reduce((total, file) => total + file.size, 0))}
                    </Typography>
                  </Box>
                ) : selectedFile ? (
                  <Box sx={{ textAlign: 'center' }}>
                    <DocIcon sx={{ fontSize: 40, mb: 1 }} />
                    <Typography>{selectedFile.name}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatFileSize(selectedFile.size)}
                    </Typography>
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center' }}>
                    <CloudUploadIcon sx={{ fontSize: 40, mb: 1 }} />
                    <Typography>Click to upload or drag and drop</Typography>
                    <Typography variant="caption" color="text.secondary">
	                      {uploadDestination === 'graphrag'
	                        ? 'PDF, Office, OpenDocument, EPUB, ZIP, images, text/data'
	                        : 'TXT, PDF, DOCX, MD, HTML, JSON, CSV'}
                    </Typography>
                  </Box>
                )}
	                <input
	                  type="file"
	                  hidden
	                  accept={uploadDestination === 'graphrag' ? GRAPHRAG_UPLOAD_ACCEPT : LOCAL_UPLOAD_ACCEPT}
	                  multiple={uploadDestination === 'graphrag'}
	                  onChange={handleFileSelect}
	                />
	              </Button>
              {uploadDestination === 'graphrag' && graphRAGUploadFiles.length > 1 && (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 1 }}>
                  {graphRAGUploadFiles.slice(0, 8).map(file => (
                    <Chip
                      key={`${file.name}-${file.size}-${file.lastModified}`}
                      label={`${file.name} (${formatFileSize(file.size)})`}
                      size="small"
                      variant="outlined"
                    />
                  ))}
                  {graphRAGUploadFiles.length > 8 && (
                    <Chip label={`+${graphRAGUploadFiles.length - 8} more`} size="small" />
                  )}
                </Box>
              )}
              {selectedFile && (
                <Button
                  size="small"
                  onClick={() => {
                    setSelectedFile(null);
                    setSelectedGraphRAGFiles([]);
                    setDocumentForm(prev => ({ ...prev, content: '' }));
                  }}
                  sx={{ mt: 1 }}
                >
                  Clear {uploadDestination === 'graphrag' && graphRAGUploadFiles.length > 1 ? 'Files' : 'File'}
                </Button>
              )}
            </Box>
          )}

          {/* Paste Content Mode */}
          {uploadMode === 'paste' && (
            <TextField
              fullWidth
              label="Content"
              value={documentForm.content}
              onChange={(e) => setDocumentForm(prev => ({ ...prev, content: e.target.value }))}
              multiline
              rows={10}
              placeholder="Paste document content here..."
            />
          )}

          {/* URL Mode */}
          {uploadMode === 'url' && (
            <Box>
              <TextField
                fullWidth
                label="URL"
                value={documentForm.source_url}
                onChange={(e) => setDocumentForm(prev => ({ ...prev, source_url: e.target.value }))}
                placeholder="https://example.com/document"
                sx={{ mb: 2 }}
              />
              <Alert severity="info">
                The document will be fetched from the URL and processed.
              </Alert>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
	            setDocumentDialogOpen(false);
	            setSelectedFile(null);
	            setSelectedGraphRAGFiles([]);
	            setUploadMode('paste');
	            setUploadDestination('local');
	            setGraphragResult(null);
          }}>
            {graphragResult ? 'Close' : 'Cancel'}
          </Button>
          {!graphragResult && (
            <Button
              onClick={handleCreateDocument}
              variant="contained"
              disabled={
	                graphragUploading ||
	                !documentForm.name ||
	                (uploadDestination === 'graphrag' && graphRAGUploadFiles.length === 0) ||
	                (uploadDestination === 'local' && !documentForm.content && uploadMode !== 'url')
	              }
	              startIcon={graphragUploading ? <CircularProgress size={16} /> : null}
	            >
	              {graphragUploading
	                ? 'Processing...'
	                : uploadDestination === 'graphrag'
	                  ? graphRAGUploadFiles.length > 1
	                    ? `Upload ${graphRAGUploadFiles.length} to GraphRAG`
	                    : 'Upload to GraphRAG'
	                  : 'Add Document'}
	            </Button>
          )}
        </DialogActions>
      </Dialog>

      {/* Process Document Dialog */}
      <Dialog open={processDialogOpen} onClose={() => setProcessDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Process Document: {selectedDocument?.name}</DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            Processing will chunk the document and create embeddings for vector search.
          </Alert>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Chunking Strategy</InputLabel>
            <Select
              value={processForm.chunking_strategy}
              label="Chunking Strategy"
              onChange={(e) => setProcessForm(prev => ({ ...prev, chunking_strategy: e.target.value }))}
            >
              <MenuItem value="fixed">Fixed Size</MenuItem>
              <MenuItem value="semantic">Semantic (Sentences/Paragraphs)</MenuItem>
              <MenuItem value="recursive">Recursive (Hierarchical)</MenuItem>
            </Select>
          </FormControl>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Chunk Size"
                type="number"
                value={processForm.chunk_size}
                onChange={(e) => setProcessForm(prev => ({ ...prev, chunk_size: parseInt(e.target.value) }))}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Chunk Overlap"
                type="number"
                value={processForm.chunk_overlap}
                onChange={(e) => setProcessForm(prev => ({ ...prev, chunk_overlap: parseInt(e.target.value) }))}
              />
            </Grid>
          </Grid>
          <FormControl fullWidth>
            <InputLabel>Embedding Model</InputLabel>
            <Select
              value={processForm.embedding_model}
              label="Embedding Model"
              onChange={(e) => setProcessForm(prev => ({ ...prev, embedding_model: e.target.value }))}
            >
              <MenuItem value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 (Fast)</MenuItem>
              <MenuItem value="all-mpnet-base-v2">all-mpnet-base-v2 (Quality)</MenuItem>
              <MenuItem value="BAAI/bge-large-en-v1.5">BGE Large (Best)</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setProcessDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleProcessDocument} variant="contained" disabled={loading}>
            {loading ? <CircularProgress size={20} /> : 'Process Document'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* View Content Dialog */}
      <Dialog open={viewDialogOpen} onClose={() => setViewDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Document Content: {selectedDocument?.name}</DialogTitle>
        <DialogContent>
          <Paper
            sx={{
              p: 2,
              bgcolor: alpha('#000', 0.3),
              maxHeight: 500,
              overflow: 'auto',
              fontFamily: 'monospace',
              fontSize: 13,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {selectedDocument?.content || 'No content available'}
          </Paper>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Batch Ingestion Dialog */}
      <BatchIngestionDialog
        open={batchIngestionDialogOpen}
        onClose={() => setBatchIngestionDialogOpen(false)}
        onSuccess={() => {
          setBatchIngestionDialogOpen(false);
          loadDocuments();
          setSuccess('Batch upload completed successfully');
        }}
        destination={uploadDestination}
      />

      <ConfirmationDialog
        action={confirmAction}
        loading={confirmingAction}
        onClose={closeConfirmDialog}
        onConfirm={handleConfirmAction}
      />
    </Box>
  );
};

export default DocumentsPage;
