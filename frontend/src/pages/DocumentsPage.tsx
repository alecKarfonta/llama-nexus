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
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Tooltip,
  Alert,
  CircularProgress,
  Divider,
  Card,
  CardContent,
  CardActions,
  Tabs,
  Tab,
  LinearProgress,
  Breadcrumbs,
  Link,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Collapse,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
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
  Settings as SettingsIcon,
  PlayArrow as ProcessIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  Storage as StorageIcon,
  Memory as MemoryIcon,
  Code as CodeIcon,
  PictureAsPdf as PdfIcon,
  Article as ArticleIcon,
  DataObject as JsonIcon,
  TableChart as CsvIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  Autorenew as ReprocessIcon,
  RemoveCircleOutline as RemoveIcon,
  Sync as SyncIcon,
  SelectAll as SelectAllIcon,
  Deselect as DeselectIcon,
  CheckBox as CheckBoxIcon,
  CheckBoxOutlineBlank as CheckBoxOutlineBlankIcon,
  AccountTree as KnowledgeGraphIcon,
} from '@mui/icons-material';
import { apiService as api } from '../services/api';
import { BatchIngestionDialog } from '../components/documents/BatchIngestionDialog';

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
}

// Document type icons
const DOC_TYPE_ICONS: Record<string, React.ReactNode> = {
  pdf: <PdfIcon />,
  docx: <ArticleIcon />,
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

// Domain Tree Component with Drag & Drop Support
const DomainTree: React.FC<{
  domains: Domain[];
  selectedDomain: string | null;
  expandedDomains: Set<string>;
  dragOverDomain: string | null;
  onSelect: (domain: Domain) => void;
  onToggle: (domainId: string) => void;
  onEdit: (domain: Domain) => void;
  onDelete: (domain: Domain) => void;
  onDragOver: (domainId: string | null) => void;
  onFileDrop: (domain: Domain, files: FileList) => void;
}> = ({ domains, selectedDomain, expandedDomains, dragOverDomain, onSelect, onToggle, onEdit, onDelete, onDragOver, onFileDrop }) => {
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

  const renderDomain = (domain: Domain, level: number = 0) => {
    const isExpanded = expandedDomains.has(domain.id);
    const isSelected = selectedDomain === domain.id;
    const hasChildren = domain.children && domain.children.length > 0;
    const isDragOver = dragOverDomain === domain.id;

    return (
      <Box key={domain.id}>
        <ListItem
          button
          selected={isSelected}
          onClick={() => onSelect(domain)}
          onDragOver={(e) => handleDragOver(e, domain.id)}
          onDragLeave={handleDragLeave}
          onDrop={(e) => handleDrop(e, domain)}
          sx={{
            pl: 2 + level * 2,
            borderRadius: 1,
            mb: 0.5,
            bgcolor: isDragOver
              ? alpha('#10B981', 0.3)
              : isSelected
                ? alpha('#6366F1', 0.15)
                : 'transparent',
            border: isDragOver ? '2px dashed #10B981' : '2px solid transparent',
            transition: 'all 0.2s ease',
            '&:hover': { bgcolor: alpha('#6366F1', 0.1) },
          }}
        >
          {hasChildren && (
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                onToggle(domain.id);
              }}
              sx={{ mr: 1 }}
            >
              {isExpanded ? <ExpandIcon /> : <ChevronRightIcon />}
            </IconButton>
          )}
          <ListItemIcon sx={{ minWidth: 36 }}>
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
            secondary={isDragOver ? 'Release to upload' : `${domain.document_count} docs`}
            primaryTypographyProps={{
              sx: { color: isDragOver ? '#10B981' : 'inherit', fontWeight: isDragOver ? 600 : 400 }
            }}
          />
          {!isDragOver && (
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
        </ListItem>
        {hasChildren && isExpanded && (
          <Box>
            {domain.children!.map(child => renderDomain(child, level + 1))}
          </Box>
        )}
      </Box>
    );
  };

  return (
    <List sx={{ py: 0 }}>
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
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (chunks.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography color="text.secondary">No chunks available</Typography>
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
      </Paper>
    </Box>
  );
};

// Main Component
const DocumentsPage: React.FC = () => {
  const theme = useTheme();
  const [domains, setDomains] = useState<Domain[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDomain, setSelectedDomain] = useState<Domain | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
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
  const [globalDragOver, setGlobalDragOver] = useState(false);

  // Processing queue
  const [processingQueue, setProcessingQueue] = useState<any>(null);

  // Dialogs
  const [domainDialogOpen, setDomainDialogOpen] = useState(false);
  const [documentDialogOpen, setDocumentDialogOpen] = useState(false);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [processDialogOpen, setProcessDialogOpen] = useState(false);
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [batchIngestionDialogOpen, setBatchIngestionDialogOpen] = useState(false);

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
  const [uploadDestination, setUploadDestination] = useState<'local' | 'graphrag'>('local');
  const [graphragUploading, setGraphragUploading] = useState(false);
  const [graphragResult, setGraphragResult] = useState<any>(null);

  // Load domains
  const loadDomains = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/rag/domains');
      const flatDomains = res.data.domains || [];

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
      setDocuments(res.data.documents || []);
      setTotalDocuments(res.data.total || 0);
    } catch (err) {
      console.error('Failed to load documents:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedDomain, searchTerm, statusFilter, typeFilter, page, rowsPerPage]);

  // Load chunks for selected document
  const loadChunks = useCallback(async (documentId: string) => {
    setChunksLoading(true);
    try {
      const res = await api.get(`/api/v1/rag/documents/${documentId}/chunks`);
      setChunks(res.data.chunks || []);
    } catch (err) {
      console.error('Failed to load chunks:', err);
    } finally {
      setChunksLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDomains();
  }, [loadDomains]);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  useEffect(() => {
    if (selectedDocument) {
      loadChunks(selectedDocument.id);
    }
  }, [selectedDocument, loadChunks]);

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
      });
      loadDomains();
    } catch (err) {
      setError('Failed to create domain');
    }
  };

  const handleDeleteDomain = async (domain: Domain) => {
    if (!confirm(`Delete domain "${domain.name}"? This will also delete all documents.`)) return;
    try {
      await api.delete(`/api/v1/rag/domains/${domain.id}`, { params: { cascade: true } });
      setSelectedDomain(null);
      loadDomains();
    } catch (err) {
      setError('Failed to delete domain');
    }
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);

    // Auto-detect document type from file extension
    const ext = file.name.split('.').pop()?.toLowerCase();
    let docType = 'txt';
    if (ext === 'pdf') docType = 'pdf';
    else if (ext === 'docx' || ext === 'doc') docType = 'docx';
    else if (ext === 'md') docType = 'md';
    else if (ext === 'html') docType = 'html';
    else if (ext === 'json') docType = 'json';
    else if (ext === 'csv') docType = 'csv';
    else if (ext === 'epub') docType = 'epub';

    // Read file content - use base64 for binary files (PDF, DOCX)
    try {
      let content: string;
      if (docType === 'pdf' || docType === 'docx' || docType === 'epub') {
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
        const ext = file.name.split('.').pop()?.toLowerCase();
        let docType = 'txt';
        if (ext === 'pdf') docType = 'pdf';
        else if (ext === 'docx' || ext === 'doc') docType = 'docx';
        else if (ext === 'md') docType = 'md';
        else if (ext === 'html' || ext === 'htm') docType = 'html';
        else if (ext === 'json') docType = 'json';
        else if (ext === 'csv') docType = 'csv';
        else if (ext === 'epub') docType = 'epub';

        // Read file content
        let content: string;
        if (docType === 'pdf' || docType === 'docx' || docType === 'epub') {
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
    if (uploadDestination === 'graphrag' && selectedFile) {
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
      setUploadMode('paste');
      loadDocuments();
      loadDomains();
    } catch (err) {
      setError('Failed to create document');
    }
  };

  const handleGraphRAGUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload');
      return;
    }

    try {
      setGraphragUploading(true);
      setGraphragResult(null);

      const result = await api.uploadToGraphRAG(selectedFile, {
        domain: selectedDomain?.name || 'general',
        useSemanticChunking: true,
        buildKnowledgeGraph: true
      });

      setGraphragResult(result);
      setSuccess(`Document processed: ${result.total_chunks} chunks, ${result.knowledge_graph?.entities || 0} entities extracted`);
      setTimeout(() => setSuccess(null), 5000);

      // Don't close dialog immediately - show results
    } catch (err: any) {
      setError(err.message || 'Failed to upload to GraphRAG');
    } finally {
      setGraphragUploading(false);
    }
  };

  const handleDeleteDocument = async (doc: Document) => {
    if (!confirm(`Delete document "${doc.name}"?`)) return;
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

  const handleReprocessDocument = async (docId: string) => {
    try {
      const response = await fetch(`/api/v1/rag/documents/${docId}/reprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chunking_strategy: 'semantic',
          embedding_model: selectedDomain?.embedding_model || 'all-MiniLM-L6-v2'
        })
      });

      if (!response.ok) throw new Error('Reprocessing failed');

      const result = await response.json();
      setSuccess(result.message);
      setTimeout(() => setSuccess(null), 3000);
      loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reprocess document');
    }
  };

  const handleRemoveFromVectorStore = async (docId: string) => {
    if (!confirm('Remove this document from the vector store? The document will remain in the database.')) return;

    try {
      const response = await fetch(`/api/v1/rag/documents/${docId}/vectors`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error('Failed to remove from vector store');

      const result = await response.json();
      setSuccess(`Removed ${result.vectors_deleted} vectors from vector store`);
      setTimeout(() => setSuccess(null), 3000);
      loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove from vector store');
    }
  };

  const handleReindexDomain = async () => {
    if (!selectedDomain) return;
    if (!confirm(`Reindex entire domain "${selectedDomain.name}"? This will reprocess all documents.`)) return;

    try {
      setBatchProcessing(true);
      const response = await fetch(`/api/v1/rag/domains/${selectedDomain.id}/reindex`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chunking_strategy: 'semantic',
          embedding_model: selectedDomain.embedding_model,
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
  };

  const handleClearCollection = async () => {
    if (!selectedDomain) return;
    if (!confirm(`Clear all vectors from "${selectedDomain.name}"? This deletes all embeddings but keeps documents. You'll need to reindex afterwards.`)) return;

    try {
      setBatchProcessing(true);
      const response = await fetch(`/api/v1/rag/domains/${selectedDomain.id}/collection`, {
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
    const interval = setInterval(loadProcessingQueue, 5000);
    loadProcessingQueue();
    return () => clearInterval(interval);
  }, [loadProcessingQueue]);

  const handleProcessDocument = async () => {
    if (!selectedDocument) return;
    try {
      setLoading(true);
      await api.post(`/api/v1/rag/documents/${selectedDocument.id}/process`, processForm);
      setProcessDialogOpen(false);
      loadDocuments();
      loadChunks(selectedDocument.id);
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

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" sx={{
            fontWeight: 700,
            mb: 0.5,
            background: 'linear-gradient(135deg, #10B981 0%, #06B6D4 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Document Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Organize, process, and manage your knowledge base documents
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            startIcon={<FolderIcon />}
            onClick={() => setDomainDialogOpen(true)}
            variant="outlined"
          >
            New Domain
          </Button>
          <Button
            startIcon={<UploadIcon />}
            onClick={() => setDocumentDialogOpen(true)}
            variant="contained"
            disabled={!selectedDomain}
            sx={{ background: 'linear-gradient(135deg, #10B981 0%, #06B6D4 100%)' }}
          >
            Add Document
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      {uploadingFiles && (
        <Alert
          severity="info"
          sx={{ mb: 2 }}
          icon={<CircularProgress size={20} />}
        >
          Uploading files... Please wait.
        </Alert>
      )}

      {/* Global Drop Zone */}
      <Box
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!selectedDomain) return;
          setGlobalDragOver(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setGlobalDragOver(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setGlobalDragOver(false);
          if (!selectedDomain) return;
          const files = e.dataTransfer.files;
          if (files.length > 0) {
            handleFileDrop(selectedDomain, files);
          }
        }}
        sx={{
          mb: 2,
          p: 3,
          borderRadius: 2,
          border: '2px dashed',
          borderColor: globalDragOver
            ? '#10B981'
            : selectedDomain
              ? alpha('#10B981', 0.3)
              : alpha('#fff', 0.1),
          bgcolor: globalDragOver
            ? alpha('#10B981', 0.08)
            : 'transparent',
          transition: 'all 0.2s ease',
          cursor: selectedDomain ? 'pointer' : 'default',
          opacity: selectedDomain ? 1 : 0.5,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 1,
        }}
        onClick={() => {
          if (selectedDomain) {
            setDocumentDialogOpen(true);
          }
        }}
      >
        <CloudUploadIcon sx={{
          fontSize: 40,
          color: globalDragOver ? '#10B981' : selectedDomain ? alpha('#10B981', 0.6) : 'text.disabled'
        }} />
        <Typography
          variant="body1"
          sx={{
            fontWeight: 600,
            color: globalDragOver ? '#10B981' : selectedDomain ? 'text.primary' : 'text.disabled'
          }}
        >
          {globalDragOver
            ? `Drop files to upload to "${selectedDomain.name}"`
            : selectedDomain
              ? `Drag files here to upload to "${selectedDomain.name}"`
              : 'Select a domain to upload documents'}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Supports TXT, PDF, DOCX, MD, HTML, JSON, CSV, EPUB
        </Typography>
        {selectedDomain && !globalDragOver && (
          <Button
            size="small"
            startIcon={<UploadIcon />}
            sx={{ mt: 1, textTransform: 'none' }}
          >
            Or click to browse files
          </Button>
        )}
      </Box>

      <Grid container spacing={3}>
        {/* Domain Tree Sidebar */}
        <Grid item xs={3}>
          <Paper sx={{
            p: 2,
            height: 'calc(100vh - 200px)',
            overflow: 'auto',
            background: 'linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(15, 15, 26, 0.9) 100%)',
            border: '1px solid rgba(16, 185, 129, 0.2)',
          }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                <FolderIcon sx={{ mr: 1, verticalAlign: 'middle', color: '#F59E0B' }} />
                Domains
              </Typography>
              <IconButton size="small" onClick={loadDomains}>
                <RefreshIcon fontSize="small" />
              </IconButton>
            </Box>

            <Button
              fullWidth
              variant="outlined"
              size="small"
              onClick={() => {
                setSelectedDomain(null);
                setPage(0);
              }}
              sx={{ mb: 2, borderStyle: 'dashed' }}
            >
              All Documents
            </Button>

            {domains.length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1, textAlign: 'center' }}>
                Drag & drop files onto a domain to upload
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
                onEdit={() => { }}
                onDelete={handleDeleteDomain}
                onDragOver={setDragOverDomain}
                onFileDrop={handleFileDrop}
              />
            )}
          </Paper>
        </Grid>

        {/* Document List */}
        <Grid item xs={selectedDocument ? 5 : 9}>
          <Paper sx={{
            p: 2,
            background: 'linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(15, 15, 26, 0.9) 100%)',
            border: '1px solid rgba(16, 185, 129, 0.2)',
          }}>
            {/* Filters */}
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <TextField
                size="small"
                placeholder="Search documents..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.disabled' }} />,
                }}
                sx={{ flex: 1 }}
              />
              <FormControl size="small" sx={{ minWidth: 120 }}>
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
              <FormControl size="small" sx={{ minWidth: 120 }}>
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

            {/* Quick Actions */}
            <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                size="small"
                color="primary"
                startIcon={<UploadIcon />}
                onClick={() => setBatchIngestionDialogOpen(true)}
              >
                Batch Upload
              </Button>
              <Button
                variant="outlined"
                size="small"
                startIcon={<ProcessIcon />}
                onClick={handleProcessAllPending}
                disabled={batchProcessing}
              >
                Process All Pending
              </Button>
              <Button
                variant="outlined"
                size="small"
                startIcon={<SyncIcon />}
                onClick={handleReindexDomain}
                disabled={!selectedDomain || batchProcessing}
              >
                Reindex Domain
              </Button>
              <Button
                variant="outlined"
                size="small"
                color="warning"
                startIcon={<StorageIcon />}
                onClick={handleClearCollection}
                disabled={!selectedDomain || batchProcessing}
              >
                Clear Vectors
              </Button>
              <Button
                variant="outlined"
                size="small"
                startIcon={selectedDocuments.size === documents.length ? <DeselectIcon /> : <SelectAllIcon />}
                onClick={selectedDocuments.size === documents.length ? clearSelection : selectAllDocuments}
              >
                {selectedDocuments.size === documents.length ? 'Deselect All' : 'Select All'}
              </Button>
            </Box>

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

            {/* Document Table */}
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell padding="checkbox">
                      <IconButton
                        size="small"
                        onClick={selectedDocuments.size === documents.length ? clearSelection : selectAllDocuments}
                      >
                        {selectedDocuments.size === documents.length ? <CheckBoxIcon /> : <CheckBoxOutlineBlankIcon />}
                      </IconButton>
                    </TableCell>
                    <TableCell>Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell align="right">Chunks</TableCell>
                    <TableCell align="right">Created</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {loading ? (
                    <TableRow>
                      <TableCell colSpan={7} sx={{ textAlign: 'center', py: 4 }}>
                        <CircularProgress />
                      </TableCell>
                    </TableRow>
                  ) : documents.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7} sx={{ textAlign: 'center', py: 4 }}>
                        <DocIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
                        <Typography variant="body2" color="text.secondary">
                          No documents found
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    documents.map(doc => {
                      const statusConfig = STATUS_CONFIG[doc.status] || STATUS_CONFIG.pending;
                      const isSelected = selectedDocuments.has(doc.id);
                      return (
                        <TableRow
                          key={doc.id}
                          hover
                          selected={selectedDocument?.id === doc.id}
                          sx={{ cursor: 'pointer' }}
                        >
                          <TableCell padding="checkbox" onClick={(e) => e.stopPropagation()}>
                            <IconButton
                              size="small"
                              onClick={() => toggleDocumentSelection(doc.id)}
                            >
                              {isSelected ? <CheckBoxIcon color="primary" /> : <CheckBoxOutlineBlankIcon />}
                            </IconButton>
                          </TableCell>
                          <TableCell onClick={() => setSelectedDocument(doc)}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              {DOC_TYPE_ICONS[doc.doc_type] || <DocIcon />}
                              <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                                {doc.name}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={doc.doc_type.toUpperCase()}
                              size="small"
                              sx={{ height: 20, fontSize: 10 }}
                            />
                          </TableCell>
                          <TableCell>
                            <Chip
                              icon={statusConfig.icon as React.ReactElement}
                              label={statusConfig.label}
                              size="small"
                              sx={{
                                height: 24,
                                bgcolor: alpha(statusConfig.color, 0.15),
                                color: statusConfig.color,
                                '& .MuiChip-icon': { color: statusConfig.color },
                              }}
                            />
                          </TableCell>
                          <TableCell align="right" onClick={() => setSelectedDocument(doc)}>{doc.chunk_count}</TableCell>
                          <TableCell align="right" onClick={() => setSelectedDocument(doc)}>
                            {new Date(doc.created_at).toLocaleDateString()}
                          </TableCell>
                          <TableCell align="right" onClick={(e) => e.stopPropagation()}>
                            <Box sx={{ display: 'flex', gap: 0.5, justifyContent: 'flex-end' }}>
                              <Tooltip title="View Details">
                                <IconButton size="small" onClick={() => setSelectedDocument(doc)}>
                                  <ViewIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                              {doc.status === 'pending' && (
                                <Tooltip title="Process Now">
                                  <IconButton
                                    size="small"
                                    color="primary"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleReprocessDocument(doc.id);
                                    }}
                                  >
                                    <ProcessIcon fontSize="small" />
                                  </IconButton>
                                </Tooltip>
                              )}
                              {doc.status === 'ready' && (
                                <>
                                  <Tooltip title="Extract to Knowledge Graph">
                                    <IconButton
                                      size="small"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleExtractKnowledge(doc.id);
                                      }}
                                      sx={{ color: '#6366F1' }}
                                    >
                                      <KnowledgeGraphIcon fontSize="small" />
                                    </IconButton>
                                  </Tooltip>
                                  <Tooltip title="Reprocess">
                                    <IconButton
                                      size="small"
                                      color="warning"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleReprocessDocument(doc.id);
                                      }}
                                    >
                                      <ReprocessIcon fontSize="small" />
                                    </IconButton>
                                  </Tooltip>
                                  <Tooltip title="Remove from Vector Store">
                                    <IconButton
                                      size="small"
                                      color="secondary"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleRemoveFromVectorStore(doc.id);
                                      }}
                                    >
                                      <RemoveIcon fontSize="small" />
                                    </IconButton>
                                  </Tooltip>
                                </>
                              )}
                              {doc.status === 'error' && (
                                <Tooltip title="Retry Processing">
                                  <IconButton
                                    size="small"
                                    color="primary"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleReprocessDocument(doc.id);
                                    }}
                                  >
                                    <RefreshIcon fontSize="small" />
                                  </IconButton>
                                </Tooltip>
                              )}
                              <Tooltip title="Delete">
                                <IconButton
                                  size="small"
                                  color="error"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleDeleteDocument(doc);
                                  }}
                                >
                                  <DeleteIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </TableCell>
                        </TableRow>
                      );
                    })
                  )}
                </TableBody>
              </Table>
            </TableContainer>

            <TablePagination
              component="div"
              count={totalDocuments}
              page={page}
              onPageChange={(_, newPage) => setPage(newPage)}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={(e) => {
                setRowsPerPage(parseInt(e.target.value, 10));
                setPage(0);
              }}
            />
          </Paper>
        </Grid>

        {/* Document Details Panel */}
        {selectedDocument && (
          <Grid item xs={4}>
            <Paper sx={{
              p: 2,
              height: 'calc(100vh - 200px)',
              overflow: 'auto',
              background: 'linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(15, 15, 26, 0.9) 100%)',
              border: '1px solid rgba(16, 185, 129, 0.2)',
            }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                    {selectedDocument.name}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Chip
                      label={selectedDocument.doc_type.toUpperCase()}
                      size="small"
                      sx={{ bgcolor: alpha('#6366F1', 0.2) }}
                    />
                    <Chip
                      label={STATUS_CONFIG[selectedDocument.status]?.label || selectedDocument.status}
                      size="small"
                      sx={{
                        bgcolor: alpha(STATUS_CONFIG[selectedDocument.status]?.color || '#6B7280', 0.2),
                        color: STATUS_CONFIG[selectedDocument.status]?.color || '#6B7280',
                      }}
                    />
                  </Box>
                </Box>
                <IconButton size="small" onClick={() => setSelectedDocument(null)}>
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Stats */}
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6}>
                  <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: alpha('#10B981', 0.1) }}>
                    <Typography variant="caption" color="text.secondary">Chunks</Typography>
                    <Typography variant="h6">{selectedDocument.chunk_count}</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ p: 1.5, borderRadius: 1, bgcolor: alpha('#3B82F6', 0.1) }}>
                    <Typography variant="caption" color="text.secondary">Tokens</Typography>
                    <Typography variant="h6">{selectedDocument.token_count}</Typography>
                  </Box>
                </Grid>
              </Grid>

              {/* Actions */}
              <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<ProcessIcon />}
                  onClick={() => setProcessDialogOpen(true)}
                  disabled={selectedDocument.status === 'processing'}
                >
                  Process
                </Button>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<ViewIcon />}
                  onClick={() => setViewDialogOpen(true)}
                >
                  View Content
                </Button>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Chunks Preview */}
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Chunks ({chunks.length})
              </Typography>
              <ChunkViewer chunks={chunks} loading={chunksLoading} />
            </Paper>
          </Grid>
        )}
      </Grid>

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
                onClick={() => setUploadDestination('local')}
                color={uploadDestination === 'local' ? 'primary' : 'default'}
                variant={uploadDestination === 'local' ? 'filled' : 'outlined'}
                icon={<StorageIcon />}
                clickable
              />
              <Chip
                label="GraphRAG (Neo4j + Knowledge Graph)"
                onClick={() => setUploadDestination('graphrag')}
                color={uploadDestination === 'graphrag' ? 'success' : 'default'}
                variant={uploadDestination === 'graphrag' ? 'filled' : 'outlined'}
                icon={<KnowledgeGraphIcon />}
                clickable
              />
            </Box>
          </Box>

          {uploadDestination === 'graphrag' && (
            <Alert severity="info" sx={{ mb: 2 }}>
              Documents will be processed through GraphRAG with automatic entity extraction and knowledge graph building.
              Only file upload is supported for GraphRAG.
            </Alert>
          )}

          {graphragResult && (
            <Alert severity="success" sx={{ mb: 2 }}>
              <Typography variant="body2" fontWeight={600}>
                GraphRAG Processing Complete
              </Typography>
              <Typography variant="caption" component="div">
                Chunks: {graphragResult.total_chunks}
              </Typography>
              {graphragResult.knowledge_graph && (
                <>
                  <Typography variant="caption" component="div">
                    Entities: {graphragResult.knowledge_graph.entities}
                  </Typography>
                  <Typography variant="caption" component="div">
                    Relationships: {graphragResult.knowledge_graph.relationships}
                  </Typography>
                </>
              )}
            </Alert>
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
                {selectedFile ? (
                  <Box sx={{ textAlign: 'center' }}>
                    <DocIcon sx={{ fontSize: 40, mb: 1 }} />
                    <Typography>{selectedFile.name}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {(selectedFile.size / 1024).toFixed(2)} KB
                    </Typography>
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center' }}>
                    <CloudUploadIcon sx={{ fontSize: 40, mb: 1 }} />
                    <Typography>Click to upload or drag and drop</Typography>
                    <Typography variant="caption" color="text.secondary">
                      TXT, PDF, DOCX, MD, HTML, JSON, CSV
                    </Typography>
                  </Box>
                )}
                <input
                  type="file"
                  hidden
                  accept=".txt,.pdf,.docx,.doc,.md,.html,.json,.csv"
                  onChange={handleFileSelect}
                />
              </Button>
              {selectedFile && (
                <Button
                  size="small"
                  onClick={() => {
                    setSelectedFile(null);
                    setDocumentForm(prev => ({ ...prev, content: '' }));
                  }}
                  sx={{ mt: 1 }}
                >
                  Clear File
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
                (uploadDestination === 'graphrag' && !selectedFile) ||
                (uploadDestination === 'local' && !documentForm.content && uploadMode !== 'url')
              }
              startIcon={graphragUploading ? <CircularProgress size={16} /> : null}
            >
              {graphragUploading ? 'Processing...' : uploadDestination === 'graphrag' ? 'Upload to GraphRAG' : 'Add Document'}
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
    </Box>
  );
};

export default DocumentsPage;
