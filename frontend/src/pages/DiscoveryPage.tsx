import { useState, useEffect, useCallback, type KeyboardEvent } from 'react';
import {
  Box,
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
  Alert,
  CircularProgress,
  LinearProgress,
  Checkbox,
  FormControlLabel,
  Skeleton,
  Tooltip,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  Close as CloseIcon,
  Visibility as ViewIcon,
  ThumbUp as ApproveIcon,
  ThumbDown as RejectIcon,
  CloudDownload as DownloadIcon,
  Queue as QueueIcon,
  TravelExplore as WebSearchIcon,
  Link as LinkIcon,
} from '@mui/icons-material';
import {
  PageShell,
  SectionCard,
  StatusChip,
  type StatusChipConfigMap,
} from '../components/common';
import { apiService as api } from '../services/api';
import { accentColors, gradients } from '../utils/theme';
import { debugLog } from '../utils/debug';

// Types
interface DiscoveredDocument {
  id: string;
  title: string;
  url: string;
  snippet: string;
  content: string;
  status: string;
  relevance_score: number;
  quality_score: number;
  source: string;
  search_query: string;
  domain_suggestion: string | null;
  word_count: number;
  discovered_at: string;
  reviewed_at: string | null;
}

interface Domain {
  id: string;
  name: string;
}

interface DiscoveryStats {
  total: number;
  by_status: Record<string, number>;
  by_source: Record<string, number>;
  avg_relevance: number;
  avg_quality: number;
}

const isRecord = (value: unknown): value is Record<string, any> => (
  typeof value === 'object' && value !== null && !Array.isArray(value)
);

const toNumber = (value: unknown, fallback = 0): number => (
  typeof value === 'number' && Number.isFinite(value) ? value : fallback
);

const normalizeDomainList = (value: unknown): Domain[] => (
  Array.isArray(value) ? value.filter(isRecord) as Domain[] : []
);

const normalizeDiscoveryDocuments = (value: unknown): DiscoveredDocument[] => (
  Array.isArray(value) ? value.filter(isRecord) as DiscoveredDocument[] : []
);

const normalizeDiscoveryStats = (value: unknown): DiscoveryStats | null => {
  if (!isRecord(value)) return null;

  return {
    total: toNumber(value.total),
    by_status: isRecord(value.by_status) ? value.by_status as Record<string, number> : {},
    by_source: isRecord(value.by_source) ? value.by_source as Record<string, number> : {},
    avg_relevance: toNumber(value.avg_relevance),
    avg_quality: toNumber(value.avg_quality),
  };
};

const STATUS_CONFIG: StatusChipConfigMap = {
  all: { color: accentColors.info, label: 'All statuses' },
  approved: { color: accentColors.success, label: 'Approved' },
  discovery: { color: accentColors.purple, label: 'Document discovery' },
  domain: { color: accentColors.purple, label: 'Domain' },
  error: { color: accentColors.slate, label: 'Error' },
  pending: { color: accentColors.warning, label: 'Pending Review' },
  processing: { color: accentColors.info, label: 'Processing' },
  query: { color: accentColors.info, label: 'Query' },
  rejected: { color: accentColors.rose, label: 'Rejected' },
  source: { color: accentColors.slate, label: 'Source' },
};

const REVIEW_STATUS_KEYS = ['pending', 'processing', 'approved', 'rejected', 'error'] as const;
const REVIEW_STATUS_OPTIONS = REVIEW_STATUS_KEYS.map((status) => [status, STATUS_CONFIG[status]] as const);

const getDiscoveryStatusConfig = (status: string) => {
  const normalizedStatus = status.toLowerCase();
  const config = STATUS_CONFIG[normalizedStatus] ?? {};

  return {
    color: config.color ?? accentColors.slate,
    label: config.label ?? status,
  };
};

const getScoreColor = (score: number) => {
  if (score > 0.7) return accentColors.success;
  if (score > 0.4) return accentColors.warning;
  return accentColors.rose;
};

const discoveryPanelSx = {
  borderRadius: '8px',
  bgcolor: 'background.paper',
  border: '1px solid',
  borderColor: 'divider',
} as const;

// Document Card Component
const DiscoveredDocumentCard = ({
  doc,
  selected,
  onSelect,
  onApprove,
  onReject,
  onExtract,
  onView,
}: {
  doc: DiscoveredDocument;
  selected: boolean;
  onSelect: () => void;
  onApprove: () => void;
  onReject: () => void;
  onExtract: () => void;
  onView: () => void;
}) => {
  const relevanceColor = getScoreColor(doc.relevance_score);
  const qualityColor = getScoreColor(doc.quality_score);

  return (
    <Box
      sx={{
        mb: 2,
        ...discoveryPanelSx,
        borderWidth: selected ? 2 : 1,
        borderColor: selected ? accentColors.primary : 'divider',
        bgcolor: selected ? alpha(accentColors.primary, 0.05) : 'transparent',
        transition: 'all 0.2s',
        '&:hover': {
          borderColor: alpha(accentColors.primary, 0.5),
          transform: 'translateY(-2px)',
          boxShadow: `0 8px 32px ${alpha(accentColors.primary, 0.15)}`,
        },
      }}
    >
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Checkbox
            checked={selected}
            onChange={onSelect}
            sx={{ mt: -0.5 }}
          />
          <Box sx={{ flex: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, flex: 1 }}>
                {doc.title || 'Untitled'}
              </Typography>
              <StatusChip
                status={doc.status}
                statusConfig={STATUS_CONFIG}
                sx={{ ml: 1, flexShrink: 0 }}
              />
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <LinkIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
              <Typography
                variant="caption"
                color="text.secondary"
                component="a"
                href={doc.url}
                target="_blank"
                sx={{
                  textDecoration: 'none',
                  '&:hover': { textDecoration: 'underline' },
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  maxWidth: 400,
                }}
              >
                {doc.url}
              </Typography>
            </Box>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {doc.snippet || 'No preview available'}
            </Typography>

            {/* Scores */}
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="caption" color="text.secondary">Relevance:</Typography>
                  <LinearProgress
                    variant="determinate"
                    value={doc.relevance_score * 100}
                    sx={{
                      flex: 1,
                      height: 6,
                      borderRadius: '3px',
                      bgcolor: alpha(relevanceColor, 0.1),
                      '& .MuiLinearProgress-bar': {
                        borderRadius: '3px',
                        bgcolor: relevanceColor,
                      },
                    }}
                  />
                  <Typography variant="caption" sx={{ minWidth: 35 }}>
                    {Math.round(doc.relevance_score * 100)}%
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="caption" color="text.secondary">Quality:</Typography>
                  <LinearProgress
                    variant="determinate"
                    value={doc.quality_score * 100}
                    sx={{
                      flex: 1,
                      height: 6,
                      borderRadius: '3px',
                      bgcolor: alpha(qualityColor, 0.1),
                      '& .MuiLinearProgress-bar': {
                        borderRadius: '3px',
                        bgcolor: qualityColor,
                      },
                    }}
                  />
                  <Typography variant="caption" sx={{ minWidth: 35 }}>
                    {Math.round(doc.quality_score * 100)}%
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="caption" color="text.secondary">Words:</Typography>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>
                    {doc.word_count.toLocaleString()}
                  </Typography>
                </Box>
              </Grid>
            </Grid>

            {/* Meta info */}
            <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
              <StatusChip
                status="source"
                label={doc.source || 'Unknown'}
                variant="outlined"
                statusConfig={STATUS_CONFIG}
                height={20}
                sx={{ fontSize: 10 }}
              />
              {doc.search_query && (
                <StatusChip
                  status="query"
                  label={`Query: ${doc.search_query}`}
                  variant="outlined"
                  statusConfig={STATUS_CONFIG}
                  height={20}
                  sx={{ fontSize: 10 }}
                />
              )}
            </Box>
          </Box>
        </Box>
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, px: 2, pb: 2, pt: 0, flexWrap: 'wrap' }}>
        {doc.status === 'pending' && !doc.content && (
          <Button
            size="small"
            startIcon={<DownloadIcon />}
            onClick={onExtract}
            variant="outlined"
          >
            Extract Content
          </Button>
        )}
        <Button
          size="small"
          startIcon={<ViewIcon />}
          onClick={onView}
        >
          View
        </Button>
        {doc.status === 'pending' && (
          <>
            <Button
              size="small"
              startIcon={<RejectIcon />}
              onClick={onReject}
              color="error"
            >
              Reject
            </Button>
            <Button
              size="small"
              startIcon={<ApproveIcon />}
              onClick={onApprove}
              variant="contained"
              sx={{ background: gradients.success }}
            >
              Approve
            </Button>
          </>
        )}
      </Box>
    </Box>
  );
};

const formatNumber = (value: number): string => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
  return value.toString();
};

// Main Component
const DiscoveryPage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<DiscoveredDocument[]>([]);
  const [queue, setQueue] = useState<DiscoveredDocument[]>([]);
  const [queueTotal, setQueueTotal] = useState(0);
  const [queueLoading, setQueueLoading] = useState(false);
  const [stats, setStats] = useState<DiscoveryStats | null>(null);
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
  const [domains, setDomains] = useState<Domain[]>([]);
  const [selectedDomain, setSelectedDomain] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState('pending');
  const [error, setError] = useState<string | null>(null);

  // Dialogs
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [viewingDoc, setViewingDoc] = useState<DiscoveredDocument | null>(null);
  const [approveDialogOpen, setApproveDialogOpen] = useState(false);
  const [approvingDoc, setApprovingDoc] = useState<DiscoveredDocument | null>(null);

  // Load domains
  useEffect(() => {
    api.get('/api/v1/rag/domains').then(res => {
      const data = isRecord(res.data) ? res.data : {};
      setDomains(normalizeDomainList(data.domains));
    }).catch((err) => {
      debugLog('discovery', 'Failed to load domains', err);
    });
  }, []);

  // Load queue
  const loadQueue = useCallback(async () => {
    setQueueLoading(true);
    try {
      const res = await api.get('/api/v1/rag/discover/queue', {
        params: { status: statusFilter || undefined, limit: 50 }
      });
      const data = isRecord(res.data) ? res.data : {};
      const documents = normalizeDiscoveryDocuments(data.documents ?? res.data);
      setQueue(documents);
      setQueueTotal(toNumber(data.total, documents.length));
    } catch (err) {
      debugLog('discovery', 'Failed to load queue', err);
      setQueue([]);
      setQueueTotal(0);
    } finally {
      setQueueLoading(false);
    }
  }, [statusFilter]);

  // Load stats
  const loadStats = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/rag/discover/statistics');
      setStats(normalizeDiscoveryStats(res.data));
    } catch (err) {
      debugLog('discovery', 'Failed to load stats', err);
      setStats(null);
    }
  }, []);

  useEffect(() => {
    loadQueue();
    loadStats();
  }, [loadQueue, loadStats]);

  // Web search
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const res = await api.post('/api/v1/rag/discover/search', {
        query: searchQuery,
        max_results: 10,
        provider: 'duckduckgo',
      });
      setSearchResults(res.data.results || []);
      loadStats();
    } catch (err) {
      setError('Search failed. Please try again.');
    } finally {
      setSearching(false);
    }
  };

  // Extract content
  const handleExtract = async (docId: string) => {
    try {
      await api.post(`/api/v1/rag/discover/${docId}/extract`);
      loadQueue();
      loadStats();
    } catch (err) {
      setError('Failed to extract content');
    }
  };

  // Approve document
  const handleApprove = async (docId: string, domainId?: string) => {
    try {
      await api.post(`/api/v1/rag/discover/${docId}/approve`, { domain_id: domainId });
      loadQueue();
      loadStats();
      setApproveDialogOpen(false);
    } catch (err) {
      setError('Failed to approve document');
    }
  };

  // Reject document
  const handleReject = async (docId: string) => {
    try {
      await api.post(`/api/v1/rag/discover/${docId}/reject`);
      loadQueue();
      loadStats();
    } catch (err) {
      setError('Failed to reject document');
    }
  };

  // Bulk approve
  const handleBulkApprove = async () => {
    if (selectedDocs.size === 0) return;
    try {
      await api.post('/api/v1/rag/discover/bulk-approve', {
        doc_ids: Array.from(selectedDocs),
        domain_id: selectedDomain || undefined,
      });
      setSelectedDocs(new Set());
      loadQueue();
      loadStats();
    } catch (err) {
      setError('Failed to bulk approve');
    }
  };

  // Toggle selection
  const toggleSelection = (docId: string) => {
    setSelectedDocs(prev => {
      const next = new Set(prev);
      if (next.has(docId)) {
        next.delete(docId);
      } else {
        next.add(docId);
      }
      return next;
    });
  };

  // Select all
  const selectAll = () => {
    if (selectedDocs.size === queue.length) {
      setSelectedDocs(new Set());
    } else {
      setSelectedDocs(new Set(queue.map(d => d.id)));
    }
  };

  const pendingCount = stats?.by_status?.pending || 0;
  const approvedCount = stats?.by_status?.approved || 0;
  const rejectedCount = stats?.by_status?.rejected || 0;
  const selectedDomainName = domains.find(domain => domain.id === selectedDomain)?.name || 'No target selected';
  const activeStatusConfig = statusFilter
    ? getDiscoveryStatusConfig(statusFilter)
    : { color: accentColors.info, label: 'All statuses' };
  const activeStatusLabel = activeStatusConfig.label;
  const averageRelevance = stats?.avg_relevance || 0;
  const topSource = stats?.by_source
    ? Object.entries(stats.by_source).sort((a, b) => b[1] - a[1])[0]
    : null;
  const hasDiscoverySummary = Boolean(
    selectedDomain ||
    topSource ||
    queueTotal > 0 ||
    approvedCount > 0 ||
    rejectedCount > 0 ||
    averageRelevance > 0
  );
  const showQueueFilters = queueTotal > 0 || searchResults.length > 0 || selectedDocs.size > 0 || Boolean(selectedDomain);
  const showBulkApproveAction = selectedDocs.size > 0;
  const showSearchAction = searching || searchQuery.trim().length > 0;

  const handleSearchKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };
  const discoveryHeaderActions = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      <TextField
        size="small"
        placeholder="Search the web..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        onKeyDown={handleSearchKeyDown}
        InputProps={{
          startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.disabled', fontSize: 18 }} />,
        }}
        sx={{ width: { xs: '100%', sm: 250, lg: 340 }, '& .MuiInputBase-root': { height: 34 } }}
      />

      {showQueueFilters && (
        <>
          <FormControl size="small" sx={{ width: { xs: '100%', sm: 160 } }}>
            <InputLabel>Status</InputLabel>
            <Select
              value={statusFilter}
              label="Status"
              onChange={(e) => setStatusFilter(e.target.value)}
              sx={{ height: 34 }}
            >
              <MenuItem value="">All</MenuItem>
              {REVIEW_STATUS_OPTIONS.map(([key, config]) => (
                <MenuItem key={key} value={key}>{config.label}</MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ width: { xs: '100%', sm: 180 } }}>
            <InputLabel>Domain</InputLabel>
            <Select
              value={selectedDomain}
              label="Domain"
              onChange={(e) => setSelectedDomain(e.target.value)}
              sx={{ height: 34 }}
            >
              <MenuItem value="">No target</MenuItem>
              {domains.map(domain => (
                <MenuItem key={domain.id} value={domain.id}>{domain.name}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </>
      )}

      <Tooltip title="Refresh queue">
        <IconButton
          aria-label="Refresh queue"
          size="small"
          onClick={() => { loadQueue(); loadStats(); }}
          sx={{
            width: 34,
            height: 34,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: 'divider',
            bgcolor: 'action.hover',
            '&:hover': { bgcolor: alpha(accentColors.purple, 0.12) },
          }}
        >
          <RefreshIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>

      {showBulkApproveAction && (
        <Button
          size="small"
          variant="outlined"
          startIcon={<ApproveIcon />}
          onClick={handleBulkApprove}
          disabled={!selectedDomain}
          sx={{ height: 34, borderRadius: '8px', fontWeight: 700, px: 1.25 }}
        >
          Approve
        </Button>
      )}

      {showSearchAction && (
        <Button
          variant="contained"
          size="small"
          startIcon={searching ? <CircularProgress size={16} color="inherit" /> : <SearchIcon />}
          onClick={handleSearch}
          disabled={searching || !searchQuery.trim()}
          sx={{ height: 34, borderRadius: '8px', fontWeight: 800, px: 1.25 }}
        >
          {searching ? 'Searching' : 'Search'}
        </Button>
      )}
    </Box>
  );
  const discoverySummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      {(stats?.total || 0) > 0 && (
        <StatusChip status="query" label={`${formatNumber(stats?.total || 0)} discovered`} statusConfig={STATUS_CONFIG} height={22} />
      )}
      {pendingCount > 0 && (
        <StatusChip status="pending" label={`${formatNumber(pendingCount)} pending`} statusConfig={STATUS_CONFIG} height={22} />
      )}
      {selectedDocs.size > 0 && (
        <StatusChip status="domain" label={`${formatNumber(selectedDocs.size)} selected`} statusConfig={STATUS_CONFIG} height={22} />
      )}
      {selectedDomain && (
        <StatusChip
          status="domain"
          label={selectedDomainName}
          statusConfig={STATUS_CONFIG}
          height={22}
          sx={{ maxWidth: 220, '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis' } }}
        />
      )}
      {hasDiscoverySummary && (
        <Typography
          variant="caption"
          color="text.secondary"
          noWrap
          sx={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}
        >
          {topSource ? `${topSource[0]} leads` : `${formatNumber(queueTotal)} queued`} / {formatNumber(approvedCount)} approved / {formatNumber(rejectedCount)} rejected / {Math.round(averageRelevance * 100)}% relevance
        </Typography>
      )}
    </Box>
  );

  return (
    <PageShell
      sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 1.25,
        py: 1.25,
        overflow: 'visible',
      }}
    >
      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Box
        data-discovery-workspace
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: 1.25,
          overflow: 'visible',
        }}
      >
      {searchResults.length > 0 && (
          <SectionCard
            title={`Search Results (${searchResults.length})`}
            subtitle="Review discovered links before adding them to the queue"
            icon={<WebSearchIcon />}
            accentColor={accentColors.purple}
          >
            <Grid container spacing={1.25}>
              {searchResults.map(doc => (
                <Grid item xs={12} md={6} key={doc.id}>
                  <Box
                    sx={{
                      ...discoveryPanelSx,
                      p: 1.5,
                      '&:hover': { borderColor: alpha(accentColors.purple, 0.5) },
                    }}
                  >
                    <Typography variant="subtitle2" noWrap sx={{ mb: 0.75 }}>
                      {doc.title}
                    </Typography>
                    <Typography
                      variant="caption"
                      component="a"
                      href={doc.url}
                      target="_blank"
                      color="text.secondary"
                      sx={{
                        display: 'block',
                        mb: 0.75,
                        textDecoration: 'none',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        '&:hover': { textDecoration: 'underline' },
                      }}
                    >
                      {doc.url}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                    }}>
                      {doc.snippet}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </SectionCard>
      )}

          <SectionCard
            title="Review Queue"
            subtitle={`${formatNumber(queueTotal)} documents matching ${activeStatusLabel.toLowerCase()}`}
            icon={<QueueIcon />}
            accentColor={accentColors.warning}
            action={discoveryHeaderActions}
          >
            {discoverySummaryRow}

            {/* Bulk actions */}
            {selectedDocs.size > 0 && (
              <Alert 
                severity="info" 
                sx={{ mb: 2 }}
                action={
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button 
                      size="small" 
                      color="inherit"
                      onClick={() => setSelectedDocs(new Set())}
                    >
                      Clear
                    </Button>
                    <Button
                      size="small"
                      variant="contained"
                      onClick={handleBulkApprove}
                      disabled={!selectedDomain}
                      sx={{ background: gradients.success }}
                    >
                      Approve {selectedDocs.size} Documents
                    </Button>
                  </Box>
                }
              >
                {selectedDocs.size} document(s) selected
              </Alert>
            )}

            {/* Select all */}
            {queue.length > 0 && (
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1.25 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={selectedDocs.size === queue.length}
                      indeterminate={selectedDocs.size > 0 && selectedDocs.size < queue.length}
                      onChange={selectAll}
                    />
                  }
                  label="Select All"
                />
              </Box>
            )}

            {/* Queue list */}
            {queueLoading ? (
              <Box sx={{ minHeight: 120, py: 1 }}>
                {[1, 2, 3].map(i => (
                  <Skeleton key={i} variant="rectangular" height={120} sx={{ mb: 1.25, borderRadius: '8px' }} />
                ))}
              </Box>
            ) : queue.length === 0 ? (
              <Box
                data-discovery-empty-queue
                sx={{
                  p: 1.25,
                  display: 'flex',
                  alignItems: { xs: 'flex-start', sm: 'center' },
                  justifyContent: 'space-between',
                  gap: 1,
                  flexWrap: 'wrap',
                  borderRadius: '8px',
                  border: '1px dashed',
                  borderColor: 'divider',
                  bgcolor: alpha(accentColors.warning, 0.04),
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
                  <QueueIcon sx={{ fontSize: 20, color: 'text.disabled', flexShrink: 0 }} />
                  <Box sx={{ minWidth: 0 }}>
                  <Typography variant="body2" color="text.primary" sx={{ fontWeight: 800 }}>
                    No documents in queue
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Search the web above to collect candidates for review.
                  </Typography>
                  </Box>
                </Box>
              </Box>
            ) : (
              <Box>
                {queue.map(doc => (
                  <DiscoveredDocumentCard
                    key={doc.id}
                    doc={doc}
                    selected={selectedDocs.has(doc.id)}
                    onSelect={() => toggleSelection(doc.id)}
                    onApprove={() => {
                      setApprovingDoc(doc);
                      setApproveDialogOpen(true);
                    }}
                    onReject={() => handleReject(doc.id)}
                    onExtract={() => handleExtract(doc.id)}
                    onView={() => {
                      setViewingDoc(doc);
                      setViewDialogOpen(true);
                    }}
                  />
                ))}
              </Box>
            )}
          </SectionCard>
      </Box>

      {/* View Document Dialog */}
      <Dialog open={viewDialogOpen} onClose={() => setViewDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <Box>
              <Typography variant="h6">{viewingDoc?.title}</Typography>
              <Typography
                variant="caption"
                component="a"
                href={viewingDoc?.url}
                target="_blank"
                color="text.secondary"
              >
                {viewingDoc?.url}
              </Typography>
            </Box>
            <IconButton onClick={() => setViewDialogOpen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          {viewingDoc?.content ? (
            <Box
              sx={{
                ...discoveryPanelSx,
                p: 2,
                bgcolor: alpha(accentColors.slate, 0.16),
                maxHeight: 500,
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
                fontFamily: 'monospace',
                fontSize: 13,
              }}
            >
              {viewingDoc.content}
            </Box>
          ) : (
            <Alert severity="info">
              Content has not been extracted yet. Click "Extract Content" to fetch the document.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          {viewingDoc?.status === 'pending' && (
            <>
              <Button
                startIcon={<RejectIcon />}
                onClick={() => {
                  handleReject(viewingDoc.id);
                  setViewDialogOpen(false);
                }}
                color="error"
              >
                Reject
              </Button>
              <Button
                startIcon={<ApproveIcon />}
                onClick={() => {
                  setViewDialogOpen(false);
                  setApprovingDoc(viewingDoc);
                  setApproveDialogOpen(true);
                }}
                variant="contained"
                sx={{ background: gradients.success }}
              >
                Approve
              </Button>
            </>
          )}
        </DialogActions>
      </Dialog>

      {/* Approve Dialog */}
      <Dialog open={approveDialogOpen} onClose={() => setApproveDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Approve Document</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Approving: <strong>{approvingDoc?.title}</strong>
          </Typography>
          <FormControl fullWidth>
            <InputLabel>Add to Domain</InputLabel>
            <Select
              value={selectedDomain}
              label="Add to Domain"
              onChange={(e) => setSelectedDomain(e.target.value)}
            >
              {domains.map(domain => (
                <MenuItem key={domain.id} value={domain.id}>{domain.name}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setApproveDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={() => approvingDoc && handleApprove(approvingDoc.id, selectedDomain)}
            variant="contained"
            disabled={!selectedDomain}
            sx={{ background: gradients.success }}
          >
            Approve & Add
          </Button>
        </DialogActions>
      </Dialog>
    </PageShell>
  );
};

export default DiscoveryPage;
