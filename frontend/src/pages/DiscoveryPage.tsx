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
  Card,
  CardContent,
  CardActions,
  Tabs,
  Tab,
  Alert,
  CircularProgress,
  Divider,
  LinearProgress,
  Checkbox,
  FormControlLabel,
  Tooltip,
  Badge,
  Skeleton,
  Rating,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  Check as CheckIcon,
  Close as CloseIcon,
  Visibility as ViewIcon,
  OpenInNew as OpenInNewIcon,
  ThumbUp as ApproveIcon,
  ThumbDown as RejectIcon,
  AutoAwesome as AIIcon,
  CloudDownload as DownloadIcon,
  FilterList as FilterIcon,
  Queue as QueueIcon,
  Explore as ExploreIcon,
  TravelExplore as WebSearchIcon,
  Article as ArticleIcon,
  Link as LinkIcon,
  BarChart as StatsIcon,
  SelectAll as SelectAllIcon,
  CheckBox as CheckBoxIcon,
  CheckBoxOutlineBlank as CheckBoxOutlineBlankIcon,
} from '@mui/icons-material';
import { apiService as api } from '../services/api';

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

// Status configuration
const STATUS_CONFIG: Record<string, { color: string; label: string }> = {
  pending: { color: '#F59E0B', label: 'Pending Review' },
  approved: { color: '#10B981', label: 'Approved' },
  rejected: { color: '#EF4444', label: 'Rejected' },
  processing: { color: '#3B82F6', label: 'Processing' },
  error: { color: '#6B7280', label: 'Error' },
};

// Document Card Component
const DiscoveredDocumentCard: React.FC<{
  doc: DiscoveredDocument;
  selected: boolean;
  onSelect: () => void;
  onApprove: () => void;
  onReject: () => void;
  onExtract: () => void;
  onView: () => void;
}> = ({ doc, selected, onSelect, onApprove, onReject, onExtract, onView }) => {
  const statusConfig = STATUS_CONFIG[doc.status] || STATUS_CONFIG.pending;

  return (
    <Card
      sx={{
        mb: 2,
        border: selected ? '2px solid' : '1px solid',
        borderColor: selected ? '#6366F1' : 'rgba(255,255,255,0.1)',
        bgcolor: selected ? alpha('#6366F1', 0.05) : 'transparent',
        transition: 'all 0.2s',
        '&:hover': {
          borderColor: alpha('#6366F1', 0.5),
          transform: 'translateY(-2px)',
          boxShadow: `0 8px 32px ${alpha('#6366F1', 0.15)}`,
        },
      }}
    >
      <CardContent>
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
              <Chip
                label={statusConfig.label}
                size="small"
                sx={{
                  bgcolor: alpha(statusConfig.color, 0.15),
                  color: statusConfig.color,
                  ml: 1,
                }}
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
                      borderRadius: 3,
                      bgcolor: alpha('#10B981', 0.1),
                      '& .MuiLinearProgress-bar': {
                        bgcolor: doc.relevance_score > 0.7 ? '#10B981' : doc.relevance_score > 0.4 ? '#F59E0B' : '#EF4444',
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
                      borderRadius: 3,
                      bgcolor: alpha('#3B82F6', 0.1),
                      '& .MuiLinearProgress-bar': {
                        bgcolor: doc.quality_score > 0.7 ? '#10B981' : doc.quality_score > 0.4 ? '#F59E0B' : '#EF4444',
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
              <Chip
                label={doc.source || 'Unknown'}
                size="small"
                variant="outlined"
                sx={{ height: 20, fontSize: 10 }}
              />
              {doc.search_query && (
                <Chip
                  label={`Query: ${doc.search_query}`}
                  size="small"
                  variant="outlined"
                  sx={{ height: 20, fontSize: 10 }}
                />
              )}
            </Box>
          </Box>
        </Box>
      </CardContent>
      
      <CardActions sx={{ px: 2, pb: 2, pt: 0, justifyContent: 'flex-end' }}>
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
              sx={{ background: 'linear-gradient(135deg, #10B981 0%, #06B6D4 100%)' }}
            >
              Approve
            </Button>
          </>
        )}
      </CardActions>
    </Card>
  );
};

// Main Component
const DiscoveryPage: React.FC = () => {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
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
      setDomains(res.data.domains || []);
    }).catch(() => {});
  }, []);

  // Load queue
  const loadQueue = useCallback(async () => {
    setQueueLoading(true);
    try {
      const res = await api.get('/api/v1/rag/discover/queue', {
        params: { status: statusFilter || undefined, limit: 50 }
      });
      setQueue(res.data.documents || []);
      setQueueTotal(res.data.total || 0);
    } catch (err) {
      console.error('Failed to load queue:', err);
    } finally {
      setQueueLoading(false);
    }
  }, [statusFilter]);

  // Load stats
  const loadStats = useCallback(async () => {
    try {
      const res = await api.get('/api/v1/rag/discover/statistics');
      setStats(res.data);
    } catch (err) {
      console.error('Failed to load stats:', err);
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

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" sx={{ 
            fontWeight: 700, 
            mb: 0.5,
            background: 'linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Document Discovery
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Search the web and review documents for your knowledge base
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <IconButton onClick={() => { loadQueue(); loadStats(); }}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Stats Cards */}
      {stats && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {[
            { label: 'Total Discovered', value: stats.total, color: '#6366F1', icon: <ExploreIcon /> },
            { label: 'Pending Review', value: stats.by_status?.pending || 0, color: '#F59E0B', icon: <QueueIcon /> },
            { label: 'Approved', value: stats.by_status?.approved || 0, color: '#10B981', icon: <ApproveIcon /> },
            { label: 'Avg Relevance', value: `${Math.round((stats.avg_relevance || 0) * 100)}%`, color: '#3B82F6', icon: <StatsIcon /> },
          ].map((stat, i) => (
            <Grid item xs={3} key={i}>
              <Card sx={{ 
                background: `linear-gradient(135deg, ${alpha(stat.color, 0.15)} 0%, ${alpha(stat.color, 0.05)} 100%)`,
                border: `1px solid ${alpha(stat.color, 0.3)}`,
              }}>
                <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 2 }}>
                  <Box sx={{ 
                    p: 1.5, 
                    borderRadius: 2, 
                    background: alpha(stat.color, 0.2),
                    color: stat.color,
                  }}>
                    {stat.icon}
                  </Box>
                  <Box>
                    <Typography variant="h5" sx={{ fontWeight: 700, color: stat.color }}>
                      {stat.value}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {stat.label}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      <Grid container spacing={3}>
        {/* Search Panel */}
        <Grid item xs={12}>
          <Paper sx={{ 
            p: 3,
            background: 'linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(15, 15, 26, 0.9) 100%)',
            border: '1px solid rgba(139, 92, 246, 0.2)',
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
              <WebSearchIcon sx={{ color: '#8B5CF6' }} />
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                Web Search
              </Typography>
            </Box>
            
            <Box sx={{ display: 'flex', gap: 2 }}>
              <TextField
                fullWidth
                placeholder="Search for documents to add to your knowledge base..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                InputProps={{
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.disabled' }} />,
                }}
              />
              <Button
                variant="contained"
                onClick={handleSearch}
                disabled={searching || !searchQuery.trim()}
                sx={{ 
                  minWidth: 140,
                  background: 'linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%)',
                }}
              >
                {searching ? <CircularProgress size={20} /> : 'Search Web'}
              </Button>
            </Box>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2 }}>
                  Search Results ({searchResults.length})
                </Typography>
                <Grid container spacing={2}>
                  {searchResults.map(doc => (
                    <Grid item xs={6} key={doc.id}>
                      <Card sx={{ 
                        p: 2,
                        border: '1px solid rgba(255,255,255,0.1)',
                        '&:hover': { borderColor: alpha('#8B5CF6', 0.5) },
                      }}>
                        <Typography variant="subtitle2" noWrap sx={{ mb: 1 }}>
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
                            mb: 1,
                            textDecoration: 'none',
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
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Review Queue */}
        <Grid item xs={12}>
          <Paper sx={{ 
            p: 3,
            background: 'linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(15, 15, 26, 0.9) 100%)',
            border: '1px solid rgba(139, 92, 246, 0.2)',
          }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <QueueIcon sx={{ color: '#F59E0B' }} />
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Review Queue
                </Typography>
                <Badge badgeContent={queueTotal} color="warning">
                  <Chip label="pending" size="small" />
                </Badge>
              </Box>
              <Box sx={{ display: 'flex', gap: 2 }}>
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
                <FormControl size="small" sx={{ minWidth: 150 }}>
                  <InputLabel>Target Domain</InputLabel>
                  <Select
                    value={selectedDomain}
                    label="Target Domain"
                    onChange={(e) => setSelectedDomain(e.target.value)}
                  >
                    <MenuItem value="">Select domain...</MenuItem>
                    {domains.map(domain => (
                      <MenuItem key={domain.id} value={domain.id}>{domain.name}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>
            </Box>

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
                      sx={{ background: 'linear-gradient(135deg, #10B981 0%, #06B6D4 100%)' }}
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
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={queue.length > 0 && selectedDocs.size === queue.length}
                    indeterminate={selectedDocs.size > 0 && selectedDocs.size < queue.length}
                    onChange={selectAll}
                  />
                }
                label="Select All"
              />
            </Box>

            {/* Queue list */}
            {queueLoading ? (
              <Box sx={{ py: 4 }}>
                {[1, 2, 3].map(i => (
                  <Skeleton key={i} variant="rectangular" height={180} sx={{ mb: 2, borderRadius: 2 }} />
                ))}
              </Box>
            ) : queue.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 8 }}>
                <QueueIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                <Typography variant="h6" color="text.secondary">
                  No documents in queue
                </Typography>
                <Typography variant="body2" color="text.disabled">
                  Use web search above to discover new documents
                </Typography>
              </Box>
            ) : (
              queue.map(doc => (
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
              ))
            )}
          </Paper>
        </Grid>
      </Grid>

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
            <Paper
              sx={{
                p: 2,
                bgcolor: alpha('#000', 0.3),
                maxHeight: 500,
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
                fontFamily: 'monospace',
                fontSize: 13,
              }}
            >
              {viewingDoc.content}
            </Paper>
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
                sx={{ background: 'linear-gradient(135deg, #10B981 0%, #06B6D4 100%)' }}
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
            sx={{ background: 'linear-gradient(135deg, #10B981 0%, #06B6D4 100%)' }}
          >
            Approve & Add
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DiscoveryPage;
