/**
 * IntelligentSearchPage - Unified search interface with GraphRAG integration
 * Provides hybrid search with LLM answer generation
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Chip,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Alert,
  Divider,
  alpha,
  useTheme,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip,
  LinearProgress,
} from '@mui/material';
import {
  Search as SearchIcon,
  Psychology as IntentIcon,
  TrendingUp as ScoreIcon,
  Description as SourceIcon,
  AccountTree as GraphIcon,
  AutoAwesome as AutoIcon,
  ShowChart as VectorIcon,
  Lightbulb as SuggestionIcon,
  ExpandMore as ExpandIcon,
  Refresh as RefreshIcon,
  Hub as HybridIcon,
  TextFields as KeywordIcon,
  Psychology,
} from '@mui/icons-material';
import { apiService as api } from '../services/api';

type SearchType = 'auto' | 'hybrid' | 'vector' | 'graph' | 'keyword';

interface SearchResult {
  content: string;
  source: string;
  score: number;
  metadata?: any;
}

interface QueryAnalysis {
  intent?: string;
  entities?: string[];
  keywords?: string[];
  complexity?: number;
}

const IntelligentSearchPage: React.FC = () => {
  const theme = useTheme();
  
  // Search state
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState<SearchType>('auto');
  const [searching, setSearching] = useState(false);
  
  // Results
  const [answer, setAnswer] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [queryAnalysis, setQueryAnalysis] = useState<QueryAnalysis | null>(null);
  const [searchMethodUsed, setSearchMethodUsed] = useState<string | null>(null);
  
  // UI state
  const [error, setError] = useState<string | null>(null);
  const [domains, setDomains] = useState<string[]>([]);
  const [selectedDomain, setSelectedDomain] = useState<string>('');
  const [expandedSource, setExpandedSource] = useState<string | null>(null);

  // Check GraphRAG status on mount
  useEffect(() => {
    checkServiceStatus();
  }, []);

  const checkServiceStatus = async () => {
    try {
      const health = await api.getGraphRAGHealth();
      if (!health.available) {
        setError('GraphRAG service is not available. Please ensure the service is running.');
      } else {
        // Load domains
        const domainsRes = await api.getGraphRAGDomains();
        setDomains(domainsRes.domains || []);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to connect to GraphRAG service');
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    setSearching(true);
    setError(null);
    setAnswer(null);
    setResults([]);
    setQueryAnalysis(null);

    try {
      const result = await api.intelligentSearch({
        query,
        search_type: searchType,
        top_k: 10,
        domain: selectedDomain || undefined,
      });

      setAnswer(result.answer);
      setResults(result.results || []);
      setQueryAnalysis(result.query_analysis || null);
      setSearchMethodUsed(result.search_method_used || searchType);
    } catch (err: any) {
      setError(err.message || 'Search failed');
    } finally {
      setSearching(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  const getSearchTypeConfig = (type: SearchType) => {
    const configs = {
      auto: { icon: <AutoIcon />, color: '#6366f1', label: 'Auto (Recommended)' },
      hybrid: { icon: <HybridIcon />, color: '#10b981', label: 'Hybrid' },
      vector: { icon: <VectorIcon />, color: '#3b82f6', label: 'Vector' },
      graph: { icon: <GraphIcon />, color: '#8b5cf6', label: 'Graph' },
      keyword: { icon: <KeywordIcon />, color: '#f59e0b', label: 'Keyword' },
    };
    return configs[type];
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight={700} gutterBottom>
          Intelligent Search
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Powered by GraphRAG - Hybrid search combining vector similarity, knowledge graph traversal, and keyword matching with LLM answer generation.
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Search Input Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <TextField
              fullWidth
              multiline
              rows={2}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask anything... (e.g., What are the main components? How does the system work?)"
              variant="outlined"
              sx={{
                '& .MuiOutlinedInput-root': {
                  fontSize: '1.1rem',
                },
              }}
            />
          </Box>

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Search Method</InputLabel>
              <Select
                value={searchType}
                label="Search Method"
                onChange={(e) => setSearchType(e.target.value as SearchType)}
              >
                {(['auto', 'hybrid', 'vector', 'graph', 'keyword'] as SearchType[]).map(type => {
                  const config = getSearchTypeConfig(type);
                  return (
                    <MenuItem key={type} value={type}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box sx={{ color: config.color, display: 'flex' }}>{config.icon}</Box>
                        {config.label}
                      </Box>
                    </MenuItem>
                  );
                })}
              </Select>
            </FormControl>

            {domains.length > 0 && (
              <FormControl size="small" sx={{ minWidth: 180 }}>
                <InputLabel>Domain Filter</InputLabel>
                <Select
                  value={selectedDomain}
                  label="Domain Filter"
                  onChange={(e) => setSelectedDomain(e.target.value)}
                >
                  <MenuItem value="">All Domains</MenuItem>
                  {domains.map(domain => (
                    <MenuItem key={domain} value={domain}>{domain}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}

            <Button
              variant="contained"
              size="large"
              startIcon={searching ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
              onClick={handleSearch}
              disabled={searching || !query.trim()}
              sx={{ minWidth: 120 }}
            >
              {searching ? 'Searching...' : 'Search'}
            </Button>
          </Box>

          {searchMethodUsed && searchMethodUsed !== searchType && (
            <Alert severity="info" sx={{ mt: 2 }}>
              Using <strong>{searchMethodUsed}</strong> search method based on query analysis
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Query Analysis */}
      {queryAnalysis && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <IntentIcon sx={{ color: '#6366f1' }} />
              <Typography variant="h6">Query Analysis</Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {queryAnalysis.intent && (
                <Chip
                  label={`Intent: ${queryAnalysis.intent}`}
                  size="small"
                  sx={{ bgcolor: alpha('#6366f1', 0.1), color: '#6366f1' }}
                />
              )}
              {queryAnalysis.complexity !== undefined && (
                <Chip
                  label={`Complexity: ${queryAnalysis.complexity}/10`}
                  size="small"
                  sx={{ bgcolor: alpha('#f59e0b', 0.1), color: '#f59e0b' }}
                />
              )}
              {queryAnalysis.entities && queryAnalysis.entities.length > 0 && (
                <>
                  <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                    Entities:
                  </Typography>
                  {queryAnalysis.entities.map((entity, idx) => (
                    <Chip
                      key={idx}
                      label={entity}
                      size="small"
                      variant="outlined"
                      color="primary"
                    />
                  ))}
                </>
              )}
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Answer Card */}
      {answer && (
        <Card sx={{ mb: 3, borderLeft: '4px solid', borderColor: '#10b981' }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <Psychology sx={{ color: '#10b981', fontSize: 28 }} />
              <Typography variant="h6" fontWeight={600}>
                Answer
              </Typography>
              <Chip
                label={`Based on ${results.length} sources`}
                size="small"
                sx={{ ml: 'auto', bgcolor: alpha('#10b981', 0.1), color: '#10b981' }}
              />
            </Box>
            <Typography
              variant="body1"
              sx={{
                lineHeight: 1.8,
                whiteSpace: 'pre-wrap',
                color: 'text.primary',
              }}
            >
              {answer}
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Sources Panel */}
      {results.length > 0 && (
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <SourceIcon sx={{ color: '#3b82f6' }} />
              <Typography variant="h6">Sources ({results.length})</Typography>
              <Chip
                label={searchMethodUsed || searchType}
                size="small"
                sx={{ ml: 'auto', textTransform: 'capitalize' }}
              />
            </Box>

            <List sx={{ width: '100%' }}>
              {results.map((result, idx) => (
                <React.Fragment key={idx}>
                  {idx > 0 && <Divider component="li" />}
                  <Accordion
                    expanded={expandedSource === `source-${idx}`}
                    onChange={(_, isExpanded) => setExpandedSource(isExpanded ? `source-${idx}` : null)}
                    elevation={0}
                    sx={{
                      '&:before': { display: 'none' },
                      bgcolor: 'transparent',
                    }}
                  >
                    <AccordionSummary expandIcon={<ExpandIcon />}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            width: 32,
                            height: 32,
                            borderRadius: '50%',
                            bgcolor: alpha('#3b82f6', 0.1),
                            color: '#3b82f6',
                            fontWeight: 600,
                            fontSize: '0.9rem',
                          }}
                        >
                          {idx + 1}
                        </Box>
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                          <Typography variant="subtitle2" noWrap>
                            {result.source}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {result.content.substring(0, 120)}...
                          </Typography>
                        </Box>
                        <Chip
                          label={`${(result.score * 100).toFixed(0)}%`}
                          size="small"
                          sx={{
                            bgcolor: alpha('#10b981', 0.1),
                            color: '#10b981',
                            fontWeight: 600,
                          }}
                        />
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          bgcolor: alpha('#6366f1', 0.03),
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 1,
                        }}
                      >
                        <Typography
                          variant="body2"
                          sx={{
                            whiteSpace: 'pre-wrap',
                            lineHeight: 1.6,
                          }}
                        >
                          {result.content}
                        </Typography>
                        {result.metadata && Object.keys(result.metadata).length > 0 && (
                          <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                              Metadata:
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                              {Object.entries(result.metadata).map(([key, value]) => (
                                <Chip
                                  key={key}
                                  label={`${key}: ${value}`}
                                  size="small"
                                  variant="outlined"
                                  sx={{ fontSize: '0.7rem' }}
                                />
                              ))}
                            </Box>
                          </Box>
                        )}
                      </Paper>
                    </AccordionDetails>
                  </Accordion>
                </React.Fragment>
              ))}
            </List>
          </CardContent>
        </Card>
      )}

      {/* Empty State */}
      {!answer && !searching && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Box
              sx={{
                py: 8,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2,
              }}
            >
              <SearchIcon sx={{ fontSize: 80, color: 'text.disabled', opacity: 0.3 }} />
              <Typography variant="h6" color="text.secondary">
                Enter a query to search across your knowledge base
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
                <Chip
                  icon={<SuggestionIcon />}
                  label="What are the main components?"
                  onClick={() => setQuery('What are the main components?')}
                  clickable
                />
                <Chip
                  icon={<SuggestionIcon />}
                  label="How does the system work?"
                  onClick={() => setQuery('How does the system work?')}
                  clickable
                />
                <Chip
                  icon={<SuggestionIcon />}
                  label="Compare X and Y"
                  onClick={() => setQuery('Compare X and Y')}
                  clickable
                />
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Loading State */}
      {searching && (
        <Card>
          <CardContent>
            <Box sx={{ py: 4, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
              <CircularProgress size={60} />
              <Typography variant="h6" color="text.secondary">
                Analyzing query and searching knowledge base...
              </Typography>
              <LinearProgress sx={{ width: '60%' }} />
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Search Method Info */}
      <Paper
        elevation={0}
        sx={{
          mt: 3,
          p: 2,
          bgcolor: alpha('#6366f1', 0.05),
          border: '1px solid',
          borderColor: alpha('#6366f1', 0.1),
        }}
      >
        <Typography variant="subtitle2" gutterBottom>
          Search Methods Explained
        </Typography>
        <Box component="ul" sx={{ m: 0, pl: 2 }}>
          <Typography component="li" variant="caption" color="text.secondary">
            <strong>Auto:</strong> Automatically selects the best search method based on query analysis
          </Typography>
          <Typography component="li" variant="caption" color="text.secondary">
            <strong>Hybrid:</strong> Combines vector similarity, knowledge graph traversal, and keyword matching
          </Typography>
          <Typography component="li" variant="caption" color="text.secondary">
            <strong>Vector:</strong> Semantic similarity search using embeddings
          </Typography>
          <Typography component="li" variant="caption" color="text.secondary">
            <strong>Graph:</strong> Knowledge graph traversal using entities and relationships
          </Typography>
          <Typography component="li" variant="caption" color="text.secondary">
            <strong>Keyword:</strong> Traditional keyword-based text search
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default IntelligentSearchPage;

