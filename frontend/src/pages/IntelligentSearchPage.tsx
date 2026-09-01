/**
 * IntelligentSearchPage - Unified search interface with GraphRAG integration
 * Provides hybrid search with LLM answer generation
 */
import { Fragment, useState, useEffect, useCallback, type KeyboardEvent } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  List,
  CircularProgress,
  Alert,
  Divider,
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
import { alpha } from '@mui/material/styles';
import {
  Search as SearchIcon,
  Psychology as IntentIcon,
  Description as SourceIcon,
  AccountTree as GraphIcon,
  AutoAwesome as AutoIcon,
  ShowChart as VectorIcon,
  ExpandMore as ExpandIcon,
  Refresh as RefreshIcon,
  Hub as HybridIcon,
  TextFields as KeywordIcon,
  Psychology,
} from '@mui/icons-material';
import {
  PageShell,
  SectionCard,
  StatusChip,
  type StatusChipConfigMap,
} from '../components/common';
import { apiService as api } from '../services/api';
import { accentColors } from '../utils/theme';

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

const INTELLIGENT_SEARCH_STATUS_CONFIG: StatusChipConfigMap = {
  answer: { color: accentColors.success, label: 'Generated answer' },
  auto: { color: accentColors.primary, label: 'Auto' },
  checking: { color: accentColors.warning, label: 'Checking GraphRAG' },
  complexity: { color: accentColors.warning, label: 'Complexity' },
  degraded: { color: accentColors.rose, label: 'GraphRAG degraded' },
  entity: { color: accentColors.primary, label: 'Entity' },
  graph: { color: accentColors.purple, label: 'Graph' },
  healthy: { color: accentColors.success, label: 'GraphRAG ready' },
  hybrid: { color: accentColors.success, label: 'Hybrid' },
  intent: { color: accentColors.primary, label: 'Intent' },
  keyword: { color: accentColors.warning, label: 'Keyword' },
  knowledge: { color: accentColors.info, label: 'Knowledge search' },
  metadata: { color: accentColors.slate, label: 'Metadata' },
  score: { color: accentColors.success, label: 'Score' },
  sources: { color: accentColors.info, label: 'Sources' },
  suggestion: { color: accentColors.info, label: 'Suggestion' },
  tools: { color: accentColors.info, label: 'Search tools' },
  vector: { color: accentColors.info, label: 'Vector' },
};

const searchSurfacePanelSx = {
  borderRadius: '8px',
  bgcolor: 'background.paper',
  border: '1px solid',
  borderColor: 'divider',
} as const;

const IntelligentSearchPage = () => {
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
  const [graphStatus, setGraphStatus] = useState<'checking' | 'healthy' | 'degraded'>('checking');

  const checkServiceStatus = useCallback(async () => {
    try {
      const health = await api.getGraphRAGHealth();
      const healthy = ['healthy', 'ok', 'available'].includes(health.status?.toLowerCase?.() || '');

      if (!healthy) {
        setGraphStatus('degraded');
        setError(health.message || health.error || 'GraphRAG service is not available. Please ensure the service is running.');
      } else {
        setGraphStatus('healthy');
        setError(null);
        const domainsRes = await api.getGraphRAGDomains();
        setDomains(domainsRes.domains || []);
      }
    } catch (err: any) {
      setGraphStatus('degraded');
      setError(err.message || 'Failed to connect to GraphRAG service');
    }
  }, []);

  // Check GraphRAG status on mount
  useEffect(() => {
    checkServiceStatus();
  }, [checkServiceStatus]);

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

      setAnswer(result.answer ?? null);
      setResults(result.results || []);
      setQueryAnalysis(result.query_analysis || null);
      setSearchMethodUsed(result.search_method_used || searchType);
    } catch (err: any) {
      setError(err.message || 'Search failed');
    } finally {
      setSearching(false);
    }
  };

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  const getSearchTypeConfig = (type: SearchType) => {
    const configs = {
      auto: { icon: <AutoIcon />, color: accentColors.primary, label: 'Auto (Recommended)' },
      hybrid: { icon: <HybridIcon />, color: accentColors.success, label: 'Hybrid' },
      vector: { icon: <VectorIcon />, color: accentColors.info, label: 'Vector' },
      graph: { icon: <GraphIcon />, color: accentColors.purple, label: 'Graph' },
      keyword: { icon: <KeywordIcon />, color: accentColors.warning, label: 'Keyword' },
    };
    return configs[type];
  };

  const entityCount = queryAnalysis?.entities?.length || 0;
  const keywordCount = queryAnalysis?.keywords?.length || 0;
  const searchUnavailable = graphStatus === 'degraded';
  const unavailableMessage = error || 'Start GraphRAG, then refresh service status to search indexed documents.';
  const searchHeaderActions = searchUnavailable ? (
    <Button
      size="small"
      variant="outlined"
      startIcon={<RefreshIcon />}
      onClick={checkServiceStatus}
      sx={{ height: 34, borderRadius: '8px', borderColor: 'divider', fontWeight: 800 }}
    >
      Retry
    </Button>
  ) : (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      <TextField
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyPress}
        placeholder="Search indexed documents..."
        size="small"
        sx={{ width: { xs: '100%', sm: 320, lg: 420 }, '& .MuiInputBase-root': { height: 34 } }}
      />

      <FormControl size="small" sx={{ width: { xs: '100%', sm: 164 } }}>
        <InputLabel>Method</InputLabel>
        <Select
          value={searchType}
          label="Method"
          onChange={(e) => setSearchType(e.target.value as SearchType)}
          renderValue={(value) => getSearchTypeConfig(value as SearchType).label.replace(' (Recommended)', '')}
          sx={{ height: 34 }}
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
        <FormControl size="small" sx={{ width: { xs: '100%', sm: 190 } }}>
          <InputLabel>Domain</InputLabel>
          <Select
            value={selectedDomain}
            label="Domain"
            onChange={(e) => setSelectedDomain(e.target.value)}
            sx={{ height: 34 }}
          >
            <MenuItem value="">All Domains</MenuItem>
            {domains.map(domain => (
              <MenuItem key={domain} value={domain}>{domain}</MenuItem>
            ))}
          </Select>
        </FormControl>
      )}

      <Tooltip title="Refresh GraphRAG status">
        <IconButton
          aria-label="Refresh GraphRAG status"
          size="small"
          onClick={checkServiceStatus}
          sx={{ width: 34, height: 34, borderRadius: '8px', border: '1px solid', borderColor: 'divider', bgcolor: 'action.hover', '&:hover': { bgcolor: alpha(accentColors.info, 0.12) } }}
        >
          <RefreshIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
      <Button
        variant="contained"
        size="small"
        startIcon={searching ? <CircularProgress size={16} color="inherit" /> : <SearchIcon />}
        onClick={handleSearch}
        disabled={searching || !query.trim() || searchUnavailable}
        sx={{ height: 34, borderRadius: '8px', fontWeight: 800, px: 1.25 }}
      >
        {searching ? 'Searching' : 'Search'}
      </Button>
    </Box>
  );
  const searchSummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
      <StatusChip status={graphStatus} statusConfig={INTELLIGENT_SEARCH_STATUS_CONFIG} height={22} />
      <StatusChip status={searchType} label={getSearchTypeConfig(searchType).label.replace(' (Recommended)', '')} statusConfig={INTELLIGENT_SEARCH_STATUS_CONFIG} height={22} />
      {results.length > 0 && (
        <StatusChip status="sources" label={`${results.length} sources`} statusConfig={INTELLIGENT_SEARCH_STATUS_CONFIG} height={22} />
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
      <SectionCard
        title="Knowledge Search"
        subtitle={searchUnavailable ? 'GraphRAG service unavailable' : 'Indexed documents, entities, and graph context'}
        icon={<SearchIcon />}
        accentColor={searchUnavailable ? accentColors.rose : accentColors.info}
        action={searchHeaderActions}
      >
        {searchUnavailable ? (
          <Box
            data-search-empty-state
            sx={{
              p: 1,
              borderRadius: '8px',
              border: '1px solid',
              borderColor: alpha(accentColors.warning, 0.28),
              bgcolor: alpha(accentColors.warning, 0.08),
            }}
          >
            <Typography variant="body2" sx={{ color: accentColors.warning, fontWeight: 700 }}>
              {unavailableMessage}
            </Typography>
          </Box>
        ) : searchSummaryRow}
      </SectionCard>

      <Box
        data-search-workspace
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: 1.25,
          overflow: 'visible',
        }}
      >
      {searchMethodUsed && searchMethodUsed !== searchType && (
        <Alert severity="info">
          Using <strong>{searchMethodUsed}</strong> search based on query analysis.
        </Alert>
      )}

      {error && !searchUnavailable && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Query Analysis */}
      {queryAnalysis && (
        <SectionCard
          title="Query Analysis"
          subtitle={`${entityCount} entities, ${keywordCount} keywords`}
          icon={<IntentIcon />}
          accentColor={accentColors.primary}
        >
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {queryAnalysis.intent && (
                <StatusChip
                  status="intent"
                  label={`Intent: ${queryAnalysis.intent}`}
                  statusConfig={INTELLIGENT_SEARCH_STATUS_CONFIG}
                />
              )}
              {queryAnalysis.complexity !== undefined && (
                <StatusChip
                  status="complexity"
                  label={`Complexity: ${queryAnalysis.complexity}/10`}
                  statusConfig={INTELLIGENT_SEARCH_STATUS_CONFIG}
                />
              )}
              {queryAnalysis.entities && queryAnalysis.entities.length > 0 && (
                <>
                  <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                    Entities:
                  </Typography>
                  {queryAnalysis.entities.map((entity, idx) => (
                    <StatusChip
                      key={idx}
                      status="entity"
                      label={entity}
                      variant="outlined"
                      statusConfig={INTELLIGENT_SEARCH_STATUS_CONFIG}
                    />
                  ))}
                </>
              )}
            </Box>
        </SectionCard>
      )}

      {/* Answer Card */}
      {answer && (
        <SectionCard
          title="Answer"
          subtitle="Generated from retrieved context"
          icon={<Psychology />}
          accentColor={accentColors.success}
          action={
              <StatusChip
                status="sources"
                label={`Based on ${results.length} sources`}
                statusConfig={INTELLIGENT_SEARCH_STATUS_CONFIG}
              />
          }
        >
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
        </SectionCard>
      )}

      {/* Sources Panel */}
      {results.length > 0 && (
        <SectionCard
          title={`Sources (${results.length})`}
          subtitle="Retrieved evidence and metadata"
          icon={<SourceIcon />}
          accentColor={accentColors.info}
          action={
              <StatusChip
                status={searchMethodUsed || searchType}
                label={searchMethodUsed || searchType}
                statusConfig={INTELLIGENT_SEARCH_STATUS_CONFIG}
                sx={{ textTransform: 'capitalize' }}
              />
          }
        >

            <List sx={{ width: '100%' }}>
              {results.map((result, idx) => (
                <Fragment key={idx}>
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
                            bgcolor: alpha(accentColors.info, 0.1),
                            color: accentColors.info,
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
                        <StatusChip
                          status="score"
                          label={`${(result.score * 100).toFixed(0)}%`}
                          statusConfig={INTELLIGENT_SEARCH_STATUS_CONFIG}
                          sx={{
                            fontWeight: 600,
                          }}
                        />
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Box
                        sx={{
                          ...searchSurfacePanelSx,
                          p: 2,
                          bgcolor: alpha(accentColors.primary, 0.03),
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
                                <StatusChip
                                  key={key}
                                  status="metadata"
                                  label={`${key}: ${value}`}
                                  variant="outlined"
                                  statusConfig={INTELLIGENT_SEARCH_STATUS_CONFIG}
                                  sx={{ fontSize: '0.7rem' }}
                                />
                              ))}
                            </Box>
                          </Box>
                        )}
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                </Fragment>
              ))}
            </List>
        </SectionCard>
      )}

      {/* Loading State */}
      {searching && (
        <Box
          sx={{
            ...searchSurfacePanelSx,
            width: '100%',
            borderTop: `2px solid ${accentColors.info}`,
            p: 1.25,
            display: 'flex',
            alignItems: 'center',
            gap: 1.25,
          }}
        >
          <CircularProgress size={20} />
          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 700 }}>
            Searching knowledge base
          </Typography>
          <LinearProgress sx={{ flex: 1, minWidth: 160 }} />
        </Box>
      )}
      </Box>
    </PageShell>
  );
};

export default IntelligentSearchPage;
