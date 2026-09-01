import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Grid,
  Alert,
  CircularProgress,
  IconButton,
  Divider,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  Search as SearchIcon,
  Code as CodeIcon,
  Functions as FunctionIcon,
  Class as ClassIcon,
  Api as ApiIcon,
  Terminal as TerminalIcon,
  Settings as ConfigIcon,
  ExpandMore as ExpandIcon,
  ContentCopy as CopyIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
  Analytics as AnalyticsIcon,
  Javascript as JavaScriptIcon,
  Storage as DatabaseIcon,
} from '@mui/icons-material';
import { useClipboard } from '@/hooks/useClipboard';
import {
  CodeHighlighter,
  MetricTile,
  PageShell,
  SectionCard,
  StatusChip,
  type StatusChipConfigMap,
} from '@/components/common';
import { accentColors } from '@/utils/theme';

type SearchType = 'semantic' | 'pattern' | 'ast';

const SEARCH_TYPE_OPTIONS: Array<{ value: SearchType; label: string }> = [
  { value: 'semantic', label: 'Semantic' },
  { value: 'pattern', label: 'Pattern' },
  { value: 'ast', label: 'AST' },
];

const AVAILABLE_LANGUAGES = [
  'python', 'javascript', 'typescript', 'java', 'go', 'rust',
  'cpp', 'c', 'csharp', 'php', 'ruby', 'swift'
];

const AVAILABLE_CODE_TYPES = [
  'function', 'class', 'method', 'variable', 'import', 'config'
];

const CODE_SEARCH_STATUS_CONFIG: StatusChipConfigMap = {
  ast: { color: accentColors.purple, label: 'AST' },
  'code-type': { color: accentColors.purple, label: 'Code type' },
  'code-type-selected': { color: accentColors.primary, label: 'Selected type' },
  dependency: { color: accentColors.slate, label: 'Dependency' },
  intelligence: { color: accentColors.primary, label: 'Code intelligence' },
  language: { color: accentColors.info, label: 'Language' },
  'language-selected': { color: accentColors.primary, label: 'Selected language' },
  checking: { color: accentColors.info, label: 'Checking service' },
  online: { color: accentColors.success, label: 'Service online' },
  offline: { color: accentColors.rose, label: 'Service offline' },
  pattern: { color: accentColors.warning, label: 'Pattern' },
  semantic: { color: accentColors.primary, label: 'Semantic' },
  search: { color: accentColors.warning, label: 'Search' },
  detection: { color: accentColors.success, label: 'Detection' },
  'score-high': { color: accentColors.success, label: 'High score' },
  'score-medium': { color: accentColors.warning, label: 'Medium score' },
  'score-low': { color: accentColors.slate, label: 'Low score' },
  summary: { color: accentColors.info, label: 'Summary' },
};

const codePanelSx = {
  borderRadius: '8px',
  bgcolor: 'background.paper',
  border: '1px solid',
  borderColor: 'divider',
} as const;

const getScoreStatus = (score: number) => {
  if (score > 0.8) return 'score-high';
  if (score > 0.55) return 'score-medium';
  return 'score-low';
};

interface CodeSearchResult {
  file_path: string;
  language: string;
  code_type: 'function' | 'class' | 'method' | 'variable' | 'import' | 'config' | 'other';
  name?: string;
  content: string;
  line_start: number;
  line_end: number;
  score: number;
  context?: string;
  dependencies?: string[];
  metadata?: Record<string, any>;
}

interface CodeDetectionResult {
  has_code: boolean;
  languages: string[];
  code_blocks: Array<{
    language: string;
    content: string;
    type: string;
    metadata?: Record<string, any>;
  }>;
  statistics: {
    total_blocks: number;
    by_language: Record<string, number>;
    by_type: Record<string, number>;
  };
}

export const CodeSearchPage: React.FC = () => {
  const copyToClipboard = useClipboard();
  const [activeTab, setActiveTab] = useState(0);
  
  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchType, setSearchType] = useState<SearchType>('semantic');
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>([]);
  const [selectedCodeTypes, setSelectedCodeTypes] = useState<string[]>([]);
  const [searchResults, setSearchResults] = useState<CodeSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  
  // Detection state
  const [detectionText, setDetectionText] = useState('');
  const [detectionResults, setDetectionResults] = useState<CodeDetectionResult | null>(null);
  const [detecting, setDetecting] = useState(false);
  
  // UI state
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [codeServiceAvailable, setCodeServiceAvailable] = useState(false);
  const [checkingService, setCheckingService] = useState(true);

  useEffect(() => {
    checkCodeServiceStatus();
  }, []);

  const checkCodeServiceStatus = async () => {
    setCheckingService(true);
    try {
      const response = await fetch('/api/v1/graphrag/code/health');
      const data = await response.json();
      setCodeServiceAvailable(data.available === true);
      if (data.available) {
        setError(null);
      }
    } catch (err) {
      setCodeServiceAvailable(false);
      setError('Unable to connect to code search service');
    } finally {
      setCheckingService(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setSearching(true);
    setError(null);
    setSearchResults([]);
    
    try {
      const response = await fetch('/api/v1/graphrag/code/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          search_type: searchType,
          languages: selectedLanguages.length > 0 ? selectedLanguages : undefined,
          code_types: selectedCodeTypes.length > 0 ? selectedCodeTypes : undefined,
          limit: 20
        })
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      setSearchResults(data.results || []);
      
      if (data.results?.length === 0) {
        setError('No code matches found. Try adjusting your search query or filters.');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Code search failed');
    } finally {
      setSearching(false);
    }
  };

  const handleDetection = async () => {
    if (!detectionText.trim()) return;
    
    setDetecting(true);
    setError(null);
    setDetectionResults(null);
    
    try {
      const response = await fetch('/api/v1/graphrag/code/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: detectionText
        })
      });

      if (!response.ok) {
        throw new Error(`Detection failed: ${response.statusText}`);
      }

      const data = await response.json();
      setDetectionResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Code detection failed');
    } finally {
      setDetecting(false);
    }
  };

  const toggleResultExpansion = (index: number) => {
    const newExpanded = new Set(expandedResults);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedResults(newExpanded);
  };

  const copyCode = (content: string) => {
    copyToClipboard(content, {
      successMessage: 'Code snippet copied.',
      successTitle: 'Code copied',
    });
  };

  const clearFilters = () => {
    setSelectedLanguages([]);
    setSelectedCodeTypes([]);
  };

  const getLanguageIcon = (language: string) => {
    switch (language.toLowerCase()) {
      case 'python':
        return <CodeIcon sx={{ color: accentColors.primary }} />;
      case 'javascript':
      case 'typescript':
        return <JavaScriptIcon sx={{ color: accentColors.warning }} />;
      case 'java':
        return <CodeIcon sx={{ color: accentColors.rose }} />;
      case 'go':
        return <CodeIcon sx={{ color: accentColors.info }} />;
      case 'rust':
        return <CodeIcon sx={{ color: accentColors.warning }} />;
      case 'sql':
        return <DatabaseIcon sx={{ color: accentColors.purple }} />;
      default:
        return <CodeIcon />;
    }
  };

  const selectedSearchType = SEARCH_TYPE_OPTIONS.find(option => option.value === searchType)?.label || searchType;
  const activeFilterCount = selectedLanguages.length + selectedCodeTypes.length;
  const resultLineCount = searchResults.reduce((total, result) => total + Math.max(result.line_end - result.line_start + 1, 0), 0);
  const detectionBlockCount = detectionResults?.statistics.total_blocks || 0;
  const detectionLanguageCount = detectionResults ? Object.keys(detectionResults.statistics.by_language).length : 0;
  const detectionTypeCount = detectionResults ? Object.keys(detectionResults.statistics.by_type).length : 0;
  const serviceStatus = checkingService ? 'checking' : codeServiceAvailable ? 'online' : 'offline';
  const serviceStatusLabel = checkingService ? 'Checking service' : codeServiceAvailable ? 'Service online' : 'Service offline';
  const codeServiceOffline = !checkingService && !codeServiceAvailable;
  const serviceUnavailableMessage = !checkingService && !codeServiceAvailable
    ? 'Code search service is unavailable; queries are disabled.'
    : null;
  const showSearchControls = codeServiceAvailable;
  const searchCardSubtitle = checkingService
    ? 'Checking code search service'
    : codeServiceAvailable
      ? `${selectedSearchType} search with ${activeFilterCount} active filters`
      : 'Service offline';

  const getCodeTypeIcon = (type: string) => {
    switch (type) {
      case 'function':
        return <FunctionIcon />;
      case 'class':
        return <ClassIcon />;
      case 'method':
        return <ApiIcon />;
      case 'config':
        return <ConfigIcon />;
      case 'import':
        return <TerminalIcon />;
      default:
      return <CodeIcon />;
    }
  };

  const refreshCodeServiceButton = (
    <Tooltip title="Refresh code service">
      <span>
        <IconButton
          aria-label="Refresh code service"
          size="small"
          onClick={checkCodeServiceStatus}
          disabled={checkingService}
          sx={{
            width: 34,
            height: 34,
            borderRadius: '8px',
            border: '1px solid',
            borderColor: 'divider',
            bgcolor: 'action.hover',
            '&:hover': { bgcolor: alpha(accentColors.primary, 0.12) },
          }}
        >
          {checkingService ? <CircularProgress size={16} /> : <RefreshIcon sx={{ fontSize: 18 }} />}
        </IconButton>
      </span>
    </Tooltip>
  );

  const codeSearchHeaderActions = codeServiceOffline ? (
    refreshCodeServiceButton
  ) : (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      <Button
        size="small"
        variant={activeTab === 0 ? 'contained' : 'outlined'}
        startIcon={<SearchIcon />}
        onClick={() => setActiveTab(0)}
        sx={{ height: 34, borderRadius: '8px', fontWeight: 800, px: 1.25 }}
      >
        Search
      </Button>
      <Button
        size="small"
        variant={activeTab === 1 ? 'contained' : 'outlined'}
        startIcon={<AnalyticsIcon />}
        onClick={() => setActiveTab(1)}
        sx={{ height: 34, borderRadius: '8px', fontWeight: 800, px: 1.25 }}
      >
        Detect
      </Button>
      {refreshCodeServiceButton}
    </Box>
  );
  const codeSearchSummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      <StatusChip status={serviceStatus} label={serviceStatusLabel} statusConfig={CODE_SEARCH_STATUS_CONFIG} height={22} />
      {searchResults.length > 0 && (
        <StatusChip status="summary" label={`${searchResults.length.toLocaleString()} results`} statusConfig={CODE_SEARCH_STATUS_CONFIG} height={22} />
      )}
      {activeFilterCount > 0 && (
        <StatusChip status="code-type" label={`${activeFilterCount} filters`} statusConfig={CODE_SEARCH_STATUS_CONFIG} height={22} />
      )}
      {(resultLineCount > 0 || detectionBlockCount > 0) && (
        <Typography variant="caption" color="text.secondary">
          {resultLineCount.toLocaleString()} matched lines / {detectionBlockCount.toLocaleString()} detected blocks
        </Typography>
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
        overflow: 'visible',
      }}
    >
      {error && !serviceUnavailableMessage && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {activeTab === 0 && (
        <Grid
          container
          spacing={1.5}
          data-code-search-workspace
          sx={{ alignItems: 'flex-start', overflow: 'visible' }}
        >
          <Grid item xs={12} lg={showSearchControls ? 8.5 : 12}>
            <SectionCard
              title="Search"
              subtitle={searchCardSubtitle}
              icon={<SearchIcon />}
              accentColor={accentColors.primary}
              action={codeSearchHeaderActions}
            >
              {!codeServiceOffline && codeSearchSummaryRow}
              {codeServiceOffline ? (
                <Box
                  data-code-search-offline-state
                  sx={{
                    p: 1.25,
                    px: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 1,
                    flexWrap: 'wrap',
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: alpha(accentColors.warning, 0.35),
                    bgcolor: alpha(accentColors.warning, 0.08),
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
                    <StatusChip status="offline" label="Offline" statusConfig={CODE_SEARCH_STATUS_CONFIG} />
                    <Typography variant="body2" color="text.secondary">
                      {serviceUnavailableMessage}
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <>
                  <Box
                    component="form"
                    onSubmit={(event) => {
                      event.preventDefault();
                      handleSearch();
                    }}
                    sx={{
                      display: 'grid',
                      gridTemplateColumns: 'minmax(0, 1fr) 38px',
                      gap: 0.75,
                      alignItems: 'center',
                      flexShrink: 0,
                    }}
                  >
                    <TextField
                      fullWidth
                      size="small"
                      placeholder="Search code by behavior, symbol, file path, or pattern"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      disabled={searching || !codeServiceAvailable}
                      InputProps={{
                        startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
                      }}
                    />
                    <Tooltip title={searching ? 'Searching code' : 'Run search'}>
                      <span>
                        <IconButton
                          type="submit"
                          aria-label="Run search"
                          disabled={searching || !searchQuery.trim() || !codeServiceAvailable}
                          sx={{
                            width: 38,
                            height: 38,
                            borderRadius: '8px',
                            border: '1px solid',
                            borderColor: alpha(accentColors.primary, 0.35),
                            color: 'primary.contrastText',
                            bgcolor: alpha(accentColors.primary, 0.82),
                            '&:hover': { bgcolor: accentColors.primary },
                            '&.Mui-disabled': {
                              color: alpha(accentColors.primary, 0.32),
                              borderColor: alpha(accentColors.primary, 0.16),
                              bgcolor: alpha(accentColors.primary, 0.08),
                            },
                          }}
                        >
                          {searching ? <CircularProgress size={16} /> : <SearchIcon sx={{ fontSize: 18 }} />}
                        </IconButton>
                      </span>
                    </Tooltip>
                  </Box>

                  <Divider sx={{ my: 1.5, flexShrink: 0 }} />

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, alignItems: 'center', mb: 1, flexShrink: 0 }}>
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
                        Results
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {searchResults.length
                          ? `${searchResults.length} matches across ${resultLineCount.toLocaleString()} lines`
                          : 'No results yet'}
                      </Typography>
                    </Box>
                  </Box>

                  {searching ? (
                    <Box sx={{ minHeight: 120, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <CircularProgress />
                    </Box>
                  ) : searchResults.length === 0 ? (
                    <Box
                      data-code-search-empty-state
                      sx={{
                        p: 1.25,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: 1,
                        flexWrap: 'wrap',
                        ...codePanelSx,
                        borderStyle: 'dashed',
                        bgcolor: alpha(accentColors.primary, 0.04),
                      }}
                    >
                      <Typography variant="body2" color="text.secondary">
                        No results yet.
                      </Typography>
                    </Box>
                  ) : (
                    <Box sx={{ display: 'grid', gap: 1.25, alignContent: 'start' }}>
                      {searchResults.map((result, index) => {
                        const lineCount = result.content.split('\n').length;
                        const isExpanded = expandedResults.has(index);

                    return (
                      <Box
                        key={`${result.file_path}-${result.line_start}-${index}`}
                        sx={{
                          p: 1.5,
                          ...codePanelSx,
                          transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
                          '&:hover': {
                            borderColor: alpha(accentColors.primary, 0.4),
                            boxShadow: `0 8px 28px ${alpha(accentColors.primary, 0.12)}`,
                          },
                        }}
                      >
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: { xs: 'flex-start', md: 'center' },
                            justifyContent: 'space-between',
                            flexDirection: { xs: 'column', md: 'row' },
                            gap: 1,
                            mb: 1,
                          }}
                        >
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
                            {getLanguageIcon(result.language)}
                            <Box sx={{ minWidth: 0 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 800 }} noWrap title={result.file_path}>
                                {result.file_path}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                Lines {result.line_start}-{result.line_end}
                              </Typography>
                            </Box>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                            <StatusChip
                              status="language"
                              label={result.language}
                              variant="outlined"
                              statusConfig={CODE_SEARCH_STATUS_CONFIG}
                            />
                            <StatusChip
                              status="code-type"
                              label={result.code_type}
                              icon={getCodeTypeIcon(result.code_type)}
                              variant="outlined"
                              statusConfig={CODE_SEARCH_STATUS_CONFIG}
                            />
                            <StatusChip
                              status={getScoreStatus(result.score)}
                              label={`Score ${result.score.toFixed(2)}`}
                              statusConfig={CODE_SEARCH_STATUS_CONFIG}
                            />
                            <Tooltip title="Copy code">
                              <IconButton size="small" onClick={() => copyCode(result.content)}>
                                <CopyIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </Box>

                        {result.name && (
                          <Typography variant="h6" sx={{ mb: 1, fontSize: '1rem', fontWeight: 800 }}>
                            {result.name}
                          </Typography>
                        )}

                        <CodeHighlighter
                          language={result.language}
                          variant="vscDarkPlus"
                          code={result.content}
                          customStyle={{
                            margin: 0,
                            borderRadius: '8px',
                            fontSize: 13,
                            maxHeight: isExpanded ? 'none' : '240px',
                            overflow: 'hidden'
                          }}
                          showLineNumbers
                          startingLineNumber={result.line_start}
                        />

                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: { xs: 'flex-start', sm: 'center' },
                            justifyContent: 'space-between',
                            flexDirection: { xs: 'column', sm: 'row' },
                            gap: 1,
                            mt: 1,
                          }}
                        >
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {result.dependencies?.map((dep, i) => (
                              <StatusChip
                                key={`${dep}-${i}`}
                                status="dependency"
                                label={dep}
                                variant="outlined"
                                statusConfig={CODE_SEARCH_STATUS_CONFIG}
                              />
                            ))}
                          </Box>
                          {lineCount > 8 && (
                            <Button
                              size="small"
                              onClick={() => toggleResultExpansion(index)}
                              startIcon={<ExpandIcon sx={{
                                transform: isExpanded ? 'rotate(180deg)' : 'none',
                                transition: 'transform 0.2s'
                              }} />}
                            >
                              {isExpanded ? 'Show Less' : 'Show More'}
                            </Button>
                          )}
                        </Box>
                      </Box>
                    );
                  })}
                    </Box>
                  )}
                </>
              )}
            </SectionCard>
          </Grid>

          {showSearchControls && (
          <Grid item xs={12} lg={3.5} sx={{ minHeight: { md: 0 } }}>
            <SectionCard
              title="Search controls"
              subtitle={activeFilterCount ? `${activeFilterCount} filters active` : 'No filters active'}
              icon={<FilterIcon />}
              accentColor={accentColors.warning}
            >
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Search Type</InputLabel>
                <Select
                  value={searchType}
                  onChange={(e) => setSearchType(e.target.value as SearchType)}
                  label="Search Type"
                >
                  {SEARCH_TYPE_OPTIONS.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                Languages
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 2 }}>
                {AVAILABLE_LANGUAGES.map(lang => (
                  <StatusChip
                    key={lang}
                    status={selectedLanguages.includes(lang) ? 'language-selected' : 'language'}
                    label={lang}
                    onClick={() => {
                      setSelectedLanguages(prev =>
                        prev.includes(lang)
                          ? prev.filter(l => l !== lang)
                          : [...prev, lang]
                      );
                    }}
                    variant={selectedLanguages.includes(lang) ? 'filled' : 'outlined'}
                    statusConfig={CODE_SEARCH_STATUS_CONFIG}
                    sx={{ cursor: 'pointer' }}
                  />
                ))}
              </Box>

              <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                Code types
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                {AVAILABLE_CODE_TYPES.map(type => (
                  <StatusChip
                    key={type}
                    status={selectedCodeTypes.includes(type) ? 'code-type-selected' : 'code-type'}
                    label={type}
                    icon={getCodeTypeIcon(type)}
                    onClick={() => {
                      setSelectedCodeTypes(prev =>
                        prev.includes(type)
                          ? prev.filter(t => t !== type)
                          : [...prev, type]
                      );
                    }}
                    variant={selectedCodeTypes.includes(type) ? 'filled' : 'outlined'}
                    statusConfig={CODE_SEARCH_STATUS_CONFIG}
                    sx={{
                      cursor: 'pointer',
                      '& .MuiChip-icon': { color: selectedCodeTypes.includes(type) ? accentColors.primary : accentColors.purple },
                    }}
                  />
                ))}
              </Box>

              {activeFilterCount > 0 && (
                <Button size="small" onClick={clearFilters} sx={{ mt: 2 }}>
                  Clear filters
                </Button>
              )}
            </SectionCard>
          </Grid>
          )}
        </Grid>
      )}

      {activeTab === 1 && (
        <Grid
          container
          spacing={1.5}
          data-code-detection-workspace
          sx={{ flex: 1, minHeight: { md: 0 }, alignItems: 'stretch', overflow: { xs: 'visible', md: 'hidden' } }}
        >
          <Grid item xs={12} lg={8.5} sx={{ minHeight: { md: 0 }, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            <SectionCard
              title="Detection input"
              subtitle={`${detectionText.length.toLocaleString()} characters queued`}
              icon={<AnalyticsIcon />}
              accentColor={accentColors.success}
              action={codeSearchHeaderActions}
              fullHeight={!detectionResults}
              sx={{
                flex: detectionResults ? '0 0 auto' : 1,
                minHeight: { md: 0 },
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
              }}
            >
              {codeSearchSummaryRow}
              <TextField
                fullWidth
                multiline
                minRows={12}
                placeholder="Paste mixed notes, markdown, logs, or source snippets"
                value={detectionText}
                onChange={(e) => setDetectionText(e.target.value)}
                sx={{
                  mb: 2,
                  flex: detectionResults ? undefined : 1,
                  '& .MuiInputBase-root': {
                    height: detectionResults ? undefined : '100%',
                    alignItems: 'flex-start',
                  },
                  '& textarea': {
                    height: detectionResults ? undefined : '100% !important',
                  },
                }}
              />
              <Button
                variant="contained"
                onClick={handleDetection}
                disabled={detecting || !detectionText.trim() || !codeServiceAvailable}
                startIcon={detecting ? <CircularProgress size={20} /> : <AnalyticsIcon />}
                sx={{ alignSelf: 'flex-start', flexShrink: 0 }}
              >
                {detecting ? 'Analyzing' : 'Analyze text'}
              </Button>
            </SectionCard>

            {detectionResults && (
              <SectionCard
                title="Detection results"
                subtitle={
                  detectionResults.has_code
                    ? `${detectionResults.statistics.total_blocks} code block${detectionResults.statistics.total_blocks === 1 ? '' : 's'} found`
                    : 'No code blocks detected'
                }
                icon={<AnalyticsIcon />}
                accentColor={detectionResults.has_code ? accentColors.success : accentColors.info}
                sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}
              >
                <Alert severity={detectionResults.has_code ? 'success' : 'info'} sx={{ mb: 2 }}>
                  {detectionResults.has_code
                    ? `Found ${detectionResults.statistics.total_blocks} code block(s)`
                    : 'No code blocks detected'}
                </Alert>

                {detectionResults.has_code && (
                  <>
                    <Grid container spacing={1.5} sx={{ mb: 2 }}>
                        {[
                          {
                            label: 'Languages',
                            value: detectionLanguageCount.toLocaleString(),
                            chips: Object.entries(detectionResults.statistics.by_language).map(([lang, count]) => ({
                              key: lang,
                              label: `${lang} (${count})`,
                              icon: getLanguageIcon(lang),
                            })),
                            color: accentColors.info,
                            status: 'language',
                          },
                          {
                            label: 'Code Types',
                            value: detectionTypeCount.toLocaleString(),
                            chips: Object.entries(detectionResults.statistics.by_type).map(([type, count]) => ({
                              key: type,
                              label: `${type} (${count})`,
                              icon: getCodeTypeIcon(type),
                            })),
                            color: accentColors.purple,
                            status: 'code-type',
                          },
                        ].map((group) => (
                          <Grid item xs={12} md={6} key={group.label}>
                            <Box
                              sx={{
                                height: '100%',
                                p: 1.5,
                                ...codePanelSx,
                                borderColor: alpha(group.color, 0.22),
                                bgcolor: alpha(group.color, 0.08),
                              }}
                            >
                              <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700 }}>
                                {group.label}
                              </Typography>
                              <Typography variant="h6" sx={{ fontWeight: 800, mb: 1 }}>
                                {group.value}
                              </Typography>
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                {group.chips.map((chip) => (
                                  <StatusChip
                                    key={chip.key}
                                    status={group.status}
                                    label={chip.label}
                                    icon={chip.icon}
                                    variant="outlined"
                                    statusConfig={CODE_SEARCH_STATUS_CONFIG}
                                  />
                                ))}
                              </Box>
                            </Box>
                          </Grid>
                        ))}
                    </Grid>

                    <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                      Detected blocks
                    </Typography>
                    {detectionResults.code_blocks.map((block, index) => (
                      <Accordion key={`${block.language}-${block.type}-${index}`} defaultExpanded={index === 0}>
                        <AccordionSummary expandIcon={<ExpandIcon />}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
                            {getLanguageIcon(block.language)}
                            <Typography sx={{ fontWeight: 700 }} noWrap>
                              {block.language} / {block.type}
                            </Typography>
                          </Box>
                        </AccordionSummary>
                        <AccordionDetails>
                          <CodeHighlighter
                            language={block.language}
                            variant="vscDarkPlus"
                            code={block.content}
                            customStyle={{
                              margin: 0,
                              borderRadius: '8px',
                              fontSize: 13
                            }}
                          />
                          <Box sx={{ mt: 1, display: 'flex', justifyContent: 'flex-end' }}>
                            <Tooltip title="Copy code">
                              <IconButton size="small" onClick={() => copyCode(block.content)}>
                                <CopyIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </AccordionDetails>
                      </Accordion>
                    ))}
                  </>
                )}
              </SectionCard>
            )}
          </Grid>

          <Grid item xs={12} lg={3.5} sx={{ minHeight: { md: 0 } }}>
            <SectionCard
              title="Detection profile"
              subtitle="Input size and detected code composition."
              icon={<ConfigIcon />}
              accentColor={accentColors.purple}
            >
              <Box sx={{ display: 'grid', gap: 1 }}>
                {[
                  { label: 'Input', value: detectionText.length.toLocaleString(), detail: 'characters', color: accentColors.primary },
                  { label: 'Blocks', value: detectionBlockCount.toLocaleString(), detail: 'detected code blocks', color: accentColors.success },
                  { label: 'Languages', value: detectionLanguageCount.toLocaleString(), detail: 'detected languages', color: accentColors.info },
                  { label: 'Types', value: detectionTypeCount.toLocaleString(), detail: 'detected code types', color: accentColors.warning },
                ].map((stat) => (
                  <MetricTile
                    key={stat.label}
                    label={stat.label}
                    value={stat.value}
                    detail={stat.detail}
                    color={stat.color}
                    variant="horizontal"
                    valueSx={{ fontSize: '1rem' }}
                  />
                ))}
              </Box>
            </SectionCard>
          </Grid>
        </Grid>
      )}
    </PageShell>
  );
};
