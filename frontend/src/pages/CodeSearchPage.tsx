import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
  Alert,
  CircularProgress,
  IconButton,
  Tabs,
  Tab,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Collapse,
  FormControlLabel,
  Checkbox,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Search as SearchIcon,
  Code as CodeIcon,
  Description as FileIcon,
  Functions as FunctionIcon,
  Class as ClassIcon,
  Api as ApiIcon,
  Terminal as TerminalIcon,
  DataObject as JsonIcon,
  Settings as ConfigIcon,
  ExpandMore as ExpandIcon,
  ContentCopy as CopyIcon,
  OpenInNew as OpenIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
  TipsAndUpdates as SuggestionIcon,
  Analytics as AnalyticsIcon,
  GitHub as GitHubIcon,
  Javascript as JavaScriptIcon,
  Storage as DatabaseIcon,
} from '@mui/icons-material';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import * as api from '../services/api';

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
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  
  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchType, setSearchType] = useState<'semantic' | 'pattern' | 'ast'>('semantic');
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
  
  // Available filters
  const availableLanguages = [
    'python', 'javascript', 'typescript', 'java', 'go', 'rust', 
    'cpp', 'c', 'csharp', 'php', 'ruby', 'swift'
  ];
  
  const availableCodeTypes = [
    'function', 'class', 'method', 'variable', 'import', 'config'
  ];

  useEffect(() => {
    checkCodeServiceStatus();
  }, []);

  const checkCodeServiceStatus = async () => {
    try {
      const response = await fetch('/api/v1/graphrag/code/health');
      const data = await response.json();
      setCodeServiceAvailable(data.available === true);
      if (!data.available) {
        setError('Code search service is not available. Please ensure the code-rag service is running.');
      }
    } catch (err) {
      setCodeServiceAvailable(false);
      setError('Unable to connect to code search service');
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
    navigator.clipboard.writeText(content);
  };

  const getLanguageIcon = (language: string) => {
    switch (language.toLowerCase()) {
      case 'python':
        return <CodeIcon sx={{ color: '#3776AB' }} />;
      case 'javascript':
      case 'typescript':
        return <JavaScriptIcon sx={{ color: '#F7DF1E' }} />;
      case 'java':
        return <CodeIcon sx={{ color: '#007396' }} />;
      case 'go':
        return <CodeIcon sx={{ color: '#00ADD8' }} />;
      case 'rust':
        return <CodeIcon sx={{ color: '#CE4A1F' }} />;
      case 'sql':
        return <DatabaseIcon sx={{ color: '#4479A1' }} />;
      default:
        return <CodeIcon />;
    }
  };

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

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        <CodeIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
        Code Intelligence
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Search and analyze code across your knowledge base with semantic understanding
      </Typography>

      {/* Service Status Alert */}
      {!codeServiceAvailable && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Code search service is currently unavailable. Some features may be limited.
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)} sx={{ mb: 3 }}>
        <Tab label="Code Search" icon={<SearchIcon />} iconPosition="start" />
        <Tab label="Code Detection" icon={<AnalyticsIcon />} iconPosition="start" />
        <Tab label="Repository Analysis" icon={<GitHubIcon />} iconPosition="start" disabled />
      </Tabs>

      {/* Code Search Tab */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={3}>
            {/* Search Filters */}
            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                <FilterIcon sx={{ fontSize: 16, mr: 0.5, verticalAlign: 'middle' }} />
                Search Filters
              </Typography>
              <Divider sx={{ my: 1 }} />
              
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Search Type</InputLabel>
                <Select
                  value={searchType}
                  onChange={(e) => setSearchType(e.target.value as any)}
                  label="Search Type"
                >
                  <MenuItem value="semantic">Semantic (AI-powered)</MenuItem>
                  <MenuItem value="pattern">Pattern Matching</MenuItem>
                  <MenuItem value="ast">AST-based</MenuItem>
                </Select>
              </FormControl>

              <Typography variant="caption" gutterBottom display="block">
                Languages
              </Typography>
              <Box sx={{ mb: 2 }}>
                {availableLanguages.map(lang => (
                  <Chip
                    key={lang}
                    label={lang}
                    size="small"
                    onClick={() => {
                      setSelectedLanguages(prev => 
                        prev.includes(lang) 
                          ? prev.filter(l => l !== lang)
                          : [...prev, lang]
                      );
                    }}
                    color={selectedLanguages.includes(lang) ? 'primary' : 'default'}
                    sx={{ m: 0.25 }}
                  />
                ))}
              </Box>

              <Typography variant="caption" gutterBottom display="block">
                Code Types
              </Typography>
              <Box>
                {availableCodeTypes.map(type => (
                  <Chip
                    key={type}
                    label={type}
                    size="small"
                    icon={getCodeTypeIcon(type)}
                    onClick={() => {
                      setSelectedCodeTypes(prev => 
                        prev.includes(type) 
                          ? prev.filter(t => t !== type)
                          : [...prev, type]
                      );
                    }}
                    color={selectedCodeTypes.includes(type) ? 'primary' : 'default'}
                    sx={{ m: 0.25 }}
                  />
                ))}
              </Box>
            </Paper>

            {/* Search Tips */}
            <Paper sx={{ p: 2, bgcolor: alpha(theme.palette.info.main, 0.05) }}>
              <Typography variant="subtitle2" gutterBottom>
                <SuggestionIcon sx={{ fontSize: 16, mr: 0.5, verticalAlign: 'middle' }} />
                Search Tips
              </Typography>
              <Typography variant="caption" component="div" color="text.secondary">
                • Use natural language for semantic search<br/>
                • Include function names or signatures<br/>
                • Specify programming patterns<br/>
                • Describe the functionality you need
              </Typography>
            </Paper>
          </Grid>

          <Grid item xs={12} md={9}>
            {/* Search Input */}
            <Paper sx={{ p: 2, mb: 2 }}>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  placeholder="Search for code... (e.g., 'authentication middleware', 'database connection pool', 'React hooks for data fetching')"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  disabled={searching || !codeServiceAvailable}
                  InputProps={{
                    startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
                  }}
                />
                <Button
                  variant="contained"
                  onClick={handleSearch}
                  disabled={searching || !searchQuery.trim() || !codeServiceAvailable}
                  sx={{ minWidth: 120 }}
                >
                  {searching ? <CircularProgress size={20} /> : 'Search'}
                </Button>
              </Box>
            </Paper>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Found {searchResults.length} code matches
                </Typography>
                {searchResults.map((result, index) => (
                  <Card key={index} sx={{ mb: 2 }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        {getLanguageIcon(result.language)}
                        <Typography variant="subtitle2" sx={{ ml: 1, flexGrow: 1 }}>
                          {result.file_path}
                        </Typography>
                        <Chip 
                          label={result.code_type} 
                          size="small" 
                          icon={getCodeTypeIcon(result.code_type)}
                        />
                        <Chip 
                          label={`Score: ${result.score.toFixed(2)}`} 
                          size="small" 
                          sx={{ ml: 1 }}
                          color={result.score > 0.8 ? 'success' : 'default'}
                        />
                      </Box>

                      {result.name && (
                        <Typography variant="h6" sx={{ mb: 1 }}>
                          {result.name}
                        </Typography>
                      )}

                      <Box sx={{ position: 'relative' }}>
                        <SyntaxHighlighter
                          language={result.language}
                          style={vscDarkPlus}
                          customStyle={{
                            margin: 0,
                            borderRadius: 4,
                            fontSize: 13,
                            maxHeight: expandedResults.has(index) ? 'none' : '200px',
                            overflow: 'hidden'
                          }}
                          showLineNumbers
                          startingLineNumber={result.line_start}
                        >
                          {result.content}
                        </SyntaxHighlighter>
                        
                        {result.content.split('\n').length > 8 && (
                          <Button
                            size="small"
                            onClick={() => toggleResultExpansion(index)}
                            sx={{ mt: 1 }}
                            startIcon={<ExpandIcon sx={{ 
                              transform: expandedResults.has(index) ? 'rotate(180deg)' : 'none',
                              transition: 'transform 0.3s'
                            }} />}
                          >
                            {expandedResults.has(index) ? 'Show Less' : 'Show More'}
                          </Button>
                        )}
                      </Box>

                      {result.dependencies && result.dependencies.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="caption" color="text.secondary">
                            Dependencies:
                          </Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                            {result.dependencies.map((dep, i) => (
                              <Chip key={i} label={dep} size="small" variant="outlined" />
                            ))}
                          </Box>
                        </Box>
                      )}
                    </CardContent>
                    <CardActions>
                      <IconButton size="small" onClick={() => copyCode(result.content)}>
                        <Tooltip title="Copy Code">
                          <CopyIcon fontSize="small" />
                        </Tooltip>
                      </IconButton>
                      <IconButton size="small">
                        <Tooltip title="Open in Editor">
                          <OpenIcon fontSize="small" />
                        </Tooltip>
                      </IconButton>
                    </CardActions>
                  </Card>
                ))}
              </Box>
            )}
          </Grid>
        </Grid>
      )}

      {/* Code Detection Tab */}
      {activeTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Paste text to detect and analyze code blocks
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={10}
                placeholder="Paste your text here... The system will automatically detect and classify any code blocks."
                value={detectionText}
                onChange={(e) => setDetectionText(e.target.value)}
                sx={{ mb: 2 }}
              />
              <Button
                variant="contained"
                onClick={handleDetection}
                disabled={detecting || !detectionText.trim() || !codeServiceAvailable}
                startIcon={detecting ? <CircularProgress size={20} /> : <AnalyticsIcon />}
              >
                {detecting ? 'Analyzing...' : 'Detect Code'}
              </Button>
            </Paper>

            {/* Detection Results */}
            {detectionResults && (
              <Paper sx={{ p: 2, mt: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Detection Results
                </Typography>
                
                <Alert severity={detectionResults.has_code ? 'success' : 'info'} sx={{ mb: 2 }}>
                  {detectionResults.has_code 
                    ? `Found ${detectionResults.statistics.total_blocks} code block(s)`
                    : 'No code blocks detected in the text'}
                </Alert>

                {detectionResults.has_code && (
                  <>
                    {/* Statistics */}
                    <Grid container spacing={2} sx={{ mb: 3 }}>
                      <Grid item xs={6}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="subtitle2" color="text.secondary">
                              Languages Detected
                            </Typography>
                            <Box sx={{ mt: 1 }}>
                              {Object.entries(detectionResults.statistics.by_language).map(([lang, count]) => (
                                <Chip 
                                  key={lang}
                                  label={`${lang} (${count})`}
                                  size="small"
                                  icon={getLanguageIcon(lang)}
                                  sx={{ mr: 0.5, mb: 0.5 }}
                                />
                              ))}
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                      <Grid item xs={6}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="subtitle2" color="text.secondary">
                              Code Types
                            </Typography>
                            <Box sx={{ mt: 1 }}>
                              {Object.entries(detectionResults.statistics.by_type).map(([type, count]) => (
                                <Chip 
                                  key={type}
                                  label={`${type} (${count})`}
                                  size="small"
                                  icon={getCodeTypeIcon(type)}
                                  sx={{ mr: 0.5, mb: 0.5 }}
                                />
                              ))}
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                    </Grid>

                    {/* Code Blocks */}
                    <Typography variant="subtitle2" gutterBottom>
                      Detected Code Blocks
                    </Typography>
                    {detectionResults.code_blocks.map((block, index) => (
                      <Accordion key={index} defaultExpanded={index === 0}>
                        <AccordionSummary expandIcon={<ExpandIcon />}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {getLanguageIcon(block.language)}
                            <Typography>
                              {block.language} - {block.type}
                            </Typography>
                          </Box>
                        </AccordionSummary>
                        <AccordionDetails>
                          <SyntaxHighlighter
                            language={block.language}
                            style={vscDarkPlus}
                            customStyle={{
                              margin: 0,
                              borderRadius: 4,
                              fontSize: 13
                            }}
                          >
                            {block.content}
                          </SyntaxHighlighter>
                          <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                            <IconButton size="small" onClick={() => copyCode(block.content)}>
                              <CopyIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        </AccordionDetails>
                      </Accordion>
                    ))}
                  </>
                )}
              </Paper>
            )}
          </Grid>

          <Grid item xs={12} md={4}>
            {/* Info Panel */}
            <Paper sx={{ p: 2, bgcolor: alpha(theme.palette.info.main, 0.05) }}>
              <Typography variant="subtitle2" gutterBottom>
                About Code Detection
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Our AI-powered code detection can:
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon><CodeIcon fontSize="small" /></ListItemIcon>
                  <ListItemText primary="Identify code in natural text" />
                </ListItem>
                <ListItem>
                  <ListItemIcon><ApiIcon fontSize="small" /></ListItemIcon>
                  <ListItemText primary="Classify code by type and purpose" />
                </ListItem>
                <ListItem>
                  <ListItemIcon><FunctionIcon fontSize="small" /></ListItemIcon>
                  <ListItemText primary="Extract function signatures" />
                </ListItem>
                <ListItem>
                  <ListItemIcon><DatabaseIcon fontSize="small" /></ListItemIcon>
                  <ListItemText primary="Detect SQL queries and schemas" />
                </ListItem>
              </List>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};
