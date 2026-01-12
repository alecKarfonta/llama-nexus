/**
 * Reasoning Playground - Interactive interface for exploring GraphRAG reasoning capabilities
 */
import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
  alpha,
  Grid,
  Stepper,
  Step,
  StepLabel,
  StepContent,
} from '@mui/material';
import {
  Psychology as ReasoningIcon,
  CompareArrows as CompareIcon,
  Timeline as CausalIcon,
  AccountTree as MultiHopIcon,
  Link as RelationIcon,
  TrendingUp as ConfidenceIcon,
  Article as SourceIcon,
} from '@mui/icons-material';
import { apiService as api } from '../services/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index} role="tabpanel">
    {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
  </div>
);

const ReasoningPlaygroundPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Multi-hop state
  const [multiHopSource, setMultiHopSource] = useState('');
  const [multiHopTarget, setMultiHopTarget] = useState('');
  const [multiHopResult, setMultiHopResult] = useState<any>(null);

  // Causal reasoning state
  const [causalQuery, setCausalQuery] = useState('');
  const [causalResult, setCausalResult] = useState<any>(null);

  // Comparative state
  const [compareEntity1, setCompareEntity1] = useState('');
  const [compareEntity2, setCompareEntity2] = useState('');
  const [compareResult, setCompareResult] = useState<any>(null);

  // Relationship state
  const [relSource, setRelSource] = useState('');
  const [relTarget, setRelTarget] = useState('');
  const [relResult, setRelResult] = useState<any>(null);

  const handleMultiHopReasoning = async () => {
    if (!multiHopSource || !multiHopTarget) return;

    setLoading(true);
    setError(null);
    setMultiHopResult(null);

    try {
      const result = await api.multiHopReasoningGraphRAG({
        query: `Find connection between ${multiHopSource} and ${multiHopTarget}`,
        max_hops: 5,
      });
      setMultiHopResult(result);
    } catch (err: any) {
      setError(err.message || 'Multi-hop reasoning failed');
    } finally {
      setLoading(false);
    }
  };

  const handleCausalReasoning = async () => {
    if (!causalQuery) return;

    setLoading(true);
    setError(null);
    setCausalResult(null);

    try {
      const result = await api.post('/api/v1/graphrag/reasoning/causal', {
        query: causalQuery,
      });
      setCausalResult(result.data);
    } catch (err: any) {
      setError(err.message || 'Causal reasoning failed');
    } finally {
      setLoading(false);
    }
  };

  const handleComparativeAnalysis = async () => {
    if (!compareEntity1 || !compareEntity2) return;

    setLoading(true);
    setError(null);
    setCompareResult(null);

    try {
      const result = await api.post('/api/v1/graphrag/reasoning/comparative', {
        query: `Compare ${compareEntity1} and ${compareEntity2}`,
      });
      setCompareResult(result.data);
    } catch (err: any) {
      setError(err.message || 'Comparative analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleRelationshipExplanation = async () => {
    if (!relSource || !relTarget) return;

    setLoading(true);
    setError(null);
    setRelResult(null);

    try {
      const result = await api.explainRelationship({
        source: relSource,
        target: relTarget,
        entities: [],
        relationships: [],
      });
      setRelResult(result);
    } catch (err: any) {
      setError(err.message || 'Relationship explanation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight={700} gutterBottom>
          Reasoning Playground
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Explore advanced reasoning capabilities powered by GraphRAG's knowledge graph
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={(_, newValue) => setTabValue(newValue)}
          variant="fullWidth"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab icon={<MultiHopIcon />} label="Multi-Hop Reasoning" />
          <Tab icon={<CausalIcon />} label="Causal Analysis" />
          <Tab icon={<CompareIcon />} label="Comparative" />
          <Tab icon={<RelationIcon />} label="Relationship" />
        </Tabs>

        {/* Multi-Hop Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Multi-Hop Reasoning
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Find paths and connections between two entities in the knowledge graph.
                  </Typography>

                  <TextField
                    fullWidth
                    label="Source Entity"
                    value={multiHopSource}
                    onChange={(e) => setMultiHopSource(e.target.value)}
                    placeholder="e.g., OpenAI"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Target Entity"
                    value={multiHopTarget}
                    onChange={(e) => setMultiHopTarget(e.target.value)}
                    placeholder="e.g., GPT-4"
                    sx={{ mb: 2 }}
                  />

                  <Button
                    variant="contained"
                    fullWidth
                    onClick={handleMultiHopReasoning}
                    disabled={loading || !multiHopSource || !multiHopTarget}
                    startIcon={loading ? <CircularProgress size={20} /> : <MultiHopIcon />}
                  >
                    {loading ? 'Reasoning...' : 'Find Connection'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              {multiHopResult && (
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Reasoning Path
                    </Typography>

                    {multiHopResult.reasoning_path && multiHopResult.reasoning_path.length > 0 ? (
                      <Stepper orientation="vertical">
                        {multiHopResult.reasoning_path.map((hop: any, idx: number) => (
                          <Step key={idx} active>
                            <StepLabel>
                              <Typography variant="body2">
                                {hop.entity || `Hop ${hop.hop}`}
                              </Typography>
                            </StepLabel>
                            <StepContent>
                              <Typography variant="caption" color="text.secondary">
                                {hop.relation || 'related to'}
                              </Typography>
                              {hop.confidence && (
                                <Chip
                                  label={`${(hop.confidence * 100).toFixed(0)}%`}
                                  size="small"
                                  sx={{ ml: 1 }}
                                />
                              )}
                            </StepContent>
                          </Step>
                        ))}
                      </Stepper>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        {multiHopResult.answer || 'No path found'}
                      </Typography>
                    )}

                    {multiHopResult.sources && multiHopResult.sources.length > 0 && (
                      <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                        <Typography variant="caption" color="text.secondary" gutterBottom>
                          Sources ({multiHopResult.sources.length})
                        </Typography>
                        <List dense>
                          {multiHopResult.sources.slice(0, 3).map((source: any, idx: number) => (
                            <ListItem key={idx}>
                              <ListItemText
                                primary={source.source || source.content?.substring(0, 60)}
                                secondary={`Score: ${((source.score || 0) * 100).toFixed(0)}%`}
                                primaryTypographyProps={{ variant: 'caption' }}
                                secondaryTypographyProps={{ variant: 'caption' }}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>

        {/* Causal Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Causal Reasoning
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Identify cause-effect relationships for a given query.
                  </Typography>

                  <TextField
                    fullWidth
                    label="Query"
                    value={causalQuery}
                    onChange={(e) => setCausalQuery(e.target.value)}
                    placeholder="e.g., What causes system failures?"
                    multiline
                    rows={3}
                    sx={{ mb: 2 }}
                  />

                  <Button
                    variant="contained"
                    fullWidth
                    onClick={handleCausalReasoning}
                    disabled={loading || !causalQuery}
                    startIcon={loading ? <CircularProgress size={20} /> : <CausalIcon />}
                  >
                    {loading ? 'Analyzing...' : 'Analyze Causality'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              {causalResult && (
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Analysis Results
                    </Typography>

                    {causalResult.answer && (
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          mb: 2,
                          bgcolor: alpha('#10b981', 0.05),
                          border: 1,
                          borderColor: alpha('#10b981', 0.2),
                        }}
                      >
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                          {causalResult.answer}
                        </Typography>
                      </Paper>
                    )}

                    {causalResult.causes && causalResult.causes.length > 0 && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Causes
                        </Typography>
                        <List dense>
                          {causalResult.causes.map((cause: string, idx: number) => (
                            <ListItem key={idx}>
                              <ListItemText primary={cause} primaryTypographyProps={{ variant: 'body2' }} />
                            </ListItem>
                          ))}
                        </List>
                      </Box>
                    )}

                    {causalResult.effects && causalResult.effects.length > 0 && (
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Effects
                        </Typography>
                        <List dense>
                          {causalResult.effects.map((effect: string, idx: number) => (
                            <ListItem key={idx}>
                              <ListItemText primary={effect} primaryTypographyProps={{ variant: 'body2' }} />
                            </ListItem>
                          ))}
                        </List>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>

        {/* Comparative Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Comparative Analysis
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Compare two entities and find their similarities and differences.
                  </Typography>

                  <TextField
                    fullWidth
                    label="First Entity"
                    value={compareEntity1}
                    onChange={(e) => setCompareEntity1(e.target.value)}
                    placeholder="e.g., Python"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Second Entity"
                    value={compareEntity2}
                    onChange={(e) => setCompareEntity2(e.target.value)}
                    placeholder="e.g., JavaScript"
                    sx={{ mb: 2 }}
                  />

                  <Button
                    variant="contained"
                    fullWidth
                    onClick={handleComparativeAnalysis}
                    disabled={loading || !compareEntity1 || !compareEntity2}
                    startIcon={loading ? <CircularProgress size={20} /> : <CompareIcon />}
                  >
                    {loading ? 'Comparing...' : 'Compare'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              {compareResult && (
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Comparison Results
                    </Typography>

                    {compareResult.answer && (
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          mb: 2,
                          bgcolor: alpha('#6366f1', 0.05),
                          border: 1,
                          borderColor: alpha('#6366f1', 0.2),
                        }}
                      >
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                          {compareResult.answer}
                        </Typography>
                      </Paper>
                    )}

                    <Grid container spacing={2}>
                      {compareResult.similarities && compareResult.similarities.length > 0 && (
                        <Grid item xs={12}>
                          <Typography variant="subtitle2" gutterBottom sx={{ color: '#10b981' }}>
                            Similarities
                          </Typography>
                          <List dense>
                            {compareResult.similarities.map((sim: string, idx: number) => (
                              <ListItem key={idx}>
                                <ListItemText primary={sim} primaryTypographyProps={{ variant: 'body2' }} />
                              </ListItem>
                            ))}
                          </List>
                        </Grid>
                      )}

                      {compareResult.differences && compareResult.differences.length > 0 && (
                        <Grid item xs={12}>
                          <Divider sx={{ my: 1 }} />
                          <Typography variant="subtitle2" gutterBottom sx={{ color: '#f59e0b' }}>
                            Differences
                          </Typography>
                          <List dense>
                            {compareResult.differences.map((diff: string, idx: number) => (
                              <ListItem key={idx}>
                                <ListItemText primary={diff} primaryTypographyProps={{ variant: 'body2' }} />
                              </ListItem>
                            ))}
                          </List>
                        </Grid>
                      )}
                    </Grid>
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>

        {/* Relationship Tab */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Explain Relationship
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Get detailed explanation of the relationship between two entities.
                  </Typography>

                  <TextField
                    fullWidth
                    label="Source Entity"
                    value={relSource}
                    onChange={(e) => setRelSource(e.target.value)}
                    placeholder="e.g., Machine Learning"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Target Entity"
                    value={relTarget}
                    onChange={(e) => setRelTarget(e.target.value)}
                    placeholder="e.g., Neural Networks"
                    sx={{ mb: 2 }}
                  />

                  <Button
                    variant="contained"
                    fullWidth
                    onClick={handleRelationshipExplanation}
                    disabled={loading || !relSource || !relTarget}
                    startIcon={loading ? <CircularProgress size={20} /> : <RelationIcon />}
                  >
                    {loading ? 'Analyzing...' : 'Explain Relationship'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              {relResult && (
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Relationship Explanation
                    </Typography>

                    {relResult.explanation && (
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          mb: 2,
                          bgcolor: alpha('#8b5cf6', 0.05),
                          border: 1,
                          borderColor: alpha('#8b5cf6', 0.2),
                        }}
                      >
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                          {relResult.explanation}
                        </Typography>
                      </Paper>
                    )}

                    {relResult.confidence && (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                        <ConfidenceIcon fontSize="small" sx={{ color: '#10b981' }} />
                        <Typography variant="body2" color="text.secondary">
                          Confidence: {(relResult.confidence * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    )}

                    {relResult.paths && relResult.paths.length > 0 && (
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Connection Paths
                        </Typography>
                        <List dense>
                          {relResult.paths.map((path: string[], idx: number) => (
                            <ListItem key={idx}>
                              <ListItemText
                                primary={Array.isArray(path) ? path.join(' → ') : String(path)}
                                primaryTypographyProps={{ variant: 'body2', fontFamily: 'monospace' }}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>

      {/* Info Card */}
      <Paper
        elevation={0}
        sx={{
          p: 2,
          bgcolor: alpha('#6366f1', 0.05),
          border: 1,
          borderColor: alpha('#6366f1', 0.1),
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
          <ReasoningIcon fontSize="small" sx={{ color: '#6366f1', mt: 0.5 }} />
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              About Reasoning Playground
            </Typography>
            <Typography variant="caption" color="text.secondary">
              This tool uses GraphRAG's advanced reasoning capabilities to explore relationships
              in your knowledge graph. Upload documents to the Documents page first to build your
              knowledge base, then use these reasoning tools to discover insights and connections.
            </Typography>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default ReasoningPlaygroundPage;

