/**
 * Reasoning Playground - Interactive interface for exploring GraphRAG reasoning capabilities
 */
import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  IconButton,
  Chip,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  Divider,
  Grid,
  Stack,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Tooltip,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  CompareArrows as CompareIcon,
  Timeline as CausalIcon,
  AccountTree as MultiHopIcon,
  Link as RelationIcon,
  TrendingUp as ConfidenceIcon,
  Article as SourceIcon,
} from '@mui/icons-material';
import {
  PageShell,
  SectionCard,
  StatusChip,
  type StatusChipConfigMap,
} from '@/components/common';
import {
  formatGraphRAGReasoningLocation,
  normalizeGraphRAGReasoningResult,
} from '@/utils/graphragReasoningResults';
import { accentColors } from '@/utils/theme';
import { apiService as api } from '../services/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <Box
    hidden={value !== index}
    role="tabpanel"
    sx={{
      display: value === index ? 'block' : 'none',
    }}
  >
    {value === index && (
      <Box
        data-reasoning-workspace
        sx={{
          '& > .MuiGrid-container': {
            alignItems: 'flex-start',
          },
          '& > .MuiGrid-container > .MuiGrid-item > .MuiCard-root': {
            width: '100%',
          },
        }}
      >
        {children}
      </Box>
    )}
  </Box>
);

const REASONING_STATUS_CONFIG: StatusChipConfigMap = {
  graphrag: { color: accentColors.rose, label: 'Graph reasoning' },
  bridge: { color: accentColors.info, label: 'Bridge' },
  'multi-hop': { color: accentColors.primary, label: 'Multi-hop' },
  causal: { color: accentColors.success, label: 'Causal' },
  comparative: { color: accentColors.warning, label: 'Comparative' },
  relationship: { color: accentColors.purple, label: 'Relationship' },
  evaluation: { color: accentColors.purple, label: 'Evaluation' },
  local_pipeline: { color: accentColors.info, label: 'Local trace' },
  remote_graph: { color: accentColors.rose, label: 'Remote graph' },
  online: { color: accentColors.success, label: 'Online' },
  offline: { color: accentColors.rose, label: 'Offline' },
  ready: { color: accentColors.success, label: 'Ready' },
  waiting: { color: accentColors.warning, label: 'Waiting' },
};

const answerSurfaceSx = (color: string) => ({
  p: 2,
  mb: 2,
  borderRadius: '8px',
  bgcolor: alpha(color, 0.06),
  border: `1px solid ${alpha(color, 0.2)}`,
});

const actionIconButtonSx = (color: string) => ({
  width: 38,
  height: 38,
  borderRadius: '8px',
  border: '1px solid',
  borderColor: alpha(color, 0.35),
  color: 'primary.contrastText',
  bgcolor: alpha(color, 0.82),
  '&:hover': { bgcolor: color },
  '&.Mui-disabled': {
    color: alpha(color, 0.32),
    borderColor: alpha(color, 0.16),
    bgcolor: alpha(color, 0.08),
  },
});

const formatPercent = (value: unknown): string | null => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
};

const formatCount = (value: unknown): number => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
};

const compactText = (value: unknown, fallback = ''): string => {
  if (typeof value === 'string') return value.replace(/\s+/g, ' ').trim() || fallback;
  if (value === null || value === undefined) return fallback;
  return String(value).replace(/\s+/g, ' ').trim() || fallback;
};

const ReasoningEvidenceSummary: React.FC<{
  result: any;
  color: string;
  maxItems?: number;
}> = ({ result, color, maxItems = 4 }) => {
  const normalized = normalizeGraphRAGReasoningResult(result);
  const localPipeline = normalized.localPipeline || {};
  const runCount = formatCount(localPipeline.run_count);
  const graphEdgeCount = formatCount(localPipeline.graph_edge_count);
  const pathCount = normalized.paths.length || formatCount(localPipeline.matched_path_count);
  const relationshipCount = normalized.relationships.length || formatCount(localPipeline.matched_relationship_count);
  const evidenceCount = normalized.evidence.length;
  const bridgeCount = normalized.bridgeEntities.length || formatCount(normalized.graphOverview?.bridge_entity_count);
  const sourceStatus = normalized.reasoningSource === 'local_pipeline' ? 'local_pipeline' : 'remote_graph';

  if (
    normalized.warnings.length === 0
    && normalized.hops.length === 0
    && normalized.paths.length === 0
    && normalized.relationships.length === 0
    && normalized.evidence.length === 0
    && !normalized.localPipeline
  ) {
    return null;
  }

  return (
    <Stack
      spacing={1.25}
      sx={{
        borderTop: '1px solid',
        borderColor: alpha(color, 0.18),
        pt: 1,
        mt: 1,
      }}
    >
      {normalized.warnings.map((warning, index) => (
        <Alert key={`${warning}-${index}`} severity="warning" sx={{ py: 0.5 }}>
          {warning}
        </Alert>
      ))}

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
        <StatusChip
          status={sourceStatus}
          label={normalized.reasoningSource === 'local_pipeline' ? 'Local pipeline evidence' : 'Remote graph evidence'}
          statusConfig={REASONING_STATUS_CONFIG}
          height={22}
        />
        {normalized.reasoningTask && (
          <StatusChip
            status={normalized.reasoningTask}
            label={normalized.reasoningTask.replace(/_/g, ' ')}
            statusConfig={REASONING_STATUS_CONFIG}
            height={22}
          />
        )}
        {runCount > 0 && <Chip label={`${runCount} run${runCount === 1 ? '' : 's'}`} size="small" variant="outlined" sx={{ height: 22 }} />}
        {pathCount > 0 && <Chip label={`${pathCount} path${pathCount === 1 ? '' : 's'}`} size="small" variant="outlined" sx={{ height: 22 }} />}
        {relationshipCount > 0 && <Chip label={`${relationshipCount} relationship${relationshipCount === 1 ? '' : 's'}`} size="small" variant="outlined" sx={{ height: 22 }} />}
        {bridgeCount > 0 && <Chip label={`${bridgeCount} bridge${bridgeCount === 1 ? '' : 's'}`} size="small" variant="outlined" sx={{ height: 22 }} />}
        {graphEdgeCount > 0 && <Chip label={`${graphEdgeCount} graph edges`} size="small" variant="outlined" sx={{ height: 22 }} />}
        {evidenceCount > 0 && <Chip label={`${evidenceCount} evidence item${evidenceCount === 1 ? '' : 's'}`} size="small" variant="outlined" sx={{ height: 22 }} />}
      </Box>

      {normalized.hops.length > 0 && (
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Reasoning hops
          </Typography>
          <Stepper orientation="vertical">
            {normalized.hops.slice(0, maxItems + 2).map((hop, index) => {
              const confidence = formatPercent(hop.confidence);
              const location = formatGraphRAGReasoningLocation(hop.metadata);
              return (
                <Step key={`${hop.text}-${index}`} active>
                  <StepLabel>
                    <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>
                      {hop.source && hop.target ? `${hop.source} -> ${hop.target}` : hop.entity}
                    </Typography>
                  </StepLabel>
                  <StepContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 0.25 }}>
                      <Chip label={hop.relation} size="small" variant="outlined" sx={{ height: 20, fontSize: '0.65rem' }} />
                      {confidence && <Chip label={confidence} size="small" color="success" variant="outlined" sx={{ height: 20, fontSize: '0.65rem' }} />}
                    </Box>
                    {(hop.evidence || hop.provenance || location) && (
                      <Typography variant="caption" color="text.secondary" component="div" sx={{ overflowWrap: 'anywhere' }}>
                        {[hop.evidence, hop.provenance || location].filter(Boolean).join(' | ')}
                      </Typography>
                    )}
                  </StepContent>
                </Step>
              );
            })}
          </Stepper>
        </Box>
      )}

      {normalized.paths.length > 0 && (
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Graph paths
          </Typography>
          <List dense disablePadding>
            {normalized.paths.slice(0, maxItems).map((path, index) => {
              const score = formatPercent(path.score);
              return (
                <ListItem key={`${path.text}-${index}`} disableGutters sx={{ alignItems: 'flex-start' }}>
                  <ListItemText
                    primary={path.text}
                    secondary={[
                      path.pathType ? `${path.pathType} path` : '',
                      path.hops !== undefined ? `${path.hops} hop${path.hops === 1 ? '' : 's'}` : '',
                      score ? `score ${score}` : '',
                    ].filter(Boolean).join(' | ')}
                    primaryTypographyProps={{ variant: 'body2', sx: { overflowWrap: 'anywhere' } }}
                    secondaryTypographyProps={{ variant: 'caption' }}
                  />
                </ListItem>
              );
            })}
          </List>
        </Box>
      )}

      {normalized.relationships.length > 0 && (
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Direct relationships
          </Typography>
          <List dense disablePadding>
            {normalized.relationships.slice(0, maxItems).map((relationship, index) => {
              const score = formatPercent(relationship.score ?? relationship.weight);
              const location = formatGraphRAGReasoningLocation(relationship.metadata);
              return (
                <ListItem key={`${relationship.text}-${index}`} disableGutters sx={{ alignItems: 'flex-start' }}>
                  <ListItemText
                    primary={relationship.text}
                    secondary={[
                      score ? `score ${score}` : '',
                      relationship.provenance || location,
                    ].filter(Boolean).join(' | ')}
                    primaryTypographyProps={{ variant: 'body2', sx: { overflowWrap: 'anywhere' } }}
                    secondaryTypographyProps={{ variant: 'caption' }}
                  />
                </ListItem>
              );
            })}
          </List>
        </Box>
      )}

      {normalized.evidence.length > 0 && (
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Evidence
          </Typography>
          <List dense disablePadding>
            {normalized.evidence.slice(0, maxItems).map((item, index) => {
              const score = formatPercent(item.score);
              const location = formatGraphRAGReasoningLocation(item.metadata);
              return (
                <ListItem key={`${item.content}-${index}`} disableGutters sx={{ alignItems: 'flex-start' }}>
                  <ListItemText
                    primary={compactText(item.content)}
                    secondary={[
                      item.source,
                      item.sourceType?.replace(/_/g, ' '),
                      score ? `score ${score}` : '',
                      location,
                    ].filter(Boolean).join(' | ')}
                    primaryTypographyProps={{ variant: 'body2', sx: { overflowWrap: 'anywhere' } }}
                    secondaryTypographyProps={{ variant: 'caption', sx: { overflowWrap: 'anywhere' } }}
                  />
                </ListItem>
              );
            })}
          </List>
        </Box>
      )}
    </Stack>
  );
};

const ReasoningPlaygroundPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<any>(null);
  const [evaluationStatus, setEvaluationStatus] = useState<any>(null);
  const [loadingCapabilities, setLoadingCapabilities] = useState(false);
  const [capabilityError, setCapabilityError] = useState<string | null>(null);

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

  useEffect(() => {
    let mounted = true;

    const loadCapabilities = async () => {
      setLoadingCapabilities(true);
      setCapabilityError(null);

      try {
        const [capabilityResult, evaluationResult] = await Promise.all([
          api.getGraphRAGCapabilities(),
          api.getGraphRAGEvaluationStatus(),
        ]);

        if (!mounted) return;
        setCapabilities(capabilityResult);
        setEvaluationStatus(evaluationResult);
      } catch (err: any) {
        if (!mounted) return;
        setCapabilityError(err.message || 'Unable to load GraphRAG capabilities');
      } finally {
        if (mounted) {
          setLoadingCapabilities(false);
        }
      }
    };

    loadCapabilities();

    return () => {
      mounted = false;
    };
  }, []);

  const canRunMultiHop = multiHopSource.trim().length > 0 && multiHopTarget.trim().length > 0;
  const canRunCausal = causalQuery.trim().length > 0;
  const canRunComparative = compareEntity1.trim().length > 0 && compareEntity2.trim().length > 0;
  const canRunRelationship = relSource.trim().length > 0 && relTarget.trim().length > 0;
  const showMultiHopAction = canRunMultiHop || loading;
  const showCausalAction = canRunCausal || loading;
  const showComparativeAction = canRunComparative || loading;
  const showRelationshipAction = canRunRelationship || loading;

  const handleMultiHopReasoning = async () => {
    if (!canRunMultiHop) return;

    setLoading(true);
    setError(null);
    setMultiHopResult(null);

    try {
      const result = await api.multiHopReasoningGraphRAG({
        query: `Find connection between ${multiHopSource.trim()} and ${multiHopTarget.trim()}`,
        source: multiHopSource.trim(),
        target: multiHopTarget.trim(),
        max_hops: 5,
        include_local_pipeline: true,
      });
      setMultiHopResult(result);
    } catch (err: any) {
      setError(err.message || 'Multi-hop reasoning failed');
    } finally {
      setLoading(false);
    }
  };

  const handleCausalReasoning = async () => {
    if (!canRunCausal) return;

    setLoading(true);
    setError(null);
    setCausalResult(null);

    try {
      const result = await api.post('/api/v1/graphrag/reasoning/causal', {
        query: causalQuery.trim(),
        include_local_pipeline: true,
      });
      setCausalResult(result.data);
    } catch (err: any) {
      setError(err.message || 'Causal reasoning failed');
    } finally {
      setLoading(false);
    }
  };

  const handleComparativeAnalysis = async () => {
    if (!canRunComparative) return;

    setLoading(true);
    setError(null);
    setCompareResult(null);

    try {
      const result = await api.post('/api/v1/graphrag/reasoning/comparative', {
        query: `Compare ${compareEntity1.trim()} and ${compareEntity2.trim()}`,
        source: compareEntity1.trim(),
        target: compareEntity2.trim(),
        include_local_pipeline: true,
      });
      setCompareResult(result.data);
    } catch (err: any) {
      setError(err.message || 'Comparative analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleRelationshipExplanation = async () => {
    if (!canRunRelationship) return;

    setLoading(true);
    setError(null);
    setRelResult(null);

    try {
      const result = await api.explainRelationship({
        source: relSource.trim(),
        target: relTarget.trim(),
        query: `Explain relationship between ${relSource.trim()} and ${relTarget.trim()}`,
        include_local_pipeline: true,
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

  const reasoningModes = [
    {
      label: 'Multi-hop',
      detail: 'Path finding',
      icon: <MultiHopIcon />,
      color: accentColors.primary,
      status: 'multi-hop',
      ready: canRunMultiHop,
      result: multiHopResult,
      action: handleMultiHopReasoning,
      actionLabel: 'Find Connection',
    },
    {
      label: 'Causal',
      detail: 'Cause and effect',
      icon: <CausalIcon />,
      color: accentColors.success,
      status: 'causal',
      ready: canRunCausal,
      result: causalResult,
      action: handleCausalReasoning,
      actionLabel: 'Analyze Causality',
    },
    {
      label: 'Comparative',
      detail: 'Similarity and difference',
      icon: <CompareIcon />,
      color: accentColors.warning,
      status: 'comparative',
      ready: canRunComparative,
      result: compareResult,
      action: handleComparativeAnalysis,
      actionLabel: 'Compare',
    },
    {
      label: 'Relationship',
      detail: 'Entity explanation',
      icon: <RelationIcon />,
      color: accentColors.purple,
      status: 'relationship',
      ready: canRunRelationship,
      result: relResult,
      action: handleRelationshipExplanation,
      actionLabel: 'Explain Relationship',
    },
  ];
  const activeMode = reasoningModes[tabValue];
  const activeSourceCount = activeMode.result
    ? normalizeGraphRAGReasoningResult(activeMode.result).evidence.length
    : 0;
  const reasoningHeaderActions = (
    <Box
      sx={{
        display: { xs: 'grid', sm: 'flex' },
        gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', sm: 'none' },
        alignItems: 'center',
        gap: 0.5,
        flexWrap: 'wrap',
        justifyContent: 'flex-end',
        width: { xs: '100%', sm: 'auto' },
      }}
    >
      {reasoningModes.map((mode, index) => (
        <Button
          key={mode.status}
          size="small"
          variant={tabValue === index ? 'contained' : 'outlined'}
          startIcon={mode.icon}
          onClick={() => setTabValue(index)}
          sx={{
            height: { xs: 30, sm: 34 },
            minWidth: 0,
            borderRadius: '8px',
            fontSize: { xs: '0.75rem', sm: '0.8125rem' },
            fontWeight: 800,
            px: { xs: 0.75, sm: 1.1 },
            '& .MuiButton-startIcon': {
              display: { xs: 'none', sm: 'inherit' },
              mr: { sm: 0.5 },
            },
          }}
        >
          {mode.label}
        </Button>
      ))}
    </Box>
  );
  const reasoningSummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      <StatusChip status="graphrag" label="Graph reasoning" statusConfig={REASONING_STATUS_CONFIG} height={22} />
      <StatusChip status={activeMode.status} label={activeMode.label} statusConfig={REASONING_STATUS_CONFIG} height={22} />
      <StatusChip
        status={activeMode.ready ? 'ready' : 'waiting'}
        label={activeMode.ready ? 'Inputs ready' : 'Needs input'}
        statusConfig={REASONING_STATUS_CONFIG}
        height={22}
      />
      {activeMode.result && (
        <StatusChip
          status="ready"
          label="Result ready"
          statusConfig={REASONING_STATUS_CONFIG}
          height={22}
        />
      )}
      <Typography variant="caption" color="text.secondary">
        {activeMode.detail} / {activeSourceCount.toLocaleString()} sources
      </Typography>
    </Box>
  );
  const enabledServiceCount = capabilities?.services
    ? Object.values(capabilities.services).filter((service: any) => service.enabled).length
    : 0;
  const capabilityCount = capabilities?.capabilities?.length || 0;
  const bridgeAvailable = Boolean(capabilities?.upstream_bridge?.available);
  const retrievalEvaluationReady = Boolean(evaluationStatus?.retrieval && !evaluationStatus.retrieval.error);
  const chunkingEvaluationReady = Boolean(evaluationStatus?.chunking && !evaluationStatus.chunking.error);
  const capabilityStrip = (
    <SectionCard
      title="GraphRAG capabilities"
      subtitle={loadingCapabilities ? 'Checking GraphRAG bridge' : `${capabilityCount} capability groups`}
      icon={<SourceIcon />}
      accentColor={bridgeAvailable ? accentColors.success : accentColors.info}
    >
      {capabilityError && (
        <Alert severity="warning" sx={{ mb: 1 }}>
          {capabilityError}
        </Alert>
      )}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
        {loadingCapabilities && <CircularProgress size={16} />}
        <StatusChip
          status={capabilities?.enabled ? 'online' : 'offline'}
          label={capabilities?.enabled ? 'GraphRAG enabled' : 'GraphRAG disabled'}
          statusConfig={REASONING_STATUS_CONFIG}
          height={22}
        />
        <StatusChip
          status={bridgeAvailable ? 'bridge' : 'waiting'}
          label={bridgeAvailable ? 'Bridge ready' : 'Bridge unavailable'}
          statusConfig={REASONING_STATUS_CONFIG}
          height={22}
        />
        <StatusChip
          status="graphrag"
          label={`${enabledServiceCount} services`}
          statusConfig={REASONING_STATUS_CONFIG}
          height={22}
        />
        <StatusChip
          status={retrievalEvaluationReady || chunkingEvaluationReady ? 'evaluation' : 'waiting'}
          label={`${Number(retrievalEvaluationReady) + Number(chunkingEvaluationReady)} eval checks`}
          statusConfig={REASONING_STATUS_CONFIG}
          height={22}
        />
        {capabilities?.capabilities?.map((capability: any) => (
          <StatusChip
            key={capability.id}
            status={capability.id === 'reasoning' ? 'multi-hop' : capability.id}
            label={String(capability.id).replace(/_/g, ' ')}
            variant="outlined"
            statusConfig={REASONING_STATUS_CONFIG}
            height={22}
          />
        ))}
      </Box>
    </SectionCard>
  );

  return (
    <PageShell
      sx={{
        py: 1.25,
        display: 'flex',
        flexDirection: 'column',
        gap: 1.25,
      }}
    >
      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 1.25 }}>
          {error}
        </Alert>
      )}

      {capabilityStrip}

      {/* Multi-Hop Tab */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={1.5}>
          <Grid item xs={12} md={multiHopResult ? 6 : 12}>
            <SectionCard
              title="Multi-hop reasoning"
              icon={<MultiHopIcon />}
              accentColor={accentColors.primary}
              action={reasoningHeaderActions}
            >
              {reasoningSummaryRow}
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: {
                    xs: '1fr',
                    sm: showMultiHopAction
                      ? 'minmax(0, 1fr) minmax(0, 1fr) 38px'
                      : 'repeat(2, minmax(0, 1fr))',
                  },
                  gap: 0.75,
                  alignItems: 'center',
                }}
              >
                <TextField
                  fullWidth
                  size="small"
                  label="Source Entity"
                  value={multiHopSource}
                  onChange={(e) => setMultiHopSource(e.target.value)}
                  placeholder="e.g., OpenAI"
                />

                <TextField
                  fullWidth
                  size="small"
                  label="Target Entity"
                  value={multiHopTarget}
                  onChange={(e) => setMultiHopTarget(e.target.value)}
                  placeholder="e.g., GPT-4"
                />

                {showMultiHopAction && (
                  <Tooltip title={loading ? 'Finding connection' : 'Find connection'}>
                    <span>
                      <IconButton
                        aria-label="Find connection"
                        onClick={handleMultiHopReasoning}
                        disabled={loading}
                        sx={actionIconButtonSx(accentColors.primary)}
                      >
                        {loading ? <CircularProgress size={16} color="inherit" /> : <MultiHopIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                    </span>
                  </Tooltip>
                )}
              </Box>
            </SectionCard>
          </Grid>

          {multiHopResult && (
          <Grid item xs={12} md={6}>
            <SectionCard
              title="Reasoning path"
              icon={<SourceIcon />}
              accentColor={accentColors.primary}
              fullHeight
            >
              {multiHopResult.answer && (
                <Box sx={answerSurfaceSx(accentColors.primary)}>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {multiHopResult.answer}
                  </Typography>
                </Box>
              )}
              <ReasoningEvidenceSummary result={multiHopResult} color={accentColors.primary} />
            </SectionCard>
          </Grid>
          )}
        </Grid>
      </TabPanel>

      {/* Causal Tab */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={1.5}>
          <Grid item xs={12} md={causalResult ? 6 : 12}>
            <SectionCard
              title="Causal reasoning"
              icon={<CausalIcon />}
              accentColor={accentColors.success}
              action={reasoningHeaderActions}
            >
              {reasoningSummaryRow}
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: showCausalAction ? 'minmax(0, 1fr) 38px' : '1fr',
                  gap: 0.75,
                  alignItems: 'start',
                }}
              >
                <TextField
                  fullWidth
                  size="small"
                  label="Query"
                  value={causalQuery}
                  onChange={(e) => setCausalQuery(e.target.value)}
                  placeholder="e.g., What causes system failures?"
                  multiline
                  rows={3}
                />

                {showCausalAction && (
                  <Tooltip title={loading ? 'Analyzing causality' : 'Analyze causality'}>
                    <span>
                      <IconButton
                        aria-label="Analyze causality"
                        onClick={handleCausalReasoning}
                        disabled={loading}
                        sx={actionIconButtonSx(accentColors.success)}
                      >
                        {loading ? <CircularProgress size={16} color="inherit" /> : <CausalIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                    </span>
                  </Tooltip>
                )}
              </Box>
            </SectionCard>
          </Grid>

          {causalResult && (
          <Grid item xs={12} md={6}>
            <SectionCard
              title="Analysis results"
              icon={<ConfidenceIcon />}
              accentColor={accentColors.success}
              fullHeight
            >
              {causalResult.answer && (
                <Box sx={answerSurfaceSx(accentColors.success)}>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {causalResult.answer}
                  </Typography>
                </Box>
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

              <ReasoningEvidenceSummary result={causalResult} color={accentColors.success} />
            </SectionCard>
          </Grid>
          )}
        </Grid>
      </TabPanel>

      {/* Comparative Tab */}
      <TabPanel value={tabValue} index={2}>
        <Grid container spacing={1.5}>
          <Grid item xs={12} md={compareResult ? 6 : 12}>
            <SectionCard
              title="Comparative analysis"
              icon={<CompareIcon />}
              accentColor={accentColors.warning}
              action={reasoningHeaderActions}
            >
              {reasoningSummaryRow}
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: {
                    xs: '1fr',
                    sm: showComparativeAction
                      ? 'minmax(0, 1fr) minmax(0, 1fr) 38px'
                      : 'repeat(2, minmax(0, 1fr))',
                  },
                  gap: 0.75,
                  alignItems: 'center',
                }}
              >
                <TextField
                  fullWidth
                  size="small"
                  label="First Entity"
                  value={compareEntity1}
                  onChange={(e) => setCompareEntity1(e.target.value)}
                  placeholder="e.g., Python"
                />

                <TextField
                  fullWidth
                  size="small"
                  label="Second Entity"
                  value={compareEntity2}
                  onChange={(e) => setCompareEntity2(e.target.value)}
                  placeholder="e.g., JavaScript"
                />

                {showComparativeAction && (
                  <Tooltip title={loading ? 'Comparing entities' : 'Compare entities'}>
                    <span>
                      <IconButton
                        aria-label="Compare entities"
                        onClick={handleComparativeAnalysis}
                        disabled={loading}
                        sx={actionIconButtonSx(accentColors.warning)}
                      >
                        {loading ? <CircularProgress size={16} color="inherit" /> : <CompareIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                    </span>
                  </Tooltip>
                )}
              </Box>
            </SectionCard>
          </Grid>

          {compareResult && (
          <Grid item xs={12} md={6}>
            <SectionCard
              title="Comparison results"
              icon={<CompareIcon />}
              accentColor={accentColors.warning}
              fullHeight
            >
              {compareResult.answer && (
                <Box sx={answerSurfaceSx(accentColors.primary)}>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {compareResult.answer}
                  </Typography>
                </Box>
              )}

              <Grid container spacing={2}>
                {compareResult.similarities && compareResult.similarities.length > 0 && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" gutterBottom sx={{ color: accentColors.success }}>
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
                    <Typography variant="subtitle2" gutterBottom sx={{ color: accentColors.warning }}>
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

              <ReasoningEvidenceSummary result={compareResult} color={accentColors.warning} />
            </SectionCard>
          </Grid>
          )}
        </Grid>
      </TabPanel>

      {/* Relationship Tab */}
      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={1.5}>
          <Grid item xs={12} md={relResult ? 6 : 12}>
            <SectionCard
              title="Explain relationship"
              icon={<RelationIcon />}
              accentColor={accentColors.purple}
              action={reasoningHeaderActions}
            >
              {reasoningSummaryRow}
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: {
                    xs: '1fr',
                    sm: showRelationshipAction
                      ? 'minmax(0, 1fr) minmax(0, 1fr) 38px'
                      : 'repeat(2, minmax(0, 1fr))',
                  },
                  gap: 0.75,
                  alignItems: 'center',
                }}
              >
                <TextField
                  fullWidth
                  size="small"
                  label="Source Entity"
                  value={relSource}
                  onChange={(e) => setRelSource(e.target.value)}
                  placeholder="e.g., Machine Learning"
                />

                <TextField
                  fullWidth
                  size="small"
                  label="Target Entity"
                  value={relTarget}
                  onChange={(e) => setRelTarget(e.target.value)}
                  placeholder="e.g., Neural Networks"
                />

                {showRelationshipAction && (
                  <Tooltip title={loading ? 'Explaining relationship' : 'Explain relationship'}>
                    <span>
                      <IconButton
                        aria-label="Explain relationship"
                        onClick={handleRelationshipExplanation}
                        disabled={loading}
                        sx={actionIconButtonSx(accentColors.purple)}
                      >
                        {loading ? <CircularProgress size={16} color="inherit" /> : <RelationIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                    </span>
                  </Tooltip>
                )}
              </Box>
            </SectionCard>
          </Grid>

          {relResult && (
          <Grid item xs={12} md={6}>
            <SectionCard
              title="Relationship explanation"
              icon={<RelationIcon />}
              accentColor={accentColors.purple}
              fullHeight
            >
              {relResult.explanation && (
                <Box sx={answerSurfaceSx(accentColors.purple)}>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {relResult.explanation}
                  </Typography>
                </Box>
              )}

              {relResult.confidence && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <ConfidenceIcon fontSize="small" sx={{ color: accentColors.success }} />
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {(relResult.confidence * 100).toFixed(0)}%
                  </Typography>
                </Box>
              )}

              <ReasoningEvidenceSummary result={relResult} color={accentColors.purple} />
            </SectionCard>
          </Grid>
          )}
        </Grid>
      </TabPanel>

    </PageShell>
  );
};

export default ReasoningPlaygroundPage;
