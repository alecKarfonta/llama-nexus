/**
 * Entity Manager - Manage entities in the knowledge graph
 * 
 * Note: Full entity extraction requires the NER service.
 * Deploy with: docker compose --profile graphrag-ner up -d
 */
import { useState, useEffect, useCallback, type ChangeEvent, type ReactNode } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
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
  Tooltip,
  InputAdornment,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  Visibility as ViewIcon,
  Person as PersonIcon,
  Business as OrgIcon,
  Place as LocationIcon,
  Event as EventIcon,
  Computer as TechIcon,
  Lightbulb as ConceptIcon,
  Category as CategoryIcon,
  Hub as GraphIcon,
  Description as DocumentsIcon,
} from '@mui/icons-material';
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from '../components/common';
import { getRoutePath } from '../config/navigation';
import { apiService as api } from '../services/api';
import { debugLog } from '../utils/debug';
import { accentColors } from '../utils/theme';

const ENTITY_TYPES = [
  'person', 'organization', 'location', 'date', 'event',
  'product', 'technology', 'concept', 'process', 'custom'
];

const TYPE_ICONS: Record<string, ReactNode> = {
  person: <PersonIcon fontSize="small" />,
  organization: <OrgIcon fontSize="small" />,
  location: <LocationIcon fontSize="small" />,
  event: <EventIcon fontSize="small" />,
  technology: <TechIcon fontSize="small" />,
  product: <TechIcon fontSize="small" />,
  concept: <ConceptIcon fontSize="small" />,
  process: <CategoryIcon fontSize="small" />,
};

const TYPE_COLORS: Record<string, string> = {
  person: accentColors.success,
  organization: accentColors.info,
  location: accentColors.warning,
  event: accentColors.rose,
  technology: accentColors.primary,
  product: accentColors.info,
  concept: accentColors.rose,
  process: accentColors.success,
  custom: accentColors.slate,
};

const ENTITY_STATUS_CONFIG: StatusChipConfigMap = {
  all: { color: accentColors.primary, label: 'All entity types' },
  confidence: { color: accentColors.info, label: 'Confidence' },
  'confidence-high': { color: accentColors.success, label: 'High confidence' },
  'confidence-low': { color: accentColors.rose, label: 'Low confidence' },
  'confidence-medium': { color: accentColors.warning, label: 'Medium confidence' },
  entities: { color: accentColors.primary, label: 'Entities' },
  manager: { color: accentColors.primary, label: 'Entity manager' },
  occurrence: { color: accentColors.info, label: 'Occurrence' },
  unknown: { color: accentColors.slate, label: 'Unknown' },
  ...ENTITY_TYPES.reduce((config, type) => {
    config[type] = {
      color: TYPE_COLORS[type] || accentColors.slate,
      label: type.charAt(0).toUpperCase() + type.slice(1),
    };
    return config;
  }, {} as StatusChipConfigMap),
};

const entityPanelSx = {
  borderRadius: '8px',
  bgcolor: 'background.paper',
  border: '1px solid',
  borderColor: 'divider',
} as const;

interface Entity {
  id: string;
  label: string;
  type: string;
  description?: string;
  confidence?: number;
  occurrence?: number;
  properties?: any;
}

const formatNumber = (value: number): string => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
  return value.toString();
};

const getConfidenceStatus = (value: number): string => {
  if (value >= 0.7) return 'confidence-high';
  if (value >= 0.4) return 'confidence-medium';
  return 'confidence-low';
};

const EntityManagerPage = () => {
  const navigate = useNavigate();
  const documentsPath = getRoutePath('documents');
  const [entities, setEntities] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [totalEntities, setTotalEntities] = useState(0);

  // Filters
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [minOccurrence, setMinOccurrence] = useState(1);

  // Selected entity
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);
  const [relationships, setRelationships] = useState<any[]>([]);

  // Load entities
  const loadEntities = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await api.getGraphRAGTopEntities({
        limit: 500,
        min_occurrence: minOccurrence,
      });

      if (result.available === false) {
        setError(result.message || result.error || 'GraphRAG service is disabled. Set GRAPHRAG_ENABLED=true');
      }

      const sourceEntities = result.entities || result.top_entities || [];
      const entitiesData: Entity[] = sourceEntities.map((entity) => ({
        id: `${entity.type || 'entity'}:${entity.name}`,
        label: entity.name,
        type: entity.type,
        description: entity.description,
        occurrence: entity.occurrence,
        confidence: entity.confidence,
        properties: {
          domain: entity.domain,
        },
      }));

      // Filter by type if selected
      let filteredEntities = entitiesData;
      if (typeFilter) {
        filteredEntities = entitiesData.filter((e: any) =>
          e.type?.toLowerCase() === typeFilter.toLowerCase()
        );
      }

      // Filter by search term
      if (searchTerm) {
        filteredEntities = filteredEntities.filter((e: any) =>
          e.label?.toLowerCase().includes(searchTerm.toLowerCase())
        );
      }

      setEntities(filteredEntities);
      setTotalEntities(filteredEntities.length);
      setPage(0);
    } catch (err: any) {
      setError(err.message || 'Failed to load entities');
    } finally {
      setLoading(false);
    }
  }, [minOccurrence, typeFilter, searchTerm]);

  useEffect(() => {
    loadEntities();
  }, [loadEntities]);

  const handleViewEntity = async (entity: Entity) => {
    setSelectedEntity(entity);
    setDetailDialogOpen(true);

    // Load relationships for this entity
    try {
      // For now, we'll show the entity details
      // In a full implementation, you'd call an endpoint to get relationships
      setRelationships([]);
    } catch (err) {
      debugLog('entity-manager', 'Failed to load relationships', err);
    }
  };

  const handleChangePage = (_: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const activeFilterLabel = typeFilter ? `${typeFilter} entities` : 'All entity types';
  const selectedTypeColor = TYPE_COLORS[typeFilter] || accentColors.primary;
  const graphRAGDisabled = Boolean(error && /graphrag/i.test(error) && /disabled|enabled/i.test(error));
  const hasEntityFilters = Boolean(searchTerm || typeFilter || minOccurrence > 1);
  const emptyTitle = hasEntityFilters ? 'No matching entities' : 'No entities yet';
  const emptyDetail = hasEntityFilters ? activeFilterLabel : 'Process documents to populate the graph.';

  const handleClearEntityFilters = () => {
    setSearchTerm('');
    setTypeFilter('');
    setMinOccurrence(1);
  };
  const refreshEntitiesButton = (
    <Tooltip title="Refresh entities">
      <IconButton
        aria-label="Refresh entities"
        size="small"
        onClick={loadEntities}
        disabled={loading}
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
        {loading ? <CircularProgress size={16} /> : <RefreshIcon sx={{ fontSize: 18 }} />}
      </IconButton>
    </Tooltip>
  );
  const entityHeaderActions = graphRAGDisabled ? refreshEntitiesButton : (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      <TextField
        size="small"
        placeholder="Search entities..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon fontSize="small" />
            </InputAdornment>
          ),
        }}
        sx={{ width: { xs: '100%', sm: 240, lg: 320 }, '& .MuiInputBase-root': { height: 34 } }}
      />

      <FormControl size="small" sx={{ width: { xs: 'calc(100% - 102px)', sm: 170 } }}>
        <InputLabel>Entity Type</InputLabel>
        <Select
          value={typeFilter}
          label="Entity Type"
          onChange={(e) => setTypeFilter(e.target.value)}
          sx={{ height: 34 }}
        >
          <MenuItem value="">All Types</MenuItem>
          {ENTITY_TYPES.map(type => (
            <MenuItem key={type} value={type}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ color: TYPE_COLORS[type] || accentColors.slate }}>
                  {TYPE_ICONS[type] || <CategoryIcon fontSize="small" />}
                </Box>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </Box>
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <TextField
        size="small"
        type="number"
        label="Min"
        value={minOccurrence}
        onChange={(e) => setMinOccurrence(parseInt(e.target.value) || 1)}
        inputProps={{ min: 1 }}
        sx={{ width: { xs: 56, sm: 84 }, '& .MuiInputBase-root': { height: 34 } }}
      />

      {refreshEntitiesButton}
    </Box>
  );
  const entitySummaryRow = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
      <StatusChip
        status={typeFilter || 'all'}
        label={activeFilterLabel}
        statusConfig={ENTITY_STATUS_CONFIG}
        height={22}
        sx={{ textTransform: 'capitalize' }}
      />
      <StatusChip
        status="entities"
        label={`${formatNumber(totalEntities)} visible`}
        statusConfig={ENTITY_STATUS_CONFIG}
        height={22}
      />
    </Box>
  );

  return (
    <PageShell sx={{ py: 1.25 }}>
      {error && !graphRAGDisabled && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 1.25 }}>
          {error}
        </Alert>
      )}

      {/* Entities Table */}
      <SectionCard
        title="Entities"
        subtitle={graphRAGDisabled ? 'GraphRAG service disabled' : `${formatNumber(totalEntities)} visible in ${activeFilterLabel.toLowerCase()}`}
        icon={<GraphIcon />}
        accentColor={selectedTypeColor}
        action={entityHeaderActions}
      >
        {!graphRAGDisabled && entitySummaryRow}
        {loading ? (
          <Box sx={{ minHeight: 72, display: 'grid', placeItems: 'center' }}>
            <CircularProgress />
          </Box>
        ) : graphRAGDisabled ? (
          <Alert severity="warning" sx={{ borderRadius: '8px', alignItems: 'center' }}>
            {error || 'GraphRAG is disabled. Enable GraphRAG services, then refresh the entity index.'}
          </Alert>
        ) : entities.length === 0 ? (
          <Box
            sx={{
              minHeight: 72,
              p: 1.25,
              borderRadius: '8px',
              border: '1px dashed',
              borderColor: 'divider',
              bgcolor: 'action.hover',
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1fr) auto' },
              gap: 1.25,
              alignItems: 'center',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, minWidth: 0 }}>
              <Box
                sx={{
                  width: 34,
                  height: 34,
                  borderRadius: '8px',
                  display: 'grid',
                  placeItems: 'center',
                  bgcolor: alpha(accentColors.primary, 0.14),
                  color: accentColors.primary,
                  flex: '0 0 auto',
                }}
              >
                <GraphIcon sx={{ fontSize: 20 }} />
              </Box>
              <Box sx={{ minWidth: 0 }}>
                <Typography variant="body2" sx={{ fontWeight: 800 }}>
                  {emptyTitle}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {emptyDetail}
                </Typography>
              </Box>
            </Box>
            <Box sx={{ display: 'flex', gap: 0.75, justifyContent: { xs: 'flex-start', lg: 'flex-end' }, flexWrap: 'wrap' }}>
              {hasEntityFilters && (
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handleClearEntityFilters}
                  sx={{ height: 34, borderRadius: '8px', borderColor: 'divider', fontWeight: 800 }}
                >
                  Clear
                </Button>
              )}
              {!hasEntityFilters && (
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<DocumentsIcon />}
                  onClick={() => navigate(documentsPath)}
                  sx={{ height: 34, borderRadius: '8px', borderColor: 'divider', fontWeight: 800 }}
                >
                  Documents
                </Button>
              )}
            </Box>
          </Box>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Entity</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Occurrence</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {entities.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((entity) => {
                  const typeColor = TYPE_COLORS[entity.type?.toLowerCase()] || accentColors.slate;
                  return (
                    <TableRow key={entity.id} hover>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box sx={{ color: typeColor }}>
                            {TYPE_ICONS[entity.type?.toLowerCase()] || <CategoryIcon fontSize="small" />}
                          </Box>
                          <Typography variant="body2" fontWeight={500}>
                            {entity.label || entity.id}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <StatusChip
                          status={entity.type?.toLowerCase() || 'unknown'}
                          label={entity.type || 'unknown'}
                          statusConfig={ENTITY_STATUS_CONFIG}
                          sx={{ textTransform: 'capitalize' }}
                        />
                      </TableCell>
                      <TableCell>
                        <StatusChip
                          status="occurrence"
                          label={String(entity.occurrence || 1)}
                          variant="outlined"
                          statusConfig={ENTITY_STATUS_CONFIG}
                        />
                      </TableCell>
                      <TableCell>
                        {entity.confidence !== undefined && (
                          <StatusChip
                            status={getConfidenceStatus(entity.confidence)}
                            label={`${(entity.confidence * 100).toFixed(0)}%`}
                            statusConfig={ENTITY_STATUS_CONFIG}
                          />
                        )}
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title="View Details">
                          <IconButton size="small" onClick={() => handleViewEntity(entity)}>
                            <ViewIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
            <TablePagination
              rowsPerPageOptions={[10, 25, 50, 100]}
              component="div"
              count={totalEntities}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
            />
          </TableContainer>
        )}
      </SectionCard>

      {/* Entity Detail Dialog */}
      <Dialog
        open={detailDialogOpen}
        onClose={() => setDetailDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {selectedEntity?.label || 'Entity Details'}
        </DialogTitle>
        <DialogContent>
          {selectedEntity && (
            <Box>
              <Box sx={{ mb: 3 }}>
                <Typography variant="overline" color="text.secondary">
                  Type
                </Typography>
                <StatusChip
                  status={selectedEntity.type?.toLowerCase() || 'unknown'}
                  label={selectedEntity.type || 'unknown'}
                  statusConfig={ENTITY_STATUS_CONFIG}
                  sx={{ ml: 1, textTransform: 'capitalize' }}
                />
              </Box>

              {selectedEntity.description && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="overline" color="text.secondary" gutterBottom>
                    Description
                  </Typography>
                  <Typography variant="body2">
                    {selectedEntity.description}
                  </Typography>
                </Box>
              )}

              <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
                {selectedEntity.occurrence !== undefined && (
                  <Box>
                    <Typography variant="overline" color="text.secondary">
                      Occurrence
                    </Typography>
                    <StatusChip
                      status="occurrence"
                      label={String(selectedEntity.occurrence)}
                      variant="outlined"
                      statusConfig={ENTITY_STATUS_CONFIG}
                      sx={{ ml: 1 }}
                    />
                  </Box>
                )}
                {selectedEntity.confidence !== undefined && (
                  <Box>
                    <Typography variant="overline" color="text.secondary">
                      Confidence
                    </Typography>
                    <StatusChip
                      status={getConfidenceStatus(selectedEntity.confidence)}
                      label={`${(selectedEntity.confidence * 100).toFixed(0)}%`}
                      statusConfig={ENTITY_STATUS_CONFIG}
                      sx={{ ml: 1 }}
                    />
                  </Box>
                )}
              </Box>

              {selectedEntity.properties && Object.keys(selectedEntity.properties).length > 0 && (
                <Box>
                  <Typography variant="overline" color="text.secondary" gutterBottom>
                    Properties
                  </Typography>
                  <Box
                    sx={{
                      p: 1.5,
                      ...entityPanelSx,
                      bgcolor: alpha(accentColors.primary, 0.03),
                    }}
                  >
                    {Object.entries(selectedEntity.properties).map(([key, value]) => (
                      <Box key={key} sx={{ mb: 0.5 }}>
                        <Typography variant="caption" color="text.secondary">
                          {key}:
                        </Typography>
                        <Typography variant="caption" sx={{ ml: 1 }}>
                          {JSON.stringify(value)}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </Box>
              )}

              {relationships.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="overline" color="text.secondary" gutterBottom>
                    Relationships
                  </Typography>
                  <List dense>
                    {relationships.map((rel, idx) => (
                      <ListItem key={idx}>
                        <ListItemText
                          primary={`${rel.target} (${rel.type})`}
                          secondary={rel.description}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </PageShell>
  );
};

export default EntityManagerPage;
