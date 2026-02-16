/**
 * Entity Manager - Manage entities in the knowledge graph
 * 
 * Note: Full entity extraction requires the NER service.
 * Deploy with: docker compose --profile graphrag-ner up -d
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
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
  Chip,
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
  AlertTitle,
  CircularProgress,
  Tooltip,
  InputAdornment,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Divider,
  alpha,
  Collapse,
} from '@mui/material';
import {
  Search as SearchIcon,
  Refresh as RefreshIcon,
  Visibility as ViewIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Link as LinkIcon,
  MergeType as MergeIcon,
  FilterList as FilterIcon,
  Person as PersonIcon,
  Business as OrgIcon,
  Place as LocationIcon,
  Event as EventIcon,
  Computer as TechIcon,
  Lightbulb as ConceptIcon,
  Category as CategoryIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { apiService as api } from '../services/api';

const ENTITY_TYPES = [
  'person', 'organization', 'location', 'date', 'event',
  'product', 'technology', 'concept', 'process', 'custom'
];

const TYPE_ICONS: Record<string, React.ReactNode> = {
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
  person: '#10b981',
  organization: '#3b82f6',
  location: '#f59e0b',
  event: '#ef4444',
  technology: '#6366f1',
  product: '#06b6d4',
  concept: '#ec4899',
  process: '#14b8a6',
  custom: '#6b7280',
};

interface Entity {
  id: string;
  label: string;
  type: string;
  description?: string;
  confidence?: number;
  occurrence?: number;
  properties?: any;
}

const EntityManagerPage: React.FC = () => {
  const [entities, setEntities] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

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
        limit: rowsPerPage,
        min_occurrence: minOccurrence,
      });

      const entitiesData = result.top_entities || [];

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
          e.name?.toLowerCase().includes(searchTerm.toLowerCase())
        );
      }

      setEntities(filteredEntities);
      setTotalEntities(filteredEntities.length);
    } catch (err: any) {
      setError(err.message || 'Failed to load entities');
    } finally {
      setLoading(false);
    }
  }, [rowsPerPage, minOccurrence, typeFilter, searchTerm]);

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
      console.error('Failed to load relationships:', err);
    }
  };

  const handleChangePage = (_: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box>
          <Typography variant="h4" fontWeight={700} gutterBottom>
            Entity Manager
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage entities in your knowledge graph
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={loadEntities}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" onClose={() => setSuccess(null)} sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      {/* Filters */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
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
            sx={{ minWidth: 250 }}
          />

          <FormControl size="small" sx={{ minWidth: 180 }}>
            <InputLabel>Entity Type</InputLabel>
            <Select
              value={typeFilter}
              label="Entity Type"
              onChange={(e) => setTypeFilter(e.target.value)}
            >
              <MenuItem value="">All Types</MenuItem>
              {ENTITY_TYPES.map(type => (
                <MenuItem key={type} value={type}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ color: TYPE_COLORS[type] || '#6b7280' }}>
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
            label="Min Occurrence"
            value={minOccurrence}
            onChange={(e) => setMinOccurrence(parseInt(e.target.value) || 1)}
            sx={{ width: 150 }}
          />

          <Chip
            label={`${totalEntities} entities`}
            sx={{
              ml: 'auto',
              bgcolor: alpha('#6366f1', 0.1),
              color: '#6366f1',
              fontWeight: 600,
            }}
          />
        </Box>
      </Paper>

      {/* Entities Table */}
      <TableContainer component={Paper}>
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
            {loading ? (
              <TableRow>
                <TableCell colSpan={5} align="center" sx={{ py: 8 }}>
                  <CircularProgress />
                </TableCell>
              </TableRow>
            ) : entities.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} align="center" sx={{ py: 8 }}>
                  <Typography color="text.secondary">
                    No entities found. Upload documents to build your knowledge graph.
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              entities.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((entity) => {
                const typeColor = TYPE_COLORS[entity.type?.toLowerCase()] || '#6b7280';
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
                      <Chip
                        label={entity.type || 'unknown'}
                        size="small"
                        sx={{
                          bgcolor: alpha(typeColor, 0.1),
                          color: typeColor,
                          textTransform: 'capitalize',
                        }}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={entity.occurrence || 1}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      {entity.confidence !== undefined && (
                        <Chip
                          label={`${(entity.confidence * 100).toFixed(0)}%`}
                          size="small"
                          sx={{
                            bgcolor: alpha('#10b981', 0.1),
                            color: '#10b981',
                          }}
                        />
                      )}
                    </TableCell>
                    <TableCell align="right">
                      <Tooltip title="View Details">
                        <IconButton size="small" onClick={() => handleViewEntity(entity)}>
                          <ViewIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Link Entity">
                        <IconButton size="small">
                          <LinkIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
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
                <Chip
                  label={selectedEntity.type || 'unknown'}
                  size="small"
                  sx={{
                    ml: 1,
                    bgcolor: alpha(TYPE_COLORS[selectedEntity.type?.toLowerCase()] || '#6b7280', 0.1),
                    color: TYPE_COLORS[selectedEntity.type?.toLowerCase()] || '#6b7280',
                    textTransform: 'capitalize',
                  }}
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
                    <Chip label={selectedEntity.occurrence} size="small" sx={{ ml: 1 }} />
                  </Box>
                )}
                {selectedEntity.confidence !== undefined && (
                  <Box>
                    <Typography variant="overline" color="text.secondary">
                      Confidence
                    </Typography>
                    <Chip
                      label={`${(selectedEntity.confidence * 100).toFixed(0)}%`}
                      size="small"
                      sx={{
                        ml: 1,
                        bgcolor: alpha('#10b981', 0.1),
                        color: '#10b981',
                      }}
                    />
                  </Box>
                )}
              </Box>

              {selectedEntity.properties && Object.keys(selectedEntity.properties).length > 0 && (
                <Box>
                  <Typography variant="overline" color="text.secondary" gutterBottom>
                    Properties
                  </Typography>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 1.5,
                      bgcolor: alpha('#6366f1', 0.03),
                      border: 1,
                      borderColor: 'divider',
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
                  </Paper>
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

      {/* Info Card */}
      <Paper
        elevation={0}
        sx={{
          mt: 3,
          p: 2,
          bgcolor: alpha('#6366f1', 0.05),
          border: 1,
          borderColor: alpha('#6366f1', 0.1),
        }}
      >
        <Typography variant="subtitle2" gutterBottom>
          About Entity Manager
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Entities are automatically extracted from documents using GraphRAG's GLiNER model.
          Upload documents to the Documents page to populate your knowledge graph with entities
          and relationships.
        </Typography>
      </Paper>
    </Box>
  );
};

export default EntityManagerPage;

