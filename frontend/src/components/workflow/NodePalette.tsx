/**
 * NodePalette - Draggable list of available nodes
 */
import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  InputAdornment,
  alpha,
  Tooltip,
  MenuItem,
} from '@mui/material';
import {
  Search as SearchIcon,
  PlayArrow as TriggerIcon,
  Psychology as LlmIcon,
  Description as RagIcon,
  Build as ToolsIcon,
  Transform as DataIcon,
  AccountTree as ControlIcon,
  Api as ApiIcon,
  Hub as McpIcon,
  Storage as DatabaseIcon,
  Output as OutputIcon,
} from '@mui/icons-material';
import {
  NODE_CATEGORIES,
  BUILTIN_NODE_TYPES,
  NodeTypeDefinition,
  NodeCategory,
} from '@/types/workflow';

// Category icons mapping
const categoryIconComponents: Record<NodeCategory, React.ReactNode> = {
  trigger: <TriggerIcon />,
  llm: <LlmIcon />,
  rag: <RagIcon />,
  tools: <ToolsIcon />,
  data: <DataIcon />,
  control: <ControlIcon />,
  api: <ApiIcon />,
  mcp: <McpIcon />,
  database: <DatabaseIcon />,
  output: <OutputIcon />,
};

interface NodePaletteProps {
  disabled?: boolean;
  onNodeSelect?: (nodeType: string) => void;
}

const NODE_RESULT_LIMIT = 8;

export const NodePalette: React.FC<NodePaletteProps> = ({ disabled, onNodeSelect }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState<NodeCategory>('trigger');

  // Handle drag start
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    if (disabled) {
      event.preventDefault();
      return;
    }
    event.dataTransfer.setData('application/workflow-node', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const normalizedQuery = searchQuery.trim().toLowerCase();

  // Filter nodes by search query or the selected category.
  const filteredNodeTypes = normalizedQuery
    ? BUILTIN_NODE_TYPES.filter(
        (node) =>
          node.displayName.toLowerCase().includes(normalizedQuery) ||
          node.description.toLowerCase().includes(normalizedQuery)
      )
    : BUILTIN_NODE_TYPES.filter((node) => node.category === activeCategory);

  const visibleNodeTypes = filteredNodeTypes.slice(0, NODE_RESULT_LIMIT);
  const hiddenNodeCount = Math.max(filteredNodeTypes.length - visibleNodeTypes.length, 0);
  const activeCategoryDef = NODE_CATEGORIES.find((category) => category.id === activeCategory);

  return (
    <Box
      sx={{
        width: { xs: 176, sm: 244 },
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderRight: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.paper',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: { xs: 0.75, sm: 1 },
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          flexDirection: 'column',
          gap: 0.75,
        }}
      >
        <TextField
          size="small"
          fullWidth
          placeholder="Search nodes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon fontSize="small" sx={{ color: 'text.secondary' }} />
              </InputAdornment>
            ),
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              bgcolor: 'background.default',
              height: { xs: 34, sm: 36 },
            },
            '& .MuiInputBase-input': {
              fontSize: { xs: '0.82rem', sm: '0.875rem' },
            },
          }}
        />
        {!normalizedQuery && (
          <TextField
            select
            fullWidth
            size="small"
            aria-label="Node category"
            value={activeCategory}
            onChange={(event) => setActiveCategory(event.target.value as NodeCategory)}
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: 'background.default',
                height: 32,
              },
              '& .MuiSelect-select': {
                display: 'flex',
                alignItems: 'center',
                fontSize: '0.8rem',
                fontWeight: 800,
              },
            }}
          >
            {NODE_CATEGORIES.map((category) => {
              const categoryId = category.id as NodeCategory;
              const nodeCount = BUILTIN_NODE_TYPES.filter((node) => node.category === category.id).length;

              return (
                <MenuItem key={category.id} value={category.id}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0, width: '100%' }}>
                    <Box sx={{ display: 'flex', color: category.color, '& svg': { fontSize: 16 } }}>
                      {categoryIconComponents[categoryId]}
                    </Box>
                    <Typography variant="body2" noWrap sx={{ flex: 1, fontWeight: 700 }}>
                      {category.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {nodeCount}
                    </Typography>
                  </Box>
                </MenuItem>
              );
            })}
          </TextField>
        )}
      </Box>

      <Box sx={{ flex: 1, minHeight: 0, p: { xs: 0.75, sm: 0.85 }, overflow: 'hidden' }}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 1,
            mb: 0.75,
            px: 0.25,
          }}
        >
          <Typography variant="caption" color="text.secondary" fontWeight={800}>
            {normalizedQuery ? 'Matches' : activeCategoryDef?.name || 'Nodes'}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: { xs: 'none', sm: 'block' } }}>
            {filteredNodeTypes.length}
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
          {visibleNodeTypes.map((node) => {
            const category = NODE_CATEGORIES.find((item) => item.id === node.category);

            return (
              <NodeItem
                key={node.type}
                node={node}
                categoryColor={category?.color || '#6366f1'}
                onDragStart={onDragStart}
                onNodeSelect={onNodeSelect}
                disabled={disabled}
              />
            );
          })}
        </Box>

        {filteredNodeTypes.length === 0 && normalizedQuery && (
          <Box sx={{ p: 1.5, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No nodes match "{searchQuery}"
            </Typography>
          </Box>
        )}

        {hiddenNodeCount > 0 && (
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: 'block', mt: 0.75, px: 0.25 }}
          >
            {hiddenNodeCount} more hidden. Search to narrow the list.
          </Typography>
        )}
      </Box>
    </Box>
  );
};

// Individual node item component
interface NodeItemProps {
  node: NodeTypeDefinition;
  categoryColor: string;
  onDragStart: (event: React.DragEvent, nodeType: string) => void;
  onNodeSelect?: (nodeType: string) => void;
  disabled?: boolean;
}

const NodeItem: React.FC<NodeItemProps> = ({
  node,
  categoryColor,
  onDragStart,
  onNodeSelect,
  disabled,
}) => {
  return (
    <Tooltip
      title={
        <Box>
          <Typography variant="body2" fontWeight={500}>
            {node.displayName}
          </Typography>
          <Typography variant="caption">{node.description}</Typography>
          {node.inputs.length > 0 && (
            <Box sx={{ mt: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                Inputs: {node.inputs.map((i) => i.name).join(', ')}
              </Typography>
            </Box>
          )}
          {node.outputs.length > 0 && (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Outputs: {node.outputs.map((o) => o.name).join(', ')}
              </Typography>
            </Box>
          )}
        </Box>
      }
      placement="right"
      arrow
    >
      <Paper
        draggable={!disabled}
        onDragStart={(e) => onDragStart(e, node.type)}
        onClick={() => {
          if (!disabled) {
            onNodeSelect?.(node.type);
          }
        }}
        elevation={0}
        sx={{
          px: 0.85,
          py: 0.55,
          minHeight: { xs: 34, sm: 36 },
          cursor: disabled ? 'not-allowed' : 'grab',
          opacity: disabled ? 0.5 : 1,
          bgcolor: 'background.default',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 1,
          transition: 'all 0.15s ease-in-out',
          '&:hover': disabled
            ? {}
            : {
                borderColor: categoryColor,
                bgcolor: alpha(categoryColor, 0.05),
              },
          '&:active': disabled
            ? {}
            : {
                cursor: 'grabbing',
              },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
          <Box
            sx={{
              width: 4,
              height: { xs: 22, sm: 24 },
              bgcolor: categoryColor,
              borderRadius: 0.5,
              flexShrink: 0,
            }}
          />
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography
              variant="body2"
              fontWeight={700}
              sx={{
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                fontSize: { xs: '0.82rem', sm: '0.875rem' },
              }}
            >
              {node.displayName}
            </Typography>
          </Box>
        </Box>
      </Paper>
    </Tooltip>
  );
};

export default NodePalette;
