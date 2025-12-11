/**
 * NodePalette - Draggable list of available nodes
 */
import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  InputAdornment,
  alpha,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore as ExpandIcon,
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
  CategoryInfo,
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
}

export const NodePalette: React.FC<NodePaletteProps> = ({ disabled }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<string[]>(['trigger', 'llm']);

  // Handle drag start
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    if (disabled) {
      event.preventDefault();
      return;
    }
    event.dataTransfer.setData('application/workflow-node', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  // Filter nodes by search query
  const filteredNodeTypes = searchQuery
    ? BUILTIN_NODE_TYPES.filter(
        (node) =>
          node.displayName.toLowerCase().includes(searchQuery.toLowerCase()) ||
          node.description.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : BUILTIN_NODE_TYPES;

  // Group by category
  const nodesByCategory = NODE_CATEGORIES.map((category) => ({
    ...category,
    nodes: filteredNodeTypes.filter((node) => node.category === category.id),
  })).filter((cat) => cat.nodes.length > 0);

  // Handle accordion expand
  const handleAccordionChange = (categoryId: string) => (
    _: React.SyntheticEvent,
    isExpanded: boolean
  ) => {
    setExpandedCategories((prev) =>
      isExpanded
        ? [...prev, categoryId]
        : prev.filter((id) => id !== categoryId)
    );
  };

  return (
    <Box
      sx={{
        width: 280,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderRight: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.paper',
      }}
    >
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="subtitle2" fontWeight={600} gutterBottom>
          Node Palette
        </Typography>
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
            },
          }}
        />
      </Box>

      {/* Node categories */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 1 }}>
        {nodesByCategory.map((category) => (
          <Accordion
            key={category.id}
            expanded={expandedCategories.includes(category.id) || !!searchQuery}
            onChange={handleAccordionChange(category.id)}
            disableGutters
            elevation={0}
            sx={{
              bgcolor: 'transparent',
              '&:before': { display: 'none' },
              '& .MuiAccordionSummary-root': {
                minHeight: 40,
                px: 1,
              },
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandIcon fontSize="small" />}
              sx={{
                '& .MuiAccordionSummary-content': {
                  alignItems: 'center',
                  gap: 1,
                  my: 0.5,
                },
              }}
            >
              <Box sx={{ color: category.color, display: 'flex' }}>
                {categoryIconComponents[category.id as NodeCategory]}
              </Box>
              <Typography variant="body2" fontWeight={500}>
                {category.name}
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  ml: 'auto',
                  mr: 1,
                  color: 'text.secondary',
                }}
              >
                {category.nodes.length}
              </Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ p: 0.5, pt: 0 }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                {category.nodes.map((node) => (
                  <NodeItem
                    key={node.type}
                    node={node}
                    categoryColor={category.color}
                    onDragStart={onDragStart}
                    disabled={disabled}
                  />
                ))}
              </Box>
            </AccordionDetails>
          </Accordion>
        ))}

        {nodesByCategory.length === 0 && searchQuery && (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No nodes match "{searchQuery}"
            </Typography>
          </Box>
        )}
      </Box>

      {/* Hint */}
      <Box
        sx={{
          p: 1.5,
          borderTop: '1px solid',
          borderColor: 'divider',
          bgcolor: alpha('#6366f1', 0.05),
        }}
      >
        <Typography variant="caption" color="text.secondary">
          Drag nodes onto the canvas to add them to your workflow
        </Typography>
      </Box>
    </Box>
  );
};

// Individual node item component
interface NodeItemProps {
  node: NodeTypeDefinition;
  categoryColor: string;
  onDragStart: (event: React.DragEvent, nodeType: string) => void;
  disabled?: boolean;
}

const NodeItem: React.FC<NodeItemProps> = ({
  node,
  categoryColor,
  onDragStart,
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
        elevation={0}
        sx={{
          px: 1.5,
          py: 1,
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
                transform: 'translateX(4px)',
              },
          '&:active': disabled
            ? {}
            : {
                cursor: 'grabbing',
                transform: 'scale(0.98)',
              },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
          <Box
            sx={{
              width: 6,
              height: '100%',
              minHeight: 32,
              bgcolor: categoryColor,
              borderRadius: 0.5,
              flexShrink: 0,
            }}
          />
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography
              variant="body2"
              fontWeight={500}
              sx={{
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {node.displayName}
            </Typography>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
                lineHeight: 1.3,
              }}
            >
              {node.description}
            </Typography>
          </Box>
        </Box>
      </Paper>
    </Tooltip>
  );
};

export default NodePalette;
