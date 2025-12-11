/**
 * BaseNode - Custom node component for workflow canvas
 */
import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Typography, Paper, alpha, Chip, keyframes } from '@mui/material';
import {
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
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
} from '@mui/icons-material';
import { WorkflowNodeData, NodeCategory, ExecutionStatus } from '@/types/workflow';

// Pulse animation for running nodes
const pulse = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(99, 102, 241, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0);
  }
`;

// Category icons
const categoryIcons: Record<NodeCategory, React.ReactNode> = {
  trigger: <TriggerIcon fontSize="small" />,
  llm: <LlmIcon fontSize="small" />,
  rag: <RagIcon fontSize="small" />,
  tools: <ToolsIcon fontSize="small" />,
  data: <DataIcon fontSize="small" />,
  control: <ControlIcon fontSize="small" />,
  api: <ApiIcon fontSize="small" />,
  mcp: <McpIcon fontSize="small" />,
  database: <DatabaseIcon fontSize="small" />,
  output: <OutputIcon fontSize="small" />,
};

// Category colors
const categoryColors: Record<NodeCategory, string> = {
  trigger: '#10b981',
  llm: '#6366f1',
  rag: '#8b5cf6',
  tools: '#f59e0b',
  data: '#06b6d4',
  control: '#ec4899',
  api: '#14b8a6',
  mcp: '#a855f7',
  database: '#f97316',
  output: '#ef4444',
};

// Status colors and icons
const statusConfig: Record<ExecutionStatus, { color: string; icon?: React.ReactNode }> = {
  idle: { color: '#64748b' },
  pending: { color: '#f59e0b', icon: <PendingIcon fontSize="inherit" /> },
  running: { color: '#6366f1' },
  success: { color: '#10b981', icon: <SuccessIcon fontSize="inherit" /> },
  error: { color: '#ef4444', icon: <ErrorIcon fontSize="inherit" /> },
  cancelled: { color: '#64748b' },
  paused: { color: '#f59e0b' },
};

// Get category from node type
function getCategoryFromType(nodeType: string): NodeCategory {
  const categoryMap: Record<string, NodeCategory> = {
    manual_trigger: 'trigger',
    http_webhook: 'trigger',
    llm_chat: 'llm',
    openai_chat: 'llm',
    embedding: 'llm',
    document_loader: 'rag',
    chunker: 'rag',
    retriever: 'rag',
    vector_store: 'rag',
    code_executor: 'tools',
    function_call: 'tools',
    template: 'data',
    json_parse: 'data',
    json_stringify: 'data',
    mapper: 'data',
    filter: 'data',
    condition: 'control',
    switch: 'control',
    loop: 'control',
    merge: 'control',
    delay: 'control',
    http_request: 'api',
    graphql_query: 'api',
    mcp_tool: 'mcp',
    mcp_resource: 'mcp',
    sql_query: 'database',
    cache_get: 'database',
    cache_set: 'database',
    output: 'output',
    webhook_response: 'output',
    log: 'output',
  };
  return categoryMap[nodeType] || 'data';
}

export const BaseNode: React.FC<NodeProps<WorkflowNodeData>> = memo(({ data, selected }) => {
  const category = getCategoryFromType(data.nodeType);
  const color = categoryColors[category];
  const icon = categoryIcons[category];
  const status = data.status || 'idle';
  const statusInfo = statusConfig[status];

  return (
    <Paper
      elevation={selected ? 8 : 2}
      sx={{
        minWidth: 200,
        maxWidth: 280,
        border: '2px solid',
        borderColor: selected ? color : alpha(color, 0.3),
        borderRadius: 2,
        overflow: 'hidden',
        transition: 'all 0.2s ease-in-out',
        animation: status === 'running' ? `${pulse} 1.5s infinite` : 'none',
        '&:hover': {
          borderColor: color,
          transform: 'translateY(-2px)',
          boxShadow: `0 8px 24px ${alpha(color, 0.3)}`,
        },
      }}
    >
      {/* Header */}
      <Box
        sx={{
          px: 1.5,
          py: 1,
          bgcolor: alpha(color, 0.15),
          borderBottom: '1px solid',
          borderColor: alpha(color, 0.2),
          display: 'flex',
          alignItems: 'center',
          gap: 1,
        }}
      >
        <Box sx={{ color, display: 'flex', alignItems: 'center' }}>
          {icon}
        </Box>
        <Typography
          variant="subtitle2"
          sx={{
            flex: 1,
            fontWeight: 600,
            color: 'text.primary',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {data.label}
        </Typography>
        {status !== 'idle' && (
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              bgcolor: statusInfo.color,
              animation: status === 'running' ? `${pulse} 1s infinite` : 'none',
            }}
          />
        )}
      </Box>

      {/* Input Handles */}
      {data.inputs.length > 0 && (
        <Box sx={{ py: 0.5 }}>
          {data.inputs.map((input, index) => (
            <Box
              key={input.id}
              sx={{
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                pl: 2,
                py: 0.25,
                minHeight: 24,
              }}
            >
              <Handle
                type="target"
                position={Position.Left}
                id={input.id}
                style={{
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: input.required ? color : alpha(color, 0.5),
                }}
              />
              <Typography
                variant="caption"
                sx={{
                  color: 'text.secondary',
                  fontSize: '0.7rem',
                }}
              >
                {input.name}
                {input.required && (
                  <Box component="span" sx={{ color: 'error.main', ml: 0.25 }}>*</Box>
                )}
              </Typography>
            </Box>
          ))}
        </Box>
      )}

      {/* Divider if both inputs and outputs */}
      {data.inputs.length > 0 && data.outputs.length > 0 && (
        <Box sx={{ borderTop: '1px solid', borderColor: 'divider' }} />
      )}

      {/* Output Handles */}
      {data.outputs.length > 0 && (
        <Box sx={{ py: 0.5 }}>
          {data.outputs.map((output, index) => (
            <Box
              key={output.id}
              sx={{
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                pr: 2,
                py: 0.25,
                minHeight: 24,
              }}
            >
              <Typography
                variant="caption"
                sx={{
                  color: 'text.secondary',
                  fontSize: '0.7rem',
                }}
              >
                {output.name}
              </Typography>
              <Handle
                type="source"
                position={Position.Right}
                id={output.id}
                style={{
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: color,
                }}
              />
            </Box>
          ))}
        </Box>
      )}

      {/* Error display */}
      {data.error && (
        <Box
          sx={{
            px: 1.5,
            py: 0.75,
            bgcolor: alpha('#ef4444', 0.15),
            borderTop: '1px solid',
            borderColor: alpha('#ef4444', 0.3),
          }}
        >
          <Typography
            variant="caption"
            sx={{
              color: '#ef4444',
              fontSize: '0.65rem',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {data.error}
          </Typography>
        </Box>
      )}

      {/* Execution time display */}
      {data.executionTime !== undefined && status === 'success' && (
        <Box
          sx={{
            px: 1.5,
            py: 0.5,
            bgcolor: alpha('#10b981', 0.1),
            borderTop: '1px solid',
            borderColor: alpha('#10b981', 0.2),
          }}
        >
          <Typography
            variant="caption"
            sx={{
              color: '#10b981',
              fontSize: '0.65rem',
            }}
          >
            {data.executionTime}ms
          </Typography>
        </Box>
      )}
    </Paper>
  );
});

BaseNode.displayName = 'BaseNode';

export default BaseNode;
