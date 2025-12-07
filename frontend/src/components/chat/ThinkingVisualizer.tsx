/**
 * ThinkingVisualizer Component
 * Displays reasoning/thinking content from the model in a collapsible format
 */

import React, { useState, memo } from 'react';
import {
  Box,
  Collapse,
  IconButton,
  Typography,
  Paper,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Psychology as ThinkingIcon,
  Timer as TimerIcon,
} from '@mui/icons-material';
import { MarkdownContent } from './MarkdownContent';

interface ThinkingVisualizerProps {
  reasoningContent: string;
  isStreaming?: boolean;
  thinkingTime?: number; // in milliseconds
  defaultExpanded?: boolean;
}

export const ThinkingVisualizer: React.FC<ThinkingVisualizerProps> = memo(({
  reasoningContent,
  isStreaming = false,
  thinkingTime,
  defaultExpanded = false,
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);

  // Count reasoning steps (paragraphs or numbered items)
  const steps = reasoningContent.split(/\n\n+/).filter(s => s.trim().length > 0);
  const stepCount = steps.length;

  // Format thinking time
  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <Paper
      variant="outlined"
      sx={{
        my: 1,
        backgroundColor: 'rgba(156, 39, 176, 0.05)',
        borderColor: 'rgba(156, 39, 176, 0.3)',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          px: 2,
          py: 1,
          cursor: 'pointer',
          '&:hover': {
            backgroundColor: 'rgba(156, 39, 176, 0.1)',
          },
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <ThinkingIcon sx={{ color: 'secondary.main', mr: 1, fontSize: 20 }} />
        <Typography variant="subtitle2" sx={{ color: 'secondary.main', fontWeight: 500 }}>
          Reasoning
        </Typography>
        
        {/* Metadata chips */}
        <Box sx={{ display: 'flex', gap: 1, ml: 2 }}>
          {stepCount > 0 && (
            <Chip
              size="small"
              label={`${stepCount} step${stepCount !== 1 ? 's' : ''}`}
              sx={{
                height: 20,
                fontSize: '0.7rem',
                backgroundColor: 'rgba(156, 39, 176, 0.2)',
              }}
            />
          )}
          {thinkingTime && (
            <Chip
              size="small"
              icon={<TimerIcon sx={{ fontSize: 14 }} />}
              label={formatTime(thinkingTime)}
              sx={{
                height: 20,
                fontSize: '0.7rem',
                backgroundColor: 'rgba(156, 39, 176, 0.2)',
                '& .MuiChip-icon': { ml: 0.5, mr: -0.5 },
              }}
            />
          )}
        </Box>
        
        {isStreaming && (
          <Typography
            variant="caption"
            sx={{ ml: 2, color: 'secondary.light', fontStyle: 'italic' }}
          >
            thinking...
          </Typography>
        )}
        
        <IconButton
          size="small"
          sx={{
            ml: 'auto',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s',
          }}
        >
          <ExpandMoreIcon fontSize="small" />
        </IconButton>
      </Box>
      
      {/* Streaming indicator */}
      {isStreaming && <LinearProgress color="secondary" sx={{ height: 2 }} />}
      
      {/* Content */}
      <Collapse in={expanded}>
        <Box
          sx={{
            px: 2,
            py: 1.5,
            borderTop: '1px solid',
            borderColor: 'rgba(156, 39, 176, 0.2)',
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
            maxHeight: 400,
            overflow: 'auto',
          }}
        >
          <Typography
            variant="body2"
            sx={{
              color: 'text.secondary',
              fontSize: '0.85rem',
              lineHeight: 1.6,
              '& p': { my: 1 },
            }}
          >
            <MarkdownContent content={reasoningContent} />
          </Typography>
        </Box>
      </Collapse>
    </Paper>
  );
});

ThinkingVisualizer.displayName = 'ThinkingVisualizer';

export default ThinkingVisualizer;
