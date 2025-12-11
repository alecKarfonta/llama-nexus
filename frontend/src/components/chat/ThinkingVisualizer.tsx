/**
 * ThinkingVisualizer Component
 * Displays reasoning/thinking content from chain-of-thought models
 * Supports DeepSeek R1, QwQ, Claude Extended Thinking, OpenAI o1, etc.
 */

import React, { useState, memo, useMemo } from 'react';
import {
  Box,
  Collapse,
  IconButton,
  Typography,
  Paper,
  LinearProgress,
  Chip,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  alpha,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Psychology as ThinkingIcon,
  Timer as TimerIcon,
  ContentCopy as CopyIcon,
  Check as CheckIcon,
  Lightbulb as InsightIcon,
  QuestionMark as QuestionIcon,
  Calculate as CalculateIcon,
  CompareArrows as CompareIcon,
  Summarize as SummaryIcon,
} from '@mui/icons-material';
import { MarkdownContent } from './MarkdownContent';

interface ThinkingVisualizerProps {
  reasoningContent: string;
  isStreaming?: boolean;
  thinkingTime?: number; // in milliseconds
  defaultExpanded?: boolean;
  modelName?: string;
}

// Detect the type of reasoning step
const detectStepType = (content: string): { icon: React.ReactNode; label: string; color: string } => {
  const lowerContent = content.toLowerCase();
  
  if (lowerContent.includes('let me') || lowerContent.includes('i need to') || lowerContent.includes('first')) {
    return { icon: <ThinkingIcon />, label: 'Planning', color: '#8b5cf6' };
  }
  if (lowerContent.includes('calculate') || lowerContent.includes('compute') || /\d+\s*[\+\-\*\/]\s*\d+/.test(content)) {
    return { icon: <CalculateIcon />, label: 'Calculating', color: '#f59e0b' };
  }
  if (lowerContent.includes('compare') || lowerContent.includes('versus') || lowerContent.includes('alternatively')) {
    return { icon: <CompareIcon />, label: 'Comparing', color: '#06b6d4' };
  }
  if (lowerContent.includes('?') || lowerContent.includes('what if') || lowerContent.includes('consider')) {
    return { icon: <QuestionIcon />, label: 'Questioning', color: '#ec4899' };
  }
  if (lowerContent.includes('therefore') || lowerContent.includes('conclusion') || lowerContent.includes('so ') || lowerContent.includes('thus')) {
    return { icon: <SummaryIcon />, label: 'Concluding', color: '#10b981' };
  }
  if (lowerContent.includes('insight') || lowerContent.includes('realize') || lowerContent.includes('notice')) {
    return { icon: <InsightIcon />, label: 'Insight', color: '#fbbf24' };
  }
  
  return { icon: <ThinkingIcon />, label: 'Reasoning', color: '#a855f7' };
};

export const ThinkingVisualizer: React.FC<ThinkingVisualizerProps> = memo(({
  reasoningContent,
  isStreaming = false,
  thinkingTime,
  defaultExpanded = false,
  modelName,
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [viewMode, setViewMode] = useState<'steps' | 'full'>('steps');
  const [copied, setCopied] = useState(false);

  // Parse reasoning into steps
  const steps = useMemo(() => {
    const rawSteps = reasoningContent.split(/\n\n+/).filter(s => s.trim().length > 0);
    return rawSteps.map((content, index) => ({
      id: index,
      content: content.trim(),
      ...detectStepType(content),
    }));
  }, [reasoningContent]);

  const stepCount = steps.length;

  // Format thinking time
  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  };

  // Word count
  const wordCount = reasoningContent.split(/\s+/).filter(w => w.length > 0).length;

  // Handle copy
  const handleCopy = async () => {
    await navigator.clipboard.writeText(reasoningContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Detect model type for styling
  const getModelBranding = () => {
    if (!modelName) return { name: 'Chain of Thought', color: '#a855f7' };
    const lower = modelName.toLowerCase();
    if (lower.includes('deepseek') || lower.includes('r1')) return { name: 'DeepSeek R1', color: '#3b82f6' };
    if (lower.includes('qwq')) return { name: 'QwQ', color: '#10b981' };
    if (lower.includes('o1') || lower.includes('o3')) return { name: 'OpenAI Reasoning', color: '#10b981' };
    if (lower.includes('claude')) return { name: 'Extended Thinking', color: '#f59e0b' };
    return { name: 'Chain of Thought', color: '#a855f7' };
  };

  const branding = getModelBranding();

  return (
    <Paper
      variant="outlined"
      sx={{
        my: 1,
        backgroundColor: alpha(branding.color, 0.03),
        borderColor: alpha(branding.color, 0.25),
        overflow: 'hidden',
        transition: 'all 0.2s ease',
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
            backgroundColor: alpha(branding.color, 0.06),
          },
        }}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Icon */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 28,
            height: 28,
            borderRadius: 1,
            bgcolor: alpha(branding.color, 0.15),
            color: branding.color,
            mr: 1.5,
          }}
        >
          <ThinkingIcon sx={{ fontSize: 16 }} />
        </Box>

        {/* Title */}
        <Typography 
          variant="subtitle2" 
          sx={{ 
            color: branding.color, 
            fontWeight: 600,
            fontSize: '0.8125rem',
          }}
        >
          {branding.name}
        </Typography>
        
        {/* Metadata chips */}
        <Box sx={{ display: 'flex', gap: 0.75, ml: 2, flexWrap: 'wrap' }}>
          {stepCount > 0 && (
            <Chip
              size="small"
              label={`${stepCount} step${stepCount !== 1 ? 's' : ''}`}
              sx={{
                height: 20,
                fontSize: '0.6875rem',
                bgcolor: alpha(branding.color, 0.1),
                color: branding.color,
                border: `1px solid ${alpha(branding.color, 0.2)}`,
              }}
            />
          )}
          {wordCount > 0 && (
            <Chip
              size="small"
              label={`${wordCount} words`}
              sx={{
                height: 20,
                fontSize: '0.6875rem',
                bgcolor: alpha(branding.color, 0.1),
                color: branding.color,
                border: `1px solid ${alpha(branding.color, 0.2)}`,
              }}
            />
          )}
          {thinkingTime && (
            <Chip
              size="small"
              icon={<TimerIcon sx={{ fontSize: 12 }} />}
              label={formatTime(thinkingTime)}
              sx={{
                height: 20,
                fontSize: '0.6875rem',
                bgcolor: alpha(branding.color, 0.1),
                color: branding.color,
                border: `1px solid ${alpha(branding.color, 0.2)}`,
                '& .MuiChip-icon': { ml: 0.5, mr: -0.5, color: branding.color },
              }}
            />
          )}
        </Box>
        
        {/* Streaming indicator */}
        {isStreaming && (
          <Box sx={{ display: 'flex', alignItems: 'center', ml: 2 }}>
            <Box
              sx={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                bgcolor: branding.color,
                animation: 'pulse 1.5s ease-in-out infinite',
                mr: 0.75,
                '@keyframes pulse': {
                  '0%, 100%': { opacity: 1 },
                  '50%': { opacity: 0.4 },
                },
              }}
            />
            <Typography
              variant="caption"
              sx={{ color: branding.color, fontStyle: 'italic', fontSize: '0.6875rem' }}
            >
              thinking...
            </Typography>
          </Box>
        )}

        {/* Actions */}
        <Box sx={{ ml: 'auto', display: 'flex', gap: 0.5 }}>
          {/* View Toggle */}
          {expanded && steps.length > 1 && (
            <Tooltip title={viewMode === 'steps' ? 'View full text' : 'View steps'}>
              <Chip
                size="small"
                label={viewMode === 'steps' ? 'Steps' : 'Full'}
                onClick={(e) => {
                  e.stopPropagation();
                  setViewMode(v => v === 'steps' ? 'full' : 'steps');
                }}
                sx={{
                  height: 20,
                  fontSize: '0.625rem',
                  cursor: 'pointer',
                  bgcolor: 'transparent',
                  border: `1px solid ${alpha('#fff', 0.2)}`,
                  '&:hover': { bgcolor: alpha('#fff', 0.05) },
                }}
              />
            </Tooltip>
          )}
          
          {/* Copy */}
          <Tooltip title={copied ? 'Copied!' : 'Copy reasoning'}>
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                handleCopy();
              }}
            >
              {copied ? <CheckIcon fontSize="small" /> : <CopyIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
          
          {/* Expand */}
          <IconButton
            size="small"
            sx={{
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s',
            }}
          >
            <ExpandMoreIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>
      
      {/* Streaming progress */}
      {isStreaming && (
        <LinearProgress
          sx={{
            height: 2,
            bgcolor: alpha(branding.color, 0.1),
            '& .MuiLinearProgress-bar': {
              bgcolor: branding.color,
            },
          }}
        />
      )}
      
      {/* Content */}
      <Collapse in={expanded}>
        <Box
          sx={{
            px: 2,
            py: 1.5,
            borderTop: '1px solid',
            borderColor: alpha(branding.color, 0.15),
            backgroundColor: alpha('#000', 0.2),
            maxHeight: 500,
            overflow: 'auto',
          }}
        >
          {viewMode === 'steps' && steps.length > 1 ? (
            /* Steps View */
            <Stepper orientation="vertical" sx={{ '& .MuiStepConnector-line': { minHeight: 16 } }}>
              {steps.map((step, index) => (
                <Step key={step.id} active expanded>
                  <StepLabel
                    StepIconComponent={() => (
                      <Box
                        sx={{
                          width: 24,
                          height: 24,
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          bgcolor: alpha(step.color, 0.15),
                          color: step.color,
                          '& .MuiSvgIcon-root': { fontSize: 14 },
                        }}
                      >
                        {step.icon}
                      </Box>
                    )}
                    sx={{
                      '& .MuiStepLabel-label': {
                        color: step.color,
                        fontWeight: 500,
                        fontSize: '0.75rem',
                      },
                    }}
                  >
                    {step.label}
                  </StepLabel>
                  <StepContent sx={{ borderColor: alpha(step.color, 0.2), ml: 1.5, pl: 2 }}>
                    <Typography
                      variant="body2"
                      sx={{
                        color: 'text.secondary',
                        fontSize: '0.8125rem',
                        lineHeight: 1.6,
                      }}
                    >
                      <MarkdownContent content={step.content} />
                    </Typography>
                  </StepContent>
                </Step>
              ))}
            </Stepper>
          ) : (
            /* Full Text View */
            <Typography
              variant="body2"
              sx={{
                color: 'text.secondary',
                fontSize: '0.8125rem',
                lineHeight: 1.7,
                '& p': { my: 1 },
              }}
            >
              <MarkdownContent content={reasoningContent} />
            </Typography>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
});

ThinkingVisualizer.displayName = 'ThinkingVisualizer';

export default ThinkingVisualizer;
