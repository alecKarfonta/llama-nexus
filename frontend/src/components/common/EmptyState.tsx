import React from 'react'
import { Box, Typography, Button, alpha } from '@mui/material'
import {
  Inbox as InboxIcon,
  Search as SearchIcon,
  CloudUpload as UploadIcon,
  Error as ErrorIcon,
  Build as BuildIcon,
  Chat as ChatIcon,
  Description as DocumentIcon,
  AccountTree as WorkflowIcon,
} from '@mui/icons-material'

type EmptyStateVariant = 
  | 'default'
  | 'search'
  | 'upload'
  | 'error'
  | 'chat'
  | 'documents'
  | 'workflows'
  | 'building'

interface EmptyStateProps {
  variant?: EmptyStateVariant
  title?: string
  description?: string
  action?: {
    label: string
    onClick: () => void
  }
  secondaryAction?: {
    label: string
    onClick: () => void
  }
  icon?: React.ReactNode
  compact?: boolean
}

const variantConfig: Record<EmptyStateVariant, { icon: React.ReactNode; title: string; description: string; color: string }> = {
  default: {
    icon: <InboxIcon />,
    title: 'No items yet',
    description: 'Get started by creating your first item.',
    color: '#6366f1',
  },
  search: {
    icon: <SearchIcon />,
    title: 'No results found',
    description: 'Try adjusting your search or filters.',
    color: '#8b5cf6',
  },
  upload: {
    icon: <UploadIcon />,
    title: 'No files uploaded',
    description: 'Drag and drop files here or click to upload.',
    color: '#10b981',
  },
  error: {
    icon: <ErrorIcon />,
    title: 'Something went wrong',
    description: 'An error occurred. Please try again.',
    color: '#ef4444',
  },
  chat: {
    icon: <ChatIcon />,
    title: 'Start a conversation',
    description: 'Send a message to begin chatting with the AI.',
    color: '#06b6d4',
  },
  documents: {
    icon: <DocumentIcon />,
    title: 'No documents',
    description: 'Upload documents to enable RAG capabilities.',
    color: '#10b981',
  },
  workflows: {
    icon: <WorkflowIcon />,
    title: 'No workflows',
    description: 'Create your first workflow to automate tasks.',
    color: '#a855f7',
  },
  building: {
    icon: <BuildIcon />,
    title: 'Under construction',
    description: 'This feature is coming soon.',
    color: '#f59e0b',
  },
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  variant = 'default',
  title,
  description,
  action,
  secondaryAction,
  icon,
  compact = false,
}) => {
  const config = variantConfig[variant]
  const displayIcon = icon || config.icon
  const displayTitle = title || config.title
  const displayDescription = description || config.description

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        py: compact ? 4 : 8,
        px: 3,
      }}
    >
      {/* Icon */}
      <Box
        sx={{
          width: compact ? 56 : 80,
          height: compact ? 56 : 80,
          borderRadius: 3,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: alpha(config.color, 0.1),
          color: config.color,
          mb: 3,
          position: 'relative',
          '& .MuiSvgIcon-root': {
            fontSize: compact ? 28 : 40,
          },
          '&::before': {
            content: '""',
            position: 'absolute',
            inset: -8,
            borderRadius: 4,
            border: `2px dashed ${alpha(config.color, 0.2)}`,
          },
        }}
      >
        {displayIcon}
      </Box>

      {/* Title */}
      <Typography
        variant={compact ? 'h6' : 'h5'}
        sx={{
          fontWeight: 600,
          color: 'text.primary',
          mb: 1,
          fontSize: compact ? '1rem' : '1.25rem',
        }}
      >
        {displayTitle}
      </Typography>

      {/* Description */}
      <Typography
        variant="body2"
        sx={{
          color: 'text.secondary',
          maxWidth: 360,
          mb: action ? 3 : 0,
          fontSize: compact ? '0.8125rem' : '0.875rem',
          lineHeight: 1.6,
        }}
      >
        {displayDescription}
      </Typography>

      {/* Actions */}
      {(action || secondaryAction) && (
        <Box sx={{ display: 'flex', gap: 1.5 }}>
          {action && (
            <Button
              variant="contained"
              onClick={action.onClick}
              sx={{
                bgcolor: config.color,
                '&:hover': {
                  bgcolor: alpha(config.color, 0.9),
                },
                textTransform: 'none',
                fontWeight: 600,
                px: 3,
              }}
            >
              {action.label}
            </Button>
          )}
          {secondaryAction && (
            <Button
              variant="outlined"
              onClick={secondaryAction.onClick}
              sx={{
                borderColor: alpha(config.color, 0.3),
                color: config.color,
                '&:hover': {
                  borderColor: config.color,
                  bgcolor: alpha(config.color, 0.08),
                },
                textTransform: 'none',
                fontWeight: 600,
                px: 3,
              }}
            >
              {secondaryAction.label}
            </Button>
          )}
        </Box>
      )}
    </Box>
  )
}

export default EmptyState
