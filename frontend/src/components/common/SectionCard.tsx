/* eslint-disable react-refresh/only-export-components */
import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Collapse,
  IconButton,
  Typography,
} from '@mui/material'
import {
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
} from '@mui/icons-material'
import { alpha } from '@mui/material/styles'
import type { SxProps, Theme } from '@mui/material/styles'

export interface SectionCardProps {
  title: string
  subtitle?: string
  icon?: React.ReactNode
  accentColor?: string
  children: React.ReactNode
  action?: React.ReactNode
  collapsible?: boolean
  defaultExpanded?: boolean
  fullHeight?: boolean
  hideHeader?: boolean
  headerSx?: SxProps<Theme>
  contentSx?: SxProps<Theme>
  sx?: SxProps<Theme>
}

export const optionalSectionGridSx: SxProps<Theme> = {
  display: 'grid',
  gridTemplateColumns: {
    xs: '1fr',
    md: 'repeat(2, minmax(0, 1fr))',
  },
  gap: 1,
  alignItems: 'start',
  '& > .MuiCard-root:only-child': {
    gridColumn: { md: '1 / -1' },
  },
  '& > .MuiCard-root[data-expanded="true"]': {
    gridColumn: { md: '1 / -1' },
  },
}

export const compactCollapsibleCardSx: SxProps<Theme> = {
  minWidth: 0,
  '&[data-expanded="false"] .MuiCardContent-root': {
    px: 1.25,
    py: 1,
    '&:last-child': {
      pb: 1,
    },
  },
  '&[data-expanded="false"] .MuiTypography-h6': {
    fontSize: '0.875rem',
  },
}

export const SectionCard: React.FC<SectionCardProps> = ({
  title,
  subtitle,
  icon,
  accentColor = '#6366f1',
  children,
  action,
  collapsible = false,
  defaultExpanded = true,
  fullHeight = false,
  hideHeader = false,
  headerSx,
  contentSx,
  sx,
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded)
    const content = (
      <Box sx={{ maxWidth: '100%', minWidth: 0, overflow: 'visible', '& > *': { minWidth: 0 } }}>
        {children}
      </Box>
    )

  return (
    <Card
      data-expanded={expanded ? 'true' : 'false'}
      sx={[
        {
          position: 'relative',
          overflow: 'hidden',
          bgcolor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: '8px',
          height: fullHeight ? '100%' : undefined,
          transition: 'border-color 0.15s ease',
          '&:hover': {
            borderColor: alpha(accentColor, 0.28),
          },
        },
        ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
      ]}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 2,
          bgcolor: accentColor,
        }}
      />

      <CardContent
        sx={[
          { p: hideHeader ? 0 : 1.5, '&:last-child': { pb: hideHeader ? 0 : 1.5 } },
          ...(Array.isArray(contentSx) ? contentSx : contentSx ? [contentSx] : []),
        ]}
      >
        {!hideHeader && (
          <Box
            sx={[
              { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap', mb: expanded ? 1.25 : 0 },
              ...(Array.isArray(headerSx) ? headerSx : headerSx ? [headerSx] : []),
            ]}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
              {icon && (
                <Box
                  sx={{
                    width: 28,
                    height: 28,
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    bgcolor: alpha(accentColor, 0.1),
                    color: accentColor,
                    flexShrink: 0,
                    '& .MuiSvgIcon-root': {
                      fontSize: 17,
                    },
                  }}
                >
                  {icon}
                </Box>
              )}
              <Box sx={{ minWidth: 0 }}>
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 600,
                    fontSize: '0.9375rem',
                    color: 'text.primary',
                    lineHeight: 1.25,
                  }}
                >
                  {title}
                </Typography>
                {subtitle && (
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'text.secondary',
                      fontSize: '0.75rem',
                    }}
                  >
                    {subtitle}
                  </Typography>
                )}
              </Box>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexShrink: 0, flexWrap: 'wrap', justifyContent: 'flex-end', ml: 'auto', maxWidth: '100%' }}>
              {action}
              {collapsible && (
                <IconButton
                  size="small"
                  onClick={() => setExpanded((value) => !value)}
                  aria-label={expanded ? `Collapse ${title}` : `Expand ${title}`}
                >
                  {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </IconButton>
              )}
            </Box>
          </Box>
        )}

        {collapsible ? <Collapse in={expanded} unmountOnExit>{content}</Collapse> : content}
      </CardContent>
    </Card>
  )
}
