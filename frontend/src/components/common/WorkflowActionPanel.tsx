import React from 'react'
import { Box, Button, Chip, Typography } from '@mui/material'
import type { ButtonProps, ChipProps } from '@mui/material'
import { alpha } from '@mui/material/styles'
import type { SxProps, Theme } from '@mui/material/styles'

export interface WorkflowAction {
  label: string
  icon?: React.ReactNode
  detail?: string
  onClick: () => void
  emphasized?: boolean
  color?: string
  endIcon?: React.ReactNode
  variant?: ButtonProps['variant']
}

interface WorkflowActionPanelProps {
  eyebrow?: string
  title: React.ReactNode
  subtitle?: React.ReactNode
  chipLabel?: string
  chipColor?: ChipProps['color']
  leadingIcon?: React.ReactNode
  accentColor?: string
  actions?: WorkflowAction[]
  emphasizedAction?: string
  actionStyle?: 'simple' | 'cards'
  actionGridTemplateColumns?: string | Record<string, string>
  dense?: boolean
  sx?: SxProps<Theme>
}

export const WorkflowActionPanel: React.FC<WorkflowActionPanelProps> = ({
  eyebrow,
  title,
  subtitle,
  chipLabel,
  chipColor = 'success',
  leadingIcon,
  accentColor = '#10b981',
  actions = [],
  emphasizedAction,
  actionStyle = 'simple',
  actionGridTemplateColumns,
  dense = false,
  sx,
}) => {
  const hasActions = actions.length > 0
  const defaultColumns = actionStyle === 'cards'
    ? { xs: '1fr', sm: `repeat(${Math.min(actions.length || 1, 3)}, minmax(0, 1fr))` }
    : { xs: '1fr', sm: `repeat(${Math.min(actions.length || 1, 3)}, minmax(0, 1fr))` }

  return (
    <Box
      sx={[
        {
          p: dense ? 1.25 : 1.5,
          borderRadius: '8px',
          border: `1px solid ${alpha(accentColor, 0.24)}`,
          bgcolor: alpha(accentColor, 0.055),
        },
        ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
      ]}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 1,
          mb: hasActions || subtitle ? 1 : 0,
          minWidth: 0,
        }}
      >
        <Box sx={{ minWidth: 0, display: 'flex', alignItems: 'center', gap: 1 }}>
          {leadingIcon && (
            <Box
              sx={{
                width: 34,
                height: 34,
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: accentColor,
                bgcolor: alpha(accentColor, 0.14),
                flexShrink: 0,
              }}
            >
              {leadingIcon}
            </Box>
          )}
          <Box sx={{ minWidth: 0 }}>
            {eyebrow && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontWeight: 800 }}>
                {eyebrow}
              </Typography>
            )}
            <Typography
              variant={dense ? 'body2' : 'body1'}
              sx={{ fontWeight: 800, overflowWrap: 'anywhere' }}
              noWrap={typeof title === 'string'}
              title={typeof title === 'string' ? title : undefined}
            >
              {title}
            </Typography>
          </Box>
        </Box>
        {chipLabel && (
          <Chip
            size="small"
            label={chipLabel}
            color={chipColor}
            variant="outlined"
            sx={{ height: 24, fontWeight: 700, flexShrink: 0 }}
          />
        )}
      </Box>

      {subtitle && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: hasActions ? 1 : 0 }}>
          {subtitle}
        </Typography>
      )}

      {hasActions && (
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: actionGridTemplateColumns || defaultColumns,
            gap: dense ? 0.75 : 1,
          }}
        >
          {actions.map((action) => {
            const actionColor = action.color || accentColor
            const isEmphasized = action.emphasized || action.label === emphasizedAction

            if (actionStyle === 'cards') {
              return (
                <Button
                  key={action.label}
                  variant={action.variant || 'outlined'}
                  startIcon={action.icon}
                  endIcon={action.endIcon}
                  onClick={action.onClick}
                  sx={{
                    justifyContent: 'space-between',
                    minHeight: 48,
                    borderRadius: '8px',
                    borderColor: alpha(actionColor, 0.35),
                    bgcolor: alpha(actionColor, 0.05),
                    color: 'text.primary',
                    textTransform: 'none',
                    '&:hover': {
                      borderColor: alpha(actionColor, 0.62),
                      bgcolor: alpha(actionColor, 0.1),
                    },
                    '& .MuiButton-startIcon': {
                      color: actionColor,
                    },
                    '& .MuiButton-endIcon': {
                      color: alpha('#fff', 0.45),
                    },
                  }}
                >
                  <Box sx={{ minWidth: 0, textAlign: 'left' }}>
                    <Typography sx={{ fontWeight: 800, fontSize: '0.78rem', lineHeight: 1.2 }}>
                      {action.label}
                    </Typography>
                    {action.detail && (
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }} noWrap>
                        {action.detail}
                      </Typography>
                    )}
                  </Box>
                </Button>
              )
            }

            return (
              <Button
                key={action.label}
                size="small"
                variant={action.variant || (isEmphasized ? 'contained' : 'outlined')}
                startIcon={action.icon}
                endIcon={action.endIcon}
                onClick={action.onClick}
                sx={{ borderRadius: '8px', fontWeight: 700, textTransform: 'none', minHeight: dense ? 34 : 36 }}
              >
                {action.label}
              </Button>
            )
          })}
        </Box>
      )}
    </Box>
  )
}
