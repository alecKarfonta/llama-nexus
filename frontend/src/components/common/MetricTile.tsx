import React from 'react'
import { Box, Typography } from '@mui/material'
import { alpha } from '@mui/material/styles'
import type { SxProps, Theme } from '@mui/material/styles'

export type MetricTileVariant = 'compact' | 'horizontal' | 'centered'

export interface MetricTileProps {
  label: string
  value: React.ReactNode
  detail?: React.ReactNode
  icon?: React.ReactNode
  color: string
  variant?: MetricTileVariant
  sx?: SxProps<Theme>
  valueSx?: SxProps<Theme>
  pulse?: boolean
}

export const MetricTile: React.FC<MetricTileProps> = ({
  label,
  value,
  detail,
  icon,
  color,
  variant = 'compact',
  sx,
  valueSx,
  pulse = false,
}) => {
  const pulseIndicator = pulse ? (
    <Box
      sx={{
        position: 'absolute',
        top: 8,
        right: 8,
        width: 8,
        height: 8,
        borderRadius: '50%',
        bgcolor: color,
        animation: 'metricTilePulse 1.5s infinite',
        '@keyframes metricTilePulse': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.4 },
        },
      }}
    />
  ) : null

  if (variant === 'horizontal') {
    return (
      <Box
        sx={[
          {
            p: 1.25,
            borderRadius: '8px',
            bgcolor: alpha(color, 0.08),
            border: `1px solid ${alpha(color, 0.15)}`,
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            minWidth: 0,
            position: 'relative',
            overflow: 'hidden',
            transition: 'background-color 0.15s',
            '&:hover': {
              bgcolor: alpha(color, 0.12),
            },
          },
          ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
        ]}
      >
        {pulseIndicator}
        {icon && (
          <Box
            sx={{
              width: 32,
              height: 32,
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: alpha(color, 0.16),
              color,
              flexShrink: 0,
              '& .MuiSvgIcon-root': { fontSize: 18 },
            }}
          >
            {icon}
          </Box>
        )}
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem', display: 'block' }} noWrap>
            {label}
          </Typography>
          <Typography
            sx={[
              { fontWeight: 700, fontSize: '1rem', color: 'text.primary', lineHeight: 1.2 },
              ...(Array.isArray(valueSx) ? valueSx : valueSx ? [valueSx] : []),
            ]}
            noWrap
            title={String(value)}
          >
            {value}
          </Typography>
          {detail && (
            <Typography variant="caption" sx={{ color: alpha(color, 0.8), fontSize: '0.65rem', display: 'block' }} noWrap title={String(detail)}>
              {detail}
            </Typography>
          )}
        </Box>
      </Box>
    )
  }

  if (variant === 'centered') {
    return (
      <Box
        sx={[
          {
            p: 1.25,
            borderRadius: '8px',
            bgcolor: alpha(color, 0.08),
            border: `1px solid ${alpha(color, 0.15)}`,
            textAlign: 'center',
            minWidth: 0,
            position: 'relative',
            overflow: 'hidden',
            transition: 'background-color 0.15s',
            '&:hover': { bgcolor: alpha(color, 0.12) },
          },
          ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
        ]}
      >
        {pulseIndicator}
        {icon && <Box sx={{ mb: 0.75, color, '& .MuiSvgIcon-root': { fontSize: 22 } }}>{icon}</Box>}
        <Typography
          sx={[
            { fontWeight: 700, fontSize: '1.2rem', color: 'text.primary', lineHeight: 1 },
            ...(Array.isArray(valueSx) ? valueSx : valueSx ? [valueSx] : []),
          ]}
          noWrap
          title={String(value)}
        >
          {value}
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem', display: 'block' }} noWrap title={label}>
          {label}
        </Typography>
        {detail && (
          <Typography variant="caption" sx={{ color: alpha(color, 0.8), fontSize: '0.65rem', display: 'block', mt: 0.25 }} noWrap title={String(detail)}>
            {detail}
          </Typography>
        )}
      </Box>
    )
  }

  return (
    <Box
      sx={[
        {
          height: '100%',
          minHeight: 82,
          p: 1,
          borderRadius: '8px',
          border: `1px solid ${alpha(color, 0.22)}`,
          bgcolor: alpha(color, 0.08),
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          gap: 0.75,
          minWidth: 0,
          position: 'relative',
          overflow: 'hidden',
        },
        ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
      ]}
    >
      {pulseIndicator}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, minWidth: 0 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, minWidth: 0 }} noWrap title={label}>
          {label}
        </Typography>
        {icon && (
          <Box
            sx={{
              width: 24,
              height: 24,
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color,
              bgcolor: alpha(color, 0.13),
              flexShrink: 0,
              '& .MuiSvgIcon-root': { fontSize: 15 },
            }}
          >
            {icon}
          </Box>
        )}
      </Box>
      <Box sx={{ minWidth: 0 }}>
        <Typography
          sx={[
            {
              fontWeight: 800,
              fontSize: '0.95rem',
              lineHeight: 1.2,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            },
            ...(Array.isArray(valueSx) ? valueSx : valueSx ? [valueSx] : []),
          ]}
          title={String(value)}
        >
          {value}
        </Typography>
        {detail && (
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: 'block', mt: 0.4, textTransform: 'capitalize', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            title={String(detail)}
          >
            {detail}
          </Typography>
        )}
      </Box>
    </Box>
  )
}
