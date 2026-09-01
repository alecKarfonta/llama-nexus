import React from 'react'
import { Chip } from '@mui/material'
import type { ChipProps } from '@mui/material'
import { alpha } from '@mui/material/styles'
import type { SxProps, Theme } from '@mui/material/styles'
import { accentColors } from '@/utils/theme'

export const statusPalette = accentColors

export interface StatusChipConfig {
  color: string
  label: string
}

export type StatusChipConfigMap = Record<string, Partial<StatusChipConfig>>

const defaultStatusConfig: Record<string, StatusChipConfig> = {
  archived: { color: statusPalette.slate, label: 'Archived' },
  cancelled: { color: statusPalette.slate, label: 'Cancelled' },
  completed: { color: statusPalette.success, label: 'Completed' },
  draft: { color: statusPalette.slate, label: 'Draft' },
  error: { color: statusPalette.rose, label: 'Error' },
  exported: { color: statusPalette.warning, label: 'Exported' },
  failed: { color: statusPalette.rose, label: 'Failed' },
  merged: { color: statusPalette.purple, label: 'Merged' },
  paused: { color: statusPalette.warning, label: 'Paused' },
  pending: { color: statusPalette.warning, label: 'Pending' },
  processing: { color: statusPalette.info, label: 'Processing' },
  queued: { color: statusPalette.purple, label: 'Queued' },
  ready: { color: statusPalette.success, label: 'Ready' },
  running: { color: statusPalette.info, label: 'Running' },
  training: { color: statusPalette.info, label: 'Training' },
}

export interface StatusChipProps extends Omit<ChipProps, 'color' | 'label' | 'sx'> {
  status: string
  label?: string
  statusConfig?: StatusChipConfigMap
  height?: number
  sx?: SxProps<Theme>
}

export const getStatusChipConfig = (
  status: string,
  statusConfig: StatusChipConfigMap = {},
): StatusChipConfig => {
  const normalizedStatus = status.toLowerCase()
  const baseConfig = defaultStatusConfig[normalizedStatus] ?? {
    color: statusPalette.slate,
    label: status,
  }
  const override = statusConfig[normalizedStatus] ?? statusConfig[status] ?? {}

  return {
    color: override.color ?? baseConfig.color,
    label: override.label ?? baseConfig.label,
  }
}

export const StatusChip: React.FC<StatusChipProps> = ({
  status,
  label,
  statusConfig,
  height = 24,
  size = 'small',
  sx,
  ...chipProps
}) => {
  const config = getStatusChipConfig(status, statusConfig)

  return (
    <Chip
      label={label ?? config.label}
      size={size}
      sx={[
        {
          bgcolor: alpha(config.color, 0.1),
          color: config.color,
          border: `1px solid ${alpha(config.color, 0.3)}`,
          fontWeight: 600,
          fontSize: '0.7rem',
          height,
        },
        ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
      ]}
      {...chipProps}
    />
  )
}
