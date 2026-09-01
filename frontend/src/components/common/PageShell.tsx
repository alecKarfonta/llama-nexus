import React from 'react'
import { Box } from '@mui/material'
import type { SxProps, Theme } from '@mui/material/styles'

export interface PageShellProps {
  children: React.ReactNode
  sx?: SxProps<Theme>
}

export const PageShell: React.FC<PageShellProps> = ({ children, sx }) => (
  <Box
    sx={[
      {
        width: '100%',
        maxWidth: '100vw',
        overflow: 'hidden',
        px: { xs: 1.5, sm: 2, md: 2.5 },
        py: 1.5,
        boxSizing: 'border-box',
        minHeight: '100%',
      },
      ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
    ]}
  >
    {children}
  </Box>
)
