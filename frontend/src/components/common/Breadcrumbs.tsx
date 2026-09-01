import React from 'react'
import { useLocation, Link as RouterLink } from 'react-router-dom'
import { Breadcrumbs as MuiBreadcrumbs, Link, Typography, Box } from '@mui/material'
import type { SxProps, Theme } from '@mui/material/styles'
import {
  Home as HomeIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material'
import { getNavigationTrail } from '@/config/navigation'

interface BreadcrumbsProps {
  maxItems?: number
  showHome?: boolean
  sx?: SxProps<Theme>
}

export const Breadcrumbs: React.FC<BreadcrumbsProps> = ({ 
  maxItems = 3,
  showHome = true,
  sx,
}) => {
  const location = useLocation()
  const trail = getNavigationTrail(location.pathname)
  const items = (showHome ? trail : trail.slice(1)).map((item, index, allItems) => ({
    ...item,
    label: item.path === '/dashboard' && showHome ? 'Home' : item.label,
    isLast: index === allItems.length - 1,
  }))

  // Don't render if we're at root or only have home
  if (items.length <= 1) {
    return null
  }

  const containerSx: SxProps<Theme> = sx
    ? (Array.isArray(sx) ? [{ mb: 2 }, ...sx] : [{ mb: 2 }, sx])
    : { mb: 2 }

  return (
    <Box sx={containerSx}>
      <MuiBreadcrumbs
        separator={<ChevronRightIcon sx={{ fontSize: 16, color: 'text.disabled' }} />}
        maxItems={maxItems}
        sx={{
          '& .MuiBreadcrumbs-ol': {
            alignItems: 'center',
          },
        }}
      >
        {items.map((item, index) => {
          if (item.isLast) {
            return (
              <Typography
                key={item.path}
                variant="body2"
                sx={{
                  color: 'text.primary',
                  fontWeight: 500,
                  fontSize: '0.8125rem',
                }}
              >
                {item.label}
              </Typography>
            )
          }

          return (
            <Link
              key={item.path}
              component={RouterLink}
              to={item.path}
              underline="hover"
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                color: 'text.secondary',
                fontSize: '0.8125rem',
                transition: 'color 0.2s ease',
                '&:hover': {
                  color: 'primary.main',
                },
              }}
            >
              {index === 0 && showHome && (
                <HomeIcon sx={{ fontSize: 16 }} />
              )}
              {item.label}
            </Link>
          )
        })}
      </MuiBreadcrumbs>
    </Box>
  )
}

export default Breadcrumbs
