import React from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  alpha,
} from '@mui/material'
import {
  Memory as ModelsIcon,
  Bolt as ActiveIcon,
  Category as FrameworksIcon,
  TrendingUp as PerformanceIcon,
} from '@mui/icons-material'

export type StatCardVariant = 'models' | 'active' | 'frameworks' | 'performance'

export interface StatCardProps {
  variant: StatCardVariant
  value: string | number
  label: string
  trend?: {
    value: number
    positive: boolean
  }
}

// Variant configuration with colors and gradients
const variantConfig: Record<StatCardVariant, {
  gradient: string
  color: string
  lightColor: string
  icon: React.ReactNode
  glowColor: string
}> = {
  models: {
    gradient: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
    color: '#6366f1',
    lightColor: '#818cf8',
    icon: <ModelsIcon />,
    glowColor: 'rgba(99, 102, 241, 0.4)',
  },
  active: {
    gradient: 'linear-gradient(135deg, #10b981 0%, #14b8a6 100%)',
    color: '#10b981',
    lightColor: '#34d399',
    icon: <ActiveIcon />,
    glowColor: 'rgba(16, 185, 129, 0.4)',
  },
  frameworks: {
    gradient: 'linear-gradient(135deg, #f59e0b 0%, #f97316 100%)',
    color: '#f59e0b',
    lightColor: '#fbbf24',
    icon: <FrameworksIcon />,
    glowColor: 'rgba(245, 158, 11, 0.4)',
  },
  performance: {
    gradient: 'linear-gradient(135deg, #06b6d4 0%, #0ea5e9 100%)',
    color: '#06b6d4',
    lightColor: '#22d3ee',
    icon: <PerformanceIcon />,
    glowColor: 'rgba(6, 182, 212, 0.4)',
  },
}

export const StatCard: React.FC<StatCardProps> = ({ variant, value, label, trend }) => {
  const config = variantConfig[variant]

  return (
    <Card
      sx={{
        position: 'relative',
        overflow: 'hidden',
        background: 'linear-gradient(145deg, rgba(30, 30, 63, 0.8) 0%, rgba(26, 26, 46, 0.9) 100%)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(255, 255, 255, 0.06)',
        borderRadius: 3,
        transition: 'all 0.3s ease-in-out',
        cursor: 'pointer',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: `0 12px 40px ${config.glowColor}`,
          borderColor: alpha(config.color, 0.3),
          '& .stat-icon-container': {
            transform: 'scale(1.1) rotate(5deg)',
          },
          '& .stat-glow': {
            opacity: 0.15,
          },
        },
      }}
    >
      {/* Background glow effect */}
      <Box
        className="stat-glow"
        sx={{
          position: 'absolute',
          top: -50,
          right: -50,
          width: 150,
          height: 150,
          borderRadius: '50%',
          background: config.gradient,
          filter: 'blur(60px)',
          opacity: 0.08,
          transition: 'opacity 0.3s ease-in-out',
        }}
      />
      
      <CardContent sx={{ 
        p: 2.5,
        '&:last-child': { pb: 2.5 },
        position: 'relative',
        zIndex: 1,
      }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          {/* Content */}
          <Box sx={{ flex: 1 }}>
            <Typography
              variant="overline"
              sx={{
                color: 'text.secondary',
                fontSize: '0.6875rem',
                fontWeight: 600,
                letterSpacing: '0.08em',
                mb: 0.5,
                display: 'block',
              }}
            >
              {label}
            </Typography>
            
            <Typography
              variant="h3"
              sx={{
                fontWeight: 700,
                fontSize: 'clamp(1.75rem, 4vw, 2.25rem)',
                lineHeight: 1.1,
                background: `linear-gradient(135deg, #f1f5f9 0%, ${config.lightColor} 100%)`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 1,
              }}
            >
              {value}
            </Typography>
            
            {/* Trend indicator */}
            {trend && (
              <Box sx={{ 
                display: 'inline-flex', 
                alignItems: 'center',
                px: 1,
                py: 0.25,
                borderRadius: 1,
                bgcolor: trend.positive ? alpha('#10b981', 0.1) : alpha('#ef4444', 0.1),
              }}>
                <Typography
                  variant="caption"
                  sx={{
                    color: trend.positive ? '#34d399' : '#f87171',
                    fontWeight: 600,
                    fontSize: '0.6875rem',
                  }}
                >
                  {trend.positive ? '+' : ''}{trend.value}%
                </Typography>
              </Box>
            )}
          </Box>
          
          {/* Icon */}
          <Box
            className="stat-icon-container"
            sx={{
              width: 48,
              height: 48,
              borderRadius: 2,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: config.gradient,
              boxShadow: `0 4px 14px ${config.glowColor}`,
              transition: 'transform 0.3s ease-in-out',
              '& .MuiSvgIcon-root': {
                fontSize: 24,
                color: 'white',
              },
            }}
          >
            {config.icon}
          </Box>
        </Box>
        
        {/* Bottom accent line */}
        <Box
          sx={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            height: 3,
            background: config.gradient,
            opacity: 0.6,
          }}
        />
      </CardContent>
    </Card>
  )
}
