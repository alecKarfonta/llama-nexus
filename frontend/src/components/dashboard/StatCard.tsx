import React from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  styled,
} from '@mui/material'

export type StatCardVariant = 'models' | 'active' | 'frameworks' | 'performance'

export interface StatCardProps {
  variant: StatCardVariant
  value: string | number
  label: string
}

// Styled components
const StyledCard = styled(Card)<{ variant: StatCardVariant }>(({ variant, theme }) => ({
  background: 'white',
  borderLeft: '4px solid',
  borderLeftColor: 
    variant === 'models' ? theme.palette.primary.main : 
    variant === 'active' ? theme.palette.success.main : 
    variant === 'frameworks' ? theme.palette.warning.main : 
    theme.palette.secondary.main,
}))

const StatValue = styled(Typography)(({ theme }) => ({
  fontSize: 'clamp(1.5rem, 4vw, 2rem)',
  fontWeight: 'bold',
  marginBottom: theme.spacing(0.5),
  lineHeight: 1.2,
}))

const StatLabel = styled(Typography)(({ theme }) => ({
  color: theme.palette.grey[500],
  fontSize: 'clamp(0.75rem, 2vw, 0.875rem)',
  lineHeight: 1.3,
}))

export const StatCard: React.FC<StatCardProps> = ({ variant, value, label }) => {
  return (
    <StyledCard variant={variant}>
      <CardContent sx={{ 
        padding: { xs: 2, sm: 3 },
        minHeight: { xs: 100, sm: 120 },
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        '&:last-child': {
          paddingBottom: { xs: 2, sm: 3 }
        }
      }}>
        <StatValue>{value}</StatValue>
        <StatLabel>{label}</StatLabel>
      </CardContent>
    </StyledCard>
  )
}
