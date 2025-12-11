import React from 'react'
import { Box, Skeleton, Paper } from '@mui/material'

interface PageSkeletonProps {
  variant?: 'dashboard' | 'list' | 'form' | 'detail' | 'chat'
}

export const PageSkeleton: React.FC<PageSkeletonProps> = ({ variant = 'dashboard' }) => {
  if (variant === 'chat') {
    return (
      <Box sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Skeleton variant="text" width={200} height={32} />
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Skeleton variant="circular" width={32} height={32} />
            <Skeleton variant="circular" width={32} height={32} />
            <Skeleton variant="circular" width={32} height={32} />
          </Box>
        </Box>
        
        {/* Messages */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-start' }}>
            <Skeleton variant="rounded" width="60%" height={80} sx={{ borderRadius: 2 }} />
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Skeleton variant="rounded" width="40%" height={48} sx={{ borderRadius: 2 }} />
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'flex-start' }}>
            <Skeleton variant="rounded" width="70%" height={120} sx={{ borderRadius: 2 }} />
          </Box>
        </Box>
        
        {/* Input */}
        <Skeleton variant="rounded" height={56} sx={{ borderRadius: 2 }} />
      </Box>
    )
  }

  if (variant === 'list') {
    return (
      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Skeleton variant="text" width={200} height={32} />
          <Skeleton variant="rounded" width={120} height={36} sx={{ borderRadius: 2 }} />
        </Box>
        
        {/* List items */}
        {[1, 2, 3, 4, 5].map((i) => (
          <Paper key={i} sx={{ p: 2, mb: 1.5, bgcolor: 'rgba(255,255,255,0.02)' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Skeleton variant="circular" width={40} height={40} />
              <Box sx={{ flex: 1 }}>
                <Skeleton variant="text" width="40%" height={24} />
                <Skeleton variant="text" width="60%" height={16} />
              </Box>
              <Skeleton variant="rounded" width={80} height={28} sx={{ borderRadius: 1 }} />
            </Box>
          </Paper>
        ))}
      </Box>
    )
  }

  if (variant === 'form') {
    return (
      <Box sx={{ p: 3, maxWidth: 800 }}>
        {/* Header */}
        <Skeleton variant="text" width={250} height={36} sx={{ mb: 1 }} />
        <Skeleton variant="text" width={400} height={20} sx={{ mb: 4 }} />
        
        {/* Form fields */}
        {[1, 2, 3].map((i) => (
          <Box key={i} sx={{ mb: 3 }}>
            <Skeleton variant="text" width={120} height={20} sx={{ mb: 1 }} />
            <Skeleton variant="rounded" height={48} sx={{ borderRadius: 1 }} />
          </Box>
        ))}
        
        {/* Actions */}
        <Box sx={{ display: 'flex', gap: 2, mt: 4 }}>
          <Skeleton variant="rounded" width={100} height={40} sx={{ borderRadius: 2 }} />
          <Skeleton variant="rounded" width={100} height={40} sx={{ borderRadius: 2 }} />
        </Box>
      </Box>
    )
  }

  if (variant === 'detail') {
    return (
      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
          <Skeleton variant="circular" width={48} height={48} />
          <Box>
            <Skeleton variant="text" width={200} height={28} />
            <Skeleton variant="text" width={300} height={16} />
          </Box>
        </Box>
        
        {/* Content cards */}
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
          {[1, 2, 3, 4].map((i) => (
            <Paper key={i} sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)' }}>
              <Skeleton variant="text" width={100} height={20} sx={{ mb: 2 }} />
              <Skeleton variant="text" width="80%" />
              <Skeleton variant="text" width="60%" />
            </Paper>
          ))}
        </Box>
      </Box>
    )
  }

  // Default: dashboard variant
  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Skeleton variant="text" width={200} height={36} />
          <Skeleton variant="text" width={300} height={20} />
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Skeleton variant="rounded" width={100} height={36} sx={{ borderRadius: 2 }} />
          <Skeleton variant="rounded" width={100} height={36} sx={{ borderRadius: 2 }} />
        </Box>
      </Box>
      
      {/* Stats cards */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: 'repeat(4, 1fr)' }, gap: 2, mb: 3 }}>
        {[1, 2, 3, 4].map((i) => (
          <Paper key={i} sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Skeleton variant="text" width={80} height={16} />
              <Skeleton variant="circular" width={24} height={24} />
            </Box>
            <Skeleton variant="text" width={100} height={32} />
            <Skeleton variant="text" width={60} height={14} />
          </Paper>
        ))}
      </Box>
      
      {/* Main content area */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' }, gap: 2 }}>
        <Paper sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', height: 300 }}>
          <Skeleton variant="text" width={150} height={24} sx={{ mb: 2 }} />
          <Skeleton variant="rounded" height={220} sx={{ borderRadius: 1 }} />
        </Paper>
        <Paper sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)', height: 300 }}>
          <Skeleton variant="text" width={120} height={24} sx={{ mb: 2 }} />
          {[1, 2, 3, 4].map((i) => (
            <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
              <Skeleton variant="circular" width={32} height={32} />
              <Box sx={{ flex: 1 }}>
                <Skeleton variant="text" width="70%" height={16} />
                <Skeleton variant="text" width="40%" height={12} />
              </Box>
            </Box>
          ))}
        </Paper>
      </Box>
    </Box>
  )
}

export default PageSkeleton
