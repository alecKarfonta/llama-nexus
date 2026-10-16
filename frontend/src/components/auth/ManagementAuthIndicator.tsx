import React from 'react'
import { IconButton, Tooltip } from '@mui/material'
import LockIcon from '@mui/icons-material/Lock'
import LockOpenIcon from '@mui/icons-material/LockOpen'
import { useManagementAuthState } from '@/hooks/useManagementAuth'
import { clearManagementToken, requestManagementLogin } from '@/utils/managementAuth'

// [2026-10-16] | INFO | MANAGEMENT_AUTH_INDICATOR.RENDER | Subtle header
// affordance: locked (token stored — click signs out) or unlocked (401 seen
// with no valid token — click reopens the login dialog). Renders nothing when
// the server has no MANAGEMENT_AUTH_TOKEN (localhost default), so normal
// installs see zero extra chrome.
export const ManagementAuthIndicator: React.FC = () => {
  const authState = useManagementAuthState()

  if (authState === 'none') return null

  if (authState === 'authenticated') {
    return (
      <Tooltip title="Sign out of management access">
        <IconButton
          aria-label="Sign out of management access"
          onClick={clearManagementToken}
          sx={{
            mr: 0.5,
            width: 34,
            height: 34,
            borderRadius: 2,
            color: 'text.secondary',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            background: 'rgba(255, 255, 255, 0.03)',
            transition: 'background-color 0.15s, border-color 0.15s',
            '&:hover': {
              background: 'rgba(255, 255, 255, 0.08)',
              borderColor: 'rgba(255, 255, 255, 0.15)',
            },
          }}
        >
          <LockIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
    )
  }

  // 'required' (fresh 401) or 'signed-out' — offer the way back in.
  return (
    <Tooltip title="Management token required — click to unlock">
      <IconButton
        aria-label="Unlock management access"
        onClick={requestManagementLogin}
        sx={{
          mr: 0.5,
          width: 34,
          height: 34,
          borderRadius: 2,
          color: 'warning.main',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          background: 'rgba(245, 158, 11, 0.08)',
          transition: 'background-color 0.15s, border-color 0.15s',
          '&:hover': {
            background: 'rgba(245, 158, 11, 0.15)',
            borderColor: 'rgba(245, 158, 11, 0.4)',
          },
        }}
      >
        <LockOpenIcon sx={{ fontSize: 18 }} />
      </IconButton>
    </Tooltip>
  )
}

export default ManagementAuthIndicator
