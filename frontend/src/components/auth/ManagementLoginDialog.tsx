import React, { useEffect, useState } from 'react'
import {
  Alert,
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  InputAdornment,
  TextField,
  Typography,
} from '@mui/material'
import LockIcon from '@mui/icons-material/Lock'
import Visibility from '@mui/icons-material/Visibility'
import VisibilityOff from '@mui/icons-material/VisibilityOff'
import { apiService } from '@/services/api'
import { isManagementAuthError, setManagementToken } from '@/utils/managementAuth'
import { useToast } from '@/contexts/ToastContext'

interface ManagementLoginDialogProps {
  open: boolean
  onClose: () => void
}

// [2026-10-16] | INFO | MANAGEMENT_LOGIN_DIALOG.RENDER | Modal gate shown only
// when the management API returned a real 401 (on localhost auth is off, so
// nothing ever probes or pops here). Remember is always-on: the token persists
// in localStorage and is re-attached automatically by the api interceptor.
export const ManagementLoginDialog: React.FC<ManagementLoginDialogProps> = ({ open, onClose }) => {
  const toast = useToast()
  const [token, setToken] = useState('')
  const [showToken, setShowToken] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [verifying, setVerifying] = useState(false)

  // Reset transient state whenever the dialog opens.
  useEffect(() => {
    if (open) {
      setToken('')
      setShowToken(false)
      setError(null)
      setVerifying(false)
    }
  }, [open])

  const handleUnlock = async () => {
    const candidate = token.trim()
    if (!candidate) {
      setError('Enter the management token to continue.')
      return
    }

    setVerifying(true)
    setError(null)
    try {
      // [2026-10-16] | DEBUG | MANAGEMENT_LOGIN_DIALOG.VERIFY | The candidate
      // rides as an explicit header so the interceptor verifies it without
      // needing it stored first. A 200 stores it and the interceptor
      // authenticates every subsequent call.
      await apiService.verifyManagementToken(candidate)
      setManagementToken(candidate)
      toast.success('Management access unlocked')
      onClose()
    } catch (err) {
      // A typed 401 means the server rejected the candidate; other failures
      // carry their own message from transformError.
      setError(isManagementAuthError(err)
        ? 'Invalid management token. Ask your operator for the correct value.'
        : err instanceof Error ? err.message : 'Token verification failed')
    } finally {
      setVerifying(false)
    }
  }

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !verifying) {
      event.preventDefault()
      handleUnlock()
    }
  }

  return (
    <Dialog open={open} onClose={verifying ? undefined : onClose} maxWidth="xs" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <LockIcon sx={{ fontSize: 20, color: 'text.secondary' }} />
        Management access locked
      </DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          This server requires a management token. It is set by the operator via{' '}
          <Typography component="code" sx={{ fontSize: '0.75rem', px: 0.5, bgcolor: 'rgba(255,255,255,0.06)', borderRadius: 0.5, display: 'inline' }}>
            MANAGEMENT_AUTH_TOKEN
          </Typography>{' '}
          on the backend.
        </Typography>
        <TextField
          label="Management token"
          type={showToken ? 'text' : 'password'}
          value={token}
          onChange={(event) => {
            setToken(event.target.value)
            if (error) setError(null)
          }}
          onKeyDown={handleKeyDown}
          disabled={verifying}
          autoFocus
          fullWidth
          size="small"
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  onClick={() => setShowToken((prev) => !prev)}
                  edge="end"
                  size="small"
                  disabled={verifying}
                  aria-label={showToken ? 'Hide token' : 'Show token'}
                >
                  {showToken ? <VisibilityOff sx={{ fontSize: 18 }} /> : <Visibility sx={{ fontSize: 18 }} />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
          The token is remembered in this browser and applied automatically to management API calls.
        </Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={verifying}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleUnlock}
          disabled={verifying || !token.trim()}
          startIcon={verifying ? <CircularProgress size={16} color="inherit" /> : <LockIcon sx={{ fontSize: 16 }} />}
        >
          Unlock
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default ManagementLoginDialog
