import React from 'react'
import {
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
} from '@mui/material'
import type { DialogProps } from '@mui/material'

export interface ConfirmationAction {
  title: string
  message: React.ReactNode
  confirmLabel?: string
  cancelLabel?: string
  destructive?: boolean
  onConfirm: () => void | Promise<void>
}

interface ConfirmationDialogProps {
  action: ConfirmationAction | null
  loading?: boolean
  maxWidth?: DialogProps['maxWidth']
  onClose: () => void
  onConfirm: () => void | Promise<void>
}

export const ConfirmationDialog: React.FC<ConfirmationDialogProps> = ({
  action,
  loading = false,
  maxWidth = 'xs',
  onClose,
  onConfirm,
}) => {
  const message = typeof action?.message === 'string'
    ? (
      <Typography variant="body2" color="text.secondary">
        {action.message}
      </Typography>
    )
    : action?.message

  return (
    <Dialog open={Boolean(action)} onClose={loading ? undefined : onClose} maxWidth={maxWidth} fullWidth>
      <DialogTitle>{action?.title}</DialogTitle>
      <DialogContent>{message}</DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          {action?.cancelLabel || 'Cancel'}
        </Button>
        <Button
          variant="contained"
          color={action?.destructive ? 'error' : 'primary'}
          onClick={onConfirm}
          disabled={loading}
          startIcon={loading ? <CircularProgress size={16} color="inherit" /> : undefined}
        >
          {action?.confirmLabel || 'Confirm'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
