import { useCallback } from 'react'
import { useToast } from '@/contexts/ToastContext'

interface CopyOptions {
  successMessage?: string
  successTitle?: string
  errorMessage?: string
  errorTitle?: string
  silent?: boolean
}

export const useClipboard = () => {
  const toast = useToast()

  return useCallback(async (text: string, options: CopyOptions = {}) => {
    try {
      await navigator.clipboard.writeText(text)
      if (!options.silent) {
        toast.success(options.successMessage || 'Copied to clipboard.', options.successTitle || 'Copied')
      }
      return true
    } catch (error) {
      toast.error(options.errorMessage || 'Clipboard access was blocked by the browser.', options.errorTitle || 'Copy failed')
      return false
    }
  }, [toast])
}
