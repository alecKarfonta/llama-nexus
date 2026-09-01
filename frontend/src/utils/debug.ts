import { preferenceStorageKeys, readStoredBoolean } from '@/utils/preferences'

const DEBUG_STORAGE_KEY = preferenceStorageKeys.debugLogs

const isDebugEnabled = () => {
  if (import.meta.env.DEV) return true

  return readStoredBoolean(DEBUG_STORAGE_KEY, false)
}

export const debugLog = (scope: string, ...args: unknown[]) => {
  if (!isDebugEnabled()) return
  console.debug(`[${scope}]`, ...args)
}

export const debugLogKey = DEBUG_STORAGE_KEY
