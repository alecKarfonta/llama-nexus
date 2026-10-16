import { useEffect, useState } from 'react';
import {
  getManagementAuthState,
  managementAuthChangeEvent,
  managementTokenStorageKey,
  syncManagementAuthFromStorage,
  type ManagementAuthChangeDetail,
  type ManagementAuthState,
} from '@/utils/managementAuth';

// [2026-10-16] | INFO | USE_MANAGEMENT_AUTH.STATE | Reactive management-auth
// state — mirrors the useLocalPreferences pattern (window event + cross-tab
// storage events). 'none' means the server has no token configured, so
// callers render zero extra chrome.
export const useManagementAuthState = (): ManagementAuthState => {
  const [state, setState] = useState<ManagementAuthState>(getManagementAuthState);

  useEffect(() => {
    const handleChange = (event: Event) => {
      setState((event as CustomEvent<ManagementAuthChangeDetail>).detail.state);
    };

    const handleStorage = (event: StorageEvent) => {
      if (event.key === null || event.key === managementTokenStorageKey) {
        syncManagementAuthFromStorage();
      }
    };

    window.addEventListener(managementAuthChangeEvent, handleChange);
    window.addEventListener('storage', handleStorage);

    return () => {
      window.removeEventListener(managementAuthChangeEvent, handleChange);
      window.removeEventListener('storage', handleStorage);
    };
  }, []);

  return state;
};
