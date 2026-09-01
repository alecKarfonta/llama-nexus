import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  getLocalPreference,
  isLocalPreferenceStorageKey,
  localPreferenceChangeEvent,
  localPreferenceOrder,
  setLocalPreference,
  type LocalPreferenceChangeDetail,
  type LocalPreferenceId,
} from '@/utils/preferences';

const readPreferenceValues = () => (
  localPreferenceOrder.reduce((values, preferenceId) => ({
    ...values,
    [preferenceId]: getLocalPreference(preferenceId),
  }), {} as Record<LocalPreferenceId, boolean>)
);

export const useLocalPreferences = () => {
  const [preferenceValues, setPreferenceValues] = useState<Record<LocalPreferenceId, boolean>>(readPreferenceValues);

  const refreshPreferences = useCallback(() => {
    setPreferenceValues(readPreferenceValues());
  }, []);

  const updatePreference = useCallback((preferenceId: LocalPreferenceId, value: boolean) => {
    setPreferenceValues((current) => ({ ...current, [preferenceId]: value }));
    setLocalPreference(preferenceId, value);
  }, []);

  useEffect(() => {
    const handlePreferenceChange = (event: Event) => {
      const { id, value } = (event as CustomEvent<LocalPreferenceChangeDetail>).detail;
      setPreferenceValues((current) => ({ ...current, [id]: value }));
    };

    const handleStorage = (event: StorageEvent) => {
      if (event.key === null || isLocalPreferenceStorageKey(event.key)) {
        refreshPreferences();
      }
    };

    window.addEventListener(localPreferenceChangeEvent, handlePreferenceChange);
    window.addEventListener('storage', handleStorage);

    return () => {
      window.removeEventListener(localPreferenceChangeEvent, handlePreferenceChange);
      window.removeEventListener('storage', handleStorage);
    };
  }, [refreshPreferences]);

  return useMemo(() => ({
    preferenceValues,
    updatePreference,
  }), [preferenceValues, updatePreference]);
};

export const useLocalPreference = (preferenceId: LocalPreferenceId) => {
  const { preferenceValues, updatePreference } = useLocalPreferences();

  return useMemo(() => ({
    value: preferenceValues[preferenceId],
    setValue: (value: boolean) => updatePreference(preferenceId, value),
  }), [preferenceId, preferenceValues, updatePreference]);
};
