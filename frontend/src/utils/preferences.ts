export const preferenceStorageKeys = {
  compactControlSurfaces: 'llama-nexus-compact-settings',
  confirmDestructiveActions: 'llama-nexus-confirm-dangerous',
  debugLogs: 'llama-nexus-debug-logs',
  sidebarCollapsedSections: 'sidebar-collapsed-sections',
} as const;

export const localPreferences = {
  compactControlSurfaces: {
    label: 'Prefer compact control surfaces',
    description: 'Use denser layouts for configuration-heavy workflows.',
    defaultValue: false,
    storageKey: preferenceStorageKeys.compactControlSurfaces,
  },
  confirmDestructiveActions: {
    label: 'Confirm destructive actions',
    description: 'Keep an extra confirmation step before deleting or stopping work.',
    defaultValue: true,
    storageKey: preferenceStorageKeys.confirmDestructiveActions,
  },
  debugLogs: {
    label: 'Debug logs',
    description: 'Emit scoped diagnostics to the browser console outside dev mode.',
    defaultValue: false,
    storageKey: preferenceStorageKeys.debugLogs,
  },
} as const;

export type LocalPreferenceId = keyof typeof localPreferences;

export const localPreferenceOrder: LocalPreferenceId[] = [
  'compactControlSurfaces',
  'confirmDestructiveActions',
  'debugLogs',
];

export const localPreferenceChangeEvent = 'llama-nexus-local-preference-change';

export interface LocalPreferenceChangeDetail {
  id: LocalPreferenceId;
  storageKey: string;
  value: boolean;
}

const getLocalStorage = () => {
  if (typeof window === 'undefined') return null;
  return window.localStorage;
};

export const readStoredBoolean = (storageKey: string, fallback: boolean) => {
  try {
    const storage = getLocalStorage();
    const stored = storage?.getItem(storageKey);
    return stored === null || stored === undefined ? fallback : stored === 'true';
  } catch {
    return fallback;
  }
};

export const writeStoredBoolean = (storageKey: string, value: boolean) => {
  try {
    getLocalStorage()?.setItem(storageKey, String(value));
  } catch {
    // Local storage may be unavailable in restricted browser contexts.
  }
};

export const getLocalPreference = (id: LocalPreferenceId) => {
  const preference = localPreferences[id];
  return readStoredBoolean(preference.storageKey, preference.defaultValue);
};

export const setLocalPreference = (id: LocalPreferenceId, value: boolean) => {
  const storageKey = localPreferences[id].storageKey;
  writeStoredBoolean(storageKey, value);

  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent<LocalPreferenceChangeDetail>(localPreferenceChangeEvent, {
      detail: {
        id,
        storageKey,
        value,
      },
    }));
  }
};

export const isLocalPreferenceStorageKey = (storageKey: string | null) => (
  Boolean(storageKey && localPreferenceOrder.some((id) => localPreferences[id].storageKey === storageKey))
);

export const readStoredJson = <T>(storageKey: string, fallback: T): T => {
  try {
    const stored = getLocalStorage()?.getItem(storageKey);
    return stored ? (JSON.parse(stored) as T) : fallback;
  } catch {
    return fallback;
  }
};

export const writeStoredJson = (storageKey: string, value: unknown) => {
  try {
    getLocalStorage()?.setItem(storageKey, JSON.stringify(value));
  } catch {
    // Local storage may be unavailable in restricted browser contexts.
  }
};
