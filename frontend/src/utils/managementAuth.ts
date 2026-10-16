/**
 * Management-API token storage + auth-state events.
 *
 * The backend gates /api/v1/* behind MANAGEMENT_AUTH_TOKEN (see
 * backend/auth_middleware.py — active only when the env var is set). This
 * module owns the browser half: the token itself (localStorage) and a window
 * event components subscribe to for login/logout/auth-required transitions
 * (same pattern as utils/preferences.ts localPreferenceChangeEvent).
 */

export const managementTokenStorageKey = 'ln-management-token';

export const managementAuthChangeEvent = 'llama-nexus-management-auth-change';

export type ManagementAuthState =
  | 'none' // No token stored and no 401 seen — auth is off server-side (localhost default)
  | 'authenticated' // A token is stored
  | 'required' // The backend returned 401 — a valid token is needed
  | 'signed-out'; // The user explicitly cleared a stored token

export interface ManagementAuthChangeDetail {
  state: ManagementAuthState;
}

// [2026-10-16] | INFO | MANAGEMENT_AUTH.ERROR | Typed rejection used by the
// api.ts 401 interceptor so callers can identify (and silently ignore)
// management-auth failures instead of toasting them on every poll.
export class ManagementAuthRequiredError extends Error {
  // Silent marker — error pipelines can check this to suppress toast spam.
  readonly isSilent = true;

  constructor(message?: string) {
    super(message || 'Management API authentication required');
    this.name = 'ManagementAuthRequiredError';
  }
}

// [2026-10-16] | INFO | MANAGEMENT_AUTH.IS_AUTH_ERROR | Helper for call-site
// catch blocks: skip toasts/notifications when the failure is just the
// pending login prompt (dialog is already open).
export const isManagementAuthError = (error: unknown): error is ManagementAuthRequiredError => (
  error instanceof ManagementAuthRequiredError
);

const getLocalStorage = () => {
  if (typeof window === 'undefined') return null;
  return window.localStorage;
};

export const getManagementToken = (): string | null => {
  try {
    return getLocalStorage()?.getItem(managementTokenStorageKey) || null;
  } catch {
    // Local storage may be unavailable in restricted browser contexts.
    return null;
  }
};

let currentAuthState: ManagementAuthState = getManagementToken() ? 'authenticated' : 'none';

export const getManagementAuthState = (): ManagementAuthState => currentAuthState;

// [2026-10-16] | DEBUG | MANAGEMENT_AUTH.EMIT | Emit on transitions only so
// repeated 401s from polling callers don't re-fire the event (and re-open
// the login dialog) in a loop.
export const emitManagementAuthChange = (state: ManagementAuthState) => {
  if (state === currentAuthState) return;
  currentAuthState = state;

  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent<ManagementAuthChangeDetail>(managementAuthChangeEvent, {
      detail: { state },
    }));
  }
};

// [2026-10-16] | INFO | MANAGEMENT_AUTH.SET_TOKEN | Persist (remember is
// always-on per spec) and announce the authenticated transition.
export const setManagementToken = (token: string) => {
  try {
    getLocalStorage()?.setItem(managementTokenStorageKey, token);
  } catch {
    // Local storage may be unavailable in restricted browser contexts.
  }
  emitManagementAuthChange('authenticated');
};

// [2026-10-16] | INFO | MANAGEMENT_AUTH.CLEAR_TOKEN | Sign out: forget the
// token and announce it. The next 401 re-enters 'required'.
export const clearManagementToken = () => {
  try {
    getLocalStorage()?.removeItem(managementTokenStorageKey);
  } catch {
    // Clearing is best-effort.
  }
  emitManagementAuthChange('signed-out');
};

// [2026-10-16] | INFO | MANAGEMENT_AUTH.REQUIRED | Called by the api.ts 401
// interceptor: any stored token is provably invalid, so drop it and open the
// login flow via the 'required' transition. 'signed-out' is sticky — the user
// just signed out deliberately, so don't nag with a reopened dialog on every
// subsequent 401; the header's unlock button stays available as the way back.
export const handleManagementAuthRequired = () => {
  try {
    getLocalStorage()?.removeItem(managementTokenStorageKey);
  } catch {
    // Clearing is best-effort.
  }
  if (currentAuthState === 'signed-out') return;
  emitManagementAuthChange('required');
};

// [2026-10-16] | DEBUG | MANAGEMENT_AUTH.SYNC | Cross-tab: another window
// changed the token — recompute state (emits only on transition). A tab that
// already knows auth is 'required' stays required; a tab that knew auth was
// on degrades to 'signed-out' rather than pretending auth is off.
export const syncManagementAuthFromStorage = () => {
  if (getManagementToken()) {
    emitManagementAuthChange('authenticated');
    return;
  }
  if (currentAuthState === 'required' || currentAuthState === 'signed-out') return;
  emitManagementAuthChange(currentAuthState === 'authenticated' ? 'signed-out' : 'none');
};

export const requestManagementLoginEvent = 'llama-nexus-management-login-request';

// [2026-10-16] | INFO | MANAGEMENT_AUTH.REQUEST_LOGIN | Ask the globally
// mounted login dialog to open (header unlock button, sign-out recovery).
export const requestManagementLogin = () => {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(requestManagementLoginEvent));
};
