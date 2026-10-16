import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Collapse,
  IconButton,
  InputAdornment,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  ContentCopy as CopyIcon,
  Key as KeyIcon,
  Lan as LanIcon,
  Refresh as RefreshIcon,
  RestartAlt as RestartWizardIcon,
  Save as SaveIcon,
  Storage as StorageIcon,
  Tune as TuneIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  VpnKey as VpnKeyIcon,
} from '@mui/icons-material';
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from '@/components/common';
import { apiService } from '@/services/api';
import { useClipboard } from '@/hooks/useClipboard';
import { useLocalPreferences } from '@/hooks/useLocalPreferences';
import { resetOnboarding } from '@/components/onboarding/WelcomeWizard';
import {
  localPreferenceOrder,
  localPreferences,
} from '@/utils/preferences';
import { accentColors } from '@/utils/theme';
import type {
  AppSettingsPayload,
  HfTokenUpdateResponse,
  LanUpdateResponse,
} from '@/types/api';

const CONFIG_STATUS_CONFIG: StatusChipConfigMap = {
  stable: { color: accentColors.success, label: 'Local' },
  server: { color: accentColors.info, label: 'Server' },
  exposed: { color: accentColors.warning, label: 'LAN' },
  localOnly: { color: accentColors.success, label: 'Local only' },
};

type SaveState = { severity: 'success' | 'info' | 'warning' | 'error'; message: string } | null;

export const ConfigurationPage: React.FC = () => {
  const { preferenceValues, updatePreference } = useLocalPreferences();
  const [copied, setCopied] = useState(false);
  const copyToClipboard = useClipboard();
  const baseUrl = window.location.origin;

  // [2026-10-16] | INFO | CONFIGURATION_PAGE.SERVER_SETTINGS | Server-backed
  // installation settings (HF token, API keys, storage paths, LAN exposure).
  const [settings, setSettings] = useState<AppSettingsPayload | null>(null);
  const [settingsLoading, setSettingsLoading] = useState(true);
  const [settingsError, setSettingsError] = useState<string | null>(null);

  const [hfTokenDraft, setHfTokenDraft] = useState('');
  const [hfTokenFocused, setHfTokenFocused] = useState(false);
  const [showHfToken, setShowHfToken] = useState(false);
  const [hfSaving, setHfSaving] = useState(false);

  const [lanSaving, setLanSaving] = useState(false);
  const [lanNotice, setLanNotice] = useState<SaveState>(null);

  const [saveState, setSaveState] = useState<SaveState>(null);

  const preferenceCount = useMemo(
    () => Object.values(preferenceValues).filter(Boolean).length,
    [preferenceValues]
  );

  const loadSettings = useCallback(async () => {
    setSettingsLoading(true);
    setSettingsError(null);
    try {
      const payload = await apiService.getAppSettings();
      setSettings(payload);
      setHfTokenDraft(payload.hf_token_masked || '');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load settings';
      setSettingsError(message);
    } finally {
      setSettingsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const saveHfToken = async () => {
    setHfSaving(true);
    setSaveState(null);
    try {
      // Masked echo → backend keeps the stored value; '' clears; hf_… replaces.
      const response: HfTokenUpdateResponse = await apiService.updateHfToken(hfTokenDraft.trim());
      setHfTokenDraft(response.hf_token_masked || '');
      setShowHfToken(false);
      setSaveState({
        severity: response.status === 'unchanged' ? 'info' : 'success',
        message:
          response.status === 'updated'
            ? 'HuggingFace token saved. Gated-model downloads now use it.'
            : response.status === 'cleared'
              ? 'HuggingFace token cleared.'
              : 'HuggingFace token unchanged.',
      });
      await loadSettings();
    } catch (error) {
      setSaveState({
        severity: 'error',
        message: error instanceof Error ? error.message : 'Failed to save token',
      });
    } finally {
      setHfSaving(false);
    }
  };

  const toggleLan = async (exposed: boolean) => {
    setLanSaving(true);
    setLanNotice(null);
    setSaveState(null);
    try {
      const response: LanUpdateResponse = await apiService.updateLanExposure(exposed);
      const parts: string[] = [];
      if (response.instructions) {
        parts.push(response.instructions);
      } else if (!response.restart_required) {
        parts.push(exposed ? 'LAN exposure enabled.' : 'LAN access restricted to localhost.');
      }
      if (response.warning) {
        parts.push(response.warning);
      }
      setLanNotice({
        severity: response.warning ? 'warning' : 'success',
        message: parts.join(' ') || (exposed ? 'LAN exposure enabled.' : 'LAN access restricted to localhost.'),
      });
      if (response.management_token_suggestion) {
        setSuggestedToken(response.management_token_suggestion);
      }
      await loadSettings();
    } catch (error) {
      setLanNotice({
        severity: 'error',
        message: error instanceof Error ? error.message : 'Failed to update LAN exposure',
      });
    } finally {
      setLanSaving(false);
    }
  };

  const [suggestedToken, setSuggestedToken] = useState<string | null>(null);

  const copyText = async (text: string, label: string) => {
    const ok = await copyToClipboard(text, {
      successMessage: `${label} copied.`,
      successTitle: 'Copied',
    });
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    }
  };

  const copyBaseUrl = async () => {
    await copyText(baseUrl, 'Base URL');
  };

  const hfTokenDirty = hfTokenDraft !== (settings?.hf_token_masked || '');

  return (
    <PageShell sx={{ py: 1.25, display: 'flex', flexDirection: 'column', gap: 1.25 }}>
      <Collapse in={Boolean(saveState)}>
        <Alert severity={saveState?.severity} onClose={() => setSaveState(null)}>
          {saveState?.message}
        </Alert>
      </Collapse>
      {copied && (
        <Alert severity="success" onClose={() => setCopied(false)}>
          Copied.
        </Alert>
      )}

      {settingsLoading && !settings ? (
        <SectionCard title="Loading settings…" icon={<TuneIcon />} accentColor={accentColors.info}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 2 }}>
            <CircularProgress size={18} />
            <Typography variant="body2" color="text.secondary">
              Fetching installation settings from the backend…
            </Typography>
          </Box>
        </SectionCard>
      ) : settingsError ? (
        <SectionCard title="Settings unavailable" icon={<TuneIcon />} accentColor={accentColors.rose}>
          <Alert
            severity="error"
            action={
              <Button color="inherit" size="small" onClick={loadSettings} startIcon={<RefreshIcon />}>
                Retry
              </Button>
            }
          >
            {settingsError}
          </Alert>
        </SectionCard>
      ) : settings ? (
        <>
          {/* HuggingFace token */}
          <SectionCard
            title="HuggingFace token"
            icon={<KeyIcon />}
            accentColor={accentColors.info}
            action={
              <StatusChip
                status="server"
                label={settings.hf_token_set ? 'Saved' : 'Not set'}
                statusConfig={CONFIG_STATUS_CONFIG}
              />
            }
          >
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Used for gated-model downloads and the HuggingFace browser. Only needed for gated
              repos — create one at huggingface.co/settings/tokens.
            </Typography>
            <Stack direction="row" spacing={1} alignItems="center">
              <TextField
                size="small"
                label="HF token"
                value={hfTokenFocused ? hfTokenDraft : (settings.hf_token_masked || '')}
                onChange={(event) => setHfTokenDraft(event.target.value)}
                onFocus={() => setHfTokenFocused(true)}
                onBlur={() => setHfTokenFocused(false)}
                type={showHfToken ? 'text' : 'password'}
                sx={{ flex: 1, minWidth: 0 }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <IconButton
                        size="small"
                        onClick={() => setShowHfToken(!showHfToken)}
                        edge="start"
                        aria-label={showHfToken ? 'Hide token' : 'Show token'}
                      >
                        {showHfToken ? <VisibilityOffIcon fontSize="small" /> : <VisibilityIcon fontSize="small" />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                helperText={
                  settings.hf_token_source === 'env'
                    ? 'A token is also present via the HUGGINGFACE_TOKEN env var; saving here overrides it.'
                    : 'Leave the masked value to keep the current token; clear the field to remove it.'
                }
              />
              <Tooltip title="Save token">
                <span>
                  <Button
                    variant="contained"
                    size="small"
                    onClick={saveHfToken}
                    disabled={hfSaving || !hfTokenDirty}
                    startIcon={hfSaving ? <CircularProgress size={14} /> : <SaveIcon />}
                  >
                    Save
                  </Button>
                </span>
              </Tooltip>
            </Stack>
          </SectionCard>

          {/* API keys (read-only masked view + copy of full values is server-side only) */}
          <SectionCard
            title="API keys"
            icon={<VpnKeyIcon />}
            accentColor={accentColors.warning}
            action={<StatusChip status="server" label="From deploy config" statusConfig={CONFIG_STATUS_CONFIG} />}
          >
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Effective keys for the deployed inference and embedding servers. Change them on the
              matching Deploy page (Server section) and redeploy to apply.
            </Typography>
            <Box sx={{ display: 'grid', gap: 0.5 }}>
              {(
                [
                  ['Inference (OpenAI-compatible /v1)', settings.api_keys.inference],
                  ['Embeddings (/v1/embeddings)', settings.api_keys.embedding],
                ] as const
              ).map(([label, key]) => (
                <Box
                  key={label}
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: 'minmax(0, 1fr) auto',
                    gap: 1,
                    alignItems: 'center',
                    minHeight: 44,
                    px: 0.75,
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: 'divider',
                    bgcolor: 'rgba(255, 255, 255, 0.018)',
                  }}
                >
                  <Box sx={{ minWidth: 0 }}>
                    <Typography variant="body2" sx={{ fontWeight: 800, lineHeight: 1.2 }}>
                      {label}
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{ fontFamily: 'monospace', color: 'text.secondary', display: 'block' }}
                    >
                      {key.masked || 'not configured'}
                    </Typography>
                  </Box>
                  {key.masked ? (
                    <Tooltip title="Copy masked hint (full values stay server-side)">
                      <IconButton size="small" onClick={() => copyText(key.masked, 'Masked key')}>
                        <CopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  ) : null}
                </Box>
              ))}
            </Box>
          </SectionCard>

          {/* Storage paths */}
          <SectionCard
            title="Storage"
            icon={<StorageIcon />}
            accentColor={accentColors.success}
            action={
              <StatusChip
                status="server"
                label={`${settings.storage.filter((row) => row.exists).length}/${settings.storage.length} present`}
                statusConfig={CONFIG_STATUS_CONFIG}
              />
            }
          >
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Where models, conversations, and caches live inside the stack. Free space is reported
              per volume.
            </Typography>
            <Box sx={{ display: 'grid', gap: 0.5 }}>
              {settings.storage.map((row) => (
                <Box
                  key={row.label}
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: 'auto minmax(0, 1fr) auto',
                    gap: 1,
                    alignItems: 'center',
                    minHeight: 40,
                    px: 0.75,
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: 'divider',
                    bgcolor: 'rgba(255, 255, 255, 0.018)',
                  }}
                >
                  <StatusChip
                    status={row.exists ? 'server' : 'stable'}
                    label={row.exists ? 'OK' : 'Missing'}
                    statusConfig={{
                      ...CONFIG_STATUS_CONFIG,
                      server: { color: accentColors.success, label: 'OK' },
                    }}
                  />
                  <Box sx={{ minWidth: 0 }}>
                    <Typography variant="body2" sx={{ fontWeight: 800, lineHeight: 1.2 }}>
                      {row.label}
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{ fontFamily: 'monospace', color: 'text.secondary', display: 'block' }}
                      noWrap
                      title={row.path}
                    >
                      {row.path}
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                    {row.free_gb !== undefined && row.total_gb !== undefined
                      ? `${row.free_gb} GB free / ${row.total_gb} GB`
                      : row.error
                        ? 'unknown'
                        : ''}
                  </Typography>
                </Box>
              ))}
            </Box>
          </SectionCard>

          {/* LAN exposure */}
          <SectionCard
            title="Network exposure"
            icon={<LanIcon />}
            accentColor={settings.lan.effective ? accentColors.warning : accentColors.success}
            action={
              <StatusChip
                status={settings.lan.effective ? 'exposed' : 'localOnly'}
                label={settings.lan.effective ? `LAN (${settings.lan.bind_host})` : 'Localhost only'}
                statusConfig={CONFIG_STATUS_CONFIG}
              />
            }
          >
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: 'auto minmax(0, 1fr)',
                gap: 1,
                alignItems: 'center',
                minHeight: 48,
                py: 0.65,
                px: 0.75,
                borderRadius: '8px',
                border: '1px solid',
                borderColor: 'divider',
                bgcolor: 'rgba(255, 255, 255, 0.018)',
              }}
            >
              <Switch
                size="small"
                checked={settings.lan.desired ?? settings.lan.effective}
                disabled={lanSaving}
                onChange={(event) => toggleLan(event.target.checked)}
              />
              <Box sx={{ minWidth: 0 }}>
                <Typography variant="body2" sx={{ fontWeight: 800, lineHeight: 1.2 }}>
                  Expose UI & API on the LAN
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.25 }}>
                  Binds the web UI and management API to {settings.lan.effective ? 'all interfaces (0.0.0.0)' : 'the host (LAN IP)'} instead of
                  localhost only. Requires a stack restart to take effect.
                </Typography>
              </Box>
            </Box>

            {settings.lan.effective && !settings.lan.management_auth_configured ? (
              <Alert severity="warning" sx={{ mt: 1 }}>
                LAN exposure is on but no management token is configured — the management API is
                open to your network. Set <code>MANAGEMENT_AUTH_TOKEN</code> in your <code>.env</code>.
                {suggestedToken ? (
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1 }}>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                      {suggestedToken}
                    </Typography>
                    <IconButton size="small" onClick={() => copyText(suggestedToken, 'Suggested token')}>
                      <CopyIcon fontSize="small" />
                    </IconButton>
                  </Stack>
                ) : null}
              </Alert>
            ) : null}

            {settings.lan.desired !== null && settings.lan.desired !== settings.lan.effective ? (
              <Alert severity="info" sx={{ mt: 1 }}>
                Pending change: the stack binds {settings.lan.bind_host} but LAN exposure is set{' '}
                {settings.lan.desired ? 'on' : 'off'}. Restart the stack on the host with{' '}
                <code>docker compose up -d</code> to apply.
              </Alert>
            ) : null}

            <Collapse in={Boolean(lanNotice)}>
              <Alert severity={lanNotice?.severity || 'info'} sx={{ mt: 1 }} onClose={() => setLanNotice(null)}>
                {lanNotice?.message}
              </Alert>
            </Collapse>
          </SectionCard>
        </>
      ) : null}

      {/* Preferences (local, browser-side) */}
      <SectionCard
        title="Preferences"
        icon={<TuneIcon />}
        accentColor={accentColors.warning}
        action={
          <StatusChip
            status="stable"
            label={`${preferenceCount.toLocaleString()} enabled`}
            statusConfig={CONFIG_STATUS_CONFIG}
          />
        }
      >
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1fr) minmax(280px, 360px)' },
            gap: 1.25,
            alignItems: 'start',
          }}
        >
          <Box sx={{ display: 'grid', gap: 0.5 }}>
            {localPreferenceOrder.map((preferenceId) => {
              const preference = localPreferences[preferenceId];

              return (
                <Box
                  key={preferenceId}
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: 'auto minmax(0, 1fr)',
                    gap: 1,
                    alignItems: 'center',
                    minHeight: 48,
                    py: 0.65,
                    px: 0.75,
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: 'divider',
                    bgcolor: 'rgba(255, 255, 255, 0.018)',
                  }}
                >
                  <Switch
                    size="small"
                    checked={preferenceValues[preferenceId]}
                    onChange={(event) => updatePreference(preferenceId, event.target.checked)}
                  />
                  <Box sx={{ minWidth: 0 }}>
                    <Typography variant="body2" sx={{ fontWeight: 800, lineHeight: 1.2 }}>
                      {preference.label}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.25 }}>
                      {preference.description}
                    </Typography>
                  </Box>
                </Box>
              );
            })}
          </Box>

          <Box
            sx={{
              p: 1,
              borderRadius: '8px',
              border: '1px solid',
              borderColor: 'divider',
              bgcolor: 'rgba(2, 6, 23, 0.28)',
              minWidth: 0,
            }}
          >
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5, fontWeight: 800 }}>
              Base URL
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
              <Typography
                variant="body2"
                sx={{ fontFamily: 'monospace', minWidth: 0, flex: 1 }}
                noWrap
                title={baseUrl}
              >
                {baseUrl}
              </Typography>
              <Tooltip title="Copy base URL">
                <IconButton size="small" onClick={copyBaseUrl} sx={{ borderRadius: '8px' }}>
                  <CopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Box>
      </SectionCard>

      {/* [2026-10-15] | INFO | CONFIGURATION_PAGE.ONBOARDING | Re-run the first-run
          setup wizard (hardware detection + guided model download) on demand. */}
      <SectionCard
        title="Setup & onboarding"
        icon={<RestartWizardIcon />}
        accentColor={accentColors.info}
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 1,
            flexWrap: 'wrap',
          }}
        >
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="body2" sx={{ fontWeight: 800, lineHeight: 1.2 }}>
              First-run setup wizard
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.25 }}>
              Re-run hardware detection and the guided model download at any time.
            </Typography>
          </Box>
          <Tooltip title="Restart the setup wizard">
            <IconButton
              size="small"
              onClick={resetOnboarding}
              sx={{
                borderRadius: '8px',
                border: '1px solid',
                borderColor: 'divider',
                px: 1.25,
                py: 0.5,
                gap: 0.5,
              }}
            >
              <RestartWizardIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </SectionCard>
    </PageShell>
  );
};
