import React, { useMemo, useState } from 'react';
import {
  Alert,
  Box,
  IconButton,
  Switch,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  ContentCopy as CopyIcon,
  RestartAlt as RestartWizardIcon,
  Tune as TuneIcon,
} from '@mui/icons-material';
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from '@/components/common';
import { useClipboard } from '@/hooks/useClipboard';
import { useLocalPreferences } from '@/hooks/useLocalPreferences';
import { resetOnboarding } from '@/components/onboarding/WelcomeWizard';
import {
  localPreferenceOrder,
  localPreferences,
} from '@/utils/preferences';
import { accentColors } from '@/utils/theme';

const CONFIG_STATUS_CONFIG: StatusChipConfigMap = {
  stable: { color: accentColors.success, label: 'Local' },
};

export const ConfigurationPage: React.FC = () => {
  const { preferenceValues, updatePreference } = useLocalPreferences();
  const [copied, setCopied] = useState(false);
  const copyToClipboard = useClipboard();
  const baseUrl = window.location.origin;

  const preferenceCount = useMemo(
    () => Object.values(preferenceValues).filter(Boolean).length,
    [preferenceValues]
  );

  const copyBaseUrl = async () => {
    const ok = await copyToClipboard(baseUrl, {
      successMessage: 'Base URL copied.',
      successTitle: 'URL copied',
    });
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    }
  };

  return (
    <PageShell sx={{ py: 1.25, display: 'flex', flexDirection: 'column', gap: 1.25 }}>
      {copied && (
        <Alert severity="success" onClose={() => setCopied(false)}>
          Base URL copied.
        </Alert>
      )}

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
