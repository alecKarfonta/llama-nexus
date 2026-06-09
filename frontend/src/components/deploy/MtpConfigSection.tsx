import React, { useState } from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Chip,
  FormControlLabel,
  Grid,
  Slider,
  Switch,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import type { ModelInfo } from '@/types/api';
import {
  DEFAULT_MTP_WORKLOAD_PROFILE,
  MTP_WORKLOAD_PROFILE_HINTS,
  MTP_WORKLOAD_PROFILES,
  type MtpWorkloadProfile,
} from '@/utils/mtpWorkloadProfiles';

export interface MtpConfigShape {
  enabled?: boolean;
  workload_profile?: MtpWorkloadProfile;
  draft_n_max?: number;
  draft_n_min?: number;
  draft_p_min?: number;
}

interface MtpConfigSectionProps {
  mtp: MtpConfigShape | undefined;
  modelMtpCapable: boolean;
  backendMtpSupported: boolean | null;
  backendBuildLabel?: string;
  parallelSlots: number;
  selectedModel?: ModelInfo | null;
  onChange: (path: string, value: unknown) => void;
  onWorkloadProfileChange: (profile: MtpWorkloadProfile) => void;
  onMtpEnabledChange: (enabled: boolean) => void;
}

const PROFILE_LABELS: Record<MtpWorkloadProfile, string> = {
  chat: 'Chat',
  agent: 'Agent',
  throughput: 'Throughput',
  custom: 'Custom',
};

export const MtpConfigSection: React.FC<MtpConfigSectionProps> = ({
  mtp,
  modelMtpCapable,
  backendMtpSupported,
  backendBuildLabel,
  parallelSlots,
  selectedModel,
  onChange,
  onWorkloadProfileChange,
  onMtpEnabledChange,
}) => {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const canEnable = modelMtpCapable && backendMtpSupported === true;
  const workloadProfile = mtp?.workload_profile ?? DEFAULT_MTP_WORKLOAD_PROFILE;
  const isCustom = workloadProfile === 'custom';

  let blockReason: string | null = null;
  if (!modelMtpCapable) {
    blockReason =
      'This GGUF has no MTP prediction heads (mtp_capable=false). Use an MTP-converted GGUF — standard quantizations of MTP-trained families will not work.';
  } else if (backendMtpSupported === false) {
    blockReason = `llama.cpp build is too old${backendBuildLabel ? ` (${backendBuildLabel})` : ''}. Rebuild llamacpp-api to b9193 or newer.`;
  } else if (backendMtpSupported !== true) {
    blockReason = 'Checking backend build support…';
  }

  const draftNMax = mtp?.draft_n_max ?? 3;
  const draftNMin = mtp?.draft_n_min ?? 0;
  const draftPMin = mtp?.draft_p_min ?? 0.75;

  const profileHint =
    workloadProfile !== 'custom' ? MTP_WORKLOAD_PROFILE_HINTS[workloadProfile] : null;

  return (
    <Accordion
      disableGutters
      defaultExpanded
      sx={{
        bgcolor: canEnable ? 'rgba(76, 175, 80, 0.04)' : 'rgba(255,255,255,0.02)',
        border: canEnable ? '1px solid rgba(76, 175, 80, 0.25)' : '1px solid rgba(255,255,255,0.08)',
        borderRadius: '4px !important',
        '&:before': { display: 'none' },
        boxShadow: 'none',
      }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Box display="flex" alignItems="center" gap={1} flexWrap="wrap">
          <Typography sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>
            Speculative Decoding (MTP)
          </Typography>
          {selectedModel?.mtpCapable && (
            <Chip label="MTP GGUF" size="small" color="success" variant="outlined" sx={{ height: 20 }} />
          )}
          {!canEnable && (
            <Chip label="Unavailable" size="small" variant="outlined" sx={{ height: 20 }} />
          )}
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: '0.75rem' }}>
          Multi-Token Prediction uses built-in heads in the same GGUF (no separate draft model).
          Optimal draft depth depends on workload — chat favors n2, tool/agent paths favor n8.
        </Typography>

        {blockReason && (
          <Alert severity="warning" sx={{ mb: 2, fontSize: '0.8125rem' }}>
            {blockReason}
          </Alert>
        )}

        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Workload profile
        </Typography>
        <ToggleButtonGroup
          exclusive
          size="small"
          value={workloadProfile}
          disabled={!canEnable}
          onChange={(_, value: MtpWorkloadProfile | null) => {
            if (value) onWorkloadProfileChange(value);
          }}
          sx={{ mb: 2, flexWrap: 'wrap' }}
        >
          {MTP_WORKLOAD_PROFILES.map((profile) => (
            <ToggleButton key={profile} value={profile} sx={{ textTransform: 'none', px: 1.5 }}>
              {PROFILE_LABELS[profile]}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>

        {profileHint && (
          <Alert severity="info" sx={{ mb: 2, fontSize: '0.8125rem' }}>
            <strong>{profileHint.title}:</strong> {profileHint.detail}
          </Alert>
        )}

        {!isCustom && workloadProfile !== 'throughput' && canEnable && !mtp?.enabled && (
          <Alert severity="info" sx={{ mb: 2, fontSize: '0.8125rem' }}>
            The <strong>{PROFILE_LABELS[workloadProfile]}</strong> profile enables MTP automatically.
            Click the profile button again if this toggle looks off after an older save.
          </Alert>
        )}

        <FormControlLabel
          control={
            <Switch
              checked={Boolean(mtp?.enabled) && canEnable}
              disabled={!canEnable}
              onChange={(e) => {
                if (e.target.checked) {
                  if (workloadProfile === 'custom') {
                    onMtpEnabledChange(true);
                  } else if (workloadProfile === 'throughput') {
                    onWorkloadProfileChange('chat');
                  } else {
                    onWorkloadProfileChange(workloadProfile);
                  }
                } else {
                  onMtpEnabledChange(false);
                }
              }}
            />
          }
          label="Enable MTP (draft-mtp)"
          sx={{ mb: 2, display: 'block' }}
        />

        {mtp?.enabled && canEnable && parallelSlots > 1 && (
          <Alert severity="info" sx={{ mb: 2, fontSize: '0.8125rem' }}>
            MTP is enabled with parallel_slots={parallelSlots}. Multi-slot deployments often
            reduce aggregate throughput — use parallel_slots=1 for latency-focused workloads.
          </Alert>
        )}

        <Box sx={{ opacity: canEnable && mtp?.enabled ? 1 : 0.5, pointerEvents: canEnable ? 'auto' : 'none' }}>
          <Typography gutterBottom sx={{ fontSize: '0.875rem' }}>
            Max draft tokens: {draftNMax}
            {!isCustom && (
              <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                (set by {PROFILE_LABELS[workloadProfile]} profile)
              </Typography>
            )}
          </Typography>
          <Slider
            value={draftNMax}
            min={1}
            max={8}
            step={1}
            marks
            disabled={!mtp?.enabled || !canEnable || !isCustom}
            onChange={(_, v) => {
              onChange('mtp.workload_profile', 'custom');
              onChange('mtp.draft_n_max', v as number);
            }}
            valueLabelDisplay="auto"
            sx={{ mb: 2, maxWidth: 400 }}
          />

          <Accordion
            expanded={advancedOpen}
            onChange={(_, exp) => setAdvancedOpen(exp)}
            disableGutters
            sx={{
              bgcolor: 'transparent',
              boxShadow: 'none',
              '&:before': { display: 'none' },
            }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="caption" color="text.secondary">
                Advanced MTP tuning
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Draft min tokens"
                    type="number"
                    fullWidth
                    size="small"
                    value={draftNMin}
                    disabled={!mtp?.enabled || !canEnable || !isCustom}
                    onChange={(e) => {
                      onChange('mtp.workload_profile', 'custom');
                      onChange('mtp.draft_n_min', parseInt(e.target.value, 10) || 0);
                    }}
                    helperText="--spec-draft-n-min (usually 0)"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Draft confidence min"
                    type="number"
                    fullWidth
                    size="small"
                    inputProps={{ min: 0, max: 1, step: 0.05 }}
                    value={draftPMin}
                    disabled={!mtp?.enabled || !canEnable || !isCustom}
                    onChange={(e) => {
                      onChange('mtp.workload_profile', 'custom');
                      onChange('mtp.draft_p_min', parseFloat(e.target.value) || 0);
                    }}
                    helperText="--spec-draft-p-min — higher values improve acceptance but may reduce speed"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Box>
      </AccordionDetails>
    </Accordion>
  );
};

export default MtpConfigSection;
