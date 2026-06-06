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
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import type { ModelInfo } from '@/types/api';

export interface MtpConfigShape {
  enabled?: boolean;
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
}

export const MtpConfigSection: React.FC<MtpConfigSectionProps> = ({
  mtp,
  modelMtpCapable,
  backendMtpSupported,
  backendBuildLabel,
  parallelSlots,
  selectedModel,
  onChange,
}) => {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const canEnable = modelMtpCapable && backendMtpSupported === true;

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
          Typical speedup is ~1.7–2× for single-stream workloads when acceptance stays ≥ 70%.
        </Typography>

        {blockReason && (
          <Alert severity="warning" sx={{ mb: 2, fontSize: '0.8125rem' }}>
            {blockReason}
          </Alert>
        )}

        <FormControlLabel
          control={
            <Switch
              checked={Boolean(mtp?.enabled) && canEnable}
              disabled={!canEnable}
              onChange={(e) => onChange('mtp.enabled', e.target.checked)}
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
          </Typography>
          <Slider
            value={draftNMax}
            min={1}
            max={6}
            step={1}
            marks
            disabled={!mtp?.enabled || !canEnable}
            onChange={(_, v) => onChange('mtp.draft_n_max', v as number)}
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
                    disabled={!mtp?.enabled || !canEnable}
                    onChange={(e) => onChange('mtp.draft_n_min', parseInt(e.target.value, 10) || 0)}
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
                    disabled={!mtp?.enabled || !canEnable}
                    onChange={(e) => onChange('mtp.draft_p_min', parseFloat(e.target.value) || 0)}
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
