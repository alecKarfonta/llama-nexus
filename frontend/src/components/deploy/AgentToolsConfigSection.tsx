import React, { useEffect, useMemo, useState } from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Chip,
  Grid,
  TextField,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import type { MtpWorkloadProfile } from '@/utils/mtpWorkloadProfiles';

export interface AgentToolsServerShape {
  chat_template_kwargs?: Record<string, unknown> | null;
  reasoning_budget?: number;
  cache_reuse?: number;
  jinja?: boolean;
}

interface AgentToolsConfigSectionProps {
  server: AgentToolsServerShape | undefined;
  workloadProfile?: MtpWorkloadProfile;
  grammarGbnfSupported?: boolean;
  llguidanceEnabled?: boolean | null;
  onChange: (path: string, value: unknown) => void;
  onSwitchToCustomProfile?: () => void;
}

function formatKwargs(value: Record<string, unknown> | null | undefined): string {
  if (value == null) return '';
  return JSON.stringify(value, null, 2);
}

function parseKwargs(
  text: string
): { ok: true; value: Record<string, unknown> | null } | { ok: false; error: string } {
  const trimmed = text.trim();
  if (!trimmed) return { ok: true, value: null };
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return { ok: false, error: 'Must be a JSON object' };
    }
    return { ok: true, value: parsed as Record<string, unknown> };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : 'Invalid JSON' };
  }
}

export const AgentToolsConfigSection: React.FC<AgentToolsConfigSectionProps> = ({
  server,
  workloadProfile,
  grammarGbnfSupported = true,
  llguidanceEnabled = null,
  onChange,
  onSwitchToCustomProfile,
}) => {
  const [kwargsText, setKwargsText] = useState(() => formatKwargs(server?.chat_template_kwargs));
  const [kwargsError, setKwargsError] = useState<string | null>(null);

  useEffect(() => {
    setKwargsText(formatKwargs(server?.chat_template_kwargs));
    setKwargsError(null);
  }, [server?.chat_template_kwargs]);

  const profileHint = useMemo(() => {
    if (workloadProfile === 'agent') {
      return 'Agent workload profile is active — these values match the experiment-backed tool path.';
    }
    if (workloadProfile === 'chat') {
      return 'Chat profile uses preserve_thinking. Switch to Agent workload for tool-call latency.';
    }
    return null;
  }, [workloadProfile]);

  const markCustom = () => {
    onSwitchToCustomProfile?.();
  };

  const handleKwargsBlur = () => {
    const result = parseKwargs(kwargsText);
    if (!result.ok) {
      setKwargsError(result.error);
      return;
    }
    setKwargsError(null);
    markCustom();
    onChange('server.chat_template_kwargs', result.value);
  };

  const syncKwargsFromServer = () => {
    setKwargsText(formatKwargs(server?.chat_template_kwargs));
    setKwargsError(null);
  };

  return (
    <Accordion
      disableGutters
      defaultExpanded
      sx={{
        bgcolor: 'rgba(33, 150, 243, 0.04)',
        border: '1px solid rgba(33, 150, 243, 0.25)',
        borderRadius: '4px !important',
        '&:before': { display: 'none' },
        boxShadow: 'none',
      }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography sx={{ fontSize: '0.9375rem', fontWeight: 600 }}>Agent &amp; Tools</Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
          <Chip
            label="GBNF grammar"
            size="small"
            color={grammarGbnfSupported ? 'success' : 'default'}
            variant="outlined"
          />
          <Chip
            label={
              llguidanceEnabled === true
                ? 'llguidance (jump-forward)'
                : llguidanceEnabled === false
                  ? 'llguidance not built'
                  : 'llguidance: checking…'
            }
            size="small"
            color={llguidanceEnabled === true ? 'success' : 'default'}
            variant="outlined"
          />
        </Box>
        {llguidanceEnabled === false && (
          <Alert severity="info" sx={{ mb: 2, fontSize: '0.8125rem' }}>
            Rebuild llamacpp-api with{' '}
            <code>ENABLE_LLGUIDANCE=true</code> for jump-forward grammar masks (optional tier).
          </Alert>
        )}

        <Alert severity="info" sx={{ mb: 2, fontSize: '0.8125rem' }}>
          When Chat clients send <code>tools</code> in the request, llama-server applies lazy tool
          grammar (GBNF) automatically — no manual grammar config needed for the tool path.
        </Alert>

        {profileHint && (
          <Alert
            severity={workloadProfile === 'agent' ? 'success' : 'warning'}
            sx={{ mb: 2, fontSize: '0.8125rem' }}
          >
            {profileHint}
          </Alert>
        )}

        <Grid container spacing={2}>
          <Grid item xs={12}>
            <TextField
              label="Chat template kwargs (JSON)"
              fullWidth
              multiline
              minRows={3}
              size="small"
              value={kwargsText}
              onChange={(e) => setKwargsText(e.target.value)}
              onBlur={handleKwargsBlur}
              error={Boolean(kwargsError)}
              helperText={
                kwargsError ??
                '--chat-template-kwargs — Agent: {"enable_thinking": false}; Chat: {"preserve_thinking": true}'
              }
              InputProps={{
                sx: { fontFamily: 'monospace', fontSize: '0.8125rem' },
              }}
            />
            <Box sx={{ mt: 0.5, display: 'flex', gap: 1 }}>
              <Button size="small" onClick={syncKwargsFromServer}>
                Reset editor
              </Button>
              <Button
                size="small"
                onClick={() => {
                  setKwargsText('{"enable_thinking": false}');
                  setKwargsError(null);
                  markCustom();
                  onChange('server.chat_template_kwargs', { enable_thinking: false });
                }}
              >
                Disable thinking (agent)
              </Button>
            </Box>
          </Grid>

          <Grid item xs={12} md={6}>
            <TextField
              label="Reasoning budget"
              type="number"
              fullWidth
              size="small"
              value={server?.reasoning_budget ?? ''}
              onChange={(e) => {
                markCustom();
                const raw = e.target.value;
                onChange(
                  'server.reasoning_budget',
                  raw === '' ? undefined : parseInt(raw, 10)
                );
              }}
              helperText="--reasoning-budget — 0 disables thinking tokens; -1 unlimited"
            />
            <Button
              size="small"
              sx={{ mt: 0.5 }}
              onClick={() => {
                markCustom();
                onChange('server.reasoning_budget', 0);
              }}
            >
              Disable for tool latency
            </Button>
          </Grid>

          <Grid item xs={12} md={6}>
            <TextField
              label="Cache reuse (tokens)"
              type="number"
              fullWidth
              size="small"
              value={server?.cache_reuse ?? ''}
              onChange={(e) => {
                markCustom();
                const raw = e.target.value;
                onChange('server.cache_reuse', raw === '' ? undefined : parseInt(raw, 10));
              }}
              helperText="--cache-reuse — prefix cache for system-prompt-heavy agents (Agent profile: 1024)"
            />
          </Grid>
        </Grid>
      </AccordionDetails>
    </Accordion>
  );
};

export default AgentToolsConfigSection;
