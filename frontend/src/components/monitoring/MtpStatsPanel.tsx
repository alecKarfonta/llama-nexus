import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Typography,
} from '@mui/material';
import type { MtpStatsSnapshot } from '@/hooks/useMtpStats';

interface MtpStatsPanelProps {
  stats: MtpStatsSnapshot | null;
  connected?: boolean;
  compact?: boolean;
}

function formatPercent(rate: number | undefined): string {
  if (rate === undefined || Number.isNaN(rate)) return '—';
  return `${(rate * 100).toFixed(1)}%`;
}

export const MtpStatsPanel: React.FC<MtpStatsPanelProps> = ({
  stats,
  connected = false,
  compact = false,
}) => {
  const rate = stats?.acceptance_rate;
  const targetOk = rate !== undefined && rate >= 0.7;
  const targetWarn = rate !== undefined && rate < 0.7 && rate >= 0.6;
  const rateColor = targetOk ? 'success.main' : targetWarn ? 'warning.main' : 'error.main';

  const body = (
    <>
      <Box display="flex" alignItems="center" gap={1} mb={1.5} flexWrap="wrap">
        <Typography variant={compact ? 'subtitle2' : 'h6'} sx={{ fontWeight: 600 }}>
          MTP live stats
        </Typography>
        <Chip
          label={connected ? 'Log stream' : 'Disconnected'}
          size="small"
          color={connected ? 'success' : 'default'}
          variant="outlined"
        />
        {rate !== undefined && (
          <Chip
            label={targetOk ? 'On target (≥70%)' : 'Tune draft settings'}
            size="small"
            color={targetOk ? 'success' : targetWarn ? 'warning' : 'error'}
            variant="outlined"
          />
        )}
      </Box>

      <GridLike compact={compact} rate={rate} rateColor={rateColor} stats={stats} />

      {rate !== undefined && (
        <Box mt={1.5}>
          <LinearProgress
            variant="determinate"
            value={Math.min(100, Math.max(0, rate * 100))}
            color={targetOk ? 'success' : targetWarn ? 'warning' : 'error'}
            sx={{ height: 6, borderRadius: 1 }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            Target ≥ 70% acceptance for steady speedup. Lower <code>draft_n_max</code> or{' '}
            <code>draft_p_min</code> if rate drops.
          </Typography>
        </Box>
      )}

      {!stats && (
        <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.8125rem' }}>
          Start the server with MTP enabled and generate tokens — acceptance stats appear in
          llama-server logs when drafting is active.
        </Typography>
      )}
    </>
  );

  if (compact) {
    return <Box sx={{ mb: 2 }}>{body}</Box>;
  }

  return (
    <Card sx={{ mb: 2, border: '1px solid rgba(255,255,255,0.08)' }}>
      <CardContent>{body}</CardContent>
    </Card>
  );
};

const GridLike: React.FC<{
  compact: boolean;
  rate: number | undefined;
  rateColor: string;
  stats: MtpStatsSnapshot | null;
}> = ({ compact, rate, rateColor, stats }) => (
  <Box
    display="grid"
    gridTemplateColumns={compact ? 'repeat(2, 1fr)' : 'repeat(3, 1fr)'}
    gap={2}
  >
    <StatBlock label="Acceptance rate" value={formatPercent(rate)} valueColor={rateColor} />
    <StatBlock
      label="Tokens accepted"
      value={stats?.tokens_accepted?.toLocaleString() ?? '—'}
    />
    <StatBlock
      label="Tokens drafted"
      value={stats?.tokens_drafted?.toLocaleString() ?? '—'}
    />
  </Box>
);

const StatBlock: React.FC<{
  label: string;
  value: string;
  valueColor?: string;
}> = ({ label, value, valueColor = 'text.primary' }) => (
  <Box
    sx={{
      p: 1.5,
      borderRadius: 1,
      bgcolor: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.06)',
    }}
  >
    <Typography variant="caption" color="text.secondary" display="block">
      {label}
    </Typography>
    <Typography variant="h6" sx={{ fontWeight: 700, color: valueColor, fontSize: '1.1rem' }}>
      {value}
    </Typography>
  </Box>
);

export default MtpStatsPanel;
