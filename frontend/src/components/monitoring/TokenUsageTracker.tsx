/**
 * Token Usage Tracker Component
 * Displays token usage statistics by model
 */

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import { useTokenUsage } from '@/hooks/useMetrics';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

interface TokenUsageTrackerProps {
  timeRange?: '1h' | '24h' | '7d' | '30d';
}

export const TokenUsageTracker: React.FC<TokenUsageTrackerProps> = ({
  timeRange = '24h',
}) => {
  const { data: tokenUsage, isLoading, error } = useTokenUsage(timeRange);

  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>Token Usage</Typography>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            Loading token usage data...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (!tokenUsage || tokenUsage.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>Token Usage</Typography>
          {error ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <ErrorOutlineIcon color="error" />
              <Typography variant="body2" color="error">
                Failed to load token usage. Ensure backend endpoint `/v1/usage/tokens` is running.
              </Typography>
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No token usage data available for the selected time period.
            </Typography>
          )}
        </CardContent>
      </Card>
    );
  }

  // Calculate total tokens
  const totalPromptTokens = tokenUsage.reduce((sum, item) => sum + item.promptTokens, 0);
  const totalCompletionTokens = tokenUsage.reduce((sum, item) => sum + item.completionTokens, 0);
  const totalTokens = totalPromptTokens + totalCompletionTokens;

  // Sort models by total token usage (descending)
  const sortedUsage = [...tokenUsage].sort((a, b) => 
    (b.promptTokens + b.completionTokens) - (a.promptTokens + a.completionTokens)
  );

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Token Usage</Typography>
          <Chip 
            label={`Last ${timeRange}`} 
            size="small" 
            color="primary" 
            variant="outlined" 
          />
        </Box>

        {/* Summary stats */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Box sx={{ textAlign: 'center', flex: 1 }}>
            <Typography variant="h4" color="primary">
              {totalTokens.toLocaleString()}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total Tokens
            </Typography>
          </Box>
          <Divider orientation="vertical" flexItem />
          <Box sx={{ textAlign: 'center', flex: 1 }}>
            <Typography variant="h4" color="secondary">
              {totalPromptTokens.toLocaleString()}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Prompt Tokens
            </Typography>
          </Box>
          <Divider orientation="vertical" flexItem />
          <Box sx={{ textAlign: 'center', flex: 1 }}>
            <Typography variant="h4" color="success.main">
              {totalCompletionTokens.toLocaleString()}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Completion Tokens
            </Typography>
          </Box>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Token usage by model */}
        <Typography variant="subtitle1" gutterBottom>
          Usage by Model
        </Typography>
        <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 300, overflow: 'auto' }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Model</TableCell>
                <TableCell align="right">Prompt</TableCell>
                <TableCell align="right">Completion</TableCell>
                <TableCell align="right">Total</TableCell>
                <TableCell align="right">% of Total</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedUsage.map((item) => {
                const modelTotal = item.promptTokens + item.completionTokens;
                const percentage = totalTokens > 0 ? (modelTotal / totalTokens) * 100 : 0;
                
                return (
                  <TableRow key={item.modelId}>
                    <TableCell>
                      <Tooltip title={`Model ID: ${item.modelId}`}>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 150 }}>
                          {item.modelName || item.modelId}
                        </Typography>
                      </Tooltip>
                    </TableCell>
                    <TableCell align="right">{item.promptTokens.toLocaleString()}</TableCell>
                    <TableCell align="right">{item.completionTokens.toLocaleString()}</TableCell>
                    <TableCell align="right">{modelTotal.toLocaleString()}</TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Box sx={{ width: '100%', mr: 1 }}>
                          <LinearProgress 
                            variant="determinate" 
                            value={percentage} 
                            sx={{ height: 6, borderRadius: 3 }}
                          />
                        </Box>
                        <Box sx={{ minWidth: 35 }}>
                          <Typography variant="body2" color="text.secondary">
                            {percentage.toFixed(1)}%
                          </Typography>
                        </Box>
                      </Box>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};
