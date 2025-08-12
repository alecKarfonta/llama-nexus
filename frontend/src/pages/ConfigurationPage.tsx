import React, { useState, useEffect } from 'react';
import {
  Typography,
  Card,
  CardContent,
  Box,
  Grid,
  TextField,
  Slider,
  Button,
  Alert,
  Tabs,
  Tab,
  Chip,
  Paper,
  CircularProgress,
  Snackbar,
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as RefreshIcon,
  RestartAlt as RestartIcon,
  Code as CodeIcon,
} from '@mui/icons-material';

interface Config {
  model: { name: string; variant: string; context_size: number; gpu_layers: number; };
  sampling: { temperature: number; top_p: number; top_k: number; repeat_penalty: number; frequency_penalty: number; presence_penalty: number; };
  performance: { threads: number; batch_size: number; num_predict: number; };
}

export const ConfigurationPage: React.FC = () => {
  const backendUrl = (import.meta as any).env?.VITE_BACKEND_URL || 'http://localhost:8700';
  const [config, setConfig] = useState<Config | null>(null);
  const [originalConfig, setOriginalConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [commandLine, setCommandLine] = useState('');
  const [presets, setPresets] = useState<any[]>([]);

  useEffect(() => {
    fetchConfig();
    fetchPresets();
  }, []);

  const fetchConfig = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${backendUrl}/api/v1/service/config`);
      if (!response.ok) throw new Error('Failed to fetch configuration');
      const data = await response.json();
      setConfig(data.config);
      setOriginalConfig(JSON.parse(JSON.stringify(data.config)));
      setCommandLine(data.command || '');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch configuration');
    } finally {
      setLoading(false);
    }
  };

  const fetchPresets = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/v1/config/presets`);
      if (response.ok) {
        const data = await response.json();
        setPresets(data);
      }
    } catch (err) {
      console.error('Failed to fetch presets:', err);
    }
  };

  const handleSave = async () => {
    if (!config) return;
    try {
      setSaving(true);
      const response = await fetch(`${backendUrl}/api/v1/service/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (!response.ok) throw new Error('Failed to save configuration');
      const data = await response.json();
      setOriginalConfig(JSON.parse(JSON.stringify(config)));
      setCommandLine(data.command || '');
      setSuccess('Configuration saved successfully');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const handleRestart = async () => {
    try {
      await fetch(`${backendUrl}/api/v1/service/restart`, { method: 'POST' });
      setSuccess('Service restarted with new configuration');
    } catch (err) {
      setError('Failed to restart service');
    }
  };

  const updateConfig = (category: keyof Config, field: string, value: any) => {
    if (!config) return;
    setConfig({
      ...config,
      [category]: { ...config[category], [field]: value },
    });
  };

  const hasChanges = () => JSON.stringify(config) !== JSON.stringify(originalConfig);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  if (!config) {
    return <Alert severity="error">Failed to load configuration.</Alert>;
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Configuration</Typography>
        <Box display="flex" gap={1}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => setConfig(originalConfig ? JSON.parse(JSON.stringify(originalConfig)) : null)}
            disabled={!hasChanges()}
          >
            Reset
          </Button>
          <Button
            variant="contained"
            startIcon={saving ? <CircularProgress size={20} /> : <SaveIcon />}
            onClick={handleSave}
            disabled={saving || !hasChanges()}
          >
            Save Changes
          </Button>
          <Button
            variant="contained"
            color="warning"
            startIcon={<RestartIcon />}
            onClick={handleRestart}
          >
            Restart Service
          </Button>
        </Box>
      </Box>

      {error && <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>{error}</Alert>}
      <Snackbar open={!!success} autoHideDuration={6000} onClose={() => setSuccess(null)} message={success} />

      {/* Presets */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Configuration Presets</Typography>
          <Box display="flex" gap={1} flexWrap="wrap">
            {presets.map((preset) => (
              <Chip key={preset.id} label={preset.name} variant="outlined" clickable />
            ))}
          </Box>
        </CardContent>
      </Card>

      <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)} sx={{ mb: 3 }}>
        <Tab label="Model" />
        <Tab label="Sampling" />
        <Tab label="Performance" />
        <Tab label="Command Line" icon={<CodeIcon />} />
      </Tabs>

      {/* Model Settings */}
      {tabValue === 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Model Configuration</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Context Size"
                  type="number"
                  value={config.model.context_size}
                  onChange={(e) => updateConfig('model', 'context_size', parseInt(e.target.value))}
                  fullWidth
                  margin="normal"
                  inputProps={{ min: 512, max: 131072 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="GPU Layers"
                  type="number"
                  value={config.model.gpu_layers}
                  onChange={(e) => updateConfig('model', 'gpu_layers', parseInt(e.target.value))}
                  fullWidth
                  margin="normal"
                  inputProps={{ min: -1, max: 999 }}
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Sampling Settings */}
      {tabValue === 1 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Sampling Parameters</Typography>
            <Box mb={3}>
              <Typography gutterBottom>Temperature: {config.sampling.temperature}</Typography>
              <Slider
                value={config.sampling.temperature}
                onChange={(_, v) => updateConfig('sampling', 'temperature', v)}
                min={0} max={2} step={0.1}
                valueLabelDisplay="auto"
              />
            </Box>
            <Box mb={3}>
              <Typography gutterBottom>Top P: {config.sampling.top_p}</Typography>
              <Slider
                value={config.sampling.top_p}
                onChange={(_, v) => updateConfig('sampling', 'top_p', v)}
                min={0} max={1} step={0.05}
                valueLabelDisplay="auto"
              />
            </Box>
            <TextField
              label="Top K"
              type="number"
              value={config.sampling.top_k}
              onChange={(e) => updateConfig('sampling', 'top_k', parseInt(e.target.value))}
              fullWidth
              margin="normal"
            />
          </CardContent>
        </Card>
      )}

      {/* Performance Settings */}
      {tabValue === 2 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Performance Settings</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Threads"
                  type="number"
                  value={config.performance.threads}
                  onChange={(e) => updateConfig('performance', 'threads', parseInt(e.target.value))}
                  fullWidth
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Batch Size"
                  type="number"
                  value={config.performance.batch_size}
                  onChange={(e) => updateConfig('performance', 'batch_size', parseInt(e.target.value))}
                  fullWidth
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Max Tokens"
                  type="number"
                  value={config.performance.num_predict}
                  onChange={(e) => updateConfig('performance', 'num_predict', parseInt(e.target.value))}
                  fullWidth
                  margin="normal"
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Command Line */}
      {tabValue === 3 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Generated Command</Typography>
            <Paper
              sx={{
                p: 2,
                backgroundColor: '#1e1e1e',
                color: '#d4d4d4',
                fontFamily: 'monospace',
                fontSize: '0.9rem',
                overflowX: 'auto',
              }}
            >
              {commandLine}
            </Paper>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};
