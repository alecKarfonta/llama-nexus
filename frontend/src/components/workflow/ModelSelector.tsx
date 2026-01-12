/**
 * ModelSelector - Dropdown for selecting deployed LLM models
 */
import React, { useState, useEffect } from 'react';
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  CircularProgress,
  Box,
  Chip,
  alpha,
  Alert,
} from '@mui/material';
import {
  CheckCircle as ReadyIcon,
  Downloading as DownloadingIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { apiService } from '@/services/api';
import type { ModelInfo } from '@/types/api';

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
  label?: string;
  description?: string;
  disabled?: boolean;
  required?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  value,
  onChange,
  label = 'Model',
  description,
  disabled,
  required,
}) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      setError(null);
      const allModels = await apiService.getModels();
      // Filter to only show deployed/ready models
      const availableModels = allModels.filter(
        (m) => m.status === 'available' || m.status === 'deployed'
      );
      setModels(availableModels);
      
      // Auto-select first model if no value is set and models are available
      if (!value && availableModels.length > 0) {
        onChange(availableModels[0].name);
      }
    } catch (err: any) {
      console.error('Failed to load models:', err);
      setError(err.message || 'Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  const getModelStatus = (model: ModelInfo) => {
    switch (model.status) {
      case 'available':
      case 'deployed':
        return { icon: <ReadyIcon fontSize="small" />, color: '#10b981', label: 'Ready' };
      case 'downloading':
        return { icon: <DownloadingIcon fontSize="small" />, color: '#f59e0b', label: 'Downloading' };
      case 'error':
        return { icon: <ErrorIcon fontSize="small" />, color: '#ef4444', label: 'Error' };
      default:
        return { icon: <WarningIcon fontSize="small" />, color: '#6b7280', label: 'Unknown' };
    }
  };

  const selectedModel = models.find((m) => m.name === value);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 2 }}>
        <CircularProgress size={20} />
        <Typography variant="body2" color="text.secondary">
          Loading models...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  if (models.length === 0) {
    return (
      <Alert severity="warning" sx={{ mb: 2 }}>
        No models are currently deployed. Please deploy a model first.
      </Alert>
    );
  }

  return (
    <Box>
      <FormControl fullWidth size="small" disabled={disabled}>
        <InputLabel required={required}>{label}</InputLabel>
        <Select
          value={value || ''}
          label={label}
          onChange={(e) => onChange(e.target.value)}
          renderValue={(selected) => {
            const model = models.find((m) => m.name === selected);
            if (!model) return selected;
            const status = getModelStatus(model);
            return (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ color: status.color, display: 'flex' }}>
                  {status.icon}
                </Box>
                <Typography variant="body2">{model.name}</Typography>
              </Box>
            );
          }}
        >
          {models.map((model) => {
            const status = getModelStatus(model);
            return (
              <MenuItem key={model.name} value={model.name}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                  <Box sx={{ color: status.color, display: 'flex' }}>
                    {status.icon}
                  </Box>
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography variant="body2" noWrap>
                      {model.name}
                    </Typography>
                    {model.variant && (
                      <Typography variant="caption" color="text.secondary" noWrap>
                        {model.variant} | {(model.size / (1024 ** 3)).toFixed(1)}GB
                      </Typography>
                    )}
                  </Box>
                  <Chip
                    label={status.label}
                    size="small"
                    sx={{
                      bgcolor: alpha(status.color, 0.1),
                      color: status.color,
                      fontSize: '0.65rem',
                      height: 20,
                    }}
                  />
                </Box>
              </MenuItem>
            );
          })}
        </Select>
        {description && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
            {description}
          </Typography>
        )}
      </FormControl>
      
      {selectedModel && (
        <Box
          sx={{
            mt: 1,
            p: 1,
            bgcolor: alpha('#6366f1', 0.05),
            borderRadius: 1,
            border: '1px solid',
            borderColor: alpha('#6366f1', 0.1),
          }}
        >
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            Selected Model Info:
          </Typography>
          <Typography variant="caption" sx={{ display: 'block' }}>
            <strong>Size:</strong> {(selectedModel.size / (1024 ** 3)).toFixed(2)} GB
          </Typography>
          {selectedModel.contextLength && (
            <Typography variant="caption" sx={{ display: 'block' }}>
              <strong>Context:</strong> {selectedModel.contextLength.toLocaleString()} tokens
            </Typography>
          )}
          {selectedModel.parameters && (
            <Typography variant="caption" sx={{ display: 'block' }}>
              <strong>Parameters:</strong> {selectedModel.parameters}
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
};

export default ModelSelector;
