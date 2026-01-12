/**
 * OpenAI API Node Configuration Component
 * Advanced configuration UI for OpenAI-compatible API endpoints
 */
import React, { useState } from 'react';
import {
  Box,
  TextField,
  Slider,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Switch,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  InputAdornment,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';

interface OpenAIAPINodeConfigProps {
  config: Record<string, any>;
  onChange: (config: Record<string, any>) => void;
}

const PRESET_ENDPOINTS = [
  { name: 'OpenAI', url: 'https://api.openai.com/v1/chat/completions' },
  { name: 'Azure OpenAI', url: 'https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions' },
  { name: 'Local (Ollama)', url: 'http://localhost:11434/v1/chat/completions' },
  { name: 'Local (LM Studio)', url: 'http://localhost:1234/v1/chat/completions' },
  { name: 'Local (Text Generation WebUI)', url: 'http://localhost:5000/v1/chat/completions' },
  { name: 'Anthropic Claude', url: 'https://api.anthropic.com/v1/messages' },
  { name: 'Custom', url: '' },
];

const PRESET_MODELS = [
  // OpenAI Models
  { category: 'OpenAI', models: ['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo', 'o1', 'o1-mini'] },
  // Open Source Models
  { category: 'Open Source', models: ['llama-3.1-70b', 'llama-3.1-8b', 'mixtral-8x7b', 'mistral-7b', 'codellama-70b'] },
  // Custom
  { category: 'Custom', models: [] },
];

export const OpenAIAPINodeConfig: React.FC<OpenAIAPINodeConfigProps> = ({ config, onChange }) => {
  const [showApiKey, setShowApiKey] = useState(false);
  const [expandedSection, setExpandedSection] = useState<string | false>('connection');

  const handleChange = (key: string, value: any) => {
    onChange({
      ...config,
      [key]: value,
    });
  };

  const handleEndpointPreset = (preset: string) => {
    const selected = PRESET_ENDPOINTS.find(p => p.name === preset);
    if (selected) {
      handleChange('endpoint', selected.url);
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      {/* Connection Settings */}
      <Accordion 
        expanded={expandedSection === 'connection'} 
        onChange={(_, isExpanded) => setExpandedSection(isExpanded ? 'connection' : false)}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1" fontWeight="bold">
            Connection Settings
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Endpoint Presets */}
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1 }}>
              {PRESET_ENDPOINTS.map(preset => (
                <Chip
                  key={preset.name}
                  label={preset.name}
                  size="small"
                  variant={config.endpoint === preset.url ? "filled" : "outlined"}
                  color={config.endpoint === preset.url ? "primary" : "default"}
                  onClick={() => handleEndpointPreset(preset.name)}
                />
              ))}
            </Box>

            {/* Endpoint URL */}
            <TextField
              fullWidth
              label="API Endpoint URL"
              value={config.endpoint || ''}
              onChange={(e) => handleChange('endpoint', e.target.value)}
              helperText="The complete URL to the chat completions endpoint"
              size="small"
            />

            {/* API Key */}
            <TextField
              fullWidth
              label="API Key"
              type={showApiKey ? 'text' : 'password'}
              value={config.apiKey || ''}
              onChange={(e) => handleChange('apiKey', e.target.value)}
              helperText="Your API key for authentication"
              size="small"
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowApiKey(!showApiKey)}
                      edge="end"
                      size="small"
                    >
                      {showApiKey ? <VisibilityOffIcon /> : <VisibilityIcon />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />

            {/* Organization (Optional) */}
            <TextField
              fullWidth
              label="Organization ID (Optional)"
              value={config.organization || ''}
              onChange={(e) => handleChange('organization', e.target.value)}
              helperText="Optional organization ID for OpenAI"
              size="small"
            />
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Model Settings */}
      <Accordion 
        expanded={expandedSection === 'model'} 
        onChange={(_, isExpanded) => setExpandedSection(isExpanded ? 'model' : false)}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1" fontWeight="bold">
            Model Settings
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Model Selection */}
            <FormControl fullWidth size="small">
              <InputLabel>Model</InputLabel>
              <Select
                value={config.model || 'gpt-4'}
                onChange={(e) => handleChange('model', e.target.value)}
                label="Model"
              >
                {PRESET_MODELS.map(category => (
                  <MenuItem key={category.category} disabled sx={{ fontWeight: 'bold' }}>
                    {category.category}
                  </MenuItem>
                )).concat(
                  PRESET_MODELS.flatMap(category =>
                    category.models.map(model => (
                      <MenuItem key={model} value={model} sx={{ pl: 3 }}>
                        {model}
                      </MenuItem>
                    ))
                  )
                )}
                <Divider />
                <MenuItem value="custom">
                  <em>Custom Model (type below)</em>
                </MenuItem>
              </Select>
            </FormControl>

            {/* Custom Model Name */}
            {config.model === 'custom' && (
              <TextField
                fullWidth
                label="Custom Model Name"
                value={config.customModel || ''}
                onChange={(e) => handleChange('customModel', e.target.value)}
                helperText="Enter your custom model name"
                size="small"
              />
            )}
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Generation Parameters */}
      <Accordion 
        expanded={expandedSection === 'generation'} 
        onChange={(_, isExpanded) => setExpandedSection(isExpanded ? 'generation' : false)}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1" fontWeight="bold">
            Generation Parameters
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Temperature */}
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2">Temperature: {config.temperature || 0.7}</Typography>
                <Tooltip title="Controls randomness. 0 = deterministic, 2 = very random">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Slider
                value={config.temperature || 0.7}
                onChange={(_, value) => handleChange('temperature', value)}
                min={0}
                max={2}
                step={0.1}
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Top P */}
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2">Top P: {config.topP || 1}</Typography>
                <Tooltip title="Nucleus sampling. Consider only tokens with cumulative probability >= top_p">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Slider
                value={config.topP || 1}
                onChange={(_, value) => handleChange('topP', value)}
                min={0}
                max={1}
                step={0.01}
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Top K */}
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2">Top K: {config.topK || 40}</Typography>
                <Tooltip title="Consider only the top K tokens. Not supported by all models">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Slider
                value={config.topK || 40}
                onChange={(_, value) => handleChange('topK', value)}
                min={1}
                max={200}
                step={1}
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Max Tokens */}
            <TextField
              fullWidth
              type="number"
              label="Max Tokens"
              value={config.maxTokens || ''}
              onChange={(e) => handleChange('maxTokens', parseInt(e.target.value))}
              helperText="Maximum number of tokens to generate"
              size="small"
              InputProps={{
                inputProps: { min: 1, max: 128000 }
              }}
            />

            {/* Presence Penalty */}
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2">Presence Penalty: {config.presencePenalty || 0}</Typography>
                <Tooltip title="Penalize tokens that have already appeared">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Slider
                value={config.presencePenalty || 0}
                onChange={(_, value) => handleChange('presencePenalty', value)}
                min={-2}
                max={2}
                step={0.1}
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Frequency Penalty */}
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2">Frequency Penalty: {config.frequencyPenalty || 0}</Typography>
                <Tooltip title="Penalize tokens based on their frequency">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Slider
                value={config.frequencyPenalty || 0}
                onChange={(_, value) => handleChange('frequencyPenalty', value)}
                min={-2}
                max={2}
                step={0.1}
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Repetition Penalty (Alternative) */}
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2">Repetition Penalty: {config.repetitionPenalty || 1}</Typography>
                <Tooltip title="Alternative repetition penalty used by some models">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Slider
                value={config.repetitionPenalty || 1}
                onChange={(_, value) => handleChange('repetitionPenalty', value)}
                min={0.1}
                max={2}
                step={0.1}
                valueLabelDisplay="auto"
              />
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Advanced Settings */}
      <Accordion 
        expanded={expandedSection === 'advanced'} 
        onChange={(_, isExpanded) => setExpandedSection(isExpanded ? 'advanced' : false)}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1" fontWeight="bold">
            Advanced Settings
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Thinking Level (for o1 models) */}
            {config.model?.startsWith('o1') && (
              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="body2">Thinking Level: {config.thinkingLevel || 0}</Typography>
                  <Tooltip title="Reasoning depth for o1 models (0=none, 10=maximum)">
                    <IconButton size="small">
                      <InfoIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
                <Slider
                  value={config.thinkingLevel || 0}
                  onChange={(_, value) => handleChange('thinkingLevel', value)}
                  min={0}
                  max={10}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                />
              </Box>
            )}

            {/* Stream Response */}
            <FormControlLabel
              control={
                <Switch
                  checked={config.stream || false}
                  onChange={(e) => handleChange('stream', e.target.checked)}
                />
              }
              label="Stream Response"
            />

            {/* Number of Completions */}
            <TextField
              fullWidth
              type="number"
              label="Number of Completions (n)"
              value={config.n || 1}
              onChange={(e) => handleChange('n', parseInt(e.target.value))}
              helperText="Number of completions to generate"
              size="small"
              InputProps={{
                inputProps: { min: 1, max: 10 }
              }}
            />

            {/* Seed */}
            <TextField
              fullWidth
              type="number"
              label="Seed (Optional)"
              value={config.seed || ''}
              onChange={(e) => handleChange('seed', e.target.value ? parseInt(e.target.value) : undefined)}
              helperText="Random seed for deterministic generation"
              size="small"
            />

            {/* Response Format */}
            <FormControl fullWidth size="small">
              <InputLabel>Response Format</InputLabel>
              <Select
                value={config.responseFormat?.type || 'text'}
                onChange={(e) => handleChange('responseFormat', { ...config.responseFormat, type: e.target.value })}
                label="Response Format"
              >
                <MenuItem value="text">Text</MenuItem>
                <MenuItem value="json_object">JSON Object</MenuItem>
                <MenuItem value="json_schema">JSON Schema</MenuItem>
              </Select>
            </FormControl>

            {/* Log Probabilities */}
            <FormControlLabel
              control={
                <Switch
                  checked={config.logprobs || false}
                  onChange={(e) => handleChange('logprobs', e.target.checked)}
                />
              }
              label="Return Log Probabilities"
            />

            {config.logprobs && (
              <TextField
                fullWidth
                type="number"
                label="Top Log Probabilities"
                value={config.topLogprobs || 5}
                onChange={(e) => handleChange('topLogprobs', parseInt(e.target.value))}
                helperText="Number of top log probabilities to return"
                size="small"
                InputProps={{
                  inputProps: { min: 0, max: 20 }
                }}
              />
            )}
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Request Configuration */}
      <Accordion 
        expanded={expandedSection === 'request'} 
        onChange={(_, isExpanded) => setExpandedSection(isExpanded ? 'request' : false)}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1" fontWeight="bold">
            Request Configuration
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Timeout */}
            <TextField
              fullWidth
              type="number"
              label="Timeout (seconds)"
              value={config.timeout || 120}
              onChange={(e) => handleChange('timeout', parseInt(e.target.value))}
              helperText="Request timeout in seconds"
              size="small"
              InputProps={{
                inputProps: { min: 1, max: 600 }
              }}
            />

            {/* Retry Attempts */}
            <TextField
              fullWidth
              type="number"
              label="Retry Attempts"
              value={config.retryAttempts || 2}
              onChange={(e) => handleChange('retryAttempts', parseInt(e.target.value))}
              helperText="Number of retry attempts on failure"
              size="small"
              InputProps={{
                inputProps: { min: 0, max: 5 }
              }}
            />

            {/* Retry Delay */}
            <TextField
              fullWidth
              type="number"
              label="Retry Delay (ms)"
              value={config.retryDelay || 1000}
              onChange={(e) => handleChange('retryDelay', parseInt(e.target.value))}
              helperText="Delay between retries in milliseconds"
              size="small"
              InputProps={{
                inputProps: { min: 100, max: 10000 }
              }}
            />

            {/* Custom Headers */}
            <Alert severity="info" sx={{ mt: 1 }}>
              Custom headers can be added programmatically if needed for specific API requirements.
            </Alert>
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Help Section */}
      <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
        <Typography variant="body2" color="text.secondary">
          This node supports any OpenAI-compatible API endpoint. Configure your endpoint URL,
          API key, and model parameters above. The node will handle retries, streaming, and
          error handling automatically.
        </Typography>
      </Box>
    </Box>
  );
};

export default OpenAIAPINodeConfig;
