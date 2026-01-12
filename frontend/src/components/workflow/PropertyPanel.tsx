/**
 * PropertyPanel - Node configuration panel
 */
import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Button,
  IconButton,
  Divider,
  Paper,
  alpha,
  Slider,
  Chip,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  Close as CloseIcon,
  Delete as DeleteIcon,
  ContentCopy as DuplicateIcon,
  PlayArrow as TestIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import {
  WorkflowNode,
  WorkflowNodeData,
  getNodeTypeDefinition,
  NodeTypeDefinition,
  PortDefinition,
} from '@/types/workflow';
import { ModelSelector } from './ModelSelector';
import { OpenAIAPINodeConfig } from './OpenAIAPINodeConfig';
import { apiService } from '@/services/api';

interface PropertyPanelProps {
  node: WorkflowNode | null;
  onNodeUpdate: (node: WorkflowNode) => void;
  onNodeDelete: (nodeId: string) => void;
  onNodeDuplicate: (nodeId: string) => void;
  onClose: () => void;
}

export const PropertyPanel: React.FC<PropertyPanelProps> = ({
  node,
  onNodeUpdate,
  onNodeDelete,
  onNodeDuplicate,
  onClose,
}) => {
  const [localData, setLocalData] = useState<WorkflowNodeData | null>(null);
  const [nodeTypeDef, setNodeTypeDef] = useState<NodeTypeDefinition | null>(null);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string; data?: any } | null>(null);

  // Update local state when node changes
  useEffect(() => {
    if (node) {
      setLocalData({ ...node.data });
      setNodeTypeDef(getNodeTypeDefinition(node.type) || null);
      setTestResult(null); // Clear test result when node changes
    } else {
      setLocalData(null);
      setNodeTypeDef(null);
      setTestResult(null);
    }
  }, [node]);

  // Handle config changes
  const handleConfigChange = (key: string, value: any) => {
    if (!node || !localData) return;

    const updatedData = {
      ...localData,
      config: {
        ...localData.config,
        [key]: value,
      },
    };
    setLocalData(updatedData);
    onNodeUpdate({
      ...node,
      data: updatedData,
    });
  };

  // Handle label change
  const handleLabelChange = (value: string) => {
    if (!node || !localData) return;

    const updatedData = {
      ...localData,
      label: value,
    };
    setLocalData(updatedData);
    onNodeUpdate({
      ...node,
      data: updatedData,
    });
  };

  // Handle node test
  const handleTestNode = async () => {
    if (!node || !localData || !nodeTypeDef) return;

    setTesting(true);
    setTestResult(null);

    try {
      // For LLM nodes, test the connection
      if (node.type === 'llm_chat' || node.type === 'openai_chat') {
        const model = localData.config?.model;
        
        if (!model) {
          setTestResult({
            success: false,
            message: 'No model selected. Please select a model first.',
          });
          return;
        }

        // Test with a simple message
        const testMessages = [{ role: 'user', content: 'Hello! Please respond with "OK" if you can read this.' }];
        
        // Call the backend API to test the LLM
        const response = await apiService.post('/v1/chat/completions', {
          model: model,
          messages: testMessages,
          max_tokens: 50,
          temperature: 0.7,
        });

        if (response.data?.choices?.[0]?.message?.content) {
          const content = response.data.choices[0].message.content;
          const usage = response.data.usage;
          
          setTestResult({
            success: true,
            message: `Model connected successfully! Response: "${content.substring(0, 100)}${content.length > 100 ? '...' : ''}"`,
            data: usage,
          });
        } else {
          setTestResult({
            success: false,
            message: 'Model responded but format was unexpected.',
          });
        }
      } else if (node.type === 'openai_api_llm') {
        // Test OpenAI API LLM node
        const endpoint = localData.config?.endpoint;
        const apiKey = localData.config?.apiKey;
        const model = localData.config?.model;
        
        if (!endpoint || !apiKey || !model) {
          setTestResult({
            success: false,
            message: 'Please configure endpoint, API key, and model first.',
          });
          return;
        }

        // Test with a simple message
        const testMessages = [{ role: 'user', content: 'Hello! Please respond with "OK" if you can read this.' }];
        
        // Test directly via the configured endpoint
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`,
            ...(localData.config?.organization && { 'OpenAI-Organization': localData.config.organization }),
          },
          body: JSON.stringify({
            model: model === 'custom' ? (localData.config?.customModel || model) : model,
            messages: testMessages,
            max_tokens: 50,
            temperature: 0.7,
          }),
        });

        if (response.ok) {
          const data = await response.json();
          const content = data.choices?.[0]?.message?.content;
          const usage = data.usage;
          
          if (content) {
            setTestResult({
              success: true,
              message: `API connected successfully! Response: "${content.substring(0, 100)}${content.length > 100 ? '...' : ''}"`,
              data: usage,
            });
          } else {
            setTestResult({
              success: false,
              message: 'API responded but format was unexpected.',
            });
          }
        } else {
          const errorText = await response.text();
          let errorMsg = `API error (${response.status})`;
          try {
            const errorData = JSON.parse(errorText);
            errorMsg += `: ${errorData.error?.message || errorText.substring(0, 100)}`;
          } catch {
            errorMsg += `: ${errorText.substring(0, 100)}`;
          }
          
          setTestResult({
            success: false,
            message: errorMsg,
          });
        }
      } else {
        setTestResult({
          success: false,
          message: 'Testing is only available for LLM nodes currently.',
        });
      }
    } catch (err: any) {
      console.error('Node test failed:', err);
      setTestResult({
        success: false,
        message: err.message || 'Failed to connect to model. Please check your configuration.',
      });
    } finally {
      setTesting(false);
    }
  };

  // Check if node can be tested
  const canTestNode = () => {
    if (!node || !localData) return false;
    
    // Only LLM nodes with a model configured can be tested
    if (node.type === 'llm_chat' || node.type === 'openai_chat') {
      return Boolean(localData.config?.model);
    }
    
    // OpenAI API LLM node needs endpoint, API key and model
    if (node.type === 'openai_api_llm') {
      return Boolean(
        localData.config?.endpoint && 
        localData.config?.apiKey && 
        localData.config?.model
      );
    }
    
    return false;
  };

  if (!node || !localData || !nodeTypeDef) {
    return (
      <Box
        sx={{
          width: 320,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          borderLeft: '1px solid',
          borderColor: 'divider',
          bgcolor: 'background.paper',
          p: 3,
        }}
      >
        <Typography variant="body2" color="text.secondary" textAlign="center">
          Select a node to view and edit its properties
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        width: 320,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderLeft: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.paper',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          gap: 1,
        }}
      >
        <Box sx={{ flex: 1 }}>
          <Typography variant="subtitle2" fontWeight={600}>
            Node Properties
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {nodeTypeDef.displayName}
          </Typography>
        </Box>
        <Tooltip title="Duplicate">
          <IconButton
            size="small"
            onClick={() => onNodeDuplicate(node.id)}
          >
            <DuplicateIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Delete">
          <IconButton
            size="small"
            onClick={() => onNodeDelete(node.id)}
            sx={{ color: 'error.main' }}
          >
            <DeleteIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <IconButton size="small" onClick={onClose}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Content */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        {/* Basic info */}
        <Box sx={{ mb: 3 }}>
          <TextField
            label="Node Name"
            value={localData.label}
            onChange={(e) => handleLabelChange(e.target.value)}
            fullWidth
            size="small"
            sx={{ mb: 2 }}
          />

          <Paper
            elevation={0}
            sx={{
              p: 1.5,
              bgcolor: alpha('#6366f1', 0.05),
              border: '1px solid',
              borderColor: alpha('#6366f1', 0.1),
              borderRadius: 1,
            }}
          >
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Description
            </Typography>
            <Typography variant="body2">
              {nodeTypeDef.description}
            </Typography>
          </Paper>
        </Box>

        {/* Inputs section */}
        {nodeTypeDef.inputs.length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Inputs
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {nodeTypeDef.inputs.map((input) => (
                <Chip
                  key={input.id}
                  label={`${input.name}${input.required ? '*' : ''}`}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.7rem' }}
                />
              ))}
            </Box>
          </Box>
        )}

        {/* Outputs section */}
        {nodeTypeDef.outputs.length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Outputs
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {nodeTypeDef.outputs.map((output) => (
                <Chip
                  key={output.id}
                  label={output.name}
                  size="small"
                  variant="outlined"
                  color="primary"
                  sx={{ fontSize: '0.7rem' }}
                />
              ))}
            </Box>
          </Box>
        )}

        <Divider sx={{ my: 2 }} />

        {/* Configuration */}
        <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
          Configuration
        </Typography>

        {/* Use custom component for OpenAI API LLM node */}
        {node.type === 'openai_api_llm' ? (
          <OpenAIAPINodeConfig
            config={localData.config}
            onChange={(newConfig) => {
              if (!node || !localData) return;
              const updatedData = {
                ...localData,
                config: newConfig,
              };
              setLocalData(updatedData);
              onNodeUpdate({
                ...node,
                data: updatedData,
              });
            }}
          />
        ) : (
          <>
            {/* Render config fields based on schema */}
            {renderConfigFields(nodeTypeDef.configSchema, localData.config, handleConfigChange)}

            {Object.keys(nodeTypeDef.configSchema.properties || {}).length === 0 && (
              <Typography variant="body2" color="text.secondary">
                This node has no configurable options.
              </Typography>
            )}
          </>
        )}
      </Box>

      {/* Footer */}
      <Box
        sx={{
          p: 2,
          borderTop: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Button
          variant="outlined"
          startIcon={testing ? <CircularProgress size={16} /> : <TestIcon />}
          fullWidth
          disabled={!canTestNode() || testing}
          onClick={handleTestNode}
          sx={{ mb: 1 }}
        >
          {testing ? 'Testing...' : 'Test Node'}
        </Button>
        
        {testResult && (
          <Paper
            elevation={0}
            sx={{
              p: 1.5,
              mb: 1,
              bgcolor: testResult.success ? alpha('#10b981', 0.1) : alpha('#ef4444', 0.1),
              border: '1px solid',
              borderColor: testResult.success ? alpha('#10b981', 0.3) : alpha('#ef4444', 0.3),
              borderRadius: 1,
            }}
          >
            <Typography
              variant="caption"
              sx={{
                color: testResult.success ? '#10b981' : '#ef4444',
                display: 'block',
                mb: testResult.data ? 0.5 : 0,
              }}
            >
              {testResult.message}
            </Typography>
            {testResult.data && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                Tokens: {testResult.data.total_tokens || 0} (prompt: {testResult.data.prompt_tokens || 0}, completion: {testResult.data.completion_tokens || 0})
              </Typography>
            )}
          </Paper>
        )}
        
        <Typography variant="caption" color="text.secondary" display="block" textAlign="center">
          Node ID: {node.id}
        </Typography>
      </Box>
    </Box>
  );
};

// Render configuration fields based on JSON schema
function renderConfigFields(
  schema: Record<string, any>,
  config: Record<string, any>,
  onChange: (key: string, value: any) => void
) {
  const properties = schema.properties || {};
  const fields = Object.entries(properties);

  if (fields.length === 0) return null;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {fields.map(([key, propSchema]: [string, any]) => (
        <ConfigField
          key={key}
          name={key}
          schema={propSchema}
          value={config[key]}
          onChange={(value) => onChange(key, value)}
        />
      ))}
    </Box>
  );
}

// Individual config field component
interface ConfigFieldProps {
  name: string;
  schema: Record<string, any>;
  value: any;
  onChange: (value: any) => void;
}

const ConfigField: React.FC<ConfigFieldProps> = ({ name, schema, value, onChange }) => {
  const label = schema.title || formatLabel(name);
  const description = schema.description;
  const type = schema.type;
  const enumValues = schema.enum;
  const format = schema.format;
  const defaultValue = schema.default;
  const minimum = schema.minimum;
  const maximum = schema.maximum;

  // Use default value if value is undefined
  const currentValue = value ?? defaultValue ?? '';

  // Select field
  if (enumValues) {
    return (
      <FormControl fullWidth size="small">
        <InputLabel>{label}</InputLabel>
        <Select
          value={currentValue}
          label={label}
          onChange={(e) => onChange(e.target.value)}
        >
          {enumValues.map((opt: string) => (
            <MenuItem key={opt} value={opt}>
              {opt}
            </MenuItem>
          ))}
        </Select>
        {description && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
            {description}
          </Typography>
        )}
      </FormControl>
    );
  }

  // Boolean field
  if (type === 'boolean') {
    return (
      <Box>
        <FormControlLabel
          control={
            <Switch
              checked={Boolean(currentValue)}
              onChange={(e) => onChange(e.target.checked)}
              size="small"
            />
          }
          label={label}
        />
        {description && (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 6 }}>
            {description}
          </Typography>
        )}
      </Box>
    );
  }

  // Number field with range (slider)
  if (type === 'number' && minimum !== undefined && maximum !== undefined) {
    return (
      <Box>
        <Typography variant="body2" gutterBottom>
          {label}: {currentValue || minimum}
        </Typography>
        <Slider
          value={currentValue || minimum}
          onChange={(_, val) => onChange(val)}
          min={minimum}
          max={maximum}
          step={schema.step || (maximum - minimum) / 100}
          size="small"
        />
        {description && (
          <Typography variant="caption" color="text.secondary">
            {description}
          </Typography>
        )}
      </Box>
    );
  }

  // Number field
  if (type === 'number' || type === 'integer') {
    return (
      <TextField
        label={label}
        type="number"
        value={currentValue}
        onChange={(e) => onChange(type === 'integer' ? parseInt(e.target.value) : parseFloat(e.target.value))}
        fullWidth
        size="small"
        helperText={description}
        inputProps={{
          min: minimum,
          max: maximum,
        }}
      />
    );
  }

  // Multiline text (code, template, etc.)
  if (format === 'code' || format === 'template' || format === 'sql' || format === 'graphql') {
    return (
      <TextField
        label={label}
        value={currentValue}
        onChange={(e) => onChange(e.target.value)}
        fullWidth
        size="small"
        multiline
        rows={6}
        helperText={description}
        sx={{
          '& .MuiInputBase-input': {
            fontFamily: 'monospace',
            fontSize: '0.8rem',
          },
        }}
      />
    );
  }

  // Password field
  if (format === 'password') {
    return (
      <TextField
        label={label}
        type="password"
        value={currentValue}
        onChange={(e) => onChange(e.target.value)}
        fullWidth
        size="small"
        helperText={description}
      />
    );
  }

  // Model selector field
  if (format === 'model-select') {
    return (
      <ModelSelector
        value={currentValue}
        onChange={onChange}
        label={label}
        description={description}
        required={schema.required}
      />
    );
  }

  // Default: text field
  return (
    <TextField
      label={label}
      value={currentValue}
      onChange={(e) => onChange(e.target.value)}
      fullWidth
      size="small"
      helperText={description}
    />
  );
};

// Helper to format field names
function formatLabel(name: string): string {
  return name
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, (str) => str.toUpperCase())
    .replace(/_/g, ' ');
}

export default PropertyPanel;
