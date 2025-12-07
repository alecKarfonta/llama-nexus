import React, { useState, useCallback } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  IconButton,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Switch,
  FormControlLabel,
  Tooltip,
  alpha,
  Divider,
  Tabs,
  Tab,
  Alert,
} from '@mui/material'
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  ContentCopy as CopyIcon,
  Code as CodeIcon,
  TextFields as StringIcon,
  Numbers as NumberIcon,
  ToggleOn as BooleanIcon,
  List as ArrayIcon,
  DataObject as ObjectIcon,
  Check as CheckIcon,
} from '@mui/icons-material'

// JSON Schema types
type SchemaType = 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object' | 'null'

interface SchemaProperty {
  id: string
  name: string
  type: SchemaType
  description?: string
  required?: boolean
  // String constraints
  minLength?: number
  maxLength?: number
  pattern?: string
  enum?: string[]
  format?: string
  // Number constraints
  minimum?: number
  maximum?: number
  multipleOf?: number
  // Array constraints
  minItems?: number
  maxItems?: number
  items?: SchemaProperty
  // Object constraints
  properties?: SchemaProperty[]
}

interface ConstraintEditorProps {
  open: boolean
  onClose: () => void
  onApply: (schema: object, grammar?: string) => void
  initialSchema?: object
}

// Type icons and colors
const typeConfig: Record<SchemaType, { icon: React.ReactNode; color: string; label: string }> = {
  string: { icon: <StringIcon />, color: '#10b981', label: 'String' },
  number: { icon: <NumberIcon />, color: '#6366f1', label: 'Number' },
  integer: { icon: <NumberIcon />, color: '#8b5cf6', label: 'Integer' },
  boolean: { icon: <BooleanIcon />, color: '#f59e0b', label: 'Boolean' },
  array: { icon: <ArrayIcon />, color: '#06b6d4', label: 'Array' },
  object: { icon: <ObjectIcon />, color: '#ec4899', label: 'Object' },
  null: { icon: <CheckIcon />, color: '#64748b', label: 'Null' },
}

// String formats
const stringFormats = [
  { value: '', label: 'None' },
  { value: 'date-time', label: 'DateTime (ISO 8601)' },
  { value: 'date', label: 'Date' },
  { value: 'time', label: 'Time' },
  { value: 'email', label: 'Email' },
  { value: 'uri', label: 'URI' },
  { value: 'uuid', label: 'UUID' },
  { value: 'hostname', label: 'Hostname' },
  { value: 'ipv4', label: 'IPv4' },
  { value: 'ipv6', label: 'IPv6' },
]

// Property Editor Component
interface PropertyEditorProps {
  property: SchemaProperty
  onChange: (property: SchemaProperty) => void
  onDelete: () => void
  depth?: number
}

const PropertyEditor: React.FC<PropertyEditorProps> = ({ property, onChange, onDelete, depth = 0 }) => {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const config = typeConfig[property.type]

  const handleChange = (field: keyof SchemaProperty, value: any) => {
    onChange({ ...property, [field]: value })
  }

  const addNestedProperty = () => {
    if (property.type === 'object') {
      const newProp: SchemaProperty = {
        id: `prop-${Date.now()}`,
        name: `property${(property.properties?.length || 0) + 1}`,
        type: 'string',
      }
      onChange({
        ...property,
        properties: [...(property.properties || []), newProp],
      })
    } else if (property.type === 'array' && !property.items) {
      onChange({
        ...property,
        items: {
          id: `items-${Date.now()}`,
          name: 'items',
          type: 'string',
        },
      })
    }
  }

  return (
    <Box
      sx={{
        p: 2,
        mb: 1,
        ml: depth * 2,
        borderRadius: 2,
        bgcolor: alpha(config.color, 0.05),
        border: `1px solid ${alpha(config.color, 0.15)}`,
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <Box sx={{ color: config.color, display: 'flex' }}>{config.icon}</Box>
        <TextField
          size="small"
          value={property.name}
          onChange={(e) => handleChange('name', e.target.value)}
          placeholder="Property name"
          sx={{ flex: 1 }}
        />
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <Select
            value={property.type}
            onChange={(e) => handleChange('type', e.target.value)}
          >
            {(Object.keys(typeConfig) as SchemaType[]).map((type) => (
              <MenuItem key={type} value={type}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {typeConfig[type].icon}
                  {typeConfig[type].label}
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={property.required || false}
              onChange={(e) => handleChange('required', e.target.checked)}
            />
          }
          label="Required"
          sx={{ ml: 1, '& .MuiTypography-root': { fontSize: '0.75rem' } }}
        />
        <IconButton size="small" onClick={onDelete} sx={{ '&:hover': { color: '#ef4444' } }}>
          <DeleteIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Description */}
      <TextField
        size="small"
        fullWidth
        value={property.description || ''}
        onChange={(e) => handleChange('description', e.target.value)}
        placeholder="Description (optional)"
        sx={{ mb: 2 }}
      />

      {/* Type-specific constraints */}
      <Button
        size="small"
        onClick={() => setShowAdvanced(!showAdvanced)}
        sx={{ mb: showAdvanced ? 2 : 0 }}
      >
        {showAdvanced ? 'Hide' : 'Show'} Constraints
      </Button>

      {showAdvanced && (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {/* String constraints */}
          {property.type === 'string' && (
            <>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <TextField
                  size="small"
                  type="number"
                  label="Min Length"
                  value={property.minLength || ''}
                  onChange={(e) => handleChange('minLength', e.target.value ? parseInt(e.target.value) : undefined)}
                  sx={{ flex: 1 }}
                />
                <TextField
                  size="small"
                  type="number"
                  label="Max Length"
                  value={property.maxLength || ''}
                  onChange={(e) => handleChange('maxLength', e.target.value ? parseInt(e.target.value) : undefined)}
                  sx={{ flex: 1 }}
                />
              </Box>
              <TextField
                size="small"
                fullWidth
                label="Pattern (Regex)"
                value={property.pattern || ''}
                onChange={(e) => handleChange('pattern', e.target.value || undefined)}
                placeholder="^[a-zA-Z]+$"
              />
              <FormControl size="small" fullWidth>
                <InputLabel>Format</InputLabel>
                <Select
                  value={property.format || ''}
                  label="Format"
                  onChange={(e) => handleChange('format', e.target.value || undefined)}
                >
                  {stringFormats.map((f) => (
                    <MenuItem key={f.value} value={f.value}>{f.label}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              <TextField
                size="small"
                fullWidth
                label="Enum Values (comma separated)"
                value={property.enum?.join(', ') || ''}
                onChange={(e) => handleChange('enum', e.target.value ? e.target.value.split(',').map(s => s.trim()) : undefined)}
                placeholder="option1, option2, option3"
              />
            </>
          )}

          {/* Number constraints */}
          {(property.type === 'number' || property.type === 'integer') && (
            <Box sx={{ display: 'flex', gap: 2 }}>
              <TextField
                size="small"
                type="number"
                label="Minimum"
                value={property.minimum ?? ''}
                onChange={(e) => handleChange('minimum', e.target.value ? parseFloat(e.target.value) : undefined)}
                sx={{ flex: 1 }}
              />
              <TextField
                size="small"
                type="number"
                label="Maximum"
                value={property.maximum ?? ''}
                onChange={(e) => handleChange('maximum', e.target.value ? parseFloat(e.target.value) : undefined)}
                sx={{ flex: 1 }}
              />
              <TextField
                size="small"
                type="number"
                label="Multiple Of"
                value={property.multipleOf ?? ''}
                onChange={(e) => handleChange('multipleOf', e.target.value ? parseFloat(e.target.value) : undefined)}
                sx={{ flex: 1 }}
              />
            </Box>
          )}

          {/* Array constraints */}
          {property.type === 'array' && (
            <>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <TextField
                  size="small"
                  type="number"
                  label="Min Items"
                  value={property.minItems ?? ''}
                  onChange={(e) => handleChange('minItems', e.target.value ? parseInt(e.target.value) : undefined)}
                  sx={{ flex: 1 }}
                />
                <TextField
                  size="small"
                  type="number"
                  label="Max Items"
                  value={property.maxItems ?? ''}
                  onChange={(e) => handleChange('maxItems', e.target.value ? parseInt(e.target.value) : undefined)}
                  sx={{ flex: 1 }}
                />
              </Box>
              {!property.items && (
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<AddIcon />}
                  onClick={addNestedProperty}
                >
                  Define Array Item Type
                </Button>
              )}
              {property.items && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                    Array Item Schema:
                  </Typography>
                  <PropertyEditor
                    property={property.items}
                    onChange={(items) => handleChange('items', items)}
                    onDelete={() => handleChange('items', undefined)}
                    depth={depth + 1}
                  />
                </Box>
              )}
            </>
          )}

          {/* Object nested properties */}
          {property.type === 'object' && (
            <>
              <Button
                variant="outlined"
                size="small"
                startIcon={<AddIcon />}
                onClick={addNestedProperty}
              >
                Add Nested Property
              </Button>
              {property.properties && property.properties.length > 0 && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                    Nested Properties:
                  </Typography>
                  {property.properties.map((nested, index) => (
                    <PropertyEditor
                      key={nested.id}
                      property={nested}
                      onChange={(updated) => {
                        const newProps = [...(property.properties || [])]
                        newProps[index] = updated
                        handleChange('properties', newProps)
                      }}
                      onDelete={() => {
                        const newProps = (property.properties || []).filter((_, i) => i !== index)
                        handleChange('properties', newProps)
                      }}
                      depth={depth + 1}
                    />
                  ))}
                </Box>
              )}
            </>
          )}
        </Box>
      )}
    </Box>
  )
}

// Convert internal schema to JSON Schema
const toJsonSchema = (properties: SchemaProperty[]): object => {
  const required: string[] = []
  const props: Record<string, any> = {}

  properties.forEach((prop) => {
    if (prop.required) {
      required.push(prop.name)
    }

    const schemaProp: Record<string, any> = {
      type: prop.type,
    }

    if (prop.description) schemaProp.description = prop.description

    // String constraints
    if (prop.type === 'string') {
      if (prop.minLength !== undefined) schemaProp.minLength = prop.minLength
      if (prop.maxLength !== undefined) schemaProp.maxLength = prop.maxLength
      if (prop.pattern) schemaProp.pattern = prop.pattern
      if (prop.format) schemaProp.format = prop.format
      if (prop.enum && prop.enum.length > 0) schemaProp.enum = prop.enum
    }

    // Number constraints
    if (prop.type === 'number' || prop.type === 'integer') {
      if (prop.minimum !== undefined) schemaProp.minimum = prop.minimum
      if (prop.maximum !== undefined) schemaProp.maximum = prop.maximum
      if (prop.multipleOf !== undefined) schemaProp.multipleOf = prop.multipleOf
    }

    // Array constraints
    if (prop.type === 'array') {
      if (prop.minItems !== undefined) schemaProp.minItems = prop.minItems
      if (prop.maxItems !== undefined) schemaProp.maxItems = prop.maxItems
      if (prop.items) {
        schemaProp.items = toJsonSchema([prop.items]).properties?.[prop.items.name] || { type: 'string' }
      }
    }

    // Object constraints
    if (prop.type === 'object' && prop.properties && prop.properties.length > 0) {
      const nested = toJsonSchema(prop.properties)
      schemaProp.properties = (nested as any).properties
      if ((nested as any).required?.length > 0) {
        schemaProp.required = (nested as any).required
      }
    }

    props[prop.name] = schemaProp
  })

  return {
    type: 'object',
    properties: props,
    ...(required.length > 0 ? { required } : {}),
  }
}

// Main Constraint Editor Component
export const ConstraintEditor: React.FC<ConstraintEditorProps> = ({
  open,
  onClose,
  onApply,
  initialSchema,
}) => {
  const [properties, setProperties] = useState<SchemaProperty[]>([])
  const [activeTab, setActiveTab] = useState(0)
  const [schemaName, setSchemaName] = useState('response')
  const [copied, setCopied] = useState(false)

  // Generate JSON Schema
  const jsonSchema = {
    $schema: 'http://json-schema.org/draft-07/schema#',
    title: schemaName,
    ...toJsonSchema(properties),
  }

  const addProperty = () => {
    const newProp: SchemaProperty = {
      id: `prop-${Date.now()}`,
      name: `property${properties.length + 1}`,
      type: 'string',
    }
    setProperties([...properties, newProp])
  }

  const updateProperty = (index: number, updated: SchemaProperty) => {
    const newProps = [...properties]
    newProps[index] = updated
    setProperties(newProps)
  }

  const deleteProperty = (index: number) => {
    setProperties(properties.filter((_, i) => i !== index))
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(jsonSchema, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleApply = () => {
    onApply(jsonSchema)
    onClose()
  }

  // Preset templates
  const applyPreset = (preset: string) => {
    switch (preset) {
      case 'answer':
        setProperties([
          { id: '1', name: 'answer', type: 'string', required: true, description: 'The answer to the question' },
          { id: '2', name: 'confidence', type: 'number', minimum: 0, maximum: 1, description: 'Confidence score' },
        ])
        setSchemaName('answer_response')
        break
      case 'classification':
        setProperties([
          { id: '1', name: 'category', type: 'string', required: true, enum: ['positive', 'negative', 'neutral'] },
          { id: '2', name: 'score', type: 'number', minimum: 0, maximum: 1, required: true },
          { id: '3', name: 'reasoning', type: 'string' },
        ])
        setSchemaName('classification_response')
        break
      case 'extraction':
        setProperties([
          { id: '1', name: 'entities', type: 'array', required: true, items: { id: 'e1', name: 'entity', type: 'object', properties: [
            { id: 'e1a', name: 'name', type: 'string', required: true },
            { id: 'e1b', name: 'type', type: 'string', required: true },
            { id: 'e1c', name: 'value', type: 'string' },
          ]}},
        ])
        setSchemaName('extraction_response')
        break
      case 'summary':
        setProperties([
          { id: '1', name: 'summary', type: 'string', required: true, maxLength: 500 },
          { id: '2', name: 'key_points', type: 'array', items: { id: 'kp', name: 'point', type: 'string' }},
          { id: '3', name: 'word_count', type: 'integer', minimum: 0 },
        ])
        setSchemaName('summary_response')
        break
    }
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <CodeIcon sx={{ color: '#6366f1' }} />
        Output Constraint Editor
        <Chip
          label="JSON Schema"
          size="small"
          sx={{
            ml: 1,
            bgcolor: alpha('#6366f1', 0.1),
            color: '#818cf8',
            fontWeight: 600,
            fontSize: '0.6875rem',
          }}
        />
      </DialogTitle>

      <DialogContent>
        <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 2 }}>
          <Tab label="Visual Builder" />
          <Tab label="JSON Schema" />
          <Tab label="GBNF Grammar" />
        </Tabs>

        {/* Visual Builder Tab */}
        {activeTab === 0 && (
          <Box>
            {/* Presets */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Quick Presets
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {[
                  { id: 'answer', label: 'Q&A Response' },
                  { id: 'classification', label: 'Classification' },
                  { id: 'extraction', label: 'Entity Extraction' },
                  { id: 'summary', label: 'Summary' },
                ].map((preset) => (
                  <Chip
                    key={preset.id}
                    label={preset.label}
                    onClick={() => applyPreset(preset.id)}
                    sx={{
                      cursor: 'pointer',
                      '&:hover': { bgcolor: alpha('#6366f1', 0.15) },
                    }}
                  />
                ))}
              </Box>
            </Box>

            <Divider sx={{ mb: 3 }} />

            {/* Schema Name */}
            <TextField
              size="small"
              label="Schema Name"
              value={schemaName}
              onChange={(e) => setSchemaName(e.target.value)}
              sx={{ mb: 3, width: 300 }}
            />

            {/* Properties */}
            <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography variant="subtitle2">Properties</Typography>
              <Button
                variant="contained"
                size="small"
                startIcon={<AddIcon />}
                onClick={addProperty}
              >
                Add Property
              </Button>
            </Box>

            {properties.length === 0 ? (
              <Alert severity="info" sx={{ mb: 2 }}>
                Add properties to define the structure of the expected output.
              </Alert>
            ) : (
              properties.map((prop, index) => (
                <PropertyEditor
                  key={prop.id}
                  property={prop}
                  onChange={(updated) => updateProperty(index, updated)}
                  onDelete={() => deleteProperty(index)}
                />
              ))
            )}
          </Box>
        )}

        {/* JSON Schema Tab */}
        {activeTab === 1 && (
          <Box>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
              <Tooltip title={copied ? 'Copied!' : 'Copy'}>
                <IconButton size="small" onClick={handleCopy}>
                  {copied ? <CheckIcon color="success" /> : <CopyIcon />}
                </IconButton>
              </Tooltip>
            </Box>
            <Box
              sx={{
                bgcolor: 'rgba(0, 0, 0, 0.3)',
                borderRadius: 2,
                p: 2,
                fontFamily: 'monospace',
                fontSize: '0.8125rem',
                overflow: 'auto',
                maxHeight: 400,
                whiteSpace: 'pre',
              }}
            >
              {JSON.stringify(jsonSchema, null, 2)}
            </Box>
          </Box>
        )}

        {/* GBNF Grammar Tab */}
        {activeTab === 2 && (
          <Box>
            <Alert severity="info" sx={{ mb: 2 }}>
              GBNF (GGML BNF) grammar is automatically generated from your JSON Schema.
              This grammar constrains the model to produce valid JSON matching your schema.
            </Alert>
            <Box
              sx={{
                bgcolor: 'rgba(0, 0, 0, 0.3)',
                borderRadius: 2,
                p: 2,
                fontFamily: 'monospace',
                fontSize: '0.75rem',
                overflow: 'auto',
                maxHeight: 400,
                whiteSpace: 'pre',
              }}
            >
              {generateGbnfGrammar(properties)}
            </Box>
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ p: 2 }}>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained"
          onClick={handleApply}
          disabled={properties.length === 0}
        >
          Apply Constraints
        </Button>
      </DialogActions>
    </Dialog>
  )
}

// Generate GBNF grammar from properties (simplified)
function generateGbnfGrammar(properties: SchemaProperty[]): string {
  let grammar = `# GBNF Grammar for structured output\n\n`
  grammar += `root ::= "{" ws `
  
  const propRules: string[] = []
  const typeRules = new Set<string>()
  
  properties.forEach((prop, i) => {
    const ruleName = `prop_${prop.name}`
    propRules.push(`"\\"${prop.name}\\"" ws ":" ws ${getTypeRule(prop.type)}`)
    typeRules.add(prop.type)
  })
  
  grammar += propRules.join(' "," ws ') + ` ws "}"\n\n`
  
  // Add type rules
  grammar += `# Type definitions\n`
  grammar += `string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""\n`
  grammar += `number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?\n`
  grammar += `integer ::= "-"? [0-9]+\n`
  grammar += `boolean ::= "true" | "false"\n`
  grammar += `null ::= "null"\n`
  grammar += `array ::= "[" ws (value ("," ws value)*)? ws "]"\n`
  grammar += `object ::= "{" ws (string ws ":" ws value ("," ws string ws ":" ws value)*)? ws "}"\n`
  grammar += `value ::= string | number | boolean | null | array | object\n`
  grammar += `ws ::= [ \\t\\n]*\n`
  
  return grammar
}

function getTypeRule(type: SchemaType): string {
  switch (type) {
    case 'string': return 'string'
    case 'number': return 'number'
    case 'integer': return 'integer'
    case 'boolean': return 'boolean'
    case 'array': return 'array'
    case 'object': return 'object'
    case 'null': return 'null'
    default: return 'value'
  }
}

export default ConstraintEditor
