import React, { useEffect, useState } from 'react'
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  TextField,
  Select,
  MenuItem,
  Button,
  Alert,
  Chip,
  CircularProgress,
  FormHelperText,
} from '@mui/material'
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  RestartAlt as RestartIcon,
  Save as SaveIcon,
  Science as TestIcon,
} from '@mui/icons-material'

export const EmbeddingDeployPage: React.FC = () => {
  // Embedding service state
  const [embeddingStatus, setEmbeddingStatus] = useState<any>(null)
  const [embeddingConfig, setEmbeddingConfig] = useState<any>(null)
  const [embeddingLoading, setEmbeddingLoading] = useState<'start' | 'stop' | 'restart' | null>(null)
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<any[]>([])
  const [embeddingError, setEmbeddingError] = useState<string | null>(null)
  const [embeddingSuccess, setEmbeddingSuccess] = useState<string | null>(null)
  
  // RAG embedding configuration
  const [ragEmbeddingConfig, setRagEmbeddingConfig] = useState<any>(null)
  const [ragEmbeddingLoading, setRagEmbeddingLoading] = useState(false)
  
  // Test embedding state
  const [testText, setTestText] = useState('The quick brown fox jumps over the lazy dog.')
  const [testResult, setTestResult] = useState<any>(null)
  const [testLoading, setTestLoading] = useState(false)
  const [testError, setTestError] = useState<string | null>(null)
  
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const init = async () => {
      try {
        setLoading(true)
        
        // Load embedding service config and status
        const embedCfgRes = await fetch('/api/v1/embedding/config')
        if (embedCfgRes.ok) {
          const embedCfg = await embedCfgRes.json()
          setEmbeddingConfig(embedCfg.config)
          setAvailableEmbeddingModels(embedCfg.available_models || [])
        }
        
        const embedStatusRes = await fetch('/api/v1/embedding/status')
        if (embedStatusRes.ok) {
          const embedStatus = await embedStatusRes.json()
          setEmbeddingStatus(embedStatus)
        }
        
        // Load RAG embedding configuration
        const ragEmbedCfgRes = await fetch('/api/v1/rag/embeddings/config')
        if (ragEmbedCfgRes.ok) {
          const ragEmbedCfg = await ragEmbedCfgRes.json()
          setRagEmbeddingConfig(ragEmbedCfg)
        }
      } catch (e) {
        console.error('Failed to fetch embedding service info:', e)
      } finally {
        setLoading(false)
      }
    }
    
    init()
  }, [])

  // Refresh embedding status periodically
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/v1/embedding/status')
        if (res.ok) {
          const status = await res.json()
          setEmbeddingStatus(status)
        }
      } catch (e) {
        // Silently fail
      }
    }, 5000)
    
    return () => clearInterval(interval)
  }, [])

  // Embedding service action handlers
  const runEmbeddingAction = async (action: 'start' | 'stop' | 'restart') => {
    try {
      setEmbeddingLoading(action)
      setEmbeddingError(null)
      
      const res = await fetch(`/api/v1/embedding/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (!res.ok) {
        throw new Error(`Failed to ${action} embedding service`)
      }
      
      const result = await res.json()
      setEmbeddingStatus(result.status)
      setEmbeddingSuccess(result.message)
      
      // Clear success message after 3 seconds
      setTimeout(() => setEmbeddingSuccess(null), 3000)
    } catch (e) {
      setEmbeddingError(e instanceof Error ? e.message : `Failed to ${action} embedding service`)
    } finally {
      setEmbeddingLoading(null)
    }
  }
  
  const updateEmbeddingConfig = (path: string, value: any) => {
    if (!embeddingConfig) return
    
    const keys = path.split('.')
    const newConfig = JSON.parse(JSON.stringify(embeddingConfig))
    
    let current = newConfig
    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) current[keys[i]] = {}
      current = current[keys[i]]
    }
    current[keys[keys.length - 1]] = value
    
    setEmbeddingConfig(newConfig)
  }
  
  const saveEmbeddingConfig = async () => {
    try {
      const res = await fetch('/api/v1/embedding/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(embeddingConfig)
      })
      
      if (!res.ok) {
        throw new Error('Failed to save embedding config')
      }
      
      const result = await res.json()
      setEmbeddingSuccess(result.message)
      setTimeout(() => setEmbeddingSuccess(null), 3000)
    } catch (e) {
      setEmbeddingError(e instanceof Error ? e.message : 'Failed to save embedding config')
    }
  }
  
  const updateRagEmbeddingConfig = (field: string, value: any) => {
    if (!ragEmbeddingConfig) return
    setRagEmbeddingConfig({ ...ragEmbeddingConfig, [field]: value })
  }
  
  const saveRagEmbeddingConfig = async () => {
    try {
      setRagEmbeddingLoading(true)
      const res = await fetch('/api/v1/rag/embeddings/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ragEmbeddingConfig)
      })
      
      if (!res.ok) {
        throw new Error('Failed to save RAG embedding config')
      }
      
      const result = await res.json()
      setEmbeddingSuccess('RAG embedding configuration saved successfully')
      setTimeout(() => setEmbeddingSuccess(null), 3000)
    } catch (e) {
      setEmbeddingError(e instanceof Error ? e.message : 'Failed to save RAG embedding config')
    } finally {
      setRagEmbeddingLoading(false)
    }
  }
  
  const testEmbedding = async () => {
    if (!testText.trim()) {
      setTestError('Please enter some text to test')
      return
    }
    
    try {
      setTestLoading(true)
      setTestError(null)
      setTestResult(null)
      
      const serviceUrl = embeddingStatus?.endpoint || 'http://localhost:8602'
      const apiKey = embeddingConfig?.server?.api_key || 'llamacpp-embed'
      const modelName = embeddingConfig?.model?.name || 'nomic-embed-text-v1.5'
      
      const startTime = Date.now()
      const res = await fetch(`${serviceUrl}/v1/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          input: testText,
          model: modelName
        })
      })
      
      if (!res.ok) {
        const errorText = await res.text()
        throw new Error(`Service returned ${res.status}: ${errorText}`)
      }
      
      const result = await res.json()
      const endTime = Date.now()
      const timeTaken = endTime - startTime
      
      setTestResult({
        ...result,
        timeTaken,
        vectorLength: result.data?.[0]?.embedding?.length || 0
      })
      
    } catch (e) {
      setTestError(e instanceof Error ? e.message : 'Failed to test embedding')
    } finally {
      setTestLoading(false)
    }
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <Box sx={{ p: { xs: 2, sm: 3, md: 4 }, maxWidth: '100%', mx: 'auto' }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
          Embedding Model Deployment
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Deploy and manage dedicated embedding models for RAG operations
        </Typography>
      </Box>

      {embeddingError && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setEmbeddingError(null)}>
          {embeddingError}
        </Alert>
      )}
      
      {embeddingSuccess && (
        <Alert severity="success" sx={{ mb: 3 }} onClose={() => setEmbeddingSuccess(null)}>
          {embeddingSuccess}
        </Alert>
      )}

      {embeddingConfig && (
        <Box>
          {/* Status and Control */}
          <Card sx={{
            mb: 3,
            borderRadius: 2,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
            border: '1px solid',
            borderColor: 'divider',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                Service Status
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                <Chip
                  label={embeddingStatus?.running ? 'Running' : 'Stopped'}
                  color={embeddingStatus?.running ? 'success' : 'default'}
                  size="small"
                />
                {embeddingStatus?.running && (
                  <>
                    <Typography variant="body2" color="text.secondary">
                      Uptime: {Math.floor(embeddingStatus.uptime / 60)}m {Math.floor(embeddingStatus.uptime % 60)}s
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Model: {embeddingStatus.model?.name}
                    </Typography>
                  </>
                )}
              </Box>
              
              {embeddingStatus?.endpoint && (
                <Alert severity="info" sx={{ mb: 3 }}>
                  <strong>Service Endpoint:</strong> {embeddingStatus.endpoint}
                </Alert>
              )}
              
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  startIcon={embeddingLoading === 'start' ? <CircularProgress size={20} /> : <StartIcon />}
                  onClick={() => runEmbeddingAction('start')}
                  disabled={embeddingLoading !== null || embeddingStatus?.running}
                  sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                >
                  Start Service
                </Button>
                <Button
                  variant="contained"
                  color="warning"
                  size="large"
                  startIcon={embeddingLoading === 'restart' ? <CircularProgress size={20} /> : <RestartIcon />}
                  onClick={() => runEmbeddingAction('restart')}
                  disabled={embeddingLoading !== null}
                  sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                >
                  Restart
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  size="large"
                  startIcon={embeddingLoading === 'stop' ? <CircularProgress size={20} /> : <StopIcon />}
                  onClick={() => runEmbeddingAction('stop')}
                  disabled={embeddingLoading !== null || !embeddingStatus?.running}
                  sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                >
                  Stop
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* Model Configuration */}
          <Card sx={{
            mb: 3,
            borderRadius: 2,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
            border: '1px solid',
            borderColor: 'divider',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                Model Configuration
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    Embedding Model
                  </Typography>
                  <Select
                    fullWidth
                    value={embeddingConfig?.model?.name || ''}
                    onChange={(e) => updateEmbeddingConfig('model.name', e.target.value)}
                    sx={{ fontSize: '0.875rem' }}
                  >
                    {availableEmbeddingModels.map((model) => (
                      <MenuItem key={model.name} value={model.name}>
                        {model.name} ({model.dimensions}D, {model.max_tokens} tokens)
                      </MenuItem>
                    ))}
                  </Select>
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    {availableEmbeddingModels.find(m => m.name === embeddingConfig?.model?.name)?.description}
                  </FormHelperText>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    Model Variant (Quantization)
                  </Typography>
                  <Select
                    fullWidth
                    value={embeddingConfig?.model?.variant || 'Q8_0'}
                    onChange={(e) => updateEmbeddingConfig('model.variant', e.target.value)}
                    sx={{ fontSize: '0.875rem' }}
                  >
                    <MenuItem value="Q8_0">Q8_0 (Recommended - High Quality)</MenuItem>
                    <MenuItem value="Q4_K_M">Q4_K_M (Balanced)</MenuItem>
                    <MenuItem value="Q4_0">Q4_0 (Smaller, Faster)</MenuItem>
                  </Select>
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    Higher quantization = better quality but larger size
                  </FormHelperText>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    Context Size
                  </Typography>
                  <TextField
                    fullWidth
                    type="number"
                    value={embeddingConfig?.model?.context_size || 8192}
                    onChange={(e) => updateEmbeddingConfig('model.context_size', parseInt(e.target.value))}
                    inputProps={{ min: 512, max: 32768 }}
                  />
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    Maximum context length for embeddings
                  </FormHelperText>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    GPU Layers
                  </Typography>
                  <TextField
                    fullWidth
                    type="number"
                    value={embeddingConfig?.model?.gpu_layers || 999}
                    onChange={(e) => updateEmbeddingConfig('model.gpu_layers', parseInt(e.target.value))}
                    inputProps={{ min: 0, max: 999 }}
                  />
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    Number of layers to offload to GPU (999 = all)
                  </FormHelperText>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    Execution Mode
                  </Typography>
                  <Select
                    fullWidth
                    value={embeddingConfig?.execution?.mode || 'gpu'}
                    onChange={(e) => updateEmbeddingConfig('execution.mode', e.target.value)}
                    sx={{ fontSize: '0.875rem' }}
                  >
                    <MenuItem value="gpu">GPU Acceleration</MenuItem>
                    <MenuItem value="cpu">CPU Only</MenuItem>
                  </Select>
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    Choose GPU or CPU execution
                  </FormHelperText>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                    CUDA Devices
                  </Typography>
                  <TextField
                    fullWidth
                    value={embeddingConfig?.execution?.cuda_devices || 'all'}
                    onChange={(e) => updateEmbeddingConfig('execution.cuda_devices', e.target.value)}
                    disabled={embeddingConfig?.execution?.mode === 'cpu'}
                  />
                  <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                    GPU devices to use (all, 0, 0,1, etc.)
                  </FormHelperText>
                </Grid>
              </Grid>
              
              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={saveEmbeddingConfig}
                  sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                >
                  Save Configuration
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* RAG Integration */}
          {ragEmbeddingConfig && (
            <Card sx={{
              mb: 3,
              borderRadius: 2,
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              border: '1px solid',
              borderColor: 'divider',
            }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                  RAG System Integration
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Configure how the RAG system uses embeddings for document processing and retrieval.
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Box sx={{ flex: 1 }}>
                        <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                          Embedding Source
                        </Typography>
                        <FormHelperText sx={{ mt: 0.5, mb: 2, fontSize: '0.75rem' }}>
                          When enabled, the RAG system will use this deployed embedding service instead of local sentence-transformers.
                          The service must be running for this to work.
                        </FormHelperText>
                      </Box>
                      <Select
                        value={ragEmbeddingConfig?.use_deployed_service ? 'deployed' : 'local'}
                        onChange={(e) => updateRagEmbeddingConfig('use_deployed_service', e.target.value === 'deployed')}
                        sx={{ minWidth: 220, fontSize: '0.875rem' }}
                      >
                        <MenuItem value="local">Local (Sentence Transformers)</MenuItem>
                        <MenuItem value="deployed">Deployed Service</MenuItem>
                      </Select>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Alert 
                      severity={ragEmbeddingConfig?.service_running ? 'success' : 'warning'} 
                      sx={{ mb: 2 }}
                    >
                      {ragEmbeddingConfig?.service_running ? (
                        <>
                          <strong>Embedding service is running</strong>
                          <br />
                          {ragEmbeddingConfig.use_deployed_service 
                            ? 'RAG system is configured to use the deployed service.' 
                            : 'Enable deployed service above to use it for RAG.'}
                        </>
                      ) : (
                        <>
                          <strong>Embedding service is not running</strong>
                          <br />
                          Start the service above to enable RAG integration. The system will fall back to local embeddings if the service is unavailable.
                        </>
                      )}
                    </Alert>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                      Default Embedding Model
                    </Typography>
                    <TextField
                      fullWidth
                      value={ragEmbeddingConfig?.default_model || ''}
                      onChange={(e) => updateRagEmbeddingConfig('default_model', e.target.value)}
                    />
                    <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                      Default model to use when no model is specified
                    </FormHelperText>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                      Service URL
                    </Typography>
                    <TextField
                      fullWidth
                      value={ragEmbeddingConfig?.service_url || ''}
                      onChange={(e) => updateRagEmbeddingConfig('service_url', e.target.value)}
                    />
                    <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                      URL of the embedding service endpoint
                    </FormHelperText>
                  </Grid>
                </Grid>
                
                <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                  <Button
                    variant="contained"
                    startIcon={ragEmbeddingLoading ? <CircularProgress size={20} /> : <SaveIcon />}
                    onClick={saveRagEmbeddingConfig}
                    disabled={ragEmbeddingLoading}
                    sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                  >
                    Save RAG Configuration
                  </Button>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Test Embedding */}
          {embeddingStatus?.running && (
            <Card sx={{
              mb: 3,
              borderRadius: 2,
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              border: '1px solid',
              borderColor: 'divider',
            }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                  Test Embedding
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Test the embedding service with sample text to verify it's working correctly.
                </Typography>
                
                {testError && (
                  <Alert severity="error" sx={{ mb: 2 }} onClose={() => setTestError(null)}>
                    {testError}
                  </Alert>
                )}
                
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Typography gutterBottom sx={{ fontSize: '0.875rem', fontWeight: 500 }}>
                      Test Text
                    </Typography>
                    <TextField
                      fullWidth
                      multiline
                      rows={3}
                      value={testText}
                      onChange={(e) => setTestText(e.target.value)}
                      placeholder="Enter text to generate embeddings..."
                    />
                    <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                      Enter any text to see how the model generates embeddings
                    </FormHelperText>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Button
                      variant="contained"
                      color="primary"
                      size="large"
                      startIcon={testLoading ? <CircularProgress size={20} /> : <TestIcon />}
                      onClick={testEmbedding}
                      disabled={testLoading || !testText.trim()}
                      sx={{ borderRadius: 1.5, fontWeight: 600, px: 3 }}
                    >
                      Generate Embedding
                    </Button>
                  </Grid>
                  
                  {testResult && (
                    <Grid item xs={12}>
                      <Alert severity="success" sx={{ mb: 2 }}>
                        <strong>Embedding Generated Successfully!</strong>
                      </Alert>
                      
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6} md={3}>
                          <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Vector Dimensions
                            </Typography>
                            <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main' }}>
                              {testResult.vectorLength}
                            </Typography>
                          </Card>
                        </Grid>
                        
                        <Grid item xs={12} sm={6} md={3}>
                          <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Processing Time
                            </Typography>
                            <Typography variant="h5" sx={{ fontWeight: 600, color: 'success.main' }}>
                              {testResult.timeTaken}ms
                            </Typography>
                          </Card>
                        </Grid>
                        
                        <Grid item xs={12} sm={6} md={3}>
                          <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Model Used
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '0.9rem' }}>
                              {testResult.model || 'N/A'}
                            </Typography>
                          </Card>
                        </Grid>
                        
                        <Grid item xs={12} sm={6} md={3}>
                          <Card variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Usage
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '0.9rem' }}>
                              {testResult.usage?.total_tokens || 'N/A'} tokens
                            </Typography>
                          </Card>
                        </Grid>
                        
                        <Grid item xs={12}>
                          <Typography variant="body2" gutterBottom sx={{ fontWeight: 500 }}>
                            Embedding Vector (first 10 dimensions):
                          </Typography>
                          <Box sx={{ 
                            p: 2, 
                            bgcolor: 'grey.100', 
                            borderRadius: 1, 
                            fontFamily: 'monospace',
                            fontSize: '0.75rem',
                            overflowX: 'auto',
                            maxHeight: 200,
                            overflowY: 'auto'
                          }}>
                            {testResult.data?.[0]?.embedding?.slice(0, 10).map((val: number, idx: number) => (
                              <span key={idx}>
                                {val.toFixed(6)}
                                {idx < 9 ? ', ' : '...'}
                              </span>
                            )) || 'No embedding data'}
                          </Box>
                          <FormHelperText sx={{ mt: 1, fontSize: '0.75rem' }}>
                            Showing first 10 of {testResult.vectorLength} dimensions
                          </FormHelperText>
                        </Grid>
                      </Grid>
                    </Grid>
                  )}
                </Grid>
              </CardContent>
            </Card>
          )}

          {/* Info */}
          <Card sx={{
            borderRadius: 2,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
            border: '1px solid',
            borderColor: 'divider',
          }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontSize: '1.125rem', fontWeight: 600 }}>
                About Embedding Service
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                The embedding service provides text embeddings for the RAG (Retrieval Augmented Generation) system.
                It runs as a separate llama.cpp server optimized for embedding generation.
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                After starting the service, it will be available at the endpoint shown above.
                The service uses the same Docker infrastructure as the main LLM service.
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Configure the RAG Integration section above to connect your RAG system to this deployed embedding service
                for faster and more efficient embedding generation during document processing and retrieval.
              </Typography>
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  )
}

export default EmbeddingDeployPage
