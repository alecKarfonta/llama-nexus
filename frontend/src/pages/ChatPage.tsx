import React, { useState, useRef, useEffect, ChangeEvent, KeyboardEvent } from 'react'
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Paper,
  Avatar,
  Stack,
  Chip,
  IconButton,
  Tooltip,
  FormControl,
  Slider,
  Grid,
  Collapse,
  Divider,
  CircularProgress,
} from '@mui/material'
import {
  Send as SendIcon,
  Person as UserIcon,
  SmartToy as BotIcon,
  Settings as SettingsIcon,
  Clear as ClearIcon,
  ContentCopy as CopyIcon,
  Build as ToolIcon,
  CheckBox as CheckBoxIcon,
  CheckBoxOutlineBlank as CheckBoxOutlineBlankIcon,
} from '@mui/icons-material'
import { apiService } from '@/services/api'
import { ToolsService } from '@/services/tools'
import type { ChatMessage, ChatCompletionRequest, Tool, ToolCall } from '@/types/api'

interface ChatPageProps {}

interface ChatSettings {
  // Connection settings
  baseUrl: string
  endpoint: string
  apiKey: string
  // Model parameters
  temperature: number
  topP: number
  topK: number
  maxTokens: number
  streamResponse: boolean
  enableTools: boolean
  selectedTools: string[]
}

const defaultSettings: ChatSettings = {
  // Connection settings
  baseUrl: '', // Empty means use current domain
  endpoint: '/v1/chat/completions',
  apiKey: '',
  // Model parameters
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  maxTokens: 2048,
  streamResponse: true,
  enableTools: false,
  selectedTools: [],
}

export const ChatPage: React.FC<ChatPageProps> = () => {
  // Load settings from localStorage or use defaults
  const loadSettings = (): ChatSettings => {
    try {
      const saved = localStorage.getItem('chat-settings');
      if (saved) {
        const parsed = JSON.parse(saved);
        return { ...defaultSettings, ...parsed };
      }
    } catch (error) {
      console.warn('Failed to load chat settings from localStorage:', error);
    }
    return defaultSettings;
  };

  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your AI assistant powered by Qwen3-Coder. How can I help you today? I have access to various tools including weather lookup, calculator, code execution, and system information.'
    }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [settings, setSettings] = useState(loadSettings)
  const [showSettings, setShowSettings] = useState(false)
  const [error, setError] = useState(null)
  const [availableTools] = useState(ToolsService.getAvailableTools())
  
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  // Save settings to localStorage whenever they change
  const saveSettings = (newSettings: ChatSettings) => {
    try {
      localStorage.setItem('chat-settings', JSON.stringify(newSettings));
      setSettings(newSettings);
    } catch (error) {
      console.warn('Failed to save chat settings to localStorage:', error);
      setSettings(newSettings); // Still update state even if save fails
    }
  };

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: ChatMessage = {
      role: 'user',
      content: input.trim()
    }

    setMessages((prev: ChatMessage[]) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setError(null)

    try {
      const selectedToolsForRequest = settings.enableTools 
        ? availableTools.filter((tool: Tool) => settings.selectedTools.includes(tool.function.name))
        : []

      const request: ChatCompletionRequest = {
        messages: [...messages, userMessage],
        temperature: settings.temperature,
        top_p: settings.topP,
        top_k: settings.topK,
        max_tokens: settings.maxTokens,
        stream: settings.streamResponse,
        tools: selectedToolsForRequest.length > 0 ? selectedToolsForRequest : undefined,
        tool_choice: selectedToolsForRequest.length > 0 ? 'auto' : undefined,
      }

      if (settings.streamResponse) {
        try {
          await handleStreamingResponse(request)
        } catch (streamError) {
          console.warn('Streaming failed, falling back to non-streaming:', streamError)
          // Fallback to non-streaming if streaming fails
          const nonStreamRequest = { ...request, stream: false }
          await handleNonStreamingResponse(nonStreamRequest)
        }
      } else {
        await handleNonStreamingResponse(request)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message')
      console.error('Chat error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  // Custom API call function that uses chat-specific settings
  const createChatCompletionStream = async (request: ChatCompletionRequest): Promise<ReadableStream<any>> => {
    const authHeaders = settings.apiKey ? { 'Authorization': `Bearer ${settings.apiKey}` } : {};
    const fullUrl = settings.baseUrl ? `${settings.baseUrl.replace(/\/$/, '')}${settings.endpoint}` : settings.endpoint;
    const response = await fetch(fullUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        'Cache-Control': 'no-cache',
        ...authHeaders,
      },
      body: JSON.stringify({ ...request, stream: true })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    return new ReadableStream({
      start(controller) {
        function pump(): Promise<void> {
          return reader.read().then(({ done, value }) => {
            if (done) {
              controller.close();
              return;
            }
            
            // Parse SSE data
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');
            
            for (const line of lines) {
              const trimmedLine = line.trim();
              if (trimmedLine.startsWith('data: ')) {
                const data = trimmedLine.slice(6);
                if (data === '[DONE]') {
                  controller.close();
                  return;
                }
                try {
                  const parsed = JSON.parse(data);
                  controller.enqueue(parsed);
                } catch (e) {
                  // Ignore malformed JSON
                  console.warn('Failed to parse SSE data:', data);
                }
              }
            }
            
            return pump();
          }).catch((error) => {
            controller.error(error);
          });
        }
        return pump();
      }
    });
  };

  const createChatCompletion = async (request: ChatCompletionRequest): Promise<any> => {
    const authHeaders = settings.apiKey ? { 'Authorization': `Bearer ${settings.apiKey}` } : {};
    const fullUrl = settings.baseUrl ? `${settings.baseUrl.replace(/\/$/, '')}${settings.endpoint}` : settings.endpoint;
    const response = await fetch(fullUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders,
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  };

  const handleStreamingResponse = async (request: ChatCompletionRequest) => {
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: ''
    }
    setMessages((prev: ChatMessage[]) => [...prev, assistantMessage])

    try {
      const stream = await createChatCompletionStream(request)
      const reader = stream.getReader()

      try {
        let accumulatedToolCalls: ToolCall[] = []
        
        while (true) {
          const { done, value } = await reader.read()
          if (done) {
            console.log('Stream ended naturally')
            break
          }

          console.log('Received stream chunk:', value)

          if (value.choices?.[0]?.delta?.content) {
            setMessages((prev: ChatMessage[]) => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage.role === 'assistant') {
                lastMessage.content += value.choices[0].delta.content
              }
              return newMessages
            })
          }

          if (value.choices?.[0]?.delta?.tool_calls) {
            accumulatedToolCalls.push(...value.choices[0].delta.tool_calls)
          }

          if (value.choices?.[0]?.finish_reason === 'tool_calls') {
            await handleToolCalls(accumulatedToolCalls)
          }
        }
      } finally {
        reader.releaseLock()
      }
    } catch (streamError) {
      console.error('Streaming error:', streamError)
      throw streamError // Re-throw to trigger fallback
    }
  }

  const handleNonStreamingResponse = async (request: ChatCompletionRequest) => {
    const response = await createChatCompletion(request)
    const assistantMessage = response.choices[0].message

    setMessages((prev: ChatMessage[]) => [...prev, assistantMessage])

    // Handle tool calls if present
    if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
      await handleToolCalls(assistantMessage.tool_calls)
    }
  }

  const handleToolCalls = async (toolCalls: ToolCall[]) => {
    for (const toolCall of toolCalls) {
      try {
        // Show tool execution indicator
        const toolMessage: ChatMessage = {
          role: 'assistant',
          content: `🔧 Executing tool: ${toolCall.function.name}...`
        }
        setMessages((prev: ChatMessage[]) => [...prev, toolMessage])

        // Execute the tool
        const result = await ToolsService.executeToolCall(toolCall)

        // Add tool result as a tool message
        const toolResultMessage: ChatMessage = {
          role: 'tool',
          content: result,
          tool_call_id: toolCall.id
        }

        // Update the tool execution message with result
        setMessages((prev: ChatMessage[]) => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage.content?.includes('🔧 Executing tool:')) {
            lastMessage.content = `✅ Tool executed: ${toolCall.function.name}\n\nResult:\n${result}`
          }
          return [...newMessages, toolResultMessage]
        })

        // Continue conversation with tool result
        const followUpRequest: ChatCompletionRequest = {
          messages: [...messages, toolResultMessage],
          temperature: settings.temperature,
          top_p: settings.topP,
          top_k: settings.topK,
          max_tokens: settings.maxTokens,
          stream: false, // Use non-streaming for tool follow-ups
        }

        const followUpResponse = await createChatCompletion(followUpRequest)
        setMessages((prev: ChatMessage[]) => [...prev, followUpResponse.choices[0].message])

      } catch (error) {
        const errorMessage: ChatMessage = {
          role: 'assistant',
          content: `❌ Error executing tool ${toolCall.function.name}: ${error instanceof Error ? error.message : 'Unknown error'}`
        }
        setMessages((prev: ChatMessage[]) => [...prev, errorMessage])
      }
    }
  }

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: 'Hello! I\'m your AI assistant powered by Qwen3-Coder. How can I help you today? I have access to various tools including weather lookup, calculator, code execution, and system information.'
      }
    ])
    setError(null)
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const formatRole = (role: string) => {
    switch (role) {
      case 'user': return 'You'
      case 'assistant': return 'Assistant'
      case 'tool': return 'Tool Result'
      case 'system': return 'System'
      default: return role
    }
  }

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'user': return <UserIcon />
      case 'assistant': return <BotIcon />
      case 'tool': return <ToolIcon />
      default: return <BotIcon />
    }
  }

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'user': return 'primary'
      case 'assistant': return 'secondary'
      case 'tool': return 'info'
      default: return 'secondary'
    }
  }

  const toggleToolSelection = (toolName: string) => {
    const newSelectedTools = settings.selectedTools.includes(toolName)
      ? settings.selectedTools.filter((name: string) => name !== toolName)
      : [...settings.selectedTools, toolName];
    
    saveSettings({ ...settings, selectedTools: newSelectedTools });
  }

  return (
    <Box sx={{ 
      height: 'calc(100vh - 120px)', 
      display: 'flex', 
      flexDirection: 'column',
      p: { xs: 2, sm: 3, md: 4 },
      maxWidth: '100%',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography 
            variant="h1" 
            sx={{ 
              fontWeight: 700, 
              color: 'text.primary',
              mb: 0.5,
              fontSize: { xs: '1.25rem', sm: '1.5rem' },
              lineHeight: 1
            }}
          >
            Chat with AI Model
          </Typography>
          <Typography 
            variant="body2" 
            color="text.secondary" 
            sx={{ 
              fontSize: '0.8125rem',
              mb: { xs: 1, sm: 2 }
            }}
          >
            Interact with your AI assistant powered by Qwen3-Coder
          </Typography>
        </Box>
        <Box>
          <Tooltip title="Chat Settings">
            <IconButton 
              onClick={() => setShowSettings(!showSettings)}
              size="small"
              sx={{
                bgcolor: showSettings ? 'action.selected' : 'transparent',
                '&:hover': { bgcolor: 'action.hover' }
              }}
            >
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Clear Chat">
            <IconButton 
              onClick={clearChat}
              size="small"
              sx={{
                '&:hover': { bgcolor: 'action.hover' }
              }}
            >
              <ClearIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Settings Panel */}
      <Collapse in={showSettings}>
        <Card sx={{ mb: 2, borderRadius: 1, boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3)' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Chat Settings
            </Typography>
            
            {/* Connection Settings */}
            <Typography variant="h6" gutterBottom sx={{ mt: 2, mb: 1 }}>
              Connection Settings
            </Typography>
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Base URL"
                  fullWidth
                  value={settings.baseUrl}
                  onChange={(e) => saveSettings({ ...settings, baseUrl: e.target.value })}
                  placeholder="https://api.openai.com (leave empty for current domain)"
                  helperText="Base URL for the API service (optional)"
                  sx={{ 
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    }
                  }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Endpoint Path"
                  fullWidth
                  value={settings.endpoint}
                  onChange={(e) => saveSettings({ ...settings, endpoint: e.target.value })}
                  placeholder="/v1/chat/completions"
                  helperText="API endpoint path"
                  sx={{ 
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    }
                  }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="API Key"
                  type="password"
                  fullWidth
                  value={settings.apiKey}
                  onChange={(e) => saveSettings({ ...settings, apiKey: e.target.value })}
                  placeholder="Enter your API key (optional)"
                  helperText="API key for authentication (leave empty if not required)"
                  sx={{ 
                    '& .MuiOutlinedInput-root': {
                      fontSize: '0.875rem',
                      borderRadius: 1,
                      backgroundColor: 'background.default',
                      '&.Mui-focused': {
                        borderColor: 'primary.main'
                      }
                    }
                  }}
                />
              </Grid>
              
              {/* Quick Presets */}
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Quick Presets:
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => saveSettings({ ...settings, baseUrl: '', endpoint: '/v1/chat/completions' })}
                    sx={{ fontSize: '0.75rem', textTransform: 'none' }}
                  >
                    Local (Current Domain)
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => saveSettings({ ...settings, baseUrl: 'http://localhost:8600', endpoint: '/v1/chat/completions' })}
                    sx={{ fontSize: '0.75rem', textTransform: 'none' }}
                  >
                    Local LlamaCPP (8600)
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => saveSettings({ ...settings, baseUrl: 'https://api.openai.com', endpoint: '/v1/chat/completions' })}
                    sx={{ fontSize: '0.75rem', textTransform: 'none' }}
                  >
                    OpenAI API
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => saveSettings({ ...settings, baseUrl: 'http://localhost:11434', endpoint: '/v1/chat/completions' })}
                    sx={{ fontSize: '0.75rem', textTransform: 'none' }}
                  >
                    Ollama (11434)
                  </Button>
                </Box>
              </Grid>

              {/* URL Preview */}
              <Grid item xs={12}>
                <Box sx={{ 
                  p: 2, 
                  backgroundColor: 'rgba(0, 0, 0, 0.1)', 
                  borderRadius: 1, 
                  border: '1px solid rgba(255, 255, 255, 0.1)' 
                }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Full URL Preview:
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', color: 'primary.main' }}>
                    {settings.baseUrl ? `${settings.baseUrl.replace(/\/$/, '')}${settings.endpoint}` : `${window.location.origin}${settings.endpoint}`}
                  </Typography>
                </Box>
              </Grid>
            </Grid>

            {/* Model Parameters */}
            <Typography variant="h6" gutterBottom sx={{ mt: 2, mb: 1 }}>
              Model Parameters
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={3}>
                <Typography gutterBottom>Temperature: {settings.temperature}</Typography>
                <Slider
                  value={settings.temperature}
                  onChange={(_: Event, value: number | number[]) => saveSettings({ ...settings, temperature: value as number })}
                  min={0}
                  max={2}
                  step={0.1}
                  marks={[
                    { value: 0, label: '0' },
                    { value: 1, label: '1' },
                    { value: 2, label: '2' },
                  ]}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography gutterBottom>Top P: {settings.topP}</Typography>
                <Slider
                  value={settings.topP}
                  onChange={(_: Event, value: number | number[]) => saveSettings({ ...settings, topP: value as number })}
                  min={0}
                  max={1}
                  step={0.05}
                  marks={[
                    { value: 0, label: '0' },
                    { value: 0.5, label: '0.5' },
                    { value: 1, label: '1' },
                  ]}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography gutterBottom>Top K: {settings.topK}</Typography>
                <Slider
                  value={settings.topK}
                  onChange={(_: Event, value: number | number[]) => saveSettings({ ...settings, topK: value as number })}
                  min={1}
                  max={100}
                  step={1}
                  marks={[
                    { value: 1, label: '1' },
                    { value: 50, label: '50' },
                    { value: 100, label: '100' },
                  ]}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography gutterBottom>Max Tokens: {settings.maxTokens}</Typography>
                <Slider
                  value={settings.maxTokens}
                  onChange={(_: Event, value: number | number[]) => saveSettings({ ...settings, maxTokens: value as number })}
                  min={100}
                  max={4096}
                  step={100}
                  marks={[
                    { value: 100, label: '100' },
                    { value: 2048, label: '2K' },
                    { value: 4096, label: '4K' },
                  ]}
                />
              </Grid>
            </Grid>

            <Divider sx={{ my: 3 }} />

            {/* Tools Section */}
            <Typography variant="h6" gutterBottom>
              Function Calling Tools
            </Typography>
            <Box sx={{ mb: 2 }}>
              <FormControl component="fieldset">
                <Button
                  variant={settings.enableTools ? 'contained' : 'outlined'}
                  startIcon={settings.enableTools ? <CheckBoxIcon /> : <CheckBoxOutlineBlankIcon />}
                  onClick={() => saveSettings({ ...settings, enableTools: !settings.enableTools })}
                  sx={{ mb: 2 }}
                >
                  Enable Tool Calling
                </Button>
              </FormControl>
            </Box>

            {settings.enableTools && (
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>
                    Available Tools (select which tools the model can use):
                  </Typography>
                </Grid>
                {availableTools.map((tool: Tool) => (
                  <Grid item xs={12} sm={6} md={4} key={tool.function.name}>
                    <Card 
                      variant="outlined" 
                      sx={{ 
                        cursor: 'pointer',
                        bgcolor: settings.selectedTools.includes(tool.function.name) ? 'action.selected' : 'background.paper',
                        '&:hover': { bgcolor: 'action.hover' }
                      }}
                      onClick={() => toggleToolSelection(tool.function.name)}
                    >
                      <CardContent sx={{ py: 1.5 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <IconButton size="small" sx={{ mr: 1 }}>
                            {settings.selectedTools.includes(tool.function.name) ? 
                              <CheckBoxIcon color="primary" /> : 
                              <CheckBoxOutlineBlankIcon />
                            }
                          </IconButton>
                          <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                            {tool.function.name}
                          </Typography>
                        </Box>
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.8rem' }}>
                          {tool.function.description}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Selected tools: {settings.selectedTools.length} / {availableTools.length}
                  </Typography>
                </Grid>
              </Grid>
            )}
          </CardContent>
        </Card>
      </Collapse>

      {/* Error Display */}
      {error && (
        <Paper sx={{ 
          p: 2, 
          mb: 2, 
          bgcolor: 'error.dark', 
          color: 'error.contrastText',
          borderRadius: 1,
          border: '1px solid',
          borderColor: 'error.main'
        }}>
          <Typography variant="body2" sx={{ fontSize: '0.8125rem' }}>
            Error: {error}
          </Typography>
        </Paper>
      )}

      {/* Messages */}
      <Box sx={{ 
        flexGrow: 1, 
        overflow: 'auto', 
        mb: 2,
        borderRadius: 1,
        bgcolor: 'background.paper',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        p: 1
      }}>
        <Stack spacing={2}>
          {messages.map((message: ChatMessage, index: number) => (
            <Box
              key={index}
              sx={{
                display: 'flex',
                justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
              }}
            >
                              <Paper
                sx={{
                  p: 2,
                  maxWidth: '80%',
                  bgcolor: message.role === 'user' ? 'primary.dark' : 'background.paper',
                  color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
                  border: '1px solid',
                  borderColor: message.role === 'user' ? 'primary.main' : 'rgba(255, 255, 255, 0.1)',
                  borderRadius: 1
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Avatar
                    sx={{
                      width: 24,
                      height: 24,
                      mr: 1,
                      bgcolor: getRoleColor(message.role) + '.main',
                    }}
                  >
                    {getRoleIcon(message.role)}
                  </Avatar>
                  <Chip
                    label={formatRole(message.role)}
                    size="small"
                    color={getRoleColor(message.role) as 'primary' | 'secondary' | 'info'}
                  />
                  <Tooltip title="Copy to clipboard">
                    <IconButton
                      size="small"
                      sx={{ ml: 'auto' }}
                      onClick={() => copyToClipboard(message.content)}
                    >
                      <CopyIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
                <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                  {message.content}
                </Typography>
              </Paper>
            </Box>
          ))}
          {isLoading && (
            <Box sx={{ display: 'flex', justifyContent: 'flex-start' }}>
              <Paper sx={{ 
                p: 2, 
                bgcolor: 'background.paper',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: 1
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <CircularProgress size={20} sx={{ mr: 2 }} />
                  <Typography variant="body2" color="text.secondary">
                    Assistant is typing...
                  </Typography>
                </Box>
              </Paper>
            </Box>
          )}
          <div ref={messagesEndRef} />
        </Stack>
      </Box>

      {/* Input */}
      <Paper sx={{ 
        p: 2,
        borderRadius: 1,
        bgcolor: 'background.paper',
        border: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <TextField
            ref={inputRef}
            fullWidth
            multiline
            maxRows={4}
            value={input}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
            disabled={isLoading}
            sx={{ 
              flexGrow: 1,
              '& .MuiOutlinedInput-root': {
                fontSize: '0.875rem',
                borderRadius: 1,
                backgroundColor: 'background.default',
                '&.Mui-focused': {
                  borderColor: 'primary.main'
                }
              }
            }}
          />
          <Button
            variant="contained"
            onClick={handleSendMessage}
            disabled={!input.trim() || isLoading}
            sx={{ 
              minWidth: 100,
              borderRadius: 1,
              fontWeight: 500,
              fontSize: '0.8125rem',
              textTransform: 'none',
              boxShadow: 'none',
              '&:hover': {
                boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)'
              }
            }}
            startIcon={isLoading ? <CircularProgress size={16} /> : <SendIcon />}
          >
            {isLoading ? 'Sending' : 'Send'}
          </Button>
        </Box>
      </Paper>
    </Box>
  )
}
