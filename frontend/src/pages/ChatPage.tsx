import React, { useState, useRef, useEffect, useCallback, ChangeEvent, KeyboardEvent } from 'react'
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
  Badge,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
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
  History as HistoryIcon,
  Save as SaveIcon,
  Add as AddIcon,
  Speed as SpeedIcon,
  Timer as TimerIcon,
  AttachFile as AttachFileIcon,
  Image as ImageIcon,
  Close as CloseIcon,
  Mic as MicIcon,
  Stop as StopIcon,
} from '@mui/icons-material'
import { apiService } from '@/services/api'
import { ToolsService } from '@/services/tools'
import { MarkdownRenderer } from '@/components/chat'
import { ConversationSidebar } from '@/components/chat/ConversationSidebar'
import type { ChatMessage, ChatCompletionRequest, Tool, ToolCall, ConversationListItem } from '@/types/api'

interface ChatPageProps {}

interface ChatSettings {
  // Connection settings
  baseUrl: string
  endpoint: string
  apiKey: string
  openaiApiKey: string // For audio transcription
  // Model parameters
  temperature: number
  topP: number
  topK: number
  maxTokens: number
  streamResponse: boolean
  enableTools: boolean
  selectedTools: string[]
  // Display settings
  showPerformanceMetrics: boolean
}

interface PerformanceMetrics {
  startTime: number | null
  firstTokenTime: number | null
  endTime: number | null
  tokensGenerated: number
  tokensPerSecond: number | null
  promptTokens: number | null
  timeToFirstToken: number | null
}

const defaultSettings: ChatSettings = {
  // Connection settings
  baseUrl: '', // Empty means use current domain
  endpoint: '/v1/chat/completions',
  apiKey: '',
  openaiApiKey: '', // For audio transcription
  // Model parameters
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  maxTokens: 2048,
  streamResponse: true,
  enableTools: false,
  selectedTools: [],
  // Display settings
  showPerformanceMetrics: false,
}

const initialPerformanceMetrics: PerformanceMetrics = {
  startTime: null,
  firstTokenTime: null,
  endTime: null,
  tokensGenerated: 0,
  tokensPerSecond: null,
  promptTokens: null,
  timeToFirstToken: null,
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

  const [currentModelName, setCurrentModelName] = useState<string>('AI Model')
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your AI assistant. How can I help you today? I have access to various tools including weather lookup, calculator, code execution, and system information.'
    }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [settings, setSettings] = useState(loadSettings)
  const [showSettings, setShowSettings] = useState(false)
  const [error, setError] = useState(null)
  const [availableTools] = useState(ToolsService.getAvailableTools())
  
  // Conversation management state
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)
  const [conversationTitle, setConversationTitle] = useState<string>('New Conversation')
  const [isSaving, setIsSaving] = useState(false)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  
  // Context window tracking state
  const [tokenCount, setTokenCount] = useState({ prompt: 0, total: 0 })
  const maxContextTokens = 128000 // Default, can be made configurable
  
  // Performance metrics tracking
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>(initialPerformanceMetrics)
  const performanceRef = useRef<PerformanceMetrics>(initialPerformanceMetrics)
  
  // Multi-modal inputs state
  const [uploadedImages, setUploadedImages] = useState<Array<{ file: File; preview: string; base64: string }>>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Audio recording state
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [audioError, setAudioError] = useState<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const [showAudioDialog, setShowAudioDialog] = useState(false)
  
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

  // Estimate token count (rough approximation: ~4 chars per token)
  const estimateTokenCount = useCallback((msgs: ChatMessage[]) => {
    let totalChars = 0;
    msgs.forEach(msg => {
      // Handle string content
      if (typeof msg.content === 'string') {
        totalChars += msg.content.length;
      } 
      // Handle array content (multi-modal)
      else if (Array.isArray(msg.content)) {
        msg.content.forEach((part: any) => {
          if (part.type === 'text' && part.text) {
            totalChars += part.text.length;
          }
          // Approximate image tokens (images typically use ~85-170 tokens in vision models)
          if (part.type === 'image_url') {
            totalChars += 100 * 4; // Estimate 100 tokens per image
          }
        });
      }
      if (msg.reasoning_content) {
        totalChars += msg.reasoning_content.length;
      }
    });
    return Math.ceil(totalChars / 4);
  }, []);

  // Save conversation to backend
  const saveConversation = useCallback(async (forceNew: boolean = false) => {
    if (messages.length <= 1) return; // Don't save empty conversations
    
    setIsSaving(true);
    try {
      const conversationData = {
        title: conversationTitle || 'New Conversation',
        messages: messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          reasoning_content: msg.reasoning_content,
          tool_calls: msg.tool_calls,
          tool_call_id: msg.tool_call_id,
        })),
        model: currentModelName,
        settings: {
          temperature: settings.temperature,
          topP: settings.topP,
          topK: settings.topK,
          maxTokens: settings.maxTokens,
        },
      };

      if (currentConversationId && !forceNew) {
        // Update existing conversation
        await apiService.updateConversation(currentConversationId, conversationData);
      } else {
        // Create new conversation
        // Auto-generate title from first user message
        const firstUserMsg = messages.find(m => m.role === 'user');
        if (firstUserMsg && conversationTitle === 'New Conversation') {
          let titleText = '';
          if (typeof firstUserMsg.content === 'string') {
            titleText = firstUserMsg.content;
          } else if (Array.isArray(firstUserMsg.content)) {
            // Extract text from multi-modal content
            const textParts = firstUserMsg.content
              .filter((part: any) => part.type === 'text' && part.text)
              .map((part: any) => part.text);
            titleText = textParts.join(' ');
          }
          conversationData.title = titleText.slice(0, 50) + (titleText.length > 50 ? '...' : '');
          setConversationTitle(conversationData.title);
        }
        
        const newConversation = await apiService.createConversation(conversationData);
        setCurrentConversationId(newConversation.id);
      }
      setHasUnsavedChanges(false);
    } catch (err) {
      console.error('Failed to save conversation:', err);
      setError('Failed to save conversation');
    } finally {
      setIsSaving(false);
    }
  }, [messages, conversationTitle, currentModelName, settings, currentConversationId]);

  // Load a conversation from the sidebar
  const handleSelectConversation = useCallback(async (conversation: ConversationListItem) => {
    try {
      const fullConversation = await apiService.getConversation(conversation.id);
      setCurrentConversationId(fullConversation.id);
      setConversationTitle(fullConversation.title);
      setMessages(fullConversation.messages.map((msg: any) => ({
        role: msg.role,
        content: msg.content,
        reasoning_content: msg.reasoning_content,
        tool_calls: msg.tool_calls,
        tool_call_id: msg.tool_call_id,
      })));
      setHasUnsavedChanges(false);
      setError(null);
    } catch (err) {
      console.error('Failed to load conversation:', err);
      setError('Failed to load conversation');
    }
  }, []);

  // Start a new conversation
  const handleNewConversation = useCallback(() => {
    setCurrentConversationId(null);
    setConversationTitle('New Conversation');
    setMessages([
      {
        role: 'assistant',
        content: `Hello! I'm your AI assistant powered by ${currentModelName}. How can I help you today? I have access to various tools including weather lookup, calculator, code execution, and system information.`
      }
    ]);
    setHasUnsavedChanges(false);
    setError(null);
  }, [currentModelName]);

  // Update token count when messages change
  useEffect(() => {
    const count = estimateTokenCount(messages);
    setTokenCount({ prompt: count, total: count });
    
    // Mark as having unsaved changes when messages change (except initial load)
    if (messages.length > 1) {
      setHasUnsavedChanges(true);
    }
  }, [messages, estimateTokenCount]);

  // Cleanup image previews on unmount
  useEffect(() => {
    return () => {
      uploadedImages.forEach(img => URL.revokeObjectURL(img.preview))
    }
  }, [uploadedImages]);

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Fetch current model info on mount
  useEffect(() => {
    const fetchCurrentModel = async () => {
      try {
        const modelInfo = await apiService.getCurrentModel()
        if (modelInfo && modelInfo.name) {
          setCurrentModelName(modelInfo.name)
          // Update the initial greeting message with the model name
          setMessages([
            {
              role: 'assistant',
              content: `Hello! I'm your AI assistant powered by ${modelInfo.name}. How can I help you today? I have access to various tools including weather lookup, calculator, code execution, and system information.`
            }
          ])
        }
      } catch (error) {
        console.warn('Failed to fetch current model:', error)
        // Keep default generic message if model fetch fails
      }
    }
    fetchCurrentModel()
  }, [])

  const handleSendMessage = async () => {
    if ((!input.trim() && uploadedImages.length === 0) || isLoading) return

    // Construct message content with images if present
    let messageContent: any = input.trim()
    
    // If images are uploaded, format the content for multi-modal models (OpenAI format)
    if (uploadedImages.length > 0) {
      // OpenAI multi-modal format: content is an array of parts
      const contentParts: any[] = []
      
      // Add text if present
      if (input.trim()) {
        contentParts.push({
          type: 'text',
          text: input.trim()
        })
      }
      
      // Add images
      uploadedImages.forEach(img => {
        contentParts.push({
          type: 'image_url',
          image_url: {
            url: img.base64
          }
        })
      })
      
      messageContent = contentParts
    }

    const userMessage: ChatMessage = {
      role: 'user',
      content: messageContent
    }

    setMessages((prev: ChatMessage[]) => [...prev, userMessage])
    setInput('')
    
    // Clear uploaded images after sending
    uploadedImages.forEach(img => URL.revokeObjectURL(img.preview))
    setUploadedImages([])
    
    setIsLoading(true)
    setError(null)
    
    // Reset performance metrics for new generation
    performanceRef.current = initialPerformanceMetrics
    setPerformanceMetrics(initialPerformanceMetrics)

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
      content: '',
      reasoning_content: '',
      tokensPerSecond: undefined
    }
    setMessages((prev: ChatMessage[]) => [...prev, assistantMessage])

    // Initialize performance tracking
    const startTime = performance.now()
    performanceRef.current = {
      ...initialPerformanceMetrics,
      startTime,
    }
    setPerformanceMetrics(performanceRef.current)

    try {
      const stream = await createChatCompletionStream(request)
      const reader = stream.getReader()

      try {
        let accumulatedToolCalls: ToolCall[] = []
        let tokenCount = 0
        let firstTokenReceived = false
        
        while (true) {
          const { done, value } = await reader.read()
          if (done) {
            console.log('Stream ended naturally')
            // Finalize performance metrics
            const endTime = performance.now()
            const totalTime = (endTime - startTime) / 1000
            performanceRef.current = {
              ...performanceRef.current,
              endTime,
              tokensPerSecond: tokenCount > 0 ? tokenCount / totalTime : null,
            }
            setPerformanceMetrics({ ...performanceRef.current })
            break
          }

          console.log('Received stream chunk:', value)

          // Check for content in multiple locations
          // content: actual response content
          // __verbose.content: llama.cpp verbose output
          const contentDelta = value.choices?.[0]?.delta?.content || 
                              value.__verbose?.content ||
                              ''
          
          if (contentDelta) {
            // Track first token time
            if (!firstTokenReceived) {
              firstTokenReceived = true
              const firstTokenTime = performance.now()
              performanceRef.current = {
                ...performanceRef.current,
                firstTokenTime,
                timeToFirstToken: firstTokenTime - startTime,
              }
              setPerformanceMetrics({ ...performanceRef.current })
            }
            
            // Count tokens (rough estimate: split by whitespace and punctuation)
            const newTokens = contentDelta.split(/[\s\n]+/).filter((t: string) => t.length > 0).length
            tokenCount += Math.max(1, newTokens) // At least 1 token per chunk
            
            // Update metrics periodically
            const currentTime = performance.now()
            const elapsedTime = (currentTime - startTime) / 1000
            performanceRef.current = {
              ...performanceRef.current,
              tokensGenerated: tokenCount,
              tokensPerSecond: tokenCount / elapsedTime,
            }
            setPerformanceMetrics({ ...performanceRef.current })
            
            setMessages((prev: ChatMessage[]) => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage.role === 'assistant') {
                lastMessage.content += contentDelta
              }
              return newMessages
            })
          }

          // Extract timing information if available (more accurate from server)
          const timings = value.timings || value.__verbose?.timings
          if (timings?.predicted_per_second) {
            performanceRef.current = {
              ...performanceRef.current,
              tokensPerSecond: timings.predicted_per_second,
            }
            setPerformanceMetrics({ ...performanceRef.current })
            
            setMessages((prev: ChatMessage[]) => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage.role === 'assistant') {
                lastMessage.tokensPerSecond = timings.predicted_per_second
              }
              return newMessages
            })
          }
          
          // Extract prompt processing info if available
          if (timings?.prompt_per_second) {
            performanceRef.current = {
              ...performanceRef.current,
              promptTokens: timings.prompt_n || performanceRef.current.promptTokens,
            }
            setPerformanceMetrics({ ...performanceRef.current })
          }

          // Handle reasoning/thinking content (for models like DeepSeek R1, QwQ)
          if (value.choices?.[0]?.delta?.reasoning_content) {
            setMessages((prev: ChatMessage[]) => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage.role === 'assistant') {
                lastMessage.reasoning_content = (lastMessage.reasoning_content || '') + value.choices[0].delta.reasoning_content
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
    const assistantMessage: ChatMessage = {
      ...response.choices[0].message,
      // Include reasoning_content if present in the response
      reasoning_content: response.choices[0].message.reasoning_content
    }

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
        content: `Hello! I'm your AI assistant powered by ${currentModelName}. How can I help you today? I have access to various tools including weather lookup, calculator, code execution, and system information.`
      }
    ])
    setError(null)
  }

  const copyToClipboard = (content: string | Array<{type: string; text?: string; image_url?: {url: string}}>) => {
    let text = '';
    if (typeof content === 'string') {
      text = content;
    } else if (Array.isArray(content)) {
      // Extract text parts from multi-modal content
      text = content
        .filter((part: any) => part.type === 'text' && part.text)
        .map((part: any) => part.text)
        .join('\n');
    }
    navigator.clipboard.writeText(text);
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

  // Image upload handlers
  const handleImageUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return

    const newImages: Array<{ file: File; preview: string; base64: string }> = []

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setError(`File ${file.name} is not an image`)
        continue
      }

      // Validate file size (max 20MB)
      if (file.size > 20 * 1024 * 1024) {
        setError(`File ${file.name} is too large (max 20MB)`)
        continue
      }

      // Create preview URL
      const preview = URL.createObjectURL(file)

      // Convert to base64
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = reject
        reader.readAsDataURL(file)
      })

      newImages.push({ file, preview, base64 })
    }

    setUploadedImages((prev) => [...prev, ...newImages])
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleRemoveImage = (index: number) => {
    setUploadedImages((prev) => {
      const updated = [...prev]
      // Revoke the preview URL to free memory
      URL.revokeObjectURL(updated[index].preview)
      updated.splice(index, 1)
      return updated
    })
  }

  // Audio recording handlers
  const startRecording = async () => {
    try {
      setAudioError(null)
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        await transcribeAudio(audioBlob)
        
        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
      setShowAudioDialog(true)
    } catch (err) {
      console.error('Failed to start recording:', err)
      setAudioError('Failed to access microphone. Please check permissions.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const cancelRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      audioChunksRef.current = []
    }
    setShowAudioDialog(false)
    setAudioError(null)
  }

  const transcribeAudio = async (audioBlob: Blob) => {
    setIsTranscribing(true)
    try {
      // Convert webm to a format OpenAI accepts (mp3, mp4, mpeg, mpga, m4a, wav, or webm)
      const formData = new FormData()
      formData.append('file', audioBlob, 'audio.webm')
      formData.append('model', 'whisper-1')

      // Get OpenAI API key from settings
      const apiKey = settings.openaiApiKey || localStorage.getItem('openai-api-key') || ''
      
      if (!apiKey) {
        throw new Error('OpenAI API key not configured. Please add it in Chat Settings.')
      }

      const response = await fetch('https://api.openai.com/v1/audio/transcriptions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`
        },
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error?.message || `Transcription failed: ${response.statusText}`)
      }

      const data = await response.json()
      const transcription = data.text

      // Append transcription to input
      setInput((prev) => prev ? `${prev} ${transcription}` : transcription)
      setShowAudioDialog(false)
      setAudioError(null)
    } catch (err) {
      console.error('Transcription error:', err)
      setAudioError(err instanceof Error ? err.message : 'Transcription failed')
    } finally {
      setIsTranscribing(false)
    }
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
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Tooltip title="Conversation History">
            <IconButton 
              onClick={() => setSidebarOpen(true)}
              size="small"
              sx={{
                bgcolor: sidebarOpen ? 'action.selected' : 'transparent',
                '&:hover': { bgcolor: 'action.hover' }
              }}
            >
              <Badge badgeContent={hasUnsavedChanges ? 1 : 0} color="warning" variant="dot">
                <HistoryIcon fontSize="small" />
              </Badge>
            </IconButton>
          </Tooltip>
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
              {conversationTitle}
            </Typography>
            <Typography 
              variant="body2" 
              color="text.secondary" 
              sx={{ 
                fontSize: '0.8125rem',
              }}
            >
              {currentModelName} | ~{tokenCount.prompt.toLocaleString()} tokens
            </Typography>
          </Box>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* Context window progress bar */}
          <Tooltip title={`Context: ${tokenCount.prompt.toLocaleString()} / ${maxContextTokens.toLocaleString()} tokens (${((tokenCount.prompt / maxContextTokens) * 100).toFixed(1)}%)`}>
            <Box sx={{ width: 100, mr: 1 }}>
              <LinearProgress 
                variant="determinate" 
                value={Math.min((tokenCount.prompt / maxContextTokens) * 100, 100)}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  bgcolor: 'action.hover',
                  '& .MuiLinearProgress-bar': {
                    bgcolor: tokenCount.prompt > maxContextTokens * 0.9 ? 'error.main' : 
                             tokenCount.prompt > maxContextTokens * 0.7 ? 'warning.main' : 'primary.main'
                  }
                }}
              />
            </Box>
          </Tooltip>
          <Tooltip title={hasUnsavedChanges ? "Save Conversation" : "Conversation Saved"}>
            <span>
              <IconButton 
                onClick={() => saveConversation()}
                size="small"
                disabled={isSaving || !hasUnsavedChanges}
                sx={{
                  '&:hover': { bgcolor: 'action.hover' }
                }}
              >
                {isSaving ? <CircularProgress size={16} /> : <SaveIcon fontSize="small" />}
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="New Conversation">
            <IconButton 
              onClick={handleNewConversation}
              size="small"
              sx={{
                '&:hover': { bgcolor: 'action.hover' }
              }}
            >
              <AddIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title={settings.showPerformanceMetrics ? "Hide Performance Metrics" : "Show Performance Metrics"}>
            <IconButton 
              onClick={() => saveSettings({ ...settings, showPerformanceMetrics: !settings.showPerformanceMetrics })}
              size="small"
              sx={{
                bgcolor: settings.showPerformanceMetrics ? 'action.selected' : 'transparent',
                '&:hover': { bgcolor: 'action.hover' }
              }}
            >
              <SpeedIcon fontSize="small" />
            </IconButton>
          </Tooltip>
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

      {/* Performance Metrics Display */}
      <Collapse in={settings.showPerformanceMetrics}>
        <Paper 
          sx={{ 
            mb: 2, 
            p: 1.5, 
            borderRadius: 1, 
            bgcolor: 'background.paper',
            border: '1px solid rgba(255, 255, 255, 0.1)'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, flexWrap: 'wrap' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <SpeedIcon fontSize="small" color="primary" />
              <Typography variant="body2" color="text.secondary">
                Performance Metrics
              </Typography>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                Generation:
              </Typography>
              <Chip 
                size="small" 
                label={performanceMetrics.tokensPerSecond 
                  ? `${performanceMetrics.tokensPerSecond.toFixed(1)} tok/s` 
                  : '--'
                }
                color={performanceMetrics.tokensPerSecond && performanceMetrics.tokensPerSecond > 20 ? 'success' : 'default'}
                sx={{ height: 20, fontSize: '0.7rem' }}
              />
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <TimerIcon fontSize="small" sx={{ color: 'text.secondary', width: 16, height: 16 }} />
              <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                TTFT:
              </Typography>
              <Chip 
                size="small" 
                label={performanceMetrics.timeToFirstToken 
                  ? `${performanceMetrics.timeToFirstToken.toFixed(0)} ms` 
                  : '--'
                }
                color={performanceMetrics.timeToFirstToken && performanceMetrics.timeToFirstToken < 500 ? 'success' : 'default'}
                sx={{ height: 20, fontSize: '0.7rem' }}
              />
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                Tokens:
              </Typography>
              <Chip 
                size="small" 
                label={performanceMetrics.tokensGenerated > 0 
                  ? performanceMetrics.tokensGenerated.toString() 
                  : '--'
                }
                variant="outlined"
                sx={{ height: 20, fontSize: '0.7rem' }}
              />
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                Total Time:
              </Typography>
              <Chip 
                size="small" 
                label={performanceMetrics.startTime && performanceMetrics.endTime
                  ? `${((performanceMetrics.endTime - performanceMetrics.startTime) / 1000).toFixed(2)}s`
                  : performanceMetrics.startTime && !performanceMetrics.endTime
                    ? 'Running...'
                    : '--'
                }
                variant="outlined"
                sx={{ height: 20, fontSize: '0.7rem' }}
              />
            </Box>
            
            {isLoading && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, ml: 'auto' }}>
                <CircularProgress size={14} />
                <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                  Generating...
                </Typography>
              </Box>
            )}
          </Box>
        </Paper>
      </Collapse>

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
              <Grid item xs={12} sm={6}>
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
              <Grid item xs={12} sm={6}>
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
              <Grid item xs={12} sm={6}>
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
              <Grid item xs={12} sm={6}>
                <TextField
                  label="OpenAI API Key (for voice)"
                  type="password"
                  fullWidth
                  value={settings.openaiApiKey}
                  onChange={(e) => saveSettings({ ...settings, openaiApiKey: e.target.value })}
                  placeholder="Enter OpenAI API key for transcription"
                  helperText="Required for voice input transcription"
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
                  {message.tokensPerSecond && message.role === 'assistant' && (
                    <Chip
                      label={`${message.tokensPerSecond.toFixed(2)} tok/s`}
                      size="small"
                      variant="outlined"
                      sx={{ ml: 1, fontSize: '0.7rem', height: 20 }}
                    />
                  )}
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
                {/* Check if message contains multi-modal content (array format) */}
                {(() => {
                  // Check if content is an array (multi-modal format)
                  if (Array.isArray(message.content)) {
                    const textParts: string[] = []
                    const imageParts: any[] = []
                    
                    // Separate text and images
                    message.content.forEach((part: any) => {
                      if (part.type === 'text') {
                        textParts.push(part.text)
                      } else if (part.type === 'image_url') {
                        imageParts.push(part.image_url.url)
                      }
                    })
                    
                    return (
                      <Box>
                        {/* Render images */}
                        {imageParts.length > 0 && (
                          <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                            {imageParts.map((imageUrl: string, idx: number) => (
                              <Box
                                key={idx}
                                sx={{
                                  maxWidth: 200,
                                  borderRadius: 1,
                                  overflow: 'hidden',
                                  border: '1px solid rgba(255, 255, 255, 0.2)',
                                }}
                              >
                                <img
                                  src={imageUrl}
                                  alt={`Attachment ${idx + 1}`}
                                  style={{
                                    width: '100%',
                                    height: 'auto',
                                    display: 'block',
                                  }}
                                />
                              </Box>
                            ))}
                          </Box>
                        )}
                        {/* Render text */}
                        {textParts.length > 0 && (
                          <MarkdownRenderer 
                            content={textParts.join('\n')} 
                            reasoning_content={message.reasoning_content}
                          />
                        )}
                      </Box>
                    )
                  }
                  
                  // String content - render normally
                  return (
                    <MarkdownRenderer 
                      content={message.content} 
                      reasoning_content={message.reasoning_content}
                    />
                  )
                })()}
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
        {/* Image Preview Section */}
        {uploadedImages.length > 0 && (
          <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {uploadedImages.map((image, index) => (
              <Box
                key={index}
                sx={{
                  position: 'relative',
                  width: 100,
                  height: 100,
                  borderRadius: 1,
                  overflow: 'hidden',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                }}
              >
                <img
                  src={image.preview}
                  alt={`Upload ${index + 1}`}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                  }}
                />
                <IconButton
                  size="small"
                  onClick={() => handleRemoveImage(index)}
                  sx={{
                    position: 'absolute',
                    top: 2,
                    right: 2,
                    bgcolor: 'rgba(0, 0, 0, 0.6)',
                    color: 'white',
                    '&:hover': {
                      bgcolor: 'rgba(0, 0, 0, 0.8)',
                    },
                    padding: '2px',
                  }}
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              </Box>
            ))}
          </Box>
        )}

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            style={{ display: 'none' }}
            onChange={handleImageUpload}
          />
          
          {/* Attachment button */}
          <Tooltip title="Attach images">
            <IconButton
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              size="small"
              sx={{
                color: 'text.secondary',
                '&:hover': { color: 'primary.main' }
              }}
            >
              <AttachFileIcon />
            </IconButton>
          </Tooltip>

          {/* Microphone button */}
          <Tooltip title={isRecording ? 'Recording...' : 'Voice input'}>
            <IconButton
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isLoading || isTranscribing}
              size="small"
              sx={{
                color: isRecording ? 'error.main' : 'text.secondary',
                '&:hover': { color: isRecording ? 'error.dark' : 'primary.main' },
                animation: isRecording ? 'pulse 1.5s ease-in-out infinite' : 'none',
                '@keyframes pulse': {
                  '0%, 100%': { opacity: 1 },
                  '50%': { opacity: 0.5 },
                },
              }}
            >
              {isRecording ? <StopIcon /> : <MicIcon />}
            </IconButton>
          </Tooltip>

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
            disabled={(!input.trim() && uploadedImages.length === 0) || isLoading}
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

      {/* Audio Recording Dialog */}
      <Dialog 
        open={showAudioDialog} 
        onClose={cancelRecording}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {isTranscribing ? 'Transcribing Audio...' : isRecording ? 'Recording Audio' : 'Audio Transcription'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ textAlign: 'center', py: 3 }}>
            {isRecording && (
              <Box>
                <Box
                  sx={{
                    width: 80,
                    height: 80,
                    borderRadius: '50%',
                    bgcolor: 'error.main',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto',
                    animation: 'pulse 1.5s ease-in-out infinite',
                    '@keyframes pulse': {
                      '0%, 100%': { transform: 'scale(1)', opacity: 1 },
                      '50%': { transform: 'scale(1.1)', opacity: 0.8 },
                    },
                  }}
                >
                  <MicIcon sx={{ fontSize: 40, color: 'white' }} />
                </Box>
                <Typography variant="body1" sx={{ mt: 2 }}>
                  Recording in progress...
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Click stop when you're done speaking
                </Typography>
              </Box>
            )}
            {isTranscribing && (
              <Box>
                <CircularProgress size={60} />
                <Typography variant="body1" sx={{ mt: 2 }}>
                  Transcribing your audio...
                </Typography>
              </Box>
            )}
            {audioError && (
              <Box>
                <Typography variant="body1" color="error" sx={{ mt: 2 }}>
                  {audioError}
                </Typography>
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={cancelRecording} disabled={isTranscribing}>
            Cancel
          </Button>
          {isRecording && (
            <Button 
              onClick={stopRecording} 
              variant="contained" 
              color="error"
              startIcon={<StopIcon />}
            >
              Stop Recording
            </Button>
          )}
        </DialogActions>
      </Dialog>

      {/* Conversation Sidebar */}
      <ConversationSidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        currentConversationId={currentConversationId || undefined}
      />
    </Box>
  )
}
