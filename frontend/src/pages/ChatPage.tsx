import React, { useState, useRef, useEffect, useCallback, ChangeEvent, KeyboardEvent } from 'react'
import { flushSync } from 'react-dom'
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
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
} from '@mui/material'
import {
  Send as SendIcon,
  Person as UserIcon,
  SmartToy as BotIcon,
  Settings as SettingsIcon,
  Clear as ClearIcon,
  ContentCopy as CopyIcon,
  Edit as EditIcon,
  Check as CheckIcon,
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
  VolumeUp as VolumeUpIcon,
  VolumeOff as VolumeOffIcon,
  GraphicEq as GraphicEqIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  RecordVoiceOver as RecordVoiceOverIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Tune as TuneIcon,
} from '@mui/icons-material'
import { useVoiceInput } from '@/hooks/useVoiceInput'
import { useTTS } from '@/hooks/useTTS'
import { apiService } from '@/services/api'
import { ToolsService, ExtendedTool } from '@/services/tools'
import { MarkdownRenderer, RAGSettingsPanel, RAGContextBlock, RAGChunk } from '@/components/chat'
import { ConversationSidebar } from '@/components/chat/ConversationSidebar'
import type { ChatMessage, ChatCompletionRequest, Tool, ToolCall, ConversationListItem } from '@/types/api'

interface ChatPageProps { }

interface ChatSettings {
  // Connection settings
  baseUrl: string
  endpoint: string
  apiKey: string
  openaiApiKey: string // For audio transcription (legacy, now uses local STT)
  model: string
  // Model parameters
  temperature: number
  topP: number
  topK: number
  maxTokens: number
  streamResponse: boolean
  enableTools: boolean
  selectedTools: string[]
  // Reasoning settings
  reasoningLevel: 'low' | 'medium' | 'high' | 'none'
  // Display settings
  showPerformanceMetrics: boolean
  // Voice settings
  voiceModeEnabled: boolean
  ttsEnabled: boolean
  ttsAutoPlay: boolean
  ttsVoice: string
  ttsSpeed: number
  sttModel: string
  sttLanguage: string
  vadSilenceThreshold: number
  vadSilenceDuration: number
  noSpeechThreshold: number  // Max no_speech_prob to accept (0-1)
  // RAG settings
  ragEnabled: boolean
  ragSearchMode: 'global' | 'domains'
  ragSelectedDomains: string[]  // domain IDs when mode is 'domains'
  ragTopK: number               // number of chunks to retrieve
  ragShowContext: boolean       // show retrieved context in UI
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
  openaiApiKey: '', // For audio transcription (legacy)
  model: '',
  // Model parameters
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  maxTokens: 2048,
  streamResponse: true,
  enableTools: false,
  selectedTools: [],
  // Reasoning settings
  reasoningLevel: 'low', // Default to low to prevent excessive thinking
  // Display settings
  showPerformanceMetrics: false,
  // Voice settings
  voiceModeEnabled: false,
  ttsEnabled: true,
  ttsAutoPlay: true,
  ttsVoice: 'alloy',
  ttsSpeed: 1.0,
  sttModel: 'base',
  sttLanguage: 'auto',
  vadSilenceThreshold: 0.04,   // 4% - increase if it triggers too easily
  vadSilenceDuration: 1500,    // 1.5 seconds
  noSpeechThreshold: 0.6,      // Reject if no_speech_prob > 60%
  // RAG settings
  ragEnabled: false,
  ragSearchMode: 'global',
  ragSelectedDomains: [],
  ragTopK: 5,
  ragShowContext: true,
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
  const [availableTools, setAvailableTools] = useState<ExtendedTool[]>(ToolsService.getBuiltInToolsExtended())
  const [mcpToolsLoading, setMcpToolsLoading] = useState(false)

  // Conversation management state
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)
  const [conversationTitle, setConversationTitle] = useState<string>('New Conversation')
  const [isSaving, setIsSaving] = useState(false)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)

  // Context window tracking state
  const [tokenCount, setTokenCount] = useState({ prompt: 0, total: 0 })
  const [contextLimit, setContextLimit] = useState(4096)

  // Performance metrics tracking
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>(initialPerformanceMetrics)
  const performanceRef = useRef<PerformanceMetrics>(initialPerformanceMetrics)

  // Multi-modal inputs state
  const [uploadedImages, setUploadedImages] = useState<Array<{ file: File; preview: string; base64: string }>>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Audio recording state (legacy manual recording)
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [audioError, setAudioError] = useState<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const [showAudioDialog, setShowAudioDialog] = useState(false)

  // Voice mode state
  const [sttServiceAvailable, setSttServiceAvailable] = useState(false)
  const [ttsServiceAvailable, setTtsServiceAvailable] = useState(false)
  const [lastAssistantMessage, setLastAssistantMessage] = useState<string | null>(null)
  const [voicePanelExpanded, setVoicePanelExpanded] = useState(false)
  const [ttsStarting, setTtsStarting] = useState(false)
  const [ttsStopping, setTtsStopping] = useState(false)

  // Message editing state
  const [editingMessageIndex, setEditingMessageIndex] = useState<number | null>(null)
  const [editingContent, setEditingContent] = useState<string>('')

  // RAG context state
  const [ragContext, setRagContext] = useState<RAGChunk[]>([])
  const [ragLoading, setRagLoading] = useState(false)

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Voice input hook with VAD
  const handleVoiceTranscription = useCallback((text: string) => {
    if (text.trim()) {
      setInput(prev => prev ? `${prev} ${text}` : text)
    }
  }, [])

  const voiceInput = useVoiceInput({
    silenceThreshold: settings.vadSilenceThreshold,
    silenceDuration: settings.vadSilenceDuration,
    minRecordingTime: 500,
    maxRecordingTime: 60000,
    sttModel: settings.sttModel,
    sttLanguage: settings.sttLanguage,
    noSpeechThreshold: settings.noSpeechThreshold,
    onTranscription: handleVoiceTranscription,
    onError: (error) => setAudioError(error),
  })

  // TTS hook
  const tts = useTTS({
    voice: settings.ttsVoice,
    model: 'tts-1',
    speed: settings.ttsSpeed,
    autoPlay: settings.ttsAutoPlay,
    onError: (error) => console.warn('TTS error:', error),
  })

  // Check STT/TTS service availability
  useEffect(() => {
    const checkServices = async () => {
      try {
        const sttStatus = await apiService.getSTTStatus()
        setSttServiceAvailable(sttStatus.running)
      } catch {
        setSttServiceAvailable(false)
      }

      try {
        const ttsStatus = await apiService.getTTSStatus()
        setTtsServiceAvailable(ttsStatus.running)
      } catch {
        setTtsServiceAvailable(false)
      }
    }

    checkServices()
    // Check periodically
    const interval = setInterval(checkServices, 30000)
    return () => clearInterval(interval)
  }, [])

  // Load MCP tools from connected servers
  useEffect(() => {
    const loadMCPTools = async () => {
      setMcpToolsLoading(true)
      try {
        const mcpTools = await ToolsService.getMCPTools()
        setAvailableTools([...ToolsService.getBuiltInToolsExtended(), ...mcpTools])
      } catch (error) {
        console.warn('Failed to load MCP tools:', error)
      } finally {
        setMcpToolsLoading(false)
      }
    }
    loadMCPTools()
  }, [])

  // Start TTS service
  const handleStartTTS = async () => {
    setTtsStarting(true)
    setError(null)
    try {
      const result = await apiService.startTTSService({
        voice: settings.ttsVoice,
      })
      if (result.success) {
        setTtsServiceAvailable(true)
      } else {
        setError(result.message || 'Failed to start TTS service')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start TTS service')
    } finally {
      setTtsStarting(false)
    }
  }

  // Stop TTS service
  const handleStopTTS = async () => {
    setTtsStopping(true)
    try {
      await apiService.stopTTSService()
      setTtsServiceAvailable(false)
      tts.stop() // Stop any playing audio
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop TTS service')
    } finally {
      setTtsStopping(false)
    }
  }

  // Auto-send when voice input transcription is complete and VAD detects end of speech
  useEffect(() => {
    if (settings.voiceModeEnabled && !voiceInput.isRecording && !voiceInput.isTranscribing && input.trim() && voiceInput.isListening) {
      // Small delay to allow for multi-utterance input
      const timer = setTimeout(() => {
        if (input.trim()) {
          handleSendMessage()
        }
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [voiceInput.isRecording, voiceInput.isTranscribing, input, settings.voiceModeEnabled, voiceInput.isListening])

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
          reasoningLevel: settings.reasoningLevel,
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
        if (modelInfo) {
          // Set context limit
          const limit = modelInfo.context_size || modelInfo.contextLength || modelInfo.model?.context_size || 4096
          if (limit) setContextLimit(limit)

          if (modelInfo.name) {
            setCurrentModelName(modelInfo.name)
            // Update the initial greeting message with the model name
            setMessages([
              {
                role: 'assistant',
                content: `Hello! I'm your AI assistant powered by ${modelInfo.name}. How can I help you today? I have access to various tools including weather lookup, calculator, code execution, and system information.`
              }
            ])
            // Pre-fill model setting if not already set
            setSettings((prev) => {
              if (prev.model) return prev
              const updated = { ...prev, model: modelInfo.name }
              // Persist so subsequent loads keep the detected model
              try {
                localStorage.setItem('chat-settings', JSON.stringify(updated))
              } catch (error) {
                console.warn('Failed to save chat settings to localStorage:', error)
              }
              return updated
            })
          }
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

      // Fetch RAG context if enabled
      let ragContextText = ''
      if (settings.ragEnabled) {
        setRagLoading(true)
        try {
          const queryText = typeof messageContent === 'string'
            ? messageContent
            : (messageContent.find((p: any) => p.type === 'text')?.text || '')

          if (queryText) {
            const ragPayload: any = {
              query: queryText,
              top_k: settings.ragTopK,
            }

            if (settings.ragSearchMode === 'global') {
              ragPayload.search_all_domains = true
            } else if (settings.ragSelectedDomains.length > 0) {
              ragPayload.domain_id = settings.ragSelectedDomains[0] // Primary domain
            }

            const ragResponse = await fetch('/api/v1/rag/retrieve', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(ragPayload),
            })

            if (ragResponse.ok) {
              const ragData = await ragResponse.json()
              const chunks: RAGChunk[] = (ragData.results || []).map((r: any) => ({
                id: r.id || r.chunk_id,
                content: r.content || r.text,
                score: r.score || r.similarity || 0,
                document_id: r.document_id,
                document_title: r.document_title || r.metadata?.title,
                chunk_index: r.chunk_index ?? 0,
                metadata: r.metadata,
              }))

              setRagContext(chunks)

              if (chunks.length > 0) {
                ragContextText = '\n\n--- Retrieved Context ---\n' +
                  chunks.map((c, i) => `[${i + 1}] ${c.content}`).join('\n\n') +
                  '\n--- End Context ---\n\nUse the above context to help answer the user\'s question.'
              }
            }
          }
        } catch (ragError) {
          console.warn('RAG retrieval failed:', ragError)
        } finally {
          setRagLoading(false)
        }
      } else {
        setRagContext([])
      }

      // Prepare messages with reasoning level system prompt if needed
      let requestMessages = [...messages, userMessage]

      // Add or update system message with reasoning level and RAG context
      const baseSystemContent = `You are a helpful AI assistant.${ragContextText}`
      if (settings.reasoningLevel !== 'none') {
        const systemMessage = {
          role: 'system' as const,
          content: `${baseSystemContent} Reasoning: ${settings.reasoningLevel}`
        }

        // Check if first message is already a system message
        if (requestMessages.length > 0 && requestMessages[0].role === 'system') {
          // Update existing system message to include reasoning level
          const existingContent = typeof requestMessages[0].content === 'string'
            ? requestMessages[0].content
            : ''
          requestMessages[0] = {
            ...requestMessages[0],
            content: existingContent.includes('Reasoning:')
              ? existingContent.replace(/Reasoning: \w+/, `Reasoning: ${settings.reasoningLevel}`) + ragContextText
              : `${existingContent}\nReasoning: ${settings.reasoningLevel}${ragContextText}`
          }
        } else {
          // Add new system message at the beginning
          requestMessages = [systemMessage, ...requestMessages]
        }
      } else if (ragContextText) {
        // No reasoning level but have RAG context - add system message for context
        const systemMessage = {
          role: 'system' as const,
          content: baseSystemContent
        }
        if (requestMessages.length > 0 && requestMessages[0].role === 'system') {
          requestMessages[0] = {
            ...requestMessages[0],
            content: (typeof requestMessages[0].content === 'string' ? requestMessages[0].content : '') + ragContextText
          }
        } else {
          requestMessages = [systemMessage, ...requestMessages]
        }
      }

      // Truncate messages to fit context window
      // Estimate tokens: Text ~3.5 chars/token, Image ~1000 tokens
      // Reserve 1000 tokens for generation
      const estimateTokens = (msg: ChatMessage) => {
        let count = 0
        // Text content
        if (typeof msg.content === 'string') {
          count += msg.content.length / 3.5
        } else if (Array.isArray(msg.content)) {
          msg.content.forEach((part: any) => {
            if (part.type === 'text') count += (part.text?.length || 0) / 3.5
            if (part.type === 'image_url') count += 1000 // Conservative estimate for image
          })
        }

        // Reasoning content
        if (msg.reasoning_content) {
          count += msg.reasoning_content.length / 3.5
        }

        return Math.ceil(count)
      }

      // Keep system prompt (index 0) and newest messages that fit
      if (requestMessages.length > 1) {
        const systemMsg = requestMessages[0]
        const systemTokens = estimateTokens(systemMsg)
        const budget = contextLimit - 1000 - systemTokens // Reserve for generation and system prompt

        let used = 0
        const keptMessages: ChatMessage[] = []

        // Iterate backwards from newest
        for (let i = requestMessages.length - 1; i >= 1; i--) {
          const msg = requestMessages[i]
          const cost = estimateTokens(msg)
          if (used + cost <= budget) {
            keptMessages.unshift(msg)
            used += cost
          } else {
            console.log(`Truncating history: Dropping message with ~${cost} tokens. Budget remaining: ${budget - used}`)
            break
          }
        }

        requestMessages = [systemMsg, ...keptMessages]
      }

      const request: ChatCompletionRequest = {
        messages: requestMessages,
        model: settings.model,
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
        let buffer = ''  // Buffer for incomplete SSE messages

        function pump(): Promise<void> {
          return reader.read().then(({ done, value }) => {
            if (done) {
              controller.close();
              return;
            }

            // Append to buffer and parse complete SSE messages
            buffer += decoder.decode(value, { stream: true });

            // Split by double-newline (SSE message separator) or single newline for simple cases
            // SSE spec: messages are separated by blank lines
            const messages = buffer.split(/\n\n/);

            // Keep the last potentially incomplete message in buffer
            buffer = messages.pop() || '';

            for (const message of messages) {
              const lines = message.split('\n');
              let data = '';

              for (const line of lines) {
                const trimmedLine = line.trim();
                if (trimmedLine.startsWith('data: ')) {
                  // Accumulate data field (may be spread across lines in some servers)
                  data += trimmedLine.slice(6);
                } else if (trimmedLine.startsWith('data:')) {
                  data += trimmedLine.slice(5);
                }
              }

              if (data === '[DONE]') {
                controller.close();
                return;
              }

              if (data) {
                try {
                  const parsed = JSON.parse(data);
                  controller.enqueue(parsed);
                } catch (e) {
                  // Try to handle case where data was split across chunks
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
        // Tool calls are accumulated here across all deltas
        let accumulatedToolCalls: ToolCall[] = []
        let tokenCount = 0
        let firstTokenReceived = false
        let toolCallsProcessed = false // Flag to prevent processing tool calls multiple times

        let fullResponseText = ''

        // TTS sentence chunking - send sentences as they complete for lower latency
        let ttsSentenceBuffer = ''
        const shouldStreamTTS = settings.voiceModeEnabled && settings.ttsEnabled && ttsServiceAvailable

        // Function to extract and speak complete sentences
        const processTTSBuffer = (forceFlush: boolean = false) => {
          if (!shouldStreamTTS || !ttsSentenceBuffer.trim()) return

          // Look for sentence boundaries: . ! ? followed by space or end
          // Also split on newlines for code/list responses
          const sentenceBreakRegex = /([.!?])\s+|(\n\n)/g

          let lastIndex = 0
          let match

          while ((match = sentenceBreakRegex.exec(ttsSentenceBuffer)) !== null) {
            // Extract the complete sentence including the punctuation
            const sentence = ttsSentenceBuffer.slice(lastIndex, match.index + 1).trim()
            if (sentence.length > 0) {
              tts.speak(sentence)
            }
            lastIndex = match.index + match[0].length
          }

          // Keep the remainder (incomplete sentence) in the buffer
          ttsSentenceBuffer = ttsSentenceBuffer.slice(lastIndex)

          // If forcing flush (end of stream), speak any remaining content
          if (forceFlush && ttsSentenceBuffer.trim().length > 0) {
            tts.speak(ttsSentenceBuffer.trim())
            ttsSentenceBuffer = ''
          }
        }

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

            // Flush any remaining TTS content
            processTTSBuffer(true)

            // Process tool calls when stream ends (not when finish_reason is received)
            // llama.cpp sends finish_reason before all tool data is accumulated
            if (accumulatedToolCalls.length > 0 && !toolCallsProcessed) {
              toolCallsProcessed = true
              console.log('=== Stream ended with accumulated tool calls ===')
              console.log('Tool calls:', JSON.stringify(accumulatedToolCalls, null, 2))

              let messagesSnapshot: ChatMessage[] = []
              flushSync(() => {
                setMessages((prev: ChatMessage[]) => {
                  const newMessages = [...prev]
                  const lastMessage = newMessages[newMessages.length - 1]
                  if (lastMessage.role === 'assistant') {
                    const updatedAssistant = {
                      ...lastMessage,
                      tool_calls: accumulatedToolCalls
                    }
                    newMessages[newMessages.length - 1] = updatedAssistant
                  }
                  messagesSnapshot = [...newMessages]
                  return newMessages
                })
              })

              await handleToolCalls(accumulatedToolCalls, messagesSnapshot)
            }
            break
          }

          console.log('Received stream chunk:', value)
          console.log('Delta object:', value.choices?.[0]?.delta)

          // Check for content in multiple locations
          // content: actual response content
          // __verbose.content: llama.cpp verbose output
          const contentDelta = value.choices?.[0]?.delta?.content ||
            value.__verbose?.content ||
            ''

          // Debug: log if we're getting reasoning_content
          if (value.choices?.[0]?.delta?.reasoning_content) {
            console.log('Reasoning content received:', value.choices[0].delta.reasoning_content)
          }

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

            // Accumulate for full response tracking
            fullResponseText += contentDelta

            // Feed TTS buffer and process any complete sentences
            if (shouldStreamTTS) {
              ttsSentenceBuffer += contentDelta
              processTTSBuffer(false)
            }

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
            // Tool calls arrive as deltas - accumulate into a single tool call
            // SIMPLE APPROACH: We only support one tool call at a time, 
            // accumulate all deltas into index 0
            console.log('=== Tool call delta received ===')
            console.log('Delta:', JSON.stringify(value.choices[0].delta.tool_calls))

            for (const delta of value.choices[0].delta.tool_calls) {
              // Initialize the single tool call if not exists
              if (accumulatedToolCalls.length === 0) {
                accumulatedToolCalls[0] = {
                  id: delta.id || `call_${Date.now()}`,
                  type: 'function',
                  function: {
                    name: '',
                    arguments: ''
                  }
                }
                console.log('Initialized tool call accumulator')
              }

              // Always accumulate into the first (and only) tool call
              if (delta.function?.name) {
                accumulatedToolCalls[0].function.name += delta.function.name
                console.log('Added to name:', delta.function.name)
              }
              if (delta.function?.arguments) {
                accumulatedToolCalls[0].function.arguments += delta.function.arguments
                console.log('Added to args:', delta.function.arguments)
              }
              // Capture ID if provided and we don't have one yet
              if (delta.id && !accumulatedToolCalls[0].id.startsWith('call_')) {
                accumulatedToolCalls[0].id = delta.id
              }
            }
            console.log('Current accumulated:', JSON.stringify(accumulatedToolCalls[0]))
          }

          // Note: Tool calls are now processed at stream end (when done === true)
          // rather than when finish_reason === 'tool_calls' because llama.cpp
          // sends finish_reason before all tool data is fully accumulated
        }
      } finally {
        reader.releaseLock()

        // After streaming completes, if content is empty but reasoning_content exists,
        // use reasoning_content as the visible content (for models that put all output in thinking)
        setMessages((prev: ChatMessage[]) => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage.role === 'assistant' && !lastMessage.content && lastMessage.reasoning_content) {
            lastMessage.content = lastMessage.reasoning_content
          }
          return newMessages
        })
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

    // Trigger TTS for the response
    const responseText = typeof assistantMessage.content === 'string' ? assistantMessage.content : ''
    if (responseText.trim() && settings.voiceModeEnabled && settings.ttsEnabled && ttsServiceAvailable) {
      tts.speak(responseText.trim())
    }

    // Handle tool calls if present
    if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
      await handleToolCalls(assistantMessage.tool_calls)
    }
  }

  const handleToolCalls = async (toolCalls: ToolCall[], messagesSnapshot?: ChatMessage[]) => {
    // Use provided snapshot or capture current messages from state
    let currentMessages: ChatMessage[] = messagesSnapshot || []

    if (!messagesSnapshot) {
      // Fallback: Get current messages state for building the follow-up request
      setMessages((prev: ChatMessage[]) => {
        currentMessages = [...prev]
        return prev // Don't modify, just capture
      })
    }

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

        // Update the tool execution message with result and add tool result
        setMessages((prev: ChatMessage[]) => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage.content?.includes('🔧 Executing tool:')) {
            lastMessage.content = `✅ Tool executed: ${toolCall.function.name}\n\nResult:\n${result}`
          }
          return [...newMessages, toolResultMessage]
        })

        // Find the assistant message that made the tool call
        // It should be the last assistant message with tool_calls in currentMessages
        const assistantToolCallMessage = [...currentMessages].reverse().find(
          (msg) => msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0
        )

        // Build proper message history for the LLM:
        // 1. Include all messages up to and including the assistant's tool_call message
        // 2. Add the tool result message
        // The chat template expects: [user msg] -> [assistant with tool_calls] -> [tool result]

        // Filter out UI-only messages (tool execution indicators)
        const messagesForLLM = currentMessages
          .filter(
            (msg) => !msg.content?.toString().includes('🔧 Executing tool:') &&
              !msg.content?.toString().includes('✅ Tool executed:')
          )
          .map((msg) => {
            // Clean up message for API - remove UI-only properties
            const cleanMsg: ChatMessage = {
              role: msg.role,
              content: msg.content,
            }

            // For assistant messages with tool_calls, content must be null (not empty string)
            if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
              cleanMsg.content = null as any // llama.cpp requires null, not empty string
              cleanMsg.tool_calls = msg.tool_calls
            }

            // Include tool_call_id for tool messages
            if (msg.role === 'tool' && msg.tool_call_id) {
              cleanMsg.tool_call_id = msg.tool_call_id
            }

            return cleanMsg
          })

        // Add the tool result message
        const followUpMessages: ChatMessage[] = [
          ...messagesForLLM,
          toolResultMessage
        ]

        // Continue conversation with tool result
        const followUpRequest: ChatCompletionRequest = {
          messages: followUpMessages,
          model: settings.model,
          temperature: settings.temperature,
          top_p: settings.topP,
          top_k: settings.topK,
          max_tokens: settings.maxTokens,
          stream: false, // Use non-streaming for tool follow-ups
        }

        // Debug: Log the request to see what we're sending
        console.log('=== Tool Follow-up Request ===')
        console.log('Messages:', JSON.stringify(followUpMessages, null, 2))
        console.log('Full request:', JSON.stringify(followUpRequest, null, 2))

        const followUpResponse = await createChatCompletion(followUpRequest)
        const responseMessage = followUpResponse.choices[0].message
        // If content is empty but reasoning_content exists, use it as the visible content
        // This handles models that put all output in reasoning/thinking tokens
        if (!responseMessage.content && responseMessage.reasoning_content) {
          responseMessage.content = responseMessage.reasoning_content
        }
        setMessages((prev: ChatMessage[]) => [...prev, responseMessage])

        // Update currentMessages for the next iteration
        currentMessages = [...currentMessages, toolResultMessage, followUpResponse.choices[0].message]

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

  // Global keyboard shortcuts for chat
  useEffect(() => {
    const handleGlobalKeyDown = (e: globalThis.KeyboardEvent) => {
      // Cmd/Ctrl + L: Clear chat
      if ((e.metaKey || e.ctrlKey) && e.key === 'l') {
        e.preventDefault()
        clearChat()
      }
      // Cmd/Ctrl + /: Toggle settings
      if ((e.metaKey || e.ctrlKey) && e.key === '/') {
        e.preventDefault()
        setShowSettings(prev => !prev)
      }
      // Cmd/Ctrl + N: New conversation
      if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
        e.preventDefault()
        handleNewConversation()
      }
      // Cmd/Ctrl + S: Save conversation
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault()
        if (hasUnsavedChanges) {
          saveConversation()
        }
      }
      // Escape: Stop TTS / Close settings
      if (e.key === 'Escape') {
        if (tts.isPlaying) {
          tts.stop()
        } else if (showSettings) {
          setShowSettings(false)
        }
      }
    }

    document.addEventListener('keydown', handleGlobalKeyDown)
    return () => document.removeEventListener('keydown', handleGlobalKeyDown)
  }, [hasUnsavedChanges, showSettings, tts.isPlaying])

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: `Hello! I'm your AI assistant powered by ${currentModelName}. How can I help you today? I have access to various tools including weather lookup, calculator, code execution, and system information.`
      }
    ])
    setError(null)
  }

  const copyToClipboard = (content: string | Array<{ type: string; text?: string; image_url?: { url: string } }>) => {
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

  // Start editing a message
  const startEditingMessage = (index: number) => {
    const message = messages[index]
    if (typeof message.content === 'string') {
      setEditingContent(message.content)
    } else if (Array.isArray(message.content)) {
      // Extract text from multi-modal content
      const textParts = message.content
        .filter((part: any) => part.type === 'text' && part.text)
        .map((part: any) => part.text)
      setEditingContent(textParts.join('\n'))
    }
    setEditingMessageIndex(index)
  }

  // Cancel editing
  const cancelEditingMessage = () => {
    setEditingMessageIndex(null)
    setEditingContent('')
  }

  // Save edited message and regenerate response
  const saveEditedMessage = async (index: number) => {
    if (!editingContent.trim()) return

    // Update the message at the given index
    const updatedMessages = messages.slice(0, index + 1)
    updatedMessages[index] = {
      ...updatedMessages[index],
      content: editingContent.trim()
    }

    // Remove all messages after the edited one (will regenerate)
    setMessages(updatedMessages)
    setEditingMessageIndex(null)
    setEditingContent('')
    setHasUnsavedChanges(true)

    // Regenerate the response
    setIsLoading(true)
    setError(null)

    try {
      const selectedToolsForRequest = settings.enableTools
        ? availableTools.filter((tool: Tool) => settings.selectedTools.includes(tool.function.name))
        : []

      // Prepare messages with reasoning level system prompt if needed
      let requestMessages = updatedMessages

      // Add or update system message with reasoning level for GPT-OSS models
      if (settings.reasoningLevel !== 'none') {
        const systemMessage = {
          role: 'system' as const,
          content: `You are a helpful AI assistant. Reasoning: ${settings.reasoningLevel}`
        }

        // Check if first message is already a system message
        if (requestMessages.length > 0 && requestMessages[0].role === 'system') {
          // Update existing system message to include reasoning level
          const existingContent = typeof requestMessages[0].content === 'string'
            ? requestMessages[0].content
            : ''
          requestMessages[0] = {
            ...requestMessages[0],
            content: existingContent.includes('Reasoning:')
              ? existingContent.replace(/Reasoning: \w+/, `Reasoning: ${settings.reasoningLevel}`)
              : `${existingContent}\nReasoning: ${settings.reasoningLevel}`
          }
        } else {
          // Add new system message at the beginning
          requestMessages = [systemMessage, ...requestMessages]
        }
      }

      const request: ChatCompletionRequest = {
        messages: requestMessages,
        model: settings.model,
        temperature: settings.temperature,
        top_p: settings.topP,
        top_k: settings.topK,
        max_tokens: settings.maxTokens,
        stream: settings.streamResponse,
        tools: selectedToolsForRequest.length > 0 ? selectedToolsForRequest : undefined,
        tool_choice: selectedToolsForRequest.length > 0 ? 'auto' : undefined,
      }

      if (settings.streamResponse) {
        await handleStreamingResponse(request)
      } else {
        await handleNonStreamingResponse(request)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to regenerate response')
    } finally {
      setIsLoading(false)
    }
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
              {currentModelName} | ~{tokenCount.prompt.toLocaleString()} tokens | Reasoning: {settings.reasoningLevel}
            </Typography>
          </Box>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* TTS Quick Launch Button - shown when TTS is not available */}
          {!ttsServiceAvailable && (
            <Tooltip title="Start Text-to-Speech Service">
              <span>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handleStartTTS}
                  disabled={ttsStarting}
                  startIcon={ttsStarting ? <CircularProgress size={14} /> : <VolumeUpIcon />}
                  sx={{
                    fontSize: '0.7rem',
                    textTransform: 'none',
                    py: 0.25,
                    px: 1,
                    minWidth: 'auto',
                    borderColor: 'rgba(255,255,255,0.3)',
                    '&:hover': { borderColor: 'primary.main' }
                  }}
                >
                  {ttsStarting ? 'Starting...' : 'Start TTS'}
                </Button>
              </span>
            </Tooltip>
          )}
          {/* Voice Mode Toggle */}
          <Tooltip title={
            !sttServiceAvailable
              ? "STT service not running - deploy from STT page first"
              : settings.voiceModeEnabled
                ? "Voice Mode Active - Click to disable"
                : "Enable Voice Mode"
          }>
            <span>
              <IconButton
                onClick={() => {
                  if (sttServiceAvailable) {
                    const newVoiceMode = !settings.voiceModeEnabled
                    saveSettings({ ...settings, voiceModeEnabled: newVoiceMode })
                    if (newVoiceMode) {
                      voiceInput.startListening()
                    } else {
                      voiceInput.stopListening()
                      tts.stop()
                    }
                  }
                }}
                disabled={!sttServiceAvailable}
                size="small"
                sx={{
                  bgcolor: settings.voiceModeEnabled ? 'success.dark' : 'transparent',
                  color: settings.voiceModeEnabled ? 'success.contrastText' : 'text.secondary',
                  '&:hover': { bgcolor: settings.voiceModeEnabled ? 'success.main' : 'action.hover' },
                  animation: voiceInput.isListening ? 'pulse 2s ease-in-out infinite' : 'none',
                  '@keyframes pulse': {
                    '0%, 100%': { boxShadow: '0 0 0 0 rgba(76, 175, 80, 0.4)' },
                    '50%': { boxShadow: '0 0 0 8px rgba(76, 175, 80, 0)' },
                  },
                }}
              >
                <RecordVoiceOverIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          {/* Context window progress bar */}
          <Tooltip title={`Context: ${tokenCount.prompt.toLocaleString()} / ${contextLimit.toLocaleString()} tokens (${((tokenCount.prompt / contextLimit) * 100).toFixed(1)}%)`}>
            <Box sx={{ width: 100, mr: 1 }}>
              <LinearProgress
                variant="determinate"
                value={Math.min((tokenCount.prompt / contextLimit) * 100, 100)}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  bgcolor: 'action.hover',
                  '& .MuiLinearProgress-bar': {
                    bgcolor: tokenCount.prompt > contextLimit * 0.9 ? 'error.main' :
                      tokenCount.prompt > contextLimit * 0.7 ? 'warning.main' : 'primary.main'
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
          <CardContent sx={{
            maxHeight: 'calc(100vh - 300px)',
            overflowY: 'auto',
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              bgcolor: 'rgba(0, 0, 0, 0.1)',
            },
            '&::-webkit-scrollbar-thumb': {
              bgcolor: 'rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              '&:hover': {
                bgcolor: 'rgba(255, 255, 255, 0.3)',
              },
            },
          }}>
            <Typography variant="h6" gutterBottom>
              Chat Settings
            </Typography>

            {/* RAG Settings */}
            <RAGSettingsPanel
              ragEnabled={settings.ragEnabled}
              ragSearchMode={settings.ragSearchMode}
              ragSelectedDomains={settings.ragSelectedDomains}
              ragTopK={settings.ragTopK}
              ragShowContext={settings.ragShowContext}
              onSettingsChange={(ragSettings) => saveSettings({ ...settings, ...ragSettings })}
            />

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
                  label="Model"
                  fullWidth
                  value={settings.model}
                  onChange={(e) => saveSettings({ ...settings, model: e.target.value })}
                  placeholder="e.g. gpt-4o-mini or llama-3.1-8b-instruct"
                  helperText="Sent as the model parameter to the chat completions endpoint"
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

            {/* Reasoning Level Control */}
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Reasoning Level</Typography>
                <Select
                  fullWidth
                  value={settings.reasoningLevel}
                  onChange={(e) => saveSettings({ ...settings, reasoningLevel: e.target.value as 'low' | 'medium' | 'high' | 'none' })}
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
                >
                  <MenuItem value="none">None (Direct responses)</MenuItem>
                  <MenuItem value="low">Low (Fast responses)</MenuItem>
                  <MenuItem value="medium">Medium (Balanced)</MenuItem>
                  <MenuItem value="high">High (Deep analysis)</MenuItem>
                </Select>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                  Controls how much the model thinks before responding. Use "Low" or "None" if the model is using too many tokens for thinking.
                </Typography>
              </Grid>
            </Grid>

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
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Available Tools (select which tools the model can use):
                    </Typography>
                    {mcpToolsLoading && (
                      <Chip label="Loading MCP tools..." size="small" variant="outlined" />
                    )}
                  </Box>
                </Grid>
                {/* Group built-in tools */}
                {availableTools.filter((t: ExtendedTool) => t._source !== 'mcp').length > 0 && (
                  <Grid item xs={12}>
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 'bold' }}>
                      📋 Built-in Tools
                    </Typography>
                  </Grid>
                )}
                {availableTools.filter((t: ExtendedTool) => t._source !== 'mcp').map((tool: ExtendedTool) => (
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
                {/* Group MCP tools by server */}
                {availableTools.filter((t: ExtendedTool) => t._source === 'mcp').length > 0 && (
                  <Grid item xs={12}>
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 'bold', mt: 2 }}>
                      🔌 MCP Tools
                    </Typography>
                  </Grid>
                )}
                {availableTools.filter((t: ExtendedTool) => t._source === 'mcp').map((tool: ExtendedTool) => (
                  <Grid item xs={12} sm={6} md={4} key={tool.function.name}>
                    <Card
                      variant="outlined"
                      sx={{
                        cursor: 'pointer',
                        bgcolor: settings.selectedTools.includes(tool.function.name) ? 'action.selected' : 'background.paper',
                        '&:hover': { bgcolor: 'action.hover' },
                        borderColor: 'primary.light',
                      }}
                      onClick={() => toggleToolSelection(tool.function.name)}
                    >
                      <CardContent sx={{ py: 1.5 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1, justifyContent: 'space-between' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <IconButton size="small" sx={{ mr: 1 }}>
                              {settings.selectedTools.includes(tool.function.name) ?
                                <CheckBoxIcon color="primary" /> :
                                <CheckBoxOutlineBlankIcon />
                              }
                            </IconButton>
                            <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                              {tool.function.name.replace(/^mcp_[^_]+_/, '')}
                            </Typography>
                          </Box>
                          <Chip
                            label={tool._serverName || 'MCP'}
                            size="small"
                            variant="outlined"
                            color="primary"
                            sx={{ fontSize: '0.65rem', height: 18 }}
                          />
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
                    {availableTools.filter((t: ExtendedTool) => t._source === 'mcp').length > 0 && (
                      <span> ({availableTools.filter((t: ExtendedTool) => t._source === 'mcp').length} from MCP servers)</span>
                    )}
                  </Typography>
                </Grid>
              </Grid>
            )}

            <Divider sx={{ my: 3 }} />

            {/* Voice Settings Section */}
            <Typography variant="h6" gutterBottom>
              Voice Settings (STT/TTS)
            </Typography>

            {/* Service Status */}
            <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
              <Chip
                label={sttServiceAvailable ? "STT Service Running" : "STT Service Not Available"}
                color={sttServiceAvailable ? "success" : "default"}
                size="small"
                icon={<MicIcon />}
              />
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip
                  label={ttsServiceAvailable ? "TTS Service Running" : "TTS Service Not Available"}
                  color={ttsServiceAvailable ? "success" : "default"}
                  size="small"
                  icon={<VolumeUpIcon />}
                />
                {!ttsServiceAvailable ? (
                  <Button
                    size="small"
                    variant="contained"
                    color="primary"
                    onClick={handleStartTTS}
                    disabled={ttsStarting}
                    startIcon={ttsStarting ? <CircularProgress size={14} /> : <PlayIcon />}
                    sx={{ fontSize: '0.75rem', textTransform: 'none', py: 0.5 }}
                  >
                    {ttsStarting ? 'Starting...' : 'Start TTS'}
                  </Button>
                ) : (
                  <Button
                    size="small"
                    variant="outlined"
                    color="error"
                    onClick={handleStopTTS}
                    disabled={ttsStopping}
                    startIcon={ttsStopping ? <CircularProgress size={14} /> : <StopIcon />}
                    sx={{ fontSize: '0.75rem', textTransform: 'none', py: 0.5 }}
                  >
                    {ttsStopping ? 'Stopping...' : 'Stop TTS'}
                  </Button>
                )}
              </Box>
            </Box>

            {!sttServiceAvailable && (
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Deploy STT service from the Deploy page. TTS can be started directly from here.
              </Typography>
            )}

            <Grid container spacing={3}>
              {/* TTS Settings */}
              <Grid item xs={12} sm={6} md={4}>
                <Typography gutterBottom>TTS Voice</Typography>
                <Select
                  fullWidth
                  size="small"
                  value={settings.ttsVoice}
                  onChange={(e) => saveSettings({ ...settings, ttsVoice: e.target.value })}
                  disabled={!ttsServiceAvailable}
                >
                  <MenuItem value="alloy">Alloy</MenuItem>
                  <MenuItem value="echo">Echo</MenuItem>
                  <MenuItem value="fable">Fable</MenuItem>
                  <MenuItem value="onyx">Onyx</MenuItem>
                  <MenuItem value="nova">Nova</MenuItem>
                  <MenuItem value="shimmer">Shimmer</MenuItem>
                </Select>
              </Grid>

              <Grid item xs={12} sm={6} md={4}>
                <Typography gutterBottom>TTS Speed: {settings.ttsSpeed}x</Typography>
                <Slider
                  value={settings.ttsSpeed}
                  onChange={(_: Event, value: number | number[]) => saveSettings({ ...settings, ttsSpeed: value as number })}
                  min={0.5}
                  max={2.0}
                  step={0.1}
                  marks={[
                    { value: 0.5, label: '0.5x' },
                    { value: 1.0, label: '1x' },
                    { value: 2.0, label: '2x' },
                  ]}
                  disabled={!ttsServiceAvailable}
                />
              </Grid>

              <Grid item xs={12} sm={6} md={4}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.ttsAutoPlay}
                      onChange={(e) => saveSettings({ ...settings, ttsAutoPlay: e.target.checked })}
                      disabled={!ttsServiceAvailable}
                    />
                  }
                  label="Auto-play TTS for responses"
                />
              </Grid>

              {/* STT Settings */}
              <Grid item xs={12} sm={6} md={4}>
                <Typography gutterBottom>STT Model</Typography>
                <Select
                  fullWidth
                  size="small"
                  value={settings.sttModel}
                  onChange={(e) => saveSettings({ ...settings, sttModel: e.target.value })}
                  disabled={!sttServiceAvailable}
                >
                  <MenuItem value="tiny">Tiny (fastest)</MenuItem>
                  <MenuItem value="base">Base (balanced)</MenuItem>
                  <MenuItem value="small">Small (better)</MenuItem>
                  <MenuItem value="medium">Medium (great)</MenuItem>
                  <MenuItem value="large-v3">Large-v3 (best)</MenuItem>
                  <MenuItem value="distil-large-v3.5-ct2">Distil-Large-v3.5 (fast & high quality)</MenuItem>
                </Select>
              </Grid>

              <Grid item xs={12} sm={6} md={4}>
                <Typography gutterBottom>STT Language</Typography>
                <Select
                  fullWidth
                  size="small"
                  value={settings.sttLanguage}
                  onChange={(e) => saveSettings({ ...settings, sttLanguage: e.target.value })}
                  disabled={!sttServiceAvailable}
                >
                  <MenuItem value="auto">Auto-detect</MenuItem>
                  <MenuItem value="en">English</MenuItem>
                  <MenuItem value="es">Spanish</MenuItem>
                  <MenuItem value="fr">French</MenuItem>
                  <MenuItem value="de">German</MenuItem>
                  <MenuItem value="it">Italian</MenuItem>
                  <MenuItem value="pt">Portuguese</MenuItem>
                  <MenuItem value="ru">Russian</MenuItem>
                  <MenuItem value="ja">Japanese</MenuItem>
                  <MenuItem value="ko">Korean</MenuItem>
                  <MenuItem value="zh">Chinese</MenuItem>
                </Select>
              </Grid>

              {/* VAD Settings */}
              <Grid item xs={12} sm={6} md={4}>
                <Typography gutterBottom>Silence Duration: {settings.vadSilenceDuration}ms</Typography>
                <Slider
                  value={settings.vadSilenceDuration}
                  onChange={(_: Event, value: number | number[]) => saveSettings({ ...settings, vadSilenceDuration: value as number })}
                  min={500}
                  max={3000}
                  step={100}
                  marks={[
                    { value: 500, label: '0.5s' },
                    { value: 1500, label: '1.5s' },
                    { value: 3000, label: '3s' },
                  ]}
                  disabled={!sttServiceAvailable}
                />
                <Typography variant="caption" color="text.secondary">
                  How long to wait after you stop speaking before sending
                </Typography>
              </Grid>

              <Grid item xs={12} sm={6} md={4}>
                <Typography gutterBottom>Silence Threshold: {(settings.vadSilenceThreshold * 100).toFixed(0)}%</Typography>
                <Slider
                  value={settings.vadSilenceThreshold}
                  onChange={(_: Event, value: number | number[]) => saveSettings({ ...settings, vadSilenceThreshold: value as number })}
                  min={0.01}
                  max={0.1}
                  step={0.01}
                  marks={[
                    { value: 0.01, label: '1%' },
                    { value: 0.05, label: '5%' },
                    { value: 0.1, label: '10%' },
                  ]}
                  disabled={!sttServiceAvailable}
                />
                <Typography variant="caption" color="text.secondary">
                  Volume level below which is considered silence
                </Typography>
              </Grid>
            </Grid>

            <Divider sx={{ my: 3 }} />

            {/* Keyboard Shortcuts */}
            <Typography variant="h6" gutterBottom>
              Keyboard Shortcuts
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {[
                { keys: 'Enter', action: 'Send message' },
                { keys: navigator.platform.includes('Mac') ? 'Cmd+L' : 'Ctrl+L', action: 'Clear chat' },
                { keys: navigator.platform.includes('Mac') ? 'Cmd+/' : 'Ctrl+/', action: 'Toggle settings' },
                { keys: navigator.platform.includes('Mac') ? 'Cmd+N' : 'Ctrl+N', action: 'New conversation' },
                { keys: navigator.platform.includes('Mac') ? 'Cmd+S' : 'Ctrl+S', action: 'Save conversation' },
                { keys: navigator.platform.includes('Mac') ? 'Cmd+K' : 'Ctrl+K', action: 'Command palette' },
                { keys: 'Esc', action: 'Stop TTS / Close' },
              ].map(({ keys, action }) => (
                <Box key={keys} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip
                    label={keys}
                    size="small"
                    sx={{
                      height: 24,
                      fontSize: '0.75rem',
                      fontFamily: 'monospace',
                      bgcolor: 'rgba(255,255,255,0.05)',
                    }}
                  />
                  <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.8125rem' }}>
                    {action}
                  </Typography>
                </Box>
              ))}
            </Box>
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
          {/* RAG Context Display */}
          {settings.ragShowContext && (ragLoading || ragContext.length > 0) && (
            <RAGContextBlock
              chunks={ragContext}
              isLoading={ragLoading}
              onFetchNeighbors={async (documentId, chunkIndex, direction) => {
                // Fetch neighboring chunks for exploration
                try {
                  const response = await fetch('/api/v1/rag/chunks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      document_id: documentId,
                      start_index: direction === 'before' ? Math.max(0, chunkIndex - 2) : chunkIndex + 1,
                      count: 2,
                    }),
                  })
                  if (response.ok) {
                    const data = await response.json()
                    return (data.chunks || []).map((c: any) => ({
                      id: c.id,
                      content: c.content,
                      score: 0,
                      document_id: documentId,
                      chunk_index: c.chunk_index,
                      metadata: c.metadata,
                    }))
                  }
                } catch (error) {
                  console.warn('Failed to fetch neighbor chunks:', error)
                }
                return []
              }}
            />
          )}
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
                  <Box sx={{ ml: 'auto', display: 'flex', gap: 0.5 }}>
                    {/* Edit button for user messages */}
                    {message.role === 'user' && editingMessageIndex !== index && (
                      <Tooltip title="Edit message">
                        <IconButton
                          size="small"
                          onClick={() => startEditingMessage(index)}
                          disabled={isLoading}
                          sx={{ color: 'text.secondary' }}
                        >
                          <EditIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    )}
                    {/* TTS Play Button for assistant messages */}
                    {message.role === 'assistant' && ttsServiceAvailable && typeof message.content === 'string' && message.content.trim() && (
                      <Tooltip title={tts.isPlaying && tts.currentText === message.content ? "Stop speaking" : "Read aloud"}>
                        <IconButton
                          size="small"
                          onClick={() => {
                            if (tts.isPlaying && tts.currentText === message.content) {
                              tts.stop()
                            } else {
                              tts.speakNow(message.content as string)
                            }
                          }}
                          sx={{
                            color: tts.isPlaying && tts.currentText === message.content ? 'primary.main' : 'text.secondary',
                          }}
                        >
                          {tts.isPlaying && tts.currentText === message.content ? <StopIcon fontSize="small" /> : <VolumeUpIcon fontSize="small" />}
                        </IconButton>
                      </Tooltip>
                    )}
                    <Tooltip title="Copy to clipboard">
                      <IconButton
                        size="small"
                        onClick={() => copyToClipboard(message.content)}
                      >
                        <CopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
                {/* Edit mode for user messages */}
                {editingMessageIndex === index ? (
                  <Box sx={{ width: '100%' }}>
                    <TextField
                      fullWidth
                      multiline
                      minRows={2}
                      maxRows={8}
                      value={editingContent}
                      onChange={(e) => setEditingContent(e.target.value)}
                      autoFocus
                      sx={{
                        mb: 1,
                        '& .MuiOutlinedInput-root': {
                          fontSize: '0.875rem',
                          bgcolor: 'rgba(0, 0, 0, 0.2)',
                        }
                      }}
                    />
                    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                      <Button
                        size="small"
                        onClick={cancelEditingMessage}
                        sx={{ textTransform: 'none', fontSize: '0.75rem' }}
                      >
                        Cancel
                      </Button>
                      <Button
                        size="small"
                        variant="contained"
                        onClick={() => saveEditedMessage(index)}
                        disabled={!editingContent.trim() || isLoading}
                        startIcon={isLoading ? <CircularProgress size={12} /> : <CheckIcon />}
                        sx={{ textTransform: 'none', fontSize: '0.75rem' }}
                      >
                        {isLoading ? 'Regenerating...' : 'Save & Regenerate'}
                      </Button>
                    </Box>
                  </Box>
                ) : (
                  /* Normal message content */
                  (() => {
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
                  })()
                )}
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

      {/* Voice Mode Compact Badge */}
      {settings.voiceModeEnabled && (
        <Box sx={{ mb: 2 }}>
          {/* Compact Badge - Always Visible */}
          <Paper
            onClick={() => setVoicePanelExpanded(!voicePanelExpanded)}
            sx={{
              p: 1,
              px: 1.5,
              borderRadius: 2,
              bgcolor: 'background.paper',
              border: '1px solid',
              borderColor: voiceInput.isRecording ? 'error.main' : voiceInput.isListening ? 'success.main' : 'rgba(255, 255, 255, 0.1)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              height: 48, // Fixed height to prevent bounce
              transition: 'border-color 0.2s ease',
              '&:hover': {
                borderColor: voiceInput.isRecording ? 'error.light' : voiceInput.isListening ? 'success.light' : 'rgba(255, 255, 255, 0.2)',
              },
            }}
          >
            {/* Left: Animated mic icon */}
            <Box sx={{
              position: 'relative',
              width: 32,
              height: 32,
              flexShrink: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              {/* Pulse animation ring */}
              <Box sx={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                borderRadius: '50%',
                bgcolor: voiceInput.isRecording ? 'error.main' : voiceInput.isListening ? 'success.main' : 'grey.700',
                opacity: 0.3,
                transform: `scale(${1 + voiceInput.volume * 1.5})`,
                transition: 'transform 0.05s ease',
              }} />
              <MicIcon sx={{
                fontSize: 18,
                color: voiceInput.isRecording ? 'error.main' : voiceInput.isListening ? 'success.main' : 'grey.500',
                zIndex: 1,
              }} />
            </Box>

            {/* Status text - fixed width */}
            <Typography
              variant="body2"
              sx={{
                fontWeight: 600,
                fontSize: '0.8rem',
                width: 100, // Fixed width
                flexShrink: 0,
                whiteSpace: 'nowrap',
              }}
            >
              {voiceInput.isTranscribing
                ? 'Processing...'
                : voiceInput.isRecording
                  ? `Rec ${(voiceInput.recordingTime / 1000).toFixed(1)}s`
                  : voiceInput.isListening
                    ? 'Listening'
                    : 'Paused'}
            </Typography>

            {/* Volume display */}
            <Typography
              variant="caption"
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.7rem',
                color: voiceInput.volume > settings.vadSilenceThreshold ? 'success.main' : 'text.secondary',
                width: 32,
                flexShrink: 0,
                textAlign: 'right',
              }}
            >
              {(voiceInput.volume * 100).toFixed(0)}%
            </Typography>

            {/* Center: Volume bar - always visible, opacity changes */}
            <Box sx={{
              flex: 1,
              height: 8,
              bgcolor: 'rgba(0,0,0,0.3)',
              borderRadius: 1,
              overflow: 'hidden',
              position: 'relative',
              opacity: voiceInput.isListening ? 1 : 0.3,
            }}>
              {/* Volume bar */}
              <Box sx={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: `${Math.min(voiceInput.volume * 200, 100)}%`,
                bgcolor: voiceInput.volume > settings.vadSilenceThreshold
                  ? voiceInput.isRecording ? 'error.main' : 'success.main'
                  : 'grey.600',
                borderRadius: 1,
              }} />
              {/* Threshold marker */}
              <Box sx={{
                position: 'absolute',
                left: `${settings.vadSilenceThreshold * 200}%`,
                top: 0,
                bottom: 0,
                width: 2,
                bgcolor: 'warning.main',
              }} />
            </Box>

            {/* Countdown indicator - single element, only color changes */}
            <Typography
              variant="caption"
              sx={{
                width: 36,
                flexShrink: 0,
                textAlign: 'right',
                fontFamily: 'monospace',
                fontSize: '0.7rem',
                fontWeight: 600,
                color: voiceInput.isRecording && voiceInput.silenceTime > 0 ? 'warning.main' : 'text.secondary',
              }}
            >
              {(voiceInput.isRecording && voiceInput.silenceTime > 0
                ? (settings.vadSilenceDuration - voiceInput.silenceTime) / 1000
                : settings.vadSilenceDuration / 1000
              ).toFixed(1)}s
            </Typography>

            {/* Right: Controls */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexShrink: 0 }} onClick={(e) => e.stopPropagation()}>
              {/* TTS toggle */}
              {ttsServiceAvailable && (
                <Tooltip title={settings.ttsEnabled ? "TTS on" : "TTS off"}>
                  <IconButton
                    onClick={() => saveSettings({ ...settings, ttsEnabled: !settings.ttsEnabled })}
                    size="small"
                    sx={{
                      p: 0.5,
                      color: settings.ttsEnabled ? 'success.main' : 'grey.500'
                    }}
                  >
                    {settings.ttsEnabled ? <VolumeUpIcon fontSize="small" /> : <VolumeOffIcon fontSize="small" />}
                  </IconButton>
                </Tooltip>
              )}

              {/* Listen toggle */}
              <Tooltip title={voiceInput.isListening ? "Stop" : "Start"}>
                <IconButton
                  onClick={() => voiceInput.toggleListening()}
                  size="small"
                  sx={{
                    p: 0.5,
                    bgcolor: voiceInput.isListening ? 'error.main' : 'success.main',
                    color: 'white',
                    '&:hover': {
                      bgcolor: voiceInput.isListening ? 'error.dark' : 'success.dark',
                    }
                  }}
                >
                  {voiceInput.isListening ? <StopIcon fontSize="small" /> : <MicIcon fontSize="small" />}
                </IconButton>
              </Tooltip>

              {/* Expand button */}
              <IconButton
                size="small"
                sx={{ p: 0.5, ml: 0.5 }}
                onClick={() => setVoicePanelExpanded(!voicePanelExpanded)}
              >
                {voicePanelExpanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
              </IconButton>
            </Box>
          </Paper>

          {/* Expanded Panel */}
          <Collapse in={voicePanelExpanded}>
            <Paper sx={{
              mt: 0.5,
              p: 1.5,
              borderRadius: 1,
              bgcolor: 'background.paper',
              border: '1px solid rgba(255, 255, 255, 0.1)',
            }}>
              {/* Full Audio Level Visualization */}
              <Box sx={{ position: 'relative', height: 32, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1, overflow: 'hidden', mb: 1.5 }}>
                {/* Current volume bar */}
                <Box sx={{
                  position: 'absolute',
                  left: 0,
                  top: 4,
                  bottom: 4,
                  width: `${Math.min(voiceInput.volume * 200, 100)}%`,
                  bgcolor: voiceInput.volume > settings.vadSilenceThreshold
                    ? voiceInput.isRecording ? 'error.main' : 'success.main'
                    : 'grey.600',
                  borderRadius: 1,
                  transition: 'width 0.05s ease-out',
                }} />

                {/* Threshold line */}
                <Box sx={{
                  position: 'absolute',
                  left: `${settings.vadSilenceThreshold * 200}%`,
                  top: 0,
                  bottom: 0,
                  width: 2,
                  bgcolor: 'warning.main',
                  zIndex: 2,
                }} />

                {/* Volume % and status */}
                <Box sx={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  px: 1,
                  zIndex: 3,
                }}>
                  <Typography variant="caption" sx={{ color: 'white', fontSize: '0.7rem' }}>
                    {voiceInput.volume > settings.vadSilenceThreshold
                      ? voiceInput.isRecording ? 'SPEAKING' : 'VOICE'
                      : 'SILENCE'}
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'white', fontSize: '0.7rem', fontWeight: 600 }}>
                    {(voiceInput.volume * 100).toFixed(0)}%
                  </Typography>
                </Box>
              </Box>

              {/* Settings Grid */}
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
                {/* Voice Threshold */}
                <Box sx={{ minWidth: 140, flex: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                      Threshold
                    </Typography>
                    <Typography variant="caption" sx={{ fontWeight: 600, color: 'primary.main', fontSize: '0.7rem' }}>
                      {(settings.vadSilenceThreshold * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                  <Slider
                    size="small"
                    value={settings.vadSilenceThreshold}
                    onChange={(_: Event, value: number | number[]) => saveSettings({ ...settings, vadSilenceThreshold: value as number })}
                    min={0.01}
                    max={0.30}
                    step={0.01}
                    sx={{ py: 0.5 }}
                  />
                </Box>

                {/* Silence Duration */}
                <Box sx={{ minWidth: 140, flex: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                      Silence Wait
                    </Typography>
                    <Typography variant="caption" sx={{ fontWeight: 600, color: 'primary.main', fontSize: '0.7rem' }}>
                      {(settings.vadSilenceDuration / 1000).toFixed(1)}s
                    </Typography>
                  </Box>
                  <Slider
                    size="small"
                    value={settings.vadSilenceDuration}
                    onChange={(_: Event, value: number | number[]) => saveSettings({ ...settings, vadSilenceDuration: value as number })}
                    min={500}
                    max={5000}
                    step={100}
                    sx={{ py: 0.5 }}
                  />
                </Box>

                {/* Speech Filter */}
                <Box sx={{ minWidth: 140, flex: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                      Speech Filter
                    </Typography>
                    <Typography variant="caption" sx={{ fontWeight: 600, color: 'primary.main', fontSize: '0.7rem' }}>
                      {(settings.noSpeechThreshold * 100).toFixed(0)}%
                      {voiceInput.lastNoSpeechProb !== null && (
                        <span style={{ color: voiceInput.lastNoSpeechProb > settings.noSpeechThreshold ? '#f44336' : '#4caf50' }}>
                          {' '}({(voiceInput.lastNoSpeechProb * 100).toFixed(0)}%)
                        </span>
                      )}
                    </Typography>
                  </Box>
                  <Slider
                    size="small"
                    value={settings.noSpeechThreshold}
                    onChange={(_: Event, value: number | number[]) => saveSettings({ ...settings, noSpeechThreshold: value as number })}
                    min={0.1}
                    max={0.95}
                    step={0.05}
                    sx={{ py: 0.5 }}
                  />
                </Box>

                {/* STT Model */}
                <Box sx={{ minWidth: 100 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5, fontSize: '0.7rem' }}>
                    STT Model
                  </Typography>
                  <Select
                    size="small"
                    value={settings.sttModel}
                    onChange={(e) => saveSettings({ ...settings, sttModel: e.target.value })}
                    sx={{
                      fontSize: '0.7rem',
                      height: 26,
                      '& .MuiSelect-select': { py: 0.3, px: 1 }
                    }}
                  >
                    <MenuItem value="tiny">Tiny</MenuItem>
                    <MenuItem value="base">Base</MenuItem>
                    <MenuItem value="small">Small</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="large-v3">Large</MenuItem>
                    <MenuItem value="distil-large-v3.5-ct2">Distil-Large</MenuItem>
                  </Select>
                </Box>
              </Box>

              {/* Error display */}
              {voiceInput.error && (
                <Typography variant="caption" color="error" sx={{ mt: 1, display: 'block' }}>
                  {voiceInput.error}
                </Typography>
              )}
            </Paper>
          </Collapse>
        </Box>
      )}

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
