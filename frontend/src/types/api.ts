/**
 * API Type Definitions for LlamaCPP Management Frontend
 * Based on Phase 1 specifications from FRONTEND_ROADMAP.md
 */

// Model Management Types
export interface ModelInfo {
  name: string;
  variant: string;
  size: number; // Size in bytes
  status: 'available' | 'downloading' | 'loading' | 'error';
  downloadProgress?: number; // 0-100 percentage
  repositoryId?: string; // HuggingFace repository ID
  contextLength?: number;
  parameters?: string; // e.g., "30B", "7B"
  quantization?: string; // e.g., "Q4_K_M", "Q8_0"
  description?: string;
  license?: string;
  vramRequired?: number; // GB required
  lastModified?: Date;
}

export interface ModelDownload {
  modelId: string;
  progress: number; // 0-100
  status: 'queued' | 'downloading' | 'completed' | 'failed' | 'cancelled';
  totalSize: number; // bytes
  downloadedSize: number; // bytes
  speed: number; // bytes per second
  eta: number; // seconds
  error?: string;
  partsInfo?: {
    total: number;
    completed: number;
    current: string;
  };
}

// Service Configuration Types
export interface ServiceConfig {
  model: {
    name: string;
    variant: string;
    contextSize: number;
    gpuLayers: number;
  };
  sampling: {
    temperature: number;
    topP: number;
    topK: number;
    minP: number;
    repeatPenalty: number;
    repeatLastN: number;
    frequencyPenalty: number;
    presencePenalty: number;
    // DRY sampling parameters
    dryMultiplier: number;
    dryBase: number;
    dryAllowedLength: number;
    dryPenaltyLastN: number;
  };
  performance: {
    threads: number;
    batchSize: number;
    ubatchSize: number;
    numKeep: number;
    numPredict: number;
  };
  server: {
    host: string;
    port: number;
    apiKey: string;
  };
}

// Resource Monitoring Types
export interface ResourceMetrics {
  cpuUsage: number; // 0-100 percentage
  memoryUsage: number; // 0-100 percentage
  memoryTotal: number; // bytes
  memoryUsed: number; // bytes
  gpuUsage: number; // 0-100 percentage
  vramUsage: number; // 0-100 percentage
  vramTotal: number; // bytes
  vramUsed: number; // bytes
  timestamp: Date;
  requestRate?: number; // requests per second
  responseTime?: number; // milliseconds
  tokensPerSecond?: number;
}

// Service Status Types
export interface ServiceStatus {
  health: 'healthy' | 'degraded' | 'unhealthy' | 'starting' | 'stopped';
  uptime: number; // seconds
  version: string;
  modelLoaded: boolean;
  modelName?: string;
  lastError?: string;
  timestamp: Date;
  endpoints: {
    chat: boolean;
    completions: boolean;
    embeddings: boolean;
    models: boolean;
  };
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: Date;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  page: number;
  pageSize: number;
  total: number;
  hasNext: boolean;
  hasPrev: boolean;
}

// Configuration Preset Types
export interface ConfigurationPreset {
  id: string;
  name: string;
  description: string;
  category: 'coding' | 'creative' | 'reasoning' | 'custom';
  config: Partial<ServiceConfig>;
  isDefault: boolean;
  createdAt: Date;
  updatedAt: Date;
}

// Webhook/Event Types for real-time updates
export interface WebSocketMessage {
  type: 'metrics' | 'status' | 'download' | 'error';
  payload: any;
  timestamp: Date;
}

export interface MetricsUpdate extends WebSocketMessage {
  type: 'metrics';
  payload: ResourceMetrics;
}

export interface StatusUpdate extends WebSocketMessage {
  type: 'status';
  payload: ServiceStatus;
}

export interface DownloadUpdate extends WebSocketMessage {
  type: 'download';
  payload: ModelDownload;
}

// Error Types
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: Date;
}

// Request Types
export interface ModelDownloadRequest {
  repositoryId: string; // e.g., "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
  filename: string; // e.g., "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
  priority?: 'low' | 'normal' | 'high';
}

export interface ConfigUpdateRequest {
  config: Partial<ServiceConfig>;
  restartService?: boolean;
  validateOnly?: boolean;
}

export interface ServiceActionRequest {
  action: 'start' | 'stop' | 'restart' | 'reload';
  config?: Partial<ServiceConfig>;
}

// Chat Completion Types (OpenAI-compatible)
export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  tokensPerSecond?: number; // Generation speed in tokens per second
}

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  tools?: Tool[];
  tool_choice?: 'none' | 'auto' | { type: 'function'; function: { name: string } };
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  stream?: boolean;
  stop?: string | string[];
  frequency_penalty?: number;
  presence_penalty?: number;
  repeat_penalty?: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: ChatMessage;
    finish_reason: 'stop' | 'length' | 'content_filter' | 'tool_calls' | null;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
      tool_calls?: ToolCall[];
    };
    finish_reason: 'stop' | 'length' | 'content_filter' | 'tool_calls' | null;
  }>;
}

// Function/Tool Calling Types
export interface FunctionParameter {
  type: string;
  description?: string;
  enum?: string[];
  items?: FunctionParameter;
  properties?: Record<string, FunctionParameter>;
  required?: string[];
}

export interface FunctionDefinition {
  name: string;
  description: string;
  parameters: FunctionParameter;
}

export interface Tool {
  type: 'function';
  function: FunctionDefinition;
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface ToolMessage extends ChatMessage {
  role: 'tool';
  tool_call_id: string;
  content: string;
}

// Token Usage Tracking Types
export interface TokenUsageData {
  modelId: string;
  modelName?: string;
  promptTokens: number;
  completionTokens: number;
  lastUsed: string;
  requests: number;
}

// Built-in example tools for testing
export interface WeatherQuery {
  location: string;
  unit?: 'celsius' | 'fahrenheit';
}

export interface CalculatorQuery {
  expression: string;
}

export interface CodeExecutionQuery {
  language: 'python' | 'javascript' | 'bash';
  code: string;
}

// LlamaCPP Commit Management Types
export interface LlamaCppCommit {
  tag: string;
  name: string;
  published_at: string;
  body: string;
  is_current: boolean;
}

export interface LlamaCppCommitsResponse {
  current_commit: string;
  releases: LlamaCppCommit[];
  recent_commits: LlamaCppCommit[];
}
