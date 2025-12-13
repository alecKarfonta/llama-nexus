/**
 * API Type Definitions for LlamaCPP Management Frontend
 * Based on Phase 1 specifications from FRONTEND_ROADMAP.md
 */

// Model Management Types
export interface ModelInfo {
  id: number | string;
  name: string;
  variant: string;
  size: number; // Size in bytes
  status: 'available' | 'downloading' | 'loading' | 'error' | 'running' | 'stopped' | 'archived';
  downloadProgress?: number; // 0-100 percentage
  repositoryId?: string; // HuggingFace repository ID
  contextLength?: number;
  parameters?: string; // e.g., "30B", "7B"
  quantization?: string; // e.g., "Q4_K_M", "Q8_0"
  description?: string;
  license?: string;
  vramRequired?: number; // GB required
  lastModified?: Date;
  // Additional fields for running models
  framework?: 'transformers' | 'llama.cpp' | 'vllm' | 'ggml' | 'onnx';
  port?: number;
  latency?: string;
  memory?: string;
  path?: string;
  // Local file information
  localPath?: string; // Relative path to the local file
  filename?: string; // Actual filename on disk
  // Archive information
  isArchived?: boolean;
  archivedAt?: Date;
  archiveRating?: number; // 1-5 stars
  archiveNotes?: string;
  performanceNotes?: string;
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

export interface ModelArchiveRequest {
  modelId: string | number;
  rating?: number; // 1-5 stars
  notes?: string;
  performanceNotes?: string;
  deleteLocalFiles?: boolean;
}

export interface ModelArchiveResponse {
  success: boolean;
  modelId: string | number;
  archivedAt: string;
  filesDeleted?: string[];
  sizeFreed?: number;
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
  content: string | Array<{type: string; text?: string; image_url?: {url: string}}>;
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  // Reasoning model support (e.g., DeepSeek R1, QwQ)
  reasoning_content?: string;
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
      reasoning_content?: string;
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

// Conversation Storage Types
export interface Conversation {
  id: string;
  title: string;
  model?: string;
  created_at: string;
  updated_at: string;
  is_archived: boolean;
  messages: ConversationMessage[];
  metadata?: Record<string, any>;
}

export interface ConversationMessage {
  id: number;
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  reasoning_content?: string;
  created_at: string;
  metadata?: Record<string, any>;
}

export interface ConversationListItem {
  id: string;
  title: string;
  model?: string;
  created_at: string;
  updated_at: string;
  is_archived: boolean;
  message_count: number;
  last_message?: string;
}

export interface ConversationListResponse {
  conversations: ConversationListItem[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface ConversationStats {
  active_conversations: number;
  archived_conversations: number;
  total_conversations: number;
  total_messages: number;
  models_used: { model: string; count: number }[];
  first_conversation?: string;
}

// VRAM Estimation Types
export interface VRAMEstimate {
  model_vram_gb: number;
  kv_cache_gb: number;
  total_vram_gb: number;
  breakdown: {
    weights: number;
    kv_cache: number;
    compute_buffer: number;
    overhead: number;
  };
  recommendation: string;
  fits_in_vram: boolean;
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

// Prompt Library Types
export interface PromptTemplate {
  id: string;
  name: string;
  description?: string;
  content: string;
  category: string;
  tags: string[];
  variables: string[];
  is_system_prompt: boolean;
  is_favorite: boolean;
  use_count: number;
  created_at: string;
  updated_at: string;
  created_by?: string;
  metadata: Record<string, any>;
}

export interface PromptCategory {
  id: string;
  name: string;
  description?: string;
  color: string;
  icon: string;
  parent_id?: string;
  sort_order: number;
  prompt_count: number;
}

export interface PromptVersion {
  id: number;
  prompt_id: string;
  version: number;
  content: string;
  change_note?: string;
  created_at: string;
}

export interface PromptListResponse {
  prompts: PromptTemplate[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface PromptLibraryStats {
  total_prompts: number;
  system_prompts: number;
  favorites: number;
  total_uses: number;
  categories_used: number;
}

// Model Registry Types
export interface CachedModel {
  id: string;
  repo_id: string;
  name: string;
  description?: string;
  author?: string;
  downloads: number;
  likes: number;
  tags: string[];
  model_type?: string;
  license?: string;
  last_modified?: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, any>;
}

export interface ModelVariant {
  id: number;
  model_id: string;
  filename: string;
  quantization: string;
  size_bytes?: number;
  vram_required_mb?: number;
  quality_score?: number;
  speed_score?: number;
  created_at: string;
}

export interface ModelUsageStats {
  model_id: string;
  variant?: string;
  load_count: number;
  inference_count: number;
  total_tokens_generated: number;
  total_inference_time_ms: number;
  last_used?: string;
  name?: string;
  repo_id?: string;
}

export interface ModelRating {
  id: number;
  model_id: string;
  variant?: string;
  rating: number;
  notes?: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface ModelRegistryStats {
  cached_models: number;
  total_variants: number;
  total_loads: number;
  total_inferences: number;
  rated_models: number;
}

export interface CachedModelListResponse {
  models: CachedModel[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}
