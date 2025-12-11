/**
 * API Service Layer for LlamaCPP Management Frontend
 * Handles all HTTP communication with the llamacpp backend
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import { settingsManager } from '@/utils/settings';
import type {
  ModelInfo,
  ModelDownloadRequest,
  ServiceConfig,
  ConfigUpdateRequest,
  ServiceStatus,
  ResourceMetrics,
  ServiceActionRequest,
  ApiResponse,
  ModelDownload,
  ConfigurationPreset,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  TokenUsageData,
  LlamaCppCommit,
  LlamaCppCommitsResponse
} from '@/types/api';

class ApiService {
  private client: AxiosInstance;
  private backendClient: AxiosInstance;
  private baseURL: string;
  private backendBaseURL: string;

  private getAuthHeaders() {
    const apiKey = settingsManager.getApiKey();
    return apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {};
  }

  constructor() {
    // Use env or relative base so dev proxy can avoid CORS; production can set env
    // In development, use empty string to let the dev server proxy handle it
    // In production, use the environment variable or default to the current origin
    const isDevelopment = import.meta.env.DEV === true;
    
    // Try to get the API base URL from various sources
    let apiBaseUrl = '';
    
    // 1. Try import.meta.env (Vite's way of exposing env vars)
    if (import.meta.env?.VITE_API_BASE_URL) {
      apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
    } 
    // 2. Try window.__ENV__ (sometimes used for runtime env injection)
    else if ((window as any).__ENV__?.VITE_API_BASE_URL) {
      apiBaseUrl = (window as any).__ENV__.VITE_API_BASE_URL;
    }
    // 3. In production, if no env var, use the current origin
    else if (!isDevelopment) {
      apiBaseUrl = window.location.origin;
    }
    
    this.baseURL = isDevelopment ? '' : apiBaseUrl;
    // Backend base URL (management API)
    let backendBaseUrl = '';
    if (import.meta.env?.VITE_BACKEND_URL) {
      backendBaseUrl = import.meta.env.VITE_BACKEND_URL;
    } else if ((window as any).__ENV__?.VITE_BACKEND_URL) {
      backendBaseUrl = (window as any).__ENV__.VITE_BACKEND_URL;
    } else if (!isDevelopment) {
      backendBaseUrl = apiBaseUrl; // fallback
    }
    this.backendBaseURL = backendBaseUrl;
    console.log('API Base URL:', this.baseURL || 'using relative URLs (proxy mode)');

    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      }
    });

    this.backendClient = axios.create({
      baseURL: this.backendBaseURL || this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      }
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor for main client
    this.client.interceptors.request.use(
      (config: any) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        // Inject API key dynamically
        const authHeaders = this.getAuthHeaders();
        config.headers = { ...config.headers, ...authHeaders };
        return config;
      },
      (error: any) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Request interceptor for backend client
    this.backendClient.interceptors.request.use(
      (config: any) => {
        console.log(`Backend Request: ${config.method?.toUpperCase()} ${config.url}`);
        // Inject API key dynamically
        const authHeaders = this.getAuthHeaders();
        config.headers = { ...config.headers, ...authHeaders };
        return config;
      },
      (error: any) => {
        console.error('Backend Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    const onResponse = (response: any) => response;
    const onError = (error: AxiosError) => {
      console.error('API Response Error:', error);
      return Promise.reject(this.transformError(error));
    };
    this.client.interceptors.response.use(
      onResponse,
      onError
    );

    this.backendClient.interceptors.response.use(onResponse, onError);
  }

  private transformError(error: AxiosError): Error {
    const data = error.response?.data as any;
    const message = data?.error || data?.detail || error.message || 'Unknown API error';
    return new Error(message);
  }

  // Generic HTTP methods for flexibility
  async get(url: string, config?: any): Promise<any> {
    return await this.backendClient.get(url, config);
  }

  async post(url: string, data?: any, config?: any): Promise<any> {
    return await this.backendClient.post(url, data, config);
  }

  async put(url: string, data?: any, config?: any): Promise<any> {
    return await this.backendClient.put(url, data, config);
  }

  async delete(url: string, config?: any): Promise<any> {
    return await this.backendClient.delete(url, config);
  }

  async patch(url: string, data?: any, config?: any): Promise<any> {
    return await this.backendClient.patch(url, data, config);
  }

  // Normalize backend download record (snake_case) to frontend ModelDownload (camelCase)
  private normalizeModelDownload(raw: any): ModelDownload {
    if (!raw || (typeof raw !== 'object')) {
      throw new Error('Invalid download record');
    }

    const modelId = raw.model_id ?? raw.modelId;
    const status = raw.status as string;
    const progress = raw.progress;
    const totalSize = raw.total_size ?? raw.totalSize;
    const downloadedSize = raw.downloaded_size ?? raw.downloadedSize;
    const speed = raw.speed;
    const eta = raw.eta;

    if (!modelId || typeof modelId !== 'string') {
      throw new Error('Download record missing model_id');
    }
    if (typeof status !== 'string') {
      throw new Error('Download record missing status');
    }
    if (typeof progress !== 'number') {
      throw new Error('Download record missing progress');
    }
    if (typeof totalSize !== 'number') {
      throw new Error('Download record missing total_size');
    }
    if (typeof downloadedSize !== 'number') {
      throw new Error('Download record missing downloaded_size');
    }
    if (typeof speed !== 'number') {
      throw new Error('Download record missing speed');
    }
    if (typeof eta !== 'number') {
      throw new Error('Download record missing eta');
    }

    // Validate and narrow status
    const allowedStatuses = ['queued', 'downloading', 'completed', 'failed', 'cancelled'] as const;
    type AllowedStatus = typeof allowedStatuses[number];
    if (!allowedStatuses.includes(status as any)) {
      throw new Error(`Invalid download status: ${status}`);
    }
    const normalizedStatus: AllowedStatus = status as AllowedStatus;

    const result: ModelDownload = {
      modelId,
      progress,
      status: normalizedStatus,
      totalSize,
      downloadedSize,
      speed,
      eta,
      error: raw.error,
    };

    // Add partsInfo if present (for multi-part downloads)
    const partsInfo = raw.parts_info ?? raw.partsInfo;
    if (partsInfo && typeof partsInfo === 'object') {
      result.partsInfo = {
        total: partsInfo.total ?? 0,
        completed: partsInfo.completed ?? 0,
        current: partsInfo.current ?? '',
      };
    }

    return result;
  }

  // Model Management APIs
  async getModels(): Promise<ModelInfo[]> {
    const response = await this.backendClient.get('/v1/models');
    const apiData = response.data;
    
    // Check if we have data in the expected format from our backend
    if (apiData && Array.isArray(apiData.data)) {
      // Transform the API response to match ModelInfo structure
      return apiData.data.map((model: any) => {
        return {
          name: model.name || '',
          variant: model.variant || 'unknown',
          size: model.size || 0,
          status: model.status || 'available',
          downloadProgress: model.downloadProgress,
          repositoryId: model.repositoryId,
          contextLength: model.contextLength,
          parameters: this.formatParameters(this.extractParameters(model.name)),
          quantization: model.variant,
          description: model.description,
          license: model.license,
          vramRequired: model.vramRequired,
          lastModified: model.lastModified ? new Date(model.lastModified) : undefined,
          // Local file information
          localPath: model.localPath,
          filename: model.filename
        };
      });
    }
    
    // Fallback to empty array if data format is unexpected
    return [];
  }

  // Template management APIs
  async listTemplates(): Promise<{ directory: string; files: string[]; selected: string }> {
    const response = await this.backendClient.get('/v1/templates');
    return (response.data as any).data;
  }

  async getTemplate(filename: string): Promise<{ filename: string; content: string }> {
    const response = await this.backendClient.get(`/v1/templates/${encodeURIComponent(filename)}`);
    return (response.data as any).data;
  }

  async updateTemplate(filename: string, content: string): Promise<void> {
    await this.backendClient.put(`/v1/templates/${encodeURIComponent(filename)}`, { content });
  }

  async selectTemplate(filename: string): Promise<void> {
    await this.backendClient.post('/v1/templates/select', { filename });
  }

  async createTemplate(filename: string, content: string = ''): Promise<{ filename: string }> {
    const response = await this.backendClient.post('/v1/templates', { filename, content });
    return (response.data as any).data;
  }
  
  // Helper to extract parameter count from model name
  private extractParameters(name: string): number | undefined {
    if (!name) return undefined;
    
    // Try to extract parameter count from model name (e.g., "30B", "7B", "1.5B")
    const match = name.match(/(\d+\.?\d*)[Bb]/);
    if (match) {
      const value = parseFloat(match[1]);
      return value * 1000000000; // Convert to actual parameter count
    }
    
    return undefined;
  }

  // Helper to format parameters (e.g., 30B, 7B)
  private formatParameters(params: number | undefined): string {
    if (!params) return '';
    const billions = params / 1000000000;
    return billions >= 1 ? `${Math.round(billions)}B` : `${Math.round(params / 1000000)}M`;
  }
  


  async getCurrentModel(): Promise<any> {
    const response = await this.backendClient.get('/v1/models/current');
    return response.data;
  }

  async listRepoFiles(repoId: string, revision: string = 'main'): Promise<string[]> {
    const response = await this.backendClient.get<ApiResponse<{ files: string[] }>>('/v1/models/repo-files', {
      params: { repo_id: repoId, revision }
    });
    const files = (response.data as any)?.data?.files || [];
    return Array.isArray(files) ? files : [];
  }

  async downloadModel(request: ModelDownloadRequest): Promise<ModelDownload> {
    const response = await this.backendClient.post<ApiResponse<ModelDownload>>('/v1/models/download', request);
    return this.normalizeModelDownload(response.data.data);
  }

  async getModelDownloads(): Promise<ModelDownload[]> {
    const response = await this.backendClient.get<ApiResponse<ModelDownload[]>>('/v1/models/downloads');
    const data = response.data.data || [];
    return (data as any[]).map((item) => this.normalizeModelDownload(item));
  }

  async cancelModelDownload(modelId: string): Promise<void> {
    await this.backendClient.delete(`/v1/models/downloads/${modelId}`);
  }

  async getLocalModelFiles(): Promise<{ files: any[]; total_size: number; total_count: number }> {
    const response = await this.backendClient.get<ApiResponse<any>>('/v1/models/local-files');
    return response.data.data || { files: [], total_size: 0, total_count: 0 };
  }

  async deleteLocalModelFile(filePath: string): Promise<{ deleted_file: string; size_freed: number }> {
    const response = await this.backendClient.delete<ApiResponse<any>>('/v1/models/local-files', {
      params: { file_path: filePath }
    });
    return response.data.data;
  }

  // Service Configuration APIs
  async getServiceConfig(): Promise<ServiceConfig> {
    const response = await this.client.get('/v1/service/config');
    // Backend returns { config, command, editable_fields }
    return (response.data as any).config as ServiceConfig;
  }

  async updateServiceConfig(request: ConfigUpdateRequest): Promise<ServiceConfig> {
    // Backend expects raw config (not wrapped).
    const body = request?.config ?? (request as any);
    const response = await this.client.put('/v1/service/config', body);
    return (response.data as any).config as ServiceConfig;
  }

  async validateServiceConfig(config: Partial<ServiceConfig>): Promise<{ valid: boolean; errors?: string[]; warnings?: string[] }> {
    const response = await this.client.post('/v1/service/config/validate', { config });
    return response.data as any;
  }

  // Service Control APIs
  async getServiceStatus(): Promise<ServiceStatus> {
    try {
      // Use the real backend service status endpoint through nginx proxy
      const statusResponse = await this.client.get('/v1/service/status');
      const data = statusResponse.data;
      
      // Transform backend response to frontend format
      return {
        health: data.running ? 'healthy' : 'stopped',
        uptime: data.uptime || 0,
        version: '1.0.0', // Backend doesn't provide version yet
        modelLoaded: data.running && !!data.model?.name,
        modelName: data.model?.name || undefined,
        timestamp: new Date(),
        endpoints: {
          chat: data.running && !!data.llamacpp_health?.healthy,
          completions: data.running && !!data.llamacpp_health?.healthy,
          embeddings: data.running && !!data.llamacpp_health?.healthy,
          models: data.running,
        },
      };
    } catch (error) {
      // Fallback to basic health check if detailed status fails
      try {
        // health endpoint is proxied in dev via '/api/health' -> backend '/health'
        const healthResponse = await this.client.get('/health');
        const isHealthy = healthResponse.status === 200;
        
        return {
          health: isHealthy ? 'healthy' : 'unhealthy',
          uptime: 0,
          version: 'unknown',
          modelLoaded: false,
          timestamp: new Date(),
          endpoints: {
            chat: isHealthy,
            completions: isHealthy,
            embeddings: false,
            models: isHealthy,
          },
        };
      } catch (fallbackError) {
        return {
          health: 'unhealthy',
          uptime: 0,
          version: 'unknown',
          modelLoaded: false,
          timestamp: new Date(),
          lastError: error instanceof Error ? error.message : 'Service unavailable',
          endpoints: {
            chat: false,
            completions: false,
            embeddings: false,
            models: false,
          },
        };
      }
    }
  }

  async performServiceAction(request: ServiceActionRequest): Promise<void> {
    await this.client.post('/v1/service/action', request);
  }

  // Resource Monitoring APIs
  async getResourceMetrics(): Promise<ResourceMetrics> {
    try {
      // Use real backend metrics endpoint
      const response = await this.client.get('/v1/resources');
      const data = response.data;
      
      // Transform backend response to frontend format
      return {
        cpuUsage: data.cpu?.percent || 0,
        memoryUsage: data.memory?.percent || 0,
        memoryTotal: (data.memory?.total_mb || 0) / 1024, // Convert MB to GB
        memoryUsed: (data.memory?.used_mb || 0) / 1024, // Convert MB to GB
        gpuUsage: data.gpu?.usage_percent || 0,
        vramUsage: data.gpu?.vram_total_mb ? (data.gpu?.vram_used_mb / data.gpu?.vram_total_mb) * 100 : 0,
        vramTotal: (data.gpu?.vram_total_mb || 0) / 1024, // Convert MB to GB
        vramUsed: (data.gpu?.vram_used_mb || 0) / 1024, // Convert MB to GB
        timestamp: new Date(data.timestamp || Date.now()),
        // These metrics aren't available from backend yet, so use reasonable defaults
        requestRate: 0,
        responseTime: 0,
        tokensPerSecond: 0,
      };
    } catch (error) {
      console.error('Failed to get resource metrics:', error);
      throw error;
    }
  }

  async getMetricsHistory(duration: string = '1h'): Promise<ResourceMetrics[]> {
    // TODO: Backend doesn't have historical metrics yet, so for now return current metrics
    // In the future, this should be replaced with real historical data from the backend
    try {
      const currentMetrics = await this.getResourceMetrics();
      
      // Generate a few historical points based on current metrics for now
      const now = new Date();
      const data: ResourceMetrics[] = [];
      const points = duration === '5m' ? 10 : duration === '15m' ? 15 : duration === '1h' ? 12 : 24;
      const intervalMs = duration === '5m' ? 30000 : duration === '15m' ? 60000 : duration === '1h' ? 300000 : 900000;
      
      for (let i = points; i >= 0; i--) {
        const timestamp = new Date(now.getTime() - (i * intervalMs));
        // Add small variations to current metrics to simulate history
        const variation = (Math.random() - 0.5) * 10; // ±5% variation
        data.push({
          ...currentMetrics,
          timestamp,
          cpuUsage: Math.max(0, Math.min(100, currentMetrics.cpuUsage + variation)),
          memoryUsage: Math.max(0, Math.min(100, currentMetrics.memoryUsage + variation * 0.5)),
          gpuUsage: Math.max(0, Math.min(100, currentMetrics.gpuUsage + variation * 0.8)),
          vramUsage: Math.max(0, Math.min(100, currentMetrics.vramUsage + variation * 0.3)),
        });
      }
      
      return data;
    } catch (error) {
      console.error('Failed to get metrics history:', error);
      throw error;
    }
  }

  // Get raw Prometheus metrics from llamacpp
  async getPrometheusMetrics(): Promise<string> {
    // Try direct metrics endpoint first, then fallback to /api/metrics
    try {
      const response = await this.client.get('/metrics', {
        headers: {
          'Accept': 'text/plain'
        }
      });
      return response.data;
    } catch (error) {
      try {
        // Fallback to /api/metrics route
        const response = await this.client.get('/api/metrics', {
          headers: {
            'Accept': 'text/plain'
          }
        });
        return response.data;
      } catch (fallbackError) {
        console.warn('Metrics endpoints not available:', error, fallbackError);
        return '# Metrics not available';
      }
    }
  }



  // Configuration Presets APIs
  async getConfigurationPresets(): Promise<ConfigurationPreset[]> {
    const response = await this.client.get<ApiResponse<ConfigurationPreset[]>>('/v1/config/presets');
    return response.data.data || [];
  }

  async createConfigurationPreset(preset: Omit<ConfigurationPreset, 'id' | 'createdAt' | 'updatedAt'>): Promise<ConfigurationPreset> {
    const response = await this.client.post<ApiResponse<ConfigurationPreset>>('/v1/config/presets', preset);
    return response.data.data!;
  }

  async updateConfigurationPreset(id: string, preset: Partial<ConfigurationPreset>): Promise<ConfigurationPreset> {
    const response = await this.client.put<ApiResponse<ConfigurationPreset>>(`/v1/config/presets/${id}`, preset);
    return response.data.data!;
  }

  async deleteConfigurationPreset(id: string): Promise<void> {
    await this.client.delete(`/v1/config/presets/${id}`);
  }

  // Health Check APIs (using existing llamacpp endpoints)
  async healthCheck(): Promise<{ status: string }> {
    try {
      await this.client.get('/health');
      return { status: 'healthy' };
    } catch (error) {
      return { status: 'unhealthy' };
    }
  }

  // Chat Completion APIs (OpenAI-compatible)
  async createChatCompletion(request: ChatCompletionRequest): Promise<ChatCompletionResponse> {
    const response = await this.client.post<ChatCompletionResponse>('/v1/chat/completions', request);
    return response.data;
  }

  async createChatCompletionStream(request: ChatCompletionRequest): Promise<ReadableStream<ChatCompletionChunk>> {
    // Use fetch API for better streaming support
    const authHeaders = this.getAuthHeaders();
    const response = await fetch(`/v1/chat/completions`, {
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
                  const parsed = JSON.parse(data) as ChatCompletionChunk;
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
  }

  // Token usage tracking
  async getTokenUsage(timeRange: string = '24h'): Promise<TokenUsageData[]> {
    try {
      const response = await this.client.get('/v1/usage/tokens', {
        params: { timeRange }
      });
      return response.data.data || [];
    } catch (error) {
      // Do not fallback to mock data; surface the error so it can be fixed
      throw error;
    }
  }

  async recordTokenUsage(params: {
    model_id: string;
    prompt_tokens: number;
    completion_tokens: number;
    model_name?: string;
    request_id?: string;
    user_id?: string;
    endpoint?: string;
    metadata?: Record<string, any>;
  }): Promise<void> {
    await this.backendClient.post('/v1/usage/tokens/record', params);
  }

  // Utility method to test API connectivity
  async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get available llama.cpp commits and releases
   */
  async getLlamaCppCommits(): Promise<LlamaCppCommitsResponse> {
    const response = await this.backendClient.get<LlamaCppCommitsResponse>('/api/v1/llamacpp/commits');
    return response.data;
  }

  /**
   * Validate a llama.cpp commit exists
   */
  async validateLlamaCppCommit(commitId: string): Promise<{ valid: boolean; error?: string; commit?: any }> {
    const response = await this.backendClient.get(`/api/v1/llamacpp/commits/${commitId}/validate`);
    return response.data;
  }

  /**
   * Apply a specific llama.cpp commit
   */
  async applyLlamaCppCommit(commitId: string): Promise<{ message: string; commit: string; commit_info?: any; requires_rebuild: boolean }> {
    const response = await this.backendClient.post(`/api/v1/llamacpp/commits/${commitId}/apply`);
    return response.data;
  }

  // --- BFCL Benchmark Testing Methods ---

  /**
   * Start BFCL benchmark testing
   */
  async startBfclBenchmark(params: {
    test_category?: string;
    max_examples?: number;
  }): Promise<{ benchmark_id: string; status: string; test_category: string; max_examples: number }> {
    const response = await this.backendClient.post('/api/v1/benchmark/bfcl/start', {}, {
      params: params
    });
    return response.data;
  }

  /**
   * Get BFCL benchmark status and results
   */
  async getBfclBenchmarkStatus(benchmarkId?: string): Promise<{
    status: 'starting' | 'running' | 'completed' | 'failed' | 'cancelled';
    test_category?: string;
    max_examples?: number;
    started_at?: string;
    completed_at?: string;
    progress?: number;
    total?: number;
    current_test?: string;
    results?: {
      total: number;
      correct: number;
      accuracy: number;
      errors: string[];
    };
    error?: string;
  }> {
    // For now, we'll use the list endpoint and get the most recent benchmark
    // In the future, we can track benchmark IDs properly
    const response = await this.backendClient.get('/api/v1/benchmark/bfcl');
    const benchmarks = response.data;
    
    // Get the most recent benchmark
    const benchmarkIds = Object.keys(benchmarks);
    if (benchmarkIds.length === 0) {
      return { status: 'starting' };
    }
    
    const mostRecent = benchmarkIds.reduce((latest, current) => {
      const latestTime = new Date(benchmarks[latest].started_at || 0);
      const currentTime = new Date(benchmarks[current].started_at || 0);
      return currentTime > latestTime ? current : latest;
    });
    
    return benchmarks[mostRecent];
  }

  /**
   * List all BFCL benchmarks
   */
  async listBfclBenchmarks(): Promise<Record<string, any>> {
    const response = await this.backendClient.get('/api/v1/benchmark/bfcl');
    return response.data;
  }

  /**
   * Stop a running BFCL benchmark
   */
  async stopBfclBenchmark(benchmarkId: string): Promise<{ status: string; benchmark_id: string }> {
    const response = await this.backendClient.delete(`/api/v1/benchmark/bfcl/${benchmarkId}`);
    return response.data;
  }

  /**
   * Clear completed benchmarks
   */
  async clearCompletedBenchmarks(): Promise<{ status: string }> {
    const response = await this.backendClient.delete('/api/v1/benchmark/bfcl');
    return response.data;
  }

  /**
   * Rebuild llama.cpp containers
   */
  async rebuildLlamaCpp(): Promise<{ message: string; stdout: string; stderr: string }> {
    const response = await this.backendClient.post('/api/v1/llamacpp/rebuild');
    return response.data;
  }

  // --- VRAM Estimation Methods ---

  /**
   * Estimate VRAM requirements for a model configuration
   */
  async estimateVram(params: {
    parameters?: number;
    quantization?: string;
    context_size?: number;
    batch_size?: number;
    num_layers?: number;
    head_dim?: number;
    num_kv_heads?: number;
    kv_cache_type?: string;
  }): Promise<{
    estimate: {
      model_weights_gb: number;
      kv_cache_gb: number;
      activation_memory_gb: number;
      cuda_overhead_gb: number;
      total_vram_gb: number;
    };
    inputs: {
      parameters_b: number;
      quantization: string;
      quantization_bits: number;
      context_size: number;
      batch_size: number;
      num_layers: number;
      head_dim: number;
      num_kv_heads: number;
      kv_cache_type: string;
    };
    recommendation: string;
  }> {
    const response = await this.backendClient.post('/api/v1/estimate/vram', params);
    return response.data;
  }

  // --- Conversation Storage Methods ---

  /**
   * List all conversations with optional filtering
   */
  async listConversations(params?: {
    limit?: number;
    offset?: number;
    search?: string;
    tags?: string[];
  }): Promise<{
    conversations: Array<{
      id: string;
      title: string;
      created_at: string;
      updated_at: string;
      model?: string;
      tags: string[];
      message_count: number;
    }>;
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
  }> {
    const response = await this.backendClient.get('/api/v1/conversations', {
      params: {
        limit: params?.limit || 50,
        offset: params?.offset || 0,
        search: params?.search,
        tags: params?.tags?.join(','),
      },
    });
    return response.data;
  }

  /**
   * Create a new conversation
   */
  async createConversation(data: {
    title?: string;
    messages?: Array<{
      role: string;
      content: string;
      reasoning_content?: string;
      tool_calls?: any[];
      tool_call_id?: string;
    }>;
    model?: string;
    settings?: Record<string, any>;
    tags?: string[];
  }): Promise<{
    id: string;
    title: string;
    messages: any[];
    created_at: string;
    updated_at: string;
    model?: string;
    settings?: Record<string, any>;
    tags: string[];
  }> {
    const response = await this.backendClient.post('/api/v1/conversations', data);
    return response.data;
  }

  /**
   * Get a specific conversation by ID
   */
  async getConversation(conversationId: string): Promise<{
    id: string;
    title: string;
    messages: Array<{
      role: string;
      content: string;
      timestamp: string;
      reasoning_content?: string;
      tool_calls?: any[];
      tool_call_id?: string;
    }>;
    created_at: string;
    updated_at: string;
    model?: string;
    settings?: Record<string, any>;
    tags: string[];
  }> {
    const response = await this.backendClient.get(`/api/v1/conversations/${conversationId}`);
    return response.data;
  }

  /**
   * Update an existing conversation
   */
  async updateConversation(conversationId: string, data: {
    title?: string;
    messages?: any[];
    model?: string;
    settings?: Record<string, any>;
    tags?: string[];
  }): Promise<{
    id: string;
    title: string;
    messages: any[];
    created_at: string;
    updated_at: string;
    model?: string;
    settings?: Record<string, any>;
    tags: string[];
  }> {
    const response = await this.backendClient.put(`/api/v1/conversations/${conversationId}`, data);
    return response.data;
  }

  /**
   * Add a message to an existing conversation
   */
  async addMessageToConversation(conversationId: string, message: {
    role: string;
    content: string;
    reasoning_content?: string;
    tool_calls?: any[];
    tool_call_id?: string;
  }): Promise<{
    id: string;
    title: string;
    messages: any[];
    created_at: string;
    updated_at: string;
  }> {
    const response = await this.backendClient.post(
      `/api/v1/conversations/${conversationId}/messages`,
      message
    );
    return response.data;
  }

  /**
   * Delete a conversation
   */
  async deleteConversation(conversationId: string): Promise<{ status: string; deleted: string }> {
    const response = await this.backendClient.delete(`/api/v1/conversations/${conversationId}`);
    return response.data;
  }

  /**
   * Export a conversation in the specified format
   */
  async exportConversation(conversationId: string, format: 'json' | 'markdown' = 'json'): Promise<string> {
    const response = await this.backendClient.get(
      `/api/v1/conversations/${conversationId}/export`,
      { params: { format }, responseType: 'text' }
    );
    return response.data;
  }

  /**
   * Search conversations by content
   */
  async searchConversations(query: string): Promise<{
    results: Array<{
      id: string;
      title: string;
      created_at: string;
      updated_at: string;
      model?: string;
      tags: string[];
      message_count: number;
    }>;
  }> {
    const response = await this.backendClient.get('/api/v1/conversations/search', {
      params: { query },
    });
    return response.data;
  }

  // ============================================================================
  // Prompt Library API
  // ============================================================================

  /**
   * Get prompt library statistics
   */
  async getPromptStats(): Promise<{
    total_prompts: number;
    system_prompts: number;
    favorites: number;
    total_uses: number;
    categories_used: number;
  }> {
    const response = await this.backendClient.get('/api/v1/prompts/stats');
    return response.data;
  }

  /**
   * List prompt categories
   */
  async listPromptCategories(): Promise<{
    categories: Array<{
      id: string;
      name: string;
      description?: string;
      color: string;
      icon: string;
      parent_id?: string;
      sort_order: number;
      prompt_count: number;
    }>;
  }> {
    const response = await this.backendClient.get('/api/v1/prompts/categories');
    return response.data;
  }

  /**
   * Create a new prompt category
   */
  async createPromptCategory(data: {
    name: string;
    description?: string;
    color?: string;
    icon?: string;
    parent_id?: string;
  }): Promise<{
    id: string;
    name: string;
    description?: string;
    color: string;
    icon: string;
    parent_id?: string;
  }> {
    const response = await this.backendClient.post('/api/v1/prompts/categories', data);
    return response.data;
  }

  /**
   * List prompts with optional filtering
   */
  async listPrompts(params?: {
    category?: string;
    search?: string;
    is_system_prompt?: boolean;
    is_favorite?: boolean;
    limit?: number;
    offset?: number;
    order_by?: string;
    order_dir?: 'ASC' | 'DESC';
  }): Promise<{
    prompts: Array<{
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
    }>;
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
  }> {
    const response = await this.backendClient.get('/api/v1/prompts', { params });
    return response.data;
  }

  /**
   * Create a new prompt template
   */
  async createPrompt(data: {
    name: string;
    content: string;
    description?: string;
    category?: string;
    tags?: string[];
    is_system_prompt?: boolean;
    metadata?: Record<string, any>;
  }): Promise<{
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
  }> {
    const response = await this.backendClient.post('/api/v1/prompts', data);
    return response.data;
  }

  /**
   * Get a prompt by ID
   */
  async getPrompt(promptId: string): Promise<{
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
    metadata: Record<string, any>;
  }> {
    const response = await this.backendClient.get(`/api/v1/prompts/${promptId}`);
    return response.data;
  }

  /**
   * Update a prompt
   */
  async updatePrompt(promptId: string, data: {
    name?: string;
    content?: string;
    description?: string;
    category?: string;
    tags?: string[];
    is_system_prompt?: boolean;
    is_favorite?: boolean;
    metadata?: Record<string, any>;
    change_note?: string;
  }): Promise<{
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
  }> {
    const response = await this.backendClient.put(`/api/v1/prompts/${promptId}`, data);
    return response.data;
  }

  /**
   * Delete a prompt
   */
  async deletePrompt(promptId: string): Promise<{ status: string; prompt_id: string }> {
    const response = await this.backendClient.delete(`/api/v1/prompts/${promptId}`);
    return response.data;
  }

  /**
   * Get prompt version history
   */
  async getPromptVersions(promptId: string): Promise<{
    versions: Array<{
      id: number;
      prompt_id: string;
      version: number;
      content: string;
      change_note?: string;
      created_at: string;
    }>;
  }> {
    const response = await this.backendClient.get(`/api/v1/prompts/${promptId}/versions`);
    return response.data;
  }

  /**
   * Restore a prompt to a specific version
   */
  async restorePromptVersion(promptId: string, version: number): Promise<{
    id: string;
    name: string;
    content: string;
    updated_at: string;
  }> {
    const response = await this.backendClient.post(`/api/v1/prompts/${promptId}/restore/${version}`);
    return response.data;
  }

  /**
   * Render a prompt with variables
   */
  async renderPrompt(promptId: string, variables: Record<string, string>): Promise<{ rendered: string }> {
    const response = await this.backendClient.post(`/api/v1/prompts/${promptId}/render`, { variables });
    return response.data;
  }

  /**
   * Export prompts to JSON
   */
  async exportPrompts(promptIds?: string[]): Promise<{ data: string }> {
    const response = await this.backendClient.post('/api/v1/prompts/export', { prompt_ids: promptIds });
    return response.data;
  }

  /**
   * Import prompts from JSON
   */
  async importPrompts(data: string, overwrite?: boolean): Promise<{
    imported: number;
    skipped: number;
    errors: string[];
  }> {
    const response = await this.backendClient.post('/api/v1/prompts/import', { data, overwrite });
    return response.data;
  }

  // ============================================================================
  // Model Registry API
  // ============================================================================

  /**
   * Get model registry statistics
   */
  async getRegistryStats(): Promise<{
    cached_models: number;
    total_variants: number;
    total_loads: number;
    total_inferences: number;
    rated_models: number;
  }> {
    const response = await this.backendClient.get('/api/v1/registry/stats');
    return response.data;
  }

  /**
   * List cached models
   */
  async listCachedModels(params?: {
    limit?: number;
    offset?: number;
    search?: string;
    model_type?: string;
  }): Promise<{
    models: Array<{
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
    }>;
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
  }> {
    const response = await this.backendClient.get('/api/v1/registry/models', { params });
    return response.data;
  }

  /**
   * Cache a model's metadata
   */
  async cacheModel(data: {
    repo_id: string;
    name: string;
    description?: string;
    author?: string;
    downloads?: number;
    likes?: number;
    tags?: string[];
    model_type?: string;
    license?: string;
    last_modified?: string;
    metadata?: Record<string, any>;
  }): Promise<{ status: string; model_id: string }> {
    const response = await this.backendClient.post('/api/v1/registry/models', data);
    return response.data;
  }

  /**
   * Get a cached model
   */
  async getCachedModel(repoId: string): Promise<{
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
    metadata: Record<string, any>;
  }> {
    const response = await this.backendClient.get(`/api/v1/registry/models/${encodeURIComponent(repoId)}`);
    return response.data;
  }

  /**
   * Delete a cached model
   */
  async deleteCachedModel(repoId: string): Promise<{ status: string; repo_id: string }> {
    const response = await this.backendClient.delete(`/api/v1/registry/models/${encodeURIComponent(repoId)}`);
    return response.data;
  }

  /**
   * Get model variants
   */
  async getModelVariants(repoId: string): Promise<{
    variants: Array<{
      id: number;
      model_id: string;
      filename: string;
      quantization: string;
      size_bytes?: number;
      vram_required_mb?: number;
      quality_score?: number;
      speed_score?: number;
    }>;
  }> {
    const response = await this.backendClient.get(`/api/v1/registry/models/${encodeURIComponent(repoId)}/variants`);
    return response.data;
  }

  /**
   * Record model load
   */
  async recordModelLoad(repoId: string, variant?: string): Promise<{ status: string }> {
    const response = await this.backendClient.post(
      `/api/v1/registry/models/${encodeURIComponent(repoId)}/usage/load`,
      null,
      { params: { variant } }
    );
    return response.data;
  }

  /**
   * Get model usage stats
   */
  async getModelUsageStats(repoId?: string): Promise<{
    usage: Array<{
      model_id: string;
      variant?: string;
      load_count: number;
      inference_count: number;
      total_tokens_generated: number;
      last_used?: string;
      name?: string;
      repo_id?: string;
    }>;
  }> {
    const response = await this.backendClient.get('/api/v1/registry/usage', {
      params: { repo_id: repoId },
    });
    return response.data;
  }

  /**
   * Get most used models
   */
  async getMostUsedModels(limit?: number): Promise<{
    models: Array<{
      id: string;
      repo_id: string;
      name: string;
      model_type?: string;
      total_loads: number;
      total_inferences: number;
      last_used?: string;
    }>;
  }> {
    const response = await this.backendClient.get('/api/v1/registry/most-used', {
      params: { limit },
    });
    return response.data;
  }

  /**
   * Set model rating
   */
  async setModelRating(repoId: string, data: {
    rating: number;
    variant?: string;
    notes?: string;
    tags?: string[];
  }): Promise<{ status: string }> {
    const response = await this.backendClient.post(
      `/api/v1/registry/models/${encodeURIComponent(repoId)}/rating`,
      data
    );
    return response.data;
  }

  /**
   * Get model rating
   */
  async getModelRating(repoId: string, variant?: string): Promise<{
    rating?: number;
    notes?: string;
    tags?: string[];
  }> {
    const response = await this.backendClient.get(
      `/api/v1/registry/models/${encodeURIComponent(repoId)}/rating`,
      { params: { variant } }
    );
    return response.data;
  }

  /**
   * Get hardware recommendations
   */
  async getHardwareRecommendations(vramGb: number, ramGb?: number): Promise<{
    recommendations: Array<{
      id: string;
      repo_id: string;
      name: string;
      model_type?: string;
      filename: string;
      quantization: string;
      size_bytes?: number;
      quality_score?: number;
      min_vram_gb?: number;
      recommended_vram_gb?: number;
      rating?: number;
    }>;
  }> {
    const response = await this.backendClient.get('/api/v1/registry/recommendations', {
      params: { vram_gb: vramGb, ram_gb: ramGb },
    });
    return response.data;
  }

  // ============================================================================
  // GraphRAG External Service API (Knowledge Graph with Neo4j + GLiNER)
  // ============================================================================

  /**
   * Check if GraphRAG service is available
   */
  async getGraphRAGHealth(): Promise<{
    status: string;
    timestamp?: string;
    message?: string;
    error?: string;
    url?: string;
  }> {
    const response = await this.backendClient.get('/api/v1/graphrag/health');
    return response.data;
  }

  /**
   * Get knowledge graph statistics from GraphRAG service
   */
  async getGraphRAGStats(): Promise<{
    nodes: number;
    edges: number;
    communities: number;
    domain: string | null;
  }> {
    const response = await this.backendClient.get('/api/v1/graphrag/stats');
    return response.data;
  }

  /**
   * Get available domains from GraphRAG service
   */
  async getGraphRAGDomains(): Promise<{
    domains: string[];
  }> {
    const response = await this.backendClient.get('/api/v1/graphrag/domains');
    return response.data;
  }

  /**
   * Get filtered knowledge graph data from GraphRAG service
   */
  async getGraphRAGGraph(params?: {
    domain?: string;
    max_entities?: number;
    max_relationships?: number;
    min_occurrence?: number;
    min_confidence?: number;
    entity_types?: string[];
    relationship_types?: string[];
    sort_by?: string;
    sort_order?: string;
  }): Promise<{
    nodes: Array<{
      id: string;
      label: string;
      type: string;
      occurrence?: number;
      properties?: Record<string, unknown>;
    }>;
    edges: Array<{
      id: string;
      source: string;
      target: string;
      label: string;
      type: string;
      weight?: number;
    }>;
    stats?: {
      nodes: number;
      edges: number;
    };
  }> {
    const response = await this.backendClient.post('/api/v1/graphrag/graph', params || {});
    return response.data;
  }

  /**
   * Get top entities by occurrence from GraphRAG service
   */
  async getGraphRAGTopEntities(params?: {
    domain?: string;
    limit?: number;
    min_occurrence?: number;
  }): Promise<{
    entities: Array<{
      name: string;
      type: string;
      occurrence: number;
      confidence: number;
      domain: string;
      description?: string;
    }>;
  }> {
    const response = await this.backendClient.get('/api/v1/graphrag/top-entities', { params });
    return response.data;
  }

  /**
   * Get top relationships by weight from GraphRAG service
   */
  async getGraphRAGTopRelationships(params?: {
    domain?: string;
    limit?: number;
    min_weight?: number;
  }): Promise<{
    relationships: Array<{
      source: string;
      target: string;
      type: string;
      weight: number;
      domain: string;
      context?: string;
    }>;
  }> {
    const response = await this.backendClient.get('/api/v1/graphrag/top-relationships', { params });
    return response.data;
  }

  /**
   * Extract entities and relationships from text using GraphRAG GLiNER
   */
  async extractGraphRAGEntities(params: {
    text: string;
    domain?: string;
  }): Promise<{
    text: string;
    domain: string;
    entities: Array<{
      name: string;
      type: string;
      description: string;
      confidence: number;
      metadata?: Record<string, unknown>;
    }>;
    relationships: Array<{
      source: string;
      target: string;
      relation: string;
      context?: string;
      confidence: number;
      metadata?: Record<string, unknown>;
    }>;
    entity_count: number;
    relationship_count: number;
    extraction_method: string;
    timestamp: string;
  }> {
    const response = await this.backendClient.post('/api/v1/graphrag/extract', params);
    return response.data;
  }

  /**
   * Hybrid search using GraphRAG service (vector + graph + keyword)
   */
  async searchGraphRAG(params: {
    query: string;
    top_k?: number;
    threshold?: number;
  }): Promise<{
    query: string;
    results: Array<{
      content: string;
      source: string;
      score: number;
      result_type: string;
      metadata?: Record<string, unknown>;
    }>;
    total_results: number;
  }> {
    const response = await this.backendClient.post('/api/v1/graphrag/search', params);
    return response.data;
  }

  /**
   * Advanced search with type selection using GraphRAG service
   */
  async advancedSearchGraphRAG(params: {
    query: string;
    search_type?: 'vector' | 'graph' | 'keyword' | 'hybrid';
    top_k?: number;
    domain?: string;
    filters?: Record<string, unknown>;
  }): Promise<{
    query: string;
    search_type: string;
    results: Array<{
      content: string;
      source: string;
      score: number;
      result_type: string;
      metadata?: Record<string, unknown>;
    }>;
    total_results: number;
  }> {
    const response = await this.backendClient.post('/api/v1/graphrag/search/advanced', params);
    return response.data;
  }

  /**
   * List documents in GraphRAG service
   */
  async listGraphRAGDocuments(): Promise<{
    documents: Array<{
      name: string;
      chunks?: number;
      entities?: number;
      relationships?: number;
    }>;
    total_documents: number;
    vector_store_documents: number;
    knowledge_graph_documents: number;
  }> {
    const response = await this.backendClient.get('/api/v1/graphrag/documents');
    return response.data;
  }

  /**
   * Delete a document from GraphRAG service
   */
  async deleteGraphRAGDocument(documentName: string): Promise<{
    status: string;
    message: string;
  }> {
    const response = await this.backendClient.delete(`/api/v1/graphrag/documents/${encodeURIComponent(documentName)}`);
    return response.data;
  }

  /**
   * Get NER service status from GraphRAG
   */
  async getGraphRAGNERStatus(): Promise<{
    ner_available: boolean;
    model_info: {
      model_name: string;
      model_type: string;
      framework: string;
      device: string;
      cuda_available: boolean;
      gpu_name?: string;
      capabilities: string[];
      default_entity_labels: string[];
      default_relation_types: Array<{
        relation: string;
        pairs_filter: string[][];
      }>;
    };
    extraction_method: string;
    timestamp: string;
  }> {
    const response = await this.backendClient.get('/api/v1/graphrag/ner/status');
    return response.data;
  }

  /**
   * Perform multi-hop reasoning using GraphRAG service
   */
  async multiHopReasoningGraphRAG(params: {
    query: string;
    max_hops?: number;
  }): Promise<{
    query: string;
    reasoning_path: Array<{
      hop: number;
      entity: string;
      relation: string;
      confidence: number;
    }>;
    answer?: string;
    sources: Array<{
      content: string;
      source: string;
      score: number;
    }>;
  }> {
    const response = await this.backendClient.post('/api/v1/graphrag/reasoning/multi-hop', params);
    return response.data;
  }

  /**
   * Get community/domain statistics from GraphRAG service
   */
  async getGraphRAGCommunities(domain?: string): Promise<{
    communities: Array<{
      id: string;
      name: string;
      entity_count: number;
      central_entities: string[];
    }>;
    stats: Record<string, unknown>;
  }> {
    const response = await this.backendClient.get('/api/v1/graphrag/communities', {
      params: domain ? { domain } : undefined
    });
    return response.data;
  }

  /**
   * Extract knowledge (entities/relationships) from a document to the knowledge graph
   */
  async extractDocumentKnowledge(documentId: string, domain?: string): Promise<{
    status: string;
    document_id: string;
    document_name: string;
    entity_count: number;
    relationship_count: number;
    extraction_method: string;
    entities: Array<{
      name: string;
      type: string;
      description: string;
      confidence: number;
    }>;
    relationships: Array<{
      source: string;
      target: string;
      relation: string;
      confidence: number;
    }>;
  }> {
    const response = await this.backendClient.post(`/api/v1/rag/documents/${documentId}/extract-knowledge`, {
      domain: domain || 'general'
    });
    return response.data;
  }

  /**
   * Batch extract knowledge from multiple documents
   */
  async batchExtractDocumentKnowledge(documentIds: string[], domain?: string): Promise<{
    status: string;
    documents_queued: number;
    document_ids: string[];
    domain: string;
  }> {
    const response = await this.backendClient.post('/api/v1/rag/documents/batch-extract-knowledge', {
      document_ids: documentIds,
      domain: domain || 'general'
    });
    return response.data;
  }

  // ============================================================================
  // STT (Speech-to-Text) API
  // ============================================================================

  /**
   * Get STT service status
   */
  async getSTTStatus(): Promise<{
    running: boolean;
    uptime: number;
    endpoint: string;
    model?: {
      name: string;
      size: string;
    };
  }> {
    const response = await this.backendClient.get('/api/v1/stt/status');
    return response.data;
  }

  /**
   * Transcribe audio file using local STT service
   */
  async transcribeAudio(audioBlob: Blob, options?: {
    model?: string;
    language?: string;
    response_format?: string;
  }): Promise<{
    text: string;
    language?: string;
    duration?: number;
    segments?: Array<{
      id: number;
      start: number;
      end: number;
      text: string;
      no_speech_prob?: number;  // Probability that segment contains no speech (0-1)
      avg_logprob?: number;     // Average log probability of tokens
    }>;
  }> {
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.webm');
    if (options?.model) formData.append('model', options.model);
    if (options?.language) formData.append('language', options.language);
    // Use verbose_json to get segment probabilities
    formData.append('response_format', options?.response_format || 'verbose_json');

    const response = await this.backendClient.post('/api/v1/stt/transcribe', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 120000, // 2 minutes for transcription
    });
    return response.data;
  }

  // ============================================================================
  // TTS (Text-to-Speech) API
  // ============================================================================

  /**
   * Get TTS service status
   */
  async getTTSStatus(): Promise<{
    running: boolean;
    uptime: number;
    endpoint: string;
    model?: {
      name: string;
      voice: string;
    };
  }> {
    const response = await this.backendClient.get('/api/v1/tts/status');
    return response.data;
  }

  /**
   * Synthesize speech from text using local TTS service
   * Returns audio as ArrayBuffer
   */
  async synthesizeSpeech(text: string, options?: {
    voice?: string;
    model?: string;
    speed?: number;
    response_format?: 'mp3' | 'wav' | 'opus' | 'flac';
  }): Promise<ArrayBuffer> {
    const response = await this.backendClient.post('/api/v1/tts/synthesize', {
      input: text,
      voice: options?.voice || 'alloy',
      model: options?.model || 'tts-1',
      speed: options?.speed || 1.0,
      response_format: options?.response_format || 'mp3',
    }, {
      responseType: 'arraybuffer',
      timeout: 120000, // 2 minutes for synthesis
    });
    return response.data;
  }
}

// Create singleton instance
export const apiService = new ApiService();

// Export default for convenience
export default apiService;
