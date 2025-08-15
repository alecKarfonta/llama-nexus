/**
 * API Service Layer for LlamaCPP Management Frontend
 * Handles all HTTP communication with the llamacpp backend
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
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
  TokenUsageData
} from '@/types/api';

class ApiService {
  private client: AxiosInstance;
  private backendClient: AxiosInstance;
  private baseURL: string;
  private backendBaseURL: string;

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
        'Authorization': 'Bearer placeholder-api-key'
      }
    });

    this.backendClient = axios.create({
      baseURL: this.backendBaseURL || this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer placeholder-api-key'
      }
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config: any) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error: any) => {
        console.error('API Request Error:', error);
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
    const response = await this.client.get('/v1/models');
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
          lastModified: model.lastModified ? new Date(model.lastModified) : undefined
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
    const response = await this.client.get('/v1/models/current');
    return response.data;
  }

  async listRepoFiles(repoId: string, revision: string = 'main'): Promise<string[]> {
    const response = await this.client.get<ApiResponse<{ files: string[] }>>('/v1/models/repo-files', {
      params: { repo_id: repoId, revision }
    });
    const files = (response.data as any)?.data?.files || [];
    return Array.isArray(files) ? files : [];
  }

  async downloadModel(request: ModelDownloadRequest): Promise<ModelDownload> {
    const response = await this.client.post<ApiResponse<ModelDownload>>('/v1/models/download', request);
    return this.normalizeModelDownload(response.data.data);
  }

  async getModelDownloads(): Promise<ModelDownload[]> {
    const response = await this.client.get<ApiResponse<ModelDownload[]>>('/v1/models/downloads');
    const data = response.data.data || [];
    return (data as any[]).map((item) => this.normalizeModelDownload(item));
  }

  async cancelModelDownload(modelId: string): Promise<void> {
    await this.client.delete(`/v1/models/downloads/${modelId}`);
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
    const response = await fetch(`/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer placeholder-api-key',
        'Accept': 'text/event-stream',
        'Cache-Control': 'no-cache',
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
}

// Create singleton instance
export const apiService = new ApiService();

// Export default for convenience
export default apiService;
