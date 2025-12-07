/**
 * WebSocket Service for Real-time Updates
 * Handles WebSocket connections for live metrics and status updates
 */

import type { WebSocketMessage, MetricsUpdate, StatusUpdate, DownloadUpdate } from '@/types/api';

export type WebSocketEventHandler = (message: WebSocketMessage) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private handlers: Map<string, Set<WebSocketEventHandler>> = new Map();
  private url: string;

  constructor() {
    // Use WebSocket protocol in development, wss in production
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    
    // Use the same host/port as the frontend - nginx will proxy /ws to backend
    let host: string;
    if ((import.meta as any).env?.DEV) {
      // In development, use localhost on the dev server port
      host = window.location.host || 'localhost:3000';
    } else {
      // In production, use the same host (nginx proxies /ws to backend)
      host = window.location.host;
    }
    
    this.url = `${protocol}//${host}/ws`;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log('Attempting to connect to WebSocket:', this.url);
        this.ws = new WebSocket(this.url);

        // Set a timeout for connection
        const connectionTimeout = setTimeout(() => {
          if (this.ws?.readyState === WebSocket.CONNECTING) {
            this.ws.close();
            reject(new Error('WebSocket connection timeout - endpoint not implemented yet'));
          }
        }, 3000); // Shorter timeout to fail faster

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          console.log('WebSocket connected successfully');
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
          console.log('WebSocket disconnected:', event.code, event.reason);
          // Don't call handleDisconnection since we're not auto-reconnecting
          // and we don't want to emit error events for expected behavior
        };

        this.ws.onerror = () => {
          clearTimeout(connectionTimeout);
          console.info('WebSocket endpoint not available - this is expected. Using polling mode for updates.');
          reject(new Error('WebSocket not available - using polling fallback'));
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnecting');
      this.ws = null;
    }
  }

  private handleMessage(message: WebSocketMessage) {
    // Emit to type-specific handlers
    const typeHandlers = this.handlers.get(message.type);
    if (typeHandlers) {
      typeHandlers.forEach(handler => handler(message));
    }

    // Emit to general handlers
    const generalHandlers = this.handlers.get('*');
    if (generalHandlers) {
      generalHandlers.forEach(handler => handler(message));
    }
  }



  // Event handling methods
  on(eventType: string, handler: WebSocketEventHandler) {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set());
    }
    this.handlers.get(eventType)!.add(handler);
  }

  off(eventType: string, handler: WebSocketEventHandler) {
    const handlers = this.handlers.get(eventType);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.handlers.delete(eventType);
      }
    }
  }



  // Convenience methods for specific event types
  onMetrics(handler: (metrics: MetricsUpdate) => void) {
    this.on('metrics', handler as WebSocketEventHandler);
  }

  onStatus(handler: (status: StatusUpdate) => void) {
    this.on('status', handler as WebSocketEventHandler);
  }

  onDownload(handler: (download: DownloadUpdate) => void) {
    this.on('download', handler as WebSocketEventHandler);
  }

  onError(handler: (error: any) => void) {
    this.on('error', handler as WebSocketEventHandler);
  }

  // Check connection status
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get connectionState(): string {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'connecting';
      case WebSocket.OPEN: return 'connected';
      case WebSocket.CLOSING: return 'closing';
      case WebSocket.CLOSED: return 'closed';
      default: return 'unknown';
    }
  }
}

// Create singleton instance
export const websocketService = new WebSocketService();

// Export default for convenience
export default websocketService;
