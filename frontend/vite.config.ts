import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const backendTarget = process.env.VITE_BACKEND_URL || 'http://127.0.0.1:8700'
const inferenceTarget = process.env.VITE_API_BASE_URL || 'http://127.0.0.1:8600'
const proxyHeaders = {
  'Authorization': 'Bearer placeholder-api-key',
}

const backendProxy = {
  target: backendTarget,
  changeOrigin: true,
  headers: proxyHeaders,
}

const inferenceProxy = {
  target: inferenceTarget,
  changeOrigin: true,
  headers: proxyHeaders,
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      // Backend management API endpoints (configuration, conversations, benchmarks, RAG, etc.).
      '/api/v1': { ...backendProxy },
      // Health endpoint from management API
      '/api/health': {
        ...backendProxy,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // LLM proxy — route through backend so active backend (llama.cpp or vLLM) is selected
      '/v1/chat/completions': { ...backendProxy },
      '/v1/completions': { ...backendProxy },
      // Service management + model registry on backend API
      '/v1/service': { ...backendProxy },
      '/v1/models': { ...backendProxy },
      // LlamaCPP API endpoints (metrics, embeddings, etc.) -> LlamaCPP API
      '/api': {
        ...inferenceProxy,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // Remaining /v1/* (not chat/models/service) -> LlamaCPP API
      '/v1': { ...inferenceProxy },
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/components': path.resolve(__dirname, './src/components'),
      '@/hooks': path.resolve(__dirname, './src/hooks'),
      '@/services': path.resolve(__dirname, './src/services'),
      '@/types': path.resolve(__dirname, './src/types'),
      '@/utils': path.resolve(__dirname, './src/utils'),
      '@/pages': path.resolve(__dirname, './src/pages')
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          mui: ['@mui/material', '@mui/icons-material'],
          charts: ['recharts', '@mui/x-charts']
        }
      }
    }
  }
})
