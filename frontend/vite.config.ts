import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      // Management API endpoints (metrics, service status, etc.) -> Backend Management API
      '/api/v1/service': {
        target: process.env.VITE_BACKEND_URL || 'http://192.168.1.77:8700',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/api'),
        headers: {
          'Authorization': 'Bearer placeholder-api-key'
        }
      },
      '/api/v1/resources': {
        target: process.env.VITE_BACKEND_URL || 'http://192.168.1.77:8700',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/api'),
        headers: {
          'Authorization': 'Bearer placeholder-api-key'
        }
      },
      '/api/v1/logs': {
        target: process.env.VITE_BACKEND_URL || 'http://192.168.1.77:8700',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/api'),
        headers: {
          'Authorization': 'Bearer placeholder-api-key'
        }
      },
      '/api/v1/config': {
        target: process.env.VITE_BACKEND_URL || 'http://192.168.1.77:8700',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/api'),
        headers: {
          'Authorization': 'Bearer placeholder-api-key'
        }
      },
      '/api/v1/benchmark': {
        target: process.env.VITE_BACKEND_URL || 'http://192.168.1.77:8700',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/api'),
        headers: {
          'Authorization': 'Bearer placeholder-api-key'
        }
      },
      // Health endpoint from management API
      '/api/health': {
        target: process.env.VITE_BACKEND_URL || 'http://192.168.1.77:8700',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        headers: {
          'Authorization': 'Bearer placeholder-api-key'
        }
      },
      // LlamaCPP API endpoints (chat, completions, models) -> LlamaCPP API
      '/api': {
        target: process.env.VITE_API_BASE_URL || 'http://192.168.1.77:8600',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        headers: {
          'Authorization': 'Bearer placeholder-api-key'
        }
      },
      // Model management endpoints to backend API
      '/v1/models': {
        target: process.env.VITE_BACKEND_URL || 'http://192.168.1.77:8700',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/v1\/models/, '/api/v1/models'),
        headers: {
          'Authorization': 'Bearer placeholder-api-key'
        }
      },
      // Direct LlamaCPP API endpoints without /api prefix (except models)
      '/v1': {
        target: process.env.VITE_API_BASE_URL || 'http://192.168.1.77:8600',
        changeOrigin: true,
        headers: {
          'Authorization': 'Bearer placeholder-api-key'
        }
      }
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
