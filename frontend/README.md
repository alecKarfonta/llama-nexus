# LlamaCPP Management Frontend

A comprehensive React-based management frontend for the llamacpp service, enabling model management, resource monitoring, and service configuration through a modern web interface.

## 🚀 Quick Start

### Development Mode

```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev
```

The development server will start on `http://localhost:3000` with API proxy configured.

### Production Deployment

```bash
# Build and start with Docker Compose (from project root)
docker compose up -d --build

# Access the frontend at http://localhost:3000
```

## 🏗️ Architecture

**Tech Stack:**
- **Framework**: React 18 with TypeScript
- **UI Library**: Material-UI 5
- **State Management**: React Query for server state + React Context for client state
- **Build Tool**: Vite for fast development and optimized builds
- **API Communication**: Axios with interceptors
- **Containerization**: Multi-stage Docker build with nginx

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/           # Reusable UI components
│   │   ├── common/          # Generic components (ErrorBoundary, etc.)
│   │   └── layout/          # Layout components (Header, Sidebar)
│   ├── hooks/               # Custom React hooks
│   ├── pages/               # Page components
│   ├── services/            # API and WebSocket services
│   ├── types/               # TypeScript type definitions
│   ├── utils/               # Utility functions and theme
│   ├── App.tsx              # Main application component
│   └── main.tsx             # Application entry point
├── public/                  # Static assets
├── Dockerfile               # Multi-stage Docker build
├── nginx.conf               # Nginx configuration
└── package.json             # Dependencies and scripts
```

## 🔌 API Integration

The frontend integrates with the llamacpp service through:

- **REST API**: Standard HTTP requests for configuration and control
- **WebSocket**: Real-time updates for metrics and status
- **Proxy**: Nginx proxies `/api/*` to the llamacpp service

### Supported Endpoints

- `GET /api/v1/models` - List available models
- `POST /api/v1/models/download` - Download new model
- `GET /api/v1/service/config` - Get service configuration
- `PUT /api/v1/service/config` - Update configuration
- `GET /api/v1/service/status` - Get service status
- `GET /api/v1/metrics` - Get resource metrics
- `GET /ws` - WebSocket for real-time updates

## 🎯 Current Status (Phase 1 Complete)

✅ **Architecture & Setup**
- React project structure with TypeScript
- Material-UI components and theming
- Vite build configuration
- Docker containerization with nginx

✅ **API Contracts**
- Type definitions for all data models
- API service layer with error handling
- WebSocket service for real-time updates

✅ **Basic UI Structure**
- Header with navigation
- Sidebar with route navigation
- Placeholder pages for all main features
- Error boundary for resilience

## 🔜 Next Steps (Phase 2)

The next phase will implement:
- Real-time service monitoring dashboard
- Resource usage visualization components
- WebSocket integration for live updates
- Service health indicators and alerts

## 🛠️ Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run test` - Run unit tests
- `npm run test:coverage` - Run tests with coverage
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

### Environment Variables

- `VITE_API_BASE_URL` - Base URL for the LlamaCPP API (default: http://192.168.1.77:8600)
- `VITE_BACKEND_URL` - Base URL for the backend management API (default: http://192.168.1.77:8700)

### Docker Development

```bash
# Build frontend container
docker build -t llamacpp-frontend ./frontend

# Run with API integration
docker compose up -d
```

## 🔐 Security

- API key authentication handled by axios interceptors
- CORS and security headers configured in nginx
- Input validation for all configuration parameters
- Error boundary prevents crashes from API failures

## 📊 Performance

- Code splitting with manual chunks for optimal loading
- Static asset caching with nginx
- Optimized Docker image with multi-stage build
- React Query for efficient data fetching and caching

## 🧪 Testing

Unit tests will be implemented in Phase 5 using:
- React Testing Library for component tests
- Vitest for test runner
- Coverage reporting with 90%+ target
