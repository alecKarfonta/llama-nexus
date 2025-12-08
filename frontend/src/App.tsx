import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Box } from '@mui/material'
import { Header } from '@/components/layout/Header'
import { Sidebar } from '@/components/layout/Sidebar'
import { DashboardPage } from '@/pages/DashboardPage'
import { ModelsPage } from '@/pages/ModelsPage'
import { ConfigurationPage } from '@/pages/ConfigurationPage'
import { DeployPage } from '@/pages/DeployPage'
import { EmbeddingDeployPage } from '@/pages/EmbeddingDeployPage'
import { ChatPage } from '@/pages/ChatPage'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'
import { TemplatesPage } from '@/pages/TemplatesPage'
import { TestingPage } from '@/pages/TestingPage'
import PromptLibraryPage from '@/pages/PromptLibraryPage'
import ModelRegistryPage from '@/pages/ModelRegistryPage'
import BenchmarkPage from '@/pages/BenchmarkPage'
import BatchProcessingPage from '@/pages/BatchProcessingPage'
import ModelComparisonPage from '@/pages/ModelComparisonPage'
import MonitoringPage from '@/pages/MonitoringPage'
import ApiDocsPage from '@/pages/ApiDocsPage'
import WorkflowBuilderPage from '@/pages/WorkflowBuilderPage'
import KnowledgeBasePage from '@/pages/KnowledgeBasePage'
import KnowledgeGraphPage from '@/pages/KnowledgeGraphPage'
import DocumentsPage from '@/pages/DocumentsPage'
import DiscoveryPage from '@/pages/DiscoveryPage'

function App() {
  const [sidebarOpen, setSidebarOpen] = React.useState(true)

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen)
  }

  return (
    <ErrorBoundary>
      <Box sx={{ 
        display: 'flex', 
        minHeight: '100vh',
        bgcolor: 'background.default',
      }}>
        <Header onMenuClick={handleSidebarToggle} />
        <Sidebar open={sidebarOpen} onToggle={handleSidebarToggle} />
        
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            pt: '64px', // Account for new header height
            ml: sidebarOpen ? '240px' : 0, // Account for sidebar width
            transition: 'margin-left 0.3s ease-in-out',
            display: 'flex',
            flexDirection: 'column',
            minHeight: '100vh',
            width: sidebarOpen ? 'calc(100% - 240px)' : '100%',
            position: 'relative',
            overflow: 'hidden',
            // Subtle background pattern
            '&::before': {
              content: '""',
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: `
                radial-gradient(ellipse at 20% 0%, rgba(99, 102, 241, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(139, 92, 246, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 0% 50%, rgba(6, 182, 212, 0.04) 0%, transparent 50%)
              `,
              pointerEvents: 'none',
              zIndex: 0,
            },
          }}
        >
          <Box sx={{ 
            flexGrow: 1, 
            position: 'relative', 
            zIndex: 1,
            overflow: 'auto',
          }}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/models" element={<ModelsPage />} />
              <Route path="/deploy" element={<DeployPage />} />
              <Route path="/embedding-deploy" element={<EmbeddingDeployPage />} />
              <Route path="/templates" element={<TemplatesPage />} />
              <Route path="/prompts" element={<PromptLibraryPage />} />
              <Route path="/registry" element={<ModelRegistryPage />} />
              <Route path="/testing" element={<TestingPage />} />
              <Route path="/benchmark" element={<BenchmarkPage />} />
              <Route path="/batch" element={<BatchProcessingPage />} />
              <Route path="/compare" element={<ModelComparisonPage />} />
              <Route path="/workflows" element={<WorkflowBuilderPage />} />
              <Route path="/knowledge" element={<KnowledgeBasePage />} />
              <Route path="/knowledge-graph" element={<KnowledgeGraphPage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/discovery" element={<DiscoveryPage />} />
              <Route path="/monitoring" element={<MonitoringPage />} />
              <Route path="/api-docs" element={<ApiDocsPage />} />
              <Route path="/configuration" element={<ConfigurationPage />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Box>
        </Box>
      </Box>
    </ErrorBoundary>
  )
}

export default App
