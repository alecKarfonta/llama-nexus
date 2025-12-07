import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Box, Paper } from '@mui/material'
import { Header } from '@/components/layout/Header'
import { Sidebar } from '@/components/layout/Sidebar'
import { DashboardPage } from '@/pages/DashboardPage'
import { ModelsPage } from '@/pages/ModelsPage'
import { ConfigurationPage } from '@/pages/ConfigurationPage'
import { DeployPage } from '@/pages/DeployPage'
import { ChatPage } from '@/pages/ChatPage'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'
import { TemplatesPage } from '@/pages/TemplatesPage'
import { TestingPage } from '@/pages/TestingPage'
import PromptLibraryPage from '@/pages/PromptLibraryPage'
import ModelRegistryPage from '@/pages/ModelRegistryPage'
import BenchmarkPage from '@/pages/BenchmarkPage'
import BatchProcessingPage from '@/pages/BatchProcessingPage'

// Placeholder pages for new routes until we implement them
const MonitoringPage = () => <Box><h1>Monitoring Page</h1><p>Coming soon...</p></Box>

function App() {
  const [sidebarOpen, setSidebarOpen] = React.useState(true)

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen)
  }

  return (
    <ErrorBoundary>
      <Box sx={{ display: 'flex', minHeight: '100vh' }}>
        <Header onMenuClick={handleSidebarToggle} />
        <Sidebar open={sidebarOpen} onToggle={handleSidebarToggle} />
        
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            pt: 6, // Account for header height
            ml: 0, // Removed left margin
            transition: 'margin-left 0.2s ease',
            display: 'flex',
            flexDirection: 'column',
            minHeight: '100vh',
            bgcolor: 'background.default',
            width: '100%', // Always use full width
          }}
        >
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/deploy" element={<DeployPage />} />
            <Route path="/templates" element={<TemplatesPage />} />
            <Route path="/prompts" element={<PromptLibraryPage />} />
            <Route path="/registry" element={<ModelRegistryPage />} />
            <Route path="/testing" element={<TestingPage />} />
            <Route path="/benchmark" element={<BenchmarkPage />} />
            <Route path="/batch" element={<BatchProcessingPage />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
            <Route path="/configuration" element={<ConfigurationPage />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Box>
      </Box>
    </ErrorBoundary>
  )
}

export default App
