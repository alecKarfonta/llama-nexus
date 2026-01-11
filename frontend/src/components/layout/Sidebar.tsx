import React, { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Box,
  alpha,
  Collapse,
} from '@mui/material'
import {
  Dashboard as DashboardIcon,
  Chat as ChatIcon,
  Memory as ModelsIcon,
  Settings as ConfigIcon,
  RocketLaunch as DeployIcon,
  Compress as QuantizationIcon,
  Science as TestingIcon,
  Insights as MonitoringIcon,
  Description as TemplatesIcon,
  MenuBook as PromptsIcon,
  Inventory2 as RegistryIcon,
  Speed as BenchmarkIcon,
  BatchPrediction as BatchIcon,
  CompareArrows as CompareIcon,
  Api as ApiIcon,
  AccountTree as WorkflowIcon,
  Storage as KnowledgeIcon,
  Hub as GraphIcon,
  Folder as DocumentsIcon,
  TravelExplore as DiscoveryIcon,
  Mic as MicIcon,
  RecordVoiceOver as TTSIcon,
  TextFields as EmbeddingIcon,
  ChevronRight as ChevronRightIcon,
  Tune as TuneIcon,
  TableChart as DatasetSidebarIcon,
  Science as DistillationIcon,
  Assessment as EvaluationIcon,
} from '@mui/icons-material'
import type { NavigationSection } from '@/types'

interface SidebarProps {
  open: boolean
  onToggle: () => void
}

// Navigation sections with items
const navigationSections: NavigationSection[] = [
  {
    id: 'overview',
    title: 'Overview',
    items: [
      { id: 'dashboard', label: 'Dashboard', path: '/dashboard', icon: 'dashboard', color: '#6366f1' },
      { id: 'models', label: 'Models', path: '/models', icon: 'models', color: '#8b5cf6' },
      { id: 'registry', label: 'Registry', path: '/registry', icon: 'registry', color: '#a855f7' },
      { id: 'quantization', label: 'Quantization', path: '/quantization', icon: 'quantization', color: '#10b981' },
    ],
  },
  {
    id: 'deployment',
    title: 'Deployment',
    items: [
      { id: 'deploy', label: 'LLM', path: '/deploy', icon: 'deploy', color: '#f59e0b' },
      { id: 'embedding-deploy', label: 'Embedding', path: '/embedding-deploy', icon: 'embedding', color: '#06b6d4' },
      { id: 'stt-deploy', label: 'STT', path: '/stt-deploy', icon: 'stt', color: '#10b981' },
      { id: 'tts-deploy', label: 'TTS', path: '/tts-deploy', icon: 'tts', color: '#8b5cf6' },
    ],
  },
  {
    id: 'development',
    title: 'Development',
    items: [
      { id: 'chat', label: 'Chat', path: '/chat', icon: 'chat', color: '#06b6d4' },
      { id: 'prompts', label: 'Prompt Library', path: '/prompts', icon: 'prompts', color: '#14b8a6' },
      { id: 'templates', label: 'Chat Templates', path: '/templates', icon: 'templates', color: '#10b981' },
      { id: 'workflows', label: 'Workflows', path: '/workflows', icon: 'workflow', color: '#a855f7' },
      { id: 'datasets', label: 'Datasets', path: '/datasets', icon: 'datasets', color: '#f59e0b' },
      { id: 'distillation', label: 'Distillation', path: '/distillation', icon: 'distillation', color: '#06b6d4' },
      { id: 'evaluation', label: 'Evaluation', path: '/evaluation', icon: 'evaluation', color: '#ec4899' },
      {
        id: 'finetuning',
        label: 'Fine-Tuning',
        path: '/finetuning',
        icon: 'finetuning',
        color: '#4ade80',
        subItems: [
          { id: 'finetuning-overview', label: 'Overview', path: '/finetuning' },
          { id: 'finetuning-templates', label: '🚀 Quick Start', path: '/finetuning/templates' },
          { id: 'finetuning-compare', label: 'Compare Adapters', path: '/finetuning/compare' },
        ]
      },
      { id: 'testing', label: 'Testing', path: '/testing', icon: 'testing', color: '#f97316' },
      { id: 'benchmark', label: 'Benchmark', path: '/benchmark', icon: 'benchmark', color: '#ef4444' },
      { id: 'batch', label: 'Batch Processing', path: '/batch', icon: 'batch', color: '#ec4899' },
      { id: 'compare', label: 'Compare Models', path: '/compare', icon: 'compare', color: '#f43f5e' },
    ],
  },
  {
    id: 'knowledge',
    title: 'Knowledge & RAG',
    items: [
      { id: 'documents', label: 'Documents', path: '/documents', icon: 'documents', color: '#10b981' },
      { id: 'knowledge-graph', label: 'Knowledge Graph', path: '/knowledge-graph', icon: 'graph', color: '#6366f1' },
      { id: 'discovery', label: 'Discovery', path: '/discovery', icon: 'discovery', color: '#8b5cf6' },
    ],
  },
  {
    id: 'operations',
    title: 'Operations',
    items: [
      { id: 'monitoring', label: 'Monitoring', path: '/monitoring', icon: 'monitoring', color: '#0ea5e9' },
      { id: 'api-docs', label: 'API Docs', path: '/api-docs', icon: 'api', color: '#14b8a6' },
      { id: 'configuration', label: 'Settings', path: '/configuration', icon: 'config', color: '#64748b' },
    ],
  },
]

const iconMap: Record<string, React.ElementType> = {
  dashboard: DashboardIcon,
  chat: ChatIcon,
  models: ModelsIcon,
  config: ConfigIcon,
  deploy: DeployIcon,
  quantization: QuantizationIcon,
  testing: TestingIcon,
  monitoring: MonitoringIcon,
  templates: TemplatesIcon,
  prompts: PromptsIcon,
  registry: RegistryIcon,
  benchmark: BenchmarkIcon,
  batch: BatchIcon,
  compare: CompareIcon,
  api: ApiIcon,
  workflow: WorkflowIcon,
  knowledge: KnowledgeIcon,
  graph: GraphIcon,
  documents: DocumentsIcon,
  discovery: DiscoveryIcon,
  stt: MicIcon,
  tts: TTSIcon,
  embedding: EmbeddingIcon,
  finetuning: TuneIcon,
  datasets: DatasetSidebarIcon,
  distillation: DistillationIcon,
  evaluation: EvaluationIcon,
}

const getIcon = (iconName: string) => {
  const Icon = iconMap[iconName] || DashboardIcon
  return <Icon />
}

// Load collapsed sections from localStorage
const loadCollapsedSections = (): Record<string, boolean> => {
  try {
    const saved = localStorage.getItem('sidebar-collapsed-sections')
    return saved ? JSON.parse(saved) : {}
  } catch {
    return {}
  }
}

export const Sidebar: React.FC<SidebarProps> = ({ open, onToggle }) => {
  const location = useLocation()
  const navigate = useNavigate()
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>(loadCollapsedSections())

  // Save collapsed state to localStorage
  useEffect(() => {
    localStorage.setItem('sidebar-collapsed-sections', JSON.stringify(collapsedSections))
  }, [collapsedSections])

  const toggleSection = (sectionId: string) => {
    setCollapsedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }))
  }

  const handleNavigation = (path: string) => {
    navigate(path)
  }

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: open ? 200 : 0,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 200,
          boxSizing: 'border-box',
          background: 'linear-gradient(180deg, rgba(26, 26, 46, 0.98) 0%, rgba(15, 15, 35, 0.98) 100%)',
          backdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(255, 255, 255, 0.06)',
          boxShadow: '4px 0 20px rgba(0, 0, 0, 0.3)',
        },
      }}
    >
      <Toolbar sx={{ minHeight: '64px !important' }} />

      {/* Sidebar Content */}
      <Box sx={{
        flex: 1,
        overflowY: 'auto',
        overflowX: 'hidden',
        py: 1,
        '&::-webkit-scrollbar': {
          width: 4,
        },
        '&::-webkit-scrollbar-thumb': {
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: 2,
        },
      }}>
        {navigationSections.map((section) => {
          const isCollapsed = collapsedSections[section.id] ?? false
          const hasActiveItem = section.items.some(item => item.path === location.pathname)

          return (
            <React.Fragment key={section.id}>
              <Box
                onClick={() => toggleSection(section.id)}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  mx: 1,
                  mt: 2,
                  mb: 0.75,
                  px: 1,
                  py: 0.25,
                  borderRadius: 1,
                  transition: 'all 0.15s ease',
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.03)',
                  },
                }}
              >
                <Typography
                  sx={{
                    fontSize: '0.6875rem',
                    textTransform: 'uppercase',
                    color: hasActiveItem ? alpha('#fff', 0.8) : alpha('#fff', 0.4),
                    letterSpacing: '0.08em',
                    fontWeight: 600,
                    transition: 'color 0.15s ease',
                  }}
                >
                  {section.title}
                </Typography>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    color: alpha('#fff', 0.3),
                    transition: 'transform 0.2s ease',
                    transform: isCollapsed ? 'rotate(0deg)' : 'rotate(90deg)',
                  }}
                >
                  <ChevronRightIcon sx={{ fontSize: 14 }} />
                </Box>
              </Box>
              <Collapse in={!isCollapsed} timeout={200}>
                <List sx={{ py: 0, px: 1 }}>
                  {section.items.map((item) => {
                    const isSelected = location.pathname === item.path
                    const itemColor = item.color || '#6366f1'

                    return (
                      <ListItem key={item.id} disablePadding sx={{ mb: 0.25 }}>
                        <ListItemButton
                          selected={isSelected}
                          onClick={() => handleNavigation(item.path)}
                          sx={{
                            py: 1,
                            px: 1.5,
                            borderRadius: 2,
                            minHeight: 42,
                            position: 'relative',
                            overflow: 'hidden',
                            transition: 'all 0.2s ease-in-out',

                            // Default state
                            bgcolor: 'transparent',

                            // Hover state
                            '&:hover': {
                              bgcolor: alpha(itemColor, 0.08),
                              '& .nav-icon': {
                                color: itemColor,
                                transform: 'scale(1.1)',
                              },
                              '& .nav-label': {
                                color: 'text.primary',
                              },
                            },

                            // Selected state
                            '&.Mui-selected': {
                              bgcolor: alpha(itemColor, 0.12),

                              // Left border glow
                              '&::before': {
                                content: '""',
                                position: 'absolute',
                                left: 0,
                                top: '50%',
                                transform: 'translateY(-50%)',
                                width: 3,
                                height: '60%',
                                bgcolor: itemColor,
                                borderRadius: '0 3px 3px 0',
                                boxShadow: `0 0 12px ${itemColor}`,
                              },

                              '& .nav-icon': {
                                color: itemColor,
                              },
                              '& .nav-label': {
                                color: 'text.primary',
                                fontWeight: 600,
                              },

                              '&:hover': {
                                bgcolor: alpha(itemColor, 0.16),
                              },
                            },
                          }}
                        >
                          <ListItemIcon
                            className="nav-icon"
                            sx={{
                              minWidth: 36,
                              color: isSelected ? itemColor : 'text.secondary',
                              transition: 'all 0.2s ease-in-out',
                              '& .MuiSvgIcon-root': {
                                fontSize: '1.25rem',
                              },
                            }}
                          >
                            {getIcon(item.icon || 'dashboard')}
                          </ListItemIcon>
                          <ListItemText
                            className="nav-label"
                            primary={item.label}
                            primaryTypographyProps={{
                              fontSize: '0.875rem',
                              fontWeight: isSelected ? 600 : 500,
                              color: isSelected ? 'text.primary' : 'text.secondary',
                              lineHeight: 1.3,
                              transition: 'all 0.2s ease-in-out',
                            }}
                          />

                          {/* Active indicator dot */}
                          {isSelected && (
                            <Box sx={{
                              width: 6,
                              height: 6,
                              borderRadius: '50%',
                              bgcolor: itemColor,
                              boxShadow: `0 0 8px ${itemColor}`,
                              animation: 'pulse 2s infinite',
                              '@keyframes pulse': {
                                '0%, 100%': {
                                  opacity: 1,
                                },
                                '50%': {
                                  opacity: 0.5,
                                },
                              },
                            }} />
                          )}
                        </ListItemButton>
                      </ListItem>
                    )
                  })}
                </List>
              </Collapse>
            </React.Fragment>
          )
        })}

      </Box>

      {/* Footer */}
      <Box sx={{
        p: 2,
        borderTop: '1px solid rgba(255, 255, 255, 0.06)',
        background: 'linear-gradient(180deg, transparent 0%, rgba(0, 0, 0, 0.2) 100%)',
      }}>
        <Box sx={{
          p: 1.5,
          borderRadius: 2,
          bgcolor: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.04)',
        }}>
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              display: 'block',
              fontSize: '0.6875rem',
              letterSpacing: '0.02em',
            }}
          >
            Llama Nexus v1.0
          </Typography>
          <Typography
            variant="caption"
            sx={{
              color: alpha('#64748b', 0.6),
              display: 'block',
              fontSize: '0.625rem',
            }}
          >
            ML Model Management
          </Typography>
        </Box>
      </Box>
    </Drawer>
  )
}
