import { lazy, type ElementType, type ReactElement } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import {
  Api as ApiIcon,
  Assessment as EvaluationIcon,
  AutoAwesome as FineTuningIcon,
  Chat as ChatIcon,
  Code as CodeIcon,
  CompareArrows as CompareIcon,
  Compress as QuantizationIcon,
  Dashboard as DashboardIcon,
  Description as TemplatesIcon,
  Extension as ExtensionIcon,
  Folder as DocumentsIcon,
  GraphicEq as StreamingSTTIcon,
  Hub as GraphIcon,
  Insights as MonitoringIcon,
  Inventory2 as RegistryIcon,
  ManageSearch as SearchIcon,
  Memory as ModelsIcon,
  MenuBook as PromptsIcon,
  Mic as MicIcon,
  Psychology as ReasoningIcon,
  RecordVoiceOver as TTSIcon,
  RocketLaunch as DeployIcon,
  Science as DistillationIcon,
  Settings as ConfigIcon,
  Speed as BenchmarkIcon,
  SyncAlt as HybridIcon,
  TableChart as DatasetIcon,
  TextFields as EmbeddingIcon,
  TravelExplore as DiscoveryIcon,
  Tune as TuneIcon,
  AccountTree as WorkflowIcon,
} from '@mui/icons-material'
import type { NavigationItem, NavigationSection } from '@/types'

const ApiDocsPage = lazy(() => import('@/pages/ApiDocsPage'))
const BenchmarkPage = lazy(() => import('@/pages/BenchmarkPage'))
const ChatPage = lazy(() => import('@/pages/ChatPage').then((module) => ({ default: module.ChatPage })))
const CodeSearchPage = lazy(() => import('@/pages/CodeSearchPage').then((module) => ({ default: module.CodeSearchPage })))
const ConfigurationPage = lazy(() => import('@/pages/ConfigurationPage').then((module) => ({ default: module.ConfigurationPage })))
const DashboardPage = lazy(() => import('@/pages/DashboardPage').then((module) => ({ default: module.DashboardPage })))
const DeployPage = lazy(() => import('@/pages/DeployPage').then((module) => ({ default: module.DeployPage })))
const DiscoveryPage = lazy(() => import('@/pages/DiscoveryPage'))
const DocumentsPage = lazy(() => import('@/pages/DocumentsPage'))
const EmbeddingDeployPage = lazy(() => import('@/pages/EmbeddingDeployPage').then((module) => ({ default: module.EmbeddingDeployPage })))
const EntityManagerPage = lazy(() => import('@/pages/EntityManagerPage'))
const FineTuningPage = lazy(() => import('@/pages/FineTuningPage'))
const GraphRAGWorkbenchPage = lazy(() => import('@/pages/GraphRAGWorkbenchPage'))
const HybridProcessingPage = lazy(() => import('@/pages/HybridProcessingPage').then((module) => ({ default: module.HybridProcessingPage })))
const IntelligentSearchPage = lazy(() => import('@/pages/IntelligentSearchPage'))
const KnowledgeGraphPage = lazy(() => import('@/pages/KnowledgeGraphPage'))
const MCPServersPage = lazy(() => import('@/pages/MCPServersPage'))
const ModelRegistryPage = lazy(() => import('@/pages/ModelRegistryPage'))
const ModelsPage = lazy(() => import('@/pages/ModelsPage').then((module) => ({ default: module.ModelsPage })))
const MonitoringPage = lazy(() => import('@/pages/MonitoringPage'))
const PromptLibraryPage = lazy(() => import('@/pages/PromptLibraryPage'))
const QuantizationPage = lazy(() => import('@/pages/QuantizationPage').then((module) => ({ default: module.QuantizationPage })))
const ReasoningPlaygroundPage = lazy(() => import('@/pages/ReasoningPlaygroundPage'))
const RetrievalAnalyticsPage = lazy(() => import('@/pages/RetrievalAnalyticsPage').then((module) => ({ default: module.RetrievalAnalyticsPage })))
const STTDeployPage = lazy(() => import('@/pages/STTDeployPage'))
const StreamingSTTDeployPage = lazy(() => import('@/pages/StreamingSTTDeployPage'))
const TTSDeployPage = lazy(() => import('@/pages/TTSDeployPage'))
const TemplatesPage = lazy(() => import('@/pages/TemplatesPage').then((module) => ({ default: module.TemplatesPage })))
const WorkflowBuilderPage = lazy(() => import('@/pages/WorkflowBuilderPage'))
const AdapterComparisonPage = lazy(() => import('@/pages/finetuning/AdapterComparisonPage').then((module) => ({ default: module.AdapterComparisonPage })))
const BenchmarkComparisonPage = lazy(() => import('@/pages/finetuning/BenchmarkComparisonPage'))
const BookDatasetPage = lazy(() => import('@/pages/finetuning/BookDatasetPage').then((module) => ({ default: module.BookDatasetPage })))
const DatasetManagementPage = lazy(() => import('@/pages/finetuning/DatasetManagementPage').then((module) => ({ default: module.DatasetManagementPage })))
const DistillationPage = lazy(() => import('@/pages/finetuning/DistillationPage').then((module) => ({ default: module.DistillationPage })))
const ModelEvaluationPage = lazy(() => import('@/pages/finetuning/ModelEvaluationPage').then((module) => ({ default: module.ModelEvaluationPage })))
const WorkflowTemplatesPage = lazy(() => import('@/pages/finetuning/WorkflowTemplatesPage').then((module) => ({ default: module.WorkflowTemplatesPage })))

export type NavigationIcon = ElementType

export interface AppPage {
  id: string
  label: string
  title: string
  description: string
  path: string
  icon: NavigationIcon
  color: string
  section: string
  keywords: string[]
  element: ReactElement
}

export interface AppRedirect {
  path: string
  element: ReactElement
}

interface PreservingRedirectProps {
  to: string
  preserveLocation?: boolean
}

const PreservingRedirect = ({ to, preserveLocation = true }: PreservingRedirectProps) => {
  const location = useLocation()

  return (
    <Navigate
      to={{
        pathname: to,
        search: preserveLocation ? location.search : '',
        hash: preserveLocation ? location.hash : '',
      }}
      replace
    />
  )
}

export type RouteMeta = Pick<AppPage, 'id' | 'label' | 'title' | 'description' | 'path' | 'color' | 'section'>

export interface SearchableNavigationItem extends Omit<NavigationItem, 'color' | 'description' | 'icon' | 'keywords'> {
  section: string
  icon: NavigationIcon
  color: string
  title: string
  description: string
  keywords: string[]
  parentLabels: string[]
}

export interface NavigationTrailItem {
  id: string
  label: string
  path: string
  color: string
  icon: NavigationIcon
  section: string
  isCurrent: boolean
}

export interface DashboardActionItem {
  id: string
  label: string
  description: string
  path: string
  icon: NavigationIcon
  color: string
}

export interface DashboardSurfaceItem extends Omit<DashboardActionItem, 'description'> {
  detail: string
}

export interface DashboardLifecycleStage extends DashboardActionItem {
  title: string
  summary: string
  links: DashboardActionItem[]
}

export interface DashboardWorkflowStep extends DashboardActionItem {
  phase: string
  title: string
  summary: string
  proof: string
  primaryAction: DashboardActionItem
  secondaryAction?: DashboardActionItem
}

export const appPages: AppPage[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    title: 'Dashboard',
    description: 'System overview and metrics',
    path: '/dashboard',
    icon: DashboardIcon,
    color: '#6366f1',
    section: 'Overview',
    keywords: ['home', 'overview', 'status', 'stats'],
    element: <DashboardPage />,
  },
  {
    id: 'chat',
    label: 'Chat',
    title: 'Chat',
    description: 'Interactive conversations',
    path: '/chat',
    icon: ChatIcon,
    color: '#06b6d4',
    section: 'Build & Run',
    keywords: ['conversation', 'talk', 'message', 'playground'],
    element: <ChatPage />,
  },
  {
    id: 'deploy',
    label: 'LLM',
    title: 'Deploy LLM',
    description: 'Configure and launch language models',
    path: '/deploy',
    icon: DeployIcon,
    color: '#f59e0b',
    section: 'Build & Run',
    keywords: ['start', 'launch', 'run', 'server', 'llm'],
    element: <DeployPage />,
  },
  {
    id: 'embedding-deploy',
    label: 'Embeddings',
    title: 'Deploy Embeddings',
    description: 'Launch embedding services',
    path: '/embedding-deploy',
    icon: EmbeddingIcon,
    color: '#06b6d4',
    section: 'Build & Run',
    keywords: ['embedding', 'embed', 'vector'],
    element: <EmbeddingDeployPage />,
  },
  {
    id: 'stt-deploy',
    label: 'Speech to Text',
    title: 'Speech to Text',
    description: 'Deploy transcription services',
    path: '/stt-deploy',
    icon: MicIcon,
    color: '#10b981',
    section: 'Build & Run',
    keywords: ['speech', 'stt', 'whisper', 'transcribe', 'audio'],
    element: <STTDeployPage />,
  },
  {
    id: 'streaming-stt-deploy',
    label: 'Streaming STT',
    title: 'Streaming Speech to Text',
    description: 'Deploy live WebSocket transcription',
    path: '/streaming-stt',
    icon: StreamingSTTIcon,
    color: '#6366f1',
    section: 'Build & Run',
    keywords: ['speech', 'stt', 'streaming', 'realtime', 'websocket', 'nemotron'],
    element: <StreamingSTTDeployPage />,
  },
  {
    id: 'tts-deploy',
    label: 'Text to Speech',
    title: 'Text to Speech',
    description: 'Deploy voice generation services',
    path: '/tts-deploy',
    icon: TTSIcon,
    color: '#8b5cf6',
    section: 'Build & Run',
    keywords: ['voice', 'tts', 'speak', 'audio'],
    element: <TTSDeployPage />,
  },
  {
    id: 'workflows',
    label: 'Workflows',
    title: 'Workflows',
    description: 'Multi-step LLM pipelines',
    path: '/workflows',
    icon: WorkflowIcon,
    color: '#a855f7',
    section: 'Build & Run',
    keywords: ['pipeline', 'chain', 'flow', 'automation'],
    element: <WorkflowBuilderPage />,
  },
  {
    id: 'models',
    label: 'Models',
    title: 'Models',
    description: 'Browse and manage model files',
    path: '/models',
    icon: ModelsIcon,
    color: '#8b5cf6',
    section: 'Models & Training',
    keywords: ['download', 'gguf', 'model files', 'llm'],
    element: <ModelsPage />,
  },
  {
    id: 'registry',
    label: 'Registry',
    title: 'Registry',
    description: 'Model metadata and cache',
    path: '/registry',
    icon: RegistryIcon,
    color: '#a855f7',
    section: 'Models & Training',
    keywords: ['cache', 'metadata', 'model registry'],
    element: <ModelRegistryPage />,
  },
  {
    id: 'benchmark',
    label: 'Benchmarks',
    title: 'Benchmarks',
    description: 'Inference performance testing',
    path: '/benchmarks',
    icon: BenchmarkIcon,
    color: '#ef4444',
    section: 'Models & Training',
    keywords: ['performance', 'speed', 'bfcl', 'throughput'],
    element: <BenchmarkPage />,
  },
  {
    id: 'quantization',
    label: 'Quantization',
    title: 'Quantization',
    description: 'Optimize and convert models',
    path: '/quantization',
    icon: QuantizationIcon,
    color: '#10b981',
    section: 'Models & Training',
    keywords: ['optimize', 'compress', 'convert', 'gguf'],
    element: <QuantizationPage />,
  },
  {
    id: 'finetuning',
    label: 'Fine-Tuning',
    title: 'Fine-Tuning',
    description: 'Training workflows and adapters',
    path: '/finetuning',
    icon: TuneIcon,
    color: '#4ade80',
    section: 'Models & Training',
    keywords: ['train', 'lora', 'adapter', 'fine tune'],
    element: <FineTuningPage />,
  },
  {
    id: 'finetuning-templates',
    label: 'Quick Start',
    title: 'Fine-Tuning Templates',
    description: 'Reusable fine-tuning workflows',
    path: '/finetuning/templates',
    icon: FineTuningIcon,
    color: '#4ade80',
    section: 'Models & Training',
    keywords: ['quick start', 'template', 'workflow'],
    element: <WorkflowTemplatesPage />,
  },
  {
    id: 'finetuning-datasets',
    label: 'Datasets',
    title: 'Datasets',
    description: 'Training data management',
    path: '/finetuning/datasets',
    icon: DatasetIcon,
    color: '#f59e0b',
    section: 'Models & Training',
    keywords: ['dataset', 'data', 'training data'],
    element: <DatasetManagementPage />,
  },
  {
    id: 'finetuning-distillation',
    label: 'Distillation',
    title: 'Distillation',
    description: 'Teacher-to-student training',
    path: '/finetuning/distillation',
    icon: DistillationIcon,
    color: '#06b6d4',
    section: 'Models & Training',
    keywords: ['distill', 'teacher', 'student'],
    element: <DistillationPage />,
  },
  {
    id: 'finetuning-evaluation',
    label: 'Evaluation',
    title: 'Model Evaluation',
    description: 'Evaluate trained models and adapters',
    path: '/finetuning/evaluation',
    icon: EvaluationIcon,
    color: '#ec4899',
    section: 'Models & Training',
    keywords: ['eval', 'lm eval', 'score', 'quality'],
    element: <ModelEvaluationPage />,
  },
  {
    id: 'finetuning-compare',
    label: 'Compare Adapters',
    title: 'Compare Adapters',
    description: 'Compare fine-tuned adapters',
    path: '/finetuning/compare',
    icon: CompareIcon,
    color: '#8b5cf6',
    section: 'Models & Training',
    keywords: ['compare', 'adapter', 'diff', 'versus'],
    element: <AdapterComparisonPage />,
  },
  {
    id: 'finetuning-book-datasets',
    label: 'Book Datasets',
    title: 'Book Datasets',
    description: 'Prepare book-derived training data',
    path: '/finetuning/book-datasets',
    icon: TemplatesIcon,
    color: '#14b8a6',
    section: 'Models & Training',
    keywords: ['book', 'dataset', 'corpus'],
    element: <BookDatasetPage />,
  },
  {
    id: 'finetuning-benchmarks',
    label: 'Training Benchmarks',
    title: 'Training Benchmarks',
    description: 'Compare benchmark runs',
    path: '/finetuning/benchmarks',
    icon: CompareIcon,
    color: '#ef4444',
    section: 'Models & Training',
    keywords: ['benchmark comparison', 'results', 'training'],
    element: <BenchmarkComparisonPage />,
  },
  {
    id: 'prompts',
    label: 'Prompt Library',
    title: 'Prompt Library',
    description: 'Reusable prompts and system instructions',
    path: '/prompts',
    icon: PromptsIcon,
    color: '#14b8a6',
    section: 'Prompts & Templates',
    keywords: ['prompt', 'system', 'instruction'],
    element: <PromptLibraryPage />,
  },
  {
    id: 'templates',
    label: 'Chat Templates',
    title: 'Chat Templates',
    description: 'Jinja chat template management',
    path: '/templates',
    icon: TemplatesIcon,
    color: '#10b981',
    section: 'Prompts & Templates',
    keywords: ['jinja', 'format', 'chat template'],
    element: <TemplatesPage />,
  },
  {
    id: 'documents',
    label: 'Documents',
    title: 'Documents',
    description: 'Document ingestion and management',
    path: '/documents',
    icon: DocumentsIcon,
    color: '#10b981',
    section: 'Knowledge & RAG',
    keywords: ['upload', 'pdf', 'file', 'ingest'],
    element: <DocumentsPage />,
  },
  {
    id: 'search',
    label: 'Search',
    title: 'Search',
    description: 'GraphRAG search and answers',
    path: '/search',
    icon: SearchIcon,
    color: '#06b6d4',
    section: 'Knowledge & RAG',
    keywords: ['semantic', 'search', 'rag', 'answer'],
    element: <IntelligentSearchPage />,
  },
  {
    id: 'knowledge-graph',
    label: 'Knowledge Graph',
    title: 'Knowledge Graph',
    description: 'Entity relationship visualization',
    path: '/knowledge-graph',
    icon: GraphIcon,
    color: '#6366f1',
    section: 'Knowledge & RAG',
    keywords: ['graph', 'entity', 'relationship', 'neo4j'],
    element: <KnowledgeGraphPage />,
  },
  {
    id: 'graphrag-workbench',
    label: 'Document Chat',
    title: 'Document Chat',
    description: 'Ask documents with graph-backed citations',
    path: '/graphrag',
    icon: GraphIcon,
    color: '#0ea5e9',
    section: 'Knowledge & RAG',
    keywords: ['documents', 'document chat', 'graphrag', 'graph rag', 'citations', 'relationships'],
    element: <GraphRAGWorkbenchPage />,
  },
  {
    id: 'entities',
    label: 'Entities',
    title: 'Entities',
    description: 'Manage extracted entities',
    path: '/entities',
    icon: GraphIcon,
    color: '#14b8a6',
    section: 'Knowledge & RAG',
    keywords: ['entity', 'ner', 'extraction'],
    element: <EntityManagerPage />,
  },
  {
    id: 'discovery',
    label: 'Discovery',
    title: 'Discovery',
    description: 'Web search and document review',
    path: '/discovery',
    icon: DiscoveryIcon,
    color: '#8b5cf6',
    section: 'Knowledge & RAG',
    keywords: ['search', 'web', 'review'],
    element: <DiscoveryPage />,
  },
  {
    id: 'code-search',
    label: 'Code Search',
    title: 'Code Search',
    description: 'Search indexed source code',
    path: '/code-search',
    icon: CodeIcon,
    color: '#f97316',
    section: 'Knowledge & RAG',
    keywords: ['code', 'semantic code', 'repository'],
    element: <CodeSearchPage />,
  },
  {
    id: 'hybrid-processing',
    label: 'Hybrid Processing',
    title: 'Hybrid Processing',
    description: 'Process code and documents together',
    path: '/hybrid-processing',
    icon: HybridIcon,
    color: '#0ea5e9',
    section: 'Knowledge & RAG',
    keywords: ['hybrid', 'processing', 'ingest'],
    element: <HybridProcessingPage />,
  },
  {
    id: 'reasoning',
    label: 'Reasoning Lab',
    title: 'Reasoning Lab',
    description: 'Explore GraphRAG reasoning',
    path: '/reasoning',
    icon: ReasoningIcon,
    color: '#ec4899',
    section: 'Knowledge & RAG',
    keywords: ['reasoning', 'multi-hop', 'causal', 'relationship'],
    element: <ReasoningPlaygroundPage />,
  },
  {
    id: 'retrieval-analytics',
    label: 'Retrieval Quality',
    title: 'Retrieval Quality',
    description: 'Analyze retrieval performance',
    path: '/retrieval-analytics',
    icon: MonitoringIcon,
    color: '#10b981',
    section: 'Knowledge & RAG',
    keywords: ['retrieval', 'quality', 'analytics', 'rag'],
    element: <RetrievalAnalyticsPage />,
  },
  {
    id: 'mcp',
    label: 'MCP Servers',
    title: 'MCP Servers',
    description: 'Manage tool server connections',
    path: '/mcp',
    icon: ExtensionIcon,
    color: '#ec4899',
    section: 'Operations',
    keywords: ['mcp', 'tools', 'servers', 'connectors'],
    element: <MCPServersPage />,
  },
  {
    id: 'monitoring',
    label: 'Monitoring',
    title: 'Monitoring',
    description: 'System metrics and logs',
    path: '/monitoring',
    icon: MonitoringIcon,
    color: '#0ea5e9',
    section: 'Operations',
    keywords: ['logs', 'metrics', 'status'],
    element: <MonitoringPage />,
  },
  {
    id: 'api-docs',
    label: 'API Docs',
    title: 'API Docs',
    description: 'Interactive API documentation',
    path: '/docs',
    icon: ApiIcon,
    color: '#14b8a6',
    section: 'Operations',
    keywords: ['swagger', 'openapi', 'docs'],
    element: <ApiDocsPage />,
  },
  {
    id: 'configuration',
    label: 'Settings',
    title: 'Settings',
    description: 'Application settings',
    path: '/configuration',
    icon: ConfigIcon,
    color: '#64748b',
    section: 'Operations',
    keywords: ['config', 'preferences', 'settings'],
    element: <ConfigurationPage />,
  },
]

export const appRedirects: AppRedirect[] = [
  { path: '/', element: <PreservingRedirect to="/dashboard" /> },
  { path: '/knowledge', element: <PreservingRedirect to="/search" /> },
  { path: '/rag', element: <PreservingRedirect to="/search" /> },
  { path: '/prompt-library', element: <PreservingRedirect to="/prompts" /> },
  { path: '/chat-templates', element: <PreservingRedirect to="/templates" /> },
  { path: '/model-registry', element: <PreservingRedirect to="/registry" /> },
  { path: '/embeddings', element: <PreservingRedirect to="/embedding-deploy" /> },
  { path: '/speech-to-text', element: <PreservingRedirect to="/stt-deploy" /> },
  { path: '/text-to-speech', element: <PreservingRedirect to="/tts-deploy" /> },
  { path: '/retrieval-quality', element: <PreservingRedirect to="/retrieval-analytics" /> },
  { path: '/fine-tuning', element: <PreservingRedirect to="/finetuning" /> },
  { path: '/fine-tuning/templates', element: <PreservingRedirect to="/finetuning/templates" /> },
  { path: '/fine-tuning/datasets', element: <PreservingRedirect to="/finetuning/datasets" /> },
  { path: '/fine-tuning/distillation', element: <PreservingRedirect to="/finetuning/distillation" /> },
  { path: '/fine-tuning/evaluation', element: <PreservingRedirect to="/finetuning/evaluation" /> },
  { path: '/fine-tuning/compare', element: <PreservingRedirect to="/finetuning/compare" /> },
  { path: '/fine-tuning/compare-adapters', element: <PreservingRedirect to="/finetuning/compare" /> },
  { path: '/fine-tuning/book-datasets', element: <PreservingRedirect to="/finetuning/book-datasets" /> },
  { path: '/fine-tuning/benchmarks', element: <PreservingRedirect to="/finetuning/benchmarks" /> },
  { path: '/finetuning/compare-adapters', element: <PreservingRedirect to="/finetuning/compare" /> },
  { path: '/datasets', element: <PreservingRedirect to="/finetuning/datasets" /> },
  { path: '/distillation', element: <PreservingRedirect to="/finetuning/distillation" /> },
  { path: '/evaluation', element: <PreservingRedirect to="/finetuning/evaluation" /> },
  { path: '/benchmark', element: <PreservingRedirect to="/benchmarks" /> },
  { path: '/benchmark-comparison', element: <PreservingRedirect to="/finetuning/benchmarks" /> },
  { path: '/api-docs', element: <PreservingRedirect to="/docs" /> },
  { path: '/settings', element: <PreservingRedirect to="/configuration" /> },
  { path: '/compare', element: <PreservingRedirect to="/finetuning/compare" /> },
  { path: '/testing', element: <PreservingRedirect to="/chat" /> },
  { path: '/batch', element: <PreservingRedirect to="/workflows" /> },
  { path: '*', element: <PreservingRedirect to="/dashboard" preserveLocation={false} /> },
]

const pageById = new Map(appPages.map((page) => [page.id, page]))
const pageByPath = new Map(appPages.map((page) => [page.path, page]))

const getPage = (id: string): AppPage => {
  const page = pageById.get(id)

  if (!page) {
    throw new Error(`Unknown navigation page: ${id}`)
  }

  return page
}

export const getRoutePath = (id: string): string => getPage(id).path

const toDashboardAction = (
  id: string,
  overrides: Partial<Omit<DashboardActionItem, 'id' | 'path' | 'icon' | 'color'>> = {}
): DashboardActionItem => {
  const page = getPage(id)

  return {
    id: page.id,
    label: page.label,
    description: page.description,
    path: page.path,
    icon: page.icon,
    color: page.color,
    ...overrides,
  }
}

const toDashboardSurface = (
  id: string,
  overrides: Partial<Omit<DashboardSurfaceItem, 'id' | 'path' | 'icon' | 'color'>> = {}
): DashboardSurfaceItem => {
  const page = getPage(id)

  return {
    id: page.id,
    label: page.label,
    detail: page.description,
    path: page.path,
    icon: page.icon,
    color: page.color,
    ...overrides,
  }
}

const toNavigationItem = (
  id: string,
  overrides: Partial<Omit<NavigationItem, 'id' | 'path' | 'icon' | 'color'>> = {}
): NavigationItem => {
  const page = getPage(id)

  return {
    id: page.id,
    label: page.label,
    path: page.path,
    icon: page.icon,
    color: page.color,
    description: page.description,
    keywords: page.keywords,
    ...overrides,
  }
}

export const dashboardQuickActions: DashboardActionItem[] = [
  toDashboardAction('chat', { description: 'Talk to a model' }),
  toDashboardAction('deploy', { label: 'Deploy', description: 'Launch runtime' }),
  toDashboardAction('models', { description: 'Manage weights' }),
  toDashboardAction('finetuning-distillation', { label: 'Distill', description: 'Create data' }),
  toDashboardAction('finetuning-evaluation', { label: 'Evaluate', description: 'Score adapters' }),
  toDashboardAction('monitoring', { label: 'Monitor', description: 'System health' }),
]

export const dashboardModelSurfaces: DashboardSurfaceItem[] = [
  toDashboardSurface('deploy', { label: 'Language Models', detail: 'Chat, tools, reasoning, MTP' }),
  toDashboardSurface('embedding-deploy', { detail: 'Vector search and RAG' }),
  toDashboardSurface('stt-deploy', { detail: 'File and recording transcription' }),
  toDashboardSurface('streaming-stt-deploy', { label: 'Streaming Speech', detail: 'Live WebSocket transcription' }),
  toDashboardSurface('tts-deploy', { label: 'Voice', detail: 'Text-to-speech services' }),
  toDashboardSurface('mcp', { label: 'Tools', detail: 'MCP and workflows' }),
  toDashboardSurface('registry', { detail: 'Local catalog and cache' }),
]

export const modelWorkflowActions: DashboardSurfaceItem[] = [
  toDashboardSurface('deploy', { label: 'Deploy', detail: 'Launch a selected model with runtime settings' }),
  toDashboardSurface('quantization', { label: 'Quantize', detail: 'Convert larger models into local-friendly formats' }),
  toDashboardSurface('finetuning', { label: 'Fine-tune', detail: 'Create adapters and training datasets' }),
  toDashboardSurface('finetuning-evaluation', { label: 'Evaluate', detail: 'Score models and adapters before promotion' }),
  toDashboardSurface('benchmark', { label: 'Benchmark', detail: 'Measure throughput, latency, and context scaling' }),
]

export const dashboardWorkflowRunway: DashboardWorkflowStep[] = [
  {
    ...toDashboardAction('models', { label: 'Assets' }),
    phase: '01',
    title: 'Asset Inventory',
    summary: 'Weights, variants, registry records, templates, and prompt artifacts.',
    proof: 'Ready for deployment',
    primaryAction: toDashboardAction('models', { label: 'Manage Models' }),
    secondaryAction: toDashboardAction('registry', { label: 'Open Registry' }),
  },
  {
    ...toDashboardAction('deploy', { label: 'Serve' }),
    phase: '02',
    title: 'Local Serving',
    summary: 'LLM, embeddings, transcription, streaming speech, and voice runtimes.',
    proof: 'Backend and endpoint',
    primaryAction: toDashboardAction('deploy', { label: 'Launch LLM' }),
    secondaryAction: toDashboardAction('embedding-deploy', { label: 'Embeddings' }),
  },
  {
    ...toDashboardAction('chat', { label: 'Interact' }),
    phase: '03',
    title: 'Context Workbench',
    summary: 'Chats, document search, tool calls, and workflow runs.',
    proof: 'Grounded sessions',
    primaryAction: toDashboardAction('chat', { label: 'Open Chat' }),
    secondaryAction: toDashboardAction('search', { label: 'Search RAG' }),
  },
  {
    ...toDashboardAction('finetuning-distillation', { label: 'Improve' }),
    phase: '04',
    title: 'Training Loop',
    summary: 'Distillation data, LoRA adapters, evaluations, and comparisons.',
    proof: 'Scores and lineage',
    primaryAction: toDashboardAction('finetuning-distillation', { label: 'Distill' }),
    secondaryAction: toDashboardAction('finetuning-evaluation', { label: 'Evaluate' }),
  },
  {
    ...toDashboardAction('benchmark', { label: 'Operate' }),
    phase: '05',
    title: 'Operations Loop',
    summary: 'Benchmarks, retrieval quality, telemetry, logs, and service health.',
    proof: 'Throughput and reliability',
    primaryAction: toDashboardAction('benchmark', { label: 'Benchmark' }),
    secondaryAction: toDashboardAction('monitoring', { label: 'Monitor' }),
  },
]

export const dashboardLifecycleStages: DashboardLifecycleStage[] = [
  {
    ...toDashboardAction('models'),
    title: 'Curate Model Assets',
    summary: 'Models, metadata, templates, and reusable prompts stay together before anything runs.',
    links: [
      toDashboardAction('models'),
      toDashboardAction('registry'),
      toDashboardAction('templates'),
    ],
  },
  {
    ...toDashboardAction('deploy', { label: 'Deploy Local Runtimes' }),
    title: 'Deploy Local Runtimes',
    summary: 'Launch LLM, embedding, speech-to-text, and voice services from one operational surface.',
    links: [
      toDashboardAction('deploy', { label: 'LLM' }),
      toDashboardAction('embedding-deploy'),
      toDashboardAction('stt-deploy', { label: 'Speech' }),
      toDashboardAction('streaming-stt-deploy', { label: 'Streaming' }),
      toDashboardAction('tts-deploy', { label: 'Voice' }),
    ],
  },
  {
    ...toDashboardAction('chat', { label: 'Interact With Context' }),
    title: 'Interact With Context',
    summary: 'Chat, retrieve documents, inspect entities, and reason across the local knowledge graph.',
    links: [
      toDashboardAction('chat'),
      toDashboardAction('search'),
      toDashboardAction('knowledge-graph', { label: 'Graph' }),
    ],
  },
  {
    ...toDashboardAction('finetuning', { label: 'Distill, Tune, Evaluate' }),
    title: 'Distill, Tune, Evaluate',
    summary: 'Move from raw examples to adapters, benchmarks, and quality comparisons without leaving the app.',
    links: [
      toDashboardAction('finetuning'),
      toDashboardAction('finetuning-distillation'),
      toDashboardAction('finetuning-evaluation'),
    ],
  },
  {
    ...toDashboardAction('benchmark', { label: 'Benchmark And Operate' }),
    title: 'Benchmark And Operate',
    summary: 'Track throughput, resource use, token volume, logs, and service health as models run.',
    links: [
      toDashboardAction('benchmark'),
      toDashboardAction('retrieval-analytics', { label: 'Quality' }),
      toDashboardAction('monitoring'),
    ],
  },
]

export const navigationSections: NavigationSection[] = [
  {
    id: 'overview',
    title: 'Overview',
    items: [toNavigationItem('dashboard')],
  },
  {
    id: 'build-run',
    title: 'Build & Run',
    items: [
      toNavigationItem('chat'),
      toNavigationItem('deploy', {
        label: 'Deploy',
        description: 'Launch local model runtimes',
        keywords: ['start', 'launch', 'run', 'server', 'llm', 'embedding', 'speech', 'voice'],
        subItems: [
          toNavigationItem('embedding-deploy'),
          toNavigationItem('stt-deploy'),
          toNavigationItem('streaming-stt-deploy'),
          toNavigationItem('tts-deploy'),
        ],
      }),
      toNavigationItem('workflows'),
    ],
  },
  {
    id: 'models-training',
    title: 'Models & Training',
    items: [
      toNavigationItem('models'),
      toNavigationItem('registry'),
      toNavigationItem('benchmark'),
      toNavigationItem('quantization'),
      toNavigationItem('finetuning', {
        subItems: [
          toNavigationItem('finetuning-templates'),
          toNavigationItem('finetuning-datasets'),
          toNavigationItem('finetuning-distillation'),
          toNavigationItem('finetuning-evaluation'),
          toNavigationItem('finetuning-compare'),
          toNavigationItem('finetuning-book-datasets'),
          toNavigationItem('finetuning-benchmarks'),
        ],
      }),
    ],
  },
  {
    id: 'prompts-templates',
    title: 'Prompts & Templates',
    items: [toNavigationItem('prompts'), toNavigationItem('templates')],
  },
  {
    id: 'knowledge-rag',
    title: 'Knowledge & RAG',
    items: [
      toNavigationItem('search'),
      toNavigationItem('documents'),
      toNavigationItem('knowledge-graph', {
        subItems: [
          toNavigationItem('entities'),
          toNavigationItem('reasoning'),
        ],
      }),
      toNavigationItem('discovery'),
      toNavigationItem('code-search'),
      toNavigationItem('hybrid-processing'),
      toNavigationItem('retrieval-analytics'),
    ],
  },
  {
    id: 'operations',
    title: 'Operations',
    items: [
      toNavigationItem('monitoring'),
      toNavigationItem('mcp'),
      toNavigationItem('api-docs'),
      toNavigationItem('configuration'),
    ],
  },
]

const toSearchableItems = (): SearchableNavigationItem[] => {
  const seen = new Set<string>()

  const collectItem = (
    item: NavigationItem,
    sectionTitle: string,
    parentLabels: string[] = [],
  ): SearchableNavigationItem[] => {
    const items: SearchableNavigationItem[] = []
    const itemIcon = item.icon || DashboardIcon
    const itemColor = item.color || '#6366f1'
    const page = pageByPath.get(item.path)

    if (!seen.has(item.path)) {
      items.push({
        ...item,
        icon: itemIcon,
        color: itemColor,
        title: page?.title || item.label,
        description: item.description || page?.description || '',
        keywords: [...(item.keywords || []), ...parentLabels, sectionTitle],
        parentLabels,
        section: sectionTitle,
      })
      seen.add(item.path)
    }

    item.subItems?.forEach((subItem) => {
      items.push(...collectItem(subItem, sectionTitle, [...parentLabels, item.label]))
    })

    return items
  }

  return navigationSections.flatMap((section) => (
    section.items.flatMap((item) => collectItem(item, section.title))
  ))
}

const normalizePath = (pathname: string): string => {
  const normalized = pathname.split('?')[0].replace(/\/+$/, '')
  return normalized || '/dashboard'
}

const toTrailItem = (
  item: NavigationItem,
  section: string,
  targetPath: string,
): NavigationTrailItem => {
  const page = pageByPath.get(item.path)

  return {
    id: item.id,
    label: item.label,
    path: item.path,
    color: item.color || page?.color || '#6366f1',
    icon: item.icon || page?.icon || DashboardIcon,
    section,
    isCurrent: item.path === targetPath,
  }
}

export const getNavigationTrail = (pathname: string): NavigationTrailItem[] => {
  const targetPath = normalizePath(pathname)
  const home = toTrailItem(toNavigationItem('dashboard'), 'Overview', targetPath)

  if (targetPath === home.path) {
    return [{ ...home, isCurrent: true }]
  }

  let bestMatch: NavigationTrailItem[] = []

  const visitItem = (
    item: NavigationItem,
    section: string,
    ancestors: NavigationItem[],
  ) => {
    const isMatch = targetPath === item.path || targetPath.startsWith(`${item.path}/`)
    const currentTrail = [...ancestors, item].map((trailItem) => toTrailItem(trailItem, section, targetPath))

    if (isMatch && currentTrail.length >= bestMatch.length) {
      bestMatch = currentTrail
    }

    item.subItems?.forEach((subItem) => visitItem(subItem, section, [...ancestors, item]))
  }

  navigationSections.forEach((section) => {
    section.items.forEach((item) => visitItem(item, section.title, []))
  })

  if (bestMatch.length === 0) {
    const routeMeta = getRouteMeta(targetPath)
    return [
      home,
      {
        id: routeMeta.id,
        label: routeMeta.label,
        path: routeMeta.path,
        color: routeMeta.color,
        icon: pageByPath.get(routeMeta.path)?.icon || DashboardIcon,
        section: routeMeta.section,
        isCurrent: true,
      },
    ]
  }

  return [home, ...bestMatch].filter((item, index, items) => (
    index === items.findIndex((candidate) => candidate.path === item.path)
  ))
}

export const searchableNavigationItems = toSearchableItems()

export const routeMetaByPath = appPages.reduce<Record<string, RouteMeta>>((acc, page) => {
  acc[page.path] = {
    id: page.id,
    label: page.label,
    title: page.title,
    description: page.description,
    path: page.path,
    color: page.color,
    section: page.section,
  }
  return acc
}, {})

export const defaultRouteMeta = routeMetaByPath['/dashboard']

export const getRouteMeta = (pathname: string): RouteMeta => {
  const exactMatch = routeMetaByPath[pathname]

  if (exactMatch) {
    return exactMatch
  }

  const parentMatch = [...pageByPath.values()]
    .filter((page) => pathname.startsWith(`${page.path}/`))
    .sort((a, b) => b.path.length - a.path.length)[0]

  return parentMatch ? routeMetaByPath[parentMatch.path] : defaultRouteMeta
}
