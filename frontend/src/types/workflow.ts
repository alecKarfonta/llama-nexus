/**
 * Workflow Builder Type Definitions
 */

// Port types for node inputs/outputs
export type PortType = 'string' | 'number' | 'boolean' | 'object' | 'array' | 'any';

export interface PortDefinition {
  id: string;
  name: string;
  type: PortType;
  required?: boolean;
  description?: string;
}

// Node categories
export type NodeCategory =
  | 'trigger'
  | 'llm'
  | 'rag'
  | 'tools'
  | 'data'
  | 'control'
  | 'api'
  | 'mcp'
  | 'database'
  | 'output';

// Execution status
export type ExecutionStatus = 'idle' | 'pending' | 'running' | 'success' | 'error' | 'cancelled' | 'paused';

// Node type definition (from registry)
export interface NodeTypeDefinition {
  type: string;
  displayName: string;
  category: NodeCategory;
  description: string;
  icon?: string;
  color?: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  configSchema: Record<string, any>; // JSON Schema
}

// Workflow node instance
export interface WorkflowNodeData {
  label: string;
  nodeType: string;
  config: Record<string, any>;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  status?: ExecutionStatus;
  error?: string;
  executionTime?: number;
}

// Connection between nodes
export interface WorkflowConnection {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

// Workflow definition
export interface Workflow {
  id: string;
  name: string;
  description?: string;
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  variables: Record<string, any>;
  settings: WorkflowSettings;
  createdAt: string;
  updatedAt: string;
  version: number;
  isActive: boolean;
}

export interface WorkflowNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: WorkflowNodeData;
}

export interface WorkflowSettings {
  timeout?: number;
  retryOnFailure?: boolean;
  maxRetries?: number;
  logLevel?: 'debug' | 'info' | 'warn' | 'error';
}

// Workflow execution
export interface NodeExecution {
  nodeId: string;
  nodeName: string;
  status: ExecutionStatus;
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  error?: string;
  logs: string[];
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  workflowVersion: number;
  status: ExecutionStatus;
  startedAt: string;
  completedAt?: string;
  triggerData: Record<string, any>;
  nodeExecutions: Record<string, NodeExecution>;
  finalOutput?: any;
  error?: string;
}

// Category metadata for UI
export interface CategoryInfo {
  id: NodeCategory;
  name: string;
  description: string;
  icon: string;
  color: string;
}

// Node registry categories
export const NODE_CATEGORIES: CategoryInfo[] = [
  {
    id: 'trigger',
    name: 'Triggers',
    description: 'Start workflow execution',
    icon: 'PlayArrow',
    color: '#10b981',
  },
  {
    id: 'llm',
    name: 'LLM / Models',
    description: 'Language model operations',
    icon: 'Psychology',
    color: '#6366f1',
  },
  {
    id: 'rag',
    name: 'RAG / Documents',
    description: 'Document retrieval and processing',
    icon: 'Description',
    color: '#8b5cf6',
  },
  {
    id: 'tools',
    name: 'Tools / Functions',
    description: 'Execute tools and code',
    icon: 'Build',
    color: '#f59e0b',
  },
  {
    id: 'data',
    name: 'Data / Transform',
    description: 'Data manipulation and transformation',
    icon: 'Transform',
    color: '#06b6d4',
  },
  {
    id: 'control',
    name: 'Control Flow',
    description: 'Branching, loops, and flow control',
    icon: 'AccountTree',
    color: '#ec4899',
  },
  {
    id: 'api',
    name: 'External APIs',
    description: 'HTTP requests and external services',
    icon: 'Api',
    color: '#14b8a6',
  },
  {
    id: 'mcp',
    name: 'MCP Servers',
    description: 'Model Context Protocol integration',
    icon: 'Hub',
    color: '#a855f7',
  },
  {
    id: 'database',
    name: 'Databases',
    description: 'Database queries and storage',
    icon: 'Storage',
    color: '#f97316',
  },
  {
    id: 'output',
    name: 'Outputs',
    description: 'Workflow outputs and responses',
    icon: 'Output',
    color: '#ef4444',
  },
];

// Built-in node type definitions
export const BUILTIN_NODE_TYPES: NodeTypeDefinition[] = [
  // Triggers
  {
    type: 'manual_trigger',
    displayName: 'Manual Trigger',
    category: 'trigger',
    description: 'Manually start workflow execution',
    color: '#10b981',
    inputs: [],
    outputs: [
      { id: 'trigger', name: 'Trigger', type: 'any' },
    ],
    configSchema: {},
  },
  {
    type: 'http_webhook',
    displayName: 'HTTP Webhook',
    category: 'trigger',
    description: 'Trigger workflow via HTTP request',
    color: '#10b981',
    inputs: [],
    outputs: [
      { id: 'body', name: 'Body', type: 'object' },
      { id: 'headers', name: 'Headers', type: 'object' },
      { id: 'query', name: 'Query Params', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Webhook path' },
        method: { type: 'string', enum: ['GET', 'POST', 'PUT', 'DELETE'] },
      },
    },
  },

  // LLM Nodes
  {
    type: 'llm_chat',
    displayName: 'LLM Chat',
    category: 'llm',
    description: 'Chat completion with local LLM',
    color: '#6366f1',
    inputs: [
      { id: 'messages', name: 'Messages', type: 'array', required: true },
      { id: 'system', name: 'System Prompt', type: 'string' },
    ],
    outputs: [
      { id: 'response', name: 'Response', type: 'string' },
      { id: 'usage', name: 'Token Usage', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        model: { type: 'string', format: 'model-select', description: 'Select a deployed model', required: true },
        temperature: { type: 'number', minimum: 0, maximum: 2, default: 0.7 },
        maxTokens: { type: 'integer', minimum: 1, default: 2048 },
      },
    },
  },
  {
    type: 'openai_chat',
    displayName: 'OpenAI Chat',
    category: 'llm',
    description: 'Chat completion via OpenAI API',
    color: '#6366f1',
    inputs: [
      { id: 'messages', name: 'Messages', type: 'array', required: true },
      { id: 'system', name: 'System Prompt', type: 'string' },
    ],
    outputs: [
      { id: 'response', name: 'Response', type: 'string' },
      { id: 'usage', name: 'Token Usage', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        model: { type: 'string', default: 'gpt-4' },
        apiKey: { type: 'string', format: 'password' },
        temperature: { type: 'number', minimum: 0, maximum: 2, default: 0.7 },
        maxTokens: { type: 'integer', minimum: 1 },
      },
    },
  },
  {
    type: 'openai_api_llm',
    displayName: 'OpenAI API LLM',
    category: 'llm',
    description: 'Advanced OpenAI-compatible API endpoint with full configuration',
    color: '#6366f1',
    inputs: [
      { id: 'messages', name: 'Messages', type: 'array', required: true },
      { id: 'system', name: 'System Prompt', type: 'string' },
      { id: 'tools', name: 'Tools', type: 'array', description: 'Function calling tools' },
    ],
    outputs: [
      { id: 'response', name: 'Response', type: 'string' },
      { id: 'usage', name: 'Token Usage', type: 'object' },
      { id: 'tool_calls', name: 'Tool Calls', type: 'array' },
      { id: 'finish_reason', name: 'Finish Reason', type: 'string' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        // Connection settings
        endpoint: { 
          type: 'string', 
          description: 'API endpoint URL',
          default: 'https://api.openai.com/v1/chat/completions',
          format: 'url'
        },
        apiKey: { 
          type: 'string', 
          format: 'password',
          description: 'API key for authentication'
        },
        organization: {
          type: 'string',
          description: 'Optional organization ID'
        },
        
        // Model settings
        model: { 
          type: 'string', 
          default: 'gpt-4',
          description: 'Model name (e.g., gpt-4, gpt-4-turbo, gpt-3.5-turbo, o1, o1-mini)'
        },
        
        // Generation parameters
        temperature: { 
          type: 'number', 
          minimum: 0, 
          maximum: 2, 
          default: 0.7,
          description: 'Controls randomness (0=deterministic, 2=very random)'
        },
        topP: { 
          type: 'number', 
          minimum: 0, 
          maximum: 1, 
          default: 1,
          description: 'Nucleus sampling probability'
        },
        topK: {
          type: 'integer',
          minimum: 1,
          maximum: 200,
          description: 'Top-k sampling (if supported by model)'
        },
        maxTokens: { 
          type: 'integer', 
          minimum: 1,
          maximum: 128000,
          description: 'Maximum tokens to generate'
        },
        presencePenalty: {
          type: 'number',
          minimum: -2,
          maximum: 2,
          default: 0,
          description: 'Penalize tokens based on presence (-2 to 2)'
        },
        frequencyPenalty: {
          type: 'number',
          minimum: -2,
          maximum: 2,
          default: 0,
          description: 'Penalize tokens based on frequency (-2 to 2)'
        },
        repetitionPenalty: {
          type: 'number',
          minimum: 0.1,
          maximum: 2,
          description: 'Alternative repetition penalty (if supported)'
        },
        
        // Thinking/reasoning (for o1 models)
        thinkingLevel: {
          type: 'integer',
          minimum: 0,
          maximum: 10,
          default: 0,
          description: 'Thinking level for reasoning models (0=none, 10=maximum)'
        },
        
        // Advanced settings
        stream: {
          type: 'boolean',
          default: false,
          description: 'Enable streaming responses'
        },
        n: {
          type: 'integer',
          minimum: 1,
          maximum: 10,
          default: 1,
          description: 'Number of completions to generate'
        },
        stop: {
          type: 'array',
          items: { type: 'string' },
          description: 'Stop sequences'
        },
        logprobs: {
          type: 'boolean',
          default: false,
          description: 'Return log probabilities'
        },
        topLogprobs: {
          type: 'integer',
          minimum: 0,
          maximum: 20,
          description: 'Number of top log probabilities to return'
        },
        seed: {
          type: 'integer',
          description: 'Random seed for deterministic generation'
        },
        responseFormat: {
          type: 'object',
          properties: {
            type: { 
              type: 'string', 
              enum: ['text', 'json_object', 'json_schema'],
              default: 'text'
            },
            schema: {
              type: 'object',
              description: 'JSON schema for structured output'
            }
          }
        },
        
        // Request configuration
        timeout: {
          type: 'integer',
          minimum: 1,
          maximum: 600,
          default: 120,
          description: 'Request timeout in seconds'
        },
        retryAttempts: {
          type: 'integer',
          minimum: 0,
          maximum: 5,
          default: 2,
          description: 'Number of retry attempts on failure'
        },
        retryDelay: {
          type: 'integer',
          minimum: 100,
          maximum: 10000,
          default: 1000,
          description: 'Delay between retries in milliseconds'
        },
        
        // Custom headers
        customHeaders: {
          type: 'object',
          description: 'Additional HTTP headers',
          additionalProperties: { type: 'string' }
        },
      },
      required: ['endpoint', 'apiKey', 'model'],
    },
  },
  {
    type: 'embedding',
    displayName: 'Generate Embedding',
    category: 'llm',
    description: 'Generate text embeddings',
    color: '#6366f1',
    inputs: [
      { id: 'text', name: 'Text', type: 'string', required: true },
    ],
    outputs: [
      { id: 'embedding', name: 'Embedding', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        model: { type: 'string', format: 'model-select', description: 'Select an embedding model' },
      },
    },
  },

  // RAG Nodes
  {
    type: 'document_loader',
    displayName: 'Document Loader',
    category: 'rag',
    description: 'Load documents from various sources',
    color: '#8b5cf6',
    inputs: [
      { id: 'source', name: 'Source', type: 'string', required: true },
    ],
    outputs: [
      { id: 'documents', name: 'Documents', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        sourceType: { type: 'string', enum: ['file', 'url', 'directory'] },
        recursive: { type: 'boolean', default: true },
      },
    },
  },
  {
    type: 'chunker',
    displayName: 'Text Chunker',
    category: 'rag',
    description: 'Split documents into chunks',
    color: '#8b5cf6',
    inputs: [
      { id: 'documents', name: 'Documents', type: 'array', required: true },
    ],
    outputs: [
      { id: 'chunks', name: 'Chunks', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        strategy: { type: 'string', enum: ['fixed', 'recursive', 'semantic'] },
        chunkSize: { type: 'integer', default: 512 },
        overlap: { type: 'integer', default: 50 },
      },
    },
  },
  {
    type: 'retriever',
    displayName: 'Semantic Search',
    category: 'rag',
    description: 'Retrieve relevant documents',
    color: '#8b5cf6',
    inputs: [
      { id: 'query', name: 'Query', type: 'string', required: true },
    ],
    outputs: [
      { id: 'documents', name: 'Documents', type: 'array' },
      { id: 'scores', name: 'Scores', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        collection: { type: 'string', description: 'Vector collection name' },
        k: { type: 'integer', default: 5 },
        threshold: { type: 'number', minimum: 0, maximum: 1 },
      },
    },
  },
  {
    type: 'vector_store',
    displayName: 'Vector Store',
    category: 'rag',
    description: 'Store or query vector database',
    color: '#8b5cf6',
    inputs: [
      { id: 'data', name: 'Data', type: 'any', required: true },
    ],
    outputs: [
      { id: 'result', name: 'Result', type: 'any' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        operation: { type: 'string', enum: ['upsert', 'query', 'delete'] },
        collection: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_search',
    displayName: 'GraphRAG Hybrid Search',
    category: 'rag',
    description: 'Hybrid search with LLM answer generation (vector + graph + keyword)',
    color: '#8b5cf6',
    inputs: [
      { id: 'query', name: 'Query', type: 'string', required: true },
      { id: 'domain', name: 'Domain', type: 'string' },
    ],
    outputs: [
      { id: 'answer', name: 'Answer', type: 'string' },
      { id: 'documents', name: 'Documents', type: 'array' },
      { id: 'sources', name: 'Sources', type: 'array' },
      { id: 'entities', name: 'Entities', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        searchType: { 
          type: 'string', 
          enum: ['auto', 'hybrid', 'vector', 'graph', 'keyword'],
          default: 'auto',
          description: 'Search method (auto selects best)'
        },
        topK: { type: 'integer', default: 10, minimum: 1, maximum: 50 },
      },
    },
  },
  {
    type: 'entity_extraction',
    displayName: 'Extract Entities',
    category: 'rag',
    description: 'Extract entities and relationships using GLiNER',
    color: '#8b5cf6',
    inputs: [
      { id: 'text', name: 'Text', type: 'string', required: true },
      { id: 'domain', name: 'Domain', type: 'string' },
    ],
    outputs: [
      { id: 'entities', name: 'Entities', type: 'array' },
      { id: 'relationships', name: 'Relationships', type: 'array' },
      { id: 'entity_count', name: 'Entity Count', type: 'number' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'multi_hop_reasoning',
    displayName: 'Multi-Hop Reasoning',
    category: 'rag',
    description: 'Find reasoning paths between entities in knowledge graph',
    color: '#8b5cf6',
    inputs: [
      { id: 'source', name: 'Source Entity', type: 'string', required: true },
      { id: 'target', name: 'Target Entity', type: 'string', required: true },
    ],
    outputs: [
      { id: 'reasoning_path', name: 'Reasoning Path', type: 'array' },
      { id: 'answer', name: 'Answer', type: 'string' },
      { id: 'sources', name: 'Sources', type: 'array' },
      { id: 'hop_count', name: 'Hop Count', type: 'number' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        maxHops: { type: 'integer', default: 3, minimum: 1, maximum: 10 },
      },
    },
  },
  {
    type: 'causal_reasoning',
    displayName: 'Causal Reasoning',
    category: 'rag',
    description: 'Find cause-effect relationships',
    color: '#8b5cf6',
    inputs: [
      { id: 'query', name: 'Query', type: 'string', required: true },
    ],
    outputs: [
      { id: 'causes', name: 'Causes', type: 'array' },
      { id: 'effects', name: 'Effects', type: 'array' },
      { id: 'reasoning', name: 'Reasoning', type: 'string' },
      { id: 'answer', name: 'Answer', type: 'string' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'comparative_reasoning',
    displayName: 'Comparative Analysis',
    category: 'rag',
    description: 'Compare entities and find similarities/differences',
    color: '#8b5cf6',
    inputs: [
      { id: 'entity1', name: 'Entity 1', type: 'string', required: true },
      { id: 'entity2', name: 'Entity 2', type: 'string', required: true },
    ],
    outputs: [
      { id: 'similarities', name: 'Similarities', type: 'array' },
      { id: 'differences', name: 'Differences', type: 'array' },
      { id: 'analysis', name: 'Analysis', type: 'string' },
      { id: 'answer', name: 'Answer', type: 'string' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'entity_linking',
    displayName: 'Entity Linking',
    category: 'rag',
    description: 'Link entity mentions to knowledge graph',
    color: '#8b5cf6',
    inputs: [
      { id: 'entities', name: 'Entities', type: 'array', required: true },
      { id: 'context', name: 'Context', type: 'string' },
    ],
    outputs: [
      { id: 'linked_entities', name: 'Linked Entities', type: 'array' },
      { id: 'disambiguated', name: 'Disambiguated', type: 'array' },
      { id: 'link_count', name: 'Link Count', type: 'number' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'graphrag_document_ingest',
    displayName: 'GraphRAG Document Ingest',
    category: 'rag',
    description: 'Ingest text content into GraphRAG vector store and knowledge graph',
    color: '#8b5cf6',
    inputs: [
      { id: 'text', name: 'Text', type: 'string', required: true },
      { id: 'filename', name: 'Filename', type: 'string' },
      { id: 'domain', name: 'Domain', type: 'string' },
      { id: 'content_type', name: 'Content Type', type: 'string' },
      { id: 'use_semantic_chunking', name: 'Semantic Chunking', type: 'boolean' },
      { id: 'build_knowledge_graph', name: 'Build Graph', type: 'boolean' },
    ],
    outputs: [
      { id: 'status', name: 'Status', type: 'string' },
      { id: 'filename', name: 'Filename', type: 'string' },
      { id: 'domain', name: 'Domain', type: 'string' },
      { id: 'total_files', name: 'Total Files', type: 'number' },
      { id: 'chunks', name: 'Chunks', type: 'array' },
      { id: 'total_chunks', name: 'Total Chunks', type: 'number' },
      { id: 'entities', name: 'Entities', type: 'number' },
      { id: 'relationships', name: 'Relationships', type: 'number' },
      { id: 'result', name: 'Result', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        filename: { type: 'string', default: 'workflow-document.txt' },
        domain: { type: 'string', default: 'general' },
        contentType: { type: 'string', default: 'text/plain' },
        useSemanticChunking: { type: 'boolean', default: true },
        buildKnowledgeGraph: { type: 'boolean', default: true },
      },
    },
  },
  {
    type: 'graphrag_document_list',
    displayName: 'GraphRAG Document List',
    category: 'rag',
    description: 'List documents currently indexed by GraphRAG',
    color: '#8b5cf6',
    inputs: [],
    outputs: [
      { id: 'documents', name: 'Documents', type: 'array' },
      { id: 'total_documents', name: 'Total Documents', type: 'number' },
      { id: 'vector_store_documents', name: 'Vector Documents', type: 'number' },
      { id: 'knowledge_graph_documents', name: 'Graph Documents', type: 'number' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'graphrag_document_delete',
    displayName: 'GraphRAG Document Delete',
    category: 'rag',
    description: 'Delete a named document from GraphRAG',
    color: '#8b5cf6',
    inputs: [
      { id: 'document_name', name: 'Document Name', type: 'string', required: true },
      { id: 'filename', name: 'Filename', type: 'string' },
    ],
    outputs: [
      { id: 'document_name', name: 'Document Name', type: 'string' },
      { id: 'status', name: 'Status', type: 'string' },
      { id: 'message', name: 'Message', type: 'string' },
      { id: 'deleted', name: 'Deleted', type: 'boolean' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        documentName: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_kg_rebuild',
    displayName: 'GraphRAG KG Rebuild',
    category: 'rag',
    description: 'Rebuild the GraphRAG knowledge graph from indexed documents',
    color: '#8b5cf6',
    inputs: [
      { id: 'domain', name: 'Domain', type: 'string' },
    ],
    outputs: [
      { id: 'domain', name: 'Domain', type: 'string' },
      { id: 'message', name: 'Message', type: 'string' },
      { id: 'documents_processed', name: 'Documents Processed', type: 'number' },
      { id: 'entities_extracted', name: 'Entities Extracted', type: 'number' },
      { id: 'relationships_extracted', name: 'Relationships Extracted', type: 'number' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        domain: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_capabilities',
    displayName: 'GraphRAG Capabilities',
    category: 'rag',
    description: 'Inspect enabled GraphRAG services, capability groups, and bridge coverage',
    color: '#8b5cf6',
    inputs: [],
    outputs: [
      { id: 'enabled', name: 'Enabled', type: 'boolean' },
      { id: 'services', name: 'Services', type: 'object' },
      { id: 'capabilities', name: 'Capabilities', type: 'array' },
      { id: 'capability_ids', name: 'Capability IDs', type: 'array' },
      { id: 'capability_count', name: 'Capability Count', type: 'number' },
      { id: 'bridge_available', name: 'Bridge Available', type: 'boolean' },
      { id: 'allowed_prefixes', name: 'Allowed Prefixes', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'graphrag_query_complexity',
    displayName: 'Query Complexity',
    category: 'rag',
    description: 'Classify GraphRAG query complexity and required reasoning strategy',
    color: '#8b5cf6',
    inputs: [
      { id: 'query', name: 'Query', type: 'string', required: true },
    ],
    outputs: [
      { id: 'query', name: 'Query', type: 'string' },
      { id: 'primary_reasoning', name: 'Primary Reasoning', type: 'string' },
      { id: 'detected_patterns', name: 'Detected Patterns', type: 'array' },
      { id: 'complexity_level', name: 'Complexity Level', type: 'string' },
      { id: 'requires_multi_hop', name: 'Requires Multi-Hop', type: 'boolean' },
      { id: 'analysis', name: 'Analysis', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'enhanced_entity_extraction',
    displayName: 'Enhanced Extraction',
    category: 'rag',
    description: 'Extract entities, relationships, claims, and linked entities using GraphRAG',
    color: '#8b5cf6',
    inputs: [
      { id: 'text', name: 'Text', type: 'string', required: true },
      { id: 'domain', name: 'Domain', type: 'string' },
    ],
    outputs: [
      { id: 'entities', name: 'Entities', type: 'array' },
      { id: 'relationships', name: 'Relationships', type: 'array' },
      { id: 'entity_count', name: 'Entity Count', type: 'number' },
      { id: 'relationship_count', name: 'Relationship Count', type: 'number' },
      { id: 'claims', name: 'Claims', type: 'array' },
      { id: 'extraction_metadata', name: 'Extraction Metadata', type: 'object' },
      { id: 'linked_entities', name: 'Linked Entities', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        useSpanbert: { type: 'boolean', default: true },
        useDependency: { type: 'boolean', default: true },
        useEntityLinking: { type: 'boolean', default: true },
      },
    },
  },
  {
    type: 'graphrag_ner_extraction',
    displayName: 'NER Extraction',
    category: 'rag',
    description: 'Extract entities from one or more texts using GraphRAG GLiNER',
    color: '#8b5cf6',
    inputs: [
      { id: 'text', name: 'Text', type: 'string' },
      { id: 'texts', name: 'Texts', type: 'array' },
    ],
    outputs: [
      { id: 'entities', name: 'Entities', type: 'array' },
      { id: 'entity_count', name: 'Entity Count', type: 'number' },
      { id: 'results', name: 'Results', type: 'array' },
      { id: 'total_texts', name: 'Total Texts', type: 'number' },
      { id: 'extraction_method', name: 'Extraction Method', type: 'string' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'graphrag_relation_extraction',
    displayName: 'Relationship Extraction',
    category: 'rag',
    description: 'Extract typed relationships from text using GraphRAG relationship extraction',
    color: '#8b5cf6',
    inputs: [
      { id: 'text', name: 'Text', type: 'string' },
      { id: 'texts', name: 'Texts', type: 'array' },
      { id: 'relations', name: 'Relations', type: 'array' },
      { id: 'entity_labels', name: 'Entity Labels', type: 'array' },
      { id: 'threshold', name: 'Threshold', type: 'number' },
    ],
    outputs: [
      { id: 'relations', name: 'Relations', type: 'array' },
      { id: 'relation_count', name: 'Relation Count', type: 'number' },
      { id: 'results', name: 'Results', type: 'array' },
      { id: 'total_texts', name: 'Total Texts', type: 'number' },
      { id: 'processing_time', name: 'Processing Time', type: 'number' },
      { id: 'model_info', name: 'Model Info', type: 'object' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        threshold: { type: 'number', default: 0.5, minimum: 0, maximum: 1 },
      },
    },
  },
  {
    type: 'graphrag_enhanced_query',
    displayName: 'Enhanced Query',
    category: 'rag',
    description: 'Process a query with graph reasoning, entity linking, and enhanced metadata',
    color: '#8b5cf6',
    inputs: [
      { id: 'query', name: 'Query', type: 'string', required: true },
      { id: 'domain', name: 'Domain', type: 'string' },
      { id: 'filters', name: 'Filters', type: 'object' },
    ],
    outputs: [
      { id: 'answer', name: 'Answer', type: 'string' },
      { id: 'results', name: 'Results', type: 'array' },
      { id: 'reasoning_paths', name: 'Reasoning Paths', type: 'array' },
      { id: 'inferred_relationships', name: 'Inferred Relationships', type: 'array' },
      { id: 'entity_clusters', name: 'Entity Clusters', type: 'array' },
      { id: 'confidence', name: 'Confidence', type: 'number' },
      { id: 'explanation', name: 'Explanation', type: 'string' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        searchType: { type: 'string', enum: ['vector', 'graph', 'keyword', 'hybrid'], default: 'hybrid' },
        topK: { type: 'integer', default: 10, minimum: 1, maximum: 50 },
      },
    },
  },
  {
    type: 'graphrag_chat_context',
    displayName: 'Chat Context',
    category: 'rag',
    description: 'Build graph-enhanced or unified RAG plus graph context for a query',
    color: '#8b5cf6',
    inputs: [
      { id: 'query', name: 'Query', type: 'string', required: true },
      { id: 'mode', name: 'Mode', type: 'string' },
      { id: 'top_k', name: 'Top K', type: 'number' },
      { id: 'domain', name: 'Domain', type: 'string' },
      { id: 'rag_enabled', name: 'RAG Enabled', type: 'boolean' },
      { id: 'rag_domains', name: 'RAG Domains', type: 'array' },
      { id: 'rag_search_mode', name: 'RAG Search Mode', type: 'string' },
      { id: 'pipeline_run_ids', name: 'Pipeline Run IDs', type: 'array' },
    ],
    outputs: [
      { id: 'mode', name: 'Mode', type: 'string' },
      { id: 'context_text', name: 'Context Text', type: 'string' },
      { id: 'chunks', name: 'Chunks', type: 'array' },
      { id: 'entities', name: 'Entities', type: 'array' },
      { id: 'relationships', name: 'Relationships', type: 'array' },
      { id: 'search_results', name: 'Search Results', type: 'array' },
      { id: 'graph_connections', name: 'Graph Connections', type: 'array' },
      { id: 'expanded_entities', name: 'Expanded Entities', type: 'array' },
      { id: 'cross_references', name: 'Cross References', type: 'array' },
      { id: 'quality_signals', name: 'Quality Signals', type: 'object' },
      { id: 'services_used', name: 'Services Used', type: 'object' },
      { id: 'errors', name: 'Errors', type: 'array' },
      { id: 'source', name: 'Source', type: 'string' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        mode: { type: 'string', enum: ['v1', 'v2', 'unified'], default: 'v2' },
        topK: { type: 'integer', default: 5, minimum: 1, maximum: 50 },
        domain: { type: 'string', default: 'general' },
        ragEnabled: { type: 'boolean', default: true },
        ragDomains: { type: 'array', items: { type: 'string' } },
        ragSearchMode: { type: 'string', enum: ['domain', 'global'], default: 'domain' },
        pipelineRunIds: { type: 'array', items: { type: 'string' } },
      },
    },
  },
  {
    type: 'graphrag_instrumentation',
    displayName: 'GraphRAG Instrumentation',
    category: 'rag',
    description: 'Collect query statistics, graph statistics, centrality, and clusters',
    color: '#8b5cf6',
    inputs: [],
    outputs: [
      { id: 'query_statistics', name: 'Query Statistics', type: 'object' },
      { id: 'graph_statistics', name: 'Graph Statistics', type: 'object' },
      { id: 'entity_centrality', name: 'Entity Centrality', type: 'object' },
      { id: 'entity_clusters', name: 'Entity Clusters', type: 'array' },
      { id: 'cluster_count', name: 'Cluster Count', type: 'number' },
      { id: 'query_response', name: 'Query Response', type: 'object' },
      { id: 'graph_response', name: 'Graph Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'graphrag_evaluation_run',
    displayName: 'Evaluation Run',
    category: 'rag',
    description: 'Run GraphRAG quality, retrieval, chunking, or comprehensive evaluations',
    color: '#8b5cf6',
    inputs: [
      { id: 'evaluation_type', name: 'Evaluation Type', type: 'string' },
      { id: 'payload', name: 'Payload', type: 'object' },
      { id: 'test_documents', name: 'Test Documents', type: 'array' },
      { id: 'ground_truth', name: 'Ground Truth', type: 'object' },
      { id: 'test_queries', name: 'Test Queries', type: 'array' },
      { id: 'expected_answers', name: 'Expected Answers', type: 'array' },
      { id: 'relevance_scores', name: 'Relevance Scores', type: 'array' },
    ],
    outputs: [
      { id: 'evaluation_type', name: 'Evaluation Type', type: 'string' },
      { id: 'result', name: 'Result', type: 'object' },
      { id: 'success', name: 'Success', type: 'boolean' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        evaluationType: {
          type: 'string',
          enum: [
            'graph_completeness',
            'entity_extraction',
            'query_responses',
            'retrieval_relevance',
            'comprehensive',
            'chunking',
            'chunking_comparative',
            'retrieval',
            'retrieval_comparative',
          ],
          default: 'graph_completeness',
        },
      },
    },
  },
  {
    type: 'graphrag_test_suite',
    displayName: 'GraphRAG Test Suite',
    category: 'rag',
    description: 'Run GraphRAG unit, integration, performance, quality, or comprehensive tests',
    color: '#8b5cf6',
    inputs: [
      { id: 'suite', name: 'Suite', type: 'string' },
    ],
    outputs: [
      { id: 'suite', name: 'Suite', type: 'string' },
      { id: 'result', name: 'Result', type: 'object' },
      { id: 'success', name: 'Success', type: 'boolean' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        suite: { type: 'string', enum: ['unit', 'integration', 'performance', 'quality', 'comprehensive'], default: 'quality' },
      },
    },
  },
  {
    type: 'graphrag_domain_stats',
    displayName: 'Domain Stats',
    category: 'rag',
    description: 'Get per-domain GraphRAG knowledge graph statistics',
    color: '#8b5cf6',
    inputs: [],
    outputs: [
      { id: 'domain_stats', name: 'Domain Stats', type: 'object' },
      { id: 'domains', name: 'Domains', type: 'array' },
      { id: 'domain_count', name: 'Domain Count', type: 'number' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'graphrag_graph_explorer',
    displayName: 'Graph Explorer',
    category: 'rag',
    description: 'Fetch filtered graph data, neighborhoods, communities, top entities, top relationships, or graph exports',
    color: '#8b5cf6',
    inputs: [
      { id: 'mode', name: 'Mode', type: 'string' },
      { id: 'filters', name: 'Filters', type: 'object' },
      { id: 'domain', name: 'Domain', type: 'string' },
      { id: 'entity_name', name: 'Entity Name', type: 'string' },
      { id: 'entity', name: 'Entity', type: 'string' },
      { id: 'max_entities', name: 'Max Entities', type: 'number' },
      { id: 'max_relationships', name: 'Max Relationships', type: 'number' },
      { id: 'min_occurrence', name: 'Min Occurrence', type: 'number' },
      { id: 'min_confidence', name: 'Min Confidence', type: 'number' },
      { id: 'entity_types', name: 'Entity Types', type: 'array' },
      { id: 'relationship_types', name: 'Relationship Types', type: 'array' },
      { id: 'entity_type', name: 'Entity Type', type: 'string' },
      { id: 'relationship_type', name: 'Relationship Type', type: 'string' },
      { id: 'limit', name: 'Limit', type: 'number' },
      { id: 'min_weight', name: 'Min Weight', type: 'number' },
    ],
    outputs: [
      { id: 'mode', name: 'Mode', type: 'string' },
      { id: 'filters', name: 'Filters', type: 'object' },
      { id: 'nodes', name: 'Nodes', type: 'array' },
      { id: 'edges', name: 'Edges', type: 'array' },
      { id: 'entities', name: 'Entities', type: 'array' },
      { id: 'relationships', name: 'Relationships', type: 'array' },
      { id: 'communities', name: 'Communities', type: 'array' },
      { id: 'community_count', name: 'Community Count', type: 'number' },
      { id: 'neighborhood', name: 'Neighborhood', type: 'object' },
      { id: 'stats', name: 'Stats', type: 'object' },
      { id: 'result', name: 'Result', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        mode: {
          type: 'string',
          enum: ['filtered_graph', 'entity_neighborhood', 'communities', 'top_entities', 'top_relationships', 'export'],
          default: 'filtered_graph',
        },
        domain: { type: 'string', default: 'general' },
        entityName: { type: 'string', default: 'GraphRAG' },
        limit: { type: 'number', default: 20 },
      },
    },
  },
  {
    type: 'graphrag_enhanced_extraction_test',
    displayName: 'Extraction Diagnostic',
    category: 'rag',
    description: 'Run GraphRAG enhanced extraction diagnostics over sample text',
    color: '#8b5cf6',
    inputs: [
      { id: 'text', name: 'Text', type: 'string', required: true },
      { id: 'domain', name: 'Domain', type: 'string' },
    ],
    outputs: [
      { id: 'success', name: 'Success', type: 'boolean' },
      { id: 'message', name: 'Message', type: 'string' },
      { id: 'entities_found', name: 'Entities Found', type: 'number' },
      { id: 'relationships_found', name: 'Relationships Found', type: 'number' },
      { id: 'extraction_methods_used', name: 'Methods Used', type: 'array' },
      { id: 'sample_entities', name: 'Sample Entities', type: 'array' },
      { id: 'sample_relationships', name: 'Sample Relationships', type: 'array' },
      { id: 'result', name: 'Result', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'graphrag_evaluation_status',
    displayName: 'Evaluation Status',
    category: 'rag',
    description: 'Check GraphRAG retrieval and chunking evaluation readiness',
    color: '#8b5cf6',
    inputs: [],
    outputs: [
      { id: 'retrieval', name: 'Retrieval', type: 'object' },
      { id: 'chunking', name: 'Chunking', type: 'object' },
      { id: 'available', name: 'Available', type: 'boolean' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'graphrag_feedback_submit',
    displayName: 'Feedback Submit',
    category: 'rag',
    description: 'Record retrieval quality feedback for a GraphRAG query',
    color: '#8b5cf6',
    inputs: [
      { id: 'query', name: 'Query', type: 'string', required: true },
      { id: 'rating', name: 'Rating', type: 'number' },
      { id: 'comment', name: 'Comment', type: 'string' },
      { id: 'quality_signals', name: 'Quality Signals', type: 'object' },
      { id: 'source', name: 'Source', type: 'string' },
      { id: 'domain', name: 'Domain', type: 'string' },
    ],
    outputs: [
      { id: 'status', name: 'Status', type: 'string' },
      { id: 'message', name: 'Message', type: 'string' },
      { id: 'query', name: 'Query', type: 'string' },
      { id: 'rating', name: 'Rating', type: 'number' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        query: { type: 'string' },
        rating: { type: 'integer', enum: [-1, 0, 1], default: 0 },
        comment: { type: 'string' },
        source: { type: 'string', default: 'workflow' },
        domain: { type: 'string', default: 'general' },
      },
    },
  },
  {
    type: 'graphrag_feedback_analytics',
    displayName: 'Feedback Analytics',
    category: 'rag',
    description: 'Fetch retrieval feedback trends, domain breakdowns, and problem queries',
    color: '#8b5cf6',
    inputs: [
      { id: 'days', name: 'Days', type: 'number' },
      { id: 'domain', name: 'Domain', type: 'string' },
    ],
    outputs: [
      { id: 'period_days', name: 'Period Days', type: 'number' },
      { id: 'domain_filter', name: 'Domain Filter', type: 'string' },
      { id: 'overall', name: 'Overall', type: 'object' },
      { id: 'daily_trends', name: 'Daily Trends', type: 'array' },
      { id: 'domain_stats', name: 'Domain Stats', type: 'array' },
      { id: 'problem_queries', name: 'Problem Queries', type: 'array' },
      { id: 'total_feedback', name: 'Total Feedback', type: 'number' },
      { id: 'satisfaction_rate', name: 'Satisfaction Rate', type: 'number' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        days: { type: 'integer', default: 30, minimum: 1, maximum: 365 },
        domain: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_social_memory_ingest',
    displayName: 'Social Memory Ingest',
    category: 'rag',
    description: 'Store chat or conversation messages in GraphRAG social memory',
    color: '#8b5cf6',
    inputs: [
      { id: 'platform', name: 'Platform', type: 'string' },
      { id: 'platform_user_id', name: 'Platform User ID', type: 'string', required: true },
      { id: 'display_name', name: 'Display Name', type: 'string' },
      { id: 'text', name: 'Text', type: 'string', required: true },
      { id: 'timestamp', name: 'Timestamp', type: 'string' },
      { id: 'message_id', name: 'Message ID', type: 'string' },
      { id: 'stream_id', name: 'Stream ID', type: 'string' },
      { id: 'session_id', name: 'Session ID', type: 'string' },
      { id: 'previous_messages', name: 'Previous Messages', type: 'array' },
      { id: 'priority', name: 'Priority', type: 'number' },
      { id: 'sync', name: 'Sync', type: 'boolean' },
    ],
    outputs: [
      { id: 'person_id', name: 'Person ID', type: 'string' },
      { id: 'episode_id', name: 'Episode ID', type: 'string' },
      { id: 'facts_written', name: 'Facts Written', type: 'number' },
      { id: 'queued', name: 'Queued', type: 'boolean' },
      { id: 'skipped_duplicate', name: 'Skipped Duplicate', type: 'boolean' },
      { id: 'queue_message_id', name: 'Queue Message ID', type: 'string' },
      { id: 'message', name: 'Message', type: 'object' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        platform: { type: 'string', enum: ['twitch', 'youtube', 'kick', 'rumble'], default: 'twitch' },
        sync: { type: 'boolean', default: false },
        priority: { type: 'integer', default: 0, minimum: 0, maximum: 10 },
      },
    },
  },
  {
    type: 'graphrag_social_memory_context',
    displayName: 'Social Memory Context',
    category: 'rag',
    description: 'Retrieve personalized context or retrieval traces from GraphRAG social memory',
    color: '#8b5cf6',
    inputs: [
      { id: 'platform', name: 'Platform', type: 'string' },
      { id: 'platform_user_id', name: 'Platform User ID', type: 'string', required: true },
      { id: 'query', name: 'Query', type: 'string' },
      { id: 'max_facts', name: 'Max Facts', type: 'number' },
      { id: 'include_recent_episodes', name: 'Recent Episodes', type: 'number' },
      { id: 'as_of', name: 'As Of', type: 'string' },
      { id: 'mode', name: 'Mode', type: 'string' },
    ],
    outputs: [
      { id: 'person_id', name: 'Person ID', type: 'string' },
      { id: 'formatted', name: 'Formatted', type: 'string' },
      { id: 'facts', name: 'Facts', type: 'array' },
      { id: 'recent_episodes', name: 'Recent Episodes', type: 'array' },
      { id: 'retrieval_path', name: 'Retrieval Path', type: 'string' },
      { id: 'cache_hit', name: 'Cache Hit', type: 'boolean' },
      { id: 'fact_ids', name: 'Fact IDs', type: 'array' },
      { id: 'mode', name: 'Mode', type: 'string' },
      { id: 'trace', name: 'Trace', type: 'object' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        platform: { type: 'string', enum: ['twitch', 'youtube', 'kick', 'rumble'], default: 'twitch' },
        mode: { type: 'string', enum: ['context', 'explain'], default: 'context' },
        maxFacts: { type: 'integer', default: 25, minimum: 1, maximum: 100 },
        includeRecentEpisodes: { type: 'integer', default: 3, minimum: 0, maximum: 15 },
      },
    },
  },
  {
    type: 'graphrag_social_memory_facts',
    displayName: 'Social Memory Facts',
    category: 'rag',
    description: 'List known social-memory facts for a resolved person',
    color: '#8b5cf6',
    inputs: [
      { id: 'person_id', name: 'Person ID', type: 'string', required: true },
      { id: 'as_of', name: 'As Of', type: 'string' },
    ],
    outputs: [
      { id: 'person_id', name: 'Person ID', type: 'string' },
      { id: 'facts', name: 'Facts', type: 'array' },
      { id: 'fact_count', name: 'Fact Count', type: 'number' },
      { id: 'as_of', name: 'As Of', type: 'string' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        personId: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_social_memory_admin',
    displayName: 'Social Memory Admin',
    category: 'rag',
    description: 'Run GraphRAG social-memory stewardship operations',
    color: '#8b5cf6',
    inputs: [
      { id: 'mode', name: 'Mode', type: 'string' },
      { id: 'admin_api_key', name: 'Admin API Key', type: 'string' },
      { id: 'payload', name: 'Payload', type: 'object' },
      { id: 'primary_person_id', name: 'Primary Person ID', type: 'string' },
      { id: 'secondary_person_id', name: 'Secondary Person ID', type: 'string' },
      { id: 'confidence', name: 'Confidence', type: 'number' },
      { id: 'reason', name: 'Reason', type: 'string' },
      { id: 'fact_id', name: 'Fact ID', type: 'string' },
      { id: 'new_fact_text', name: 'New Fact Text', type: 'string' },
      { id: 'corrected_by', name: 'Corrected By', type: 'string' },
      { id: 'predicate', name: 'Predicate', type: 'string' },
      { id: 'entity_name', name: 'Entity Name', type: 'string' },
      { id: 'entity_type', name: 'Entity Type', type: 'string' },
      { id: 'transcript_path', name: 'Transcript Path', type: 'string' },
      { id: 'sandbox_tag', name: 'Sandbox Tag', type: 'string' },
      { id: 'sync', name: 'Sync', type: 'boolean' },
      { id: 'suite', name: 'Suite', type: 'string' },
      { id: 'baseline_path', name: 'Baseline Path', type: 'string' },
    ],
    outputs: [
      { id: 'mode', name: 'Mode', type: 'string' },
      { id: 'status_code', name: 'Status Code', type: 'number' },
      { id: 'result', name: 'Result', type: 'object' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        mode: {
          type: 'string',
          enum: [
            'community_report',
            'migrate',
            'merge_identities',
            'correct_fact',
            'replay_transcript',
            'purge_replay',
            'run_eval',
          ],
          default: 'community_report',
        },
        suite: { type: 'string', enum: ['all', 'locomo', 'vtuber'], default: 'all' },
        adminApiKey: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_reddit_crawl',
    displayName: 'Reddit Crawl',
    category: 'rag',
    description: 'Start a Reddit crawl through GraphRAG with optional auto-ingestion',
    color: '#8b5cf6',
    inputs: [
      { id: 'query', name: 'Query', type: 'string', required: true },
      { id: 'subreddits', name: 'Subreddits', type: 'array' },
      { id: 'sort_by', name: 'Sort By', type: 'string' },
      { id: 'time_filter', name: 'Time Filter', type: 'string' },
      { id: 'max_posts', name: 'Max Posts', type: 'number' },
      { id: 'max_comments_per_post', name: 'Max Comments', type: 'number' },
      { id: 'comment_depth', name: 'Comment Depth', type: 'number' },
      { id: 'include_replies', name: 'Include Replies', type: 'boolean' },
      { id: 'auto_ingest', name: 'Auto Ingest', type: 'boolean' },
      { id: 'filter_nsfw', name: 'Filter NSFW', type: 'boolean' },
    ],
    outputs: [
      { id: 'crawl_id', name: 'Crawl ID', type: 'string' },
      { id: 'status', name: 'Status', type: 'string' },
      { id: 'posts_found', name: 'Posts Found', type: 'number' },
      { id: 'comments_found', name: 'Comments Found', type: 'number' },
      { id: 'total_content_length', name: 'Content Length', type: 'number' },
      { id: 'estimated_processing_time', name: 'Estimated Time', type: 'string' },
      { id: 'ingestion_status', name: 'Ingestion Status', type: 'string' },
      { id: 'query', name: 'Query', type: 'string' },
      { id: 'subreddits', name: 'Subreddits', type: 'array' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        query: { type: 'string' },
        subreddits: { type: 'array', items: { type: 'string' } },
        sortBy: { type: 'string', enum: ['relevance', 'hot', 'top', 'new', 'comments'], default: 'relevance' },
        timeFilter: { type: 'string', enum: ['hour', 'day', 'week', 'month', 'year', 'all'], default: 'month' },
        maxPosts: { type: 'integer', default: 50, minimum: 1, maximum: 500 },
        maxCommentsPerPost: { type: 'integer', default: 20, minimum: 0, maximum: 250 },
        commentDepth: { type: 'integer', default: 2, minimum: 1, maximum: 3 },
        includeReplies: { type: 'boolean', default: true },
        autoIngest: { type: 'boolean', default: true },
        filterNsfw: { type: 'boolean', default: true },
      },
    },
  },
  {
    type: 'graphrag_reddit_crawl_status',
    displayName: 'Reddit Crawl Status',
    category: 'rag',
    description: 'Poll progress for a GraphRAG Reddit crawl',
    color: '#8b5cf6',
    inputs: [
      { id: 'crawl_id', name: 'Crawl ID', type: 'string', required: true },
    ],
    outputs: [
      { id: 'crawl_id', name: 'Crawl ID', type: 'string' },
      { id: 'status', name: 'Status', type: 'string' },
      { id: 'progress', name: 'Progress', type: 'number' },
      { id: 'posts_processed', name: 'Posts Processed', type: 'number' },
      { id: 'comments_processed', name: 'Comments Processed', type: 'number' },
      { id: 'current_post', name: 'Current Post', type: 'string' },
      { id: 'error', name: 'Error', type: 'string' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        crawlId: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_reddit_crawl_results',
    displayName: 'Reddit Crawl Results',
    category: 'rag',
    description: 'Fetch completed GraphRAG Reddit crawl results and ingestion status',
    color: '#8b5cf6',
    inputs: [
      { id: 'crawl_id', name: 'Crawl ID', type: 'string', required: true },
    ],
    outputs: [
      { id: 'crawl_id', name: 'Crawl ID', type: 'string' },
      { id: 'status', name: 'Status', type: 'string' },
      { id: 'results', name: 'Results', type: 'array' },
      { id: 'result_count', name: 'Result Count', type: 'number' },
      { id: 'statistics', name: 'Statistics', type: 'object' },
      { id: 'ingestion_status', name: 'Ingestion Status', type: 'string' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        crawlId: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_upstream',
    displayName: 'GraphRAG Upstream',
    category: 'rag',
    description: 'Call an allowlisted GraphRAG endpoint through the upstream bridge',
    color: '#8b5cf6',
    inputs: [
      { id: 'path', name: 'Path', type: 'string' },
      { id: 'method', name: 'Method', type: 'string' },
      { id: 'params', name: 'Params', type: 'object' },
      { id: 'body', name: 'Body', type: 'any' },
    ],
    outputs: [
      { id: 'response', name: 'Response', type: 'any' },
      { id: 'status_code', name: 'Status Code', type: 'number' },
      { id: 'path', name: 'Path', type: 'string' },
      { id: 'method', name: 'Method', type: 'string' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        method: { type: 'string', enum: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'], default: 'GET' },
        path: { type: 'string' },
        timeout: { type: 'integer', default: 120, minimum: 1, maximum: 300 },
      },
    },
  },

  // Tools Nodes
  {
    type: 'code_executor',
    displayName: 'Code Executor',
    category: 'tools',
    description: 'Execute Python or JavaScript code',
    color: '#f59e0b',
    inputs: [
      { id: 'input', name: 'Input', type: 'any' },
    ],
    outputs: [
      { id: 'result', name: 'Result', type: 'any' },
      { id: 'stdout', name: 'Stdout', type: 'string' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        language: { type: 'string', enum: ['python', 'javascript'] },
        code: { type: 'string', format: 'code' },
        timeout: { type: 'integer', default: 30 },
      },
    },
  },
  {
    type: 'function_call',
    displayName: 'Function Call',
    category: 'tools',
    description: 'Call a registered function/tool',
    color: '#f59e0b',
    inputs: [
      { id: 'args', name: 'Arguments', type: 'object' },
    ],
    outputs: [
      { id: 'result', name: 'Result', type: 'any' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        functionName: { type: 'string' },
      },
    },
  },
  {
    type: 'code_detection',
    displayName: 'Code Detection',
    category: 'tools',
    description: 'Detect code in documents using GraphRAG',
    color: '#f59e0b',
    inputs: [
      { id: 'content', name: 'Content', type: 'string', required: true },
      { id: 'filename', name: 'Filename', type: 'string' },
    ],
    outputs: [
      { id: 'is_code', name: 'Is Code', type: 'boolean' },
      { id: 'language', name: 'Language', type: 'string' },
      { id: 'confidence', name: 'Confidence', type: 'number' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    type: 'code_search',
    displayName: 'Code Search',
    category: 'tools',
    description: 'Search for code examples using GraphRAG',
    color: '#f59e0b',
    inputs: [
      { id: 'query', name: 'Query', type: 'string', required: true },
      { id: 'language', name: 'Language', type: 'string' },
    ],
    outputs: [
      { id: 'code_examples', name: 'Code Examples', type: 'array' },
      { id: 'sources', name: 'Sources', type: 'array' },
      { id: 'total_results', name: 'Total Results', type: 'number' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        topK: { type: 'integer', default: 10, minimum: 1, maximum: 50 },
      },
    },
  },
  {
    type: 'graphrag_code_ingest',
    displayName: 'Code-RAG Ingest',
    category: 'tools',
    description: 'Start Code-RAG indexing for one or more code file paths',
    color: '#f59e0b',
    inputs: [
      { id: 'file_paths', name: 'File Paths', type: 'array', required: true },
      { id: 'project_name', name: 'Project Name', type: 'string' },
      { id: 'language', name: 'Language', type: 'string' },
    ],
    outputs: [
      { id: 'task_id', name: 'Task ID', type: 'string' },
      { id: 'status', name: 'Status', type: 'string' },
      { id: 'message', name: 'Message', type: 'string' },
      { id: 'project_name', name: 'Project Name', type: 'string' },
      { id: 'file_count', name: 'File Count', type: 'number' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        projectName: { type: 'string', default: 'default_project' },
        language: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_code_ingestion_status',
    displayName: 'Code-RAG Ingest Status',
    category: 'tools',
    description: 'Check progress for a Code-RAG indexing task',
    color: '#f59e0b',
    inputs: [
      { id: 'task_id', name: 'Task ID', type: 'string', required: true },
    ],
    outputs: [
      { id: 'task_id', name: 'Task ID', type: 'string' },
      { id: 'status', name: 'Status', type: 'string' },
      { id: 'progress', name: 'Progress', type: 'number' },
      { id: 'processed_files', name: 'Processed Files', type: 'number' },
      { id: 'total_files', name: 'Total Files', type: 'number' },
      { id: 'entities_found', name: 'Entities Found', type: 'number' },
      { id: 'error', name: 'Error', type: 'string' },
      { id: 'elapsed_time', name: 'Elapsed Time', type: 'number' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        taskId: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_code_analysis',
    displayName: 'Code-RAG Analysis',
    category: 'tools',
    description: 'Parse a code file and return entities and relationships from Code-RAG',
    color: '#f59e0b',
    inputs: [
      { id: 'file_path', name: 'File Path', type: 'string', required: true },
    ],
    outputs: [
      { id: 'file_path', name: 'File Path', type: 'string' },
      { id: 'language', name: 'Language', type: 'string' },
      { id: 'success', name: 'Success', type: 'boolean' },
      { id: 'entities', name: 'Entities', type: 'array' },
      { id: 'relationships', name: 'Relationships', type: 'array' },
      { id: 'entity_count', name: 'Entity Count', type: 'number' },
      { id: 'relationship_count', name: 'Relationship Count', type: 'number' },
      { id: 'parse_time_ms', name: 'Parse Time', type: 'number' },
      { id: 'error', name: 'Error', type: 'string' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        filePath: { type: 'string' },
      },
    },
  },
  {
    type: 'graphrag_code_stats',
    displayName: 'Code-RAG Statistics',
    category: 'tools',
    description: 'Get Code-RAG search index and ingestion task statistics',
    color: '#f59e0b',
    inputs: [],
    outputs: [
      { id: 'search_engine', name: 'Search Engine', type: 'object' },
      { id: 'active_tasks', name: 'Active Tasks', type: 'number' },
      { id: 'completed_tasks', name: 'Completed Tasks', type: 'number' },
      { id: 'failed_tasks', name: 'Failed Tasks', type: 'number' },
      { id: 'response', name: 'Response', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {},
    },
  },

  // Data Transform Nodes
  {
    type: 'template',
    displayName: 'Template',
    category: 'data',
    description: 'Render template with variables',
    color: '#06b6d4',
    inputs: [
      { id: 'vars', name: 'Variables', type: 'object' },
    ],
    outputs: [
      { id: 'output', name: 'Output', type: 'string' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        template: { type: 'string', format: 'template' },
      },
    },
  },
  {
    type: 'json_parse',
    displayName: 'JSON Parse',
    category: 'data',
    description: 'Parse JSON string to object',
    color: '#06b6d4',
    inputs: [
      { id: 'input', name: 'Input', type: 'string', required: true },
    ],
    outputs: [
      { id: 'output', name: 'Output', type: 'object' },
    ],
    configSchema: {},
  },
  {
    type: 'json_stringify',
    displayName: 'JSON Stringify',
    category: 'data',
    description: 'Convert object to JSON string',
    color: '#06b6d4',
    inputs: [
      { id: 'input', name: 'Input', type: 'object', required: true },
    ],
    outputs: [
      { id: 'output', name: 'Output', type: 'string' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        pretty: { type: 'boolean', default: false },
      },
    },
  },
  {
    type: 'mapper',
    displayName: 'Array Map',
    category: 'data',
    description: 'Map over array items',
    color: '#06b6d4',
    inputs: [
      { id: 'items', name: 'Items', type: 'array', required: true },
    ],
    outputs: [
      { id: 'results', name: 'Results', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        expression: { type: 'string', description: 'JavaScript expression using item' },
      },
    },
  },
  {
    type: 'filter',
    displayName: 'Array Filter',
    category: 'data',
    description: 'Filter array items',
    color: '#06b6d4',
    inputs: [
      { id: 'items', name: 'Items', type: 'array', required: true },
    ],
    outputs: [
      { id: 'results', name: 'Results', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        condition: { type: 'string', description: 'JavaScript condition using item' },
      },
    },
  },

  // Control Flow Nodes
  {
    type: 'condition',
    displayName: 'Condition',
    category: 'control',
    description: 'Branch based on condition',
    color: '#ec4899',
    inputs: [
      { id: 'input', name: 'Input', type: 'any', required: true },
    ],
    outputs: [
      { id: 'true', name: 'True', type: 'any' },
      { id: 'false', name: 'False', type: 'any' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        condition: { type: 'string', description: 'JavaScript condition' },
      },
    },
  },
  {
    type: 'switch',
    displayName: 'Switch',
    category: 'control',
    description: 'Multi-way branch',
    color: '#ec4899',
    inputs: [
      { id: 'input', name: 'Input', type: 'any', required: true },
    ],
    outputs: [
      { id: 'default', name: 'Default', type: 'any' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        cases: { 
          type: 'array', 
          items: { 
            type: 'object',
            properties: {
              value: { type: 'string' },
              label: { type: 'string' },
            }
          }
        },
      },
    },
  },
  {
    type: 'loop',
    displayName: 'Loop',
    category: 'control',
    description: 'Iterate over items',
    color: '#ec4899',
    inputs: [
      { id: 'items', name: 'Items', type: 'array', required: true },
    ],
    outputs: [
      { id: 'item', name: 'Current Item', type: 'any' },
      { id: 'index', name: 'Index', type: 'number' },
      { id: 'done', name: 'Done', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        maxIterations: { type: 'integer', default: 100 },
      },
    },
  },
  {
    type: 'merge',
    displayName: 'Merge',
    category: 'control',
    description: 'Merge multiple inputs',
    color: '#ec4899',
    inputs: [
      { id: 'input1', name: 'Input 1', type: 'any' },
      { id: 'input2', name: 'Input 2', type: 'any' },
      { id: 'input3', name: 'Input 3', type: 'any' },
    ],
    outputs: [
      { id: 'output', name: 'Output', type: 'array' },
    ],
    configSchema: {},
  },
  {
    type: 'delay',
    displayName: 'Delay',
    category: 'control',
    description: 'Wait for specified duration',
    color: '#ec4899',
    inputs: [
      { id: 'input', name: 'Input', type: 'any' },
    ],
    outputs: [
      { id: 'output', name: 'Output', type: 'any' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        ms: { type: 'integer', default: 1000, description: 'Delay in milliseconds' },
      },
    },
  },

  // API Nodes
  {
    type: 'http_request',
    displayName: 'HTTP Request',
    category: 'api',
    description: 'Make HTTP request',
    color: '#14b8a6',
    inputs: [
      { id: 'url', name: 'URL', type: 'string' },
      { id: 'body', name: 'Body', type: 'any' },
      { id: 'headers', name: 'Headers', type: 'object' },
    ],
    outputs: [
      { id: 'response', name: 'Response', type: 'any' },
      { id: 'status', name: 'Status', type: 'number' },
      { id: 'headers', name: 'Headers', type: 'object' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        method: { type: 'string', enum: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'], default: 'GET' },
        timeout: { type: 'integer', default: 30 },
        baseUrl: { type: 'string' },
      },
    },
  },
  {
    type: 'graphql_query',
    displayName: 'GraphQL Query',
    category: 'api',
    description: 'Execute GraphQL query',
    color: '#14b8a6',
    inputs: [
      { id: 'variables', name: 'Variables', type: 'object' },
    ],
    outputs: [
      { id: 'data', name: 'Data', type: 'object' },
      { id: 'errors', name: 'Errors', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        endpoint: { type: 'string' },
        query: { type: 'string', format: 'graphql' },
      },
    },
  },

  // MCP Nodes
  {
    type: 'mcp_tool',
    displayName: 'MCP Tool',
    category: 'mcp',
    description: 'Call MCP server tool',
    color: '#a855f7',
    inputs: [
      { id: 'args', name: 'Arguments', type: 'object' },
    ],
    outputs: [
      { id: 'result', name: 'Result', type: 'any' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        server: { type: 'string', description: 'MCP server name' },
        tool: { type: 'string', description: 'Tool name' },
      },
    },
  },
  {
    type: 'mcp_resource',
    displayName: 'MCP Resource',
    category: 'mcp',
    description: 'Access MCP resource',
    color: '#a855f7',
    inputs: [],
    outputs: [
      { id: 'content', name: 'Content', type: 'any' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        server: { type: 'string' },
        resource: { type: 'string' },
      },
    },
  },
  {
    type: 'mcp_prompt',
    displayName: 'MCP Prompt',
    category: 'mcp',
    description: 'Use a prompt template from MCP server',
    color: '#a855f7',
    inputs: [
      { id: 'args', name: 'Arguments', type: 'object' },
    ],
    outputs: [
      { id: 'messages', name: 'Messages', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        server: { type: 'string', description: 'MCP server name' },
        prompt: { type: 'string', description: 'Prompt name' },
      },
    },
  },

  // Database Nodes
  {
    type: 'sql_query',
    displayName: 'SQL Query',
    category: 'database',
    description: 'Execute SQL query',
    color: '#f97316',
    inputs: [
      { id: 'params', name: 'Parameters', type: 'object' },
    ],
    outputs: [
      { id: 'rows', name: 'Rows', type: 'array' },
      { id: 'count', name: 'Row Count', type: 'number' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        connection: { type: 'string', description: 'Connection name' },
        query: { type: 'string', format: 'sql' },
      },
    },
  },
  {
    type: 'cache_get',
    displayName: 'Cache Get',
    category: 'database',
    description: 'Get value from cache',
    color: '#f97316',
    inputs: [
      { id: 'key', name: 'Key', type: 'string', required: true },
    ],
    outputs: [
      { id: 'value', name: 'Value', type: 'any' },
      { id: 'hit', name: 'Cache Hit', type: 'boolean' },
    ],
    configSchema: {},
  },
  {
    type: 'cache_set',
    displayName: 'Cache Set',
    category: 'database',
    description: 'Set value in cache',
    color: '#f97316',
    inputs: [
      { id: 'key', name: 'Key', type: 'string', required: true },
      { id: 'value', name: 'Value', type: 'any', required: true },
    ],
    outputs: [
      { id: 'success', name: 'Success', type: 'boolean' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        ttl: { type: 'integer', description: 'TTL in seconds' },
      },
    },
  },
  {
    type: 'cache_delete',
    displayName: 'Cache Delete',
    category: 'database',
    description: 'Delete value from cache',
    color: '#f97316',
    inputs: [
      { id: 'key', name: 'Key', type: 'string', required: true },
    ],
    outputs: [
      { id: 'deleted', name: 'Deleted', type: 'boolean' },
    ],
    configSchema: {},
  },
  {
    type: 'qdrant_search',
    displayName: 'Qdrant Search',
    category: 'database',
    description: 'Search Qdrant vector database',
    color: '#f97316',
    inputs: [
      { id: 'vector', name: 'Vector', type: 'array', required: true },
    ],
    outputs: [
      { id: 'points', name: 'Points', type: 'array' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        collection: { type: 'string', description: 'Collection name' },
        limit: { type: 'integer', default: 5 },
        scoreThreshold: { type: 'number', description: 'Minimum score' },
      },
    },
  },

  // Tools - Additional
  {
    type: 'shell_command',
    displayName: 'Shell Command',
    category: 'tools',
    description: 'Execute a shell command (restricted)',
    color: '#f59e0b',
    inputs: [],
    outputs: [
      { id: 'stdout', name: 'Stdout', type: 'string' },
      { id: 'stderr', name: 'Stderr', type: 'string' },
      { id: 'code', name: 'Exit Code', type: 'number' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'Shell command to run' },
        timeout: { type: 'integer', default: 30, description: 'Timeout in seconds' },
      },
    },
  },

  // Output Nodes
  {
    type: 'output',
    displayName: 'Output',
    category: 'output',
    description: 'Workflow output',
    color: '#ef4444',
    inputs: [
      { id: 'value', name: 'Value', type: 'any', required: true },
    ],
    outputs: [],
    configSchema: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'Output name' },
      },
    },
  },
  {
    type: 'webhook_response',
    displayName: 'Webhook Response',
    category: 'output',
    description: 'HTTP response for webhook trigger',
    color: '#ef4444',
    inputs: [
      { id: 'body', name: 'Body', type: 'any', required: true },
      { id: 'headers', name: 'Headers', type: 'object' },
    ],
    outputs: [],
    configSchema: {
      type: 'object',
      properties: {
        statusCode: { type: 'integer', default: 200 },
        contentType: { type: 'string', default: 'application/json' },
      },
    },
  },
  {
    type: 'log',
    displayName: 'Log',
    category: 'output',
    description: 'Log message',
    color: '#ef4444',
    inputs: [
      { id: 'message', name: 'Message', type: 'any', required: true },
    ],
    outputs: [
      { id: 'passthrough', name: 'Passthrough', type: 'any' },
    ],
    configSchema: {
      type: 'object',
      properties: {
        level: { type: 'string', enum: ['debug', 'info', 'warn', 'error'], default: 'info' },
      },
    },
  },
];

// Helper to get node type by type string
export function getNodeTypeDefinition(type: string): NodeTypeDefinition | undefined {
  return BUILTIN_NODE_TYPES.find((n) => n.type === type);
}

// Helper to get nodes by category
export function getNodeTypesByCategory(category: NodeCategory): NodeTypeDefinition[] {
  return BUILTIN_NODE_TYPES.filter((n) => n.category === category);
}
