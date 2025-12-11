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
        model: { type: 'string', description: 'Model to use' },
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
        model: { type: 'string', description: 'Embedding model' },
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
