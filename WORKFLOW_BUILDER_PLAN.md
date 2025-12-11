# Workflow Builder Overhaul Plan

## Executive Summary

This document outlines a complete overhaul of the workflow builder system in LlamaNexus. The goal is to create a visual, drag-and-drop workflow creation experience that allows users to connect various services, models, tools, document loaders, database connections, external APIs, OpenAI-compatible endpoints, and MCP servers into composable pipelines.

---

## Current State Analysis

### Existing Implementation (`WorkflowBuilderPage.tsx`)

The current implementation is a basic prototype with significant limitations:

- **Node Types**: Only 6 basic types (input, llm, code, condition, output, loop)
- **No True Canvas**: Nodes are rendered in a flex-wrap layout, not positioned on a canvas
- **No Visual Connections**: Connections are stored as IDs but not rendered as lines/curves
- **No Backend**: Workflow execution is simulated with `setTimeout`
- **No Persistence**: Workflows only saved to localStorage
- **No Integration**: Not connected to existing services (RAG, models, tools, etc.)

### What Needs to Change

1. Replace with a proper graph/flow editor library
2. Build backend workflow engine
3. Create comprehensive node type system
4. Implement visual connection rendering
5. Add real-time execution monitoring
6. Integrate with all existing LlamaNexus services

---

## Architecture Overview

```
+------------------+     +------------------+     +------------------+
|    Frontend      |     |    Backend       |     |    Execution     |
|  (React Flow)    |<--->|   (FastAPI)      |<--->|    Engine        |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
|  Node Registry   |     | Workflow Store   |     | Service Layer    |
|  (UI Components) |     | (SQLite/JSON)    |     | (Integrations)   |
+------------------+     +------------------+     +------------------+
```

---

## Phase 1: Foundation (Week 1-2)

### 1.1 Frontend Canvas Library

**Recommended: React Flow** (https://reactflow.dev/)

Rationale:
- MIT licensed, production-ready
- Built-in drag-and-drop, zooming, panning
- Customizable node and edge components
- Good TypeScript support
- Active community and maintenance

Alternative considered: Rete.js, JointJS

**Implementation Tasks:**

1. Install React Flow dependencies:
   ```bash
   npm install reactflow
   ```

2. Create base canvas component with:
   - Grid background
   - Mini-map
   - Controls (zoom, fit view, lock)
   - Keyboard shortcuts

3. Define base node component structure:
   ```typescript
   interface WorkflowNode {
     id: string;
     type: string;
     position: { x: number; y: number };
     data: {
       label: string;
       config: Record<string, any>;
       inputs: PortDefinition[];
       outputs: PortDefinition[];
       status?: 'idle' | 'running' | 'success' | 'error';
     };
   }
   
   interface PortDefinition {
     id: string;
     name: string;
     type: 'string' | 'number' | 'boolean' | 'object' | 'array' | 'any';
     required?: boolean;
   }
   ```

### 1.2 Node Categories and Types

#### Category: Inputs/Triggers
| Node Type | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `manual_trigger` | Manually triggered workflow | - | `trigger: any` |
| `http_webhook` | HTTP endpoint trigger | - | `body: object`, `headers: object` |
| `schedule` | Cron-based scheduler | - | `timestamp: string` |
| `file_watch` | Watch directory for changes | - | `filepath: string`, `event: string` |
| `event_listener` | Listen to internal events | - | `event: object` |

#### Category: LLM/Models
| Node Type | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `llm_chat` | Chat completion with local model | `messages: array`, `model?: string` | `response: string`, `usage: object` |
| `llm_completion` | Text completion | `prompt: string` | `completion: string`, `usage: object` |
| `openai_chat` | OpenAI API chat | `messages: array`, `api_key: string` | `response: string`, `usage: object` |
| `anthropic_chat` | Claude API | `messages: array`, `api_key: string` | `response: string` |
| `embedding` | Generate embeddings | `text: string`, `model?: string` | `embedding: array` |
| `model_router` | Route to best model based on criteria | `prompt: string`, `criteria: object` | `model: string`, `response: string` |

#### Category: RAG/Documents
| Node Type | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `document_loader` | Load documents from sources | `source: string`, `type: string` | `documents: array` |
| `chunker` | Split documents into chunks | `documents: array`, `strategy: string` | `chunks: array` |
| `vector_store` | Store/retrieve from vector DB | `operation: string`, `data: any` | `results: array` |
| `retriever` | Semantic search | `query: string`, `k: number` | `documents: array`, `scores: array` |
| `reranker` | Rerank retrieved results | `query: string`, `documents: array` | `documents: array` |
| `graph_query` | Query knowledge graph | `query: string` | `nodes: array`, `edges: array` |

#### Category: Tools/Functions
| Node Type | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `function_call` | Execute tool/function | `name: string`, `args: object` | `result: any` |
| `tool_router` | Route to appropriate tool | `query: string`, `tools: array` | `tool: string`, `args: object` |
| `code_executor` | Run Python/JS code | `code: string`, `language: string` | `result: any`, `stdout: string` |
| `shell_command` | Execute shell command | `command: string` | `stdout: string`, `stderr: string`, `code: number` |

#### Category: Data/Transform
| Node Type | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `json_parse` | Parse JSON string | `input: string` | `output: object` |
| `json_stringify` | Convert to JSON string | `input: object` | `output: string` |
| `template` | Render Jinja/Mustache template | `template: string`, `vars: object` | `output: string` |
| `extract_json` | Extract JSON from text | `text: string` | `json: object` |
| `regex_extract` | Extract with regex | `text: string`, `pattern: string` | `matches: array` |
| `mapper` | Map over array | `items: array`, `expression: string` | `results: array` |
| `filter` | Filter array | `items: array`, `condition: string` | `results: array` |
| `aggregator` | Aggregate multiple inputs | `inputs: array` | `output: any` |

#### Category: Control Flow
| Node Type | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `condition` | Branch based on condition | `input: any`, `condition: string` | `true: any`, `false: any` |
| `switch` | Multi-way branch | `input: any`, `cases: object` | `[case_name]: any` |
| `loop` | Iterate over items | `items: array` | `item: any`, `index: number` |
| `while_loop` | Loop while condition true | `condition: string` | `iteration: number` |
| `parallel` | Execute branches in parallel | `inputs: array` | `outputs: array` |
| `merge` | Merge parallel branches | `inputs: array` | `output: array` |
| `delay` | Wait for duration | `ms: number` | `elapsed: number` |
| `retry` | Retry on failure | `input: any`, `max_retries: number` | `output: any` |

#### Category: External APIs
| Node Type | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `http_request` | Make HTTP request | `url: string`, `method: string`, `body?: any` | `response: any`, `status: number` |
| `graphql_query` | GraphQL request | `endpoint: string`, `query: string` | `data: object` |
| `websocket` | WebSocket connection | `url: string`, `message?: string` | `response: any` |
| `openapi_call` | Call OpenAPI endpoint | `spec_url: string`, `operation: string` | `response: any` |

#### Category: MCP (Model Context Protocol)
| Node Type | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `mcp_server` | Connect to MCP server | `server_url: string` | `connection: object` |
| `mcp_tool` | Call MCP tool | `server: string`, `tool: string`, `args: object` | `result: any` |
| `mcp_resource` | Access MCP resource | `server: string`, `resource: string` | `content: any` |
| `mcp_prompt` | Use MCP prompt | `server: string`, `prompt: string`, `args: object` | `messages: array` |

#### Category: Databases
| Node Type | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `sql_query` | Execute SQL query | `connection: string`, `query: string` | `rows: array` |
| `nosql_query` | MongoDB/Redis query | `connection: string`, `query: object` | `results: array` |
| `qdrant_search` | Qdrant vector search | `collection: string`, `query: array` | `points: array` |
| `cache_get` | Get from cache | `key: string` | `value: any`, `hit: boolean` |
| `cache_set` | Set in cache | `key: string`, `value: any`, `ttl?: number` | `success: boolean` |

#### Category: Outputs
| Node Type | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `output` | Workflow output | `value: any` | - |
| `webhook_response` | HTTP response | `body: any`, `status: number` | - |
| `file_write` | Write to file | `path: string`, `content: string` | `written: boolean` |
| `notification` | Send notification | `channel: string`, `message: string` | `sent: boolean` |
| `log` | Log to console/storage | `level: string`, `message: string` | - |

---

## Phase 2: Backend Workflow Engine (Week 2-3)

### 2.1 Data Models

```python
# backend/modules/workflow/models.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class NodeType(str, Enum):
    # All node types from above
    MANUAL_TRIGGER = "manual_trigger"
    LLM_CHAT = "llm_chat"
    # ... etc

class PortType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    ANY = "any"

class Port(BaseModel):
    id: str
    name: str
    type: PortType
    required: bool = False

class Connection(BaseModel):
    id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str

class WorkflowNode(BaseModel):
    id: str
    type: NodeType
    position: Dict[str, float]
    config: Dict[str, Any]
    
class Workflow(BaseModel):
    id: str
    name: str
    description: Optional[str]
    nodes: List[WorkflowNode]
    connections: List[Connection]
    variables: Dict[str, Any] = {}
    settings: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    version: int = 1
    is_active: bool = True

class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class NodeExecution(BaseModel):
    node_id: str
    status: ExecutionStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    error: Optional[str]
    logs: List[str]

class WorkflowExecution(BaseModel):
    id: str
    workflow_id: str
    workflow_version: int
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime]
    trigger_data: Dict[str, Any]
    node_executions: Dict[str, NodeExecution]
    final_output: Optional[Any]
    error: Optional[str]
```

### 2.2 Workflow Engine Architecture

```python
# backend/modules/workflow/engine.py

class WorkflowEngine:
    """
    Executes workflows by traversing the node graph and running each node.
    """
    
    def __init__(self):
        self.node_executors: Dict[str, NodeExecutor] = {}
        self.running_executions: Dict[str, WorkflowExecution] = {}
        self.register_builtin_executors()
    
    async def execute_workflow(
        self,
        workflow: Workflow,
        trigger_data: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> WorkflowExecution:
        """Execute a workflow from trigger to completion."""
        pass
    
    async def execute_node(
        self,
        node: WorkflowNode,
        inputs: Dict[str, Any],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute a single node and return outputs."""
        pass
    
    def build_execution_graph(self, workflow: Workflow) -> ExecutionGraph:
        """Build topologically sorted execution order."""
        pass
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        pass
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause execution at next node boundary."""
        pass
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution."""
        pass
```

### 2.3 Node Executor Interface

```python
# backend/modules/workflow/executors/base.py

from abc import ABC, abstractmethod

class NodeExecutor(ABC):
    """Base class for all node executors."""
    
    node_type: str
    display_name: str
    category: str
    description: str
    
    input_ports: List[PortDefinition]
    output_ports: List[PortDefinition]
    config_schema: Dict[str, Any]  # JSON Schema for config
    
    @abstractmethod
    async def execute(
        self,
        config: Dict[str, Any],
        inputs: Dict[str, Any],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute the node and return outputs."""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate node configuration, return list of errors."""
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate inputs against port definitions."""
        pass
```

### 2.4 API Endpoints

```python
# backend/main.py - New workflow endpoints

# Workflow CRUD
@app.get("/api/v1/workflows")
async def list_workflows(limit: int = 50, offset: int = 0):
    """List all workflows."""

@app.post("/api/v1/workflows")
async def create_workflow(workflow: WorkflowCreate):
    """Create a new workflow."""

@app.get("/api/v1/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow by ID."""

@app.put("/api/v1/workflows/{workflow_id}")
async def update_workflow(workflow_id: str, workflow: WorkflowUpdate):
    """Update a workflow."""

@app.delete("/api/v1/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow."""

# Workflow Execution
@app.post("/api/v1/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, trigger_data: Dict[str, Any] = {}):
    """Execute a workflow."""

@app.get("/api/v1/workflows/{workflow_id}/executions")
async def list_executions(workflow_id: str):
    """List workflow executions."""

@app.get("/api/v1/workflows/executions/{execution_id}")
async def get_execution(execution_id: str):
    """Get execution details."""

@app.post("/api/v1/workflows/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """Cancel running execution."""

@app.post("/api/v1/workflows/executions/{execution_id}/pause")
async def pause_execution(execution_id: str):
    """Pause execution."""

@app.post("/api/v1/workflows/executions/{execution_id}/resume")
async def resume_execution(execution_id: str):
    """Resume paused execution."""

# Node Registry
@app.get("/api/v1/workflows/nodes")
async def list_node_types():
    """List available node types with schemas."""

@app.get("/api/v1/workflows/nodes/{node_type}")
async def get_node_type(node_type: str):
    """Get node type details and schema."""

# Workflow Validation
@app.post("/api/v1/workflows/validate")
async def validate_workflow(workflow: WorkflowCreate):
    """Validate workflow structure and connections."""

# Workflow Templates
@app.get("/api/v1/workflows/templates")
async def list_templates():
    """List workflow templates."""

@app.post("/api/v1/workflows/templates/{template_id}/instantiate")
async def instantiate_template(template_id: str, params: Dict[str, Any] = {}):
    """Create workflow from template."""
```

---

## Phase 3: Frontend Implementation (Week 3-4)

### 3.1 Component Structure

```
frontend/src/
  components/
    workflow/
      canvas/
        WorkflowCanvas.tsx        # Main React Flow canvas
        CanvasControls.tsx        # Zoom, fit, minimap controls
        MiniMap.tsx               # Canvas minimap
      nodes/
        BaseNode.tsx              # Base node component
        NodeHandle.tsx            # Input/output ports
        NodeHeader.tsx            # Node title and status
        NodeConfig.tsx            # Configuration panel
        categories/
          InputNodes.tsx          # Trigger node variants
          LLMNodes.tsx            # LLM/model nodes
          RAGNodes.tsx            # Document/retrieval nodes
          ToolNodes.tsx           # Function/code nodes
          DataNodes.tsx           # Transform nodes
          ControlNodes.tsx        # Flow control nodes
          APINodes.tsx            # External API nodes
          MCPNodes.tsx            # MCP server nodes
          DatabaseNodes.tsx       # Database nodes
          OutputNodes.tsx         # Output nodes
      edges/
        BaseEdge.tsx              # Basic connection line
        AnimatedEdge.tsx          # Edge with data flow animation
        ConditionalEdge.tsx       # Edge with condition indicator
      panels/
        NodePalette.tsx           # Draggable node list
        PropertyPanel.tsx         # Selected node properties
        ExecutionPanel.tsx        # Run history and status
        VariablesPanel.tsx        # Workflow variables
      dialogs/
        SaveWorkflowDialog.tsx
        LoadWorkflowDialog.tsx
        WorkflowSettingsDialog.tsx
        NodeConfigDialog.tsx
      hooks/
        useWorkflow.ts            # Workflow state management
        useExecution.ts           # Execution state and polling
        useNodeRegistry.ts        # Node type registry
        useWorkflowStorage.ts     # Persistence
  pages/
    WorkflowBuilderPage.tsx       # Main page
    WorkflowListPage.tsx          # Workflow management
  services/
    workflowApi.ts                # Workflow API client
  types/
    workflow.ts                   # TypeScript types
```

### 3.2 React Flow Integration

```typescript
// components/workflow/canvas/WorkflowCanvas.tsx

import ReactFlow, {
  Node,
  Edge,
  Controls,
  MiniMap,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  NodeTypes,
  EdgeTypes,
} from 'reactflow';
import 'reactflow/dist/style.css';

const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({
  workflow,
  onWorkflowChange,
  execution,
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState(workflow.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(workflow.connections);
  
  const nodeTypes: NodeTypes = useMemo(() => ({
    llm_chat: LLMChatNode,
    retriever: RetrieverNode,
    condition: ConditionNode,
    // ... all node types
  }), []);
  
  const edgeTypes: EdgeTypes = useMemo(() => ({
    default: BaseEdge,
    animated: AnimatedEdge,
    conditional: ConditionalEdge,
  }), []);
  
  const onConnect = useCallback((params: Connection) => {
    // Validate connection compatibility
    if (validateConnection(params, nodes)) {
      setEdges((eds) => addEdge(params, eds));
    }
  }, [nodes]);
  
  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    const nodeType = event.dataTransfer.getData('application/reactflow');
    // Create new node at drop position
  }, []);
  
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onDrop={onDrop}
      onDragOver={(e) => e.preventDefault()}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      fitView
    >
      <Controls />
      <MiniMap />
      <Background variant="dots" gap={16} size={1} />
    </ReactFlow>
  );
};
```

### 3.3 Node Palette with Drag-and-Drop

```typescript
// components/workflow/panels/NodePalette.tsx

const NodePalette: React.FC = () => {
  const { nodeTypes, categories } = useNodeRegistry();
  
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };
  
  return (
    <Box sx={{ width: 280, overflow: 'auto' }}>
      {categories.map((category) => (
        <Accordion key={category.id}>
          <AccordionSummary>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {category.icon}
              <Typography>{category.name}</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={1}>
              {nodeTypes
                .filter((n) => n.category === category.id)
                .map((nodeType) => (
                  <Paper
                    key={nodeType.type}
                    draggable
                    onDragStart={(e) => onDragStart(e, nodeType.type)}
                    sx={{
                      p: 1,
                      cursor: 'grab',
                      '&:hover': { bgcolor: 'action.hover' },
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {nodeType.icon}
                      <Box>
                        <Typography variant="body2">
                          {nodeType.displayName}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {nodeType.description}
                        </Typography>
                      </Box>
                    </Box>
                  </Paper>
                ))}
            </Stack>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
};
```

### 3.4 Custom Node Component

```typescript
// components/workflow/nodes/BaseNode.tsx

interface BaseNodeProps {
  id: string;
  data: {
    label: string;
    config: Record<string, any>;
    inputs: PortDefinition[];
    outputs: PortDefinition[];
    status?: ExecutionStatus;
    error?: string;
  };
  selected: boolean;
}

const BaseNode: React.FC<BaseNodeProps> = ({ id, data, selected }) => {
  const { status, error } = data;
  
  const getStatusColor = () => {
    switch (status) {
      case 'running': return 'info.main';
      case 'success': return 'success.main';
      case 'error': return 'error.main';
      default: return 'grey.500';
    }
  };
  
  return (
    <Paper
      elevation={selected ? 8 : 2}
      sx={{
        minWidth: 200,
        border: selected ? '2px solid' : '1px solid',
        borderColor: selected ? 'primary.main' : 'divider',
        borderRadius: 2,
      }}
    >
      {/* Header */}
      <Box
        sx={{
          px: 1.5,
          py: 1,
          bgcolor: 'background.default',
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          gap: 1,
        }}
      >
        {status && (
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              bgcolor: getStatusColor(),
              animation: status === 'running' ? 'pulse 1s infinite' : 'none',
            }}
          />
        )}
        <Typography variant="subtitle2" fontWeight={600}>
          {data.label}
        </Typography>
      </Box>
      
      {/* Input Handles */}
      <Box sx={{ position: 'relative', minHeight: 40, py: 1 }}>
        {data.inputs.map((input, index) => (
          <Box
            key={input.id}
            sx={{
              display: 'flex',
              alignItems: 'center',
              pl: 1,
              py: 0.25,
            }}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={input.id}
              style={{ top: 16 + index * 24 }}
            />
            <Typography variant="caption" sx={{ ml: 1 }}>
              {input.name}
              {input.required && <span style={{ color: 'red' }}>*</span>}
            </Typography>
          </Box>
        ))}
      </Box>
      
      {/* Output Handles */}
      <Box sx={{ position: 'relative', minHeight: 40, py: 1 }}>
        {data.outputs.map((output, index) => (
          <Box
            key={output.id}
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              pr: 1,
              py: 0.25,
            }}
          >
            <Typography variant="caption" sx={{ mr: 1 }}>
              {output.name}
            </Typography>
            <Handle
              type="source"
              position={Position.Right}
              id={output.id}
              style={{ top: 16 + index * 24 }}
            />
          </Box>
        ))}
      </Box>
      
      {/* Error Display */}
      {error && (
        <Box sx={{ px: 1.5, py: 1, bgcolor: 'error.dark' }}>
          <Typography variant="caption" color="error.contrastText">
            {error}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};
```

---

## Phase 4: Service Integrations (Week 4-5)

### 4.1 LLM Integration

```python
# backend/modules/workflow/executors/llm_executors.py

class LLMChatExecutor(NodeExecutor):
    node_type = "llm_chat"
    display_name = "LLM Chat"
    category = "llm"
    
    input_ports = [
        PortDefinition(id="messages", name="Messages", type="array", required=True),
        PortDefinition(id="system", name="System Prompt", type="string"),
    ]
    
    output_ports = [
        PortDefinition(id="response", name="Response", type="string"),
        PortDefinition(id="usage", name="Token Usage", type="object"),
    ]
    
    config_schema = {
        "type": "object",
        "properties": {
            "model": {"type": "string", "description": "Model to use"},
            "temperature": {"type": "number", "minimum": 0, "maximum": 2},
            "max_tokens": {"type": "integer", "minimum": 1},
            "tools": {"type": "array", "items": {"type": "object"}},
        }
    }
    
    async def execute(self, config, inputs, context):
        messages = inputs.get("messages", [])
        system_prompt = inputs.get("system")
        
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Use internal LLM service
        response = await context.llm_service.chat_completion(
            messages=messages,
            model=config.get("model"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens"),
            tools=config.get("tools"),
        )
        
        return {
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
```

### 4.2 RAG Integration

```python
# backend/modules/workflow/executors/rag_executors.py

class RetrieverExecutor(NodeExecutor):
    node_type = "retriever"
    display_name = "Semantic Search"
    category = "rag"
    
    input_ports = [
        PortDefinition(id="query", name="Query", type="string", required=True),
    ]
    
    output_ports = [
        PortDefinition(id="documents", name="Documents", type="array"),
        PortDefinition(id="scores", name="Scores", type="array"),
    ]
    
    config_schema = {
        "type": "object",
        "properties": {
            "collection": {"type": "string", "description": "Vector collection name"},
            "k": {"type": "integer", "default": 5, "description": "Number of results"},
            "threshold": {"type": "number", "description": "Minimum similarity score"},
            "filter": {"type": "object", "description": "Metadata filter"},
        }
    }
    
    async def execute(self, config, inputs, context):
        query = inputs["query"]
        
        # Use RAG retriever service
        results = await context.rag_service.retrieve(
            query=query,
            collection=config.get("collection", "default"),
            k=config.get("k", 5),
            threshold=config.get("threshold"),
            filter=config.get("filter"),
        )
        
        return {
            "documents": [r.content for r in results],
            "scores": [r.score for r in results],
        }


class DocumentLoaderExecutor(NodeExecutor):
    node_type = "document_loader"
    display_name = "Document Loader"
    category = "rag"
    
    input_ports = [
        PortDefinition(id="source", name="Source", type="string", required=True),
    ]
    
    output_ports = [
        PortDefinition(id="documents", name="Documents", type="array"),
    ]
    
    config_schema = {
        "type": "object",
        "properties": {
            "source_type": {
                "type": "string",
                "enum": ["file", "url", "directory", "database"],
            },
            "file_types": {
                "type": "array",
                "items": {"type": "string"},
            },
            "recursive": {"type": "boolean", "default": True},
        }
    }
    
    async def execute(self, config, inputs, context):
        source = inputs["source"]
        source_type = config.get("source_type", "file")
        
        documents = await context.document_service.load(
            source=source,
            source_type=source_type,
            file_types=config.get("file_types"),
            recursive=config.get("recursive", True),
        )
        
        return {"documents": documents}
```

### 4.3 MCP Integration

```python
# backend/modules/workflow/executors/mcp_executors.py

class MCPServerExecutor(NodeExecutor):
    node_type = "mcp_server"
    display_name = "MCP Server"
    category = "mcp"
    
    output_ports = [
        PortDefinition(id="connection", name="Connection", type="object"),
    ]
    
    config_schema = {
        "type": "object",
        "properties": {
            "server_url": {"type": "string", "format": "uri"},
            "server_name": {"type": "string"},
            "transport": {"type": "string", "enum": ["stdio", "sse", "websocket"]},
        },
        "required": ["server_url"]
    }
    
    async def execute(self, config, inputs, context):
        connection = await context.mcp_service.connect(
            url=config["server_url"],
            name=config.get("server_name"),
            transport=config.get("transport", "stdio"),
        )
        return {"connection": connection}


class MCPToolExecutor(NodeExecutor):
    node_type = "mcp_tool"
    display_name = "MCP Tool"
    category = "mcp"
    
    input_ports = [
        PortDefinition(id="connection", name="Connection", type="object"),
        PortDefinition(id="args", name="Arguments", type="object"),
    ]
    
    output_ports = [
        PortDefinition(id="result", name="Result", type="any"),
    ]
    
    config_schema = {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string"},
            "timeout": {"type": "integer", "default": 30000},
        },
        "required": ["tool_name"]
    }
    
    async def execute(self, config, inputs, context):
        connection = inputs.get("connection")
        args = inputs.get("args", {})
        
        result = await context.mcp_service.call_tool(
            connection=connection,
            tool_name=config["tool_name"],
            arguments=args,
            timeout=config.get("timeout", 30000),
        )
        
        return {"result": result}
```

### 4.4 External API Integration

```python
# backend/modules/workflow/executors/api_executors.py

class HTTPRequestExecutor(NodeExecutor):
    node_type = "http_request"
    display_name = "HTTP Request"
    category = "api"
    
    input_ports = [
        PortDefinition(id="url", name="URL", type="string", required=True),
        PortDefinition(id="body", name="Body", type="any"),
        PortDefinition(id="headers", name="Headers", type="object"),
    ]
    
    output_ports = [
        PortDefinition(id="response", name="Response", type="any"),
        PortDefinition(id="status", name="Status Code", type="number"),
        PortDefinition(id="headers", name="Response Headers", type="object"),
    ]
    
    config_schema = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "default": "GET"
            },
            "timeout": {"type": "integer", "default": 30},
            "follow_redirects": {"type": "boolean", "default": True},
            "auth": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["none", "basic", "bearer", "api_key"]},
                    "credentials": {"type": "object"}
                }
            }
        }
    }
    
    async def execute(self, config, inputs, context):
        import httpx
        
        url = inputs["url"]
        method = config.get("method", "GET")
        body = inputs.get("body")
        headers = inputs.get("headers", {})
        
        # Handle authentication
        auth = config.get("auth", {})
        if auth.get("type") == "bearer":
            headers["Authorization"] = f"Bearer {auth['credentials']['token']}"
        elif auth.get("type") == "api_key":
            headers[auth['credentials']['header']] = auth['credentials']['key']
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                json=body if body else None,
                headers=headers,
                timeout=config.get("timeout", 30),
                follow_redirects=config.get("follow_redirects", True),
            )
        
        try:
            response_data = response.json()
        except:
            response_data = response.text
        
        return {
            "response": response_data,
            "status": response.status_code,
            "headers": dict(response.headers),
        }


class OpenAPICallExecutor(NodeExecutor):
    node_type = "openapi_call"
    display_name = "OpenAPI Call"
    category = "api"
    
    input_ports = [
        PortDefinition(id="params", name="Parameters", type="object"),
        PortDefinition(id="body", name="Request Body", type="any"),
    ]
    
    output_ports = [
        PortDefinition(id="response", name="Response", type="any"),
        PortDefinition(id="status", name="Status Code", type="number"),
    ]
    
    config_schema = {
        "type": "object",
        "properties": {
            "spec_url": {"type": "string", "format": "uri"},
            "operation_id": {"type": "string"},
            "server": {"type": "string"},
            "auth": {"type": "object"},
        },
        "required": ["spec_url", "operation_id"]
    }
    
    async def execute(self, config, inputs, context):
        # Load and parse OpenAPI spec
        spec = await context.openapi_service.load_spec(config["spec_url"])
        operation = spec.get_operation(config["operation_id"])
        
        # Build and execute request
        response = await context.openapi_service.execute_operation(
            spec=spec,
            operation=operation,
            params=inputs.get("params", {}),
            body=inputs.get("body"),
            server=config.get("server"),
            auth=config.get("auth"),
        )
        
        return {
            "response": response.data,
            "status": response.status_code,
        }
```

---

## Phase 5: Execution and Monitoring (Week 5-6)

### 5.1 Real-Time Execution Updates

```typescript
// frontend/hooks/useExecution.ts

import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';

export const useExecution = (executionId: string | null) => {
  const [execution, setExecution] = useState<WorkflowExecution | null>(null);
  const [nodeStatuses, setNodeStatuses] = useState<Record<string, NodeExecution>>({});
  
  const { subscribe, unsubscribe } = useWebSocket();
  
  useEffect(() => {
    if (!executionId) return;
    
    // Subscribe to execution updates
    const channel = `execution:${executionId}`;
    
    subscribe(channel, (event) => {
      switch (event.type) {
        case 'execution_started':
          setExecution(event.execution);
          break;
        case 'node_started':
          setNodeStatuses((prev) => ({
            ...prev,
            [event.nodeId]: { ...event.data, status: 'running' },
          }));
          break;
        case 'node_completed':
          setNodeStatuses((prev) => ({
            ...prev,
            [event.nodeId]: { ...event.data, status: 'success' },
          }));
          break;
        case 'node_failed':
          setNodeStatuses((prev) => ({
            ...prev,
            [event.nodeId]: { ...event.data, status: 'error' },
          }));
          break;
        case 'execution_completed':
          setExecution(event.execution);
          break;
        case 'execution_failed':
          setExecution(event.execution);
          break;
      }
    });
    
    return () => unsubscribe(channel);
  }, [executionId, subscribe, unsubscribe]);
  
  return { execution, nodeStatuses };
};
```

### 5.2 Execution Visualization

```typescript
// components/workflow/panels/ExecutionPanel.tsx

const ExecutionPanel: React.FC<{ executionId: string }> = ({ executionId }) => {
  const { execution, nodeStatuses } = useExecution(executionId);
  
  if (!execution) return <Typography>No execution selected</Typography>;
  
  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      {/* Execution Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6">Execution #{execution.id.slice(0, 8)}</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
          <Chip
            label={execution.status}
            color={getStatusColor(execution.status)}
            size="small"
          />
          <Typography variant="caption">
            Started: {formatDate(execution.started_at)}
          </Typography>
        </Box>
      </Box>
      
      {/* Node Timeline */}
      <Timeline>
        {Object.entries(nodeStatuses).map(([nodeId, status]) => (
          <TimelineItem key={nodeId}>
            <TimelineSeparator>
              <TimelineDot color={getStatusColor(status.status)} />
              <TimelineConnector />
            </TimelineSeparator>
            <TimelineContent>
              <Typography variant="body2">{status.node_name}</Typography>
              <Typography variant="caption" color="text.secondary">
                {status.duration ? `${status.duration}ms` : 'Running...'}
              </Typography>
              {status.error && (
                <Alert severity="error" sx={{ mt: 1 }}>
                  {status.error}
                </Alert>
              )}
            </TimelineContent>
          </TimelineItem>
        ))}
      </Timeline>
      
      {/* Output Display */}
      {execution.final_output && (
        <Box sx={{ p: 2, bgcolor: 'background.default' }}>
          <Typography variant="subtitle2">Output</Typography>
          <Paper sx={{ p: 1, mt: 1, fontFamily: 'monospace' }}>
            <pre>{JSON.stringify(execution.final_output, null, 2)}</pre>
          </Paper>
        </Box>
      )}
    </Box>
  );
};
```

---

## Phase 6: Advanced Features (Week 6-8)

### 6.1 Workflow Templates

Pre-built workflow templates for common use cases:

1. **RAG Q&A Pipeline**
   - Document loader -> Chunker -> Embedder -> Vector Store
   - Query input -> Retriever -> LLM Chat -> Output

2. **Multi-Model Router**
   - Input -> Model Classifier -> (Route to appropriate model) -> Output

3. **Agentic Loop**
   - Input -> LLM with Tools -> Tool Router -> Tool Execution -> Loop back or Output

4. **Document Processing Pipeline**
   - File Watch -> Document Loader -> Chunker -> Embedder -> Vector Store -> Notification

5. **API Gateway**
   - HTTP Webhook -> Input Validation -> LLM Process -> JSON Output -> HTTP Response

### 6.2 Workflow Versioning

```python
class WorkflowVersion(BaseModel):
    id: str
    workflow_id: str
    version: int
    snapshot: Workflow  # Full workflow state at this version
    change_summary: str
    created_by: Optional[str]
    created_at: datetime
```

### 6.3 Workflow Sharing and Import/Export

```python
@app.post("/api/v1/workflows/{workflow_id}/export")
async def export_workflow(workflow_id: str, format: str = "json"):
    """Export workflow as JSON or YAML."""

@app.post("/api/v1/workflows/import")
async def import_workflow(file: UploadFile):
    """Import workflow from JSON/YAML file."""

@app.post("/api/v1/workflows/{workflow_id}/share")
async def share_workflow(workflow_id: str, settings: ShareSettings):
    """Generate shareable link for workflow."""
```

### 6.4 Workflow Testing

```typescript
// Test mode for workflows
interface WorkflowTestCase {
  id: string;
  name: string;
  triggerData: Record<string, any>;
  expectedOutput: Record<string, any>;
  nodeAssertions: Record<string, {
    output?: Record<string, any>;
    shouldError?: boolean;
  }>;
}

// Test runner
async function runWorkflowTests(
  workflowId: string,
  testCases: WorkflowTestCase[]
): Promise<TestResults> {
  // Execute workflow with test data
  // Compare outputs with expected values
  // Report pass/fail for each test case
}
```

### 6.5 Workflow Analytics

- Execution time per node
- Success/failure rates
- Token usage tracking
- Cost estimation
- Performance bottleneck detection

---

## Database Schema

```sql
-- Workflows
CREATE TABLE workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    nodes_json TEXT NOT NULL,
    connections_json TEXT NOT NULL,
    variables_json TEXT,
    settings_json TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1
);

-- Workflow Versions
CREATE TABLE workflow_versions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    change_summary TEXT,
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE,
    UNIQUE(workflow_id, version)
);

-- Workflow Executions
CREATE TABLE workflow_executions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    workflow_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    trigger_data_json TEXT,
    node_executions_json TEXT,
    final_output_json TEXT,
    error TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
);

-- Workflow Templates
CREATE TABLE workflow_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT,
    workflow_json TEXT NOT NULL,
    parameters_schema_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scheduled Workflows
CREATE TABLE workflow_schedules (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    cron_expression TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    next_run_at TIMESTAMP,
    last_run_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_workflows_active ON workflows(is_active);
CREATE INDEX idx_executions_workflow ON workflow_executions(workflow_id);
CREATE INDEX idx_executions_status ON workflow_executions(status);
CREATE INDEX idx_schedules_next_run ON workflow_schedules(next_run_at);
```

---

## Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Foundation | React Flow integration, base canvas, node component structure |
| 2-3 | Backend Engine | Workflow models, engine, executors, API endpoints |
| 3-4 | Frontend UI | Node palette, property panel, execution panel, dialogs |
| 4-5 | Integrations | LLM, RAG, MCP, API, Database executors |
| 5-6 | Execution | Real-time updates, WebSocket, execution visualization |
| 6-7 | Advanced | Templates, versioning, import/export |
| 7-8 | Polish | Testing, analytics, documentation |

---

## Dependencies to Add

### Frontend
```json
{
  "reactflow": "^11.10.0",
  "@reactflow/background": "^11.3.0",
  "@reactflow/controls": "^11.2.0",
  "@reactflow/minimap": "^11.7.0"
}
```

### Backend
```
# requirements.txt additions
networkx>=3.0  # For graph operations
jsonschema>=4.0  # For config validation
apscheduler>=3.10  # For scheduled workflows
```

---

## Success Metrics

1. **Usability**
   - Time to create first workflow < 5 minutes
   - Node connection success rate > 95%
   - User-reported ease of use score > 4/5

2. **Performance**
   - Canvas renders 100+ nodes at 60fps
   - Execution status updates < 100ms latency
   - Workflow save/load < 500ms

3. **Reliability**
   - Workflow execution success rate > 99%
   - No data loss on browser refresh
   - Graceful error handling for all node types

4. **Adoption**
   - 50% of users create at least one workflow
   - Average workflows per user > 3
   - Template usage rate > 30%

---

## Open Questions

1. Should we support sub-workflows (workflow as a node)?
2. How to handle secrets/credentials in workflow configs?
3. Should workflows be exportable to other formats (e.g., Langchain, DSPy)?
4. Multi-user collaboration on workflows?
5. Rate limiting and resource quotas for workflow execution?

---

## Next Steps

1. Review and approve this plan
2. Set up React Flow in frontend
3. Create base node component and canvas
4. Implement workflow backend models and storage
5. Build first set of node executors (LLM, RAG, HTTP)
6. Connect frontend to backend
7. Iterate based on testing and feedback
