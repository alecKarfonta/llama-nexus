"""
Workflow Data Models
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid


class PortType(str, Enum):
    """Data types for node ports"""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    ANY = "any"


class NodeCategory(str, Enum):
    """Categories for workflow nodes"""
    TRIGGER = "trigger"
    LLM = "llm"
    RAG = "rag"
    TOOLS = "tools"
    DATA = "data"
    CONTROL = "control"
    API = "api"
    MCP = "mcp"
    DATABASE = "database"
    OUTPUT = "output"


class ExecutionStatus(str, Enum):
    """Status of workflow/node execution"""
    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class PortDefinition(BaseModel):
    """Definition of a node input/output port"""
    id: str
    name: str
    type: PortType
    required: bool = False
    description: Optional[str] = None


class Position(BaseModel):
    """Position on the canvas"""
    x: float
    y: float


class WorkflowNodeData(BaseModel):
    """Data associated with a workflow node"""
    label: str
    nodeType: str
    config: Dict[str, Any] = Field(default_factory=dict)
    inputs: List[PortDefinition] = Field(default_factory=list)
    outputs: List[PortDefinition] = Field(default_factory=list)


class WorkflowNode(BaseModel):
    """A node in the workflow"""
    id: str
    type: str  # Node type (llm_chat, retriever, etc.)
    position: Position
    data: WorkflowNodeData


class WorkflowConnection(BaseModel):
    """A connection between two nodes"""
    id: str
    source: str  # Source node ID
    sourceHandle: str  # Source port ID
    target: str  # Target node ID
    targetHandle: str  # Target port ID


class WorkflowSettings(BaseModel):
    """Workflow execution settings"""
    timeout: int = 300  # Seconds
    retryOnFailure: bool = False
    maxRetries: int = 3
    logLevel: str = "info"


class WorkflowVariables(BaseModel):
    """Workflow-level variables"""
    variables: Dict[str, Any] = Field(default_factory=dict)


class Workflow(BaseModel):
    """Complete workflow definition"""
    id: str = Field(default_factory=lambda: f"wf-{uuid.uuid4().hex[:12]}")
    name: str
    description: Optional[str] = None
    nodes: List[WorkflowNode] = Field(default_factory=list)
    connections: List[WorkflowConnection] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    settings: WorkflowSettings = Field(default_factory=WorkflowSettings)
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    isActive: bool = True


class WorkflowCreate(BaseModel):
    """Request model for creating a workflow"""
    name: str
    description: Optional[str] = None
    nodes: List[WorkflowNode] = Field(default_factory=list)
    connections: List[WorkflowConnection] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    settings: Optional[WorkflowSettings] = None


class WorkflowUpdate(BaseModel):
    """Request model for updating a workflow"""
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[WorkflowNode]] = None
    connections: Optional[List[WorkflowConnection]] = None
    variables: Optional[Dict[str, Any]] = None
    settings: Optional[WorkflowSettings] = None
    isActive: Optional[bool] = None


class NodeExecution(BaseModel):
    """Execution state of a single node"""
    nodeId: str
    nodeName: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    startedAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None
    duration: Optional[int] = None  # Milliseconds
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    logs: List[str] = Field(default_factory=list)


class WorkflowExecution(BaseModel):
    """Execution state of a workflow"""
    id: str = Field(default_factory=lambda: f"exec-{uuid.uuid4().hex[:12]}")
    workflowId: str
    workflowVersion: int = 1
    status: ExecutionStatus = ExecutionStatus.PENDING
    startedAt: datetime = Field(default_factory=datetime.utcnow)
    completedAt: Optional[datetime] = None
    triggerData: Dict[str, Any] = Field(default_factory=dict)
    nodeExecutions: Dict[str, NodeExecution] = Field(default_factory=dict)
    finalOutput: Optional[Any] = None
    error: Optional[str] = None


class NodeTypeDefinition(BaseModel):
    """Definition of a node type for the registry"""
    type: str
    displayName: str
    category: NodeCategory
    description: str
    color: Optional[str] = None
    inputs: List[PortDefinition] = Field(default_factory=list)
    outputs: List[PortDefinition] = Field(default_factory=list)
    configSchema: Dict[str, Any] = Field(default_factory=dict)
