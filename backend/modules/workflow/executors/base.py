"""
Base Node Executor - Abstract base class for all node executors
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context passed to node executors during execution"""
    
    # Workflow context
    workflow_id: str
    execution_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Service references (populated by engine)
    llm_service: Any = None
    rag_service: Any = None
    http_client: Any = None
    
    # Execution state
    logs: List[str] = field(default_factory=list)
    cancelled: bool = False
    
    def log(self, message: str, level: str = "info"):
        """Add a log message"""
        self.logs.append(f"[{level.upper()}] {message}")
        getattr(logger, level, logger.info)(f"[{self.execution_id}] {message}")
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a workflow variable"""
        return self.variables.get(name, default)
    
    def set_variable(self, name: str, value: Any):
        """Set a workflow variable"""
        self.variables[name] = value


class NodeExecutor(ABC):
    """
    Abstract base class for node executors.
    
    Each node type must implement an executor that:
    1. Validates configuration
    2. Validates inputs
    3. Executes the node logic
    4. Returns outputs
    """
    
    # Node metadata (override in subclasses)
    node_type: str = "base"
    display_name: str = "Base Node"
    category: str = "data"
    description: str = "Base node executor"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the executor with node configuration.
        
        Args:
            config: Node configuration from the workflow
        """
        self.config = config
        self.validate_config()
    
    def validate_config(self) -> List[str]:
        """
        Validate node configuration.
        
        Returns:
            List of error messages (empty if valid)
        """
        return []
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """
        Validate inputs before execution.
        
        Args:
            inputs: Input values from connected nodes
            
        Returns:
            List of error messages (empty if valid)
        """
        return []
    
    @abstractmethod
    async def execute(
        self, 
        inputs: Dict[str, Any], 
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute the node logic.
        
        Args:
            inputs: Input values from connected nodes
            context: Execution context with services and state
            
        Returns:
            Dictionary of output values keyed by port ID
        """
        raise NotImplementedError
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Helper to get a config value with default"""
        return self.config.get(key, default)


class PassthroughExecutor(NodeExecutor):
    """Simple executor that passes input through unchanged"""
    
    node_type = "passthrough"
    display_name = "Passthrough"
    description = "Passes input through unchanged"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        return {"output": inputs.get("input")}
