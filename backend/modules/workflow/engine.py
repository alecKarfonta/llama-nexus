"""
Workflow Engine - Executes workflows by traversing the node graph
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

from .models import (
    Workflow,
    WorkflowNode,
    WorkflowConnection,
    WorkflowExecution,
    NodeExecution,
    ExecutionStatus,
)
from .storage import WorkflowStorage
from .executors import get_executor, NODE_EXECUTORS
from .executors.base import ExecutionContext, NodeExecutor

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """
    Executes workflows by traversing the node graph.
    
    The engine:
    1. Builds an execution graph from the workflow
    2. Determines execution order (topological sort)
    3. Executes nodes in order, passing outputs to inputs
    4. Handles control flow (conditions, loops)
    5. Reports progress via callbacks
    """
    
    def __init__(self, storage: WorkflowStorage):
        self.storage = storage
        self.running_executions: Dict[str, WorkflowExecution] = {}
        self.cancelled_executions: Set[str] = set()
        self.progress_callbacks: Dict[str, callable] = {}
    
    async def execute_workflow(
        self,
        workflow: Workflow,
        trigger_data: Dict[str, Any] = None,
        execution_id: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> WorkflowExecution:
        """
        Execute a workflow from start to finish.
        
        Args:
            workflow: The workflow to execute
            trigger_data: Data from the trigger (webhook body, etc.)
            execution_id: Optional custom execution ID
            progress_callback: Callback for progress updates
            
        Returns:
            WorkflowExecution with final status and outputs
        """
        # Create execution record
        execution = self.storage.create_execution(
            workflow.id,
            trigger_data or {}
        )
        if execution_id:
            execution.id = execution_id
        
        execution.status = ExecutionStatus.RUNNING
        self.running_executions[execution.id] = execution
        
        if progress_callback:
            self.progress_callbacks[execution.id] = progress_callback
        
        logger.info(f"Starting workflow execution: {execution.id}")
        
        try:
            # Build execution context
            context = ExecutionContext(
                workflow_id=workflow.id,
                execution_id=execution.id,
                variables={
                    "_trigger_data": trigger_data or {},
                    **workflow.variables,
                },
            )
            
            # Build execution graph
            graph = self._build_execution_graph(workflow)
            
            # Execute nodes in order
            node_outputs: Dict[str, Dict[str, Any]] = {}
            
            for node in graph:
                if execution.id in self.cancelled_executions:
                    execution.status = ExecutionStatus.CANCELLED
                    break
                
                # Gather inputs from connected nodes
                inputs = self._gather_inputs(node, workflow.connections, node_outputs)
                
                # Execute node
                node_exec = await self._execute_node(node, inputs, context)
                execution.nodeExecutions[node.id] = node_exec
                
                # Store outputs
                if node_exec.status == ExecutionStatus.SUCCESS:
                    node_outputs[node.id] = node_exec.outputs
                else:
                    # Node failed
                    execution.status = ExecutionStatus.ERROR
                    execution.error = node_exec.error
                    break
                
                # Update storage and notify
                self.storage.update_execution(execution)
                await self._notify_progress(execution, node_exec)
            
            # Determine final output
            if execution.status != ExecutionStatus.ERROR and execution.status != ExecutionStatus.CANCELLED:
                execution.status = ExecutionStatus.SUCCESS
                execution.finalOutput = self._extract_final_output(node_outputs, context)
            
            execution.completedAt = datetime.utcnow()
            self.storage.update_execution(execution)
            
            logger.info(f"Workflow execution completed: {execution.id} - {execution.status}")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {execution.id} - {e}")
            execution.status = ExecutionStatus.ERROR
            execution.error = str(e)
            execution.completedAt = datetime.utcnow()
            self.storage.update_execution(execution)
        
        finally:
            # Cleanup
            self.running_executions.pop(execution.id, None)
            self.cancelled_executions.discard(execution.id)
            self.progress_callbacks.pop(execution.id, None)
        
        return execution
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        if execution_id in self.running_executions:
            self.cancelled_executions.add(execution_id)
            logger.info(f"Cancellation requested for: {execution_id}")
            return True
        return False
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running execution (not fully implemented)"""
        if execution_id in self.running_executions:
            execution = self.running_executions[execution_id]
            execution.status = ExecutionStatus.PAUSED
            self.storage.update_execution(execution)
            return True
        return False
    
    def _build_execution_graph(self, workflow: Workflow) -> List[WorkflowNode]:
        """
        Build execution order using topological sort.
        Returns nodes in order they should be executed.
        """
        # Build adjacency list
        edges: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = {node.id: 0 for node in workflow.nodes}
        
        for conn in workflow.connections:
            edges[conn.source].append(conn.target)
            in_degree[conn.target] = in_degree.get(conn.target, 0) + 1
        
        # Kahn's algorithm for topological sort
        queue = [
            node.id for node in workflow.nodes 
            if in_degree.get(node.id, 0) == 0
        ]
        order = []
        
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            
            for neighbor in edges[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(order) != len(workflow.nodes):
            logger.warning("Workflow contains cycles, some nodes may not execute")
        
        # Map IDs back to nodes
        node_map = {node.id: node for node in workflow.nodes}
        return [node_map[nid] for nid in order if nid in node_map]
    
    def _gather_inputs(
        self,
        node: WorkflowNode,
        connections: List[WorkflowConnection],
        node_outputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Gather inputs for a node from connected node outputs"""
        inputs = {}
        
        for conn in connections:
            if conn.target == node.id:
                source_outputs = node_outputs.get(conn.source, {})
                source_value = source_outputs.get(conn.sourceHandle)
                
                if source_value is not None:
                    inputs[conn.targetHandle] = source_value
        
        return inputs
    
    async def _execute_node(
        self,
        node: WorkflowNode,
        inputs: Dict[str, Any],
        context: ExecutionContext
    ) -> NodeExecution:
        """Execute a single node"""
        node_exec = NodeExecution(
            nodeId=node.id,
            nodeName=node.data.label,
            status=ExecutionStatus.RUNNING,
            startedAt=datetime.utcnow(),
            inputs=inputs,
        )
        
        try:
            # Get executor class
            executor_class = get_executor(node.type)
            if not executor_class:
                raise ValueError(f"Unknown node type: {node.type}")
            
            # Create executor instance
            executor: NodeExecutor = executor_class(node.data.config)
            
            # Validate inputs
            errors = executor.validate_inputs(inputs)
            if errors:
                raise ValueError(f"Input validation failed: {errors}")
            
            context.log(f"Executing node: {node.data.label} ({node.type})")
            
            # Execute
            start_time = datetime.utcnow()
            outputs = await executor.execute(inputs, context)
            end_time = datetime.utcnow()
            
            # Update execution record
            node_exec.status = ExecutionStatus.SUCCESS
            node_exec.outputs = outputs
            node_exec.completedAt = end_time
            node_exec.duration = int((end_time - start_time).total_seconds() * 1000)
            node_exec.logs = context.logs.copy()
            
            context.log(f"Node completed: {node.data.label} ({node_exec.duration}ms)")
            
        except Exception as e:
            logger.error(f"Node execution failed: {node.id} - {e}")
            node_exec.status = ExecutionStatus.ERROR
            node_exec.error = str(e)
            node_exec.completedAt = datetime.utcnow()
            node_exec.logs = context.logs.copy()
        
        return node_exec
    
    def _extract_final_output(
        self,
        node_outputs: Dict[str, Dict[str, Any]],
        context: ExecutionContext
    ) -> Any:
        """Extract the final output from workflow execution"""
        # Check for explicit outputs
        outputs = {}
        for key, value in context.variables.items():
            if key.startswith("_output_"):
                output_name = key[8:]  # Remove "_output_" prefix
                outputs[output_name] = value
        
        if outputs:
            return outputs if len(outputs) > 1 else list(outputs.values())[0]
        
        # Return last node's output
        if node_outputs:
            last_outputs = list(node_outputs.values())[-1]
            if "_final_output" in last_outputs:
                return last_outputs["_final_output"]
            return last_outputs
        
        return None
    
    async def _notify_progress(
        self,
        execution: WorkflowExecution,
        node_exec: NodeExecution
    ):
        """Notify progress callback"""
        callback = self.progress_callbacks.get(execution.id)
        if callback:
            try:
                await callback({
                    "execution_id": execution.id,
                    "node_id": node_exec.nodeId,
                    "node_name": node_exec.nodeName,
                    "status": node_exec.status.value,
                    "duration": node_exec.duration,
                })
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def get_available_node_types(self) -> List[Dict[str, Any]]:
        """Get list of available node types"""
        node_types = []
        
        for node_type, executor_class in NODE_EXECUTORS.items():
            node_types.append({
                "type": node_type,
                "displayName": getattr(executor_class, 'display_name', node_type),
                "category": getattr(executor_class, 'category', 'data'),
                "description": getattr(executor_class, 'description', ''),
            })
        
        return node_types
