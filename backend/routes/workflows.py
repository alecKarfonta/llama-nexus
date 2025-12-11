"""
Workflow API Routes
Provides endpoints for workflow management, execution, and templates.
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


def get_workflow_components(request: Request):
    """Helper to get workflow components from app state."""
    workflow_available = getattr(request.app.state, 'workflow_available', False)
    if not workflow_available:
        raise HTTPException(status_code=503, detail="Workflow system not available")
    storage = getattr(request.app.state, 'workflow_storage', None)
    engine = getattr(request.app.state, 'workflow_engine', None)
    if not storage or not engine:
        raise HTTPException(status_code=503, detail="Workflow system not initialized")
    return {'storage': storage, 'engine': engine}


@router.get("")
async def list_workflows(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None
):
    """List all workflows."""
    wf = get_workflow_components(request)
    result = wf['storage'].list_workflows(limit=limit, offset=offset, search=search)
    return {
        "workflows": [w.dict() for w in result['workflows']],
        "total": result['total'],
        "limit": result['limit'],
        "offset": result['offset'],
        "has_more": result['has_more'],
    }


@router.post("")
async def create_workflow(request: Request):
    """Create a new workflow."""
    from modules.workflow.models import WorkflowCreate
    
    wf = get_workflow_components(request)
    data = await request.json()
    
    workflow_create = WorkflowCreate(**data)
    workflow = wf['storage'].create_workflow(workflow_create)
    
    return workflow.dict()


# Workflow Templates - must be before /{workflow_id} routes
@router.get("/templates")
async def list_workflow_templates(request: Request):
    """List available workflow templates."""
    workflow_available = getattr(request.app.state, 'workflow_available', False)
    if not workflow_available:
        raise HTTPException(status_code=503, detail="Workflow system not available")
    
    try:
        from modules.workflow.templates import get_workflow_templates
        templates = get_workflow_templates()
        return {"templates": templates}
    except ImportError:
        return {"templates": []}


@router.get("/templates/{template_id}")
async def get_workflow_template(request: Request, template_id: str):
    """Get a specific workflow template."""
    workflow_available = getattr(request.app.state, 'workflow_available', False)
    if not workflow_available:
        raise HTTPException(status_code=503, detail="Workflow system not available")
    
    try:
        from modules.workflow.templates import get_workflow_template as get_template
        template = get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        return template
    except ImportError:
        raise HTTPException(status_code=404, detail="Templates not available")


@router.post("/templates/{template_id}/instantiate")
async def instantiate_workflow_template(request: Request, template_id: str):
    """Create a new workflow from a template."""
    from modules.workflow.models import WorkflowCreate
    
    wf = get_workflow_components(request)
    
    try:
        from modules.workflow.templates import get_workflow_template as get_template
        template = get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Get optional parameters from request body
        try:
            params = await request.json()
        except:
            params = {}
        
        # Create workflow from template
        workflow_data = {
            "name": params.get("name", template["name"]),
            "description": params.get("description", template["description"]),
            "nodes": template["nodes"],
            "connections": template["connections"],
            "variables": params.get("variables", {}),
        }
        
        workflow_create = WorkflowCreate(**workflow_data)
        workflow = wf['storage'].create_workflow(workflow_create)
        
        return workflow.dict()
        
    except ImportError:
        raise HTTPException(status_code=404, detail="Templates not available")


@router.get("/nodes/types")
async def list_node_types(request: Request):
    """List available node types."""
    workflow_available = getattr(request.app.state, 'workflow_available', False)
    if not workflow_available:
        raise HTTPException(status_code=503, detail="Workflow system not available")
    
    wf = get_workflow_components(request)
    node_types = wf['engine'].get_available_node_types()
    
    return {"node_types": node_types}


@router.post("/validate")
async def validate_workflow(request: Request):
    """Validate a workflow definition."""
    from modules.workflow.models import WorkflowCreate
    from modules.workflow.executors import NODE_EXECUTORS
    
    wf = get_workflow_components(request)
    data = await request.json()
    
    errors = []
    warnings = []
    
    try:
        # Try to create workflow object
        workflow_create = WorkflowCreate(**data)
        
        # Check nodes
        for node in workflow_create.nodes:
            node_type = node.type
            if node_type not in NODE_EXECUTORS:
                errors.append(f"Unknown node type: {node_type}")
        
        # Check connections reference valid nodes
        node_ids = {n.id for n in workflow_create.nodes}
        for conn in workflow_create.connections:
            if conn.source not in node_ids:
                errors.append(f"Connection references unknown source node: {conn.source}")
            if conn.target not in node_ids:
                errors.append(f"Connection references unknown target node: {conn.target}")
        
        # Check for cycles (basic check)
        if len(workflow_create.nodes) > 0 and len(workflow_create.connections) >= len(workflow_create.nodes):
            warnings.append("Workflow may contain cycles")
        
    except Exception as e:
        errors.append(f"Invalid workflow structure: {e}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


@router.get("/executions/{execution_id}")
async def get_execution(request: Request, execution_id: str):
    """Get execution details."""
    wf = get_workflow_components(request)
    execution = wf['storage'].get_execution(execution_id)
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    return execution.dict()


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(request: Request, execution_id: str):
    """Cancel a running execution."""
    wf = get_workflow_components(request)
    
    if await wf['engine'].cancel_execution(execution_id):
        return {"status": "cancelled", "execution_id": execution_id}
    
    raise HTTPException(status_code=404, detail="Execution not found or not running")


@router.get("/{workflow_id}")
async def get_workflow(request: Request, workflow_id: str):
    """Get a workflow by ID."""
    wf = get_workflow_components(request)
    workflow = wf['storage'].get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return workflow.dict()


@router.put("/{workflow_id}")
async def update_workflow(request: Request, workflow_id: str):
    """Update a workflow."""
    from modules.workflow.models import WorkflowUpdate
    
    wf = get_workflow_components(request)
    data = await request.json()
    
    workflow_update = WorkflowUpdate(**data)
    workflow = wf['storage'].update_workflow(workflow_id, workflow_update)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return workflow.dict()


@router.delete("/{workflow_id}")
async def delete_workflow(request: Request, workflow_id: str):
    """Delete a workflow."""
    wf = get_workflow_components(request)
    
    if wf['storage'].delete_workflow(workflow_id):
        return {"status": "deleted", "workflow_id": workflow_id}
    
    raise HTTPException(status_code=404, detail="Workflow not found")


@router.get("/{workflow_id}/versions")
async def get_workflow_versions(request: Request, workflow_id: str):
    """Get version history for a workflow."""
    wf = get_workflow_components(request)
    versions = wf['storage'].get_workflow_versions(workflow_id)
    return {"versions": versions}


@router.post("/{workflow_id}/restore/{version}")
async def restore_workflow_version(request: Request, workflow_id: str, version: int):
    """Restore a workflow to a specific version."""
    wf = get_workflow_components(request)
    workflow = wf['storage'].restore_workflow_version(workflow_id, version)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Version not found")
    
    return workflow.dict()


@router.post("/{workflow_id}/execute")
async def execute_workflow(
    request: Request, 
    workflow_id: str,
    background_tasks: BackgroundTasks
):
    """Execute a workflow."""
    wf = get_workflow_components(request)
    
    # Get workflow
    workflow = wf['storage'].get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Get trigger data from request body
    try:
        trigger_data = await request.json()
    except:
        trigger_data = {}
    
    # Create execution record first
    execution = wf['storage'].create_execution(workflow_id, trigger_data)
    
    # Start execution in background
    background_tasks.add_task(
        wf['engine'].execute_workflow,
        workflow,
        trigger_data,
        execution.id
    )
    
    return {
        "execution_id": execution.id,
        "workflow_id": workflow_id,
        "status": "started",
    }


@router.get("/{workflow_id}/executions")
async def list_workflow_executions(
    request: Request,
    workflow_id: str,
    limit: int = 50,
    offset: int = 0
):
    """List executions for a workflow."""
    wf = get_workflow_components(request)
    result = wf['storage'].list_executions(workflow_id=workflow_id, limit=limit, offset=offset)
    return {
        "executions": [e.dict() for e in result['executions']],
        "total": result['total'],
        "limit": result['limit'],
        "offset": result['offset'],
        "has_more": result['has_more'],
    }
