"""
MCP API Routes

Provides REST API endpoints for managing MCP server connections,
listing tools, and executing tool calls.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mcp", tags=["mcp"])


# Request/Response Models

class AddServerRequest(BaseModel):
    """Request to add a new MCP server."""
    name: str = Field(..., description="Unique server name")
    transport: str = Field(..., description="Transport type: 'stdio' or 'http'")
    # Stdio fields
    command: Optional[str] = Field(None, description="Command for stdio transport")
    args: Optional[List[str]] = Field(default_factory=list, description="Command arguments")
    env: Optional[Dict[str, str]] = Field(default_factory=dict, description="Environment variables")
    # HTTP fields  
    url: Optional[str] = Field(None, description="URL for HTTP transport")
    auth_token: Optional[str] = Field(None, description="Auth token for HTTP")
    # Common
    description: Optional[str] = Field(None, description="Server description")
    icon: Optional[str] = Field(None, description="Icon emoji")
    enabled: bool = Field(True, description="Whether server is enabled")
    auto_connect: bool = Field(True, description="Connect on startup")


class AddPopularServerRequest(BaseModel):
    """Request to add a popular pre-configured MCP server."""
    server_id: str = Field(..., description="Popular server ID (e.g., 'filesystem', 'github')")
    path: Optional[str] = Field(None, description="Path for servers that require it")
    env_vars: Optional[Dict[str, str]] = Field(default_factory=dict, description="Environment variables")


class UpdateServerRequest(BaseModel):
    """Request to update an MCP server configuration."""
    enabled: Optional[bool] = None
    auto_connect: Optional[bool] = None
    description: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    auth_token: Optional[str] = None


class ExecuteToolRequest(BaseModel):
    """Request to execute an MCP tool."""
    arguments: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Tool arguments")


# Helper functions

def get_mcp_components(request: Request):
    """Get MCP components from app state."""
    mcp_client = getattr(request.app.state, 'mcp_client_manager', None)
    if not mcp_client:
        raise HTTPException(status_code=503, detail="MCP module not initialized")
    return mcp_client


def get_mcp_config_store(request: Request):
    """Get MCP config store from app state."""
    mcp_client = get_mcp_components(request)
    return mcp_client.config_store


# API Routes

@router.get("/status")
async def get_mcp_status(request: Request):
    """Get overall MCP module status."""
    try:
        mcp_client = get_mcp_components(request)
        return {
            "available": mcp_client.is_available,
            "connected_servers": len([c for c in mcp_client.connections.values() if c.connected]),
            "total_tools": len(mcp_client.get_all_tools_openai_format()),
        }
    except HTTPException:
        return {
            "available": False,
            "connected_servers": 0,
            "total_tools": 0,
        }


@router.get("/servers")
async def list_servers(request: Request):
    """List all configured MCP servers with their status."""
    mcp_client = get_mcp_components(request)
    config_store = mcp_client.config_store
    
    servers = []
    for config in config_store.list_servers():
        status = mcp_client.get_connection_status(config.name)
        servers.append({
            "name": config.name,
            "transport": config.transport,
            "description": config.description,
            "icon": config.icon,
            "enabled": config.enabled,
            "auto_connect": config.auto_connect,
            "connected": status.connected if status else False,
            "tools_count": status.tools_count if status else 0,
            "resources_count": status.resources_count if status else 0,
            "error": status.error if status else None,
            "latency_ms": status.latency_ms if status else None,
            "last_connected": status.last_connected if status else None,
            # Config details
            "command": config.command,
            "args": config.args,
            "url": config.url,
            "created_at": config.created_at,
        })
    
    return {"servers": servers}


@router.post("/servers")
async def add_server(request: Request, body: AddServerRequest):
    """Add a new MCP server configuration."""
    mcp_client = get_mcp_components(request)
    config_store = mcp_client.config_store
    
    # Import here to avoid circular imports
    from modules.mcp.config import MCPServerConfig, TransportType
    
    try:
        transport = TransportType(body.transport)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid transport: {body.transport}")
    
    # Validate transport-specific fields
    if transport == TransportType.STDIO and not body.command:
        raise HTTPException(status_code=400, detail="Stdio transport requires 'command'")
    if transport == TransportType.HTTP and not body.url:
        raise HTTPException(status_code=400, detail="HTTP transport requires 'url'")
    
    config = MCPServerConfig(
        name=body.name,
        transport=transport,
        command=body.command,
        args=body.args or [],
        env=body.env or {},
        url=body.url,
        auth_token=body.auth_token,
        description=body.description,
        icon=body.icon,
        enabled=body.enabled,
        auto_connect=body.auto_connect,
    )
    
    try:
        config_store.add_server(config)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    
    # Auto-connect if enabled
    connection_status = None
    if body.auto_connect:
        try:
            connection = await mcp_client.connect(body.name)
            connection_status = {
                "connected": connection.connected,
                "error": connection.error,
                "tools_count": len(connection.tools),
            }
        except Exception as e:
            connection_status = {"connected": False, "error": str(e)}
    
    return {
        "success": True,
        "server": config.model_dump(),
        "connection": connection_status,
    }


@router.post("/servers/popular")
async def add_popular_server(request: Request, body: AddPopularServerRequest):
    """Add a popular pre-configured MCP server."""
    mcp_client = get_mcp_components(request)
    config_store = mcp_client.config_store
    
    from modules.mcp.popular_servers import get_popular_server_config, POPULAR_MCP_SERVERS
    
    if body.server_id not in POPULAR_MCP_SERVERS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown server: {body.server_id}. Available: {list(POPULAR_MCP_SERVERS.keys())}"
        )
    
    try:
        config = get_popular_server_config(
            body.server_id,
            path=body.path,
            env_vars=body.env_vars,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Check for duplicates
    existing = config_store.get_server(config.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Server '{config.name}' already exists")
    
    config_store.add_server(config)
    
    # Auto-connect
    connection_status = None
    try:
        connection = await mcp_client.connect(config.name)
        connection_status = {
            "connected": connection.connected,
            "error": connection.error,
            "tools_count": len(connection.tools),
        }
    except Exception as e:
        connection_status = {"connected": False, "error": str(e)}
    
    return {
        "success": True,
        "server": config.model_dump(),
        "connection": connection_status,
    }


@router.get("/servers/popular")
async def list_popular_servers(request: Request):
    """List available popular MCP servers."""
    from modules.mcp.popular_servers import list_popular_servers
    
    return {"servers": list_popular_servers()}


@router.get("/servers/{name}")
async def get_server(request: Request, name: str):
    """Get a specific server's configuration and status."""
    mcp_client = get_mcp_components(request)
    config = mcp_client.config_store.get_server(name)
    
    if not config:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
    
    status = mcp_client.get_connection_status(name)
    
    return {
        "config": config.model_dump(),
        "status": status.model_dump() if status else None,
    }


@router.patch("/servers/{name}")
async def update_server(request: Request, name: str, body: UpdateServerRequest):
    """Update an MCP server configuration."""
    mcp_client = get_mcp_components(request)
    config_store = mcp_client.config_store
    
    config = config_store.get_server(name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
    
    # Apply updates
    if body.enabled is not None:
        config.enabled = body.enabled
    if body.auto_connect is not None:
        config.auto_connect = body.auto_connect
    if body.description is not None:
        config.description = body.description
    if body.env is not None:
        config.env = body.env
    if body.auth_token is not None:
        config.auth_token = body.auth_token
    
    config_store.update_server(name, config)
    
    return {"success": True, "server": config.model_dump()}


@router.delete("/servers/{name}")
async def delete_server(request: Request, name: str):
    """Remove an MCP server configuration."""
    mcp_client = get_mcp_components(request)
    
    # Disconnect first
    await mcp_client.disconnect(name)
    
    # Remove config
    removed = mcp_client.config_store.remove_server(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
    
    return {"success": True}


@router.post("/servers/{name}/connect")
async def connect_server(request: Request, name: str):
    """Connect to an MCP server."""
    mcp_client = get_mcp_components(request)
    
    config = mcp_client.config_store.get_server(name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
    
    try:
        connection = await mcp_client.connect(name)
        return {
            "success": True,
            "connected": connection.connected,
            "error": connection.error,
            "tools_count": len(connection.tools),
            "resources_count": len(connection.resources),
            "latency_ms": connection.latency_ms,
        }
    except Exception as e:
        return {
            "success": False,
            "connected": False,
            "error": str(e),
        }


@router.post("/servers/{name}/disconnect")
async def disconnect_server(request: Request, name: str):
    """Disconnect from an MCP server."""
    mcp_client = get_mcp_components(request)
    
    disconnected = await mcp_client.disconnect(name)
    
    return {"success": disconnected}


@router.get("/servers/{name}/tools")
async def list_server_tools(request: Request, name: str):
    """List tools available from an MCP server."""
    mcp_client = get_mcp_components(request)
    
    if name not in mcp_client.connections:
        raise HTTPException(status_code=400, detail=f"Server '{name}' is not connected")
    
    connection = mcp_client.connections[name]
    
    if not connection.connected:
        raise HTTPException(status_code=400, detail=f"Server '{name}' is not connected")
    
    # Return both raw MCP tools and OpenAI format
    tools_openai = mcp_client.get_server_tools(name)
    
    tools_raw = []
    for tool in connection.tools:
        if hasattr(tool, 'name'):
            tools_raw.append({
                "name": tool.name,
                "description": getattr(tool, 'description', ''),
                "inputSchema": getattr(tool, 'inputSchema', {}),
            })
        else:
            tools_raw.append(tool)
    
    return {
        "server": name,
        "tools": tools_raw,
        "tools_openai_format": tools_openai,
    }


@router.get("/servers/{name}/resources")
async def list_server_resources(request: Request, name: str):
    """List resources available from an MCP server."""
    mcp_client = get_mcp_components(request)
    
    if name not in mcp_client.connections:
        raise HTTPException(status_code=400, detail=f"Server '{name}' is not connected")
    
    connection = mcp_client.connections[name]
    
    if not connection.connected:
        raise HTTPException(status_code=400, detail=f"Server '{name}' is not connected")
    
    resources = []
    for resource in connection.resources:
        if hasattr(resource, 'uri'):
            resources.append({
                "uri": resource.uri,
                "name": getattr(resource, 'name', ''),
                "description": getattr(resource, 'description', ''),
                "mimeType": getattr(resource, 'mimeType', None),
            })
        else:
            resources.append(resource)
    
    return {
        "server": name,
        "resources": resources,
    }


@router.post("/servers/{name}/tools/{tool_name}")
async def execute_tool(request: Request, name: str, tool_name: str, body: ExecuteToolRequest):
    """Execute a tool on an MCP server."""
    mcp_client = get_mcp_components(request)
    
    if name not in mcp_client.connections:
        raise HTTPException(status_code=400, detail=f"Server '{name}' is not connected")
    
    try:
        result = await mcp_client.call_tool(name, tool_name, body.arguments)
        
        # Format result for response
        if hasattr(result, 'content'):
            # MCP SDK result object
            content = []
            for item in result.content:
                if hasattr(item, 'text'):
                    content.append({"type": "text", "text": item.text})
                elif hasattr(item, 'data'):
                    content.append({"type": "data", "data": item.data})
                else:
                    content.append({"type": "unknown", "value": str(item)})
            
            return {
                "success": True,
                "result": content,
                "isError": getattr(result, 'isError', False),
            }
        else:
            return {
                "success": True,
                "result": result,
                "isError": False,
            }
            
    except Exception as e:
        logger.error(f"Tool execution failed: {name}/{tool_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


@router.get("/tools")
async def list_all_tools(request: Request):
    """List all tools from all connected MCP servers (OpenAI format)."""
    mcp_client = get_mcp_components(request)
    
    tools = mcp_client.get_all_tools_openai_format()
    
    return {
        "tools": tools,
        "count": len(tools),
    }


@router.get("/tools/summary")
async def get_tools_summary(request: Request):
    """Get a human-readable summary of all MCP tools."""
    mcp_client = get_mcp_components(request)
    
    return {
        "summary": mcp_client.get_tools_summary(),
    }


@router.post("/tools/execute")
async def execute_tool_by_name(request: Request, tool_name: str, body: ExecuteToolRequest):
    """Execute an MCP tool by its namespaced name (mcp_server_tool)."""
    mcp_client = get_mcp_components(request)
    
    from modules.mcp.tools_bridge import ToolsBridge
    
    server_name, actual_tool_name = ToolsBridge.parse_mcp_tool_call(tool_name)
    
    if not server_name or not actual_tool_name:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid MCP tool name: {tool_name}. Expected format: mcp_servername_toolname"
        )
    
    if server_name not in mcp_client.connections:
        raise HTTPException(status_code=400, detail=f"Server '{server_name}' is not connected")
    
    try:
        result = await mcp_client.call_tool(server_name, actual_tool_name, body.arguments)
        
        # Format result
        formatted = ToolsBridge.format_tool_result(result)
        
        return {
            "success": True,
            "server": server_name,
            "tool": actual_tool_name,
            "result": formatted,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")
