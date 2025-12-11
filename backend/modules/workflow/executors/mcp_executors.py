"""
MCP (Model Context Protocol) Node Executors
Integrate with MCP servers for tools and resources
"""

from typing import Dict, Any, List, Optional
import httpx
import os
import json
from .base import NodeExecutor, ExecutionContext


class MCPToolExecutor(NodeExecutor):
    """Call a tool on an MCP server"""
    
    node_type = "mcp_tool"
    display_name = "MCP Tool"
    category = "mcp"
    description = "Call a tool on an MCP server"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        args = inputs.get("args", {})
        
        server = self.get_config_value("server", "")
        tool_name = self.get_config_value("tool", "")
        timeout = self.get_config_value("timeout", 30000) / 1000  # Convert to seconds
        
        if not server or not tool_name:
            raise ValueError("MCP server and tool name are required")
        
        context.log(f"Calling MCP tool: {tool_name} on {server}")
        
        # Determine how to connect to the MCP server
        # This supports both HTTP-based MCP servers and stdio servers via a proxy
        mcp_proxy_url = os.environ.get("MCP_PROXY_URL", "http://localhost:3100")
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{mcp_proxy_url}/tools/call",
                    json={
                        "server": server,
                        "tool": tool_name,
                        "arguments": args,
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                context.log(f"MCP tool result received")
                return {"result": result.get("result", result)}
                
        except httpx.HTTPError as e:
            context.log(f"MCP tool call failed: {e}", level="error")
            raise ValueError(f"MCP tool call failed: {e}")


class MCPResourceExecutor(NodeExecutor):
    """Access a resource from an MCP server"""
    
    node_type = "mcp_resource"
    display_name = "MCP Resource"
    category = "mcp"
    description = "Access a resource from an MCP server"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        server = self.get_config_value("server", "")
        resource_uri = self.get_config_value("resource", "")
        
        if not server or not resource_uri:
            raise ValueError("MCP server and resource URI are required")
        
        context.log(f"Accessing MCP resource: {resource_uri} on {server}")
        
        mcp_proxy_url = os.environ.get("MCP_PROXY_URL", "http://localhost:3100")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{mcp_proxy_url}/resources/read",
                    json={
                        "server": server,
                        "uri": resource_uri,
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                content = result.get("content", result)
                context.log(f"MCP resource content received")
                return {"content": content}
                
        except httpx.HTTPError as e:
            context.log(f"MCP resource access failed: {e}", level="error")
            raise ValueError(f"MCP resource access failed: {e}")


class MCPPromptExecutor(NodeExecutor):
    """Use a prompt template from an MCP server"""
    
    node_type = "mcp_prompt"
    display_name = "MCP Prompt"
    category = "mcp"
    description = "Use a prompt template from an MCP server"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        args = inputs.get("args", {})
        
        server = self.get_config_value("server", "")
        prompt_name = self.get_config_value("prompt", "")
        
        if not server or not prompt_name:
            raise ValueError("MCP server and prompt name are required")
        
        context.log(f"Getting MCP prompt: {prompt_name} from {server}")
        
        mcp_proxy_url = os.environ.get("MCP_PROXY_URL", "http://localhost:3100")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{mcp_proxy_url}/prompts/get",
                    json={
                        "server": server,
                        "prompt": prompt_name,
                        "arguments": args,
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                messages = result.get("messages", [])
                context.log(f"MCP prompt returned {len(messages)} messages")
                return {"messages": messages}
                
        except httpx.HTTPError as e:
            context.log(f"MCP prompt fetch failed: {e}", level="error")
            raise ValueError(f"MCP prompt fetch failed: {e}")


class MCPServerListExecutor(NodeExecutor):
    """List available MCP servers and their capabilities"""
    
    node_type = "mcp_servers"
    display_name = "MCP Servers"
    category = "mcp"
    description = "List available MCP servers and their capabilities"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        context.log("Listing MCP servers")
        
        mcp_proxy_url = os.environ.get("MCP_PROXY_URL", "http://localhost:3100")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{mcp_proxy_url}/servers")
                response.raise_for_status()
                result = response.json()
                
                servers = result.get("servers", [])
                context.log(f"Found {len(servers)} MCP servers")
                return {"servers": servers}
                
        except httpx.HTTPError as e:
            context.log(f"MCP servers list failed: {e}", level="error")
            # Return empty list if proxy is not available
            return {"servers": []}








