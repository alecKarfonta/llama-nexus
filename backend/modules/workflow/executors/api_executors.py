"""
API Node Executors - External HTTP and GraphQL requests
"""

from typing import Dict, Any
import httpx
import json
from .base import NodeExecutor, ExecutionContext


class HttpRequestExecutor(NodeExecutor):
    """Make HTTP request"""
    
    node_type = "http_request"
    display_name = "HTTP Request"
    category = "api"
    description = "Make HTTP requests to external APIs"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        url = inputs.get("url") or self.get_config_value("baseUrl", "")
        body = inputs.get("body")
        headers = inputs.get("headers", {})
        
        method = self.get_config_value("method", "GET")
        timeout = self.get_config_value("timeout", 30)
        base_url = self.get_config_value("baseUrl", "")
        
        # Handle auth
        auth_config = self.get_config_value("auth", {})
        if auth_config.get("type") == "bearer":
            token = auth_config.get("credentials", {}).get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif auth_config.get("type") == "api_key":
            creds = auth_config.get("credentials", {})
            header_name = creds.get("header", "X-API-Key")
            api_key = creds.get("key")
            if api_key:
                headers[header_name] = api_key
        
        # Build full URL
        if base_url and not url.startswith("http"):
            full_url = f"{base_url.rstrip('/')}/{url.lstrip('/')}"
        else:
            full_url = url
        
        if not full_url:
            raise ValueError("URL is required for HTTP request")
        
        context.log(f"HTTP {method} {full_url}")
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method in ["POST", "PUT", "PATCH"]:
                response = await client.request(
                    method,
                    full_url,
                    json=body if body else None,
                    headers=headers,
                )
            else:
                response = await client.request(
                    method,
                    full_url,
                    params=body if isinstance(body, dict) else None,
                    headers=headers,
                )
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            context.log(f"HTTP response: {response.status_code}")
            
            return {
                "response": response_data,
                "status": response.status_code,
                "headers": dict(response.headers),
            }


class GraphQLExecutor(NodeExecutor):
    """Execute GraphQL query"""
    
    node_type = "graphql_query"
    display_name = "GraphQL Query"
    category = "api"
    description = "Execute GraphQL queries and mutations"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        variables = inputs.get("variables", {})
        
        endpoint = self.get_config_value("endpoint", "")
        query = self.get_config_value("query", "")
        
        if not endpoint:
            raise ValueError("GraphQL endpoint is required")
        
        if not query:
            raise ValueError("GraphQL query is required")
        
        context.log(f"GraphQL query to {endpoint}")
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                endpoint,
                json={
                    "query": query,
                    "variables": variables,
                },
                headers={
                    "Content-Type": "application/json",
                }
            )
            
            result = response.json()
            
            data = result.get("data")
            errors = result.get("errors", [])
            
            if errors:
                context.log(f"GraphQL errors: {errors}", level="warning")
            
            return {
                "data": data,
                "errors": errors,
            }


class WebSocketExecutor(NodeExecutor):
    """WebSocket connection"""
    
    node_type = "websocket"
    display_name = "WebSocket"
    category = "api"
    description = "Connect and send messages via WebSocket"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        url = inputs.get("url") or self.get_config_value("url", "")
        message = inputs.get("message")
        
        if not url:
            raise ValueError("WebSocket URL is required")
        
        context.log(f"WebSocket connection to {url}")
        
        import websockets
        
        async with websockets.connect(url) as ws:
            if message:
                await ws.send(json.dumps(message) if isinstance(message, dict) else str(message))
                context.log("Sent message via WebSocket")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=30)
                try:
                    response_data = json.loads(response)
                except:
                    response_data = response
                
                return {"response": response_data}
            except asyncio.TimeoutError:
                context.log("WebSocket timeout", level="warning")
                return {"response": None}


class OpenAPICallExecutor(NodeExecutor):
    """Call OpenAPI endpoint"""
    
    node_type = "openapi_call"
    display_name = "OpenAPI Call"
    category = "api"
    description = "Call an endpoint from an OpenAPI specification"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        params = inputs.get("params", {})
        body = inputs.get("body")
        
        spec_url = self.get_config_value("specUrl", "")
        operation_id = self.get_config_value("operationId", "")
        server = self.get_config_value("server", "")
        
        if not operation_id:
            raise ValueError("Operation ID is required")
        
        context.log(f"OpenAPI call: {operation_id}")
        
        # For now, this is a simplified implementation
        # A full implementation would parse the OpenAPI spec and build the request
        
        # If server is provided, treat operation_id as the path
        if server:
            url = f"{server.rstrip('/')}/{operation_id.lstrip('/')}"
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.request(
                    "POST",
                    url,
                    params=params,
                    json=body,
                )
                
                try:
                    response_data = response.json()
                except:
                    response_data = response.text
                
                return {
                    "response": response_data,
                    "status": response.status_code,
                }
        
        return {"response": None, "status": 0}
