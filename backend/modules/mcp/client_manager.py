"""
MCP Client Manager

Manages connections to external MCP servers via stdio or HTTP transports.
Handles tool discovery, execution, and connection lifecycle.
"""

import asyncio
import logging
import os
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import MCPConfigStore, MCPServerConfig, MCPServerStatus, TransportType
from .tools_bridge import ToolsBridge

logger = logging.getLogger(__name__)

# Import MCP SDK - gracefully handle if not installed
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.streamable_http import streamablehttp_client
    MCP_SDK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MCP SDK not installed: {e}. Install with: pip install mcp")
    MCP_SDK_AVAILABLE = False
    ClientSession = None
    stdio_client = None
    StdioServerParameters = None
    streamablehttp_client = None


@dataclass
class MCPConnection:
    """Represents an active connection to an MCP server."""
    config: MCPServerConfig
    session: Optional[Any] = None  # ClientSession
    exit_stack: Optional[AsyncExitStack] = None
    connected: bool = False
    error: Optional[str] = None
    tools: List[Any] = field(default_factory=list)
    resources: List[Any] = field(default_factory=list)
    prompts: List[Any] = field(default_factory=list)
    last_connected: Optional[datetime] = None
    latency_ms: Optional[float] = None


class MCPClientManager:
    """Manages connections to external MCP servers."""
    
    def __init__(self, config_store: Optional[MCPConfigStore] = None):
        """Initialize the client manager.
        
        Args:
            config_store: Configuration store for server configs
        """
        self.config_store = config_store or MCPConfigStore()
        self.connections: Dict[str, MCPConnection] = {}
        self._lock = asyncio.Lock()
    
    @property
    def is_available(self) -> bool:
        """Check if MCP SDK is available."""
        return MCP_SDK_AVAILABLE
    
    async def connect(self, server_name: str) -> MCPConnection:
        """Connect to an MCP server by name.
        
        Args:
            server_name: Name of the server (must be configured)
            
        Returns:
            MCPConnection with connection status
            
        Raises:
            ValueError: If server not found or SDK not available
        """
        if not MCP_SDK_AVAILABLE:
            raise ValueError("MCP SDK not installed. Run: pip install mcp")
        
        config = self.config_store.get_server(server_name)
        if not config:
            raise ValueError(f"Server '{server_name}' not found in configuration")
        
        async with self._lock:
            # Disconnect existing connection if any
            if server_name in self.connections:
                await self._disconnect_internal(server_name)
            
            connection = MCPConnection(config=config)
            
            try:
                start_time = time.time()
                
                if config.transport == TransportType.STDIO:
                    await self._connect_stdio(connection)
                elif config.transport == TransportType.HTTP:
                    await self._connect_http(connection)
                else:
                    raise ValueError(f"Unsupported transport: {config.transport}")
                
                connection.latency_ms = (time.time() - start_time) * 1000
                connection.connected = True
                connection.last_connected = datetime.utcnow()
                connection.error = None
                
                # Discover available tools, resources, prompts
                await self._discover_capabilities(connection)
                
                logger.info(
                    f"Connected to MCP server '{server_name}' - "
                    f"{len(connection.tools)} tools, {len(connection.resources)} resources"
                )
                
            except Exception as e:
                connection.error = str(e)
                connection.connected = False
                logger.error(f"Failed to connect to MCP server '{server_name}': {e}")
            
            self.connections[server_name] = connection
            return connection
    
    async def _connect_stdio(self, connection: MCPConnection):
        """Connect via stdio transport."""
        config = connection.config
        
        if not config.command:
            raise ValueError("Stdio transport requires a command")
        
        # Build environment with any custom env vars
        env = dict(os.environ)
        if config.env:
            env.update(config.env)
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args or [],
            env=env if config.env else None,
        )
        
        # Set up exit stack for resource management
        connection.exit_stack = AsyncExitStack()
        
        # Connect to the server
        stdio_transport = await connection.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        
        read_stream, write_stream = stdio_transport
        
        # Create session
        connection.session = await connection.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        # Initialize the session
        await connection.session.initialize()
    
    async def _connect_http(self, connection: MCPConnection):
        """Connect via HTTP transport."""
        config = connection.config
        
        if not config.url:
            raise ValueError("HTTP transport requires a URL")
        
        # Set up headers for auth if provided
        headers = {}
        if config.auth_token:
            headers["Authorization"] = f"Bearer {config.auth_token}"
        
        # Set up exit stack for resource management
        connection.exit_stack = AsyncExitStack()
        
        # Connect to the server
        http_transport = await connection.exit_stack.enter_async_context(
            streamablehttp_client(config.url, headers=headers if headers else None)
        )
        
        read_stream, write_stream, _ = http_transport
        
        # Create session
        connection.session = await connection.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        # Initialize the session
        await connection.session.initialize()
    
    async def _discover_capabilities(self, connection: MCPConnection):
        """Discover tools, resources, and prompts from connected server."""
        if not connection.session:
            return
        
        # Discover tools
        try:
            tools_response = await connection.session.list_tools()
            connection.tools = tools_response.tools if hasattr(tools_response, 'tools') else []
        except Exception as e:
            logger.warning(f"Failed to list tools: {e}")
            connection.tools = []
        
        # Discover resources
        try:
            resources_response = await connection.session.list_resources()
            connection.resources = resources_response.resources if hasattr(resources_response, 'resources') else []
        except Exception as e:
            logger.debug(f"Failed to list resources (may not be supported): {e}")
            connection.resources = []
        
        # Discover prompts
        try:
            prompts_response = await connection.session.list_prompts()
            connection.prompts = prompts_response.prompts if hasattr(prompts_response, 'prompts') else []
        except Exception as e:
            logger.debug(f"Failed to list prompts (may not be supported): {e}")
            connection.prompts = []
    
    async def disconnect(self, server_name: str) -> bool:
        """Disconnect from an MCP server.
        
        Args:
            server_name: Name of the server to disconnect
            
        Returns:
            True if disconnected, False if not connected
        """
        async with self._lock:
            return await self._disconnect_internal(server_name)
    
    async def _disconnect_internal(self, server_name: str) -> bool:
        """Internal disconnect without lock."""
        if server_name not in self.connections:
            return False
        
        connection = self.connections[server_name]
        
        try:
            if connection.exit_stack:
                await connection.exit_stack.aclose()
        except Exception as e:
            logger.warning(f"Error closing connection to '{server_name}': {e}")
        
        del self.connections[server_name]
        logger.info(f"Disconnected from MCP server '{server_name}'")
        return True
    
    async def disconnect_all(self):
        """Disconnect from all servers."""
        server_names = list(self.connections.keys())
        for name in server_names:
            await self.disconnect(name)
    
    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a tool on a connected MCP server.
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If server not connected or tool not found
        """
        if server_name not in self.connections:
            raise ValueError(f"Server '{server_name}' is not connected")
        
        connection = self.connections[server_name]
        
        if not connection.connected or not connection.session:
            raise ValueError(f"Server '{server_name}' is not connected")
        
        try:
            result = await connection.session.call_tool(
                tool_name,
                arguments=arguments or {}
            )
            return result
        except Exception as e:
            logger.error(f"Tool call failed: {server_name}/{tool_name}: {e}")
            raise
    
    def get_connection_status(self, server_name: str) -> Optional[MCPServerStatus]:
        """Get the status of a server connection.
        
        Args:
            server_name: Name of the server
            
        Returns:
            MCPServerStatus or None if not configured
        """
        # Check if server is connected
        if server_name in self.connections:
            conn = self.connections[server_name]
            return MCPServerStatus(
                name=server_name,
                connected=conn.connected,
                error=conn.error,
                tools_count=len(conn.tools),
                resources_count=len(conn.resources),
                last_connected=conn.last_connected.isoformat() if conn.last_connected else None,
                latency_ms=conn.latency_ms,
            )
        
        # Check if server is configured but not connected
        config = self.config_store.get_server(server_name)
        if config:
            return MCPServerStatus(
                name=server_name,
                connected=False,
            )
        
        return None
    
    def list_all_statuses(self) -> List[MCPServerStatus]:
        """Get statuses for all configured servers."""
        statuses = []
        
        for config in self.config_store.list_servers():
            status = self.get_connection_status(config.name)
            if status:
                statuses.append(status)
        
        return statuses
    
    def get_server_tools(self, server_name: str) -> List[Dict]:
        """Get tools from a connected server in OpenAI format.
        
        Args:
            server_name: Name of the server
            
        Returns:
            List of tools in OpenAI function calling format
        """
        if server_name not in self.connections:
            return []
        
        connection = self.connections[server_name]
        
        if not connection.connected:
            return []
        
        return ToolsBridge.mcp_to_openai(server_name, connection.tools)
    
    def get_all_tools_openai_format(self) -> List[Dict]:
        """Get all tools from all connected servers in OpenAI format.
        
        Returns:
            Merged list of all MCP tools in OpenAI format
        """
        all_tools = []
        
        for name, connection in self.connections.items():
            if connection.connected:
                all_tools.extend(ToolsBridge.mcp_to_openai(name, connection.tools))
        
        return all_tools
    
    async def connect_auto_connect_servers(self):
        """Connect to all servers configured for auto-connect."""
        servers = self.config_store.get_auto_connect_servers()
        
        for config in servers:
            try:
                await self.connect(config.name)
            except Exception as e:
                logger.error(f"Failed to auto-connect to '{config.name}': {e}")
    
    def get_tools_summary(self) -> str:
        """Get a human-readable summary of all available MCP tools.
        
        Returns:
            Formatted summary string
        """
        if not self.connections:
            return "No MCP servers connected"
        
        summaries = []
        for name, connection in self.connections.items():
            if connection.connected:
                summaries.append(ToolsBridge.create_tool_summary(name, connection.tools))
        
        return "\n\n".join(summaries) if summaries else "No tools available"
