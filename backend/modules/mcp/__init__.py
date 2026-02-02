"""
MCP (Model Context Protocol) Module for Llama-Nexus

This module provides MCP client functionality to connect to external MCP servers
and expose their tools to the LLM chat interface.

Components:
- MCPClientManager: Manages connections to external MCP servers
- MCPConfigStore: Persists server configurations
- ToolsBridge: Converts MCP tools to OpenAI-compatible format
- PopularServers: Pre-configured popular MCP server definitions
"""

from .client_manager import MCPClientManager
from .config import MCPConfigStore, MCPServerConfig, TransportType
from .tools_bridge import ToolsBridge
from .popular_servers import POPULAR_MCP_SERVERS, get_popular_server_config

__all__ = [
    'MCPClientManager',
    'MCPConfigStore', 
    'MCPServerConfig',
    'TransportType',
    'ToolsBridge',
    'POPULAR_MCP_SERVERS',
    'get_popular_server_config',
]
