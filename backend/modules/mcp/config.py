"""
MCP Configuration Models and Persistence

Defines configuration structures for MCP servers and handles persistence
to the local filesystem.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)


class TransportType(str, Enum):
    """MCP transport types."""
    STDIO = "stdio"
    HTTP = "http"


class MCPServerConfig(BaseModel):
    """Configuration for an external MCP server."""
    name: str = Field(..., description="Unique name for this server")
    transport: TransportType = Field(..., description="Transport type (stdio or http)")
    
    # Stdio transport configuration
    command: Optional[str] = Field(None, description="Command to run for stdio transport")
    args: Optional[List[str]] = Field(default_factory=list, description="Arguments for the command")
    env: Optional[Dict[str, str]] = Field(default_factory=dict, description="Environment variables")
    
    # HTTP transport configuration
    url: Optional[str] = Field(None, description="URL for HTTP transport")
    auth_token: Optional[str] = Field(None, description="Authentication token for HTTP")
    
    # Common settings
    enabled: bool = Field(True, description="Whether this server is enabled")
    auto_connect: bool = Field(True, description="Connect automatically on startup")
    description: Optional[str] = Field(None, description="Human-readable description")
    icon: Optional[str] = Field(None, description="Icon name or emoji")
    
    # Metadata
    created_at: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    
    class Config:
        use_enum_values = True


class MCPServerStatus(BaseModel):
    """Runtime status of an MCP server connection."""
    name: str
    connected: bool = False
    error: Optional[str] = None
    tools_count: int = 0
    resources_count: int = 0
    last_connected: Optional[str] = None
    latency_ms: Optional[float] = None


class MCPConfigStore:
    """Persists MCP server configurations to the filesystem."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the config store.
        
        Args:
            data_dir: Directory to store configurations. Defaults to ./data/mcp
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "mcp"
        
        self.data_dir = Path(data_dir)
        self.config_file = self.data_dir / "servers.json"
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """Create data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.config_file.exists():
            self._save_configs([])
    
    def _load_configs(self) -> List[dict]:
        """Load raw configs from file."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_configs(self, configs: List[dict]):
        """Save raw configs to file."""
        with open(self.config_file, 'w') as f:
            json.dump(configs, f, indent=2)
    
    def list_servers(self) -> List[MCPServerConfig]:
        """List all configured MCP servers."""
        raw_configs = self._load_configs()
        servers = []
        for cfg in raw_configs:
            try:
                servers.append(MCPServerConfig(**cfg))
            except Exception as e:
                logger.warning(f"Failed to parse server config: {e}")
        return servers
    
    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get a specific server by name."""
        for server in self.list_servers():
            if server.name == name:
                return server
        return None
    
    def add_server(self, config: MCPServerConfig) -> MCPServerConfig:
        """Add a new server configuration.
        
        Raises:
            ValueError: If a server with this name already exists
        """
        configs = self._load_configs()
        
        # Check for duplicates
        if any(c.get('name') == config.name for c in configs):
            raise ValueError(f"Server '{config.name}' already exists")
        
        config.created_at = datetime.utcnow().isoformat()
        configs.append(config.model_dump())
        self._save_configs(configs)
        
        logger.info(f"Added MCP server: {config.name}")
        return config
    
    def update_server(self, name: str, config: MCPServerConfig) -> MCPServerConfig:
        """Update an existing server configuration.
        
        Raises:
            ValueError: If the server doesn't exist
        """
        configs = self._load_configs()
        
        for i, c in enumerate(configs):
            if c.get('name') == name:
                config.updated_at = datetime.utcnow().isoformat()
                # Preserve created_at
                config.created_at = c.get('created_at', config.created_at)
                configs[i] = config.model_dump()
                self._save_configs(configs)
                logger.info(f"Updated MCP server: {name}")
                return config
        
        raise ValueError(f"Server '{name}' not found")
    
    def remove_server(self, name: str) -> bool:
        """Remove a server configuration.
        
        Returns:
            True if server was removed, False if not found
        """
        configs = self._load_configs()
        original_len = len(configs)
        configs = [c for c in configs if c.get('name') != name]
        
        if len(configs) < original_len:
            self._save_configs(configs)
            logger.info(f"Removed MCP server: {name}")
            return True
        return False
    
    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get all enabled servers."""
        return [s for s in self.list_servers() if s.enabled]
    
    def get_auto_connect_servers(self) -> List[MCPServerConfig]:
        """Get servers configured for auto-connect."""
        return [s for s in self.list_servers() if s.enabled and s.auto_connect]
