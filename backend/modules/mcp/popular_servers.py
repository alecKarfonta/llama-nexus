"""
Popular MCP Server Definitions

Pre-configured definitions for popular MCP servers that users can easily add.
These servers are from the official MCP servers repository.
"""

from typing import Dict, List, Optional
from .config import MCPServerConfig, TransportType


# Popular MCP servers with pre-configured settings
POPULAR_MCP_SERVERS: Dict[str, dict] = {
    "filesystem": {
        "name": "filesystem",
        "description": "Read and write files on the local filesystem",
        "icon": "📁",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem"],
        "requires_path": True,  # User must provide a path
        "default_path_placeholder": "/path/to/directory",
        "tools": ["read_file", "write_file", "list_directory", "create_directory", "delete_file"],
        "npm_package": "@modelcontextprotocol/server-filesystem",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
    },
    "github": {
        "name": "github", 
        "description": "Interact with GitHub repositories, issues, and pull requests",
        "icon": "🐙",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "requires_env": ["GITHUB_TOKEN"],
        "tools": ["search_repositories", "get_repository", "list_issues", "create_issue", "list_pull_requests"],
        "npm_package": "@modelcontextprotocol/server-github",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/github",
    },
    "slack": {
        "name": "slack",
        "description": "Send messages and interact with Slack workspaces",
        "icon": "💬",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "requires_env": ["SLACK_BOT_TOKEN"],
        "tools": ["send_message", "list_channels", "search_messages"],
        "npm_package": "@modelcontextprotocol/server-slack",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/slack",
    },
    "postgres": {
        "name": "postgres",
        "description": "Query and interact with PostgreSQL databases",
        "icon": "🐘",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres"],
        "requires_env": ["POSTGRES_CONNECTION_STRING"],
        "tools": ["query", "list_tables", "describe_table"],
        "npm_package": "@modelcontextprotocol/server-postgres",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/postgres",
    },
    "puppeteer": {
        "name": "puppeteer",
        "description": "Browser automation and web scraping",
        "icon": "🎭",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "tools": ["navigate", "screenshot", "click", "type", "get_content"],
        "npm_package": "@modelcontextprotocol/server-puppeteer",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer",
    },
    "brave-search": {
        "name": "brave-search",
        "description": "Web search using Brave Search API",
        "icon": "🦁",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "requires_env": ["BRAVE_API_KEY"],
        "tools": ["brave_web_search", "brave_local_search"],
        "npm_package": "@modelcontextprotocol/server-brave-search",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search",
    },
    "google-drive": {
        "name": "google-drive",
        "description": "Access and manage Google Drive files",
        "icon": "📂",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-gdrive"],
        "requires_oauth": True,
        "tools": ["search_files", "read_file", "list_files"],
        "npm_package": "@modelcontextprotocol/server-gdrive",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/gdrive",
    },
    "memory": {
        "name": "memory",
        "description": "Knowledge graph-based memory for persistent context",
        "icon": "🧠",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "tools": ["create_entities", "create_relations", "search_nodes", "delete_entities"],
        "npm_package": "@modelcontextprotocol/server-memory",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/memory",
    },
    "fetch": {
        "name": "fetch",
        "description": "Fetch and process content from URLs",
        "icon": "🌐",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "tools": ["fetch"],
        "npm_package": "@modelcontextprotocol/server-fetch",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/fetch",
    },
    "sqlite": {
        "name": "sqlite",
        "description": "Query SQLite databases",
        "icon": "🗃️",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite"],
        "requires_path": True,
        "default_path_placeholder": "/path/to/database.db",
        "tools": ["query", "list_tables", "describe_table"],
        "npm_package": "@modelcontextprotocol/server-sqlite",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/sqlite",
    },
    "sentry": {
        "name": "sentry",
        "description": "Retrieve and analyze issues from Sentry",
        "icon": "🛡️",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sentry"],
        "requires_env": ["SENTRY_AUTH_TOKEN"],
        "tools": ["get_sentry_issue"],
        "npm_package": "@modelcontextprotocol/server-sentry",
        "docs_url": "https://docs.sentry.io/product/sentry-mcp/",
    },
    "sequential-thinking": {
        "name": "sequential-thinking",
        "description": "Dynamic problem-solving through thought sequences",
        "icon": "💭",
        "transport": TransportType.STDIO,
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "tools": ["create_thought_sequence", "add_thought", "get_sequence"],
        "npm_package": "@modelcontextprotocol/server-sequential-thinking",
        "docs_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/sequential-thinking",
    },
}


def get_popular_server_config(
    server_id: str,
    path: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
) -> MCPServerConfig:
    """Create an MCPServerConfig from a popular server definition.
    
    Args:
        server_id: ID of the popular server (e.g., 'filesystem', 'github')
        path: Path argument for servers that require it (filesystem, sqlite)
        env_vars: Environment variables to set
    
    Returns:
        MCPServerConfig ready to be added to the config store
        
    Raises:
        ValueError: If server_id is not a known popular server
    """
    if server_id not in POPULAR_MCP_SERVERS:
        raise ValueError(f"Unknown popular server: {server_id}. Available: {list(POPULAR_MCP_SERVERS.keys())}")
    
    server_def = POPULAR_MCP_SERVERS[server_id]
    
    # Build args list
    args = list(server_def.get("args", []))
    
    # Append path if required
    if server_def.get("requires_path"):
        if not path:
            raise ValueError(f"Server '{server_id}' requires a path argument")
        args.append(path)
    
    return MCPServerConfig(
        name=server_def["name"],
        description=server_def.get("description"),
        icon=server_def.get("icon"),
        transport=server_def["transport"],
        command=server_def.get("command"),
        args=args,
        env=env_vars or {},
        enabled=True,
        auto_connect=True,
    )


def list_popular_servers() -> List[dict]:
    """List all popular servers with their metadata.
    
    Returns:
        List of server definitions with metadata for frontend display
    """
    return [
        {
            "id": server_id,
            "name": server_def["name"],
            "description": server_def.get("description", ""),
            "icon": server_def.get("icon", "🔧"),
            "transport": server_def["transport"],
            "requires_path": server_def.get("requires_path", False),
            "requires_env": server_def.get("requires_env", []),
            "requires_oauth": server_def.get("requires_oauth", False),
            "tools": server_def.get("tools", []),
            "docs_url": server_def.get("docs_url"),
            "npm_package": server_def.get("npm_package"),
        }
        for server_id, server_def in POPULAR_MCP_SERVERS.items()
    ]
